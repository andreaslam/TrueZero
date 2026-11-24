import argparse
import hashlib
import io
import json
import os
import queue
import random
import re
import socket
import threading
import time
import warnings

import network as network
import numpy as np
import torch

# the Rust self-play engine loads TorchScript (.pt) via libtorch, so torch.export is not
# a drop-in replacement; silence the jit.save/jit.load deprecation notice
warnings.filterwarnings(
    "ignore", message=r"`torch\.jit\.(load|save)` is deprecated", category=FutureWarning
)
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.loop import LoopBuffer
from lib.train import ScalarTarget, TrainSettings
from paths import data_dir, data_path
from ladder import LadderRunner
from runconfig import record_received_file, start_run
from telemetry import (
    LogTailer,
    SelfplayHealth,
    Telemetry,
    buffer_row,
    gate_row,
    load_observe_config,
    logger_row,
    resolve_data_paths,
)
from torch import optim


class Server:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.file = None

    def connect(self):
        while not self.try_connect():
            time.sleep(1)

    def try_connect(self) -> bool:
        # a socket whose connect failed cannot be reused, so start from a fresh one
        self.close()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.file = None
        try:
            self.socket.connect((self.host, self.port))
        except OSError as e:
            print(f"Connection failed: {e}. Retrying...")
            return False
        print("Connected to server!")
        self.file = self.socket.makefile("r")
        return True

    def send(self, message):
        message = {"purpose": message}
        obj = json.dumps(message) + "\n"

        try:
            self.socket.sendall(obj.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"[Server] Failed to send message: {e}")

    def receive(self):
        assert self.file is not None
        return self.file.readline()

    def close(self):
        try:
            if self.file is not None:
                self.file.close()
        except OSError:
            pass

        try:
            self.socket.close()
        except OSError:
            pass


def serialise_net(model):
    buffer = io.BytesIO()

    with torch.no_grad():
        model.eval()
        torch.jit.save(model, buffer)

    return buffer.getvalue()


def model_weights_checksum(model):
    checksum = hashlib.sha256()

    with torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            tensor = tensor.detach().cpu().contiguous()

            checksum.update(name.encode("utf-8"))
            checksum.update(str(tensor.dtype).encode("utf-8"))
            checksum.update(str(tuple(tensor.shape)).encode("utf-8"))
            checksum.update(tensor.numpy().tobytes())

    return checksum.hexdigest()


class FullSearchPolicyFile(DataFile):
    """A replay file whose fast-search positions carry no policy target.

    With playout cap randomisation most moves are fast searches (few visits, no
    root noise, possibly stopped early). Their visit distribution is close to the
    network's own policy, so training the policy on it mostly teaches the net to
    copy itself; KataGo trains the policy on full searches only. The inherited
    trainer ignores `is_full_search`, so here every fast-search position gets
    policy values of -1, lib's own marker for "no policy target": such a row
    adds no policy loss or gradient and is left out of policy accuracy. Value,
    WDL and moves-left targets are unchanged. Neither lib nor the file format is
    modified, and this applies to files already in the replay buffer.
    """

    def load_position(self, pi):
        position = super().load_position(pi)
        if not position.is_full_search:
            position.policy_values = np.full_like(position.policy_values, -1.0)
        return position

    def with_new_handles(self):
        # The samplers reopen files through this, so it must keep the subclass.
        other = super().with_new_handles()
        return FullSearchPolicyFile(other.info, other.bin_handle, other.off_handle)


def load_file(games_path):
    f = DataFile.open(Game.find("chess"), games_path)
    return FullSearchPolicyFile(f.info, f.bin_handle, f.off_handle)


# Weight of the policy loss per position that has a policy target.
POLICY_WEIGHT = 1.0


def policy_target_fraction(batch):
    """Fraction of the batch's positions that have a policy target (full searches)."""
    has_target = (batch.policy_values != -1).any(dim=1)
    return has_target.float().mean().item()


def policy_entropy_row(model, batch, prefix):
    """Mean entropy (nats) of the net's policy over legal moves and of the search target,
    over positions with a policy target (full searches). A net whose entropy collapses
    far below the target's is overconfident; one stuck high is not learning the policy."""
    scalars, logits = model(batch.input_full)
    has_target = (batch.policy_values != -1).any(dim=1)
    if not has_target.any():
        return {}
    legal = batch.policy_values != -1
    picked = torch.gather(logits.flatten(1), 1, batch.policy_indices)
    picked = picked.masked_fill(~legal, float("-inf"))
    logp = torch.log_softmax(picked[has_target], dim=1)
    p = logp.exp()
    net_h = -(p * logp.masked_fill(~legal[has_target], 0.0)).sum(dim=1).mean()
    target = batch.policy_values[has_target].clamp(min=0.0)
    target_h = -(target * torch.log(target.clamp(min=1e-12))).sum(dim=1).mean()
    return {
        f"policy-entropy/{prefix} net": net_h.item(),
        f"policy-entropy/{prefix} target": target_h.item(),
    }


def set_policy_weight(train_settings, batch):
    """Keep the policy's share of the loss independent of how many positions are masked.

    lib averages the policy loss over the whole batch, so masking fast searches
    (FullSearchPolicyFile) would shrink it by the full-search fraction p and shift
    the balance towards value. Scaling by 1/p restores the per-target weight it had
    before playout cap randomisation. Returns p.
    """
    fraction = policy_target_fraction(batch)
    train_settings.policy_weight = POLICY_WEIGHT / max(fraction, 1.0 / len(batch))
    return fraction


HOST = os.environ.get("TZ_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("TZ_SERVER_PORT", "38475"))

# The training hyperparameters below are the defaults (equal to run.example.toml);
# main() replaces them with the run's run.toml through apply_run_config.
BUFFER_SIZE = 1500000
BATCH_SIZE = 2048

# Do not train until the replay buffer has at least this many positions.
MIN_BUFFER_TO_TRAIN = 250_000

# Normal training target:
# approximately one pass over newly generated positions.
EPOCHS_PER_GEN = 1.0

# Fraction of normal training batches sampled from the newest generation.
RECENT_BATCH_FRACTION = 0.5

# Small files still need to produce a candidate.
# This is intentionally modest because each file is only ~2k-5k positions.
MIN_TRAINING_STEPS = 16

# IMPORTANT:
# Startup training is NOT treated as one epoch over the entire replay
# buffer. That would massively oversample the newest 2k-5k positions
# through the recent sampler.
#
# Instead, perform a small bootstrap session over the full replay buffer.
STARTUP_TRAINING_STEPS = 16

TEST_FRACTION = 0.2
WEIGHT_DECAY = 1e-4
LOSS_WEIGHTS = {
    "value_weight": 1.0,
    "wdl_weight": 1.0,
    "moves_left_weight": 0.1,
    "moves_left_delta": 0.1,
    "clip_norm": 5.0,
}

assert BATCH_SIZE > 0 and (BATCH_SIZE & (BATCH_SIZE - 1)) == 0

data_dir("hidden")

DATAFILE_PATH = data_path("hidden/datafile.txt")
TRAINING_LOG_PATH = data_path("hidden/traininglog.txt")
LOG_NPZ_PATH = data_path("log.npz")
TRAINER_STATE_PATH = data_path("hidden/trainer_state.pt")
TRAINER_STATE_VERSION = 1

LAST_SENT_NET_CHECKSUM = None

# This trainer's run identity (runconfig.StartIdentity), set by main().
RUN_IDENTITY = None

# Optional W&B telemetry (telemetry.py), set by main(); None in tests and tools.
TELEMETRY = None
SELFPLAY_HEALTH = SelfplayHealth()


def telemetry_log(row_fn):
    """Log `row_fn()` if telemetry is on. Metrics must never stop training."""
    if TELEMETRY is None or not TELEMETRY.enabled:
        return
    try:
        TELEMETRY.log(row_fn())
    except Exception as e:
        print(f"[telemetry][Warning] {e!r}")


def create_optimizer(model):
    # The learning rate is set before every step by LrSchedule.apply.
    return optim.AdamW(
        params=model.parameters(),
        lr=1e-3,
        weight_decay=WEIGHT_DECAY,
    )


# ---------------------------------------------------------------------------
# Learning-rate schedule
#
# Modelled on what AlphaZero-style engines do for continual self-play training:
#
# - A warm-up at the start: KataGo trains its first 2M samples at 1/20 rising to
#   1x of the full rate; Lc0 ramps linearly over `warmup_steps`.
# - Then piecewise-constant drops keyed on how much has been trained, never on
#   a planned end point: AlphaZero 0.2 -> 0.02 -> 0.002 -> 0.0002 over 700k
#   steps; Lc0 `lr_values`/`lr_boundaries`; KataGo's `-lr-schedule` keyed on
#   samples trained. A cosine-to-zero schedule needs a known run length, which
#   continual training does not have, so it is not used here.
#
# The position in the schedule is the number of samples trained (steps times
# batch size, like KataGo's global_step_samples), so the schedule is unchanged
# if BATCH_SIZE changes. It is saved in the trainer checkpoint with the
# optimiser. A fresh optimiser starts the schedule at zero, so it warms up
# again rather than taking full-rate steps from empty Adam moments.
#
# Set in run.toml [trainer]:
#   lr_schedule        "samples:lr,..." piecewise-constant points, starting
#                      at 0; counts take K/M/B suffixes, e.g. "0:1e-3,50M:5e-4"
#   lr_warmup_samples  warm-up length in samples (0 disables warm-up)
# ---------------------------------------------------------------------------

LR_SCHEDULE = "0:1e-3,50M:5e-4,100M:2.5e-4,200M:1e-4"
LR_WARMUP_SAMPLES = "1M"

# The warm-up starts at this fraction of the scheduled rate (KataGo: 1/20).
LR_WARMUP_START_FRACTION = 0.05


def parse_sample_count(text):
    text = text.strip().upper()
    scale = {"K": 10**3, "M": 10**6, "B": 10**9}.get(text[-1:], 1)
    if scale != 1:
        text = text[:-1]
    value = float(text) * scale
    if value < 0 or value != int(value):
        raise ValueError(f"invalid sample count {text!r}")
    return int(value)


def parse_lr_schedule(text):
    """Parse "samples:lr,samples:lr,..." into sorted (samples, lr) points."""
    points = []
    for item in text.split(","):
        if not item.strip():
            continue
        samples, lr = item.split(":")
        lr = float(lr)
        if not lr > 0:
            raise ValueError(f"learning rate must be positive, got {lr}")
        points.append((parse_sample_count(samples), lr))

    points.sort()
    if not points or points[0][0] != 0:
        raise ValueError("the LR schedule must start at sample 0")
    if len({s for s, _ in points}) != len(points):
        raise ValueError("the LR schedule repeats a sample count")
    return points


class LrSchedule:
    """Warm-up then piecewise-constant learning rate, keyed on samples trained."""

    def __init__(self, points, warmup_samples, samples_trained=0):
        self.points = points
        self.warmup_samples = warmup_samples
        self.samples_trained = samples_trained

    @classmethod
    def from_config(cls):
        """A schedule at sample 0 from LR_SCHEDULE and LR_WARMUP_SAMPLES (run.toml)."""
        return cls(
            parse_lr_schedule(LR_SCHEDULE),
            parse_sample_count(LR_WARMUP_SAMPLES),
        )

    def lr(self):
        lr = self.points[0][1]
        for start, value in self.points:
            if self.samples_trained >= start:
                lr = value

        if self.samples_trained < self.warmup_samples:
            progress = self.samples_trained / self.warmup_samples
            lr *= LR_WARMUP_START_FRACTION + (1 - LR_WARMUP_START_FRACTION) * progress

        return lr

    def apply(self, op):
        """Set the learning rate for the next optimiser step and return it."""
        lr = self.lr()
        for group in op.param_groups:
            group["lr"] = lr
        return lr

    def advance(self, batch_size):
        self.samples_trained += batch_size

    def state_dict(self):
        return {"samples_trained": self.samples_trained}

    def load_state_dict(self, state):
        samples_trained = int(state["samples_trained"])
        if samples_trained < 0:
            raise ValueError(f"invalid samples_trained {samples_trained}")
        self.samples_trained = samples_trained


def scheduled_train_step(train_settings, batch, model, op, schedule, log):
    """One optimiser step at the scheduled learning rate."""
    lr = schedule.apply(op)
    fraction = set_policy_weight(train_settings, batch)
    train_settings.train_step(
        batch,
        network=model,
        optimizer=op,
        logger=log,
    )
    schedule.advance(BATCH_SIZE)
    log.log("schedule", "lr", lr)
    # Separate group so the plotter does not draw lr against a count in the millions.
    log.log("samples", "trained", schedule.samples_trained)
    log.log("policy-targets", "fraction", fraction)
    if TELEMETRY is not None:
        TELEMETRY.set_samples(schedule.samples_trained)
    telemetry_log(
        lambda: {
            **logger_row(log, ("train",), groups=("grad_norm", "param_norm")),
            "train/lr": lr,
            "train/policy_target_fraction": fraction,
        }
    )


def save_trainer_state(model, op, schedule, accepted_gen, candidate_gen, path=None):
    """Checkpoint the training weights, optimiser and LR schedule after a round.

    The accepted net on disk is not enough to resume training: a rejected
    candidate does not reset the training weights, and AdamW's moment
    estimates and step counts exist only in the optimiser. Restarting from the
    accepted net with a fresh optimiser rolls training back to the last
    accepted net, and the first steps afterwards move every parameter by about
    lr whatever its gradient scale (bias-corrected moments from one gradient).

    The checkpoint is serialised in memory, written to a temporary file and
    renamed over the old one, so a crash or a dropped share mid-write leaves the
    previous checkpoint intact. An I/O error only skips this checkpoint.
    """
    path = path or TRAINER_STATE_PATH
    tmp_path = path + ".tmp"

    buffer = io.BytesIO()
    torch.save(
        {
            "version": TRAINER_STATE_VERSION,
            "accepted_gen": accepted_gen,
            "candidate_gen": candidate_gen,
            "model": model.state_dict(),
            "optimizer": op.state_dict(),
            "schedule": schedule.state_dict(),
        },
        buffer,
    )

    try:
        with open(tmp_path, "wb") as f:
            f.write(buffer.getvalue())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except OSError as e:
        print(f"[Warning] failed to save trainer state: {e!r}")
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return False

    return True


def load_trainer_state(model, accepted_gen, device, path=None):
    """Restore the training weights, optimiser and schedule saved by save_trainer_state.

    Returns the optimiser and LR schedule to train with. The checkpoint is used only if it
    continues from the accepted net: it was written while `accepted_gen` was
    accepted, or by the round whose candidate became `accepted_gen` (whose
    weights are exactly that net). Otherwise, or if the file is unreadable or
    does not fit the model, `model` keeps the accepted net's weights, and a fresh
    optimiser and a schedule at sample 0 (so in warm-up) are returned.
    """
    path = path or TRAINER_STATE_PATH

    # Parsed outside the try below: a bad lr_schedule must stop the trainer,
    # not be reported as an unreadable checkpoint.
    LrSchedule.from_config()

    if not os.path.exists(path):
        print("[Training] No trainer state found; starting a fresh optimiser.")
        return create_optimizer(model), LrSchedule.from_config()

    try:
        state = torch.load(path, map_location=device, weights_only=True)

        if state.get("version") != TRAINER_STATE_VERSION:
            raise ValueError(f"unsupported version {state.get('version')!r}")

        if not state["accepted_gen"] <= accepted_gen <= state["candidate_gen"]:
            raise ValueError(
                f"saved for generations {state['accepted_gen']}..."
                f"{state['candidate_gen']}, but the accepted net is {accepted_gen}"
            )

        current = model.state_dict()
        saved = state["model"]
        if set(saved) != set(current) or any(
            saved[k].shape != current[k].shape for k in current
        ):
            raise ValueError("saved weights do not match the network architecture")

        schedule = LrSchedule.from_config()
        schedule.load_state_dict(state["schedule"])
        op = create_optimizer(model)
        op.load_state_dict(state["optimizer"])
        model.load_state_dict(saved)

    except Exception as e:
        print(f"[Warning] ignoring trainer state {path}: {e!r}")
        return create_optimizer(model), LrSchedule.from_config()

    print(
        "[Training] Resumed training weights, optimiser and LR schedule from "
        f"{path} (saved after training candidate {state['candidate_gen']}; "
        f"{schedule.samples_trained:,} samples trained, lr {schedule.lr():.3g})."
    )
    return op, schedule


def apply_run_config(cfg):
    """Set the training hyperparameters from a validated run.toml."""
    global BUFFER_SIZE, BATCH_SIZE, MIN_BUFFER_TO_TRAIN, EPOCHS_PER_GEN
    global RECENT_BATCH_FRACTION, MIN_TRAINING_STEPS, STARTUP_TRAINING_STEPS
    global TEST_FRACTION, LR_SCHEDULE, LR_WARMUP_SAMPLES, WEIGHT_DECAY, POLICY_WEIGHT
    t = cfg["trainer"]
    BUFFER_SIZE = t["buffer_size"]
    BATCH_SIZE = t["batch_size"]
    MIN_BUFFER_TO_TRAIN = t["min_buffer_to_train"]
    EPOCHS_PER_GEN = float(t["epochs_per_gen"])
    RECENT_BATCH_FRACTION = float(t["recent_batch_fraction"])
    MIN_TRAINING_STEPS = t["min_training_steps"]
    STARTUP_TRAINING_STEPS = t["startup_training_steps"]
    TEST_FRACTION = float(t["test_fraction"])
    LR_SCHEDULE = t["lr_schedule"]
    LR_WARMUP_SAMPLES = t["lr_warmup_samples"]
    WEIGHT_DECAY = float(t["weight_decay"])
    POLICY_WEIGHT = float(t["policy_weight"])
    for key in LOSS_WEIGHTS:
        LOSS_WEIGHTS[key] = float(t[key])
    # Fail now, not at the first training round.
    LrSchedule.from_config()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TrueZero trainer")
    parser.add_argument(
        "--allow-config-change",
        action="store_true",
        help="start although run.toml differs from the one the run was locked with",
    )
    parser.add_argument(
        "--allow-code-change",
        action="store_true",
        help="start although the code is not at the run's pinned commit, or is dirty",
    )
    return parser.parse_args(argv)


def receiver_loop(server, message_queue, receiver_stop_event):
    """
    Continuously receive messages from the server.

    Network reception is deliberately separated from training so that
    server.receive() cannot block the training scheduler.
    """
    while not receiver_stop_event.is_set():
        try:
            received_data = server.receive()

            # Empty string means the peer closed the connection.
            if received_data == "":
                print("[Receiver] Server connection closed.")
                receiver_stop_event.set()
                return

            message_queue.put(received_data)

        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            OSError,
        ) as e:
            print(f"[Receiver] Connection lost: {e}")
            receiver_stop_event.set()
            return

        except Exception as e:
            print(f"[Receiver] Unexpected receive error: {e}")
            receiver_stop_event.set()
            return


def get_message(message_queue, timeout=0.05):
    try:
        return message_queue.get(timeout=timeout)
    except queue.Empty:
        return None


def main():
    args = parse_args()
    # run.toml, the run lock and the start log come first: a refused start does nothing else.
    run_cfg, run_identity = start_run(
        "trainer",
        allow_config_change=args.allow_config_change,
        allow_code_change=args.allow_code_change,
    )
    apply_run_config(run_cfg)
    global RUN_IDENTITY, TELEMETRY
    RUN_IDENTITY = run_identity
    observe = load_observe_config()
    TELEMETRY = Telemetry(observe, run_cfg, run_identity)
    LogTailer(
        TELEMETRY,
        resolve_data_paths(observe["tail"]["logs"]),
        flush_seconds=float(observe["tail"]["flush_seconds"]),
    ).start()
    ladder = LadderRunner(observe["ladder"], TELEMETRY)
    seed_everything(run_cfg["run"]["seed"])

    game = Game.find("chess")

    # Select exactly one device.
    if torch.cuda.is_available():
        device = "cuda"
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        device = "mps"
    else:
        device = "cpu"

    print(f"Using: {device}")

    data_dir("nets")
    data_dir("games")

    training_nets = check_net_exists(
        device,
        r"tz_(\d+)\.pt",
    )

    model_path = get_model_path(training_nets)

    model = torch.jit.load(
        model_path,
        map_location=device,
    ).eval()

    starting_gen = int(
        re.findall(
            r"tz_(\d+)\.pt",
            model_path,
        )[0]
    )

    accepted_model_path = model_path
    accepted_gen = starting_gen

    pending_candidate_path = None
    pending_candidate_gen = None

    print(f"starting generation: {starting_gen}")

    server = Server(HOST, PORT)
    server.connect()

    # Verification happens synchronously before the receiver thread starts.
    get_verification(
        server,
        "PythonTraining",
    )

    data_paths = get_previous_data_paths()

    loopbuf = LoopBuffer(
        game,
        target_positions=BUFFER_SIZE,
        test_fraction=TEST_FRACTION,
    )

    train_settings = TrainSettings(
        game=game,
        scalar_target=ScalarTarget.Final,
        value_weight=LOSS_WEIGHTS["value_weight"],
        wdl_weight=LOSS_WEIGHTS["wdl_weight"],
        moves_left_weight=LOSS_WEIGHTS["moves_left_weight"],
        moves_left_delta=LOSS_WEIGHTS["moves_left_delta"],
        policy_weight=POLICY_WEIGHT,
        sim_weight=0.0,
        train_in_eval_mode=False,
        clip_norm=LOSS_WEIGHTS["clip_norm"],
        mask_policy=True,
    )

    # Resume the training weights (which may be ahead of the accepted net) and
    # the optimiser. Self-play still gets `accepted_model_path`.
    op, schedule = load_trainer_state(model, accepted_gen, device)
    print(
        f"[Training] LR schedule {schedule.points}, warm-up "
        f"{schedule.warmup_samples:,} samples; now at "
        f"{schedule.samples_trained:,} samples, lr {schedule.lr():.3g}"
    )

    log = load_previous_data(
        data_paths,
        loopbuf,
    )

    # ---------------------------------------------------------------
    # Startup bootstrap
    #
    # If the replay buffer is already warm, we train a SMALL number
    # of batches from the FULL replay buffer.
    #
    # We do NOT use the recent-generation sampler here.
    # ---------------------------------------------------------------

    startup_training_due = loopbuf.position_count >= MIN_BUFFER_TO_TRAIN

    if startup_training_due:
        print(
            "[Training] Replay buffer already contains "
            f"{loopbuf.position_count:,} positions."
        )

        print(
            "[Training] Buffer is warm; scheduling "
            f"{STARTUP_TRAINING_STEPS} startup training steps."
        )
    else:
        print(
            "[Training] Replay buffer below minimum "
            f"({loopbuf.position_count:,}/"
            f"{MIN_BUFFER_TO_TRAIN:,}); waiting for data."
        )

    # Normal training is triggered by NEW data.
    training_due = False

    # Receiver thread.
    message_queue = queue.Queue()

    receiver_stop_event = threading.Event()

    receiver_thread = threading.Thread(
        target=receiver_loop,
        args=(
            server,
            message_queue,
            receiver_stop_event,
        ),
        daemon=True,
        name="server-receiver",
    )

    receiver_thread.start()

    stop_requested = False
    next_reconnect_time = 0.0

    while True:
        # ===========================================================
        # 1. Process incoming messages without blocking the trainer.
        # ===========================================================

        received_data = get_message(
            message_queue,
            timeout=0.05,
        )

        if received_data is not None:
            log.start_batch()

            try:
                raw_data = json.loads(received_data)
            except json.JSONDecodeError as e:
                print(f"[Server] Ignoring malformed JSON message: {e}")
                continue

            received_data_string = str(raw_data)

            # -------------------------------------------------------
            # Network request.
            # -------------------------------------------------------

            if "RequestingNet" in received_data_string:
                # Self-play always uses the last net that passed the gate.
                # The training weights may have moved on since (a rejected
                # candidate does not reset them), so never send `model`.
                send_model_from_path(
                    accepted_model_path,
                    device,
                    server,
                )

            # -------------------------------------------------------
            # SPRT result.
            # -------------------------------------------------------

            test_result = extract_test_result(raw_data)

            if test_result is not None:
                if pending_candidate_gen is not None:
                    telemetry_log(lambda: gate_row(test_result, pending_candidate_gen))

                if pending_candidate_path is None or pending_candidate_gen is None:
                    print("[Training] Ignoring SPRT result with no pending candidate")

                elif test_result.get(
                    "accept_new_net",
                    False,
                ):
                    accepted_gen = pending_candidate_gen

                    accepted_model_path = promote_candidate_net(
                        pending_candidate_path,
                        accepted_gen,
                    )

                    print(f"[Training] Candidate accepted: {accepted_model_path}")
                    # Training is paused while a candidate is pending, so the schedule's
                    # position is exactly the samples this net was trained on.
                    ladder.maybe_run(
                        accepted_model_path, accepted_gen, schedule.samples_trained
                    )

                    pending_candidate_path = None
                    pending_candidate_gen = None

                else:
                    # AlphaGo Zero-style gating: the gate decides which net
                    # generates self-play data, not what the trainer keeps.
                    # Reverting here would discard every step since the last
                    # accepted net, so one generation (~16 steps) would have to
                    # clear the 55% bar on its own for training to progress.
                    print(
                        "[Training] Candidate rejected; self-play keeps "
                        f"{accepted_model_path}, training continues"
                    )

                    delete_file_if_exists(pending_candidate_path)

                    pending_candidate_path = None
                    pending_candidate_gen = None

            # -------------------------------------------------------
            # New data via path.
            # -------------------------------------------------------

            if "JobSendPath" in received_data_string:
                try:
                    data = extract_incoming_data_given_path(
                        loopbuf,
                        log,
                        raw_data,
                    )
                except OSError as e:
                    print(f"[Warning] skipping incoming replay file: {e!r}")
                    continue

                if loopbuf.position_count >= MIN_BUFFER_TO_TRAIN:
                    training_due = True

                    print("[Training] New generation received; training marked due.")
                else:
                    print(
                        "[Training] New data received, but replay "
                        "buffer is still below training minimum."
                    )

            # -------------------------------------------------------
            # New data via bytes.
            # -------------------------------------------------------

            if "JobSendData" in received_data_string:
                try:
                    data = extract_incoming_data_given_bytes(
                        loopbuf,
                        log,
                        raw_data,
                    )
                except OSError as e:
                    print(f"[Warning] skipping incoming replay file: {e!r}")
                    continue

                if loopbuf.position_count >= MIN_BUFFER_TO_TRAIN:
                    training_due = True

                    print("[Training] New generation received; training marked due.")
                else:
                    print(
                        "[Training] New data received, but replay "
                        "buffer is still below training minimum."
                    )

            # -------------------------------------------------------
            # Explicit shutdown.
            # -------------------------------------------------------

            if "StopServer" in received_data_string:
                print("[Server] Stop requested.")
                stop_requested = True

        # ===========================================================
        # 2. STARTUP BOOTSTRAP TRAINING
        #
        # This happens independently of the generator sending a file.
        # It uses the FULL replay buffer and only 16 steps.
        # ===========================================================

        if (
            startup_training_due
            and loopbuf.position_count >= MIN_BUFFER_TO_TRAIN
            and pending_candidate_path is None
        ):
            candidate_gen = accepted_gen + 1

            print(
                "[Training] Starting startup bootstrap training "
                f"for generation {candidate_gen}."
            )

            candidate_path = train_round_or_skip(
                model=model,
                starting_gen=candidate_gen,
                server=server,
                loopbuf=loopbuf,
                train_settings=train_settings,
                op=op,
                schedule=schedule,
                log=log,
                data=None,
                startup=True,
            )

            startup_training_due = False

            if candidate_path is not None:
                pending_candidate_path = candidate_path
                pending_candidate_gen = candidate_gen

                print(
                    f"[Training] Startup candidate {candidate_gen} is now pending SPRT."
                )

        # ===========================================================
        # 3. NORMAL NEW-GENERATION TRAINING
        #
        # Only actual new data makes this due.
        # ===========================================================

        if (
            training_due
            and loopbuf.position_count >= MIN_BUFFER_TO_TRAIN
            and pending_candidate_path is None
        ):
            candidate_gen = accepted_gen + 1

            print(f"[Training] Starting training for generation {candidate_gen}.")

            candidate_path = train_round_or_skip(
                model=model,
                starting_gen=candidate_gen,
                server=server,
                loopbuf=loopbuf,
                train_settings=train_settings,
                op=op,
                schedule=schedule,
                log=log,
                data=data if "data" in locals() else None,
                startup=False,
            )

            training_due = False

            if candidate_path is not None:
                pending_candidate_path = candidate_path
                pending_candidate_gen = candidate_gen

                print(f"[Training] Candidate {candidate_gen} is now pending SPRT.")

        # ===========================================================
        # 4. Shutdown.
        # ===========================================================

        if stop_requested:
            receiver_stop_event.set()
            server.close()

            print("[Server] Connection closed.")
            break

        # A dropped connection must not stop training: reconnect, re-verify and
        # restart the receiver. Messages sent while disconnected are lost; a
        # pending candidate stays pending until the server reports on it.
        if receiver_stop_event.is_set() and time.time() >= next_reconnect_time:
            print("[Server] Connection lost; reconnecting.")
            server.close()
            try:
                connected = server.try_connect()
                if connected:
                    get_verification(server, "PythonTraining")
            except (OSError, ConnectionError, json.JSONDecodeError) as e:
                print(f"[Server] Reconnect failed: {e!r}")
                connected = False

            if connected:
                receiver_stop_event = threading.Event()
                receiver_thread = threading.Thread(
                    target=receiver_loop,
                    args=(
                        server,
                        message_queue,
                        receiver_stop_event,
                    ),
                    daemon=True,
                    name="server-receiver",
                )
                receiver_thread.start()
                print("[Server] Reconnected.")
            else:
                next_reconnect_time = time.time() + RECONNECT_INTERVAL_S


def extract_incoming_data_given_bytes(
    loopbuf,
    log,
    raw_data,
):
    bin_data = raw_data["purpose"]["JobSendData"][0]
    off_data = raw_data["purpose"]["JobSendData"][1]
    meta_data = raw_data["purpose"]["JobSendData"][2]

    bin_data, off_data, meta_data = (
        bytes(dict(bin_data)["BinFile"]),
        bytes(dict(off_data)["OffFile"]),
        bytes(dict(meta_data)["MetaDataFile"]),
    )

    data_dir("python_client_games")

    # nanoseconds: two files arriving within the same second must not overwrite each other
    path = data_path(f"python_client_games/temp_games_{time.time_ns()}")

    with open(path + ".bin", "wb") as file:
        file.write(bin_data)

    with open(path + ".off", "wb") as file:
        file.write(off_data)

    decoded_string = meta_data.decode("utf-8")
    data = json.loads(decoded_string)

    with open(path + ".json", "w") as file:
        json.dump(
            data,
            file,
            indent=4,
        )

    with open(DATAFILE_PATH, "a") as f:
        f.write(path + "\n")

    print(path)
    record_received_file(path, bin_data, RUN_IDENTITY)

    data = load_file(path)
    telemetry_log(lambda: SELFPLAY_HEALTH.file_row(data))

    loopbuf.append(
        log,
        data,
    )

    print(
        "[loaded files] buffer size:",
        loopbuf.position_count,
    )

    log.finished_data()

    try:
        log.save(LOG_NPZ_PATH)
    except Exception:
        print("[Warning] failed to save log.npz")

    return data


RECONNECT_INTERVAL_S = 5.0


def reopen_buffer_files(loopbuf):
    """Replace every replay-buffer file with freshly opened handles.

    The buffer keeps files open for the whole run. On a network drive the old
    handles can go stale after the share reconnects, and reads then fail
    (Windows reports e.g. "network name deleted" as EINVAL). A file that cannot be
    reopened is dropped from the buffer, with the reason printed.
    """
    kept = []
    for old in loopbuf.files:
        base = os.path.splitext(str(old.info.bin_path))[0]
        try:
            old.close()
        except OSError:
            pass
        try:
            kept.append(load_file(base))
        except (OSError, AssertionError) as e:
            print(f"[Warning] dropping replay file {base} from the buffer: {e!r}")
            loopbuf.position_count -= len(old.positions)
            loopbuf.simulation_count -= old.info.simulation_count
    loopbuf.files = kept


def train_round_or_skip(**kwargs):
    """Run one training round on fresh file handles; skip the round on an I/O error."""
    loopbuf = kwargs["loopbuf"]
    reopen_buffer_files(loopbuf)
    if loopbuf.position_count < MIN_BUFFER_TO_TRAIN:
        print("[Training] Buffer below minimum after dropping unreadable files.")
        return None
    try:
        return full_train_and_send(**kwargs)
    except OSError as e:
        print(f"[Warning] training round skipped after an I/O error: {e!r}")
        return None


def full_train_and_send(
    model,
    starting_gen,
    server,
    loopbuf,
    train_settings,
    op,
    schedule,
    log,
    data=None,
    startup=False,
):
    if loopbuf.position_count < MIN_BUFFER_TO_TRAIN:
        return None

    if startup:
        (
            train_sampler,
            test_sampler,
            last_gen_test_sampler,
        ) = initialise_startup_samplers(loopbuf)

        num_steps_training = STARTUP_TRAINING_STEPS

        model.train()

        print(f"[Training] startup bootstrap: {num_steps_training} full-replay steps")

        train_startup(
            model=model,
            train_settings=train_settings,
            op=op,
            schedule=schedule,
            log=log,
            train_sampler=train_sampler,
            num_steps_training=num_steps_training,
        )

    else:
        (
            train_sampler,
            recent_train_sampler,
            test_sampler,
            last_gen_test_sampler,
        ) = initialise_samplers(loopbuf)

        if data is None:
            print(
                "[Training] WARNING: normal training requested "
                "without a data object; using minimum training steps."
            )

            num_steps_training = MIN_TRAINING_STEPS
        else:
            num_steps_training = get_num_steps_training(data)

        model.train()

        print("[Training] training model!")
        print(
            "[Training] new positions:",
            len(data.positions) if data is not None else 0,
        )
        print(
            "[Training] num_steps_training:",
            num_steps_training,
        )

        train_net(
            model=model,
            train_settings=train_settings,
            op=op,
            schedule=schedule,
            log=log,
            train_sampler=train_sampler,
            recent_train_sampler=recent_train_sampler,
            num_steps_training=num_steps_training,
        )

    save_trainer_state(
        model,
        op,
        schedule,
        accepted_gen=starting_gen - 1,
        candidate_gen=starting_gen,
    )

    entropy = test_net(
        model,
        train_settings,
        log,
        test_sampler,
        last_gen_test_sampler,
    )
    telemetry_log(
        lambda: {
            **logger_row(log, ("test", "last gen test")),
            **entropy,
            **buffer_row(loopbuf),
            "train/candidate_gen": starting_gen,
        }
    )

    log.finished_data()

    try:
        log.save(LOG_NPZ_PATH)
    except Exception:
        print("[Warning] failed to save log.npz")

    model_path = save_candidate_net(
        model,
        starting_gen,
    )

    if send_new_net(
        model_path,
        model,
        server,
    ):
        return model_path

    delete_file_if_exists(model_path)

    return None


def train_startup(
    model,
    train_settings,
    op,
    schedule,
    log,
    train_sampler,
    num_steps_training,
):
    """
    Startup bootstrap training.

    IMPORTANT:
    This sampler is the full replay sampler, NOT the recent-generation
    sampler. Therefore the 16 startup batches are distributed across the
    replay buffer rather than hammering the final 2k-5k position file.
    """
    for step in range(num_steps_training):
        if step != 0:
            log.start_batch()

        batch = train_sampler.next_batch()

        scheduled_train_step(train_settings, batch, model, op, schedule, log)

    train_sampler.close()


def send_new_net(
    model_path,
    model,
    server,
):
    if is_duplicate_net(model):
        print("[Training] Network weights unchanged; skipping duplicate net send")

        return False

    msg = {"NewNetworkPath": model_path}

    server.send(msg)

    return send_net_in_bytes(
        model,
        server,
        force=True,
    )


def save_candidate_net(
    model,
    starting_gen,
):
    model_path = data_path(f"nets/tz_candidate_{starting_gen}.pt")

    print(model_path)

    model.eval()

    with torch.no_grad():
        torch.jit.save(
            model,
            model_path,
        )

    if not os.path.exists(DATAFILE_PATH):
        with open(DATAFILE_PATH, "w"):
            pass

    return model_path


def promote_candidate_net(
    candidate_path,
    accepted_gen,
):
    accepted_path = data_path(f"nets/tz_{accepted_gen}.pt")

    if os.path.abspath(candidate_path) != os.path.abspath(accepted_path):
        os.replace(
            candidate_path,
            accepted_path,
        )

    with open(
        TRAINING_LOG_PATH,
        "a",
    ) as f:
        f.write(accepted_path + "\n")

    return accepted_path


def delete_file_if_exists(path):
    if path and os.path.exists(path):
        os.remove(path)


def extract_test_result(raw_data):
    purpose = raw_data.get(
        "purpose",
        {},
    )

    if isinstance(purpose, dict) and "TestResult" in purpose:
        return purpose["TestResult"]

    return None


def test_net(
    model,
    train_settings,
    log,
    test_sampler,
    last_gen_test_sampler,
):
    with torch.no_grad():
        model.eval()

        test_batch = test_sampler.next_batch()

        set_policy_weight(train_settings, test_batch)
        train_settings.evaluate_batch(
            network=model,
            batch=test_batch,
            log_prefix="test",
            logger=log,
        )
        entropy = {}
        if TELEMETRY is not None and TELEMETRY.enabled:
            entropy = policy_entropy_row(model, test_batch, "test")

        last_gen_test_batch = last_gen_test_sampler.next_batch()

        set_policy_weight(train_settings, last_gen_test_batch)
        train_settings.evaluate_batch(
            network=model,
            batch=last_gen_test_batch,
            log_prefix="last gen test",
            logger=log,
        )

    test_sampler.close()
    last_gen_test_sampler.close()
    return entropy


def train_net(
    model,
    train_settings,
    op,
    schedule,
    log,
    train_sampler,
    recent_train_sampler,
    num_steps_training,
):
    recent_interval = max(
        1,
        round(1 / RECENT_BATCH_FRACTION),
    )

    for gen in range(num_steps_training):
        if gen != 0:
            log.start_batch()

        use_recent = RECENT_BATCH_FRACTION > 0 and gen % recent_interval == 0

        sampler = recent_train_sampler if use_recent else train_sampler

        batch = sampler.next_batch()

        scheduled_train_step(train_settings, batch, model, op, schedule, log)

    train_sampler.close()
    recent_train_sampler.close()


def get_num_steps_training(data):
    """
    Normal generation training.

    A 2k-5k position file therefore produces approximately:

        2k -> max(16, ~2) = 16 steps
        5k -> max(16, ~5) = 16 steps

    With 50% recent sampling, the newest generation is therefore seen
    roughly 8 batches = 16,384 sampled positions at minimum.

    This is substantially less aggressive than the previous startup
    behaviour, where a 328k-position buffer generated ~321 steps with
    ~160 recent batches.
    """
    new_positions = len(data.positions)

    recent_fraction = max(
        RECENT_BATCH_FRACTION,
        1e-6,
    )

    num_steps_training = (new_positions / BATCH_SIZE) * EPOCHS_PER_GEN / recent_fraction

    num_steps_training = max(
        num_steps_training,
        MIN_TRAINING_STEPS,
    )

    return int(num_steps_training)


def initialise_samplers(loopbuf):
    train_sampler = loopbuf.sampler(
        batch_size=BATCH_SIZE,
        unroll_steps=None,
        include_final=False,
        random_symmetries=False,
        only_last_gen=False,
        test=False,
    )

    recent_train_sampler = loopbuf.sampler(
        batch_size=BATCH_SIZE,
        unroll_steps=None,
        include_final=False,
        random_symmetries=False,
        only_last_gen=True,
        test=False,
    )

    test_sampler = loopbuf.sampler(
        batch_size=BATCH_SIZE,
        unroll_steps=None,
        include_final=False,
        random_symmetries=False,
        only_last_gen=False,
        test=True,
    )

    last_gen_test_sampler = loopbuf.sampler(
        batch_size=BATCH_SIZE,
        unroll_steps=None,
        include_final=False,
        random_symmetries=False,
        only_last_gen=True,
        test=True,
    )

    return (
        train_sampler,
        recent_train_sampler,
        test_sampler,
        last_gen_test_sampler,
    )


def initialise_startup_samplers(loopbuf):
    """
    Startup-specific samplers.

    The training sampler covers the FULL replay buffer.

    There is deliberately no recent training sampler here. Startup
    training is a bootstrap operation, not a full training generation.
    """
    train_sampler = loopbuf.sampler(
        batch_size=BATCH_SIZE,
        unroll_steps=None,
        include_final=False,
        random_symmetries=False,
        only_last_gen=False,
        test=False,
    )

    test_sampler = loopbuf.sampler(
        batch_size=BATCH_SIZE,
        unroll_steps=None,
        include_final=False,
        random_symmetries=False,
        only_last_gen=False,
        test=True,
    )

    last_gen_test_sampler = loopbuf.sampler(
        batch_size=BATCH_SIZE,
        unroll_steps=None,
        include_final=False,
        random_symmetries=False,
        only_last_gen=True,
        test=True,
    )

    return (
        train_sampler,
        test_sampler,
        last_gen_test_sampler,
    )


def extract_incoming_data_given_path(
    loopbuf,
    log,
    raw_data,
):
    file_path = raw_data["purpose"]["JobSendPath"]

    with open(
        DATAFILE_PATH,
        "a",
    ) as f:
        f.write(file_path + "\n")

    data = load_file(file_path)
    telemetry_log(lambda: SELFPLAY_HEALTH.file_row(data))

    loopbuf.append(
        log,
        data,
    )

    print(
        "[loaded files] buffer size:",
        loopbuf.position_count,
    )

    log.finished_data()

    try:
        log.save(LOG_NPZ_PATH)
    except Exception:
        print("[Warning] failed to save log.npz")

    return data


def is_duplicate_net(model):
    return LAST_SENT_NET_CHECKSUM == model_weights_checksum(model)


def send_net_in_bytes(
    model,
    server,
    force=False,
):
    global LAST_SENT_NET_CHECKSUM

    checksum = model_weights_checksum(model)

    if not force and LAST_SENT_NET_CHECKSUM == checksum:
        print("[Training] Network weights unchanged; skipping duplicate net send")

        return False

    net_send = serialise_net(model)

    msg = {"NewNetworkData": [byte for byte in net_send]}

    server.send(msg)

    LAST_SENT_NET_CHECKSUM = checksum

    return True


def send_model_from_path(
    model_path,
    device,
    server,
):
    print(f"[Training] Sending accepted net: {model_path}")

    accepted_model = torch.jit.load(
        model_path,
        map_location=device,
    ).eval()

    return send_net_in_bytes(
        accepted_model,
        server,
        force=True,
    )


def load_previous_data(
    data_paths,
    loopbuf,
):
    log = Logger()

    if data_paths:
        data_paths = list(dict.fromkeys(data_paths))

        for file in data_paths:
            try:
                data = load_file(file)

                loopbuf.append(
                    None,
                    data,
                )

            except Exception as e:
                print(f"[Warning] skipping replay file {file}: {e}")
                continue

    if os.path.exists(LOG_NPZ_PATH):
        try:
            log = log.load(LOG_NPZ_PATH)

            print("loaded log")

        except Exception as e:
            print("[Error]", e)

            os.remove(LOG_NPZ_PATH)

    print(
        "[loaded files] buffer size:",
        loopbuf.position_count,
    )

    return log


def get_verification(
    server,
    identity,
):
    while True:
        server.send({"Initialise": identity})

        received_data = server.receive()

        if received_data == "":
            raise ConnectionError("Server disconnected during verification.")

        received_data = json.loads(received_data)

        purpose = str(received_data)

        if "IdentityConfirmation" in purpose and identity in purpose:
            break

    print("identity verified")


def get_previous_data_paths():
    data_paths = None

    if os.path.isfile(DATAFILE_PATH):
        with open(
            DATAFILE_PATH,
            "r",
        ) as f:
            data_paths = f.readlines()

        data_paths = [item.strip() for item in data_paths if item != ""]

        data_paths = [
            x
            for x in data_paths
            if os.path.isfile(x.strip() + ".bin")
            and os.path.isfile(x.strip() + ".json")
            and os.path.isfile(x.strip() + ".off")
        ]

    else:
        with open(
            DATAFILE_PATH,
            "w+",
        ):
            pass

    return data_paths


def get_model_path(training_nets):
    if os.path.isfile(TRAINING_LOG_PATH):
        with open(
            TRAINING_LOG_PATH,
            "r",
        ) as f:
            recorded_sessions = f.readlines()

        recorded_sessions = [item.strip() for item in recorded_sessions if item != ""]

        if recorded_sessions != training_nets:
            with open(
                TRAINING_LOG_PATH,
                "w",
            ) as f:
                f.write("\n".join(training_nets) + "\n")

            recorded_sessions = training_nets

    else:
        with open(
            TRAINING_LOG_PATH,
            "w",
        ) as f:
            f.write(training_nets[-1] + "\n")

        recorded_sessions = training_nets

    model_path = recorded_sessions[-1].strip()

    return model_path


def check_net_exists(
    device,
    pattern,
):
    training_nets = []
    net_id = {}

    for net in os.listdir(data_dir("nets")):
        match = re.match(
            pattern,
            net,
        )

        if match:
            group = int(match.groups()[0])

            net_id[data_path(f"nets/{net}")] = group

            training_nets.append(net)

    net_id = dict(
        sorted(
            net_id.items(),
            key=lambda x: x[1],
        )
    )

    training_nets = list(net_id.keys())

    if not os.listdir(data_dir("nets")) or not training_nets:
        with torch.no_grad():
            net = torch.jit.script(network.TrueNetXS(num_hidden=64).to(device)).eval()

            torch.jit.save(
                net,
                data_path("nets/tz_0.pt"),
            )

        with open(
            TRAINING_LOG_PATH,
            "w+",
        ) as f:
            f.write(data_path("nets/tz_0.pt") + "\n")

        # A brand new run must not inherit metrics from a previous run.
        if os.path.exists(LOG_NPZ_PATH):
            os.remove(LOG_NPZ_PATH)
            print("[Training] New run: reset log.npz")

        training_nets.append(data_path("nets/tz_0.pt"))

    return training_nets


if __name__ == "__main__":
    main()
