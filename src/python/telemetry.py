"""Optional Weights & Biases telemetry for the trainer.

Configured by ``<data root>/observe.toml`` (template: ``observe.example.toml``), which is
NOT part of the run identity: how a run is watched can change mid-run, what it is cannot.
W&B is on by default (project ``truezero``, your default entity). Online logging needs a
W&B login (``wandb login`` once, or ``WANDB_API_KEY``); without one the trainer never
prompts, it logs offline instead (``wandb sync`` uploads later). With ``mode = "off"`` or
without the ``wandb`` package every call here is a no-op and training is unaffected.

Everything is logged against ``samples_trained`` (the LR schedule's position), never
wall-clock, so runs compare across hardware and restarts. One W&B run per TrueZero run:
its id is kept in ``hidden/wandb_run_id.<run_id>`` so a restarted trainer resumes the same W&B run;
the W&B group is the parent run (or the run itself), so branches sit with their parent.

Also here, because they feed the logs:
- ``logger_row``: the lib Logger's values for the current batch (losses, accuracies).
- ``SelfplayHealth``: per replay file, game length, draw rate, move-limit rate and a
  rolling first-move distribution (a collapse to one opening shows here first).
- ``LogTailer``: follows the Rust processes' stdout logs (``[Statistics-*]`` and executor
  timing lines) and forwards averages, so Rust needs no W&B dependency.
"""

import codecs
import math
import os
import re
import threading
import time
import uuid
from collections import Counter, deque

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

from paths import data_path

OBSERVE_DEFAULTS = {
    "wandb": {"mode": "online", "project": "truezero", "entity": ""},
    "tail": {"logs": [], "flush_seconds": 60.0},
    "ladder": {
        "enabled": False,
        "every_accepted_nets": 5,
        "binary": "",
        "anchors": [],
        "games": 100,
        "nodes": 200,
        "openings": "hidden/8moves_v3.epd",
        "tactics": "",
        "tactics_nodes": 400,
    },
}


def load_observe_config(path=None):
    """observe.toml merged over the defaults. Unknown keys are reported, not fatal: this
    file only controls observation."""
    path = path or data_path("observe.toml")
    cfg = {k: dict(v) for k, v in OBSERVE_DEFAULTS.items()}
    if not os.path.exists(path):
        return cfg
    if tomllib is None:
        print("[telemetry] observe.toml needs Python 3.11+ (tomllib); telemetry is off")
        return cfg
    with open(path, "rb") as f:
        user = tomllib.load(f)
    for table, values in user.items():
        if table not in cfg or not isinstance(values, dict):
            print(f"[telemetry][Warning] {path}: unknown table [{table}] ignored")
            continue
        for key, value in values.items():
            if key not in cfg[table]:
                print(f"[telemetry][Warning] {path}: unknown key [{table}] {key} ignored")
                continue
            cfg[table][key] = value
    return cfg


def wandb_key_available(environ=os.environ, home=None):
    """Whether W&B has an API key, without contacting it or prompting: WANDB_API_KEY, or an
    api.wandb.ai entry in the netrc file `wandb login` writes (~/.netrc, ~/_netrc on
    Windows, or $NETRC)."""
    if environ.get("WANDB_API_KEY"):
        return True
    home = home or os.path.expanduser("~")
    candidates = [environ.get("NETRC"), os.path.join(home, ".netrc"), os.path.join(home, "_netrc")]
    host = environ.get("WANDB_BASE_URL", "https://api.wandb.ai").split("//")[-1].split("/")[0]
    for path in filter(None, candidates):
        try:
            with open(path) as f:
                if host in f.read():
                    return True
        except OSError:
            pass
    return False


class Telemetry:
    """Thread-safe front for W&B. Never raises into the trainer."""

    def __init__(
        self,
        observe,
        run_cfg=None,
        identity=None,
        wandb_module=None,
        key_check=wandb_key_available,
    ):
        self.samples_trained = 0
        self._lock = threading.Lock()
        self._wandb = None
        self._run = None
        mode = observe["wandb"]["mode"]
        if mode == "off":
            print("[telemetry] W&B off (observe.toml [wandb] mode)")
            return
        if mode not in ("online", "offline"):
            print(f"[telemetry][Warning] unknown W&B mode {mode!r}; telemetry is off")
            return
        if mode == "online" and not key_check():
            # wandb.init would stop and ask for a key; a trainer must never wait on that.
            print(
                "[telemetry] no W&B login found (run `wandb login` once, or set "
                "WANDB_API_KEY); logging offline for now, upload later with `wandb sync`"
            )
            mode = "offline"
        if wandb_module is None:
            try:
                import wandb as wandb_module
            except ImportError:
                print("[telemetry][Warning] wandb is not installed; telemetry is off")
                return
        try:
            self._run = self._init(wandb_module, observe, run_cfg, identity, mode)
            self._wandb = wandb_module
        except Exception as e:
            print(f"[telemetry][Warning] W&B init failed, telemetry is off: {e!r}")

    @staticmethod
    def _init(wandb, observe, run_cfg, identity, mode):
        run = (run_cfg or {}).get("run", {})
        run_id = run.get("run_id", "truezero")
        # Keyed by run_id, so a branch started from a copy of its parent's hidden/ gets
        # its own W&B run instead of resuming the parent's.
        id_path = data_path(f"hidden/wandb_run_id.{run_id}")
        if os.path.exists(id_path):
            with open(id_path) as f:
                wandb_id = f.read().strip()
        else:
            wandb_id = uuid.uuid4().hex[:12]
            os.makedirs(os.path.dirname(id_path), exist_ok=True)
            with open(id_path, "w") as f:
                f.write(wandb_id)
        config = {"run_toml": run_cfg or {}}
        if identity is not None:
            config.update(
                config_sha256=identity.config_sha256,
                git_sha=identity.git_sha,
                git_dirty=identity.git_dirty,
            )
        # wandb infers the program's path from the git root and the cwd with os.path.relpath,
        # which raises on Windows when they are on different mounts (a mapped drive U: whose
        # git root resolves to \\server\share). The commit is already in config (git_sha),
        # so skip wandb's own git and code capture unless the user asked for it.
        os.environ.setdefault("WANDB_DISABLE_GIT", "true")
        os.environ.setdefault("WANDB_DISABLE_CODE", "true")
        wrun = wandb.init(
            project=observe["wandb"]["project"],
            entity=observe["wandb"]["entity"] or None,
            mode=mode,
            id=wandb_id,
            resume="allow",
            name=run_id,
            group=run.get("parent_run_id") or run_id,
            config=config,
            dir=data_path("hidden"),
        )
        wandb.define_metric("samples_trained")
        wandb.define_metric("*", step_metric="samples_trained")
        print(f"[telemetry] W&B {mode} run {run_id} (id {wandb_id})")
        return wrun

    @property
    def enabled(self):
        return self._wandb is not None

    def set_samples(self, samples_trained):
        self.samples_trained = int(samples_trained)

    def log(self, metrics, samples_trained=None):
        """Log a dict of numbers (NaN and None are dropped) at `samples_trained`."""
        if not self.enabled:
            return
        row = {
            k: v
            for k, v in metrics.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }
        if not row:
            return
        row["samples_trained"] = (
            self.samples_trained if samples_trained is None else int(samples_trained)
        )
        try:
            with self._lock:
                self._wandb.log(row)
        except Exception as e:
            print(f"[telemetry][Warning] W&B log failed: {e!r}")

    def finish(self):
        if self.enabled:
            try:
                self._wandb.finish()
            except Exception:
                pass


def logger_row(log, prefixes, groups=()):
    """The lib Logger's values for its current batch whose name starts with one of
    `prefixes` (or whose group is in `groups`), as {"group/name": value}. lib's policy accuracy ("pol-acc/<p> acc") is
    averaged over positions with a policy target, i.e. full searches only (fast-search
    targets are masked)."""
    row = {}
    b = log.curr_batch
    if b < 0:
        return row
    for (group, name), values in log.data.items():
        if (name.startswith(prefixes) or group in groups) and b < len(values.values):
            v = float(values.values[b])
            if not math.isnan(v):
                row[f"{group}/{name}"] = v
    return row


def buffer_row(loopbuf, now=None):
    """Replay-buffer size and age: how stale the data being trained on is."""
    now = time.time() if now is None else now
    stamps = [f.info.timestamp for f in loopbuf.files]
    row = {"buffer/positions": loopbuf.position_count, "buffer/files": len(loopbuf.files)}
    if stamps:
        row["buffer/oldest_age_s"] = now - min(stamps)
        row["buffer/newest_age_s"] = now - max(stamps)
    return row


# ---------------------------------------------------------------------------
# Self-play health
# ---------------------------------------------------------------------------


_STARTPOS_BOOLS = None


def _is_startpos(position):
    """Whether a stored position is the standard start position (White to move), so its
    side-to-move policy index needs no flip and its move is an opening first move."""
    global _STARTPOS_BOOLS
    if _STARTPOS_BOOLS is None:
        import chess
        import numpy as np
        from decoder import convert_board

        tensor, _ = convert_board(chess.Board())
        _STARTPOS_BOOLS = tensor[8:].numpy().astype(bool)
        _STARTPOS_BOOLS = np.ascontiguousarray(_STARTPOS_BOOLS)
    return (
        position.input_scalars[0] == 1
        and position.input_bools.shape == _STARTPOS_BOOLS.shape
        and bool((position.input_bools.astype(bool) == _STARTPOS_BOOLS).all())
    )


def _first_move_name(index):
    try:
        from decoder import CONTENTS

        return CONTENTS[index].uci()
    except Exception:
        return str(index)


class SelfplayHealth:
    """Per replay file statistics, with the first move tracked over the last `window`
    games that began from the standard start position."""

    def __init__(self, window=2000):
        self.first_moves = deque(maxlen=window)

    def file_row(self, datafile):
        sims = datafile.simulations
        lengths, draws = [], 0
        for si in range(len(sims)):
            sim = sims[si]
            lengths.append(sim.move_count)
            start = datafile.load_position(sim.start_file_pi)
            if (
                start.played_mv is not None
                and start.played_mv >= 0
                and _is_startpos(start)
            ):
                self.first_moves.append(start.played_mv)
            # final_wdl is (win, draw, loss); a draw is the same from either side.
            if start.final_wdl[1] > 0.5:
                draws += 1
        n = len(lengths)
        # Only each game's first position is read: a file has ~25k positions and the
        # buffer may sit on a network drive.
        row = {"selfplay/games": n, "selfplay/positions": len(datafile.positions)}
        if n:
            lengths.sort()
            row.update(
                {
                    "selfplay/game_length_mean": sum(lengths) / n,
                    "selfplay/game_length_median": lengths[n // 2],
                    "selfplay/game_length_max": lengths[-1],
                    "selfplay/draw_rate": draws / n,
                    "selfplay/decisive_rate": 1 - draws / n,
                }
            )
        row.update(self.first_move_row())
        return row

    def first_move_row(self, top=8):
        counts = Counter(self.first_moves)
        total = sum(counts.values())
        if not total:
            return {}
        probs = [c / total for c in counts.values()]
        row = {
            "first_move/top_share": max(probs),
            "first_move/entropy_bits": -sum(p * math.log2(p) for p in probs),
            "first_move/distinct": len(counts),
        }
        for index, c in counts.most_common(top):
            row[f"first_move/share/{_first_move_name(index)}"] = c / total
        return row


# ---------------------------------------------------------------------------
# Rust log tailing
# ---------------------------------------------------------------------------

_STAT_RE = re.compile(r"\[Statistics-(nps|evals)\]\s+(\d+)")
_EXEC_RE = re.compile(
    r"\[(?P<name>[^\]]+)\]\s+[\d.]+s:\s+(?P<batches>\d+) batches, mean batch (?P<mean>[\d.]+) "
    r"\(min (?P<min>\d+), max (?P<max>\d+)\), (?P<evals>[\d.]+) evals/s; time: "
    r"waiting (?P<waiting>[\d.]+)%, fill (?P<fill>[\d.]+)%, forward\+copy (?P<fwd>[\d.]+)%, "
    r"unpack (?P<unpack>[\d.]+)%"
)


def parse_rust_line(line):
    """Metrics in one line of server/self-play stdout, or {}."""
    m = _STAT_RE.search(line)
    if m:
        key = "selfplay/nodes_per_s" if m.group(1) == "nps" else "selfplay/evals_per_s"
        return {key: float(m.group(2))}
    m = _EXEC_RE.search(line)
    if m:
        name = m.group("name")
        return {
            f"executor/{name}/mean_batch": float(m.group("mean")),
            f"executor/{name}/evals_per_s": float(m.group("evals")),
            f"executor/{name}/waiting_pct": float(m.group("waiting")),
            f"executor/{name}/fill_pct": float(m.group("fill")),
            f"executor/{name}/forward_pct": float(m.group("fwd")),
            f"executor/{name}/unpack_pct": float(m.group("unpack")),
        }
    return {}


class _FollowedFile:
    """Follows a growing text file from its end at start-up. Handles a missing file, a
    restart that truncates it, and PowerShell's UTF-16 output (Tee-Object / `>` in
    Windows PowerShell 5.1 write UTF-16 LE with a BOM)."""

    def __init__(self, path):
        self.path = path
        self.pos = None
        self.decoder = None
        self.buf = ""

    def read_lines(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return []
        if self.pos is None or size < self.pos:
            with open(self.path, "rb") as f:
                head = f.read(4)
            if head.startswith(codecs.BOM_UTF16_LE) or head.startswith(codecs.BOM_UTF16_BE):
                enc = "utf-16"
            else:
                enc = "utf-8"
            self.decoder = codecs.getincrementaldecoder(enc)(errors="replace")
            self.buf = ""
            if self.pos is None:
                # Start at the end, but at an even offset so UTF-16 stays aligned.
                self.pos = size - (size % 2 if enc == "utf-16" else 0)
                if enc == "utf-16":
                    self.decoder.decode(head[:2])  # consume the BOM's byte order
            else:
                self.pos = 0
        if size == self.pos:
            return []
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read(size - self.pos)
        except OSError:
            return []
        self.pos += len(data)
        self.buf += self.decoder.decode(data)
        *lines, self.buf = self.buf.split("\n")
        return [line.rstrip("\r") for line in lines]


class LogTailer:
    """Follows the Rust logs and logs per-key means every `flush_seconds`."""

    def __init__(self, telemetry, paths, flush_seconds=60.0, poll_seconds=1.0):
        self.telemetry = telemetry
        self.files = [_FollowedFile(p) for p in paths]
        self.flush_seconds = flush_seconds
        self.poll_seconds = poll_seconds
        self.sums = {}
        self.counts = {}
        self._stop = threading.Event()
        self._thread = None

    def poll(self):
        for f in self.files:
            for line in f.read_lines():
                for k, v in parse_rust_line(line).items():
                    self.sums[k] = self.sums.get(k, 0.0) + v
                    self.counts[k] = self.counts.get(k, 0) + 1

    def flush(self):
        if not self.sums:
            return {}
        row = {k: self.sums[k] / self.counts[k] for k in self.sums}
        nps, evals = row.get("selfplay/nodes_per_s"), row.get("selfplay/evals_per_s")
        if nps and evals:
            # Visits per network evaluation: >1 means the cache and terminals are saving work.
            row["selfplay/nodes_per_eval"] = nps / evals
        self.sums, self.counts = {}, {}
        self.telemetry.log(row)
        return row

    def _run(self):
        last = time.monotonic()
        while not self._stop.wait(self.poll_seconds):
            try:
                self.poll()
                if time.monotonic() - last >= self.flush_seconds:
                    self.flush()
                    last = time.monotonic()
            except Exception as e:  # a log hiccup must never stop training
                print(f"[telemetry][Warning] log tailer: {e!r}")

    def start(self):
        if not self.files or not self.telemetry.enabled:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True, name="log-tailer")
        self._thread.start()
        print(f"[telemetry] following {[f.path for f in self.files]}")
        return self

    def stop(self):
        self._stop.set()


def resolve_data_paths(paths):
    return [p if os.path.isabs(p) else data_path(p) for p in paths]


def gate_row(test_result, candidate_gen):
    """The gate's verdict (trainenginetest SPRT) as metrics."""
    lo, mid, hi = (test_result.get("elo") or (None, None, None))[:3]
    return {
        "gate/accepted": 1.0 if test_result.get("accept_new_net") else 0.0,
        "gate/elo": mid,
        "gate/elo_low": lo,
        "gate/elo_high": hi,
        "gate/candidate_gen": candidate_gen,
    }
