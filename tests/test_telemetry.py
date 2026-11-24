"""Telemetry must report what happened and never get in training's way.

Run from `src/python`:

    PYTHONPATH=. uv run python -m unittest discover -s ../../tests
"""

import codecs
import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.loop import LoopBuffer

import client
import network
import telemetry

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "kzero", "kzero_chess"
)

# Lines exactly as the Rust binaries print them (server.rs, executor.rs).
NPS_LINE = "[Statistics-nps] 52000"
EVALS_LINE = "[Statistics-evals] 104000"
EXEC_LINE = (
    "[executor-0] 30s: 1520 batches, mean batch 430 (min 12, max 2048), 21786 evals/s; "
    "time: waiting 2%, fill 7%, forward+copy 63%, unpack 15% (slice 8%, send 7%)"
)


class FakeWandb:
    """Records calls; stands in for the wandb module."""

    def __init__(self):
        self.rows = []
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return object()

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, row):
        self.rows.append(row)

    def finish(self):
        pass


def online(fake, run_cfg=None):
    observe = telemetry.load_observe_config("/nonexistent/observe.toml")
    observe["wandb"]["mode"] = "offline"
    return telemetry.Telemetry(observe, run_cfg, wandb_module=fake)


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        patcher = mock.patch.object(
            telemetry, "data_path", lambda rel: os.path.join(self.dir, rel)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_off_by_default_and_without_wandb(self):
        t = telemetry.Telemetry(telemetry.load_observe_config("/nonexistent"))
        self.assertFalse(t.enabled)
        t.log({"x": 1.0})  # no-op, no error
        observe = telemetry.load_observe_config("/nonexistent")
        observe["wandb"]["mode"] = "online"
        with mock.patch.dict("sys.modules", {"wandb": None}):
            self.assertFalse(telemetry.Telemetry(observe).enabled)

    def test_rows_carry_samples_trained_and_drop_nan(self):
        fake = FakeWandb()
        t = online(fake)
        t.set_samples(4096)
        t.log({"a": 1.0, "b": float("nan"), "c": None})
        self.assertEqual(fake.rows, [{"a": 1.0, "samples_trained": 4096}])

    def test_run_id_persists_and_group_is_the_parent(self):
        cfg = {"run": {"run_id": "branch-pcr", "parent_run_id": "base"}}
        fake = FakeWandb()
        online(fake, cfg)
        wandb_id = fake.init_kwargs["id"]
        self.assertEqual(fake.init_kwargs["group"], "base")
        self.assertEqual(fake.init_kwargs["resume"], "allow")
        with open(os.path.join(self.dir, "hidden", "wandb_run_id.branch-pcr")) as f:
            self.assertEqual(f.read(), wandb_id)
        again = FakeWandb()
        online(again, cfg)
        self.assertEqual(again.init_kwargs["id"], wandb_id)  # a restart resumes it
        root = FakeWandb()
        online(root, {"run": {"run_id": "base", "parent_run_id": ""}})
        self.assertEqual(root.init_kwargs["group"], "base")
        self.assertNotEqual(root.init_kwargs["id"], wandb_id)  # never the branch's run

    def test_a_failing_wandb_never_raises(self):
        fake = FakeWandb()
        t = online(fake)
        fake.log = mock.Mock(side_effect=RuntimeError("network"))
        t.log({"a": 1.0})

    def test_observe_unknown_keys_are_ignored(self):
        path = os.path.join(self.dir, "observe.toml")
        with open(path, "w") as f:
            f.write('[wandb]\nmode = "offline"\nbogus = 1\n[tail]\nlogs = ["a.log"]\n')
        cfg = telemetry.load_observe_config(path)
        self.assertEqual(cfg["wandb"]["mode"], "offline")
        self.assertEqual(cfg["tail"]["logs"], ["a.log"])
        self.assertNotIn("bogus", cfg["wandb"])


class RustLogTests(unittest.TestCase):
    def test_parse_statistics_and_executor_lines(self):
        self.assertEqual(
            telemetry.parse_rust_line(NPS_LINE), {"selfplay/nodes_per_s": 52000.0}
        )
        self.assertEqual(
            telemetry.parse_rust_line(EVALS_LINE), {"selfplay/evals_per_s": 104000.0}
        )
        row = telemetry.parse_rust_line(EXEC_LINE)
        self.assertEqual(row["executor/executor-0/mean_batch"], 430.0)
        self.assertEqual(row["executor/executor-0/evals_per_s"], 21786.0)
        self.assertEqual(row["executor/executor-0/forward_pct"], 63.0)
        self.assertEqual(telemetry.parse_rust_line("[Server] New connection"), {})

    def follow(self, encoding):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        path = os.path.join(d, "server.log")
        bom = codecs.BOM_UTF16_LE if encoding == "utf-16-le" else b""
        with open(path, "wb") as f:
            f.write(bom + "old line before start\r\n".encode(encoding))
        fake = FakeWandb()
        tailer = telemetry.LogTailer(online(fake), [path])
        tailer.poll()  # starts at the end: history is not replayed
        with open(path, "ab") as f:
            f.write(f"{NPS_LINE}\r\n{EVALS_LINE}\r\n{NPS_LINE}".encode(encoding))
        tailer.poll()
        with open(path, "ab") as f:
            f.write("\r\n".encode(encoding))  # completes the last line
        tailer.poll()
        return tailer.flush()

    def test_tailer_utf8_and_powershell_utf16(self):
        for enc in ("utf-8", "utf-16-le"):
            with self.subTest(enc):
                row = self.follow(enc)
                self.assertEqual(row["selfplay/nodes_per_s"], 52000.0)
                self.assertEqual(row["selfplay/evals_per_s"], 104000.0)
                self.assertAlmostEqual(row["selfplay/nodes_per_eval"], 0.5)

    def test_missing_log_is_not_an_error(self):
        tailer = telemetry.LogTailer(online(FakeWandb()), ["/nonexistent/x.log"])
        tailer.poll()
        self.assertEqual(tailer.flush(), {})


class SelfplayHealthTests(unittest.TestCase):
    def test_fixture_file(self):
        """The fixture's games (fixture_games.txt): 7 games, 5 drawn, four start from the
        standard position with f2f3, e2e4, g1f3 and h2h4. The game starting after 1. f3
        (Black to move) is not an opening first move and must not count as e2e4."""
        f = DataFile.open(Game.find("chess"), FIXTURE)
        self.addCleanup(f.close)
        row = telemetry.SelfplayHealth().file_row(f)
        self.assertEqual(row["selfplay/games"], 7)
        self.assertAlmostEqual(row["selfplay/draw_rate"], 5 / 7)
        self.assertEqual(row["selfplay/game_length_max"], 15)
        self.assertEqual(row["first_move/distinct"], 4)
        for mv in ("f2f3", "e2e4", "g1f3", "h2h4"):
            self.assertAlmostEqual(row[f"first_move/share/{mv}"], 0.25)
        self.assertAlmostEqual(row["first_move/entropy_bits"], 2.0)


class TrainerMetricTests(unittest.TestCase):
    def test_logger_row_picks_the_current_batch(self):
        log = Logger()
        log.start_batch()
        log.log("loss-value", "train value", 0.5)
        log.log("loss-value", "test value", 0.7)
        log.start_batch()
        log.log("loss-value", "train value", 0.4)
        log.log("grad_norm", "max", 3.0)
        self.assertEqual(
            telemetry.logger_row(log, ("train",), groups=("grad_norm",)),
            {"loss-value/train value": 0.4, "grad_norm/max": 3.0},
        )

    def test_policy_entropy_on_real_batch(self):
        buf = LoopBuffer(Game.find("chess"), target_positions=10**6, test_fraction=0.0)
        buf.append(None, client.load_file(FIXTURE))
        sampler = buf.sampler(
            batch_size=16,
            unroll_steps=None,
            include_final=False,
            random_symmetries=False,
            only_last_gen=False,
            test=False,
        )
        self.addCleanup(sampler.close)
        batch = sampler.next_batch()
        torch.manual_seed(0)
        model = torch.jit.script(network.TrueNetXS(num_resBlocks=1, num_hidden=8)).eval()
        with torch.no_grad():
            row = client.policy_entropy_row(model, batch, "test")
        has = (batch.policy_values != -1).any(dim=1)
        legal = (batch.policy_values != -1).sum(dim=1)[has].float()
        net_h = row["policy-entropy/test net"]
        # Entropy over legal moves lies in [0, log(#legal)].
        self.assertGreaterEqual(net_h, 0.0)
        self.assertLessEqual(net_h, torch.log(legal).mean().item() + 1e-5)
        self.assertGreaterEqual(row["policy-entropy/test target"], 0.0)

    def test_gate_row(self):
        row = telemetry.gate_row({"elo": [-10.0, 5.0, 20.0], "accept_new_net": False}, 7)
        self.assertEqual(row["gate/accepted"], 0.0)
        self.assertEqual(row["gate/elo"], 5.0)
        self.assertEqual(row["gate/candidate_gen"], 7)


if __name__ == "__main__":
    unittest.main()
