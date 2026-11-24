"""truezero.toml, runs and provenance: the trainer side of src/rust/runconfig.rs (same
rules; the overlay cases are shared through tests/fixtures/config_overlay_cases.json).

Run from `src/python`:

    PYTHONPATH=. uv run python -m unittest discover -s ../../tests
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import client
import paths
import runconfig

EXAMPLE_TEXT = runconfig.EXAMPLE_PATH.read_text()
CASES = Path(__file__).parent / "fixtures" / "config_overlay_cases.json"


def defaults():
    return runconfig.config_from_texts(EXAMPLE_TEXT, "")


def identity(sha="abc", config="a", dirty=False, run_id="truenet-128x10-20261124-1503"):
    return runconfig.StartIdentity(run_id, "truenet-128x10", config * 64, sha, dirty, "test")


class ConfigTests(unittest.TestCase):
    def test_example_reproduces_the_previous_trainer_defaults(self):
        """The example's [trainer] equals the trainer's built-in defaults."""
        names = [
            "BUFFER_SIZE",
            "BATCH_SIZE",
            "MIN_BUFFER_TO_TRAIN",
            "EPOCHS_PER_GEN",
            "RECENT_BATCH_FRACTION",
            "MIN_TRAINING_STEPS",
            "STARTUP_TRAINING_STEPS",
            "TEST_FRACTION",
            "LR_SCHEDULE",
            "LR_WARMUP_SAMPLES",
            "WEIGHT_DECAY",
            "POLICY_WEIGHT",
        ]
        before = {n: getattr(client, n) for n in names}
        weights_before = dict(client.LOSS_WEIGHTS)
        with mock.patch.multiple(client, **before), mock.patch.dict(client.LOSS_WEIGHTS):
            client.apply_run_config(defaults())
            self.assertEqual({n: getattr(client, n) for n in names}, before)
            self.assertEqual(client.LOSS_WEIGHTS, weights_before)

    def test_overlay_cases_shared_with_rust(self):
        cases = json.loads(CASES.read_text())["cases"]
        for case in cases:
            with self.subTest(case["name"]):
                if not case["ok"]:
                    with self.assertRaises(runconfig.ConfigError):
                        runconfig.config_from_texts(EXAMPLE_TEXT, case["user"])
                    continue
                cfg = runconfig.config_from_texts(EXAMPLE_TEXT, case["user"])
                for table, key, value in case.get("expect", []):
                    self.assertEqual(cfg[table][key], value)

    def test_trainer_ranges(self):
        for bad in (
            "[trainer]\nbatch_size = 2000\n",
            "[trainer]\ntest_fraction = 1.5\n",
            '[wandb]\nmode = "sometimes"\n',
        ):
            with self.subTest(bad), self.assertRaises(runconfig.ConfigError):
                runconfig.config_from_texts(EXAMPLE_TEXT, bad)

    def test_first_use_creates_truezero_toml(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        (d / runconfig.EXAMPLE_FILE).write_text(EXAMPLE_TEXT)
        loaded = runconfig.load_config(d)
        self.assertEqual((d / runconfig.CONFIG_FILE).read_text(), EXAMPLE_TEXT)
        self.assertEqual(loaded.config_sha256, hashlib.sha256(EXAMPLE_TEXT.encode()).hexdigest())
        # The user's edits are kept on later loads.
        (d / runconfig.CONFIG_FILE).write_text("[selfplay]\nvirtual_leaves = 2\n")
        self.assertEqual(runconfig.load_config(d).config["selfplay"]["virtual_leaves"], 2)

    def test_retired_env_vars_warn_and_are_ignored(self):
        w = runconfig.retired_env_var_warning({"TZ_LR_SCHEDULE": "0:1e-3", "TZ_SERVER_PORT": "1"})
        self.assertIn("TZ_LR_SCHEDULE", w)
        self.assertIn("[server] port", w)
        self.assertIsNone(runconfig.retired_env_var_warning({"TZ_DEVICE": "cuda"}))
        cfg = defaults()
        with mock.patch.dict(os.environ, {"TZ_LR_SCHEDULE": "0:9"}), mock.patch.multiple(
            client, LR_SCHEDULE=client.LR_SCHEDULE, LR_WARMUP_SAMPLES=client.LR_WARMUP_SAMPLES
        ), mock.patch.dict(client.LOSS_WEIGHTS):
            client.apply_run_config(cfg)
            self.assertEqual(client.LR_SCHEDULE, cfg["trainer"]["lr_schedule"])

    def test_new_net_follows_net_config(self):
        import network

        self.assertIsInstance(client.new_net(defaults()["net"]), network.TrueNet)
        small = client.new_net({"arch": "TrueNetXS", "blocks": 1, "channels": 8})
        self.assertIsInstance(small, network.TrueNetXS)
        self.assertEqual(len(small.backBone), 1)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)

    def rust_run(self, config_sha256="a" * 64, git_sha="abc"):
        """A run folder as runconfig.rs create_run writes it."""
        run = self.dir / "truenet-128x10-20261124-1503"
        (run / "hidden").mkdir(parents=True)
        (run / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run.name,
                    "name": "truenet-128x10",
                    "created_at": 1.0,
                    "git_sha": git_sha,
                    "git_dirty": False,
                    "config_sha256": config_sha256,
                }
            )
        )
        (self.dir / runconfig.CURRENT_RUN_FILE).write_text(run.name + "\n")
        return run

    def test_current_run(self):
        self.assertIsNone(runconfig.current_run_id(self.dir))
        run = self.rust_run()
        self.assertEqual(runconfig.current_run_id(self.dir), run.name)

    def test_unchanged_start_is_logged_quietly(self):
        run = self.rust_run()
        self.assertEqual(runconfig.record_start(run, identity(), b"x"), [])
        lines = (run / "hidden" / "run_log.jsonl").read_text().splitlines()
        self.assertEqual(json.loads(lines[0])["run_id"], identity().run_id)

    def test_changed_config_or_commit_is_logged_not_refused(self):
        run = self.rust_run()
        notes = runconfig.record_start(run, identity(sha="def", config="b"), b"[run]\nseed = 1\n")
        self.assertEqual(len(notes), 2)
        copy = run / "hidden" / "config_history" / ("b" * 12 + ".toml")
        self.assertEqual(copy.read_bytes(), b"[run]\nseed = 1\n")

    def test_start_waits_for_the_server_then_fixes_the_data_root(self):
        run = self.rust_run()
        cfg = defaults()
        cfg["paths"]["runs_dir"] = str(self.dir)
        loaded = runconfig.LoadedConfig(cfg, b"", "c" * 64)
        old_root = paths._root
        self.addCleanup(paths.set_root, old_root)
        env = {k: v for k, v in os.environ.items() if k != "TZ_DATA_ROOT"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
            runconfig, "load_config", return_value=loaded
        ):
            got_cfg, ident, run_dir = runconfig.start("test")
        self.assertEqual(run_dir, run)
        self.assertEqual(ident.run_id, run.name)
        self.assertEqual(paths.data_path("nets"), str(run / "nets"))

    def test_run_change_messages_from_the_server(self):
        """Wire forms as runconfig.rs run_messages_on_the_wire pins them."""
        me = "r-20261124-1503"
        same = json.loads('{"purpose":{"RunInfo":"r-20261124-1503"}}')
        other = json.loads('{"purpose":{"RunInfo":"r-20261124-1600"}}')
        reset = json.loads('{"purpose":{"ResetRun":"r2"}}')
        self.assertIsNone(client.run_change(same, me))
        self.assertIn("r-20261124-1600", client.run_change(other, me))
        self.assertIn("r2", client.run_change(reset, me))
        self.assertIsNone(client.run_change({"purpose": {"RequestingNet": []}}, me))
        self.assertIsNone(client.run_change({"purpose": "x"}, me))


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.dir, "provenance"))

    def tearDown(self):
        shutil.rmtree(self.dir)

    def datagen_line(self, bin_bytes, **fields):
        """A line as main.rs collector_main writes it."""
        ident = identity()
        rec = {
            "event": "file",
            "generator_id": 3,
            "run_id": ident.run_id,
            "config_sha256": ident.config_sha256,
            "git_sha": ident.git_sha,
            "net_sha256": "n" * 64,
            "bin_sha256": hashlib.sha256(bin_bytes).hexdigest(),
            "games": 100,
            "positions": 25000,
        }
        rec.update(fields)
        with open(os.path.join(self.dir, "provenance", "datagen.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def trainer_lines(self):
        with open(os.path.join(self.dir, "provenance", "trainer.jsonl")) as f:
            return [json.loads(line) for line in f]

    def test_received_file_joins_to_its_generator(self):
        self.datagen_line(b"other")
        self.datagen_line(b"games")
        rec = runconfig.record_received_file("p", b"games", identity(), hidden_dir=self.dir)
        self.assertEqual(rec["generator_id"], 3)
        [line] = self.trainer_lines()
        self.assertEqual(line["generator_id"], 3)
        self.assertEqual(line["mismatch"], [])

    def test_file_from_another_config_is_flagged_not_dropped(self):
        self.datagen_line(b"games", config_sha256="b" * 64)
        rec = runconfig.record_received_file("p", b"games", identity(), hidden_dir=self.dir)
        self.assertIsNotNone(rec)
        self.assertEqual(len(self.trainer_lines()[0]["mismatch"]), 1)

    def test_unknown_file_is_still_recorded(self):
        self.assertIsNone(
            runconfig.record_received_file("p", b"x", identity(), hidden_dir=self.dir)
        )
        self.assertIsNone(self.trainer_lines()[0]["generator_id"])


if __name__ == "__main__":
    unittest.main()
