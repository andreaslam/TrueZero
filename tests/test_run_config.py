"""run.toml and the run lock: the trainer side of src/rust/runconfig.rs (same rules).

Run from `src/python`:

    PYTHONPATH=. uv run python -m unittest discover -s ../../tests
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import client
import runconfig

EXAMPLE = runconfig.EXAMPLE_PATH


def identity(sha="abc", config="a", dirty=False, run_id=None):
    cfg, _ = runconfig.load_run_config(EXAMPLE)
    run = dict(cfg["run"])
    if run_id is not None:
        run["run_id"] = run_id
    return runconfig.StartIdentity(run, config * 64, sha, dirty, "test")


class RunConfigTests(unittest.TestCase):
    def test_example_reproduces_the_previous_trainer_defaults(self):
        """Adopting run.example.toml must not change the running trainer."""
        cfg, _ = runconfig.load_run_config(EXAMPLE)
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
            client.apply_run_config(cfg)
            self.assertEqual({n: getattr(client, n) for n in names}, before)
            self.assertEqual(client.LOSS_WEIGHTS, weights_before)

    def test_missing_unknown_and_mistyped_keys_are_errors(self):
        text = EXAMPLE.read_text()
        cases = {
            "missing": text.replace("batch_size = 2048\n", ""),
            "unknown": text.replace("batch_size = 2048", "batch_sise = 2048"),
            "type": text.replace("batch_size = 2048", 'batch_size = "2048"'),
            "not a power of two": text.replace("batch_size = 2048", "batch_size = 2000"),
            "table": text + "\n[extra]\nx = 1\n",
            "fork without parent": text.replace("fork_sample = 0", "fork_sample = 5"),
        }
        for name, bad in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                runconfig.validate(runconfig.tomllib.loads(bad))

    def test_hash_is_of_the_file_bytes(self):
        """The Rust side hashes the same bytes, so the two agree without a shared parser."""
        import hashlib

        _, sha = runconfig.load_run_config(EXAMPLE)
        self.assertEqual(sha, hashlib.sha256(EXAMPLE.read_bytes()).hexdigest())

    def test_retired_env_vars_are_refused(self):
        with self.assertRaises(runconfig.RunRefused):
            runconfig.refuse_retired_env_vars({"TZ_LR_SCHEDULE": "0:1e-3"})
        runconfig.refuse_retired_env_vars({"TZ_SERVER_PORT": "1", "TZ_NUM_EXECUTORS": "2"})


class RunLockTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def lock(self):
        with open(os.path.join(self.dir, "run_lock.json")) as f:
            return json.load(f)

    def test_first_start_creates_the_lock_and_identical_restart_passes(self):
        self.assertEqual(runconfig.check_run_lock(self.dir, identity()), [])
        self.assertEqual(self.lock()["git_sha"], "abc")
        self.assertEqual(self.lock()["seed"], 0)
        self.assertEqual(runconfig.check_run_lock(self.dir, identity()), [])
        with open(os.path.join(self.dir, "run_log.jsonl")) as f:
            self.assertEqual(len(f.readlines()), 2)

    def test_dirty_tree_cannot_create_the_lock_without_override(self):
        with self.assertRaises(runconfig.RunRefused):
            runconfig.check_run_lock(self.dir, identity(dirty=True))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "run_lock.json")))
        used = runconfig.check_run_lock(
            self.dir, identity(dirty=True), allow_code_change=True
        )
        self.assertEqual(len(used), 1)

    def test_other_commit_or_config_is_refused_unless_overridden(self):
        runconfig.check_run_lock(self.dir, identity())
        with self.assertRaises(runconfig.RunRefused):
            runconfig.check_run_lock(self.dir, identity(sha="def"))
        self.assertEqual(
            len(runconfig.check_run_lock(self.dir, identity(sha="def"), allow_code_change=True)),
            1,
        )
        with self.assertRaises(runconfig.RunRefused):
            runconfig.check_run_lock(self.dir, identity(config="b"), allow_code_change=True)
        used = runconfig.check_run_lock(
            self.dir, identity(config="b"), allow_config_change=True
        )
        self.assertEqual(len(used), 1)

    def test_another_run_id_is_always_refused(self):
        runconfig.check_run_lock(self.dir, identity())
        with self.assertRaises(runconfig.RunRefused):
            runconfig.check_run_lock(
                self.dir,
                identity(run_id="other"),
                allow_config_change=True,
                allow_code_change=True,
            )

    def test_lock_written_by_rust_is_read(self):
        """Field names match RunLock in runconfig.rs."""
        with open(os.path.join(self.dir, "run_lock.json"), "w") as f:
            json.dump(
                {
                    "run_id": identity().run["run_id"],
                    "parent_run_id": "",
                    "fork_sample": 0,
                    "seed": 0,
                    "git_sha": "abc",
                    "config_sha256": "a" * 64,
                    "created_at": 1.0,
                    "created_by": "datagen",
                },
                f,
            )
        self.assertEqual(runconfig.check_run_lock(self.dir, identity()), [])


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.dir, "provenance"))

    def tearDown(self):
        shutil.rmtree(self.dir)

    def datagen_line(self, bin_bytes, **fields):
        """A line as main.rs collector_main writes it."""
        import hashlib

        ident = identity()
        rec = {
            "event": "file",
            "generator_id": 3,
            "run_id": ident.run["run_id"],
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
