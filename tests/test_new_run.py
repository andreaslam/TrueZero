"""A new run (no tz_N.pt in nets/) starts from a fresh TrueNet and a fresh log.npz.

Run from `src/python`:

    PYTHONPATH=. uv run python -m unittest discover -s ../../tests
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch

import client


class NewRunTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "nets"))
        os.makedirs(os.path.join(self.root, "hidden"))
        self.log_npz = os.path.join(self.root, "log.npz")
        path = lambda rel: os.path.join(self.root, rel)
        self.patches = [
            mock.patch.object(client, "data_path", path),
            mock.patch.object(client, "data_dir", path),
            mock.patch.object(client, "LOG_NPZ_PATH", self.log_npz),
            mock.patch.object(client, "TRAINING_LOG_PATH", path("hidden/traininglog.txt")),
            # keep the test fast: the architecture is what matters, not the size
            mock.patch.object(client, "NEW_NET_RES_BLOCKS", 1),
            mock.patch.object(client, "NEW_NET_HIDDEN", 8),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.root)

    def test_new_run_creates_truenet_and_resets_log(self):
        with open(self.log_npz, "wb") as f:
            f.write(b"old run")
        nets = client.check_net_exists("cpu", r"tz_(\d+)\.pt")
        self.assertEqual(nets, [os.path.join(self.root, "nets/tz_0.pt")])
        self.assertFalse(os.path.exists(self.log_npz))
        net = torch.jit.load(nets[0])
        self.assertEqual(net.original_name, "TrueNet")

    def test_existing_run_keeps_log(self):
        with open(self.log_npz, "wb") as f:
            f.write(b"this run")
        client.check_net_exists("cpu", r"tz_(\d+)\.pt")  # creates tz_0
        with open(self.log_npz, "wb") as f:
            f.write(b"this run")
        client.check_net_exists("cpu", r"tz_(\d+)\.pt")
        self.assertTrue(os.path.exists(self.log_npz))


if __name__ == "__main__":
    unittest.main()
