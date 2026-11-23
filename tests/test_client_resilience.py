"""client.py must survive I/O and network failures instead of crashing.

Run from `src/python`:

    PYTHONPATH=. python -m unittest discover -s ../../tests
"""

import errno
import os
import shutil
import socket
import tempfile
import unittest
from unittest import mock

from lib.data.group import DataGroup
from lib.games import Game
from lib.loop import LoopBuffer

import client

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "kzero", "kzero_chess"
)


class StaleHandle:
    """A handle whose share went away: reads fail like Windows reports them (EINVAL)."""

    def seek(self, *args):
        return 0

    def read(self, *args):
        raise OSError(errno.EINVAL, "Invalid argument")

    def close(self):
        pass


class ReplayBufferTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.game = Game.find("chess")
        self.buffer = LoopBuffer(self.game, target_positions=10_000, test_fraction=0.2)
        self.bases = []
        for i in range(2):
            base = os.path.join(self.dir, f"games_{i}")
            for ext in (".bin", ".off", ".json"):
                shutil.copy(FIXTURE + ext, base + ext)
            self.bases.append(base)
            self.buffer.append(None, client.load_file(base))

    def tearDown(self):
        for f in self.buffer.files:
            f.close()
        shutil.rmtree(self.dir)

    def test_stale_handles_are_replaced_before_sampling(self):
        self.buffer.files[0].off_handle.close()
        self.buffer.files[0].off_handle = StaleHandle()
        with self.assertRaises(OSError):
            DataGroup.from_files(self.game, self.buffer.files)

        client.reopen_buffer_files(self.buffer)

        group = DataGroup.from_files(self.game, self.buffer.files)
        self.assertEqual(len(group.positions), self.buffer.position_count)
        self.assertEqual(len(self.buffer.files), 2)

    def test_unreadable_file_is_dropped_with_counts_updated(self):
        per_file = self.buffer.position_count // 2
        os.remove(self.bases[0] + ".off")

        client.reopen_buffer_files(self.buffer)

        self.assertEqual(len(self.buffer.files), 1)
        self.assertEqual(self.buffer.position_count, per_file)
        group = DataGroup.from_files(self.game, self.buffer.files)
        self.assertEqual(len(group.positions), per_file)

    def test_io_error_during_training_skips_the_round(self):
        def failing_round(**kwargs):
            raise OSError(errno.EINVAL, "Invalid argument")

        with mock.patch.object(client, "MIN_BUFFER_TO_TRAIN", 1), mock.patch.object(
            client, "full_train_and_send", failing_round
        ):
            self.assertIsNone(client.train_round_or_skip(loopbuf=self.buffer))


class ServerConnectTests(unittest.TestCase):
    def test_failed_connect_returns_false_and_a_later_attempt_succeeds(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = client.Server("127.0.0.1", port)
        try:
            # nothing is listening yet: refused, no exception
            self.assertFalse(server.try_connect())
            listener.listen(1)
            # the same Server object can connect after a failed attempt
            self.assertTrue(server.try_connect())
            self.assertIsNotNone(server.file)
        finally:
            server.close()
            listener.close()


if __name__ == "__main__":
    unittest.main()
