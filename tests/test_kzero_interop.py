"""TrueZero against data written by KZero itself.

`fixtures/kzero/kzero_chess.*` was written by kZero's `BinaryOutput` with `ChessStdMapper`
(kZero commit 5bf3343, generator in `write_fixture.rs.txt`) from `fixture_games.txt`.
`fixtures/truezero_legacy/legacy_chess.*` holds the same games written by TrueZero's writer
before KZero compatibility was restored (commit 1c83ad6). Run from `src/python`:

    PYTHONPATH=. python -m unittest discover -s ../../tests
"""

import json
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import kzero_data
import network
from lib.data.check import check_data_file
from lib.data.group import DataGroup
from lib.data.sampler import sample_position
from lib.logger import Logger
from lib.loop import LoopBuffer
from lib.train import ScalarTarget, TrainSettings

try:
    import chess
except ImportError:
    chess = None

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
KZERO = os.path.join(FIXTURES, "kzero", "kzero_chess")
LEGACY = os.path.join(FIXTURES, "truezero_legacy", "legacy_chess")
GAMES = os.path.join(FIXTURES, "kzero", "fixture_games.txt")
KDL_SCALAR = kzero_data.S["kdl_policy"]


def read_records(path):
    """Per position: (scalars + inputs as bytes, sorted (policy index, value) pairs)."""
    off = Path(path + ".off").read_bytes()
    meta = json.loads(Path(path + ".json").read_text())
    count = meta["position_count"]
    offsets = list(np.frombuffer(off, np.uint64, count).astype(int))
    data = Path(path + ".bin").read_bytes()
    bounds = offsets + [len(data)]
    head_len = 4 * len(kzero_data.SCALAR_NAMES) + kzero_data.BOOL_BYTES + 4 * 8
    result = []
    for start, end in zip(bounds, bounds[1:]):
        head = bytearray(data[start : start + head_len])
        n = (end - start - head_len) // 8
        indices = np.frombuffer(data, np.int32, n, start + head_len)
        values = np.frombuffer(data, np.float32, n, start + head_len + 4 * n)
        result.append((head, sorted(zip(indices.tolist(), values.tolist()))))
    return result


class KZeroFileTests(unittest.TestCase):
    def setUp(self):
        self.file = kzero_data.open_data_file(KZERO)

    def tearDown(self):
        self.file.close()

    def test_inherited_checker_accepts_file(self):
        self.assertEqual(kzero_data.detect_layout(self.file), "kzero")
        check_data_file(self.file)

    def test_sampler_attaches_the_appended_final_position(self):
        group = DataGroup(self.file.info.game, [self.file.simulations])
        random.seed(0)
        for _ in range(200):
            _, pos = sample_position(
                group, include_final=False, include_final_for_each=True
            )
            final = pos.final_position
            self.assertTrue(final.is_final)
            self.assertEqual(final.simulation, pos.simulation)
            self.assertEqual(final.move_index, pos.simulation.move_count)

    def test_end_to_end_training_step(self):
        torch.manual_seed(0)
        game = self.file.info.game
        buffer = LoopBuffer(game, target_positions=1000, test_fraction=0.2)
        log = Logger()
        log.start_batch()
        buffer.append(log, self.file)

        model = torch.jit.script(network.TrueNetXS(num_resBlocks=1, num_hidden=8))
        settings = TrainSettings(
            game=game,
            scalar_target=ScalarTarget.Final,
            value_weight=1.0,
            wdl_weight=1.0,
            moves_left_weight=0.1,
            moves_left_delta=0.1,
            policy_weight=1,
            sim_weight=0.0,
            train_in_eval_mode=False,
            clip_norm=5.0,
            mask_policy=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in model.parameters()]

        train = buffer.sampler(8, None, False, False, False, test=False)
        test = buffer.sampler(8, None, False, False, True, test=True)
        try:
            batch = train.next_batch()
            self.assertTrue(
                torch.all(batch.input_full[:, 7] == batch.input_full[:, 7].round())
            )
            settings.train_step(batch, network=model, optimizer=optimizer, logger=log)
            with torch.no_grad():
                settings.evaluate_batch(model, test.next_batch(), "test", log)
        finally:
            train.close()
            test.close()

        changed = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        self.assertTrue(changed, "training step did not update the network")


@unittest.skipIf(chess is None, "python-chess is not installed")
class PythonEncoderTests(unittest.TestCase):
    """`decoder.py` (python-chess) must encode positions exactly like KZero's ChessStdMapper."""

    def test_inputs_and_policy_indices_match_kzero(self):
        import decoder

        file = kzero_data.open_data_file(KZERO)
        pi = 0
        try:
            for line in Path(GAMES).read_text().splitlines():
                if not line.strip():
                    continue
                fen, moves = line.split("|")
                board = (
                    chess.Board()
                    if fen.strip() == "startpos"
                    else chess.Board(fen.strip())
                )
                for mv in moves.split() + [None]:
                    pos = file.positions[pi]
                    tensor, _ = decoder.convert_board(board)
                    np.testing.assert_array_equal(
                        tensor[:8, 0, 0].numpy(),
                        pos.input_scalars,
                        err_msg=f"position {pi}",
                    )
                    np.testing.assert_array_equal(
                        tensor[8:].numpy(), pos.input_bools, err_msg=f"position {pi}"
                    )
                    if mv is not None:
                        _, indices = decoder.extract_policy(board)
                        self.assertEqual(
                            sorted(indices), sorted(pos.policy_indices.tolist())
                        )
                        board.push_uci(mv)
                    pi += 1
        finally:
            file.close()
        self.assertEqual(pi, len(file.positions))


class LegacyMigrationTests(unittest.TestCase):
    def test_legacy_file_is_refused(self):
        with self.assertRaises(kzero_data.LegacyDataError):
            kzero_data.open_data_file(LEGACY)

    def test_migration_reproduces_the_kzero_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "migrated")
            kzero_data.migrate_legacy_file(LEGACY, out)

            self.assertEqual(
                Path(out + ".off").read_bytes(), Path(KZERO + ".off").read_bytes()
            )
            migrated = read_records(out)
            expected = read_records(KZERO)
            self.assertEqual(len(migrated), len(expected))

            for pi, (m, e) in enumerate(zip(migrated, expected)):
                head_m, policy_m = m
                head_e, policy_e = e
                # kdl_policy cannot be recovered: legacy files store NaN
                k = slice(4 * KDL_SCALAR, 4 * KDL_SCALAR + 4)
                self.assertTrue(np.isnan(np.frombuffer(head_m[k], np.float32)[0]))
                head_m[k] = head_e[k]
                self.assertEqual(head_m, head_e, f"position {pi}")
                # (index, value) pairs are stored in move-generator order, which differs per crate
                self.assertEqual(policy_m, policy_e, f"position {pi}")

            self.assertEqual(
                json.loads(Path(out + ".json").read_text()),
                json.loads(Path(KZERO + ".json").read_text()),
            )

    def test_migration_refuses_a_kzero_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(kzero_data.LegacyDataError):
                kzero_data.migrate_legacy_file(KZERO, os.path.join(tmp, "x"))


if __name__ == "__main__":
    unittest.main()
