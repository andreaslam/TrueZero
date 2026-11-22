"""TrueZero against data written by KZero itself.

`fixtures/kzero/kzero_chess.*` was written by kZero's `BinaryOutput` with `ChessStdMapper`
(kZero commit 5bf3343, generator in `write_fixture.rs.txt`) from `fixture_games.txt`.
Run from `src/python`:

    PYTHONPATH=. python -m unittest discover -s ../../tests
"""

import os
import random
import unittest
from pathlib import Path

import numpy as np
import torch

from lib.data.file import DataFile
from lib.games import Game
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
GAMES = os.path.join(FIXTURES, "kzero", "fixture_games.txt")


class KZeroFileTests(unittest.TestCase):
    def setUp(self):
        self.file = DataFile.open(Game.find("chess"), KZERO)

    def tearDown(self):
        self.file.close()

    def test_inherited_checker_accepts_file(self):
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

        file = DataFile.open(Game.find("chess"), KZERO)
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


if __name__ == "__main__":
    unittest.main()
