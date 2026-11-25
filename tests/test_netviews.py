"""Smoke and orientation tests for the network visualiser (`netviews.py`).

Run from `src/python`:

    PYTHONPATH=. uv run python -m unittest discover -s ../../tests
"""

import unittest

import numpy as np
import torch

try:
    import chess
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    chess = None

FENS = [
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/5P2/PPPP2PP/RNB1K1NR b KQkq - 0 4",
]


@unittest.skipIf(chess is None, "python-chess / matplotlib not installed")
class NetViewsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import netviews
        from network import TrueNet

        torch.manual_seed(0)
        cls.nv = netviews
        cls.net = TrueNet(num_resBlocks=2, num_hidden=16).eval()

    def test_input_planes_land_on_physical_squares(self):
        # "us K" (input channel 13) displayed via to_display_grid must light up the
        # side to move's king square as drawn by square_grid, for either side to move.
        for fen in FENS:
            board = chess.Board(fen)
            probe = self.nv.Probe(self.net, board)
            shown = self.nv.to_display_grid(probe.acts["input"][13], board)
            expected = self.nv.square_grid({board.king(board.turn): 1.0})
            np.testing.assert_array_equal(shown, expected, err_msg=fen)

    def test_policy_moves_are_legal_and_normalised(self):
        for fen in FENS:
            board = chess.Board(fen)
            moves = self.nv.Probe(self.net, board).physical_moves()
            self.assertEqual({m for m, _, _ in moves}, set(board.legal_moves))
            self.assertAlmostEqual(sum(p for _, p, _ in moves), 1.0, places=5)

    def test_occlusion_baseline_is_the_position_value(self):
        board = chess.Board(FENS[1])
        probe = self.nv.Probe(self.net, board)
        occ = probe.occlusion()
        kings = {board.king(chess.WHITE), board.king(chess.BLACK)}
        self.assertEqual(set(occ), set(board.piece_map()) - kings)
        # with no repetition history the stack-free baseline equals the probe's value
        self.assertAlmostEqual(
            probe.white_value(),
            -float(torch.tanh(self.net(probe.x.unsqueeze(0))[0][0, 0])),
            places=5,
        )

    def test_every_figure_renders(self):
        for fen in FENS:
            for name, fig in self.nv.all_figures(self.net, chess.Board(fen), "test"):
                with self.subTest(fen=fen, view=name):
                    self.assertGreater(len(fig.axes), 0)
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
