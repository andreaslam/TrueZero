import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import numpy as np

import network
from lib.data.position import write_input
from lib.data.sampler import sample_position
from lib.games import Game
from lib.train import _masked_mean, evaluate_policy


class ReplayContractTests(unittest.TestCase):
    def test_final_input_uses_appended_final_record(self):
        # KZero layout: a non-final position stores final_moves_left = game_length + 1 - pos_index,
        # so the appended final position is at pi + final_moves_left - 1 (here 5 + 4 - 1 = 8).
        source = SimpleNamespace(
            is_final=False,
            final_moves_left=4.0,
            simulation=SimpleNamespace(includes_final=True),
            final_position=None,
        )
        last_non_final = SimpleNamespace(is_final=False)
        final = SimpleNamespace(is_final=True)
        positions = [SimpleNamespace(is_final=False) for _ in range(5)]
        positions.extend(
            [source, last_non_final, SimpleNamespace(is_final=False), final]
        )
        group = SimpleNamespace(positions=positions)

        with patch("lib.data.sampler.random.randrange", return_value=5):
            _, sampled = sample_position(
                group, include_final=False, include_final_for_each=True
            )

        self.assertIs(sampled.final_position, final)

    def test_policy_loss_rejects_excess_target_mass(self):
        with self.assertRaisesRegex(AssertionError, "Invalid value mass"):
            evaluate_policy(
                torch.zeros((1, 2)),
                torch.tensor([[0, 1]]),
                torch.tensor([[0.75, 0.75]]),
                mask_invalid_moves=True,
            )

    def test_masked_mean_uses_only_terminal_entries(self):
        values = torch.tensor([1.0, 10.0, 5.0])
        terminal = torch.tensor([True, False, True])

        self.assertEqual(_masked_mean(values, terminal).item(), 3.0)

    def test_stored_input_is_the_network_input(self):
        # KZero contract: write_input expands the stored scalars unchanged
        game = Game.find("chess")
        scalars = np.array([1, 0, 1, 1, 1, 1, 1, 50], dtype=np.float32)
        position = SimpleNamespace(
            input_scalars=scalars,
            input_bools=np.zeros(game.input_bool_shape, dtype=np.uint8),
        )
        target = torch.empty(game.full_input_shape)

        write_input(game, target, position)

        self.assertEqual(target[7, 0, 0].item(), 50.0)
        self.assertEqual(target[6, 0, 0].item(), 1.0)

    def test_network_scales_the_raw_halfmove_clock(self):
        x = torch.ones((2, 21, 8, 8))
        x[:, network.HALFMOVE_CLOCK_CHANNEL] = 50.0

        scaled = network.scale_input(x)

        self.assertTrue(torch.all(scaled[:, 7] == 0.5))
        others = [c for c in range(21) if c != 7]
        self.assertTrue(torch.equal(scaled[:, others], x[:, others]))
        self.assertTrue(torch.all(x[:, 7] == 50.0))  # input not mutated
        self.assertEqual((network.HALFMOVE_CLOCK_CHANNEL, network.HALFMOVE_CLOCK_SCALE), (7, 100.0))
        scripted = torch.jit.script(network.TrueNetXS(num_resBlocks=1, num_hidden=8)).eval()
        eager = network.TrueNetXS(num_resBlocks=1, num_hidden=8).eval()
        eager.load_state_dict(scripted.state_dict())
        with torch.no_grad():
            for a, b in zip(scripted(x), eager(x)):
                self.assertTrue(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
