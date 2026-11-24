"""Fast-search (playout cap randomisation) positions must not train the policy.

client.load_file marks their policy targets with lib's "no policy target" value
(-1) and the policy loss weight is rescaled by the batch's share of positions
with a target. Run from `src/python`:

    PYTHONPATH=. uv run python -m unittest discover -s ../../tests
"""

import os
import random
import unittest

import numpy as np
import torch
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.loop import LoopBuffer
from lib.train import ScalarTarget, TrainSettings, evaluate_policy

import client
import network

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "kzero", "kzero_chess"
)


def train_settings():
    return TrainSettings(
        game=Game.find("chess"),
        scalar_target=ScalarTarget.Final,
        value_weight=1.0,
        wdl_weight=1.0,
        moves_left_weight=0.1,
        moves_left_delta=0.1,
        policy_weight=client.POLICY_WEIGHT,
        sim_weight=0.0,
        train_in_eval_mode=False,
        clip_norm=5.0,
        mask_policy=True,
    )


class FastSearchPolicyTests(unittest.TestCase):
    def setUp(self):
        self.raw = DataFile.open(Game.find("chess"), FIXTURE)
        self.masked = client.load_file(FIXTURE)
        self.non_final = [
            pi
            for pi in range(self.raw.info.position_count)
            if not self.raw.positions[pi].is_final
        ]
        flags = {self.raw.positions[pi].is_full_search for pi in self.non_final}
        # the fixture must contain both kinds, or these tests prove nothing
        self.assertEqual(flags, {True, False})

    def tearDown(self):
        self.raw.close()
        self.masked.close()

    def check_masking(self, masked_file):
        for pi in self.non_final:
            raw = self.raw.positions[pi]
            masked = masked_file.positions[pi]
            if raw.is_full_search:
                np.testing.assert_array_equal(masked.policy_values, raw.policy_values)
            else:
                self.assertEqual(len(masked.policy_values), raw.available_mv_count)
                self.assertTrue(np.all(masked.policy_values == -1))
            # everything else is untouched
            np.testing.assert_array_equal(masked.policy_indices, raw.policy_indices)
            np.testing.assert_array_equal(masked.final_wdl, raw.final_wdl)
            self.assertEqual(masked.final_v, raw.final_v)
            self.assertEqual(masked.final_moves_left, raw.final_moves_left)

    def test_fast_searches_lose_only_their_policy_target(self):
        self.check_masking(self.masked)

    def test_masking_survives_reopened_handles(self):
        # The samplers read through with_new_handles(), not the original file.
        reopened = self.masked.with_new_handles()
        self.assertIsInstance(reopened, client.FullSearchPolicyFile)
        self.check_masking(reopened)

    def test_sampled_batches_have_targets_only_for_full_searches(self):
        buffer = LoopBuffer(Game.find("chess"), target_positions=1000, test_fraction=0.0)
        buffer.append(None, self.masked)
        full = sum(self.raw.positions[pi].is_full_search for pi in self.non_final)
        sampler = buffer.sampler(256, None, False, False, False, test=False)
        try:
            random.seed(0)
            fraction = client.policy_target_fraction(sampler.next_batch())
        finally:
            sampler.close()
        expected = full / len(self.non_final)
        self.assertLess(fraction, 1.0)
        self.assertAlmostEqual(fraction, expected, delta=0.15)

    def test_masked_rows_give_no_policy_gradient_and_weight_restores_the_balance(self):
        logits = torch.randn(4, 1880, requires_grad=True)
        indices = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        values = torch.tensor([[0.7, 0.3], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]])

        evaluation = evaluate_policy(logits, indices, values, True)
        evaluation.train_loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[1:].abs().sum().item(), 0.0)

        class Batch:
            policy_values = values

            def __len__(self):
                return 4

        settings = train_settings()
        self.assertEqual(client.set_policy_weight(settings, Batch()), 0.25)
        self.assertEqual(settings.policy_weight, 4 * client.POLICY_WEIGHT)
        # weighted batch loss == the loss averaged over positions with a target
        self.assertAlmostEqual(
            (settings.policy_weight * evaluation.train_loss).item(),
            evaluation.norm_loss.item(),
            places=5,
        )

    def test_weight_is_unchanged_without_fast_searches_and_finite_without_targets(self):
        class Batch:
            def __init__(self, values):
                self.policy_values = values

            def __len__(self):
                return len(self.policy_values)

        settings = train_settings()
        client.set_policy_weight(settings, Batch(torch.tensor([[1.0, 0.0], [0.5, 0.5]])))
        self.assertEqual(settings.policy_weight, client.POLICY_WEIGHT)
        client.set_policy_weight(settings, Batch(torch.full((8, 2), -1.0)))
        self.assertEqual(settings.policy_weight, 8 * client.POLICY_WEIGHT)

    def test_training_step_on_masked_data(self):
        torch.manual_seed(0)
        buffer = LoopBuffer(Game.find("chess"), target_positions=1000, test_fraction=0.2)
        log = Logger()
        log.start_batch()
        buffer.append(log, self.masked)
        model = torch.jit.script(network.TrueNetXS(num_resBlocks=1, num_hidden=8))
        op = client.create_optimizer(model)
        schedule = client.LrSchedule(client.parse_lr_schedule("0:1e-3"), 0)
        settings = train_settings()
        before = [p.detach().clone() for p in model.parameters()]

        train = buffer.sampler(16, None, False, False, False, test=False)
        try:
            client.scheduled_train_step(
                settings, train.next_batch(), model, op, schedule, log
            )
        finally:
            train.close()

        after = list(model.parameters())
        self.assertTrue(all(torch.isfinite(p).all() for p in after))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(after, before)))
        self.assertLess(log.data[("policy-targets", "fraction")][0], 1.0)


if __name__ == "__main__":
    unittest.main()
