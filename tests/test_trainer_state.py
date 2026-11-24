"""A restarted trainer must continue exactly where it stopped.

client.py checkpoints the training weights and the optimiser after every
training round and restores them on start-up, but only when the checkpoint
continues from the accepted net. Run from `src/python`:

    PYTHONPATH=. uv run python -m unittest discover -s ../../tests
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch

import client
import network


def scripted_net(seed, num_hidden=8):
    torch.manual_seed(seed)
    return torch.jit.script(network.TrueNetXS(num_resBlocks=1, num_hidden=num_hidden))


def train_steps(model, op, steps, seed):
    """Stand-in for train_step: a few deterministic AdamW steps in train mode."""
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(steps):
        x = torch.randn(4, 21, 8, 8, generator=gen)
        value, policy = model(x)
        loss = value.square().mean() + policy.square().mean()
        op.zero_grad(set_to_none=True)
        loss.backward()
        op.step()


def weights(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def assert_same_weights(test, a, b):
    test.assertEqual(set(a), set(b))
    for k in a:
        test.assertTrue(torch.equal(a[k], b[k]), k)


class TrainerStateTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "trainer_state.pt")
        self.accepted = scripted_net(seed=0)  # the accepted net tz_5.pt
        self.accepted_weights = weights(self.accepted)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def trained_and_saved(self, accepted_gen=5, candidate_gen=6):
        """Train from the accepted net, then checkpoint as full_train_and_send does."""
        model = scripted_net(seed=0)
        op = client.create_optimizer(model)
        train_steps(model, op, steps=3, seed=1)
        self.assertTrue(
            client.save_trainer_state(model, op, accepted_gen, candidate_gen, self.path)
        )
        return model, op

    def fresh_load(self, accepted_gen):
        model = scripted_net(seed=0)
        op = client.load_trainer_state(model, accepted_gen, "cpu", self.path)
        return model, op

    def test_resume_continues_bit_for_bit(self):
        # Candidate 6 rejected, then a crash: tz_5 is still the accepted net.
        trained, trained_op = self.trained_and_saved()

        resumed, resumed_op = self.fresh_load(accepted_gen=5)

        assert_same_weights(self, weights(resumed), weights(trained))
        # Training on from either gives identical weights: the optimiser's
        # moments and step counts were restored, not just the weights.
        train_steps(trained, trained_op, steps=2, seed=2)
        train_steps(resumed, resumed_op, steps=2, seed=2)
        assert_same_weights(self, weights(resumed), weights(trained))

    def test_resume_after_candidate_was_accepted(self):
        # Candidate 6 accepted (tz_6.pt now holds exactly these weights), then a crash.
        trained, _ = self.trained_and_saved()

        resumed, resumed_op = self.fresh_load(accepted_gen=6)

        assert_same_weights(self, weights(resumed), weights(trained))
        self.assertTrue(resumed_op.state)

    def test_checkpoint_from_another_lineage_is_ignored(self):
        self.trained_and_saved(accepted_gen=5, candidate_gen=6)

        for accepted_gen in (4, 7):  # nets rolled back, or checkpoint stale
            resumed, resumed_op = self.fresh_load(accepted_gen)
            assert_same_weights(self, weights(resumed), self.accepted_weights)
            self.assertFalse(resumed_op.state)

    def test_missing_or_corrupt_checkpoint_gives_fresh_optimiser(self):
        resumed, resumed_op = self.fresh_load(accepted_gen=5)
        self.assertFalse(resumed_op.state)

        with open(self.path, "wb") as f:
            f.write(b"half a checkpoint")
        resumed, resumed_op = self.fresh_load(accepted_gen=5)
        assert_same_weights(self, weights(resumed), self.accepted_weights)
        self.assertFalse(resumed_op.state)

    def test_architecture_mismatch_is_ignored(self):
        self.trained_and_saved()

        model = scripted_net(seed=0, num_hidden=16)
        before = weights(model)
        op = client.load_trainer_state(model, 5, "cpu", self.path)

        assert_same_weights(self, weights(model), before)
        self.assertFalse(op.state)

    def test_failed_write_keeps_previous_checkpoint(self):
        trained, _ = self.trained_and_saved()
        with open(self.path, "rb") as f:
            before = f.read()

        model = scripted_net(seed=3)
        op = client.create_optimizer(model)
        with mock.patch.object(
            client.os, "replace", side_effect=OSError(22, "Invalid argument")
        ):
            self.assertFalse(client.save_trainer_state(model, op, 6, 7, self.path))

        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), before)
        self.assertFalse(os.path.exists(self.path + ".tmp"))

        resumed, _ = self.fresh_load(accepted_gen=5)
        assert_same_weights(self, weights(resumed), weights(trained))

    def test_unwritable_location_does_not_raise(self):
        model = scripted_net(seed=0)
        op = client.create_optimizer(model)
        path = os.path.join(self.dir, "missing_dir", "trainer_state.pt")
        self.assertFalse(client.save_trainer_state(model, op, 5, 6, path))


if __name__ == "__main__":
    unittest.main()
