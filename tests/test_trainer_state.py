"""A restarted trainer must continue exactly where it stopped.

client.py checkpoints the training weights, the optimiser and the position in
the LR schedule after every training round and restores them on start-up, but
only when the checkpoint continues from the accepted net. Run from `src/python`:

    PYTHONPATH=. python -m unittest discover -s ../../tests
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch
from lib.logger import Logger

import client
import network

# Short enough that the tests cross the warm-up and a drop.
SCHEDULE_ENV = {
    "TZ_LR_SCHEDULE": "0:1e-3,20K:5e-4",
    "TZ_LR_WARMUP_SAMPLES": "8K",
}


def scripted_net(seed, num_hidden=8):
    torch.manual_seed(seed)
    return torch.jit.script(network.TrueNetXS(num_resBlocks=1, num_hidden=num_hidden))


class StubTrainSettings:
    """Stand-in for TrainSettings: a deterministic AdamW step in train mode."""

    def __init__(self, seed):
        self.gen = torch.Generator().manual_seed(seed)
        self.lrs_seen = []

    def train_step(self, batch, network, optimizer, logger):
        self.lrs_seen.append(optimizer.param_groups[0]["lr"])
        network.train()
        x = torch.randn(4, 21, 8, 8, generator=self.gen)
        value, policy = network(x)
        loss = value.square().mean() + policy.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def train_steps(model, op, schedule, steps, seed):
    """The trainer's loop: one scheduled step per logger batch."""
    settings = StubTrainSettings(seed)
    log = Logger()
    for _ in range(steps):
        log.start_batch()
        client.scheduled_train_step(settings, None, model, op, schedule, log)
    return settings.lrs_seen


def weights(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def assert_same_weights(test, a, b):
    test.assertEqual(set(a), set(b))
    for k in a:
        test.assertTrue(torch.equal(a[k], b[k]), k)


@mock.patch.dict(os.environ, SCHEDULE_ENV)
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
        schedule = client.LrSchedule.from_env()
        train_steps(model, op, schedule, steps=3, seed=1)
        self.assertTrue(
            client.save_trainer_state(
                model, op, schedule, accepted_gen, candidate_gen, self.path
            )
        )
        return model, op, schedule

    def fresh_load(self, accepted_gen):
        model = scripted_net(seed=0)
        op, schedule = client.load_trainer_state(model, accepted_gen, "cpu", self.path)
        return model, op, schedule

    def assert_fresh(self, op, schedule):
        self.assertFalse(op.state)
        self.assertEqual(schedule.samples_trained, 0)

    def test_resume_continues_bit_for_bit(self):
        # Candidate 6 rejected, then a crash: tz_5 is still the accepted net.
        trained, trained_op, trained_schedule = self.trained_and_saved()

        resumed, resumed_op, resumed_schedule = self.fresh_load(accepted_gen=5)

        assert_same_weights(self, weights(resumed), weights(trained))
        self.assertEqual(resumed_schedule.samples_trained, 3 * client.BATCH_SIZE)
        # Training on from either gives identical weights and learning rates:
        # the optimiser's moments and the schedule position were restored too.
        # These steps run through the end of the warm-up and the drop at 20K.
        lrs_trained = train_steps(trained, trained_op, trained_schedule, 8, seed=2)
        lrs_resumed = train_steps(resumed, resumed_op, resumed_schedule, 8, seed=2)
        self.assertEqual(lrs_resumed, lrs_trained)
        self.assertIn(1e-3, lrs_resumed)
        self.assertEqual(lrs_resumed[-1], 5e-4)
        assert_same_weights(self, weights(resumed), weights(trained))

    def test_resume_after_candidate_was_accepted(self):
        # Candidate 6 accepted (tz_6.pt now holds exactly these weights), then a crash.
        trained, _, _ = self.trained_and_saved()

        resumed, resumed_op, resumed_schedule = self.fresh_load(accepted_gen=6)

        assert_same_weights(self, weights(resumed), weights(trained))
        self.assertTrue(resumed_op.state)
        self.assertEqual(resumed_schedule.samples_trained, 3 * client.BATCH_SIZE)

    def test_checkpoint_from_another_lineage_is_ignored(self):
        self.trained_and_saved(accepted_gen=5, candidate_gen=6)

        for accepted_gen in (4, 7):  # nets rolled back, or checkpoint stale
            resumed, resumed_op, resumed_schedule = self.fresh_load(accepted_gen)
            assert_same_weights(self, weights(resumed), self.accepted_weights)
            self.assert_fresh(resumed_op, resumed_schedule)

    def test_missing_or_corrupt_checkpoint_starts_fresh_in_warmup(self):
        resumed, resumed_op, resumed_schedule = self.fresh_load(accepted_gen=5)
        self.assert_fresh(resumed_op, resumed_schedule)
        self.assertAlmostEqual(
            resumed_schedule.lr(), client.LR_WARMUP_START_FRACTION * 1e-3
        )

        with open(self.path, "wb") as f:
            f.write(b"half a checkpoint")
        resumed, resumed_op, resumed_schedule = self.fresh_load(accepted_gen=5)
        assert_same_weights(self, weights(resumed), self.accepted_weights)
        self.assert_fresh(resumed_op, resumed_schedule)

    def test_architecture_mismatch_is_ignored(self):
        self.trained_and_saved()

        model = scripted_net(seed=0, num_hidden=16)
        before = weights(model)
        op, schedule = client.load_trainer_state(model, 5, "cpu", self.path)

        assert_same_weights(self, weights(model), before)
        self.assert_fresh(op, schedule)

    def test_bad_schedule_setting_is_an_error_not_a_bad_checkpoint(self):
        self.trained_and_saved()
        with mock.patch.dict(os.environ, {"TZ_LR_SCHEDULE": "10K:1e-3"}):
            with self.assertRaises(ValueError):
                self.fresh_load(accepted_gen=5)

    def test_failed_write_keeps_previous_checkpoint(self):
        trained, _, _ = self.trained_and_saved()
        with open(self.path, "rb") as f:
            before = f.read()

        model = scripted_net(seed=3)
        op = client.create_optimizer(model)
        with mock.patch.object(
            client.os, "replace", side_effect=OSError(22, "Invalid argument")
        ):
            self.assertFalse(
                client.save_trainer_state(
                    model, op, client.LrSchedule.from_env(), 6, 7, self.path
                )
            )

        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), before)
        self.assertFalse(os.path.exists(self.path + ".tmp"))

        resumed, _, _ = self.fresh_load(accepted_gen=5)
        assert_same_weights(self, weights(resumed), weights(trained))

    def test_unwritable_location_does_not_raise(self):
        model = scripted_net(seed=0)
        op = client.create_optimizer(model)
        path = os.path.join(self.dir, "missing_dir", "trainer_state.pt")
        self.assertFalse(
            client.save_trainer_state(
                model, op, client.LrSchedule.from_env(), 5, 6, path
            )
        )


class LrScheduleTests(unittest.TestCase):
    def schedule(self, samples, points="0:1e-3,100:1e-4,300:1e-5", warmup=0):
        return client.LrSchedule(client.parse_lr_schedule(points), warmup, samples)

    def test_piecewise_constant_drops_at_the_boundaries(self):
        expected = {0: 1e-3, 99: 1e-3, 100: 1e-4, 299: 1e-4, 300: 1e-5, 10**9: 1e-5}
        for samples, lr in expected.items():
            self.assertEqual(self.schedule(samples).lr(), lr, samples)

    def test_warmup_ramps_linearly_from_a_fraction_to_the_scheduled_rate(self):
        f = client.LR_WARMUP_START_FRACTION
        lr = lambda samples: self.schedule(samples, warmup=200).lr()
        self.assertAlmostEqual(lr(0), f * 1e-3)
        self.assertAlmostEqual(lr(50), (f + (1 - f) * 0.25) * 1e-3)
        # The warm-up scales whatever rate the schedule gives at that point.
        self.assertAlmostEqual(lr(150), (f + (1 - f) * 0.75) * 1e-4)
        self.assertEqual(lr(200), 1e-4)

    def test_apply_sets_every_group_and_advance_counts_samples(self):
        model = torch.nn.Linear(2, 2)
        op = torch.optim.AdamW(
            [{"params": [model.weight]}, {"params": [model.bias]}], lr=1.0
        )
        schedule = self.schedule(0)
        self.assertEqual(schedule.apply(op), 1e-3)
        self.assertEqual([g["lr"] for g in op.param_groups], [1e-3, 1e-3])
        schedule.advance(64)
        schedule.advance(64)
        self.assertEqual(schedule.samples_trained, 128)
        self.assertEqual(schedule.apply(op), 1e-4)

    def test_scheduled_step_uses_and_logs_the_rate(self):
        model = scripted_net(seed=0)
        op = client.create_optimizer(model)
        schedule = client.LrSchedule(client.parse_lr_schedule("0:2e-3"), 0)
        settings = StubTrainSettings(seed=0)
        log = Logger()
        log.start_batch()

        client.scheduled_train_step(settings, None, model, op, schedule, log)

        self.assertEqual(settings.lrs_seen, [2e-3])
        self.assertEqual(log.data[("schedule", "lr")][0], 2e-3)
        self.assertEqual(
            log.data[("schedule", "samples_trained")][0], client.BATCH_SIZE
        )

    def test_parsing(self):
        self.assertEqual(
            client.parse_lr_schedule(" 50M:5e-4, 0:1e-3 ,1.5B:1e-5"),
            [(0, 1e-3), (50_000_000, 5e-4), (1_500_000_000, 1e-5)],
        )
        self.assertEqual(client.parse_sample_count("2k"), 2000)
        for bad in ("", "10:1e-3", "0:1e-3,0:1e-4", "0:0", "0:-1", "0:1e-3,1.5:1"):
            with self.assertRaises(ValueError, msg=bad):
                client.parse_lr_schedule(bad)

    def test_defaults_parse_and_start_in_warmup(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TZ_LR_SCHEDULE", None)
            os.environ.pop("TZ_LR_WARMUP_SAMPLES", None)
            schedule = client.LrSchedule.from_env()
        self.assertEqual(schedule.points[0], (0, 1e-3))
        self.assertGreater(schedule.warmup_samples, 0)
        self.assertLess(schedule.lr(), 1e-3)


if __name__ == "__main__":
    unittest.main()
