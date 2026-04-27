import unittest
from types import SimpleNamespace

from src.trainer import scheduled_lr


def make_args(**overrides):
    args = dict(
        lr_init=1.0,
        lr_final=0.1,
        warmup_steps=0,
        cooldown_steps=0,
        epoch_steps=100,
        my_exit_tokens=0,
        ctx_len=10,
        real_bsz=2,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class LRScheduleTest(unittest.TestCase):
    def test_no_cooldown_keeps_lr_flat_after_warmup(self):
        args = make_args(cooldown_steps=0, warmup_steps=0)

        self.assertEqual(scheduled_lr(args, 0), 1.0)
        self.assertEqual(scheduled_lr(args, 50), 1.0)
        self.assertEqual(scheduled_lr(args, 100), 1.0)

    def test_warmup_is_preserved(self):
        args = make_args(warmup_steps=10)

        self.assertAlmostEqual(scheduled_lr(args, 0), 0.01)
        self.assertAlmostEqual(scheduled_lr(args, 5), 0.505)
        self.assertAlmostEqual(scheduled_lr(args, 10), 1.0)

    def test_fixed_step_cooldown_decays_tail_to_final_lr(self):
        args = make_args(cooldown_steps=20, epoch_steps=100)

        self.assertEqual(scheduled_lr(args, 79), 1.0)
        self.assertEqual(scheduled_lr(args, 80), 1.0)
        self.assertAlmostEqual(scheduled_lr(args, 90), 0.55)
        self.assertAlmostEqual(scheduled_lr(args, 100), 0.1)
        self.assertAlmostEqual(scheduled_lr(args, 120), 0.1)

    def test_token_schedule_takes_precedence_over_fixed_step_cooldown(self):
        args = make_args(
            cooldown_steps=20,
            epoch_steps=100,
            my_exit_tokens=100 * 10 * 2,
        )

        self.assertAlmostEqual(scheduled_lr(args, 50), 0.55)


if __name__ == "__main__":
    unittest.main()
