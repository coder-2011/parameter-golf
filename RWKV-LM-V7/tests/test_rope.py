import os
import unittest
from types import SimpleNamespace

import torch

os.environ.setdefault("RWKV_COMPILE_ON", "0")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE", "4")
os.environ.setdefault("RWKV_MY_TESTING", "")

from src.model import RWKV_Tmix_x070


def reference_interleaved_rope(x, n_head, head_size, rope_dims, theta):
    bsz, seq_len, channels = x.shape
    x = x.view(bsz, seq_len, n_head, head_size)
    x_rope = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rope_dims, 2, dtype=torch.float32) / rope_dims)
    )
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    cos = freqs.cos().to(dtype=x.dtype).view(1, seq_len, 1, rope_dims // 2)
    sin = freqs.sin().to(dtype=x.dtype).view(1, seq_len, 1, rope_dims // 2)

    pairs = x_rope.view(bsz, seq_len, n_head, rope_dims // 2, 2)
    rotated = torch.empty_like(pairs)
    rotated[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
    rotated[..., 1] = pairs[..., 1] * cos + pairs[..., 0] * sin
    return torch.cat((rotated.view(bsz, seq_len, n_head, rope_dims), x_pass), dim=-1).view(
        bsz, seq_len, channels
    )


def make_args(**overrides):
    args = dict(
        vocab_size=1892,
        ctx_len=16,
        n_layer=2,
        n_embd=8,
        dim_att=8,
        dim_ffn=32,
        head_size=4,
        my_testing="x070",
        rope_mode="none",
        rope_theta=10000.0,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class RopeTest(unittest.TestCase):
    def test_rope_preserves_pair_norms_and_position_zero(self):
        module = RWKV_Tmix_x070(make_args(rope_mode="rk"), layer_id=0)
        x = torch.randn(2, 5, 8)

        y = module._apply_rope(x)

        torch.testing.assert_close(y[:, 0], x[:, 0])
        x_pairs = x.view(2, 5, 2, 2, 2).float().square().sum(dim=-1)
        y_pairs = y.view(2, 5, 2, 2, 2).float().square().sum(dim=-1)
        torch.testing.assert_close(y_pairs, x_pairs, rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(y[:, 1:], x[:, 1:]))

    def test_partial_rope_leaves_tail_dims_unchanged(self):
        module = RWKV_Tmix_x070(make_args(rope_mode="rk", rope_dims=2), layer_id=0)
        x = torch.randn(2, 5, 8)

        y = module._apply_rope(x)

        x_heads = x.view(2, 5, 2, 4)
        y_heads = y.view(2, 5, 2, 4)
        torch.testing.assert_close(y_heads[..., 2:], x_heads[..., 2:])
        self.assertFalse(torch.allclose(y_heads[:, 1:, :, :2], x_heads[:, 1:, :, :2]))

    def test_rope_matches_interleaved_reference(self):
        for rope_dims in (2, 4):
            module = RWKV_Tmix_x070(
                make_args(rope_mode="rk", rope_dims=rope_dims), layer_id=0
            )
            x = torch.randn(2, 5, 8)

            y = module._apply_rope(x)
            expected = reference_interleaved_rope(
                x, n_head=2, head_size=4, rope_dims=rope_dims, theta=10000.0
            )

            torch.testing.assert_close(y, expected)

    def test_rope_state_is_not_persistent(self):
        disabled = RWKV_Tmix_x070(make_args(rope_mode="none"), layer_id=0)
        enabled = RWKV_Tmix_x070(make_args(rope_mode="rk"), layer_id=0)

        self.assertFalse(hasattr(disabled, "rope_inv_freq"))
        self.assertTrue(hasattr(enabled, "rope_inv_freq"))
        self.assertNotIn("rope_inv_freq", enabled.state_dict())

    def test_rope_requires_even_rope_dims(self):
        with self.assertRaisesRegex(ValueError, "positive even value <= head_size"):
            RWKV_Tmix_x070(
                make_args(rope_mode="rk", n_embd=6, dim_att=6, head_size=3),
                layer_id=0,
            )


if __name__ == "__main__":
    unittest.main()
