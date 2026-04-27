import os
import unittest
from types import SimpleNamespace

import torch

os.environ.setdefault("RWKV_COMPILE_ON", "0")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE", "64")
os.environ.setdefault("RWKV_MY_TESTING", "")

from src.model import CausalSelfAttention, RWKV, RWKV_Tmix_x070, is_attention_layer


def reference_attention_rope(x, theta, rope_dims=None):
    bsz, seq_len, n_head, head_dim = x.shape
    rope_dims = rope_dims or head_dim
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rope_dims, 2, dtype=torch.float32) / rope_dims)
    )
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    cos = freqs.cos().to(dtype=x.dtype).view(1, seq_len, 1, rope_dims // 2)
    sin = freqs.sin().to(dtype=x.dtype).view(1, seq_len, 1, rope_dims // 2)
    x_rope = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    pairs = x_rope.view(bsz, seq_len, n_head, rope_dims // 2, 2)
    rotated = torch.empty_like(pairs)
    rotated[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
    rotated[..., 1] = pairs[..., 1] * cos + pairs[..., 0] * sin
    return torch.cat((rotated.view(bsz, seq_len, n_head, rope_dims), x_pass), dim=-1)


def make_args(**overrides):
    args = dict(
        vocab_size=1892,
        ctx_len=16,
        n_layer=8,
        n_embd=64,
        dim_att=64,
        dim_ffn=256,
        head_size=64,
        my_testing="",
        rope_mode="none",
        rope_theta=10000.0,
        rope_dims=0,
        norm_type="layernorm",
        learned_shift_state=0,
        attn_every=0,
        attn_offset=0,
        attn_heads=0,
        attn_dim=0,
        attn_dropout=0.0,
        attn_rope=1,
        tie_embeddings=0,
        optimizer="adamw",
        grad_cp=0,
        lr_init=0.0,
        betas=(0.9, 0.99),
        adam_eps=1e-18,
        weight_decay=0.0,
        strategy="auto",
        accelerator="cpu",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class HybridAttentionTest(unittest.TestCase):
    def test_attention_schedule_keeps_layer_zero_recurrent(self):
        args = make_args(attn_every=4, attn_offset=4)

        selected = [i for i in range(args.n_layer) if is_attention_layer(args, i)]

        self.assertEqual(selected, [4])
        self.assertFalse(is_attention_layer(args, 0))

    def test_model_default_off_uses_rwkv_time_mix(self):
        model = RWKV(make_args())

        self.assertTrue(all(isinstance(block.att, RWKV_Tmix_x070) for block in model.blocks))

    def test_model_replaces_selected_layers_with_attention(self):
        model = RWKV(make_args(attn_every=4, attn_offset=4))

        self.assertIsInstance(model.blocks[4].att, CausalSelfAttention)
        self.assertIsInstance(model.blocks[0].att, RWKV_Tmix_x070)
        self.assertIsInstance(model.blocks[1].att, RWKV_Tmix_x070)

    def test_attention_forward_backward(self):
        module = CausalSelfAttention(make_args(attn_dim=64, attn_heads=1), layer_id=4)
        x = torch.randn(2, 8, 64, requires_grad=True)

        y, v_first = module(x, None)
        loss = y.square().mean()
        loss.backward()

        self.assertIsNone(v_first)
        self.assertEqual(y.shape, x.shape)
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_attention_rope_matches_interleaved_reference(self):
        module = CausalSelfAttention(
            make_args(attn_dim=64, attn_heads=2, rope_theta=10000.0), layer_id=4
        )
        x = torch.randn(2, 5, 2, 32)

        y = module._apply_rope(x)
        expected = reference_attention_rope(x, theta=10000.0)

        torch.testing.assert_close(y, expected)
        torch.testing.assert_close(y[:, 0], x[:, 0])
        self.assertFalse(torch.allclose(y[:, 1:], x[:, 1:]))

    def test_attention_partial_rope_leaves_tail_dims_unchanged(self):
        module = CausalSelfAttention(
            make_args(attn_dim=64, attn_heads=2, rope_dims=16), layer_id=4
        )
        x = torch.randn(2, 5, 2, 32)

        y = module._apply_rope(x)
        expected = reference_attention_rope(x, theta=10000.0, rope_dims=16)

        torch.testing.assert_close(y, expected)
        torch.testing.assert_close(y[..., 16:], x[..., 16:])
        self.assertFalse(torch.allclose(y[:, 1:, :, :16], x[:, 1:, :, :16]))

    def test_attention_rope_requires_valid_rope_dims(self):
        for rope_dims in (15, 34):
            with self.subTest(rope_dims=rope_dims):
                with self.assertRaisesRegex(ValueError, "attention rope_dims"):
                    CausalSelfAttention(
                        make_args(attn_dim=64, attn_heads=2, rope_dims=rope_dims),
                        layer_id=4,
                    )

    def test_attention_state_dict_roundtrip(self):
        module = CausalSelfAttention(make_args(attn_dim=64, attn_heads=1), layer_id=4)
        fresh = CausalSelfAttention(make_args(attn_dim=64, attn_heads=1), layer_id=4)

        fresh.load_state_dict(module.state_dict(), strict=True)

        self.assertNotIn("rope_inv_freq", module.state_dict())
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, fresh.state_dict()[key])


if __name__ == "__main__":
    unittest.main()
