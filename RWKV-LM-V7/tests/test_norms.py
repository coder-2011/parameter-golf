import os
import unittest
from types import SimpleNamespace

import torch

os.environ.setdefault("RWKV_COMPILE_ON", "0")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE", "4")
os.environ.setdefault("RWKV_MY_TESTING", "")

from src.model import RMSNorm, RWKV, make_norm


def make_model_args(**overrides):
    args = dict(
        vocab_size=32,
        ctx_len=8,
        n_layer=2,
        n_embd=32,
        dim_att=32,
        dim_ffn=128,
        head_size=4,
        my_testing="x070",
        rope_mode="none",
        rope_theta=10000.0,
        rope_dims=0,
        grad_cp=0,
        strategy="auto",
        lr_init=0.0,
        betas=(0.9, 0.99),
        adam_eps=1e-18,
        weight_decay=0.0,
        accelerator="CPU",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class NormTest(unittest.TestCase):
    def test_rmsnorm_matches_definition(self):
        norm = RMSNorm(4, eps=1e-5)
        norm.weight.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        y = norm(x)
        expected = (
            x
            * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5)
            * norm.weight
        )

        torch.testing.assert_close(y, expected)

    def test_make_norm_defaults_to_layernorm(self):
        norm = make_norm(SimpleNamespace(n_embd=4))

        self.assertIsInstance(norm, torch.nn.LayerNorm)

    def test_make_norm_accepts_rmsnorm(self):
        norm = make_norm(SimpleNamespace(n_embd=4, norm_type="rmsnorm"))

        self.assertIsInstance(norm, RMSNorm)

    def test_rwkv_uses_requested_norm_type(self):
        layernorm_model = RWKV(make_model_args(norm_type="layernorm"))
        rmsnorm_model = RWKV(make_model_args(norm_type="rmsnorm"))

        self.assertIsInstance(layernorm_model.blocks[0].ln1, torch.nn.LayerNorm)
        self.assertIsInstance(layernorm_model.ln_out, torch.nn.LayerNorm)
        self.assertIsInstance(rmsnorm_model.blocks[0].ln0, RMSNorm)
        self.assertIsInstance(rmsnorm_model.blocks[0].ln1, RMSNorm)
        self.assertIsInstance(rmsnorm_model.blocks[0].ln2, RMSNorm)
        self.assertIsInstance(rmsnorm_model.ln_out, RMSNorm)
        self.assertEqual(list(rmsnorm_model.blocks[0].ln1.state_dict()), ["weight"])


if __name__ == "__main__":
    unittest.main()
