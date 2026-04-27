import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

os.environ.setdefault("RWKV_COMPILE_ON", "0")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE", "4")
os.environ.setdefault("RWKV_MY_TESTING", "")

from src.model import RWKV
from src.quant import (
    dequantize_state_dict_int,
    load_quantized_state_dict,
    pack_quantized_tensor,
    quantize_state_dict_int,
    save_quantized_state_dict,
    unpack_quantized_tensor,
)


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
        norm_type="layernorm",
        tie_embeddings=0,
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


class QuantTest(unittest.TestCase):
    def test_pack_unpack_int6_roundtrip(self):
        q = torch.tensor([-31, -17, -1, 0, 1, 18, 31], dtype=torch.int8)

        packed = pack_quantized_tensor(q, bits=6)
        unpacked = unpack_quantized_tensor(packed, tuple(q.shape), bits=6)

        torch.testing.assert_close(unpacked, q)
        self.assertLess(packed.numel(), q.numel())

    def test_pack_rejects_values_outside_int6_range(self):
        q = torch.tensor([-32, 0, 31], dtype=torch.int8)

        with self.assertRaisesRegex(ValueError, "out of int6 range"):
            pack_quantized_tensor(q, bits=6)

    def test_quantized_state_dict_dequantizes_shapes(self):
        state = {
            "matrix": torch.linspace(-1.0, 1.0, steps=12).reshape(3, 4),
            "vector": torch.linspace(-0.5, 0.5, steps=5),
            "step": torch.tensor(7, dtype=torch.int64),
        }

        obj = quantize_state_dict_int(state, bits=6)
        dequant = dequantize_state_dict_int(obj)

        self.assertEqual(dequant["matrix"].shape, state["matrix"].shape)
        self.assertEqual(dequant["vector"].shape, state["vector"].shape)
        torch.testing.assert_close(dequant["step"], state["step"])
        self.assertEqual(obj["__quant_format__"], "rwkv_int6_per_row_2d_packed_v1")

    def test_zero_and_tiny_values_do_not_underflow_to_bad_scales(self):
        state = {
            "zeros": torch.zeros(4, 8),
            "tiny": torch.full((4,), 1.0e-45),
        }

        obj = quantize_state_dict_int(state, bits=6)
        dequant = dequantize_state_dict_int(obj)

        for scale in obj["scales"].values():
            self.assertTrue(torch.isfinite(scale.float()).all())
            self.assertGreaterEqual(float(scale.float().min()), 2.0**-24)
        torch.testing.assert_close(dequant["zeros"], state["zeros"])
        torch.testing.assert_close(dequant["tiny"], torch.zeros_like(state["tiny"]))

    def test_large_values_do_not_overflow_or_exceed_int6_range(self):
        state = {
            "large_matrix": torch.tensor(
                [[-1.0e20, 0.0, 1.0e20], [3.0e19, -2.0e19, 1.0e19]]
            ),
            "large_vector": torch.tensor([-1.0e20, 0.0, 1.0e20]),
        }

        obj = quantize_state_dict_int(state, bits=6)
        dequant = dequantize_state_dict_int(obj)

        for packed, shape in zip(obj["quantized"].values(), obj["shapes"].values()):
            q = unpack_quantized_tensor(packed, tuple(shape), bits=6)
            self.assertGreaterEqual(int(q.min()), -31)
            self.assertLessEqual(int(q.max()), 31)
        self.assertTrue(torch.isfinite(dequant["large_matrix"]).all())
        self.assertTrue(torch.isfinite(dequant["large_vector"]).all())

    def test_save_load_quantized_checkpoint_strict_model_load(self):
        model = RWKV(make_model_args())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rwkv-final.int6.ptz"
            raw_bytes, file_bytes = save_quantized_state_dict(
                model.state_dict(), path, bits=6
            )
            self.assertGreater(raw_bytes, 0)
            self.assertGreater(file_bytes, 0)

            dequant = load_quantized_state_dict(path)

        fresh = RWKV(make_model_args())
        fresh.load_state_dict(dequant, strict=True)


if __name__ == "__main__":
    unittest.main()
