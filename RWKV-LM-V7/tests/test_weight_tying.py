import os
import unittest
from types import SimpleNamespace

import torch

os.environ.setdefault("RWKV_COMPILE_ON", "0")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE", "4")
os.environ.setdefault("RWKV_MY_TESTING", "")

from src.muon import classify_rwkv_muon_parameters
from src.model import RWKV


def make_args(**overrides):
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
        grad_cp=0,
        strategy="auto",
        lr_init=0.0,
        betas=(0.9, 0.99),
        adam_eps=1e-18,
        weight_decay=0.0,
        accelerator="CPU",
        optimizer="adamw",
        tie_embeddings=0,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class WeightTyingTest(unittest.TestCase):
    def test_missing_flag_defaults_to_untied(self):
        args = make_args()
        delattr(args, "tie_embeddings")

        model = RWKV(args)

        self.assertIsNot(model.head.weight, model.emb.weight)
        self.assertNotEqual(model.head.weight.data_ptr(), model.emb.weight.data_ptr())

    def test_default_keeps_embedding_and_head_separate(self):
        model = RWKV(make_args())

        self.assertIsNot(model.head.weight, model.emb.weight)
        self.assertNotEqual(
            model.head.weight.data_ptr(),
            model.emb.weight.data_ptr(),
        )

    def test_tied_embedding_and_head_share_parameter(self):
        model = RWKV(make_args(tie_embeddings=1))

        self.assertIs(model.head.weight, model.emb.weight)
        self.assertEqual(
            model.head.weight.data_ptr(),
            model.emb.weight.data_ptr(),
        )
        self.assertEqual(
            sum(1 for name, _ in model.named_parameters() if name in {"emb.weight", "head.weight"}),
            1,
        )

    def test_muon_parameter_groups_follow_tying(self):
        untied_groups = classify_rwkv_muon_parameters(
            RWKV(make_args(tie_embeddings=0)).named_parameters()
        )
        tied_groups = classify_rwkv_muon_parameters(
            RWKV(make_args(tie_embeddings=1)).named_parameters()
        )

        self.assertEqual([name for name, _ in untied_groups["embed"]], ["emb.weight"])
        self.assertEqual([name for name, _ in untied_groups["head"]], ["head.weight"])
        self.assertEqual([name for name, _ in tied_groups["embed"]], ["emb.weight"])
        self.assertEqual(tied_groups["head"], [])

    def test_tied_model_accepts_untied_state_dict(self):
        untied = RWKV(make_args())
        tied = RWKV(make_args(tie_embeddings=1))
        state = untied.state_dict()

        tied.load_state_dict(state, strict=True)

        torch.testing.assert_close(tied.emb.weight, state["head.weight"])
        self.assertEqual(tied.head.weight.data_ptr(), tied.emb.weight.data_ptr())

    def test_untied_model_accepts_tied_state_dict(self):
        tied = RWKV(make_args(tie_embeddings=1))
        untied = RWKV(make_args(tie_embeddings=0))
        state = tied.state_dict()

        untied.load_state_dict(state, strict=True)

        torch.testing.assert_close(untied.emb.weight, state["emb.weight"])
        torch.testing.assert_close(untied.head.weight, state["head.weight"])
        self.assertNotEqual(untied.head.weight.data_ptr(), untied.emb.weight.data_ptr())


if __name__ == "__main__":
    unittest.main()
