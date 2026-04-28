import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_gpt_parcae as pg


def _tiny_h() -> pg.Hyperparameters:
    h = pg.Hyperparameters()
    h.vocab_size = 32
    h.train_seq_len = 8
    h.model_dim = 16
    h.recurrent_dim = 16
    h.num_heads = 2
    h.num_kv_heads = 1
    h.recurrent_num_heads = 2
    h.mlp_mult = 2
    h.recurrent_intermediate_dim = 32
    h.n_layers_in_prelude = 1
    h.n_layers_in_recurrent_block = 2
    h.n_layers_in_coda = 1
    h.mean_recurrence = 1
    h.mean_backprop_depth = 1
    h.recurrent_iteration_method = "per-batch"
    h.sampling_scheme = "fixed"
    h.curriculum_target = "forward"
    h.injection_type = "diagonal"
    h.injection_swiglu_scale = 0.0
    h.residual_mode = "sequential"
    h.parallel_residual_scope = "none"
    h.parallel_residual_start = -1
    h.parallel_residual_ln_scale = True
    h.state_init = "like-init"
    h.prelude_norm = False
    h.qk_norm = False
    h.qk_bias = False
    h.clip_qkv = None
    h.outer_rope_dims = 0
    h.recurrent_rope_dims = 0
    h.recurrent_layer_rope_dims = ""
    h.bigram_hash_buckets = 0
    h.use_value_embeddings = False
    h.gradient_checkpointing = False
    h.activation_checkpoint_impl = "none"
    h.monitoring = False
    h.tie_embeddings = True
    h.poe_num_experts = 1
    h.laurel_scope = "none"
    h.laurel_rank = 0
    h.coda_moe_num_experts = 0
    h.deepseek_moe_num_base_experts = 0
    h.qat_bits = 0
    h.qat_linear = True
    h.qat_tied_output = True
    return h


def _block(*, parallel_residual: bool, record_residual: bool) -> pg.TransformerPreNormBlock:
    torch.manual_seed(123)
    return pg.TransformerPreNormBlock(
        dim=16,
        num_heads=2,
        num_kv_heads=1,
        hidden_dim=32,
        rope_base=10000.0,
        layer_id=3,
        n_layer=6,
        qk_norm=False,
        qk_bias=False,
        clip_qkv=None,
        mlp_class_name="BaseMLP",
        rope_dims=8,
        parallel_residual=parallel_residual,
        record_residual=record_residual,
        ln_scale=True,
        residual_layer_idx=3,
    )


def _freqs() -> torch.Tensor:
    return pg.precompute_freqs_cis(8, 5, 10000.0)


def test_record_parallel_block_matches_reference_formula():
    block = _block(parallel_residual=True, record_residual=True)
    x = torch.randn(2, 5, 16)
    x0 = torch.randn(2, 5, 16)

    actual = block(x, _freqs(), x0=x0)

    mix = block.resid_mix.to(dtype=x.dtype)
    x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    attn_out = block.attn(block.norm_1(x_in) * block.ln_scale_factor, _freqs())
    mlp_out = block.mlp(block.norm_2(x_in) * block.ln_scale_factor)
    expected = (
        x_in
        + block.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        + block.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
    )

    assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_record_sequential_block_matches_reference_formula():
    block = _block(parallel_residual=False, record_residual=True)
    x = torch.randn(2, 5, 16)
    x0 = torch.randn(2, 5, 16)

    actual = block(x, _freqs(), x0=x0)

    mix = block.resid_mix.to(dtype=x.dtype)
    x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    attn_out = block.attn(block.norm_1(x_in) * block.ln_scale_factor, _freqs())
    x_after_attn = x_in + block.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
    expected = x_after_attn + block.mlp_scale.to(dtype=x.dtype)[None, None, :] * block.mlp(
        block.norm_2(x_after_attn) * block.ln_scale_factor
    )

    assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_parallel_residual_scope_and_start_use_physical_layer_indices():
    h = _tiny_h()
    h.residual_mode = "parallel"
    h.parallel_residual_scope = "all"
    h.parallel_residual_start = 2
    model = pg.GPT(h)

    assert model.prelude[0].record_residual
    assert not model.prelude[0].parallel_residual
    assert model.core_block[0].record_residual
    assert not model.core_block[0].parallel_residual
    assert model.core_block[1].record_residual
    assert model.core_block[1].parallel_residual
    assert model.coda[0].record_residual
    assert model.coda[0].parallel_residual


def test_core_scope_does_not_add_record_controls_outside_core():
    h = _tiny_h()
    h.residual_mode = "parallel"
    h.parallel_residual_scope = "core"
    model = pg.GPT(h)

    assert not model.prelude[0].record_residual
    assert all(block.record_residual and block.parallel_residual for block in model.core_block)
    assert not model.coda[0].record_residual


def test_control_tensors_restore_to_fp32_after_bfloat16_cast():
    h = _tiny_h()
    h.residual_mode = "parallel"
    h.parallel_residual_scope = "core"
    model = pg.GPT(h).bfloat16()
    pg.restore_low_dim_params_to_fp32(model)

    control_names = [
        "core_block.0.attn_scale",
        "core_block.0.mlp_scale",
        "core_block.0.resid_mix",
    ]
    params = dict(model.named_parameters())
    for name in control_names:
        assert params[name].dtype == torch.float32
    assert params["core_block.0.attn.c_q.weight"].dtype == torch.bfloat16


def test_parallel_residual_tiny_forward_backward_has_finite_control_grads():
    torch.manual_seed(456)
    h = _tiny_h()
    h.residual_mode = "parallel"
    h.parallel_residual_scope = "core"
    model = pg.GPT(h)
    input_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len))
    target_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len))

    loss = model.forward_model(
        input_ids,
        labels=target_ids,
        num_steps_pair=torch.tensor([0, 1]),
    )["loss"]
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()

    params = dict(model.named_parameters())
    for name in (
        "core_block.0.attn_scale",
        "core_block.0.mlp_scale",
        "core_block.0.resid_mix",
    ):
        grad = params[name].grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_parallel_residual_quantized_state_dict_loads_strictly():
    torch.manual_seed(789)
    h = _tiny_h()
    h.residual_mode = "parallel"
    h.parallel_residual_scope = "core"
    model = pg.GPT(h)

    quant_obj, _ = pg.quantize_state_dict_int(model.state_dict(), bits=8)
    restored_state = pg.dequantize_state_dict_int(quant_obj)
    restored = pg.GPT(h)
    restored.load_state_dict(restored_state, strict=True)

    input_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len))
    logits = restored.forward_logits(input_ids, num_steps_pair=torch.tensor([0, 1]))
    assert torch.isfinite(logits).all()
    assert restored_state["core_block.0.attn_scale"].dtype == torch.float32
    assert restored_state["core_block.0.mlp_scale"].dtype == torch.float32
    assert restored_state["core_block.0.resid_mix"].dtype == torch.float32


if __name__ == "__main__":
    for test_name, test_fn in list(globals().items()):
        if test_name.startswith("test_"):
            test_fn()
