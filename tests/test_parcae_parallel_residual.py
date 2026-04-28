import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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
    h.coda_moe_num_experts = 0
    h.deepseek_moe_num_base_experts = 0
    h.qat_bits = 0
    h.qat_linear = True
    h.qat_tied_output = True
    h.xsa_last_n = 0
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


def _freqs() -> tuple[torch.Tensor, torch.Tensor]:
    return pg.precompute_freqs_cos_sin(8, 5, 10000.0)


def test_smear_gate_matches_shifted_blend():
    smear = pg.SmearGate(3)
    with torch.no_grad():
        smear.gate.copy_(torch.tensor([-2.0, 0.0, 2.0]))
    x = torch.randn(2, 4, 3)

    actual = smear(x)

    g = torch.sigmoid(smear.gate.to(dtype=x.dtype))[None, None, :]
    x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
    expected = (1 - g) * x + g * x_prev
    assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_smear_gate_is_wired_into_gpt_backward():
    torch.manual_seed(123)
    h = _tiny_h()
    model = pg.GPT(h)
    input_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len))
    target_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len))

    loss = model.forward_model(input_ids, labels=target_ids, num_steps_pair=torch.tensor([0, 1]))["loss"]

    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    assert model.smear.gate.grad is not None
    assert torch.isfinite(model.smear.gate.grad).all()


def test_xsa_efficient_matches_explicit_gqa_projection():
    attn = pg.ParcaeCausalSelfAttention(
        dim=16,
        num_heads=4,
        num_kv_heads=2,
        rope_base=10000.0,
        layer_id=0,
        n_layer=4,
        qk_norm=False,
        qk_bias=False,
        clip_qkv=None,
        rope_dims=4,
    )
    y = torch.randn(2, 5, 4, 4)
    v = torch.randn(2, 5, 2, 4)

    actual = attn._xsa_efficient(y, v)

    v_expanded = v.repeat_interleave(2, dim=2)
    v_norm = F.normalize(v_expanded, dim=-1)
    expected = y - (y * v_norm).sum(dim=-1, keepdim=True) * v_norm

    assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_xsa_zero_values_are_finite_and_leave_attention_output_unchanged():
    attn = pg.ParcaeCausalSelfAttention(
        dim=16,
        num_heads=4,
        num_kv_heads=2,
        rope_base=10000.0,
        layer_id=0,
        n_layer=4,
        qk_norm=False,
        qk_bias=False,
        clip_qkv=None,
        rope_dims=4,
    )
    y = torch.randn(2, 5, 4, 4)
    v = torch.zeros(2, 5, 2, 4)

    actual = attn._xsa_efficient(y, v)

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, y, atol=0.0, rtol=0.0)


def test_xsa_routes_last_effective_layers_with_recurrence_depth():
    h = _tiny_h()
    h.mean_recurrence = 2
    h.xsa_last_n = 4
    model = pg.GPT(h)

    assert model.xsa_active_layer_ids() == [2, 3, 4, 5]
    assert [model._use_xsa_for_effective_layer(i) for i in range(6)] == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]

    h = _tiny_h()
    h.n_layers_in_prelude = 2
    h.n_layers_in_recurrent_block = 2
    h.n_layers_in_coda = 2
    h.mean_recurrence = 2
    h.xsa_last_n = 3
    model = pg.GPT(h)

    assert model.xsa_active_layer_ids() == [5, 6, 7]


def test_xsa_does_not_change_state_dict_contract_or_iterate_return_shape():
    h = _tiny_h()
    baseline = pg.GPT(h)

    h_xsa = _tiny_h()
    h_xsa.xsa_last_n = 2
    xsa_model = pg.GPT(h_xsa)

    assert baseline.state_dict().keys() == xsa_model.state_dict().keys()
    assert sum(p.numel() for p in baseline.parameters()) == sum(p.numel() for p in xsa_model.parameters())

    input_embeds = torch.randn(2, h_xsa.train_seq_len, h_xsa.model_dim)
    freqs_cos = xsa_model.recurrent_freqs_cos[:, : h_xsa.train_seq_len]
    freqs_sin = xsa_model.recurrent_freqs_sin[:, : h_xsa.train_seq_len]
    xsa_model._current_input_ids = torch.randint(0, h_xsa.vocab_size, (2, h_xsa.train_seq_len))
    out = xsa_model.iterate_forward(input_embeds, freqs_cos, freqs_sin, None, torch.tensor([0, 1]))

    assert len(out) == 4
    assert xsa_model._last_total_steps_int == 1


def test_xsa_enabled_tiny_forward_backward_is_finite():
    torch.manual_seed(654)
    h = _tiny_h()
    h.xsa_last_n = 2
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
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.parameters()
    )


def test_record_parallel_block_matches_reference_formula():
    block = _block(parallel_residual=True, record_residual=True)
    x = torch.randn(2, 5, 16)
    x0 = torch.randn(2, 5, 16)

    freqs_cos, freqs_sin = _freqs()
    actual = block(x, freqs_cos, freqs_sin, x0=x0)

    mix = block.resid_mix.to(dtype=x.dtype)
    x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    attn_out = block.attn(block.norm_1(x_in) * block.ln_scale_factor, freqs_cos, freqs_sin)
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

    freqs_cos, freqs_sin = _freqs()
    actual = block(x, freqs_cos, freqs_sin, x0=x0)

    mix = block.resid_mix.to(dtype=x.dtype)
    x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    attn_out = block.attn(block.norm_1(x_in) * block.ln_scale_factor, freqs_cos, freqs_sin)
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


def test_qat_schedule_is_latched_outside_forward():
    h = _tiny_h()
    h.qat_bits = 6
    h.qat_start_step = 3
    model = pg.GPT(h)

    model.set_training_step(2)
    assert not model._qat_enabled
    assert all(not module.qat_enabled for module in model.qat_linears)

    model.set_training_step(3)
    assert model._qat_enabled
    assert all(module.qat_enabled for module in model.qat_linears)

    model.step = 0
    x = torch.randint(0, h.vocab_size, (2, h.train_seq_len))
    y = torch.randint(0, h.vocab_size, (2, h.train_seq_len))
    loss = model(x, y)

    assert torch.isfinite(loss)
    assert model._qat_enabled


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


def test_ema_state_matches_record_update_formula():
    model = torch.nn.Linear(2, 1, bias=False).bfloat16()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 5.0]]))

    ema = pg.ParameterAverager(model, cpu=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 9.0]]))
    ema.update_ema(decay=0.25)
    ema.load_into(model)

    assert torch.allclose(model.weight.float(), torch.tensor([[2.5, 8.0]]))


def test_swa_state_averages_checkpoints_and_loads_strictly():
    model = pg.GPT(_tiny_h())
    state0 = {name: t.detach().clone() for name, t in model.state_dict().items()}
    swa = pg.ParameterAverager(model, cpu=True)

    with torch.no_grad():
        for param in model.parameters():
            param.add_(2.0)
    state1 = {name: t.detach().clone() for name, t in model.state_dict().items()}
    swa.add_swa()
    swa.load_into(model, divisor=2)

    loaded = model.state_dict()
    for name, t0 in state0.items():
        if t0.is_floating_point():
            expected = ((t0.float() + state1[name].float()) / 2).to(dtype=loaded[name].dtype)
            assert torch.allclose(loaded[name], expected)
        else:
            assert torch.equal(loaded[name], state1[name])


def test_ema_application_changes_model_used_for_validation_forward():
    torch.manual_seed(321)
    h = _tiny_h()
    h.state_init = "zero"
    model = pg.GPT(h)
    model.eval()
    input_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len))
    num_steps_pair = torch.tensor([0, 1])
    before = model.forward_logits(input_ids, num_steps_pair=num_steps_pair)

    avg_state = pg.ParameterAverager(model, cpu=False)
    for tensor in avg_state.avg_params:
        tensor.zero_()
    avg_state.load_into(model)

    after = model.forward_logits(input_ids, num_steps_pair=num_steps_pair)
    assert torch.isfinite(after).all()
    assert not torch.allclose(before, after)
    assert torch.count_nonzero(after).item() == 0


if __name__ == "__main__":
    for test_name, test_fn in list(globals().items()):
        if test_name.startswith("test_"):
            test_fn()
