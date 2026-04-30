import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

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
    h.residual_mode = "sequential"
    h.parallel_residual_scope = "none"
    h.parallel_residual_start = -1
    h.parallel_residual_ln_scale = True
    h.parallel_residual_impl = "immediate"
    h.parallel_residual_record_controls = True
    h.parallel_residual_tied_norm = False
    h.parallel_residual_in_fp32 = False
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
    h.sparse_attn_gate = False
    h.sparse_attn_gate_window = 4
    h.sparse_attn_gate_init_std = 0.0
    h.sparse_attn_gate_scale = 1.0
    h.qat_bits = 0
    h.qat_linear = True
    h.qat_embeddings = True
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


def test_smear_gate_masks_previous_token_at_bos_positions():
    smear = pg.SmearGate(2)
    with torch.no_grad():
        smear.gate.zero_()
    x = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    input_ids = torch.tensor([[1, 7, 1, 9]])

    actual = smear(x, input_ids=input_ids)

    g = torch.sigmoid(smear.gate.to(dtype=x.dtype))[None, None, :]
    x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
    x_prev[:, 0] = 0
    x_prev[:, 2] = 0
    expected = (1 - g) * x + g * x_prev
    assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_smear_gate_is_wired_into_gpt_backward():
    torch.manual_seed(123)
    h = _tiny_h()
    model = pg.GPT(h)
    model.eval()
    input_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len))
    target_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len))

    loss = model.forward_model(input_ids, labels=target_ids, num_steps_pair=torch.tensor([0, 1]))["loss"]

    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    assert model.smear.gate.grad is not None
    assert torch.isfinite(model.smear.gate.grad).all()


def test_liger_style_gated_mlp_matches_packed_gated_mlp_with_split_weights():
    torch.manual_seed(123)
    packed = pg.GatedMLP(dim=5, hidden_dim=7)
    split = pg.LigerStyleGatedMLP(dim=5, hidden_dim=7)
    with torch.no_grad():
        split.gate_proj.weight.copy_(packed.fc.weight[:7])
        split.up_proj.weight.copy_(packed.fc.weight[7:])
        split.down_proj.weight.copy_(packed.proj.weight)
    x = torch.randn(3, 4, 5)

    actual = split(x)
    expected = packed(x)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_liger_style_gated_mlp_is_wired_into_gpt():
    torch.manual_seed(123)
    h = _tiny_h()
    h.mlp_class_name = "LigerStyleGatedMLP"
    h.recurrent_mlp_class_name = "LigerGEGLUMLP"
    model = pg.GPT(h)

    assert isinstance(model.prelude[0].mlp, pg.LigerStyleGatedMLP)
    assert all(isinstance(block.mlp, pg.LigerStyleGatedMLP) for block in model.core_block)
    assert isinstance(model.coda[0].mlp, pg.LigerStyleGatedMLP)
    assert isinstance(model.core_block[0].mlp.gate_proj, pg.CastedLinear)
    assert isinstance(model.core_block[0].mlp.up_proj, pg.CastedLinear)
    assert isinstance(model.core_block[0].mlp.down_proj, pg.CastedLinear)

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
    assert model.core_block[0].mlp.gate_proj.weight.grad is not None
    assert torch.isfinite(model.core_block[0].mlp.gate_proj.weight.grad).all()


def test_fused_leaky_relu_sq_mlp_matches_naive_forward_backward():
    torch.manual_seed(123)
    fused = pg.FusedLeakyReLUSqMLP(dim=5, hidden_dim=7)
    naive_fc = torch.nn.Linear(5, 7, bias=False)
    naive_proj = torch.nn.Linear(7, 5, bias=False)
    with torch.no_grad():
        naive_fc.weight.copy_(fused.fc.weight)
        naive_proj.weight.copy_(fused.proj.weight)
    x = torch.randn(3, 4, 5, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()

    actual = fused(x)
    hidden = F.leaky_relu(naive_fc(x_ref), negative_slope=0.5).square()
    expected = naive_proj(hidden)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused.fc.weight.grad, naive_fc.weight.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused.proj.weight.grad, naive_proj.weight.grad, atol=1e-6, rtol=1e-6)


def test_fused_leaky_relu_sq_mlp_is_wired_into_gpt():
    torch.manual_seed(123)
    h = _tiny_h()
    h.mlp_class_name = "FusedLeakyReLUSqMLP"
    h.recurrent_mlp_class_name = "LeakyReLUSqMLP"
    model = pg.GPT(h)

    assert isinstance(model.prelude[0].mlp, pg.FusedLeakyReLUSqMLP)
    assert all(isinstance(block.mlp, pg.FusedLeakyReLUSqMLP) for block in model.core_block)
    assert isinstance(model.coda[0].mlp, pg.FusedLeakyReLUSqMLP)

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
    assert model.core_block[0].mlp.fc.weight.grad is not None
    assert torch.isfinite(model.core_block[0].mlp.fc.weight.grad).all()


def test_ttt_lora_adapters_are_transparent_trainable_and_removed():
    torch.manual_seed(123)
    h = _tiny_h()
    model = pg.GPT(h)
    model.eval()
    input_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len))
    target_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len))
    num_steps_pair = torch.tensor([0, 1])
    before = model.forward_logits(input_ids, num_steps_pair=num_steps_pair)
    for p in model.parameters():
        p.requires_grad_(False)

    params = pg.install_ttt_lora_adapters(model, rank=2, alpha=4.0, min_params=0, device=torch.device("cpu"))

    assert params
    assert all(p.requires_grad for p in params)
    assert torch.allclose(model.forward_logits(input_ids, num_steps_pair=num_steps_pair), before, atol=0.0, rtol=0.0)
    model.train()
    loss = model.forward_model(input_ids, labels=target_ids, num_steps_pair=num_steps_pair)["loss"]
    assert loss is not None
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert any(torch.count_nonzero(p.grad).item() > 0 for p in params)

    pg.remove_ttt_lora_adapters(model)

    assert not any("ttt_lora" in name for name, _ in model.named_parameters())
    model.eval()
    assert torch.allclose(model.forward_logits(input_ids, num_steps_pair=num_steps_pair), before, atol=0.0, rtol=0.0)


def test_ttt_no_qv_mask_allows_only_k_rows_in_packed_qkv():
    h = _tiny_h()
    model = pg.GPT(h)
    params = pg.install_ttt_lora_adapters(
        model,
        rank=2,
        alpha=4.0,
        min_params=0,
        device=torch.device("cpu"),
        mask="no_qv",
    )

    masks = pg._ttt_grad_masks(model, "no_qv")
    qkv = model.prelude[0].attn.qkv_proj
    weight_mask = masks[id(qkv.weight)]
    lora_b_mask = masks[id(qkv.ttt_lora_B)]

    assert params
    assert torch.count_nonzero(weight_mask[: qkv.out_features // 2]).item() == 0
    assert torch.count_nonzero(lora_b_mask[: qkv.out_features // 2]).item() == 0
    assert torch.count_nonzero(weight_mask[-qkv.out_features // 4 :]).item() == 0
    assert torch.count_nonzero(lora_b_mask[-qkv.out_features // 4 :]).item() == 0
    assert torch.count_nonzero(weight_mask[qkv.out_features // 2 : -qkv.out_features // 4]).item() > 0
    assert torch.count_nonzero(lora_b_mask[qkv.out_features // 2 : -qkv.out_features // 4]).item() > 0
    pg.remove_ttt_lora_adapters(model)


def test_eval_val_ttt_lora_branch_removes_adapters_after_smoke():
    torch.manual_seed(123)
    h = _tiny_h()
    h.eval_stride = 4
    h.ttt_chunk_tokens = 16
    h.ttt_batch_seqs = 2
    h.ttt_epochs = 1
    h.ttt_lora_rank = 2
    h.ttt_lora_alpha = 4.0
    h.ttt_lora_lr = 0.001
    h.ttt_lora_wd = 0.0
    h.ttt_lora_min_params = 0
    model = pg.GPT(h)
    val_tokens = torch.randint(0, h.vocab_size, (41,), dtype=torch.int64)
    base_bytes_lut = torch.ones(h.vocab_size, dtype=torch.int16)
    has_leading_space_lut = torch.zeros(h.vocab_size, dtype=torch.bool)
    is_boundary_token_lut = torch.zeros(h.vocab_size, dtype=torch.bool)
    before = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, pg.CastedLinear) and (".attn." in name or ".mlp." in name)
    }

    loss, bpb = pg.eval_val_ttt(
        h,
        model,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )

    assert torch.isfinite(torch.tensor(loss))
    assert torch.isfinite(torch.tensor(bpb))
    assert not any("ttt_lora" in name for name, _ in model.named_parameters())
    after = {
        name: module.weight.detach()
        for name, module in model.named_modules()
        if isinstance(module, pg.CastedLinear) and name in before
    }
    assert any(not torch.allclose(after[name], weight) for name, weight in before.items())


def test_packed_qkv_attention_loads_legacy_separate_qkv_weights_strictly():
    torch.manual_seed(123)
    original = pg.ParcaeCausalSelfAttention(
        dim=16,
        num_heads=4,
        num_kv_heads=2,
        rope_base=10000.0,
        layer_id=0,
        n_layer=4,
        qk_norm=True,
        qk_bias=True,
        clip_qkv=3.0,
        rope_dims=4,
    )
    legacy_state = original.state_dict()
    qkv = legacy_state.pop("qkv_proj.weight")
    q_w, k_w, v_w = qkv.split((original.q_dim, original.kv_dim, original.kv_dim), dim=0)
    legacy_state["c_q.weight"] = q_w
    legacy_state["c_k.weight"] = k_w
    legacy_state["c_v.weight"] = v_w

    restored = pg.ParcaeCausalSelfAttention(
        dim=16,
        num_heads=4,
        num_kv_heads=2,
        rope_base=10000.0,
        layer_id=0,
        n_layer=4,
        qk_norm=True,
        qk_bias=True,
        clip_qkv=3.0,
        rope_dims=4,
    )
    restored.load_state_dict(legacy_state, strict=True)

    x = torch.randn(2, 5, 16)
    freqs_cos, freqs_sin = pg.precompute_freqs_cos_sin(4, 5, 10000.0)
    mask = torch.zeros(1, 1, 5, 5)
    expected = original(x, freqs_cos, freqs_sin, mask=mask)
    actual = restored(x, freqs_cos, freqs_sin, mask=mask)

    assert "qkv_proj.weight" in restored.state_dict()
    assert "c_q.weight" not in restored.state_dict()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_packed_qkv_is_wired_into_gpt_forward():
    h = _tiny_h()
    model = pg.GPT(h)

    assert isinstance(model.prelude[0].attn.qkv_proj, pg.CastedLinear)
    assert isinstance(model.core_block[0].attn.qkv_proj, pg.CastedLinear)

    input_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len))
    target_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len))
    loss = model.forward_model(
        input_ids,
        labels=target_ids,
        num_steps_pair=torch.tensor([0, 1]),
    )["loss"]

    assert loss is not None
    assert torch.isfinite(loss)


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


@pytest.mark.skipif(not torch.cuda.is_available() or pg.triton is None, reason="requires CUDA and Triton")
@pytest.mark.parametrize("qk_norm", [False, True])
def test_fused_qkv_postprocess_matches_reference_forward_backward(qk_norm: bool):
    torch.manual_seed(123)
    batch, seq_len, n_head, n_kv_head, head_dim = 2, 7, 4, 2, 8
    rope_dims = 6
    qkv = torch.randn(
        batch,
        seq_len,
        n_head + 2 * n_kv_head,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    qkv_ref = qkv.detach().clone().requires_grad_(True)
    freqs_cos, freqs_sin = [
        t.to(device="cuda") for t in pg.precompute_freqs_cos_sin(rope_dims, seq_len, 10000.0)
    ]

    q, k, v = pg.fused_qkv_postprocess(
        qkv,
        freqs_cos,
        freqs_sin,
        n_head,
        n_kv_head,
        head_dim,
        rope_dims,
        qk_norm,
    )
    q_ref = qkv_ref[:, :, :n_head]
    k_ref = qkv_ref[:, :, n_head : n_head + n_kv_head]
    v_ref = qkv_ref[:, :, n_head + n_kv_head :]
    q_ref, k_ref = pg.apply_rotary_emb_complex_like(q_ref, k_ref, freqs_cos, freqs_sin, rope_dims)
    if qk_norm:
        q_ref = F.rms_norm(q_ref, (q_ref.size(-1),))
        k_ref = F.rms_norm(k_ref, (k_ref.size(-1),))

    qk_atol = 2e-2 if qk_norm else 0.0
    qk_rtol = 2e-2 if qk_norm else 0.0
    assert torch.allclose(q, q_ref, atol=qk_atol, rtol=qk_rtol)
    assert torch.allclose(k, k_ref, atol=qk_atol, rtol=qk_rtol)
    assert torch.allclose(v, v_ref, atol=0.0, rtol=0.0)

    dq = torch.randn_like(q)
    dk = torch.randn_like(k)
    dv = torch.randn_like(v)
    (q.float() * dq.float()).sum().add_((k.float() * dk.float()).sum()).add_((v.float() * dv.float()).sum()).backward()
    (q_ref.float() * dq.float()).sum().add_((k_ref.float() * dk.float()).sum()).add_((v_ref.float() * dv.float()).sum()).backward()

    assert qkv.grad is not None
    assert qkv_ref.grad is not None
    assert torch.allclose(qkv.grad, qkv_ref.grad, atol=2e-2, rtol=2e-2)


def test_sparse_attn_gate_zero_init_halves_attention_output_and_gets_gradient():
    torch.manual_seed(123)
    base = pg.ParcaeCausalSelfAttention(
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
    gated = pg.ParcaeCausalSelfAttention(
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
        sparse_attn_gate=True,
        sparse_attn_gate_window=4,
        sparse_attn_gate_init_std=0.0,
    )
    gated.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(2, 5, 16)
    freqs_cos, freqs_sin = pg.precompute_freqs_cos_sin(4, 5, 10000.0)
    mask = torch.zeros(1, 1, 5, 5)

    expected = base(x, freqs_cos, freqs_sin, mask=mask) * 0.5
    actual = gated(x, freqs_cos, freqs_sin, mask=mask)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    assert gated.attn_gate_w.grad is not None
    assert torch.isfinite(gated.attn_gate_w.grad).all()


def test_sparse_attn_gate_is_wired_into_gpt_and_control_tensor_patterns():
    torch.manual_seed(123)
    h = _tiny_h()
    h.sparse_attn_gate = True
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
    gate_params = [(name, p) for name, p in model.named_parameters() if name.endswith("attn_gate_w")]
    assert gate_params
    assert all(any(pattern in name for pattern in pg.CONTROL_TENSOR_NAME_PATTERNS) for name, _ in gate_params)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in gate_params)


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


def test_delayed_parallel_block_stream_matches_immediate_parallel_stack():
    torch.manual_seed(321)
    block0 = _block(parallel_residual=True, record_residual=False)
    block1 = _block(parallel_residual=True, record_residual=False)
    x = torch.randn(2, 5, 16)
    freqs_cos, freqs_sin = _freqs()

    expected = block1(block0(x, freqs_cos, freqs_sin), freqs_cos, freqs_sin)
    hidden1, hidden2, residual = block0.forward_parallel_delayed(
        x,
        None,
        None,
        freqs_cos,
        freqs_sin,
    )
    hidden1, hidden2, residual = block1.forward_parallel_delayed(
        hidden1,
        hidden2,
        residual,
        freqs_cos,
        freqs_sin,
    )
    actual = pg.flush_parallel_residual_stream(hidden1, hidden2, residual, output_dtype=x.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


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
    pytest.importorskip("torchao")
    h = _tiny_h()
    h.qat_bits = 4
    h.qat_group_size = 8
    h.qat_activation_bits = 8
    h.qat_start_step = 3
    model = pg.GPT(h)
    pg.prepare_torchao_qat(model, h)

    model.set_training_step(2)
    assert not model._qat_enabled
    assert model.qat_fake_quant_modules
    assert all(
        not getattr(fake_quantizer, "enabled", False)
        for module in model.qat_fake_quant_modules
        for fake_quantizer in (
            getattr(module, "activation_fake_quantizer", None),
            getattr(module, "weight_fake_quantizer", None),
        )
        if fake_quantizer is not None
    )

    model.set_training_step(3)
    assert model._qat_enabled
    assert all(
        getattr(fake_quantizer, "enabled", False)
        for module in model.qat_fake_quant_modules
        for fake_quantizer in (
            getattr(module, "activation_fake_quantizer", None),
            getattr(module, "weight_fake_quantizer", None),
        )
        if fake_quantizer is not None
    )

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
    assert params["core_block.0.attn.qkv_proj.weight"].dtype == torch.bfloat16


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


def test_delayed_parallel_residual_gpt_matches_immediate_without_record_controls():
    torch.manual_seed(987)
    h = _tiny_h()
    h.residual_mode = "parallel"
    h.parallel_residual_scope = "core"
    h.parallel_residual_record_controls = False
    immediate = pg.GPT(h)

    h_delayed = _tiny_h()
    h_delayed.residual_mode = "parallel"
    h_delayed.parallel_residual_scope = "core"
    h_delayed.parallel_residual_record_controls = False
    h_delayed.parallel_residual_impl = "delayed"
    delayed = pg.GPT(h_delayed)
    delayed.load_state_dict(immediate.state_dict(), strict=True)
    immediate.eval()
    delayed.eval()

    input_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len))
    steps = torch.tensor([0, 1])
    expected = immediate.forward_logits(input_ids, num_steps_pair=steps)
    actual = delayed.forward_logits(input_ids, num_steps_pair=steps)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_delayed_parallel_residual_tied_norm_and_fp32_stream_are_finite():
    torch.manual_seed(654)
    h = _tiny_h()
    h.residual_mode = "parallel"
    h.parallel_residual_scope = "core"
    h.parallel_residual_record_controls = False
    h.parallel_residual_impl = "delayed"
    h.parallel_residual_tied_norm = True
    h.parallel_residual_in_fp32 = True
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
    assert model.core_block[0].norm_1.weight.grad is not None
    assert torch.isfinite(model.core_block[0].norm_1.weight.grad).all()
    assert model.core_block[0].norm_2.weight.grad is None


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
