from types import SimpleNamespace
import io
import zlib

import torch

import train_gpt_parcae as pg


def _args(**overrides):
    base = dict(
        quant_bits=8,
        mixed_quant_bits=True,
        quant_embed_bits=7,
        quant_attn_bits=6,
        quant_mlp_bits=5,
        quant_low_bits=4,
        quant_control_bits=0,
        quant_low_bit_patterns=("project_out",),
        quant_keep_float_patterns=(),
        rans_int6=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _assert_sparse_gate_int8_quantized(obj, name, original):
    assert name in obj["quantized"]
    assert name in obj["scales"]
    assert name in obj["dtypes"]
    assert name not in obj["passthrough"]
    assert obj["quantized"][name].dtype == torch.int8
    assert obj["quantized"][name].shape == original.shape
    assert obj["scales"][name].dtype == torch.float16
    assert obj["scales"][name].shape == (original.shape[0],)
    assert obj["dtypes"][name] == str(original.dtype).removeprefix("torch.")

    restored = pg.dequantize_state_dict_int(obj)[name]
    row_scale = obj["scales"][name].float().view(-1, 1)
    assert restored.dtype == original.dtype
    assert restored.shape == original.shape
    assert torch.all((restored.float() - original.float()).abs() <= row_scale + 1e-6)


def test_mixed_quant_export_records_per_tensor_bits_and_roundtrips_shapes():
    torch.manual_seed(123)
    state = {
        "tok_emb.weight": torch.randn(33000, 3, dtype=torch.float32),
        "prelude.0.attn.c_qkv.weight": torch.randn(300, 300, dtype=torch.float32),
        "prelude.0.mlp.fc.weight": torch.randn(300, 300, dtype=torch.float32),
        "project_out.weight": torch.randn(300, 300, dtype=torch.float32),
        "misc.weight": torch.randn(300, 300, dtype=torch.float32),
        "core_block.0.attn.q_gain": torch.randn(8, dtype=torch.float32),
    }

    obj, stats = pg.quantize_state_dict_mixed_int(state, _args())
    restored = pg.dequantize_state_dict_int(obj)

    assert obj["__quant_format__"] == "mixed_int_clean_per_row_packed_v1"
    assert obj["tensor_bits"] == {
        "tok_emb.weight": 7,
        "prelude.0.attn.c_qkv.weight": 6,
        "prelude.0.mlp.fc.weight": 5,
        "project_out.weight": 4,
        "misc.weight": 8,
    }
    assert obj["passthrough"]["core_block.0.attn.q_gain"].dtype == torch.float16
    assert stats["num_float_tensors"] == len(state) - 1
    assert restored.keys() == state.keys()
    for name, tensor in state.items():
        assert restored[name].shape == tensor.shape
        assert restored[name].dtype == tensor.dtype


def test_sparse_attn_gate_uses_dedicated_int8_even_when_controls_passthrough():
    torch.manual_seed(123)
    gate_name = "core_block.0.attn.attn_gate_w"
    state = {
        gate_name: torch.randn(8, 12, dtype=torch.float32),
        "core_block.0.attn.q_gain": torch.randn(8, dtype=torch.float32),
    }

    obj, stats = pg.quantize_state_dict_mixed_int(state, _args(quant_control_bits=0))

    assert obj["tensor_bits"][gate_name] == 8
    assert obj["passthrough"]["core_block.0.attn.q_gain"].dtype == torch.float16
    assert stats["num_float_tensors"] == 1
    _assert_sparse_gate_int8_quantized(obj, gate_name, state[gate_name])


def test_sparse_attn_gate_int8_survives_uniform_int6_rans_format():
    torch.manual_seed(123)
    gate_name = "coda.0.attn.attn_gate_w"
    state = {gate_name: torch.randn(8, 12, dtype=torch.float32)}

    obj, stats = pg.quantize_state_dict_int(state, bits=6, use_rans_int6=True)

    assert obj["__quant_format__"] == "int6_rans_per_row_v1"
    assert gate_name not in obj["q_shapes"]
    assert stats["num_float_tensors"] == 1
    _assert_sparse_gate_int8_quantized(obj, gate_name, state[gate_name])


def test_gptq_export_uses_dedicated_sparse_gate_method_before_control_passthrough():
    torch.manual_seed(123)
    gate_name = "prelude.0.attn.attn_gate_w"
    state = {
        gate_name: torch.randn(8, 12, dtype=torch.float32),
        "prelude.0.attn.q_gain": torch.randn(8, dtype=torch.float32),
    }

    obj, stats = pg.quantize_state_dict_gptq_int(state, hessians={}, args=_args(quant_control_bits=0))

    assert obj["methods"][gate_name] == "sparse_attn_gate_int8_row"
    assert obj["tensor_bits"][gate_name] == 8
    assert obj["methods"]["prelude.0.attn.q_gain"] == "passthrough_control_fp16"
    assert stats["num_float_tensors"] == 1
    _assert_sparse_gate_int8_quantized(obj, gate_name, state[gate_name])


def test_mixed_quant_rejects_rans_int6():
    state = {"misc.weight": torch.randn(300, 300)}
    try:
        pg.quantize_state_dict_mixed_int(state, _args(rans_int6=True))
    except ValueError as exc:
        assert "RANS_INT6" in str(exc)
    else:
        raise AssertionError("expected mixed quant export to reject RANS_INT6")


def test_grouped_artifact_roundtrips_quant_object_and_legacy_zlib():
    torch.manual_seed(123)
    state = {
        "tok_emb.weight": torch.randn(33000, 3, dtype=torch.float32),
        "prelude.0.attn.c_qkv.weight": torch.randn(300, 300, dtype=torch.float32),
        "prelude.0.attn.attn_gate_w": torch.randn(8, 12, dtype=torch.float32),
        "prelude.0.mlp.fc.weight": torch.randn(300, 300, dtype=torch.float32),
        "core_block.0.attn.q_gain": torch.randn(8, dtype=torch.float32),
    }
    obj, _ = pg.quantize_state_dict_mixed_int(state, _args())

    grouped_blob, group_stats = pg.serialize_quant_artifact_grouped(obj)
    grouped_state = pg.load_quant_artifact(grouped_blob)
    grouped_restored = pg.dequantize_state_dict_int(grouped_state)

    legacy_buf = io.BytesIO()
    torch.save(obj, legacy_buf)
    legacy_state = pg.load_quant_artifact(zlib.compress(legacy_buf.getvalue(), level=9))
    legacy_restored = pg.dequantize_state_dict_int(legacy_state)

    assert grouped_blob.startswith(pg.GROUPED_ARTIFACT_MAGIC)
    assert group_stats
    assert grouped_state["tensor_bits"] == obj["tensor_bits"]
    _assert_sparse_gate_int8_quantized(grouped_state, "prelude.0.attn.attn_gate_w", state["prelude.0.attn.attn_gate_w"])
    assert grouped_restored.keys() == legacy_restored.keys()
    for name in grouped_restored:
        assert torch.equal(grouped_restored[name], legacy_restored[name])


def test_grouped_artifact_roundtrips_lane_grouped_passthrough_tensors():
    obj = {
        "__quant_format__": "raw_state_grouped_v1",
        "bits": 0,
        "quantized": {},
        "q_shapes": {},
        "scales": {},
        "dtypes": {},
        "passthrough": {
            "fp32.weight": torch.randn(17, 19, dtype=torch.float32),
            "bf16.weight": torch.randn(13, 7, dtype=torch.bfloat16),
        },
    }

    grouped_blob, group_stats = pg.serialize_quant_artifact_grouped(obj)
    restored = pg.load_quant_artifact(grouped_blob)["passthrough"]

    assert any(".b0" in group for group in group_stats)
    for name, tensor in obj["passthrough"].items():
        assert torch.equal(restored[name], tensor)
