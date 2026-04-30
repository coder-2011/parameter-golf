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
