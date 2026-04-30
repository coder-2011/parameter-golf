from __future__ import annotations

import argparse
import io
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_gpt_parcae as pg


def _compress_payload(raw: bytes, compressor: str) -> bytes:
    if compressor == "store":
        return raw
    if compressor == "zlib":
        return zlib.compress(raw, level=9)
    if compressor == "brotli":
        try:
            import brotli  # type: ignore
        except Exception:
            return b""
        return brotli.compress(raw, quality=11)
    raise ValueError(f"unsupported probe compressor {compressor!r}")


def _serialize_grouped_with_compressor(obj: dict[str, object], compressor: str) -> bytes:
    quant_format = str(obj.get("__quant_format__", ""))
    tensor_sections = ("quantized", "scales", "passthrough")
    manifest: dict[str, object] = {
        "version": 1,
        "metadata": {key: value for key, value in obj.items() if key not in tensor_sections},
        "tensors": [],
        "groups": [],
    }
    group_raw: dict[str, bytearray] = {}
    for section in tensor_sections:
        tensors = obj.get(section, {})
        if not isinstance(tensors, dict):
            continue
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"expected tensor at {section}.{name}, got {type(tensor)!r}")
            raw = pg._tensor_to_raw_bytes(tensor)
            group = pg._artifact_group_for_tensor(section, name, tensor, quant_format)
            if pg._should_lane_group_tensor(tensor):
                element_size = tensor.element_size()
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, element_size)
                lanes = []
                for lane_idx in range(element_size):
                    lane_raw = arr[:, lane_idx].tobytes()
                    lane_group = f"{group}.b{lane_idx}"
                    lane_offset = len(group_raw.setdefault(lane_group, bytearray()))
                    group_raw[lane_group].extend(lane_raw)
                    lanes.append(
                        {
                            "group": lane_group,
                            "offset": lane_offset,
                            "nbytes": len(lane_raw),
                        }
                    )
                manifest["tensors"].append(
                    {
                        "section": section,
                        "name": name,
                        "shape": list(tensor.shape),
                        "dtype": pg._torch_dtype_name(tensor.dtype),
                        "lanes": lanes,
                    }
                )
                continue
            shuffle = pg._should_shuffle_tensor_bytes(tensor)
            if shuffle:
                raw = pg._byte_shuffle(raw, tensor.element_size())
            offset = len(group_raw.setdefault(group, bytearray()))
            group_raw[group].extend(raw)
            manifest["tensors"].append(
                {
                    "section": section,
                    "name": name,
                    "group": group,
                    "offset": offset,
                    "nbytes": len(raw),
                    "shape": list(tensor.shape),
                    "dtype": pg._torch_dtype_name(tensor.dtype),
                    "shuffle": int(shuffle),
                }
            )

    group_blobs = bytearray()
    for group in sorted(group_raw):
        raw = bytes(group_raw[group])
        compressed = _compress_payload(raw, compressor)
        if compressor == "brotli" and not compressed:
            return b""
        offset = len(group_blobs)
        group_blobs.extend(compressed)
        manifest["groups"].append(
            {
                "name": group,
                "compressor": compressor,
                "offset": offset,
                "nbytes": len(compressed),
                "raw_nbytes": len(raw),
            }
        )
    manifest_bytes = zlib.compress(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        level=9,
    )
    return (
        pg.GROUPED_ARTIFACT_MAGIC
        + len(manifest_bytes).to_bytes(8, "little")
        + manifest_bytes
        + bytes(group_blobs)
    )


def _torchsave_bytes(obj: object) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _state_dicts_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[name], right[name]) for name in left)


def _raw_state_artifact_obj(state: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "__quant_format__": "raw_state_grouped_v1",
        "bits": 0,
        "quantized": {},
        "q_shapes": {},
        "scales": {},
        "dtypes": {},
        "passthrough": {
            name: tensor.detach().to("cpu").contiguous()
            for name, tensor in state.items()
            if isinstance(tensor, torch.Tensor)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe grouped quant artifact compression")
    parser.add_argument("--model", type=Path, default=Path("final_model.pt"))
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--rans-int6", action="store_true")
    parser.add_argument("--raw-state", action="store_true", help="Group/compress raw checkpoint tensors without quantizing")
    args = parser.parse_args()

    state = torch.load(args.model, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"expected state_dict-like dict, got {type(state)!r}")

    quant_obj = _raw_state_artifact_obj(state) if args.raw_state else pg.quantize_state_dict_int(
        state,
        bits=args.bits,
        use_rans_int6=args.rans_int6,
    )[0]
    current_torchsave_zlib = zlib.compress(_torchsave_bytes(state if args.raw_state else quant_obj), level=9)
    grouped_zlib = _serialize_grouped_with_compressor(quant_obj, "zlib")
    grouped_brotli = _serialize_grouped_with_compressor(quant_obj, "brotli")
    grouped_best, _ = pg.serialize_quant_artifact_grouped(quant_obj)

    if args.raw_state:
        current_roundtrip = {
            name: tensor.detach().to("cpu").contiguous()
            for name, tensor in state.items()
            if isinstance(tensor, torch.Tensor)
        }
        grouped_roundtrip = pg.load_quant_artifact(grouped_best)["passthrough"]
    else:
        current_roundtrip = pg.dequantize_state_dict_int(quant_obj)
        grouped_roundtrip = pg.dequantize_state_dict_int(pg.load_quant_artifact(grouped_best))
    roundtrip_equal = int(_state_dicts_equal(current_roundtrip, grouped_roundtrip))

    print(f"current_torchsave_zlib_bytes={len(current_torchsave_zlib)}")
    print(f"grouped_zlib_bytes={len(grouped_zlib)}")
    print(f"grouped_brotli_bytes={len(grouped_brotli) if grouped_brotli else -1}")
    print(f"grouped_best_bytes={len(grouped_best)}")
    print(f"roundtrip_equal={roundtrip_equal}")


if __name__ == "__main__":
    main()
