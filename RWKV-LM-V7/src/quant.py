from __future__ import annotations

import io
import math
import zlib
from pathlib import Path

import numpy as np
import torch

QUANT_SCALE_EPS = 2.0**-24


def signed_quant_max(bits: int) -> int:
    if bits < 2 or bits > 8:
        raise ValueError(f"quant bits must be in [2, 8], got {bits}")
    return (1 << (bits - 1)) - 1


def pack_quantized_tensor(q: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        return q.to(torch.int8).contiguous()
    qmax = signed_quant_max(bits)
    if q.numel() > 0:
        q_min = int(q.min().item())
        q_max = int(q.max().item())
        if q_min < -qmax or q_max > qmax:
            raise ValueError(
                f"quantized values out of int{bits} range "
                f"[-{qmax}, {qmax}]: min={q_min} max={q_max}"
            )
    flat = (q.detach().cpu().reshape(-1).to(torch.int16).numpy() + qmax).astype(
        np.uint16, copy=False
    )
    shifts = np.arange(bits, dtype=np.uint16)
    bitstream = ((flat[:, None] >> shifts[None, :]) & 1).astype(
        np.uint8, copy=False
    ).reshape(-1)
    return torch.from_numpy(np.packbits(bitstream, bitorder="little")).contiguous()


def unpack_quantized_tensor(
    packed: torch.Tensor, shape: tuple[int, ...], bits: int
) -> torch.Tensor:
    if bits == 8:
        return packed.to(torch.int8).reshape(shape).contiguous()
    qmax = signed_quant_max(bits)
    n = math.prod(shape)
    packed_np = packed.detach().cpu().numpy().astype(np.uint8, copy=False)
    bitstream = np.unpackbits(packed_np, bitorder="little")[: n * bits].reshape(
        n, bits
    )
    shifts = np.arange(bits, dtype=np.uint16)
    vals = (bitstream.astype(np.uint16, copy=False) << shifts[None, :]).sum(axis=1)
    q = vals.astype(np.int16, copy=False) - qmax
    return torch.from_numpy(q.astype(np.int8, copy=False)).reshape(shape).contiguous()


def quantize_float_tensor(tensor: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = signed_quant_max(bits)
    t32 = tensor.detach().float().cpu()
    if t32.ndim == 2:
        scale = t32.abs().amax(dim=1).div(qmax).clamp_min(QUANT_SCALE_EPS)
        q = torch.clamp(torch.round(t32 / scale[:, None]), -qmax, qmax).to(torch.int8)
        return q.contiguous(), scale.contiguous()

    scale = t32.abs().amax().div(qmax).clamp_min(QUANT_SCALE_EPS)
    q = torch.clamp(torch.round(t32 / scale), -qmax, qmax).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def quantize_state_dict_int(state_dict: dict[str, torch.Tensor], bits: int) -> dict[str, object]:
    signed_quant_max(bits)
    quantized: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    passthrough: dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            passthrough[name] = tensor.detach().cpu().contiguous()
            continue
        q, scale = quantize_float_tensor(tensor, bits)
        quantized[name] = pack_quantized_tensor(q, bits)
        scales[name] = scale
        shapes[name] = tuple(tensor.shape)

    return {
        "__quant_format__": f"rwkv_int{bits}_per_row_2d_packed_v1",
        "bits": bits,
        "quantized": quantized,
        "scales": scales,
        "shapes": shapes,
        "passthrough": passthrough,
    }


def dequantize_state_dict_int(obj: dict[str, object]) -> dict[str, torch.Tensor]:
    bits = int(obj["bits"])
    signed_quant_max(bits)
    quantized: dict[str, torch.Tensor] = obj["quantized"]
    scales: dict[str, torch.Tensor] = obj["scales"]
    shapes: dict[str, tuple[int, ...]] = obj["shapes"]
    passthrough: dict[str, torch.Tensor] = obj.get("passthrough", {})

    state_dict = {name: tensor for name, tensor in passthrough.items()}
    for name, packed in quantized.items():
        shape = tuple(shapes[name])
        q = unpack_quantized_tensor(packed, shape, bits).float()
        scale = scales[name].float()
        if q.ndim == 2:
            value = q * scale[:, None]
        else:
            value = q * scale
        state_dict[name] = value.contiguous()
    return state_dict


def save_quantized_state_dict(
    state_dict: dict[str, torch.Tensor], path: str | Path, bits: int, compress_level: int = 9
) -> tuple[int, int]:
    quant_obj = quantize_state_dict_int(state_dict, bits)
    raw_buf = io.BytesIO()
    torch.save(quant_obj, raw_buf)
    raw = raw_buf.getvalue()
    blob = zlib.compress(raw, level=compress_level)
    Path(path).write_bytes(blob)
    return len(raw), len(blob)


def load_quantized_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    blob = Path(path).read_bytes()
    obj = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
    if "__quant_format__" not in obj or "bits" not in obj:
        raise ValueError(f"not a recognized RWKV quantized checkpoint: {path}")
    return dequantize_state_dict_int(obj)
