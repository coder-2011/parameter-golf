from __future__ import annotations

import argparse
import io
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_gpt_parcae as pg


RANS_PREC = 15
RANS_TOTAL = 1 << RANS_PREC
RANS_LOWER = 1 << 23
RANS_ALPHABET = 64
RANS_MAGIC = b"R6A1"


def _build_freq_table(symbols: np.ndarray, alphabet_size: int = RANS_ALPHABET) -> list[int]:
    counts = np.bincount(symbols.astype(np.int64, copy=False), minlength=alphabet_size)
    present = counts > 0
    n_present = int(present.sum())
    if n_present == 0:
        raise ValueError("cannot build rANS table for an empty symbol stream")
    remaining = RANS_TOTAL - n_present
    if remaining < 0:
        raise ValueError(f"too many symbols for RANS_TOTAL={RANS_TOTAL}: {n_present}")

    freqs = np.zeros(alphabet_size, dtype=np.int64)
    freqs[present] = 1
    total = int(counts[present].sum())
    freqs[present] += np.floor(counts[present] / total * remaining).astype(np.int64)

    deficit = RANS_TOTAL - int(freqs.sum())
    if deficit:
        for idx in np.argsort(-counts):
            if deficit == 0:
                break
            if present[idx]:
                freqs[idx] += 1
                deficit -= 1

    if int(freqs.sum()) != RANS_TOTAL:
        raise AssertionError("rANS frequency table does not sum to RANS_TOTAL")
    return freqs.tolist()


def _rans_encode(symbols: np.ndarray, freqs: list[int]) -> bytes:
    cum = [0] * (len(freqs) + 1)
    for i, freq in enumerate(freqs):
        cum[i + 1] = cum[i] + int(freq)

    state = RANS_LOWER
    out = bytearray()
    for sym in symbols[::-1]:
        s = int(sym)
        freq = int(freqs[s])
        start = int(cum[s])
        if freq <= 0:
            raise ValueError(f"symbol {s} has zero frequency")

        max_state = (freq * (RANS_LOWER >> RANS_PREC)) << 8
        while state >= max_state:
            out.append(state & 0xFF)
            state >>= 8
        state = (state // freq) * RANS_TOTAL + start + (state % freq)

    for _ in range(4):
        out.append(state & 0xFF)
        state >>= 8
    out.reverse()
    return bytes(out)


def _rans_decode(data: bytes, freqs: list[int], count: int) -> np.ndarray:
    cum = [0] * (len(freqs) + 1)
    for i, freq in enumerate(freqs):
        cum[i + 1] = cum[i] + int(freq)
    if cum[-1] != RANS_TOTAL:
        raise ValueError(f"rANS frequency total must be {RANS_TOTAL}, got {cum[-1]}")

    sym_table = np.empty(RANS_TOTAL, dtype=np.uint8)
    for s, freq in enumerate(freqs):
        if freq:
            sym_table[cum[s] : cum[s + 1]] = s

    view = memoryview(data)
    if len(view) < 4:
        raise ValueError("rANS stream is too short to contain an initial state")
    pos = 0
    state = 0
    for _ in range(4):
        state = (state << 8) | int(view[pos])
        pos += 1

    symbols = np.empty(count, dtype=np.uint8)
    for i in range(count):
        slot = state & (RANS_TOTAL - 1)
        s = int(sym_table[slot])
        symbols[i] = s
        state = int(freqs[s]) * (state >> RANS_PREC) + slot - int(cum[s])
        while state < RANS_LOWER:
            if pos >= len(view):
                raise ValueError(f"rANS stream exhausted at decoded symbol {i}/{count}")
            state = (state << 8) | int(view[pos])
            pos += 1
    return symbols


def compress_int6_q(q: Tensor) -> bytes:
    q_np = q.detach().cpu().reshape(-1).to(torch.int16).numpy()
    symbols = (q_np + 32).astype(np.uint8, copy=False)
    if symbols.size and (int(symbols.min()) < 0 or int(symbols.max()) >= RANS_ALPHABET):
        raise ValueError("int6 tensor contains values outside [-32, 31]")
    freqs = _build_freq_table(symbols)
    encoded = _rans_encode(symbols, freqs)
    header = bytearray(RANS_MAGIC)
    header.extend(int(symbols.size).to_bytes(8, "little"))
    for freq in freqs:
        header.extend(int(freq).to_bytes(2, "little"))
    return bytes(header) + encoded


def _blob_to_uint8_tensor(blob: bytes) -> Tensor:
    return torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy()).contiguous()


def _blob_to_bytes(blob: bytes | Tensor) -> bytes:
    if isinstance(blob, Tensor):
        return blob.detach().cpu().contiguous().numpy().tobytes()
    return blob


def decompress_int6_q(blob: bytes | Tensor, shape: tuple[int, ...]) -> Tensor:
    blob = _blob_to_bytes(blob)
    if blob[:4] != RANS_MAGIC:
        raise ValueError(f"bad rANS magic {blob[:4]!r}")
    count = int.from_bytes(blob[4:12], "little")
    freqs = [int.from_bytes(blob[12 + 2 * i : 14 + 2 * i], "little") for i in range(RANS_ALPHABET)]
    symbols = _rans_decode(blob[12 + 2 * RANS_ALPHABET :], freqs, count)
    q = symbols.astype(np.int16, copy=False) - 32
    if math.prod(shape) != count:
        raise ValueError(f"decoded count {count} does not match shape {shape}")
    return torch.from_numpy(q.astype(np.int8, copy=False)).reshape(shape).contiguous()


def dequantize_rans_object(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    bits = int(obj["bits"])
    if bits != 6:
        raise ValueError(f"this probe only supports int6, got bits={bits}")
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    q_shapes = obj["q_shapes"]
    for name, blob in obj["rans_quantized"].items():
        q = decompress_int6_q(blob, tuple(q_shapes[name]))
        dtype = getattr(torch, obj["dtypes"][name])
        scale = obj["scales"][name]
        if scale.ndim > 0:
            value = q.float() * scale.float().view(q.shape[0], *([1] * (q.ndim - 1)))
        else:
            value = q.float() * float(scale.item())
        out[name] = value.to(dtype=dtype).contiguous()
    for name, tensor in obj["passthrough"].items():
        value = tensor.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            value = value.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = value
    return out


def build_rans_quant_object(state_dict: dict[str, Tensor]) -> tuple[dict[str, object], dict[str, int]]:
    bits = 6
    rans_quantized: dict[str, Tensor] = {}
    q_shapes: dict[str, tuple[int, ...]] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    stats = dict.fromkeys(
        (
            "num_tensors",
            "num_quantized",
            "num_passthrough",
            "baseline_tensor_bytes",
            "packed_payload_bytes",
            "rans_payload_bytes",
            "scale_bytes",
            "passthrough_bytes",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += pg.tensor_nbytes(t)
        if (not t.is_floating_point()) or pg.should_keep_float_tensor(name, t):
            kept = t if not t.is_floating_point() else pg.keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["num_passthrough"] += 1
            stats["passthrough_bytes"] += pg.tensor_nbytes(kept)
            continue

        q, scale = pg.quantize_float_tensor(t, bits=bits)
        packed = pg.pack_quantized_tensor(q, bits)
        blob = compress_int6_q(q)
        decoded = decompress_int6_q(blob, tuple(q.shape))
        if not torch.equal(q, decoded):
            raise AssertionError(f"rANS int6 roundtrip mismatch for {name}")

        rans_quantized[name] = _blob_to_uint8_tensor(blob)
        q_shapes[name] = tuple(q.shape)
        scales[name] = scale
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["num_quantized"] += 1
        stats["packed_payload_bytes"] += pg.tensor_nbytes(packed)
        stats["rans_payload_bytes"] += len(blob)
        stats["scale_bytes"] += pg.tensor_nbytes(scale)

    obj: dict[str, object] = {
        "__quant_format__": "int6_rans_per_tensor_v1",
        "bits": bits,
        "rans_quantized": rans_quantized,
        "q_shapes": q_shapes,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def serialized_bytes(obj: object) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def serialized_size(obj: object) -> int:
    return len(serialized_bytes(obj))


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe rANS int6 compression on a Parcae .pt state dict")
    parser.add_argument("--model", type=Path, default=Path("final_model.pt"))
    parser.add_argument("--out", type=Path, default=Path("runs/rans_probe/final_model.int6.rans.pt"))
    parser.add_argument("--write-decoded", action="store_true")
    args = parser.parse_args()

    state = torch.load(args.model, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"expected a state_dict-like dict, got {type(state)!r}")

    rans_obj, stats = build_rans_quant_object(state)
    packed_obj, _ = pg.quantize_state_dict_int(state, bits=6)
    packed_state = pg.dequantize_state_dict_int(packed_obj)
    rans_state = dequantize_rans_object(rans_obj)
    if packed_state.keys() != rans_state.keys():
        raise AssertionError("decoded state keys differ")
    mismatches = [name for name in packed_state if not torch.equal(packed_state[name], rans_state[name])]
    if mismatches:
        raise AssertionError(f"decoded state mismatch in {len(mismatches)} tensors, first={mismatches[0]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rans_obj, args.out)
    decoded_path = args.out.with_suffix(".decoded.pt")
    decoded_size = serialized_size(rans_state)
    if args.write_decoded:
        torch.save(rans_state, decoded_path)

    print(f"source_path={args.model}")
    print(f"source_file_bytes={args.model.stat().st_size}")
    print(f"quantized_tensors={stats['num_quantized']} passthrough_tensors={stats['num_passthrough']}")
    print(f"baseline_tensor_bytes={stats['baseline_tensor_bytes']}")
    print(f"current_bitpack_payload_bytes={stats['packed_payload_bytes']}")
    print(f"rans_payload_bytes={stats['rans_payload_bytes']}")
    print(f"scale_bytes={stats['scale_bytes']} passthrough_bytes={stats['passthrough_bytes']}")
    packed_serialized = serialized_bytes(packed_obj)
    rans_serialized = serialized_bytes(rans_obj)
    print(f"current_torchsave_int6_bytes={len(packed_serialized)}")
    print(f"current_torchsave_zlib_int6_bytes={len(zlib.compress(packed_serialized, level=9))}")
    print(f"rans_torchsave_int6_bytes={args.out.stat().st_size}")
    print(f"rans_torchsave_zlib_int6_bytes={len(zlib.compress(rans_serialized, level=9))}")
    print(f"decoded_torchsave_state_bytes={decoded_size}")
    print(f"decoded_equal_to_current_int6_roundtrip=1")
    if args.write_decoded:
        print(f"decoded_path={decoded_path}")


if __name__ == "__main__":
    main()
