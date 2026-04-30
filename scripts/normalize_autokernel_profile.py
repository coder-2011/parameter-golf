#!/usr/bin/env python3
"""Normalize AutoKernel profiler shape strings for extract.py.

AutoKernel stores PyTorch profiler input shapes as Python-list strings, while
its extractor expects compact `key=value` strings. This script converts the
common Parcae ops and fills unshaped CUDA rows from the first parseable same-op
row in the profile.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


def _prod(xs: list[int]) -> int:
    return math.prod(xs) if xs else 1


def _shape_list(raw: str) -> list[Any] | None:
    if not raw:
        return None
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, list) else None


def _is_shape(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(dim, int) for dim in value)


def _first_shape(shapes: list[Any]) -> list[int] | None:
    for shape in shapes:
        if _is_shape(shape) and shape:
            return shape
    return None


def _format(op_type: str, raw: str) -> str | None:
    shapes = _shape_list(raw)
    if shapes is None:
        return None

    if op_type == "matmul" and len(shapes) >= 2 and _is_shape(shapes[0]) and _is_shape(shapes[1]):
        a, b = shapes[0], shapes[1]
        if len(a) == 2 and len(b) == 2:
            return f"M={a[0]}, N={b[1]}, K={a[1]}"

    if op_type in {"layernorm", "rmsnorm", "softmax", "reduce"}:
        x = _first_shape(shapes)
        if x:
            rows = _prod(x[:-1])
            cols = x[-1]
            if op_type == "layernorm":
                return f"batch={rows}, dim={cols}"
            if op_type == "softmax":
                return f"rows={rows}, cols={cols}"
            return f"M={rows}, N={cols}"

    if op_type == "flash_attention":
        q = _first_shape(shapes)
        if q and len(q) == 4:
            # Parcae/SDPA profiler rows use B,T,H,D layout.
            batch, seq_len, heads, head_dim = q
            return f"batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}"

    if op_type == "cross_entropy":
        logits = _first_shape(shapes)
        if logits and len(logits) >= 2:
            return f"batch={_prod(logits[:-1])}, vocab={logits[-1]}"

    if op_type == "rotary_embedding":
        x = _first_shape(shapes)
        if x and len(x) == 4:
            batch, seq_len, heads, head_dim = x
            return f"batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}"

    if op_type == "fused_mlp":
        x = _first_shape(shapes)
        weights = [shape for shape in shapes if _is_shape(shape) and len(shape) == 2]
        if x and weights:
            hidden = max(shape[0] for shape in weights)
            return f"batch={_prod(x[:-1])}, dim={x[-1]}, hidden={hidden}"

    return None


def normalize_profile(profile: dict[str, Any]) -> int:
    top_kernels = profile.get("top_kernels", [])
    fallback_by_op: dict[str, str] = {}
    changed = 0

    for kernel in top_kernels:
        op_type = str(kernel.get("op_type", ""))
        shape_info = str(kernel.get("shape_info", ""))
        normalized = _format(op_type, shape_info)
        if normalized is not None:
            fallback_by_op.setdefault(op_type, normalized)

    # Parcae uses RMSNorm; if CUDA layer_norm rows do not expose shapes, this
    # gives AutoKernel's layernorm starter the same row shape instead of a huge
    # unrelated default.
    if "layernorm" not in fallback_by_op and "rmsnorm" in fallback_by_op:
        fallback_by_op["layernorm"] = fallback_by_op["rmsnorm"].replace("M=", "batch=").replace("N=", "dim=")

    for kernel in top_kernels:
        op_type = str(kernel.get("op_type", ""))
        original = str(kernel.get("shape_info", ""))
        normalized = _format(op_type, original) or fallback_by_op.get(op_type)
        if normalized is None or normalized == original:
            continue
        kernel["raw_shape_info"] = original
        kernel["shape_info"] = normalized
        changed += 1

    profile["shape_info_normalized"] = True
    profile["shape_info_normalized_count"] = changed
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    changed = normalize_profile(profile)
    output = args.output or args.profile
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"normalized {changed} AutoKernel shape rows in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
