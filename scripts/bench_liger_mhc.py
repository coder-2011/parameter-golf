#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _maybe_add_liger_src() -> None:
    liger_src = os.environ.get("LIGER_KERNEL_SRC", "")
    if liger_src:
        sys.path.insert(0, liger_src)


_maybe_add_liger_src()


class LinearResidual(nn.Module):
    def __init__(self, c: int, dtype: torch.dtype):
        super().__init__()
        self.layer = nn.Linear(c, c, bias=False, device="cuda", dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x)


class MLPResidual(nn.Module):
    def __init__(self, c: int, mult: int, dtype: torch.dtype):
        super().__init__()
        hidden = c * mult
        self.up = nn.Linear(c, hidden, bias=False, device="cuda", dtype=dtype)
        self.down = nn.Linear(hidden, c, bias=False, device="cuda", dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(F.gelu(self.up(x)))


class MLPOnly(nn.Module):
    def __init__(self, c: int, mult: int, dtype: torch.dtype):
        super().__init__()
        hidden = c * mult
        self.up = nn.Linear(c, hidden, bias=False, device="cuda", dtype=dtype)
        self.down = nn.Linear(hidden, c, bias=False, device="cuda", dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


@dataclass
class BenchResult:
    ms: float
    peak_mib: int


def _zero_grads(module: nn.Module, x: torch.Tensor) -> None:
    for param in module.parameters():
        param.grad = None
    x.grad = None


def bench_step(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = module(x)
    loss = out.float().square().mean()
    loss.backward()
    return loss.detach()


def bench_module(module: nn.Module, x: torch.Tensor, warmup: int, iters: int) -> BenchResult:
    for _ in range(warmup):
        _zero_grads(module, x)
        bench_step(module, x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _zero_grads(module, x)
        bench_step(module, x)
    end.record()
    torch.cuda.synchronize()
    return BenchResult(
        ms=start.elapsed_time(end) / iters,
        peak_mib=torch.cuda.max_memory_allocated() // 1024 // 1024,
    )


def parse_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LigerMHC overhead on simple residual layers.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--channels", type=int, default=384)
    parser.add_argument("--hc", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--mlp-mult", type=int, default=4)
    parser.add_argument("--tmax", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LigerMHC benchmarking")

    try:
        from liger_kernel.transformers.mhc import LigerMHC
    except Exception as exc:
        raise RuntimeError(
            "Could not import LigerMHC. With the current autoresearch venv, set "
            "LIGER_KERNEL_SRC=/tmp/Liger-Kernel/src after cloning upstream Liger-Kernel."
        ) from exc

    dtype = parse_dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    device_name = torch.cuda.get_device_name()
    print(
        f"device={device_name} torch={torch.__version__} dtype={args.dtype} "
        f"shape=[{args.batch},{args.seq_len},{args.channels}] warmup={args.warmup} iters={args.iters}"
    )

    baseline_specs = [
        ("linear", LinearResidual(args.channels, dtype), nn.Linear(args.channels, args.channels, bias=False, device="cuda", dtype=dtype)),
        ("mlp", MLPResidual(args.channels, args.mlp_mult, dtype), MLPOnly(args.channels, args.mlp_mult, dtype)),
    ]
    for name, baseline, mhc_inner in baseline_specs:
        baseline = baseline.cuda()
        x = torch.randn(args.batch, args.seq_len, args.channels, device="cuda", dtype=dtype, requires_grad=True)
        base_result = bench_module(baseline, x, args.warmup, args.iters)
        print(f"{name}:baseline_ms={base_result.ms:.3f} peak_mib={base_result.peak_mib}")
        for hc in args.hc:
            inner = copy.deepcopy(mhc_inner).cuda()
            mhc = LigerMHC(inner, hc=hc, c=args.channels, tmax=args.tmax, phi_dtype=dtype).cuda()
            x_mhc = torch.randn(
                args.batch,
                args.seq_len,
                hc,
                args.channels,
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )
            result = bench_module(mhc, x_mhc, args.warmup, args.iters)
            print(
                f"{name}:hc={hc} mhc_ms={result.ms:.3f} slowdown={result.ms / base_result.ms:.2f}x "
                f"peak_mib={result.peak_mib} mem_ratio={result.peak_mib / max(base_result.peak_mib, 1):.2f}x"
            )


if __name__ == "__main__":
    main()
