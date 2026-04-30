#!/usr/bin/env python3
"""Benchmark Parcae's fused QKV postprocess kernel against the Python path."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_gpt_parcae as pg


def _events_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def _reference(
    qkv: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    rope_dims: int,
    qk_norm: bool,
    rope_impl: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = qkv[:, :, :n_head]
    k = qkv[:, :, n_head : n_head + n_kv_head]
    v = qkv[:, :, n_head + n_kv_head :]
    if rope_impl == "liger":
        q, k = pg.apply_liger_rotary_emb(q, k, freqs_cos, freqs_sin, rope_dims)
    elif rope_impl == "tridao":
        rotated = pg.apply_tridao_packed_qkv_rotary_emb(qkv, freqs_cos, freqs_sin, rope_dims, n_head)
        q = rotated[:, :, :n_head]
        k = rotated[:, :, n_head : n_head + n_kv_head]
        v = rotated[:, :, n_head + n_kv_head :]
    else:
        q, k = pg.apply_rotary_emb_complex_like(q, k, freqs_cos, freqs_sin, rope_dims)
    if qk_norm:
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
    return q.contiguous(), k.contiguous(), v.contiguous()


def _backward_step(
    fn,
    qkv: torch.Tensor,
    grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    if qkv.grad is not None:
        qkv.grad = None
    q, k, v = fn(qkv)
    gq, gk, gv = grads
    loss = (q.float() * gq.float()).sum()
    loss = loss + (k.float() * gk.float()).sum()
    loss = loss + (v.float() * gv.float()).sum()
    loss.backward()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-kv-head", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=48)
    parser.add_argument("--rope-dims", type=int, default=32)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--reference-rope", choices=("torch", "liger", "tridao"), default="torch")
    parser.add_argument("--qk-norm", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available() or pg.triton is None:
        raise SystemExit("CUDA and Triton are required")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.manual_seed(123)
    device = torch.device("cuda")
    total_heads = args.n_head + 2 * args.n_kv_head
    qkv = torch.randn(
        args.batch,
        args.seq_len,
        total_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    freqs_cos, freqs_sin = [
        t.to(device=device) for t in pg.precompute_freqs_cos_sin(args.rope_dims, args.seq_len, 10000.0)
    ]

    def ref_fn(x: torch.Tensor):
        return _reference(
            x,
            freqs_cos,
            freqs_sin,
            args.n_head,
            args.n_kv_head,
            args.head_dim,
            args.rope_dims,
            bool(args.qk_norm),
            args.reference_rope,
        )

    def fused_fn(x: torch.Tensor):
        return pg.fused_qkv_postprocess(
            x,
            freqs_cos,
            freqs_sin,
            args.n_head,
            args.n_kv_head,
            args.head_dim,
            args.rope_dims,
            bool(args.qk_norm),
        )

    q_ref, k_ref, v_ref = ref_fn(qkv)
    q_fused, k_fused, v_fused = fused_fn(qkv)
    torch.cuda.synchronize()
    qk_tol = 2e-2 if args.qk_norm else 0.0
    print(f"q_allclose={torch.allclose(q_ref, q_fused, atol=qk_tol, rtol=qk_tol)} max={float((q_ref.float() - q_fused.float()).abs().max()):.4g}")
    print(f"k_allclose={torch.allclose(k_ref, k_fused, atol=qk_tol, rtol=qk_tol)} max={float((k_ref.float() - k_fused.float()).abs().max()):.4g}")
    print(f"v_allclose={torch.equal(v_ref, v_fused)}")

    ref_fwd_ms = _events_ms(lambda: ref_fn(qkv), args.warmup, args.iters)
    fused_fwd_ms = _events_ms(lambda: fused_fn(qkv), args.warmup, args.iters)

    qkv_ref = qkv.detach().clone().requires_grad_(True)
    qkv_fused = qkv.detach().clone().requires_grad_(True)
    grads = tuple(torch.randn_like(t) for t in ref_fn(qkv))
    ref_bwd_ms = _events_ms(lambda: _backward_step(ref_fn, qkv_ref, grads), args.warmup, args.iters)
    fused_bwd_ms = _events_ms(lambda: _backward_step(fused_fn, qkv_fused, grads), args.warmup, args.iters)

    print(f"shape=B{args.batch} T{args.seq_len} Hq{args.n_head} Hkv{args.n_kv_head} D{args.head_dim} rope{args.rope_dims}")
    print(f"forward_ms reference={ref_fwd_ms:.4f} fused={fused_fwd_ms:.4f} speedup={ref_fwd_ms / fused_fwd_ms:.3f}x")
    print(f"fwd_bwd_ms reference={ref_bwd_ms:.4f} fused={fused_bwd_ms:.4f} speedup={ref_bwd_ms / fused_bwd_ms:.3f}x")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    torch.cuda.synchronize()
    print(f"elapsed_s={time.perf_counter() - start:.2f}")
