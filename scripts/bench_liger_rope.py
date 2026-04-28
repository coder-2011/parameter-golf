from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_gpt_parcae as pg


def _bench(name, fn, q, k, cos, sin, rope_dims, iters, warmup, backward):
    for _ in range(warmup):
        q_in = q.detach().clone().requires_grad_(True) if backward else q
        k_in = k.detach().clone().requires_grad_(True) if backward else k
        q_out, k_out = fn(q_in, k_in, cos, sin, rope_dims)
        if backward:
            (q_out.float().square().mean() + k_out.float().square().mean()).backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        q_in = q.detach().clone().requires_grad_(True) if backward else q
        k_in = k.detach().clone().requires_grad_(True) if backward else k
        q_out, k_out = fn(q_in, k_in, cos, sin, rope_dims)
        if backward:
            (q_out.float().square().mean() + k_out.float().square().mean()).backward()
    end.record()
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"{name}: {start.elapsed_time(end) / iters:.4f} ms/iter peak_alloc:{peak_mib:.1f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--rope-dims", type=int, default=64)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Liger RoPE benchmark")
    if pg.LigerRopeFunction is None:
        raise SystemExit("liger-kernel is not importable; install it before running this benchmark")
    if args.rope_dims <= 0 or args.rope_dims > args.head_dim or args.rope_dims % 2 != 0:
        raise SystemExit("--rope-dims must be a positive even value <= --head-dim")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    q = torch.randn(
        args.batch_size,
        args.seq_len,
        args.q_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        args.batch_size,
        args.seq_len,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    cos, sin = pg.precompute_freqs_cos_sin(args.rope_dims, args.seq_len)
    cos = cos.to(device=device)
    sin = sin.to(device=device)

    print(
        f"shape B={args.batch_size} T={args.seq_len} Hq={args.q_heads} "
        f"Hkv={args.kv_heads} D={args.head_dim} rope_dims={args.rope_dims} "
        f"dtype={dtype} backward={args.backward}"
    )
    _bench(
        "pytorch_adjacent_pair_rope",
        pg.apply_rotary_emb_complex_like,
        q,
        k,
        cos,
        sin,
        args.rope_dims,
        args.iters,
        args.warmup,
        args.backward,
    )
    _bench(
        "liger_split_half_rope",
        pg.apply_liger_rotary_emb,
        q,
        k,
        cos,
        sin,
        args.rope_dims,
        args.iters,
        args.warmup,
        args.backward,
    )


if __name__ == "__main__":
    main()
