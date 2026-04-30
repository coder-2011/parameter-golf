#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only on non-Triton envs.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _leaky_relu_sq_matmul_kernel(
        A,
        B,
        C,
        AUX,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        FORWARD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_in_group = pid % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            a = tl.load(
                A + offs_m[:, None] * stride_am + k_idxs[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                B + offs_n[:, None] * stride_bn + k_idxs[None, :] * stride_bk,
                mask=(offs_n[:, None] < N) & (k_idxs[None, :] < K),
                other=0.0,
            )
            acc += tl.dot(a, tl.trans(b))

        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        aux_ptrs = AUX + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        if FORWARD:
            pre = acc.to(tl.bfloat16)
            post = tl.where(pre > 0.0, pre, 0.5 * pre)
            post = post * post
            tl.store(c_ptrs, pre, mask=mask)
            tl.store(aux_ptrs, post, mask=mask)
        else:
            pre = tl.load(aux_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad = acc * tl.where(pre > 0.0, 2.0 * pre, 0.5 * pre)
            tl.store(c_ptrs, grad, mask=mask)


def _torch_leaky_relu_sq_matmul(a: torch.Tensor, b: torch.Tensor, aux: torch.Tensor | None = None):
    c = F.linear(a, b)
    if aux is None:
        post = F.leaky_relu(c, negative_slope=0.5).square()
        return c, post
    return c * torch.where(aux > 0, 2.0 * aux, 0.5 * aux)


def _fused_leaky_relu_sq(a: torch.Tensor, b: torch.Tensor, aux: torch.Tensor | None = None):
    if triton is None or not a.is_cuda:
        return _torch_leaky_relu_sq_matmul(a, b, aux)
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("fused kernel expects bf16 inputs")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("fused kernel expects contiguous inputs")
    m, k = a.shape
    n, k2 = b.shape
    if k != k2:
        raise ValueError(f"shape mismatch: {a.shape=} {b.shape=}")
    if m < 4096:
        return _torch_leaky_relu_sq_matmul(a, b, aux)
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    forward = aux is None
    if forward:
        aux = torch.empty_like(c)
    elif not aux.is_contiguous():
        aux = aux.contiguous()

    block_m, block_n, block_k, group_m = 64, 128, 64, 8
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _leaky_relu_sq_matmul_kernel[grid](
        a,
        b,
        c,
        aux,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        FORWARD=forward,
        num_warps=4,
        num_stages=4 if forward else 3,
    )
    return (c, aux) if forward else c


class FusedLeakyReLUSqMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(-1, x.shape[-1]).contiguous()
        pre, post = _fused_leaky_relu_sq(x_flat, up_w.contiguous())
        out = F.linear(post, down_w)
        ctx.save_for_backward(x_flat, up_w, down_w, pre, post)
        return out.reshape(*x.shape[:-1], down_w.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_flat, up_w, down_w, pre, post = ctx.saved_tensors
        go = grad_output.reshape(-1, grad_output.shape[-1]).to(dtype=post.dtype).contiguous()
        d_down_w = go.T @ post
        dpre = _fused_leaky_relu_sq(go, down_w.T.contiguous(), aux=pre)
        d_up_w = dpre.T @ x_flat
        dx = dpre @ up_w
        return dx.reshape_as(grad_output), d_up_w, d_down_w


def naive_mlp(x: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    return F.linear(F.leaky_relu(F.linear(x, up_w), negative_slope=0.5).square(), down_w)


def fused_mlp(x: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    if triton is None or not x.is_cuda or x.dtype != torch.bfloat16 or x.reshape(-1, x.shape[-1]).shape[0] < 4096:
        return naive_mlp(x, up_w, down_w)
    return FusedLeakyReLUSqMLP.apply(x, up_w, down_w)


def bench(fn, x, up_w, down_w, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        loss = fn(x, up_w, down_w).float().square().mean()
        loss.backward()
        for t in (x, up_w, down_w):
            t.grad = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        loss = fn(x, up_w, down_w).float().square().mean()
        loss.backward()
        for t in (x, up_w, down_w):
            t.grad = None
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Triton benchmark")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    x = torch.randn(args.batch, args.seq, args.dim, device=device, dtype=dtype, requires_grad=True)
    up_w = torch.randn(args.hidden, args.dim, device=device, dtype=dtype, requires_grad=True) / args.dim**0.5
    down_w = torch.randn(args.dim, args.hidden, device=device, dtype=dtype, requires_grad=True) / args.hidden**0.5
    up_w = up_w.detach().requires_grad_()
    down_w = down_w.detach().requires_grad_()

    x_ref = x.detach().clone().requires_grad_()
    up_ref = up_w.detach().clone().requires_grad_()
    down_ref = down_w.detach().clone().requires_grad_()
    y_ref = naive_mlp(x_ref, up_ref, down_ref)
    y = fused_mlp(x, up_w, down_w)
    (y_ref.float().square().mean()).backward()
    (y.float().square().mean()).backward()
    torch.cuda.synchronize()

    print(f"shape batch={args.batch} seq={args.seq} dim={args.dim} hidden={args.hidden}")
    print(f"triton_available={triton is not None}")
    print(f"max_abs_out={float((y.detach().float() - y_ref.detach().float()).abs().max()):.6g}")
    print(f"max_abs_dx={float((x.grad.float() - x_ref.grad.float()).abs().max()):.6g}")
    print(f"max_abs_d_up={float((up_w.grad.float() - up_ref.grad.float()).abs().max()):.6g}")
    print(f"max_abs_d_down={float((down_w.grad.float() - down_ref.grad.float()).abs().max()):.6g}")

    for t in (x, up_w, down_w, x_ref, up_ref, down_ref):
        t.grad = None
    naive_ms = bench(naive_mlp, x_ref, up_ref, down_ref, args.iters, args.warmup)
    fused_ms = bench(fused_mlp, x, up_w, down_w, args.iters, args.warmup)
    print(f"naive_fwd_bwd_ms={naive_ms:.3f}")
    print(f"fused_fwd_bwd_ms={fused_ms:.3f}")
    print(f"speedup={naive_ms / fused_ms:.3f}x")


if __name__ == "__main__":
    main()
