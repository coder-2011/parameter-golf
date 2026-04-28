from __future__ import annotations

import argparse

import torch

try:
    from liger_kernel.ops.softmax import LigerSoftmaxFunction
except Exception:
    LigerSoftmaxFunction = None


def _time_ms(fn, x: torch.Tensor, grad: torch.Tensor, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        x_in = x.detach().clone().requires_grad_(True)
        y = fn(x_in)
        (y * grad).sum().backward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        x_in = x.detach().clone().requires_grad_(True)
        y = fn(x_in)
        (y * grad).sum().backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=1892)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if LigerSoftmaxFunction is None:
        raise SystemExit("liger-kernel softmax is not importable")

    torch.manual_seed(123)
    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.batch, args.vocab, device="cuda", dtype=dtype)
    grad = torch.randn_like(x)

    x_pt = x.detach().clone().requires_grad_(True)
    y_pt = torch.softmax(x_pt, dim=-1)
    (y_pt * grad).sum().backward()

    x_liger = x.detach().clone().requires_grad_(True)
    y_liger = LigerSoftmaxFunction.apply(x_liger)
    (y_liger * grad).sum().backward()

    print(f"shape=({args.batch}, {args.vocab}) dtype={dtype}")
    print(f"max_abs_forward_diff={(y_pt - y_liger).abs().max().item():.6g}")
    print(f"max_abs_grad_diff={(x_pt.grad - x_liger.grad).abs().max().item():.6g}")

    pt_ms = _time_ms(lambda t: torch.softmax(t, dim=-1), x, grad, args.iters, args.warmup)
    liger_ms = _time_ms(lambda t: LigerSoftmaxFunction.apply(t), x, grad, args.iters, args.warmup)
    print(f"torch_softmax_fwd_bwd_ms={pt_ms:.4f}")
    print(f"liger_softmax_fwd_bwd_ms={liger_ms:.4f}")
    print(f"speedup={pt_ms / liger_ms:.3f}x")


if __name__ == "__main__":
    main()
