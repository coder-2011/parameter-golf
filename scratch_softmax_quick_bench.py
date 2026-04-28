import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _softmax_forward(input_ptr, output_ptr, n_rows: tl.constexpr, n_cols: tl.constexpr, block_size: tl.constexpr):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    vals = tl.load(input_ptr + row_id * n_cols + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    vals = vals - tl.max(vals, axis=0)
    numer = tl.exp(vals)
    denom = tl.sum(numer, axis=0)
    out = numer / denom
    tl.store(output_ptr + row_id * n_cols + offsets, out, mask=mask)


def triton_softmax(x):
    original_shape = x.shape
    x_2d = x.contiguous().view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape
    out = torch.empty_like(x_2d)
    block_size = max(triton.next_power_of_2(n_cols), 64)
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 8192:
        num_warps = 16
    _softmax_forward[(n_rows,)](x_2d, out, n_rows, n_cols, block_size, num_warps=num_warps)
    return out.view(original_shape)


def naive_softmax(x):
    shifted = x - x.max(dim=-1, keepdim=True).values
    numer = torch.exp(shifted)
    return numer / numer.sum(dim=-1, keepdim=True)


def gbps(ms, x):
    return 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)


def main():
    torch.manual_seed(1337)
    device = "cuda"
    dtype = torch.float16
    shapes = [
        (4096, 1024),
        (4096, 1892),
        (4096, 8192),
        (512, 32000),
    ]
    providers = [
        ("triton_fused", triton_softmax),
        ("torch_softmax", lambda x: F.softmax(x, dim=-1)),
        ("naive", naive_softmax),
    ]

    print(
        f"device={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"triton={triton.__version__} dtype=float16"
    )
    print("M,N,provider,ms,GB/s,max_abs_diff_vs_torch")
    for m, n in shapes:
        x = torch.randn((m, n), dtype=dtype, device=device) * 2.0
        ref = F.softmax(x, dim=-1)
        torch.cuda.synchronize()
        for name, fn in providers:
            y = fn(x)
            torch.cuda.synchronize()
            ms = float(triton.testing.do_bench(lambda: fn(x), warmup=10, rep=50))
            y = fn(x)
            torch.cuda.synchronize()
            diff = (y.float() - ref.float()).abs().max().item()
            print(f"{m},{n},{name},{ms:.6f},{gbps(ms, x):.2f},{diff:.8g}")
        del x, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
