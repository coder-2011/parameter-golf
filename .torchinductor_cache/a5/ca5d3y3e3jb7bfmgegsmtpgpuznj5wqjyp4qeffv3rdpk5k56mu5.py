# AOT ID: ['1_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# Original path: /workspace/parameter-golf/train_gpt_parcae.py:4795
_leaky_relu_sq_matmul_kernel_0 = async_compile.triton('_leaky_relu_sq_matmul_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 4, 'num_stages': 4}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_leaky_relu_sq_matmul_kernel_0', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'A': '*bf16', 'B': '*bf16', 'C': '*bf16', 'AUX': '*bf16', 'M': 'constexpr', 'N': 'constexpr', 'K': 'constexpr', 'stride_am': 'constexpr', 'stride_ak': 'constexpr', 'stride_bn': 'constexpr', 'stride_bk': 'constexpr', 'stride_cm': 'constexpr', 'stride_cn': 'constexpr', 'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr', 'BLOCK_K': 'constexpr', 'GROUP_M': 'constexpr', 'FORWARD': 'constexpr', 'NEGATIVE_SLOPE': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'M': 131072, 'N': 1536, 'K': 384, 'stride_am': 384, 'stride_ak': 1, 'stride_bn': 384, 'stride_bk': 1, 'stride_cm': 1536, 'stride_cn': 1, 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'FORWARD': True, 'NEGATIVE_SLOPE': 0.5}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
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
    NEGATIVE_SLOPE: tl.constexpr,
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
        post = tl.where(pre > 0.0, pre, NEGATIVE_SLOPE * pre)
        post = post * post
        tl.store(c_ptrs, pre, mask=mask)
        tl.store(aux_ptrs, post, mask=mask)
    else:
        pre = tl.load(aux_ptrs, mask=mask, other=0.0).to(tl.float32)
        grad = acc * tl.where(pre > 0.0, 2.0 * pre, 2.0 * NEGATIVE_SLOPE * NEGATIVE_SLOPE * pre)
        tl.store(c_ptrs, grad, mask=mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/u6/cu6j4koyzs5cfbsxy23m4y2tfaqvuuf2xuyrytn2c6oi6t4jqeh3.py
# Topologically Sorted Source Nodes: [outs, add], Original ATen: [aten.view, aten.add]
# Source node to ATen node mapping:
#   add => add
#   outs => view_1
# Graph fragment:
#   %mm : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm]
#   %primals_4 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=primals_4]
#   %view_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 384]), kwargs = {})
#   %add : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_4), kwargs = {})
#   return %add
triton_poi_fused_add_view_0 = async_compile.triton('triton_poi_fused_add_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_view_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 402653184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_view_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_4 = args
        args.clear()
        assert_size_stride(primals_1, (384, 1536), (1536, 1))
        assert_size_stride(primals_2, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(primals_3, (1536, 384), (384, 1))
        assert_size_stride(primals_4, (128, 1024, 384), (393216, 384, 1))
        with torch.cuda._DeviceGuard(2):
            torch.cuda.set_device(2)
            buf0 = empty_strided_cuda((131072, 1536), (1536, 1), torch.bfloat16)
            buf1 = empty_strided_cuda((131072, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [reshape, triton_kernel_wrapper_mutation], Original ATen: [aten.view]
            stream2 = get_raw_stream(2)
            _leaky_relu_sq_matmul_kernel_0.run(reinterpret_tensor(primals_2, (131072, 384), (384, 1), 0), primals_3, buf0, buf1, 24576, 1, 1, stream=stream2)
            buf4 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getattr_2, out], Original ATen: [aten.permute, aten.mm]
            extern_kernels.mm(buf1, reinterpret_tensor(primals_1, (1536, 384), (1, 1536), 0), out=buf4)
            buf5 = reinterpret_tensor(buf4, (128, 1024, 384), (393216, 384, 1), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [outs, add], Original ATen: [aten.view, aten.add]
            stream2 = get_raw_stream(2)
            triton_poi_fused_add_view_0.run(buf5, primals_4, 50331648, stream=stream2)
            del primals_4
        return (buf5, primals_1, primals_3, reinterpret_tensor(primals_2, (131072, 384), (384, 1), 0), buf0, buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((384, 1536), (1536, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_2 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_3 = rand_strided((1536, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_4 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    return [primals_1, primals_2, primals_3, primals_4]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
