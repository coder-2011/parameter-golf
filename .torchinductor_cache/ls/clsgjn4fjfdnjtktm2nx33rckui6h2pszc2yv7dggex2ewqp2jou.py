# AOT ID: ['17_inference']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/av/cavhshkf6x5pwy3wg3y74gtdf3b6fj5pvdynaofxtn4dx2zpnayc.py
# Topologically Sorted Source Nodes: [dt, neg, A, mul, decay, mul_1, matmul, mul_2, x], Original ATen: [aten.softplus, aten.neg, aten.exp, aten.mul, aten._unsafe_view, aten.add]
# Source node to ATen node mapping:
#   A => exp_1
#   decay => exp_2
#   dt => exp, gt, log1p, where
#   matmul => view_1
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   neg => neg
#   x => add
# Graph fragment:
#   %arg2_1 : Tensor "f32[512, 1000, 384][384000, 384, 1]cuda:5" = PlaceHolder[target=arg2_1]
#   %arg0_1 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=arg1_1]
#   %mm : Tensor "bf16[512000, 384][384, 1]cuda:5" = PlaceHolder[target=mm]
#   %gt : Tensor "b8[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%arg0_1, 20), kwargs = {})
#   %exp : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%arg0_1,), kwargs = {})
#   %log1p : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %where : Tensor "f32[384][1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %arg0_1, %log1p), kwargs = {})
#   %neg : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where,), kwargs = {})
#   %exp_1 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%arg1_1,), kwargs = {})
#   %mul : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %exp_1), kwargs = {})
#   %exp_2 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul,), kwargs = {})
#   %mul_1 : Tensor "f32[512, 1000, 384][384000, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, %exp_2), kwargs = {})
#   %view_1 : Tensor "bf16[512, 1000, 384][384000, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [512, 1000, 384]), kwargs = {})
#   %mul_2 : Tensor "f32[512, 1000, 384][384000, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %view_1), kwargs = {})
#   %add : Tensor "f32[512, 1000, 384][384000, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   return %add
triton_poi_fused__unsafe_view_add_exp_mul_neg_softplus_0 = async_compile.triton('triton_poi_fused__unsafe_view_add_exp_mul_neg_softplus_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_exp_mul_neg_softplus_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2752515072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_exp_mul_neg_softplus_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp2 = tl.full([1], 20.0, tl.float32)
    tmp3 = tmp1 > tmp2
    tmp4 = libdevice.exp(tmp1)
    tmp5 = libdevice.log1p(tmp4)
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp7 = -tmp6
    tmp9 = libdevice.exp(tmp8)
    tmp10 = tmp7 * tmp9
    tmp11 = libdevice.exp(tmp10)
    tmp12 = tmp0 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp6 * tmp14
    tmp16 = tmp12 + tmp15
    tl.store(out_ptr0 + (x2), tmp16, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
        args.clear()
        assert_size_stride(arg0_1, (384, ), (1, ))
        assert_size_stride(arg1_1, (384, ), (1, ))
        assert_size_stride(arg2_1, (512, 1000, 384), (384000, 384, 1))
        assert_size_stride(arg3_1, (384, 384), (384, 1))
        assert_size_stride(arg4_1, (512, 1000, 384), (384000, 384, 1))
        with torch.cuda._DeviceGuard(5):
            torch.cuda.set_device(5)
            buf0 = empty_strided_cuda((512000, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, getattr_1], Original ATen: [aten.view, aten.permute, aten.mm]
            extern_kernels.mm(reinterpret_tensor(arg4_1, (512000, 384), (384, 1), 0), reinterpret_tensor(arg3_1, (384, 384), (1, 384), 0), out=buf0)
            del arg3_1
            del arg4_1
            buf1 = empty_strided_cuda((512, 1000, 384), (384000, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [dt, neg, A, mul, decay, mul_1, matmul, mul_2, x], Original ATen: [aten.softplus, aten.neg, aten.exp, aten.mul, aten._unsafe_view, aten.add]
            stream5 = get_raw_stream(5)
            triton_poi_fused__unsafe_view_add_exp_mul_neg_softplus_0.run(arg2_1, arg0_1, arg1_1, buf0, buf1, 196608000, stream=stream5)
            del arg0_1
            del arg1_1
            del arg2_1
            del buf0
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    arg1_1 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    arg2_1 = rand_strided((512, 1000, 384), (384000, 384, 1), device='cuda:5', dtype=torch.float32)
    arg3_1 = rand_strided((384, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    arg4_1 = rand_strided((512, 1000, 384), (384000, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
