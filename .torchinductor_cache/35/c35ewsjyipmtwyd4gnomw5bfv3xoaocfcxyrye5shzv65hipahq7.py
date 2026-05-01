# AOT ID: ['5_inference']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/55/c55glqyyxu6srfgsl5zkthpen75ktbul7orfriibwdmz7yybuuxn.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_2
# Graph fragment:
#   %arg1_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=arg1_1]
#   %convert_element_type_2 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 402653184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/jv/cjv6rrzifbijjaxslktbni2qo4aekmac3vgw5ixftod545ka5han.py
# Topologically Sorted Source Nodes: [linear, leaky_relu, square, linear_1], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy]
# Source node to ATen node mapping:
#   leaky_relu => convert_element_type_5, gt, mul, where
#   linear => view_1
#   linear_1 => convert_element_type_10
#   square => pow_1
# Graph fragment:
#   %mm : Tensor "bf16[131072, 1024][1024, 1]cuda:2" = PlaceHolder[target=mm]
#   %view_1 : Tensor "bf16[128, 1024, 1024][1048576, 1024, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 1024]), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:2"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %gt : Tensor "b8[128, 1024, 1024][1048576, 1024, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_5, 0), kwargs = {})
#   %mul : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, 0.5), kwargs = {})
#   %where : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convert_element_type_5, %mul), kwargs = {})
#   %pow_1 : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where, 2), kwargs = {})
#   %convert_element_type_10 : Tensor "bf16[128, 1024, 1024][1048576, 1024, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_10
triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_1 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 805306368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_1(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.0, tl.float32)
    tmp3 = tmp1 > tmp2
    tmp4 = tl.full([1], 0.5, tl.float32)
    tmp5 = tmp1 * tmp4
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp7 = tmp6 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/lc/clc3toq7vjjcpofykjahp4eoywujnyuua2fogxkpmirgxsn2i72q.py
# Topologically Sorted Source Nodes: [mul, add, linear_1, mul_22, add_1], Original ATen: [aten.mul, aten.add, aten._unsafe_view]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   linear_1 => view_3
#   mul => mul_1
#   mul_22 => mul_2
# Graph fragment:
#   %arg5_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=arg5_1]
#   %arg3_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=arg4_1]
#   %arg6_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2" = PlaceHolder[target=arg6_1]
#   %mm_1 : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm_1]
#   %mul_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, %arg4_1), kwargs = {})
#   %add : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg5_1, %mul_1), kwargs = {})
#   %view_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [128, 1024, 384]), kwargs = {})
#   %mul_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg6_1, %view_3), kwargs = {})
#   %add_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %mul_2), kwargs = {})
#   return %add_1
triton_poi_fused__unsafe_view_add_mul_2 = async_compile.triton('triton_poi_fused__unsafe_view_add_mul_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_mul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 805309440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_mul_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp5 + tmp9
    tl.store(out_ptr0 + (x2), tmp10, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1024, 384), (384, 1))
        assert_size_stride(arg1_1, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(arg2_1, (384, 1024), (1024, 1))
        assert_size_stride(arg3_1, (1, 1, 384), (384, 384, 1))
        assert_size_stride(arg4_1, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(arg5_1, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(arg6_1, (1, 1, 384), (384, 384, 1))
        with torch.cuda._DeviceGuard(2):
            torch.cuda.set_device(2)
            buf0 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
            stream2 = get_raw_stream(2)
            triton_poi_fused__to_copy_0.run(arg1_1, buf0, 50331648, stream=stream2)
            del arg1_1
            buf1 = empty_strided_cuda((131072, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf0, (131072, 384), (384, 1), 0), reinterpret_tensor(arg0_1, (384, 1024), (1, 384), 0), out=buf1)
            del arg0_1
            del buf0
            buf2 = reinterpret_tensor(buf1, (128, 1024, 1024), (1048576, 1024, 1), 0); del buf1  # reuse
            # Topologically Sorted Source Nodes: [linear, leaky_relu, square, linear_1], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy]
            stream2 = get_raw_stream(2)
            triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_1.run(buf2, 134217728, stream=stream2)
            buf3 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, leaky_relu, square, linear_1], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf2, (131072, 1024), (1024, 1), 0), reinterpret_tensor(arg2_1, (1024, 384), (1, 1024), 0), out=buf3)
            del arg2_1
            del buf2
            buf4 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul, add, linear_1, mul_22, add_1], Original ATen: [aten.mul, aten.add, aten._unsafe_view]
            stream2 = get_raw_stream(2)
            triton_poi_fused__unsafe_view_add_mul_2.run(arg5_1, arg3_1, arg4_1, arg6_1, buf3, buf4, 50331648, stream=stream2)
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            del buf3
        return (buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1024, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    arg1_1 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.float32)
    arg2_1 = rand_strided((384, 1024), (1024, 1), device='cuda:2', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:2', dtype=torch.float32)
    arg4_1 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    arg5_1 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.float32)
    arg6_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:2', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
