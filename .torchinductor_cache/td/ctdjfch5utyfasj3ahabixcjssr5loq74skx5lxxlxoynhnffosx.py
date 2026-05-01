# AOT ID: ['8_forward']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ju/cju6vffyvcs4njybr464fbmo7llewzoywsz37dradub2e3ocihyd.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear => convert_element_type_default_5, permute
# Graph fragment:
#   %primals_1 : Tensor "bf16[1024, 384][384, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %convert_element_type_default_5 : Tensor "bf16[1024, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.bfloat16), kwargs = {})
#   %permute : Tensor "bf16[384, 1024][1, 384]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_default_5, [1, 0]), kwargs = {})
#   return %permute
triton_poi_fused__to_copy_t_0 = async_compile.triton('triton_poi_fused__to_copy_t_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/cc/cccy6z7t4xu545khd4bvfcjkpdhzf4blodgth6sqn45xgsouheci.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_2
# Graph fragment:
#   %primals_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %convert_element_type_2 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 402653184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/4b/c4bjsqxys6opna2s7d6eh3wrujh4w7pgavtllh7chdazcahmb4po.py
# Topologically Sorted Source Nodes: [linear, leaky_relu, square, linear_1], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy]
# Source node to ATen node mapping:
#   leaky_relu => convert_element_type_5, gt, mul, where
#   linear => view_1
#   linear_1 => convert_element_type_10
#   square => pow_1
# Graph fragment:
#   %mm : Tensor "bf16[131072, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_1 : Tensor "bf16[128, 1024, 1024][1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 1024]), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %gt : Tensor "b8[128, 1024, 1024][1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_5, 0), kwargs = {})
#   %mul : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, 0.5), kwargs = {})
#   %where : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convert_element_type_5, %mul), kwargs = {})
#   %pow_1 : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where, 2), kwargs = {})
#   %convert_element_type_10 : Tensor "bf16[128, 1024, 1024][1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_10
triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_2 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 805306368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.0, tl.float32)
    tmp3 = tmp1 > tmp2
    tmp4 = tl.full([1], 0.5, tl.float32)
    tmp5 = tmp1 * tmp4
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp7 = tmp6 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/7c/c7c2ke35owncolxybqsdvfqitth2llffvm2zfhgp335fit7ndpkg.py
# Topologically Sorted Source Nodes: [linear_1, mul, add, mul_22, add_1], Original ATen: [aten._unsafe_view, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   linear_1 => view_3
#   mul => mul_1
#   mul_22 => mul_2
# Graph fragment:
#   %primals_6 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0" = PlaceHolder[target=primals_6]
#   %primals_4 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_5 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:0" = PlaceHolder[target=primals_5]
#   %primals_7 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %mm_1 : Tensor "bf16[131072, 384][384, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %view_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [128, 1024, 384]), kwargs = {})
#   %mul_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, %primals_5), kwargs = {})
#   %add : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_6, %mul_1), kwargs = {})
#   %mul_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_7, %view_3), kwargs = {})
#   %add_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %mul_2), kwargs = {})
#   return %add_1
triton_poi_fused__unsafe_view_add_mul_3 = async_compile.triton('triton_poi_fused__unsafe_view_add_mul_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_mul_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 805309440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_mul_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
        args.clear()
        assert_size_stride(primals_1, (1024, 384), (384, 1))
        assert_size_stride(primals_2, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(primals_3, (384, 1024), (1024, 1))
        assert_size_stride(primals_4, (1, 1, 384), (384, 384, 1))
        assert_size_stride(primals_5, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(primals_6, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(primals_7, (1, 1, 384), (384, 384, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((384, 1024), (1, 384), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_0.run(primals_1, buf0, 393216, stream=stream0)
            del primals_1
            buf1 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_2, buf1, 50331648, stream=stream0)
            del primals_2
            buf2 = empty_strided_cuda((131072, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1, (131072, 384), (384, 1), 0), buf0, out=buf2)
            buf3 = empty_strided_cuda((1024, 384), (1, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_0.run(primals_3, buf3, 393216, stream=stream0)
            del primals_3
            buf4 = empty_strided_cuda((128, 1024, 1024), (1048576, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, leaky_relu, square, linear_1], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_2.run(buf2, buf4, 134217728, stream=stream0)
            buf5 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, leaky_relu, square, linear_1], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf4, (131072, 1024), (1024, 1), 0), buf3, out=buf5)
            buf6 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, mul, add, mul_22, add_1], Original ATen: [aten._unsafe_view, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_3.run(primals_6, primals_4, primals_5, primals_7, buf5, buf6, 50331648, stream=stream0)
            del primals_6
        return (buf6, primals_4, primals_5, primals_7, reinterpret_tensor(buf1, (131072, 384), (384, 1), 0), buf2, reinterpret_tensor(buf4, (131072, 1024), (1024, 1), 0), buf5, reinterpret_tensor(buf3, (384, 1024), (1024, 1), 0), reinterpret_tensor(buf0, (1024, 384), (384, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((1024, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((384, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
