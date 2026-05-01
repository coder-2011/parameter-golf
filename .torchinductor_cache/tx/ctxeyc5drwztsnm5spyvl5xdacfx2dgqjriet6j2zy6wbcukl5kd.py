# AOT ID: ['8_backward']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/dn/cdneiw35mysetud2do5pmgoekm5cmifiw4o57bydp2zvfqsxejp6.py
# Topologically Sorted Source Nodes: [linear, mul_3, sum_1, mul_5, sum_2], Original ATen: [aten._unsafe_view, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   linear => view_1
#   mul_3 => mul_3
#   mul_5 => mul_5
#   sum_1 => sum_1
#   sum_2 => sum_2
# Graph fragment:
#   %tangents_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=tangents_1]
#   %mm : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm]
#   %primals_4 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=primals_4]
#   %view_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 384]), kwargs = {})
#   %mul_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %view_1), kwargs = {})
#   %sum_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [0, 1], True), kwargs = {dtype: torch.float32})
#   %mul_5 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %primals_4), kwargs = {})
#   %sum_2 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_5, [0, 1], True), kwargs = {dtype: torch.float32})
#   return %buf0,%buf2
triton_red_fused__unsafe_view_mul_sum_0 = async_compile.triton('triton_red_fused__unsafe_view_mul_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_mul_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 404797440, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_mul_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 134016
    r0_numel = 376
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 384
    x0 = (xindex % 384)
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 376*x1
        tmp1 = tl.full([1, 1], 131072, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp12 = tl.load(in_ptr2 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp3 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask & xmask, tmp19, _tmp18)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/rk/crkhv4wwbvldzt5j3dzd5e2crh5s5gudposov5ncxilrx6dt2iv3.py
# Topologically Sorted Source Nodes: [linear, mul_3, sum_1], Original ATen: [aten._unsafe_view, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   linear => view_1
#   mul_3 => mul_3
#   sum_1 => sum_1
# Graph fragment:
#   %buf0 : Tensor "f32[1, 1, 384, 349][134016, 134016, 1, 384]cuda:2" = PlaceHolder[target=buf0]
#   %view_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 384]), kwargs = {})
#   %mul_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %view_1), kwargs = {})
#   %sum_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [0, 1], True), kwargs = {dtype: torch.float32})
#   return %sum_1
triton_red_fused__unsafe_view_mul_sum_1 = async_compile.triton('triton_red_fused__unsafe_view_mul_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_mul_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 539136, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_mul_sum_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 384
    r0_numel = 349
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 384*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/35/c35rosf46auvv3haizg4mimq7ylz2fd4pyvpzxpyh7absathnivu.py
# Topologically Sorted Source Nodes: [mul_2, convert_element_type_5, mul_4, convert_element_type_6], Original ATen: [aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   convert_element_type_5 => convert_element_type_5
#   convert_element_type_6 => convert_element_type_6
#   mul_2 => mul_2
#   mul_4 => mul_4
# Graph fragment:
#   %tangents_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=tangents_1]
#   %primals_3 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2" = PlaceHolder[target=primals_3]
#   %primals_6 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2" = PlaceHolder[target=primals_6]
#   %mul_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %primals_6), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %mul_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %primals_3), kwargs = {})
#   %convert_element_type_6 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   return %convert_element_type_6,%convert_element_type_5
triton_poi_fused__to_copy_mul_2 = async_compile.triton('triton_poi_fused__to_copy_mul_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 603982848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp0 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp3, None)
    tl.store(out_ptr1 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ux/cuxeqjxghcks64q5w4s26h65bjjij4t72vvmmtmdcsmihhziblt7.py
# Topologically Sorted Source Nodes: [view_3, convert_element_type_11], Original ATen: [aten.view, aten._to_copy]
# Source node to ATen node mapping:
#   convert_element_type_11 => convert_element_type_11
#   view_3 => view_3
# Graph fragment:
#   %mm_2 : Tensor "bf16[131072, 1536][1536, 1]cuda:2" = PlaceHolder[target=mm_2]
#   %view_3 : Tensor "bf16[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [128, 1024, 1536]), kwargs = {})
#   %convert_element_type_11 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   return %convert_element_type_11
triton_poi_fused__to_copy_view_3 = async_compile.triton('triton_poi_fused__to_copy_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2013265920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201326592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
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
        primals_3, primals_4, primals_6, view, mm, permute_3, tangents_1 = args
        args.clear()
        assert_size_stride(primals_3, (1, 1, 384), (384, 384, 1))
        assert_size_stride(primals_4, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(primals_6, (1, 1, 384), (384, 384, 1))
        assert_size_stride(view, (131072, 1536), (1536, 1))
        assert_size_stride(mm, (131072, 384), (384, 1))
        assert_size_stride(permute_3, (384, 1536), (1536, 1))
        assert_size_stride(tangents_1, (128, 1024, 384), (393216, 384, 1))
        with torch.cuda._DeviceGuard(2):
            torch.cuda.set_device(2)
            buf0 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf2 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            # Topologically Sorted Source Nodes: [linear, mul_3, sum_1, mul_5, sum_2], Original ATen: [aten._unsafe_view, aten.mul, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused__unsafe_view_mul_sum_0.run(tangents_1, mm, primals_4, buf0, buf2, 134016, 376, stream=stream2)
            del mm
            del primals_4
            buf1 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, mul_3, sum_1], Original ATen: [aten._unsafe_view, aten.mul, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused__unsafe_view_mul_sum_1.run(buf0, buf1, 384, 349, stream=stream2)
            del buf0
            buf3 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_5, sum_2], Original ATen: [aten.mul, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused__unsafe_view_mul_sum_1.run(buf2, buf3, 384, 349, stream=stream2)
            del buf2
            buf4 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            buf5 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_2, convert_element_type_5, mul_4, convert_element_type_6], Original ATen: [aten.mul, aten._to_copy]
            stream2 = get_raw_stream(2)
            triton_poi_fused__to_copy_mul_2.run(tangents_1, primals_3, primals_6, buf4, buf5, 50331648, stream=stream2)
            del primals_3
            del primals_6
            buf6 = empty_strided_cuda((384, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_2, convert_element_type_5, view_2, permute_1, mm_1], Original ATen: [aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf5, (384, 131072), (1, 384), 0), view, out=buf6)
            del view
            buf7 = empty_strided_cuda((131072, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_2, convert_element_type_5, view_2, mm_2], Original ATen: [aten.mul, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf5, (131072, 384), (384, 1), 0), permute_3, out=buf7)
            del buf5
            del permute_3
            buf8 = empty_strided_cuda((128, 1024, 1536), (1572864, 1536, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_3, convert_element_type_11], Original ATen: [aten.view, aten._to_copy]
            stream2 = get_raw_stream(2)
            triton_poi_fused__to_copy_view_3.run(buf7, buf8, 201326592, stream=stream2)
            del buf7
        return (buf6, buf8, buf3, buf4, tangents_1, buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_3 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:2', dtype=torch.float32)
    primals_4 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_6 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:2', dtype=torch.float32)
    view = rand_strided((131072, 1536), (1536, 1), device='cuda:2', dtype=torch.bfloat16)
    mm = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_3 = rand_strided((384, 1536), (1536, 1), device='cuda:2', dtype=torch.bfloat16)
    tangents_1 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.float32)
    return [primals_3, primals_4, primals_6, view, mm, permute_3, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
