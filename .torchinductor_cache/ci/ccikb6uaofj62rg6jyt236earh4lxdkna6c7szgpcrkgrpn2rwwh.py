# AOT ID: ['6_backward']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/un/cunodkkuxuvvebhze3wkpm2not7xlokepxts2u5natv7vkw2dyrp.py
# Topologically Sorted Source Nodes: [matmul, mul_4, sum_1, mul_5, sum_2], Original ATen: [aten._unsafe_view, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   matmul => view_1
#   mul_4 => mul_4
#   mul_5 => mul_5
#   sum_1 => sum_1
#   sum_2 => sum_2
# Graph fragment:
#   %tangents_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=tangents_1]
#   %mm : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm]
#   %primals_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=primals_3]
#   %view_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 384]), kwargs = {})
#   %mul_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %view_1), kwargs = {})
#   %sum_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [0, 1], True), kwargs = {dtype: torch.float32})
#   %mul_5 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %primals_3), kwargs = {})
#   %sum_2 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_5, [0, 1], True), kwargs = {dtype: torch.float32})
#   return %buf0,%buf5
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_mul_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 505460736, 'r0_': 0}}
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
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
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
        tmp12 = tl.load(in_ptr2 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp3 * tmp12
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/2w/c2wqtlbgjlrgin63zqfrl4j7yuo3z4en2l57iidhd47rhca7jtsu.py
# Topologically Sorted Source Nodes: [matmul, mul_4, sum_1], Original ATen: [aten._unsafe_view, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   matmul => view_1
#   mul_4 => mul_4
#   sum_1 => sum_1
# Graph fragment:
#   %buf0 : Tensor "f32[1, 1, 384, 349][134016, 134016, 1, 384]cuda:5" = PlaceHolder[target=buf0]
#   %view_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 384]), kwargs = {})
#   %mul_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %view_1), kwargs = {})
#   %sum_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [0, 1], True), kwargs = {dtype: torch.float32})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/yq/cyqz73m7kvcwlkol66fsbcnwk5rpqmiyiyhex33iiykocr3ryjqy.py
# Topologically Sorted Source Nodes: [dt, mul_3, convert_element_type_2], Original ATen: [aten.softplus, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   convert_element_type_2 => convert_element_type_2
#   dt => exp, gt, log1p, where
#   mul_3 => mul_3
# Graph fragment:
#   %tangents_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=tangents_1]
#   %primals_1 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_1]
#   %exp : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%primals_1,), kwargs = {})
#   %log1p : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %gt : Tensor "b8[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%primals_1, 20), kwargs = {})
#   %where : Tensor "f32[384][1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %primals_1, %log1p), kwargs = {})
#   %mul_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %where), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2
triton_poi_fused__to_copy_mul_softplus_2 = async_compile.triton('triton_poi_fused__to_copy_mul_softplus_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_softplus_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 402654720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_softplus_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.full([1], 20.0, tl.float32)
    tmp3 = tmp1 > tmp2
    tmp4 = libdevice.exp(tmp1)
    tmp5 = libdevice.log1p(tmp4)
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/qa/cqauacjdlchinmbqqpzr3zyajoe7sqszwtikv62nwvj7uonrma5l.py
# Topologically Sorted Source Nodes: [dt, view_2, mul_5, sum_2, view_5, A, neg, mul, decay, mul_6, mul_7, mul_8, neg_1, add_1, mul_9, mul_10, exp_3, gt_1, mul_12, add_2, div, where_1], Original ATen: [aten.softplus, aten.view, aten.mul, aten.sum, aten.exp, aten.neg, aten.add, aten.softplus_backward]
# Source node to ATen node mapping:
#   A => exp_1
#   add_1 => add_1
#   add_2 => add_2
#   decay => exp_2
#   div => div
#   dt => exp, gt, log1p, where
#   exp_3 => exp_3
#   gt_1 => gt_1
#   mul => mul
#   mul_10 => mul_10
#   mul_12 => mul_12
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   neg => neg
#   neg_1 => neg_1
#   sum_2 => sum_2
#   view_2 => view_2
#   view_5 => view_5
#   where_1 => where_1
# Graph fragment:
#   %buf5 : Tensor "f32[1, 1, 384, 349][134016, 134016, 1, 384]cuda:5" = PlaceHolder[target=buf5]
#   %sum_2 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5" = PlaceHolder[target=sum_2]
#   %primals_1 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_2]
#   %sum_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5" = PlaceHolder[target=sum_1]
#   %exp : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%primals_1,), kwargs = {})
#   %log1p : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %gt : Tensor "b8[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%primals_1, 20), kwargs = {})
#   %where : Tensor "f32[384][1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %primals_1, %log1p), kwargs = {})
#   %view_2 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_1, [384]), kwargs = {})
#   %mul_5 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %primals_3), kwargs = {})
#   %sum_2 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_5, [0, 1], True), kwargs = {dtype: torch.float32})
#   %view_5 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_2, [384]), kwargs = {})
#   %exp_1 : Tensor "f32[384][1]cuda:5"[num_users=3] = call_function[target=torch.ops.aten.exp.default](args = (%primals_2,), kwargs = {})
#   %neg : Tensor "f32[384][1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%where,), kwargs = {})
#   %mul : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %exp_1), kwargs = {})
#   %exp_2 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul,), kwargs = {})
#   %mul_6 : Tensor "f32[384][1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %exp_2), kwargs = {})
#   %mul_7 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %neg), kwargs = {})
#   %mul_8 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %exp_1), kwargs = {})
#   %neg_1 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_8,), kwargs = {})
#   %add_1 : Tensor "f32[384][1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %neg_1), kwargs = {})
#   %mul_9 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %exp_1), kwargs = {})
#   %mul_10 : Tensor "f32[384][1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, 1), kwargs = {})
#   %exp_3 : Tensor "f32[384][1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_10,), kwargs = {})
#   %gt_1 : Tensor "b8[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_10, 20), kwargs = {})
#   %mul_12 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %exp_3), kwargs = {})
#   %add_2 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_3, 1.0), kwargs = {})
#   %div : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_12, %add_2), kwargs = {})
#   %where_1 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_1, %div), kwargs = {})
#   return %sum_2,%mul_9,%where_1
triton_red_fused_add_exp_mul_neg_softplus_softplus_backward_sum_view_3 = async_compile.triton('triton_red_fused_add_exp_mul_neg_softplus_softplus_backward_sum_view_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_exp_mul_neg_softplus_softplus_backward_sum_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 546816, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_add_exp_mul_neg_softplus_softplus_backward_sum_view_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.full([1, 1], 20.0, tl.float32)
    tmp6 = tmp4 > tmp5
    tmp7 = libdevice.exp(tmp4)
    tmp8 = libdevice.log1p(tmp7)
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tmp10 = -tmp9
    tmp12 = libdevice.exp(tmp11)
    tmp13 = tmp10 * tmp12
    tmp14 = libdevice.exp(tmp13)
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 * tmp10
    tmp17 = tmp16 * tmp12
    tmp18 = tl.full([1, 1], 1.0, tl.float32)
    tmp19 = tmp4 * tmp18
    tmp20 = tmp19 > tmp5
    tmp22 = tmp15 * tmp12
    tmp23 = -tmp22
    tmp24 = tmp21 + tmp23
    tmp25 = libdevice.exp(tmp19)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp25 + tmp18
    tmp28 = (tmp26 / tmp27)
    tmp29 = tl.where(tmp20, tmp24, tmp28)
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp29, xmask)
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
        primals_1, primals_2, primals_3, primals_4, view, mm, tangents_1 = args
        args.clear()
        assert_size_stride(primals_1, (384, ), (1, ))
        assert_size_stride(primals_2, (384, ), (1, ))
        assert_size_stride(primals_3, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(primals_4, (384, 384), (384, 1))
        assert_size_stride(view, (131072, 384), (384, 1))
        assert_size_stride(mm, (131072, 384), (384, 1))
        assert_size_stride(tangents_1, (128, 1024, 384), (393216, 384, 1))
        with torch.cuda._DeviceGuard(5):
            torch.cuda.set_device(5)
            buf0 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf5 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            # Topologically Sorted Source Nodes: [matmul, mul_4, sum_1, mul_5, sum_2], Original ATen: [aten._unsafe_view, aten.mul, aten.sum]
            stream5 = get_raw_stream(5)
            triton_red_fused__unsafe_view_mul_sum_0.run(tangents_1, mm, primals_3, buf0, buf5, 134016, 376, stream=stream5)
            del mm
            del primals_3
            buf1 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [matmul, mul_4, sum_1], Original ATen: [aten._unsafe_view, aten.mul, aten.sum]
            stream5 = get_raw_stream(5)
            triton_red_fused__unsafe_view_mul_sum_1.run(buf0, buf1, 384, 349, stream=stream5)
            del buf0
            buf2 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dt, mul_3, convert_element_type_2], Original ATen: [aten.softplus, aten.mul, aten._to_copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused__to_copy_mul_softplus_2.run(tangents_1, primals_1, buf2, 50331648, stream=stream5)
            del tangents_1
            buf3 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dt, mul_3, convert_element_type_2, view_3, permute_1, mm_1], Original ATen: [aten.softplus, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf2, (384, 131072), (1, 384), 0), view, out=buf3)
            del view
            buf4 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dt, mul_3, convert_element_type_2, view_3, getattr_1, permute_3, mm_2], Original ATen: [aten.softplus, aten.mul, aten._to_copy, aten.view, aten.permute, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf2, (131072, 384), (384, 1), 0), primals_4, out=buf4)
            del buf2
            del primals_4
            buf7 = empty_strided_cuda((384, ), (1, ), torch.float32)
            buf8 = reinterpret_tensor(buf1, (384, ), (1, ), 0); del buf1  # reuse
            # Topologically Sorted Source Nodes: [dt, view_2, mul_5, sum_2, view_5, A, neg, mul, decay, mul_6, mul_7, mul_8, neg_1, add_1, mul_9, mul_10, exp_3, gt_1, mul_12, add_2, div, where_1], Original ATen: [aten.softplus, aten.view, aten.mul, aten.sum, aten.exp, aten.neg, aten.add, aten.softplus_backward]
            stream5 = get_raw_stream(5)
            triton_red_fused_add_exp_mul_neg_softplus_softplus_backward_sum_view_3.run(buf8, buf5, primals_1, primals_2, buf7, 384, 349, stream=stream5)
            del buf5
            del primals_1
            del primals_2
        return (buf8, buf7, None, buf3, reinterpret_tensor(buf4, (128, 1024, 384), (393216, 384, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_2 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_3 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.float32)
    primals_4 = rand_strided((384, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    view = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    mm = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    tangents_1 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, view, mm, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
