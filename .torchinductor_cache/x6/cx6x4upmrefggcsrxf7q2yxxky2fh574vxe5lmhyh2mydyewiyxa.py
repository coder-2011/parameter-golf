# AOT ID: ['26_inference']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/yy/cyy73kjxxhi5l3ovzfjaoiagxwwoeecrb2cibtqsczxqbfqithty.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_5
# Graph fragment:
#   %arg7_1 : Tensor "f32[8, 12][12, 1]cuda:4" = PlaceHolder[target=arg7_1]
#   %convert_element_type_5 : Tensor "bf16[8, 12][12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg7_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_5
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/vq/cvqanlnmte5rv2izm4qqrd4wu4ewhajs3kcvw2eln6c4v4yvd6ky.py
# Topologically Sorted Source Nodes: [getitem, getitem_1, mul, getitem_2, getitem_3, mul_1, add, rms_norm, mul_2, linear, rms_norm_1, mul_5, linear_3], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._fused_rms_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add => add
#   getitem => select
#   getitem_1 => unsqueeze, unsqueeze_1
#   getitem_2 => select_1
#   getitem_3 => unsqueeze_2, unsqueeze_3
#   linear => convert_element_type_2
#   linear_3 => convert_element_type_13
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_4
#   mul_5 => mul_9
#   rms_norm => add_1, mean, mul_2, mul_3, pow_1, rsqrt
#   rms_norm_1 => add_2, mean_1, mul_7, mul_8, pow_2, rsqrt_1
# Graph fragment:
#   %arg0_1 : Tensor "f32[2, 384][384, 1]cuda:4" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:4" = PlaceHolder[target=arg2_1]
#   %buf0 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:4" = PlaceHolder[target=buf0]
#   %arg3_1 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=arg3_1]
#   %buf21 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:4" = PlaceHolder[target=buf21]
#   %arg11_1 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=arg11_1]
#   %mul_4 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4" = PlaceHolder[target=mul_4]
#   %select : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %mul : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %arg1_1), kwargs = {})
#   %select_1 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 0), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 1), kwargs = {})
#   %mul_1 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %arg2_1), kwargs = {})
#   %add : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %pow_1 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add_1 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul_2 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt), kwargs = {})
#   %mul_3 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg3_1), kwargs = {})
#   %mul_4 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 0.7071067811865475), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   %pow_2 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean_1 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_2 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_7 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt_1), kwargs = {})
#   %mul_8 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %arg11_1), kwargs = {})
#   %mul_9 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, 0.7071067811865475), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.bfloat16), kwargs = {})
#   return %buf0,%buf21,%mul_4,%convert_element_type_13,%convert_element_type_2
triton_red_fused__fused_rms_norm__to_copy_add_mul_select_unsqueeze_1 = async_compile.triton('triton_red_fused__fused_rms_norm__to_copy_add_mul_select_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 524288, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm__to_copy_add_mul_select_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 3, 'num_reduction': 2, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 2956806144}}
)
@triton.jit
def triton_red_fused__fused_rms_norm__to_copy_add_mul_select_unsqueeze_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 350000
    r0_numel = 384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tmp2 + tmp6
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr0 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp12 * tmp13
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp15 * tmp17
        tmp19 = tmp14 + tmp18
        tmp20 = tl.full([1, 1], 384.0, tl.float32)
        tmp21 = (tmp10 / tmp20)
        tmp22 = tl.full([1, 1], 1e-06, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp25 * tmp26
        tmp28 = tl.full([1, 1], 0.7071067811865475, tl.float32)
        tmp29 = tmp27 * tmp28
        tmp31 = tmp25 * tmp30
        tmp32 = tmp31 * tmp28
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp29.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 384*x0), tmp29, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_1 + 384*x0), tmp33, r0_mask & xmask)
        tl.store(out_ptr4 + (r0_1 + 384*x0), tmp34, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/wy/cwy33yodsfytfcsyp5dqepeyf7es4cgbbvwbpfupk3eo4ftjhild.py
# Topologically Sorted Source Nodes: [getitem_7, contiguous_6, linear_1], Original ATen: [aten.slice, aten.clone, aten._to_copy]
# Source node to ATen node mapping:
#   contiguous_6 => clone_1
#   getitem_7 => slice_2
#   linear_1 => convert_element_type_6
# Graph fragment:
#   %mul_4 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4" = PlaceHolder[target=mul_4]
#   %slice_2 : Tensor "f32[350, 1000, 12][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_4, 2, 0, 12), kwargs = {})
#   %clone_1 : Tensor "f32[350, 1000, 12][12000, 12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_6 : Tensor "bf16[350, 1000, 12][12000, 12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_6
triton_poi_fused__to_copy_clone_slice_2 = async_compile.triton('triton_poi_fused__to_copy_clone_slice_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_slice_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 33600000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_slice_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4200000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 12)
    x3 = xindex // 12
    x2 = xindex // 12000
    x4 = (xindex % 12000)
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x3), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x4 + 12032*x2), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/v6/cv6t7fsuevamxyrwjpjm42dzp3eljzfbdasm5noyknisek5tba63.py
# Topologically Sorted Source Nodes: [getitem_7, contiguous_6, linear_1], Original ATen: [aten.slice, aten.clone, aten._to_copy, aten.view, aten.t, aten.mm]
# Source node to ATen node mapping:
#   contiguous_6 => clone_1
#   getitem_7 => slice_2
#   linear_1 => convert_element_type_5, convert_element_type_6, mm_1, permute_7, view_3
# Graph fragment:
#   %convert_element_type_6 : Tensor "bf16[350, 1000, 12][12032, 12, 1]cuda:4" = PlaceHolder[target=convert_element_type_6]
#   %slice_2 : Tensor "f32[350, 1000, 12][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_4, 2, 0, 12), kwargs = {})
#   %clone_1 : Tensor "f32[350, 1000, 12][12000, 12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_6 : Tensor "bf16[350, 1000, 12][12000, 12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_1, torch.bfloat16), kwargs = {})
#   %view_3 : Tensor "bf16[350000, 12][12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_6, [350000, 12]), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[8, 12][12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg7_1, torch.bfloat16), kwargs = {})
#   %permute_7 : Tensor "bf16[12, 8][1, 12]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_5, [1, 0]), kwargs = {})
#   %mm_1 : Tensor "bf16[350000, 8][8, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_3, %permute_7), kwargs = {})
#   return %buf16
triton_poi_fused__to_copy_clone_mm_slice_t_view_3 = async_compile.triton('triton_poi_fused__to_copy_clone_mm_slice_t_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_mm_slice_t_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 25200000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_mm_slice_t_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4200000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 12)
    x1 = xindex // 12
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 12*((x1 % 1000)) + 12032*(x1 // 1000)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# Original path: /workspace/parameter-golf/train_gpt_parcae.py:4418
_fused_qkv_postprocess_fwd_kernel_0 = async_compile.triton('_fused_qkv_postprocess_fwd_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 2, 'num_stages': 3}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_fused_qkv_postprocess_fwd_kernel_0', 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'QKV': '*bf16', 'Q': '*bf16', 'K': '*bf16', 'COS': '*bf16', 'SIN': '*bf16', 'TOTAL_ROWS': 'constexpr', 'SEQLEN': 'constexpr', 'H_Q': 'constexpr', 'H_KV': 'constexpr', 'HEAD_DIM': 'constexpr', 'ROPE_DIMS': 'constexpr', 'DO_QK_NORM': 'constexpr', 'ROPE_INTERLEAVED': 'constexpr', 'RMS_EPS': 'constexpr', 'BLOCK_D': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'TOTAL_ROWS': 350000, 'SEQLEN': 1000, 'H_Q': 8, 'H_KV': 8, 'HEAD_DIM': 48, 'ROPE_DIMS': 48, 'DO_QK_NORM': True, 'ROPE_INTERLEAVED': False, 'RMS_EPS': 1e-06, 'BLOCK_D': 64}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _fused_qkv_postprocess_fwd_kernel(
    QKV,
    Q,
    K,
    COS,
    SIN,
    TOTAL_ROWS: tl.constexpr,
    SEQLEN: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIMS: tl.constexpr,
    DO_QK_NORM: tl.constexpr,
    ROPE_INTERLEAVED: tl.constexpr,
    RMS_EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    h_total: tl.constexpr = H_Q + 2 * H_KV
    h_qk: tl.constexpr = H_Q + H_KV
    row = pid // h_qk
    head = pid - row * h_qk
    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM

    is_q = head < H_Q
    k_head = head - H_Q
    q_head_safe = tl.minimum(head, H_Q - 1)
    k_head_safe = tl.maximum(k_head, 0)
    qkv_head = tl.where(is_q, q_head_safe, H_Q + k_head_safe)
    qkv_base = (row * h_total + qkv_head) * HEAD_DIM
    x = tl.load(QKV + qkv_base + offs, mask=mask, other=0.0).to(tl.float32)

    y = x
    rope_mask = offs < ROPE_DIMS
    rope_half: tl.constexpr = ROPE_DIMS // 2
    if ROPE_INTERLEAVED:
        pair = offs // 2
        is_first = (offs & 1) == 0
        mate_offs = tl.where(is_first, offs + 1, offs - 1)
    else:
        is_first = offs < rope_half
        pair = tl.where(is_first, offs, offs - rope_half)
        mate_offs = tl.where(is_first, offs + rope_half, offs - rope_half)
    mate = tl.load(QKV + qkv_base + mate_offs, mask=mask & rope_mask, other=0.0).to(tl.float32)
    token_idx = row - (row // SEQLEN) * SEQLEN
    cos = tl.load(COS + token_idx * (ROPE_DIMS // 2) + pair, mask=rope_mask, other=1.0).to(tl.float32)
    sin = tl.load(SIN + token_idx * (ROPE_DIMS // 2) + pair, mask=rope_mask, other=0.0).to(tl.float32)
    rot = tl.where(is_first, x * cos - mate * sin, mate * sin + x * cos)
    y = tl.where(rope_mask, rot, x)
    if DO_QK_NORM:
        ss = tl.sum(tl.where(mask, y * y, 0.0), axis=0)
        inv_rms = tl.rsqrt(ss / HEAD_DIM + RMS_EPS)
        y = y * inv_rms

    q_base = (row * H_Q + q_head_safe) * HEAD_DIM
    k_base = (row * H_KV + k_head_safe) * HEAD_DIM
    tl.store(Q + q_base + offs, y, mask=mask & is_q)
    tl.store(K + k_base + offs, y, mask=mask & ~is_q)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/oc/cocx3iundq3nygneynpmcwjg6vq2fzsx6uco4gx5umhjjywmx2ye.py
# Topologically Sorted Source Nodes: [linear, view, scaled_dot_product_attention, getitem_4, contiguous, transpose_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
# Source node to ATen node mapping:
#   contiguous => clone
#   getitem_4 => slice_1
#   linear => view_1
#   scaled_dot_product_attention => _scaled_dot_product_flash_attention, permute_4, permute_5
#   transpose_2 => permute_3
#   view => view_2
# Graph fragment:
#   %mm : Tensor "bf16[350000, 1152][1152, 1]cuda:4" = PlaceHolder[target=mm]
#   %view_1 : Tensor "bf16[350, 1000, 1152][1152000, 1152, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [350, 1000, 1152]), kwargs = {})
#   %view_2 : Tensor "bf16[350, 1000, 24, 48][1152000, 1152, 48, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [350, 1000, 24, 48]), kwargs = {})
#   %permute_4 : Tensor "bf16[350, 8, 1000, 48][384000, 48, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty, [0, 2, 1, 3]), kwargs = {})
#   %permute_5 : Tensor "bf16[350, 8, 1000, 48][384000, 48, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty_1, [0, 2, 1, 3]), kwargs = {})
#   %slice_1 : Tensor "bf16[350, 1000, 8, 48][1152000, 1152, 48, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_2, 2, 16, 9223372036854775807), kwargs = {})
#   %clone : Tensor "bf16[350, 1000, 8, 48][384000, 384, 48, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_1,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_3 : Tensor "bf16[350, 8, 1000, 48][384000, 48, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%clone, [0, 2, 1, 3]), kwargs = {})
#   %_scaled_dot_product_flash_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%permute_4, %permute_5, %permute_3, 0.0, True), kwargs = {scale: 0.14433756729740646})
#   return %buf8
triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_4 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 806400000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134400000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = xindex // 384
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + 1152*x1), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/em/cem4y65cipt6ypg6ur6tyqlnp77gxeb6xjmg6uof3x47mso6pjp7.py
# Topologically Sorted Source Nodes: [transpose_3, linear_1, mul_3, sigmoid, getitem_8, mul_4], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
# Source node to ATen node mapping:
#   getitem_8 => unsqueeze_4
#   linear_1 => view_4
#   mul_3 => mul_5
#   mul_4 => mul_6
#   sigmoid => sigmoid
#   transpose_3 => permute_6
# Graph fragment:
#   %getitem_2 : Tensor "bf16[350, 8, 1000, 48][384000, 48, 384, 1]cuda:4" = PlaceHolder[target=getitem_2]
#   %mm_1 : Tensor "bf16[350000, 8][8, 1]cuda:4" = PlaceHolder[target=mm_1]
#   %permute_6 : Tensor "bf16[350, 1000, 8, 48][384000, 384, 48, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [0, 2, 1, 3]), kwargs = {})
#   %view_4 : Tensor "bf16[350, 1000, 8][8000, 8, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [350, 1000, 8]), kwargs = {})
#   %mul_5 : Tensor "bf16[350, 1000, 8][8000, 8, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, 0.5), kwargs = {})
#   %sigmoid : Tensor "bf16[350, 1000, 8][8000, 8, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_5,), kwargs = {})
#   %unsqueeze_4 : Tensor "bf16[350, 1000, 8, 1][8000, 8, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid, 3), kwargs = {})
#   %mul_6 : Tensor "bf16[350, 1000, 8, 48][384000, 384, 48, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_6, %unsqueeze_4), kwargs = {})
#   return %mul_6
triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5 = async_compile.triton('triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 806400000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134400000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 48
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/5w/c5wxfyhveq5xk6y6mqc3kmudmuhe7u2yj7m6bthft7gibin3bbjs.py
# Topologically Sorted Source Nodes: [linear_3, leaky_relu, square, linear_4], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy]
# Source node to ATen node mapping:
#   leaky_relu => convert_element_type_16, gt, mul_10, where
#   linear_3 => view_9
#   linear_4 => convert_element_type_21
#   square => pow_3
# Graph fragment:
#   %mm_3 : Tensor "bf16[350000, 1536][1536, 1]cuda:4" = PlaceHolder[target=mm_3]
#   %view_9 : Tensor "bf16[350, 1000, 1536][1536000, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [350, 1000, 1536]), kwargs = {})
#   %convert_element_type_16 : Tensor "f32[350, 1000, 1536][1536000, 1536, 1]cuda:4"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_9, torch.float32), kwargs = {})
#   %gt : Tensor "b8[350, 1000, 1536][1536000, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_16, 0), kwargs = {})
#   %mul_10 : Tensor "f32[350, 1000, 1536][1536000, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_16, 0.5), kwargs = {})
#   %where : Tensor "f32[350, 1000, 1536][1536000, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convert_element_type_16, %mul_10), kwargs = {})
#   %pow_3 : Tensor "f32[350, 1000, 1536][1536000, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where, 2), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[350, 1000, 1536][1536000, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_3, torch.bfloat16), kwargs = {})
#   return %convert_element_type_21
triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_6 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3225600000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_6(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 537600000
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/mi/cmilvfysdzz4ivq7y7fqitorchqdaivahqalrolqawcmzgortd3x.py
# Topologically Sorted Source Nodes: [getitem, getitem_1, mul, getitem_2, getitem_3, mul_1, add, getitem_11, getitem_12, getitem_9, linear_2, mul_6, add_1, getitem_10, linear_4, mul_7, add_2, mul_8, getitem_13, getitem_14, mul_9, add_3, rms_norm_2, mul_10, linear_5, rms_norm_3, mul_13, linear_8], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._unsafe_view, aten._fused_rms_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_3
#   add_2 => add_4
#   add_3 => add_5
#   getitem => select
#   getitem_1 => unsqueeze, unsqueeze_1
#   getitem_10 => unsqueeze_7, unsqueeze_8
#   getitem_11 => select_6
#   getitem_12 => unsqueeze_10, unsqueeze_9
#   getitem_13 => select_7
#   getitem_14 => unsqueeze_11, unsqueeze_12
#   getitem_2 => select_1
#   getitem_3 => unsqueeze_2, unsqueeze_3
#   getitem_9 => unsqueeze_5, unsqueeze_6
#   linear_2 => view_7
#   linear_4 => view_11
#   linear_5 => convert_element_type_26
#   linear_8 => convert_element_type_37
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_17
#   mul_13 => mul_22
#   mul_6 => mul_11
#   mul_7 => mul_12
#   mul_8 => mul_13
#   mul_9 => mul_14
#   rms_norm_2 => add_6, mean_2, mul_15, mul_16, pow_4, rsqrt_2
#   rms_norm_3 => add_7, mean_3, mul_20, mul_21, pow_5, rsqrt_3
# Graph fragment:
#   %arg14_1 : Tensor "f32[2, 384][384, 1]cuda:4" = PlaceHolder[target=arg14_1]
#   %arg0_1 : Tensor "f32[2, 384][384, 1]cuda:4" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:4" = PlaceHolder[target=arg2_1]
#   %arg9_1 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=arg9_1]
#   %mm_2 : Tensor "bf16[350000, 384][384, 1]cuda:4" = PlaceHolder[target=mm_2]
#   %arg10_1 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=arg10_1]
#   %mm_4 : Tensor "bf16[350000, 384][384, 1]cuda:4" = PlaceHolder[target=mm_4]
#   %mul_13 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4" = PlaceHolder[target=mul_13]
#   %buf27 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:4" = PlaceHolder[target=buf27]
#   %arg15_1 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=arg15_1]
#   %buf48 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:4" = PlaceHolder[target=buf48]
#   %arg21_1 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=arg21_1]
#   %mul_17 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4" = PlaceHolder[target=mul_17]
#   %select : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %mul : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %arg1_1), kwargs = {})
#   %select_1 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 0), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 1), kwargs = {})
#   %mul_1 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %arg2_1), kwargs = {})
#   %add : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %select_6 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg14_1, 0, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_6, 0), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, 0), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 1), kwargs = {})
#   %view_7 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [350, 1000, 384]), kwargs = {})
#   %mul_11 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_6, %view_7), kwargs = {})
#   %add_3 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %mul_11), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 1), kwargs = {})
#   %view_11 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [350, 1000, 384]), kwargs = {})
#   %mul_12 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_8, %view_11), kwargs = {})
#   %add_4 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mul_12), kwargs = {})
#   %mul_13 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_10, %add_4), kwargs = {})
#   %select_7 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg14_1, 0, 1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_7, 0), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_11, 1), kwargs = {})
#   %mul_14 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_12, %arg2_1), kwargs = {})
#   %add_5 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %mul_14), kwargs = {})
#   %pow_4 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_5, 2), kwargs = {})
#   %mean_2 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [2], True), kwargs = {})
#   %add_6 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_15 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %rsqrt_2), kwargs = {})
#   %mul_16 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %arg15_1), kwargs = {})
#   %mul_17 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, 0.5773502691896258), kwargs = {})
#   %convert_element_type_26 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_17, torch.bfloat16), kwargs = {})
#   %pow_5 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_5, 2), kwargs = {})
#   %mean_3 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [2], True), kwargs = {})
#   %add_7 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_3, 1e-06), kwargs = {})
#   %rsqrt_3 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %mul_20 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %rsqrt_3), kwargs = {})
#   %mul_21 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %arg21_1), kwargs = {})
#   %mul_22 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, 0.5773502691896258), kwargs = {})
#   %convert_element_type_37 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_22, torch.bfloat16), kwargs = {})
#   return %mul_13,%buf27,%buf48,%mul_17,%convert_element_type_37,%convert_element_type_26
triton_red_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_7 = async_compile.triton('triton_red_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 524288, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 15, 'num_store': 4, 'num_reduction': 2, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 4569612288}}
)
@triton.jit
def triton_red_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 350000
    r0_numel = 384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr5 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr7 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr0 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 * tmp2
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 * tmp6
        tmp8 = tmp3 + tmp7
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp9 * tmp11
        tmp13 = tmp8 + tmp12
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 * tmp16
        tmp18 = tmp13 + tmp17
        tmp19 = tmp0 * tmp18
        tmp21 = tmp20 * tmp6
        tmp22 = tmp19 + tmp21
        tmp23 = tmp22 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask & xmask, tmp26, _tmp25)
        tl.store(out_ptr0 + (r0_1 + 384*x0), tmp19, r0_mask & xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp27 = tl.load(out_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr0 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.load(in_ptr3 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp39 = tl.load(in_ptr8 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp43 = tl.load(in_ptr9 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp28 * tmp30
        tmp32 = tmp27 + tmp31
        tmp33 = tl.full([1, 1], 384.0, tl.float32)
        tmp34 = (tmp25 / tmp33)
        tmp35 = tl.full([1, 1], 1e-06, tl.float32)
        tmp36 = tmp34 + tmp35
        tmp37 = libdevice.rsqrt(tmp36)
        tmp38 = tmp32 * tmp37
        tmp40 = tmp38 * tmp39
        tmp41 = tl.full([1, 1], 0.5773502691896258, tl.float32)
        tmp42 = tmp40 * tmp41
        tmp44 = tmp38 * tmp43
        tmp45 = tmp44 * tmp41
        tmp46 = tmp45.to(tl.float32)
        tmp47 = tmp42.to(tl.float32)
        tl.store(out_ptr3 + (r0_1 + 384*x0), tmp42, r0_mask & xmask)
        tl.store(out_ptr4 + (r0_1 + 384*x0), tmp46, r0_mask & xmask)
        tl.store(out_ptr5 + (r0_1 + 384*x0), tmp47, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/xa/cxa232pywvvbhnwc6ummrfz7kihwiyc32yq2me4o7zbf544suztj.py
# Topologically Sorted Source Nodes: [getitem_13, getitem_14, mul_9, add_3, getitem_22, getitem_23, getitem_20, linear_7, mul_14, add_4, getitem_21, linear_9, mul_15, add_5, mul_16, getitem_24, getitem_25, mul_17, add_6, rms_norm_4, mul_18, linear_10, rms_norm_5, mul_21, linear_13], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._unsafe_view, aten._fused_rms_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add_3 => add_5
#   add_4 => add_8
#   add_5 => add_9
#   add_6 => add_10
#   getitem_13 => select_7
#   getitem_14 => unsqueeze_11, unsqueeze_12
#   getitem_20 => unsqueeze_14, unsqueeze_15
#   getitem_21 => unsqueeze_16, unsqueeze_17
#   getitem_22 => select_12
#   getitem_23 => unsqueeze_18, unsqueeze_19
#   getitem_24 => select_13
#   getitem_25 => unsqueeze_20, unsqueeze_21
#   linear_10 => convert_element_type_50
#   linear_13 => convert_element_type_61
#   linear_7 => view_19
#   linear_9 => view_23
#   mul_14 => mul_24
#   mul_15 => mul_25
#   mul_16 => mul_26
#   mul_17 => mul_27
#   mul_18 => mul_30
#   mul_21 => mul_35
#   mul_9 => mul_14
#   rms_norm_4 => add_11, mean_4, mul_28, mul_29, pow_7, rsqrt_4
#   rms_norm_5 => add_12, mean_5, mul_33, mul_34, pow_8, rsqrt_5
# Graph fragment:
#   %arg24_1 : Tensor "f32[2, 384][384, 1]cuda:4" = PlaceHolder[target=arg24_1]
#   %mul_13 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4" = PlaceHolder[target=mul_13]
#   %arg14_1 : Tensor "f32[2, 384][384, 1]cuda:4" = PlaceHolder[target=arg14_1]
#   %arg2_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:4" = PlaceHolder[target=arg2_1]
#   %arg19_1 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=arg19_1]
#   %mm_7 : Tensor "bf16[350000, 384][384, 1]cuda:4" = PlaceHolder[target=mm_7]
#   %arg20_1 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=arg20_1]
#   %mm_9 : Tensor "bf16[350000, 384][384, 1]cuda:4" = PlaceHolder[target=mm_9]
#   %add_10 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4" = PlaceHolder[target=add_10]
#   %buf54 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:4" = PlaceHolder[target=buf54]
#   %arg25_1 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=arg25_1]
#   %buf68 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:4" = PlaceHolder[target=buf68]
#   %arg31_1 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=arg31_1]
#   %select_7 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg14_1, 0, 1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_7, 0), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_11, 1), kwargs = {})
#   %mul_14 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_12, %arg2_1), kwargs = {})
#   %add_5 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %mul_14), kwargs = {})
#   %select_12 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg24_1, 0, 0), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_12, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 1), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg19_1, 0), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, 1), kwargs = {})
#   %view_19 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [350, 1000, 384]), kwargs = {})
#   %mul_24 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_15, %view_19), kwargs = {})
#   %add_8 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_24), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg20_1, 0), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 1), kwargs = {})
#   %view_23 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [350, 1000, 384]), kwargs = {})
#   %mul_25 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_17, %view_23), kwargs = {})
#   %add_9 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %mul_25), kwargs = {})
#   %mul_26 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_19, %add_9), kwargs = {})
#   %select_13 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg24_1, 0, 1), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_13, 0), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, 1), kwargs = {})
#   %mul_27 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_21, %arg2_1), kwargs = {})
#   %add_10 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_27), kwargs = {})
#   %pow_7 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_10, 2), kwargs = {})
#   %mean_4 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [2], True), kwargs = {})
#   %add_11 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_4, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_28 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %rsqrt_4), kwargs = {})
#   %mul_29 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %arg25_1), kwargs = {})
#   %mul_30 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, 0.5), kwargs = {})
#   %convert_element_type_50 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_30, torch.bfloat16), kwargs = {})
#   %pow_8 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_10, 2), kwargs = {})
#   %mean_5 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [2], True), kwargs = {})
#   %add_12 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_5, 1e-06), kwargs = {})
#   %rsqrt_5 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_33 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %rsqrt_5), kwargs = {})
#   %mul_34 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %arg31_1), kwargs = {})
#   %mul_35 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, 0.5), kwargs = {})
#   %convert_element_type_61 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_35, torch.bfloat16), kwargs = {})
#   return %add_10,%buf54,%buf68,%convert_element_type_50,%convert_element_type_61
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_8 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 11, 'num_store': 4, 'num_reduction': 2, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2800000, 'r0_': 3494410752}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 350000
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr4 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr6 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr7 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.load(in_ptr8 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp6 = tmp1 + tmp5
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp11 = tmp6 + tmp10
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp16 = tmp11 + tmp15
    tmp17 = tmp0 * tmp16
    tmp19 = tmp18 * tmp4
    tmp20 = tmp17 + tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(r0_mask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
    tmp26 = tl.full([1, 1], 384.0, tl.float32)
    tmp27 = (tmp25 / tmp26)
    tmp28 = tl.full([1, 1], 1e-06, tl.float32)
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp20 * tmp30
    tmp33 = tmp31 * tmp32
    tmp34 = tl.full([1, 1], 0.5, tl.float32)
    tmp35 = tmp33 * tmp34
    tmp36 = tmp35.to(tl.float32)
    tmp38 = tmp31 * tmp37
    tmp39 = tmp38 * tmp34
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp20, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 384*x0), tmp36, r0_mask & xmask)
    tl.store(out_ptr3 + (r0_1 + 384*x0), tmp40, r0_mask & xmask)
    tl.store(out_ptr0 + (x0), tmp25, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/tg/ctg334polo32vdyy5foefhf4aimjlpeyei2pvhn42mrdn3iyzruf.py
# Topologically Sorted Source Nodes: [rms_norm_4, mul_18, getitem_29, contiguous_22, linear_11], Original ATen: [aten._fused_rms_norm, aten.mul, aten.slice, aten.clone, aten._to_copy]
# Source node to ATen node mapping:
#   contiguous_22 => clone_5
#   getitem_29 => slice_6
#   linear_11 => convert_element_type_54
#   mul_18 => mul_30
#   rms_norm_4 => add_11, mean_4, mul_28, mul_29, pow_7, rsqrt_4
# Graph fragment:
#   %add_10 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4" = PlaceHolder[target=add_10]
#   %buf54 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:4" = PlaceHolder[target=buf54]
#   %arg25_1 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=arg25_1]
#   %pow_7 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_10, 2), kwargs = {})
#   %mean_4 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [2], True), kwargs = {})
#   %add_11 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_4, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_28 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %rsqrt_4), kwargs = {})
#   %mul_29 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %arg25_1), kwargs = {})
#   %mul_30 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, 0.5), kwargs = {})
#   %slice_6 : Tensor "f32[350, 1000, 12][384000, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_30, 2, 0, 12), kwargs = {})
#   %clone_5 : Tensor "f32[350, 1000, 12][12000, 12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_6,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_54 : Tensor "bf16[350, 1000, 12][12000, 12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_5, torch.bfloat16), kwargs = {})
#   return %convert_element_type_54
triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_9 = async_compile.triton('triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 33600048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4200000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 12)
    x3 = xindex // 12
    x2 = xindex // 12000
    x4 = (xindex % 12000)
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([1], 384.0, tl.float32)
    tmp3 = (tmp1 / tmp2)
    tmp4 = tl.full([1], 1e-06, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.rsqrt(tmp5)
    tmp7 = tmp0 * tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tl.full([1], 0.5, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x4 + 12032*x2), tmp12, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/du/cducdzkyesxzl3t3w6fcykbn36dfdpmg3pfj4s72pjvdskjs353v.py
# Topologically Sorted Source Nodes: [linear_13, leaky_relu_2, square_2], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow]
# Source node to ATen node mapping:
#   leaky_relu_2 => convert_element_type_64, gt_2, mul_36, where_2
#   linear_13 => view_33
#   square_2 => pow_9
# Graph fragment:
#   %mm_13 : Tensor "bf16[350000, 1536][1536, 1]cuda:4" = PlaceHolder[target=mm_13]
#   %view_33 : Tensor "bf16[350, 1000, 1536][1536000, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_13, [350, 1000, 1536]), kwargs = {})
#   %convert_element_type_64 : Tensor "f32[350, 1000, 1536][1536000, 1536, 1]cuda:4"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_33, torch.float32), kwargs = {})
#   %gt_2 : Tensor "b8[350, 1000, 1536][1536000, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_64, 0), kwargs = {})
#   %mul_36 : Tensor "f32[350, 1000, 1536][1536000, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_64, 0.5), kwargs = {})
#   %where_2 : Tensor "f32[350, 1000, 1536][1536000, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %convert_element_type_64, %mul_36), kwargs = {})
#   %pow_9 : Tensor "f32[350, 1000, 1536][1536000, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_2, 2), kwargs = {})
#   return %pow_9
triton_poi_fused__unsafe_view_leaky_relu_pow_10 = async_compile.triton('triton_poi_fused__unsafe_view_leaky_relu_pow_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_leaky_relu_pow_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 5376000000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_leaky_relu_pow_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 537600000
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
    tl.store(out_ptr0 + (x0), tmp7, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1 = args
        args.clear()
        assert_size_stride(arg0_1, (2, 384), (384, 1))
        assert_size_stride(arg1_1, (350, 1000, 384), (384000, 384, 1))
        assert_size_stride(arg2_1, (350, 1000, 384), (384000, 384, 1))
        assert_size_stride(arg3_1, (384, ), (1, ))
        assert_size_stride(arg4_1, (1152, 384), (384, 1))
        assert_size_stride(arg5_1, (1, 1000, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg6_1, (1, 1000, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg7_1, (8, 12), (12, 1))
        assert_size_stride(arg8_1, (384, 384), (384, 1))
        assert_size_stride(arg9_1, (384, ), (1, ))
        assert_size_stride(arg10_1, (384, ), (1, ))
        assert_size_stride(arg11_1, (384, ), (1, ))
        assert_size_stride(arg12_1, (1536, 384), (384, 1))
        assert_size_stride(arg13_1, (384, 1536), (1536, 1))
        assert_size_stride(arg14_1, (2, 384), (384, 1))
        assert_size_stride(arg15_1, (384, ), (1, ))
        assert_size_stride(arg16_1, (1152, 384), (384, 1))
        assert_size_stride(arg17_1, (8, 12), (12, 1))
        assert_size_stride(arg18_1, (384, 384), (384, 1))
        assert_size_stride(arg19_1, (384, ), (1, ))
        assert_size_stride(arg20_1, (384, ), (1, ))
        assert_size_stride(arg21_1, (384, ), (1, ))
        assert_size_stride(arg22_1, (1536, 384), (384, 1))
        assert_size_stride(arg23_1, (384, 1536), (1536, 1))
        assert_size_stride(arg24_1, (2, 384), (384, 1))
        assert_size_stride(arg25_1, (384, ), (1, ))
        assert_size_stride(arg26_1, (1152, 384), (384, 1))
        assert_size_stride(arg27_1, (8, 12), (12, 1))
        assert_size_stride(arg28_1, (384, 384), (384, 1))
        assert_size_stride(arg29_1, (384, ), (1, ))
        assert_size_stride(arg30_1, (384, ), (1, ))
        assert_size_stride(arg31_1, (384, ), (1, ))
        assert_size_stride(arg32_1, (1536, 384), (384, 1))
        with torch.cuda._DeviceGuard(4):
            torch.cuda.set_device(4)
            buf17 = empty_strided_cuda((8, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_0.run(arg7_1, buf17, 96, stream=stream4)
            del arg7_1
            buf44 = empty_strided_cuda((8, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_0.run(arg17_1, buf44, 96, stream=stream4)
            del arg17_1
            buf74 = empty_strided_cuda((8, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_0.run(arg27_1, buf74, 96, stream=stream4)
            del arg27_1
            buf4 = empty_strided_cuda((350, 1000, 8, 48), (384000, 384, 48, 1), torch.bfloat16)
            buf5 = empty_strided_cuda((350, 1000, 8, 48), (384000, 384, 48, 1), torch.bfloat16)
            buf31 = empty_strided_cuda((350, 1000, 8, 48), (384000, 384, 48, 1), torch.bfloat16)
            buf32 = empty_strided_cuda((350, 1000, 8, 48), (384000, 384, 48, 1), torch.bfloat16)
            buf57 = empty_strided_cuda((350, 1000, 8, 48), (384000, 384, 48, 1), torch.bfloat16)
            buf58 = empty_strided_cuda((350, 1000, 8, 48), (384000, 384, 48, 1), torch.bfloat16)
            buf1 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.float32)
            buf22 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            buf2 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, getitem_1, mul, getitem_2, getitem_3, mul_1, add, rms_norm, mul_2, linear, rms_norm_1, mul_5, linear_3], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._fused_rms_norm, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_red_fused__fused_rms_norm__to_copy_add_mul_select_unsqueeze_1.run(arg0_1, arg1_1, arg2_1, arg3_1, arg11_1, buf1, buf22, buf2, 350000, 384, stream=stream4)
            del arg11_1
            del arg3_1
            buf15 = empty_strided_cuda((350, 1000, 12), (12032, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_7, contiguous_6, linear_1], Original ATen: [aten.slice, aten.clone, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_clone_slice_2.run(buf1, buf15, 4200000, stream=stream4)
            del buf1
            buf16 = empty_strided_cuda((350000, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_7, contiguous_6, linear_1], Original ATen: [aten.slice, aten.clone, aten._to_copy, aten.view, aten.t, aten.mm]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_clone_mm_slice_t_view_3.run(buf15, buf16, 4200000, stream=stream4)
            buf18 = empty_strided_cuda((350000, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_7, contiguous_6, linear_1], Original ATen: [aten.slice, aten.clone, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf16, reinterpret_tensor(buf17, (12, 8), (1, 12), 0), out=buf18)
            del buf17
            buf3 = empty_strided_cuda((350000, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf2, (350000, 384), (384, 1), 0), reinterpret_tensor(arg4_1, (384, 1152), (1, 384), 0), out=buf3)
            del arg4_1
            # Topologically Sorted Source Nodes: [linear, view, getitem_5, getitem_6, triton_kernel_wrapper_mutation], Original ATen: [aten._unsafe_view, aten.view, aten.select]
            stream4 = get_raw_stream(4)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf3, (350, 1000, 24, 48), (1152000, 1152, 48, 1), 0), buf4, buf5, reinterpret_tensor(arg5_1, (1000, 24), (24, 1), 0), reinterpret_tensor(arg6_1, (1000, 24), (24, 1), 0), 5600000, 1, 1, stream=stream4)
            buf8 = reinterpret_tensor(buf2, (350, 8, 1000, 48), (384000, 48, 384, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [linear, view, scaled_dot_product_attention, getitem_4, contiguous, transpose_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
            stream4 = get_raw_stream(4)
            triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_4.run(buf3, buf8, 134400000, stream=stream4)
            del buf3
            # Topologically Sorted Source Nodes: [linear, view, scaled_dot_product_attention, getitem_4, contiguous, transpose_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
            buf9 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf4, (350, 8, 1000, 48), (384000, 48, 384, 1), 0), reinterpret_tensor(buf5, (350, 8, 1000, 48), (384000, 48, 384, 1), 0), buf8, 0.0, True, scale=0.14433756729740646)
            del buf4
            del buf5
            buf10 = buf9[0]
            assert_size_stride(buf10, (350, 8, 1000, 48), (384000, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf10, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf9
            buf19 = reinterpret_tensor(buf10, (350, 1000, 8, 48), (384000, 384, 48, 1), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, linear_1, mul_3, sigmoid, getitem_8, mul_4], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
            stream4 = get_raw_stream(4)
            triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5.run(buf19, buf18, 134400000, stream=stream4)
            buf20 = reinterpret_tensor(buf8, (350000, 384), (384, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, linear_1, mul_3, sigmoid, getitem_8, mul_4, reshape, linear_2], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf19, (350000, 384), (384, 1), 0), reinterpret_tensor(arg8_1, (384, 384), (1, 384), 0), out=buf20)
            del arg8_1
            del buf19
            buf23 = empty_strided_cuda((350000, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, getitem_1, mul, getitem_2, getitem_3, mul_1, add, rms_norm_1, mul_5, linear_3], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._fused_rms_norm, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf22, (350000, 384), (384, 1), 0), reinterpret_tensor(arg12_1, (384, 1536), (1, 384), 0), out=buf23)
            del arg12_1
            buf24 = reinterpret_tensor(buf23, (350, 1000, 1536), (1536000, 1536, 1), 0); del buf23  # reuse
            # Topologically Sorted Source Nodes: [linear_3, leaky_relu, square, linear_4], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_6.run(buf24, 537600000, stream=stream4)
            buf25 = reinterpret_tensor(buf22, (350000, 384), (384, 1), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [linear_3, leaky_relu, square, linear_4], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf24, (350000, 1536), (1536, 1), 0), reinterpret_tensor(arg13_1, (1536, 384), (1, 1536), 0), out=buf25)
            del arg13_1
            del buf24
            buf26 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.float32)
            buf28 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.float32)
            buf49 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            buf29 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, getitem_1, mul, getitem_2, getitem_3, mul_1, add, getitem_11, getitem_12, getitem_9, linear_2, mul_6, add_1, getitem_10, linear_4, mul_7, add_2, mul_8, getitem_13, getitem_14, mul_9, add_3, rms_norm_2, mul_10, linear_5, rms_norm_3, mul_13, linear_8], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._unsafe_view, aten._fused_rms_norm, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_red_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_7.run(arg14_1, arg0_1, arg1_1, arg2_1, arg9_1, buf20, arg10_1, buf25, arg15_1, arg21_1, buf26, buf28, buf49, buf29, 350000, 384, stream=stream4)
            del arg0_1
            del arg10_1
            del arg15_1
            del arg1_1
            del arg21_1
            del arg9_1
            del buf20
            del buf25
            buf42 = buf15; del buf15  # reuse
            # Topologically Sorted Source Nodes: [getitem_18, contiguous_14, linear_6], Original ATen: [aten.slice, aten.clone, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_clone_slice_2.run(buf28, buf42, 4200000, stream=stream4)
            del buf28
            buf43 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [getitem_18, contiguous_14, linear_6], Original ATen: [aten.slice, aten.clone, aten._to_copy, aten.view, aten.t, aten.mm]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_clone_mm_slice_t_view_3.run(buf42, buf43, 4200000, stream=stream4)
            buf45 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [getitem_18, contiguous_14, linear_6], Original ATen: [aten.slice, aten.clone, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf43, reinterpret_tensor(buf44, (12, 8), (1, 12), 0), out=buf45)
            del buf44
            buf30 = empty_strided_cuda((350000, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf29, (350000, 384), (384, 1), 0), reinterpret_tensor(arg16_1, (384, 1152), (1, 384), 0), out=buf30)
            del arg16_1
            # Topologically Sorted Source Nodes: [linear_5, view_1, getitem_16, getitem_17, triton_kernel_wrapper_mutation_1], Original ATen: [aten._unsafe_view, aten.view, aten.select]
            stream4 = get_raw_stream(4)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf30, (350, 1000, 24, 48), (1152000, 1152, 48, 1), 0), buf31, buf32, reinterpret_tensor(arg5_1, (1000, 24), (24, 1), 0), reinterpret_tensor(arg6_1, (1000, 24), (24, 1), 0), 5600000, 1, 1, stream=stream4)
            buf35 = reinterpret_tensor(buf29, (350, 8, 1000, 48), (384000, 48, 384, 1), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [linear_5, view_1, scaled_dot_product_attention_1, getitem_15, contiguous_8, transpose_6], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
            stream4 = get_raw_stream(4)
            triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_4.run(buf30, buf35, 134400000, stream=stream4)
            del buf30
            # Topologically Sorted Source Nodes: [linear_5, view_1, scaled_dot_product_attention_1, getitem_15, contiguous_8, transpose_6], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
            buf36 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf31, (350, 8, 1000, 48), (384000, 48, 384, 1), 0), reinterpret_tensor(buf32, (350, 8, 1000, 48), (384000, 48, 384, 1), 0), buf35, 0.0, True, scale=0.14433756729740646)
            del buf31
            del buf32
            buf37 = buf36[0]
            assert_size_stride(buf37, (350, 8, 1000, 48), (384000, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf37, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf36
            buf46 = reinterpret_tensor(buf37, (350, 1000, 8, 48), (384000, 384, 48, 1), 0); del buf37  # reuse
            # Topologically Sorted Source Nodes: [transpose_7, linear_6, mul_11, sigmoid_1, getitem_19, mul_12], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
            stream4 = get_raw_stream(4)
            triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5.run(buf46, buf45, 134400000, stream=stream4)
            buf47 = reinterpret_tensor(buf35, (350000, 384), (384, 1), 0); del buf35  # reuse
            # Topologically Sorted Source Nodes: [transpose_7, linear_6, mul_11, sigmoid_1, getitem_19, mul_12, reshape_1, linear_7], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf46, (350000, 384), (384, 1), 0), reinterpret_tensor(arg18_1, (384, 384), (1, 384), 0), out=buf47)
            del arg18_1
            del buf46
            buf50 = empty_strided_cuda((350000, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_13, getitem_14, mul_9, add_3, rms_norm_3, mul_13, linear_8], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._fused_rms_norm, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf49, (350000, 384), (384, 1), 0), reinterpret_tensor(arg22_1, (384, 1536), (1, 384), 0), out=buf50)
            del arg22_1
            buf51 = reinterpret_tensor(buf50, (350, 1000, 1536), (1536000, 1536, 1), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [linear_8, leaky_relu_1, square_1, linear_9], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_6.run(buf51, 537600000, stream=stream4)
            buf52 = reinterpret_tensor(buf49, (350000, 384), (384, 1), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [linear_8, leaky_relu_1, square_1, linear_9], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf51, (350000, 1536), (1536, 1), 0), reinterpret_tensor(arg23_1, (1536, 384), (1, 1536), 0), out=buf52)
            del arg23_1
            buf53 = buf26; del buf26  # reuse
            buf54 = empty_strided_cuda((350, 1000, 1), (1000, 1, 350016), torch.float32)
            buf55 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            buf69 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_13, getitem_14, mul_9, add_3, getitem_22, getitem_23, getitem_20, linear_7, mul_14, add_4, getitem_21, linear_9, mul_15, add_5, mul_16, getitem_24, getitem_25, mul_17, add_6, rms_norm_4, mul_18, linear_10, rms_norm_5, mul_21, linear_13], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._unsafe_view, aten._fused_rms_norm, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_8.run(buf53, arg24_1, arg14_1, arg2_1, arg19_1, buf47, arg20_1, buf52, arg25_1, arg31_1, buf54, buf55, buf69, 350000, 384, stream=stream4)
            del arg14_1
            del arg19_1
            del arg20_1
            del arg24_1
            del arg2_1
            del arg31_1
            del buf47
            del buf52
            buf72 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_4, mul_18, getitem_29, contiguous_22, linear_11], Original ATen: [aten._fused_rms_norm, aten.mul, aten.slice, aten.clone, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_9.run(buf53, buf54, arg25_1, buf72, 4200000, stream=stream4)
            del arg25_1
            del buf54
            buf73 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_4, mul_18, getitem_29, contiguous_22, linear_11], Original ATen: [aten._fused_rms_norm, aten.mul, aten.slice, aten.clone, aten._to_copy, aten.view, aten.t, aten.mm]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_clone_mm_slice_t_view_3.run(buf72, buf73, 4200000, stream=stream4)
            del buf72
            buf75 = buf45; del buf45  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_4, mul_18, getitem_29, contiguous_22, linear_11], Original ATen: [aten._fused_rms_norm, aten.mul, aten.slice, aten.clone, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf73, reinterpret_tensor(buf74, (12, 8), (1, 12), 0), out=buf75)
            del buf73
            del buf74
            buf56 = empty_strided_cuda((350000, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_4, mul_18, linear_10], Original ATen: [aten._fused_rms_norm, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf55, (350000, 384), (384, 1), 0), reinterpret_tensor(arg26_1, (384, 1152), (1, 384), 0), out=buf56)
            del arg26_1
            # Topologically Sorted Source Nodes: [linear_10, view_2, getitem_27, getitem_28, triton_kernel_wrapper_mutation_2], Original ATen: [aten._unsafe_view, aten.view, aten.select]
            stream4 = get_raw_stream(4)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf56, (350, 1000, 24, 48), (1152000, 1152, 48, 1), 0), buf57, buf58, reinterpret_tensor(arg5_1, (1000, 24), (24, 1), 0), reinterpret_tensor(arg6_1, (1000, 24), (24, 1), 0), 5600000, 1, 1, stream=stream4)
            del arg5_1
            del arg6_1
            buf61 = reinterpret_tensor(buf55, (350, 8, 1000, 48), (384000, 48, 384, 1), 0); del buf55  # reuse
            # Topologically Sorted Source Nodes: [linear_10, view_2, scaled_dot_product_attention_2, getitem_26, contiguous_16, transpose_10], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
            stream4 = get_raw_stream(4)
            triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_4.run(buf56, buf61, 134400000, stream=stream4)
            del buf56
            # Topologically Sorted Source Nodes: [linear_10, view_2, scaled_dot_product_attention_2, getitem_26, contiguous_16, transpose_10], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
            buf62 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf57, (350, 8, 1000, 48), (384000, 48, 384, 1), 0), reinterpret_tensor(buf58, (350, 8, 1000, 48), (384000, 48, 384, 1), 0), buf61, 0.0, True, scale=0.14433756729740646)
            del buf57
            del buf58
            buf63 = buf62[0]
            assert_size_stride(buf63, (350, 8, 1000, 48), (384000, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf63, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf62
            buf76 = reinterpret_tensor(buf63, (350, 1000, 8, 48), (384000, 384, 48, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [transpose_11, linear_11, mul_19, sigmoid_2, getitem_30, mul_20], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
            stream4 = get_raw_stream(4)
            triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5.run(buf76, buf75, 134400000, stream=stream4)
            del buf75
            buf77 = reinterpret_tensor(buf61, (350000, 384), (384, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [transpose_11, linear_11, mul_19, sigmoid_2, getitem_30, mul_20, reshape_2, linear_12], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf76, (350000, 384), (384, 1), 0), reinterpret_tensor(arg28_1, (384, 384), (1, 384), 0), out=buf77)
            del arg28_1
            del buf76
            buf70 = reinterpret_tensor(buf51, (350000, 1536), (1536, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_5, mul_21, linear_13], Original ATen: [aten._fused_rms_norm, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf69, (350000, 384), (384, 1), 0), reinterpret_tensor(arg32_1, (384, 1536), (1, 384), 0), out=buf70)
            del arg32_1
            del buf69
            buf71 = empty_strided_cuda((350, 1000, 1536), (1536000, 1536, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_13, leaky_relu_2, square_2], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow]
            stream4 = get_raw_stream(4)
            triton_poi_fused__unsafe_view_leaky_relu_pow_10.run(buf70, buf71, 537600000, stream=stream4)
            del buf70
        return (buf71, reinterpret_tensor(arg29_1, (1, 1, 384), (384, 384, 1), 0), reinterpret_tensor(buf77, (350, 1000, 384), (384000, 384, 1), 0), buf53, reinterpret_tensor(arg30_1, (1, 1, 384), (384, 384, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2, 384), (384, 1), device='cuda:4', dtype=torch.float32)
    arg1_1 = rand_strided((350, 1000, 384), (384000, 384, 1), device='cuda:4', dtype=torch.float32)
    arg2_1 = rand_strided((350, 1000, 384), (384000, 384, 1), device='cuda:4', dtype=torch.bfloat16)
    arg3_1 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    arg4_1 = rand_strided((1152, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    arg5_1 = rand_strided((1, 1000, 1, 24), (24576, 24, 24, 1), device='cuda:4', dtype=torch.bfloat16)
    arg6_1 = rand_strided((1, 1000, 1, 24), (24576, 24, 24, 1), device='cuda:4', dtype=torch.bfloat16)
    arg7_1 = rand_strided((8, 12), (12, 1), device='cuda:4', dtype=torch.float32)
    arg8_1 = rand_strided((384, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    arg9_1 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    arg10_1 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    arg11_1 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    arg12_1 = rand_strided((1536, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    arg13_1 = rand_strided((384, 1536), (1536, 1), device='cuda:4', dtype=torch.bfloat16)
    arg14_1 = rand_strided((2, 384), (384, 1), device='cuda:4', dtype=torch.float32)
    arg15_1 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    arg16_1 = rand_strided((1152, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    arg17_1 = rand_strided((8, 12), (12, 1), device='cuda:4', dtype=torch.float32)
    arg18_1 = rand_strided((384, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    arg19_1 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    arg20_1 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    arg21_1 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    arg22_1 = rand_strided((1536, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    arg23_1 = rand_strided((384, 1536), (1536, 1), device='cuda:4', dtype=torch.bfloat16)
    arg24_1 = rand_strided((2, 384), (384, 1), device='cuda:4', dtype=torch.float32)
    arg25_1 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    arg26_1 = rand_strided((1152, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    arg27_1 = rand_strided((8, 12), (12, 1), device='cuda:4', dtype=torch.float32)
    arg28_1 = rand_strided((384, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    arg29_1 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    arg30_1 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    arg31_1 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    arg32_1 = rand_strided((1536, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
