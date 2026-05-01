# AOT ID: ['1_backward']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/du/cdu6ett37mt72nslbhbdrbrydklbockdoghpmye3nxbd42eenui5.py
# Topologically Sorted Source Nodes: [convert_element_type_25, to_11, convert_element_type_27, mul_12, rms_norm_1, mul_14, sum_2, div, mul_15, sub_1, mul_16, mul_17, sum_3, convert_element_type_28, add_5], Original ATen: [aten._fused_rms_norm_backward, aten._to_copy, aten._fused_rms_norm, aten.add]
# Source node to ATen node mapping:
#   add_5 => add_5
#   convert_element_type_25 => convert_element_type_25
#   convert_element_type_27 => convert_element_type_27
#   convert_element_type_28 => convert_element_type_28
#   div => div
#   mul_12 => mul_12
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   rms_norm_1 => convert_element_type_23, mul_10
#   sub_1 => sub_1
#   sum_2 => sum_2
#   sum_3 => sum_3
#   to_11 => convert_element_type_22
# Graph fragment:
#   %add_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1" = PlaceHolder[target=add_3]
#   %rsqrt_1 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:1" = PlaceHolder[target=rsqrt_1]
#   %tangents_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1" = PlaceHolder[target=tangents_1]
#   %primals_16 : Tensor "f32[384][1]cuda:1" = PlaceHolder[target=primals_16]
#   %tangents_2 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1" = PlaceHolder[target=tangents_2]
#   %sum_2 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:1" = PlaceHolder[target=sum_2]
#   %convert_element_type_25 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.float32), kwargs = {})
#   %convert_element_type_22 : Tensor "bf16[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_16, torch.bfloat16), kwargs = {})
#   %convert_element_type_27 : Tensor "f32[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_22, torch.float32), kwargs = {})
#   %mul_12 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_25, %convert_element_type_27), kwargs = {})
#   %convert_element_type_23 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float32), kwargs = {})
#   %mul_10 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_23, %rsqrt_1), kwargs = {})
#   %mul_14 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %mul_12), kwargs = {})
#   %sum_2 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_14, [2], True), kwargs = {})
#   %div : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_10, 384), kwargs = {})
#   %mul_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sum_2), kwargs = {})
#   %sub_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_12, %mul_15), kwargs = {})
#   %mul_16 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_17 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_25, %mul_10), kwargs = {})
#   %sum_3 : Tensor "f32[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_17, [0, 1]), kwargs = {})
#   %convert_element_type_28 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_16, torch.bfloat16), kwargs = {})
#   %add_5 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %convert_element_type_28), kwargs = {})
#   return %sum_2,%add_5
triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_0 = async_compile.triton('triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 0, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
    xnumel = 131072
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * RSPLIT_SIZE
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    accum0 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        x0 = xindex
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr4 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp5 * tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = tl.where(r0_mask, tmp11, 0)
        tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
        tmp16 = tl.full([1, 1], 0.0026041666666666665, tl.float32)
        tmp17 = tmp3 * tmp16
        tmp18 = tmp17 * tmp14
        tmp19 = tmp9 - tmp18
        tmp20 = tmp19 * tmp2
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp15 + tmp21
        tmp23 = tmp5 * tmp3
        tl.store(out_ptr1 + (r0_1 + 384*x0), tmp22, r0_mask)
        tmp24 = tl.sum(tmp23, 0)
        tmp25 = accum0 + tmp24
        accum0 = tmp25
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/zj/czjc267y2xtd6xqfceukesgovvsfd5zfwmvek43bazaj6visi2a2.py
# Topologically Sorted Source Nodes: [view_14, view_15, transpose_3, mul_18, linear_3, mul_6, sigmoid_2, getitem_7, mul_19, sum_4, convert_element_type_35, squeeze, convert_element_type_36, convert_element_type_37, sub_2, mul_20, mul_21, convert_element_type_38, mul_22, permute_19, _scaled_dot_product_flash_attention_backward], Original ATen: [aten.view, aten.transpose, aten.mul, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten._scaled_dot_product_flash_attention_backward]
# Source node to ATen node mapping:
#   _scaled_dot_product_flash_attention_backward => _scaled_dot_product_flash_attention_backward
#   convert_element_type_35 => convert_element_type_35
#   convert_element_type_36 => convert_element_type_36
#   convert_element_type_37 => convert_element_type_37
#   convert_element_type_38 => convert_element_type_38
#   getitem_7 => unsqueeze_3
#   linear_3 => view_8
#   mul_18 => mul_18
#   mul_19 => mul_19
#   mul_20 => mul_20
#   mul_21 => mul_21
#   mul_22 => mul_22
#   mul_6 => mul_8
#   permute_19 => permute_19
#   sigmoid_2 => sigmoid_2
#   squeeze => squeeze
#   sub_2 => sub_2
#   sum_4 => sum_4
#   transpose_3 => permute_8
#   view_14 => view_14
#   view_15 => view_15
# Graph fragment:
#   %mm_5 : Tensor "bf16[131072, 384][384, 1]cuda:1" = PlaceHolder[target=mm_5]
#   %getitem_2 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:1" = PlaceHolder[target=getitem_2]
#   %mm_2 : Tensor "bf16[131072, 8][8, 1]cuda:1" = PlaceHolder[target=mm_2]
#   %sum_4 : Tensor "f32[128, 1024, 8, 1][8192, 8, 1, 1048576]cuda:1" = PlaceHolder[target=sum_4]
#   %view_14 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [128, 1024, 384]), kwargs = {})
#   %view_15 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_14, [128, 1024, 8, 48]), kwargs = {})
#   %permute_8 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [0, 2, 1, 3]), kwargs = {})
#   %mul_18 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, %permute_8), kwargs = {})
#   %view_8 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [128, 1024, 8]), kwargs = {})
#   %mul_8 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, 0.5), kwargs = {})
#   %sigmoid_2 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_8,), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[128, 1024, 8, 1][8192, 8, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_2, 3), kwargs = {})
#   %mul_19 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, %unsqueeze_3), kwargs = {})
#   %sum_4 : Tensor "f32[128, 1024, 8, 1][8192, 8, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_18, [3], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_35 : Tensor "bf16[128, 1024, 8, 1][8192, 8, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_4, torch.bfloat16), kwargs = {})
#   %squeeze : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%convert_element_type_35, 3), kwargs = {})
#   %convert_element_type_36 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%squeeze, torch.float32), kwargs = {})
#   %convert_element_type_37 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_2, torch.float32), kwargs = {})
#   %sub_2 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_37), kwargs = {})
#   %mul_20 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_37, %sub_2), kwargs = {})
#   %mul_21 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_36, %mul_20), kwargs = {})
#   %convert_element_type_38 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_21, torch.bfloat16), kwargs = {})
#   %mul_22 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_38, 0.5), kwargs = {})
#   %permute_19 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_19, [0, 2, 1, 3]), kwargs = {})
#   %_scaled_dot_product_flash_attention_backward : [num_users=3] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention_backward.default](args = (%permute_19, %permute_6, %permute_7, %permute_5, %getitem_2, %getitem_3, None, None, 1024, 1024, 0.0, True, %getitem_8, %getitem_9), kwargs = {scale: 0.14433756729740646})
#   return %sum_4,%buf12,%mul_22
triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_transpose_unsqueeze_view_1 = async_compile.triton('triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_transpose_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1048576, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_transpose_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6291456, 'r0_': 402653184}}
)
@triton.jit
def triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_transpose_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    r0_numel = 48
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 48*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 48*x0), r0_mask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 0.5, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tl.sigmoid(tmp10)
    tmp12 = tmp0 * tmp11
    tmp13 = tmp7.to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11.to(tl.float32)
    tmp16 = tl.full([1, 1], 1.0, tl.float32)
    tmp17 = tmp16 - tmp15
    tmp18 = tmp15 * tmp17
    tmp19 = tmp14 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp9
    tl.store(out_ptr1 + (r0_1 + 48*x0), tmp12, r0_mask)
    tl.store(out_ptr2 + (x0), tmp21, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/mj/cmjqsdtur6errc2cv5agpoqs2x6hlrvybfhznfoqbeptwxxho4fa.py
# Topologically Sorted Source Nodes: [constant_pad_nd_default], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   constant_pad_nd_default => constant_pad_nd_default
# Graph fragment:
#   %view_7 : Tensor "bf16[131072, 12][12, 1]cuda:1" = PlaceHolder[target=view_7]
#   %constant_pad_nd_default : Tensor "bf16[131072, 16][16, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_7, [0, 4, 0, 0]), kwargs = {})
#   return %constant_pad_nd_default
triton_poi_fused_mm_2 = async_compile.triton('triton_poi_fused_mm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 11534336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 12, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 12*x1), tmp2, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/yz/cyzog5cbejzbm735j5axxrnyk3wsqusvqshtye7skgza4lnjch67.py
# Topologically Sorted Source Nodes: [slice_tensor, convert_element_type_43], Original ATen: [aten.mm, aten._to_copy]
# Source node to ATen node mapping:
#   convert_element_type_43 => convert_element_type_43
#   slice_tensor => slice_tensor
# Graph fragment:
#   %mm_default : Tensor "bf16[8, 16][16, 1]cuda:1" = PlaceHolder[target=mm_default]
#   %slice_tensor : Tensor "bf16[8, 12][16, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_default, 1, 0, -4), kwargs = {})
#   %convert_element_type_43 : Tensor "f32[8, 12][12, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_tensor, torch.float32), kwargs = {})
#   return %convert_element_type_43
triton_poi_fused__to_copy_mm_3 = async_compile.triton('triton_poi_fused__to_copy_mm_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mm_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 12)
    x1 = xindex // 12
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ve/cveefqzlqzfrs4i5m4n3gkau65nf35zx2fugf7roepbkg6yigucw.py
# Topologically Sorted Source Nodes: [permute_20, permute_21, permute_22, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
# Source node to ATen node mapping:
#   copy_ => copy
#   getitem => slice_5
#   permute_20 => permute_20
#   permute_21 => permute_21
#   permute_22 => permute_22
#   triton_kernel_wrapper_mutation => triton_kernel_wrapper_mutation
# Graph fragment:
#   %getitem_13 : Tensor "bf16[128, 4, 1024, 48][196608, 48, 192, 1]cuda:1" = PlaceHolder[target=getitem_13]
#   %permute_20 : Tensor "bf16[128, 1024, 4, 48][196608, 192, 48, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_13, [0, 2, 1, 3]), kwargs = {})
#   %permute_21 : Tensor "bf16[128, 1024, 4, 48][196608, 192, 48, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_12, [0, 2, 1, 3]), kwargs = {})
#   %permute_22 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_11, [0, 2, 1, 3]), kwargs = {})
#   %slice_5 : Tensor "bf16[128, 1024, 4, 48][786432, 768, 48, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%empty_2, 2, 12, 9223372036854775807), kwargs = {})
#   %copy : Tensor "bf16[128, 1024, 4, 48][786432, 768, 48, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_5, %permute_20), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[128, 1024, 16, 48][786432, 768, 48, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty_2, %copy, 2, 12, 9223372036854775807), kwargs = {})
#   %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 12, constant_args_idx: 9, grid: [(1572864, 1, 1)], tma_descriptor_metadata: {}, kwargs: {DQ: %permute_22, DK: %permute_21, QKV: %view_6, DQKV: %slice_scatter_default_1, COS: %select_1, SIN: %select_3}})
#   return %buf18
triton_poi_fused_copy_slice_transpose_4 = async_compile.triton('triton_poi_fused_copy_slice_transpose_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_transpose_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 452984832}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_slice_transpose_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100663296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 48) % 16)
    x2 = xindex // 768
    x3 = (xindex % 768)
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 12, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-576) + x3 + 192*x2), tmp2, other=0.0).to(tl.float32)
    tmp4 = tl.full([1], float("nan"), tl.float32)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x4), tmp5, None)
''', device_str='cuda')


# Original path: /workspace/parameter-golf/train_gpt_parcae.py:4480
_fused_qkv_postprocess_bwd_kernel_0 = async_compile.triton('_fused_qkv_postprocess_bwd_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 2, 'num_stages': 3}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_fused_qkv_postprocess_bwd_kernel_0', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'DQ': '*bf16', 'DK': '*bf16', 'QKV': '*bf16', 'DQKV': '*bf16', 'COS': '*bf16', 'SIN': '*bf16', 'TOTAL_ROWS': 'constexpr', 'SEQLEN': 'constexpr', 'H_Q': 'constexpr', 'H_KV': 'constexpr', 'HEAD_DIM': 'constexpr', 'ROPE_DIMS': 'constexpr', 'DO_QK_NORM': 'constexpr', 'ROPE_INTERLEAVED': 'constexpr', 'RMS_EPS': 'constexpr', 'BLOCK_D': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'TOTAL_ROWS': 131072, 'SEQLEN': 1024, 'H_Q': 8, 'H_KV': 4, 'HEAD_DIM': 48, 'ROPE_DIMS': 48, 'DO_QK_NORM': True, 'ROPE_INTERLEAVED': False, 'RMS_EPS': 1e-06, 'BLOCK_D': 64}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _fused_qkv_postprocess_bwd_kernel(
    DQ,
    DK,
    QKV,
    DQKV,
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
    q_base = (row * H_Q + q_head_safe) * HEAD_DIM
    k_base = (row * H_KV + k_head_safe) * HEAD_DIM
    gq = tl.load(DQ + q_base + offs, mask=mask & is_q, other=0.0).to(tl.float32)
    gk = tl.load(DK + k_base + offs, mask=mask & ~is_q, other=0.0).to(tl.float32)
    g = gq + gk

    x = tl.load(QKV + qkv_base + offs, mask=mask, other=0.0).to(tl.float32)
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
    z_rot = tl.where(is_first, x * cos - mate * sin, mate * sin + x * cos)
    z = tl.where(rope_mask, z_rot, x)
    if DO_QK_NORM:
        ss = tl.sum(tl.where(mask, z * z, 0.0), axis=0)
        inv_rms = tl.rsqrt(ss / HEAD_DIM + RMS_EPS)
        y = z * inv_rms
        mean_gy = tl.sum(tl.where(mask, g * y, 0.0), axis=0) / HEAD_DIM
        g = inv_rms * (g - y * mean_gy)

    mate_gq = tl.load(DQ + q_base + mate_offs, mask=mask & rope_mask & is_q, other=0.0).to(tl.float32)
    mate_gk = tl.load(DK + k_base + mate_offs, mask=mask & rope_mask & ~is_q, other=0.0).to(tl.float32)
    mate_g = mate_gq + mate_gk
    if DO_QK_NORM:
        mate_x = mate
        mate_mate = x
        mate_z_rot = tl.where(is_first, mate_mate * sin + mate_x * cos, mate_x * cos - mate_mate * sin)
        mate_z = tl.where(rope_mask, mate_z_rot, mate_x)
        mate_y = mate_z * inv_rms
        mate_g = inv_rms * (mate_g - mate_y * mean_gy)
    dx_rot = tl.where(is_first, g * cos + mate_g * sin, -mate_g * sin + g * cos)
    dx = tl.where(rope_mask, dx_rot, g)
    tl.store(DQKV + qkv_base + offs, dx, mask=mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/qx/cqx35lkbwobrlf4dmorw53gpckj4tydp6wasum46n2m7pbmaqspy.py
# Topologically Sorted Source Nodes: [add_6, view_17, full_default_1, view_25, add_7, convert_element_type_48, to_7, convert_element_type_50, mul_23, rms_norm, mul_25, sum_5, div_1, mul_26, sub_3, mul_27, mul_28, sum_6, convert_element_type_51, add_8], Original ATen: [aten.add, aten.view, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten._fused_rms_norm]
# Source node to ATen node mapping:
#   add_6 => add_6
#   add_7 => add_7
#   add_8 => add_8
#   convert_element_type_48 => convert_element_type_48
#   convert_element_type_50 => convert_element_type_50
#   convert_element_type_51 => convert_element_type_51
#   div_1 => div_1
#   full_default_1 => full_default_1
#   mul_23 => mul_23
#   mul_25 => mul_25
#   mul_26 => mul_26
#   mul_27 => mul_27
#   mul_28 => mul_28
#   rms_norm => convert_element_type_13, mul_6
#   sub_3 => sub_3
#   sum_5 => sum_5
#   sum_6 => sum_6
#   to_7 => convert_element_type_12
#   view_17 => view_17
#   view_25 => view_25
# Graph fragment:
#   %add_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1" = PlaceHolder[target=add_1]
#   %rsqrt : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:1" = PlaceHolder[target=rsqrt]
#   %mm_7 : Tensor "bf16[131072, 12][12, 1]cuda:1" = PlaceHolder[target=mm_7]
#   %mm_9 : Tensor "bf16[131072, 384][384, 1]cuda:1" = PlaceHolder[target=mm_9]
#   %primals_10 : Tensor "f32[384][1]cuda:1" = PlaceHolder[target=primals_10]
#   %tangents_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1" = PlaceHolder[target=tangents_3]
#   %add_5 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1" = PlaceHolder[target=add_5]
#   %sum_5 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:1" = PlaceHolder[target=sum_5]
#   %add_6 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_3, %add_5), kwargs = {})
#   %view_17 : Tensor "bf16[128, 1024, 12][12288, 12, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [128, 1024, 12]), kwargs = {})
#   %full_default_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([128, 1024, 384], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %slice_scatter_default : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_1, %view_17, 2, 0, 12), kwargs = {})
#   %view_25 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [128, 1024, 384]), kwargs = {})
#   %add_7 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %view_25), kwargs = {})
#   %convert_element_type_48 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_7, torch.float32), kwargs = {})
#   %convert_element_type_12 : Tensor "bf16[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_10, torch.bfloat16), kwargs = {})
#   %convert_element_type_50 : Tensor "f32[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_12, torch.float32), kwargs = {})
#   %mul_23 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_48, %convert_element_type_50), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %mul_6 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_13, %rsqrt), kwargs = {})
#   %mul_25 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %mul_23), kwargs = {})
#   %sum_5 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_25, [2], True), kwargs = {})
#   %div_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_6, 384), kwargs = {})
#   %mul_26 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sum_5), kwargs = {})
#   %sub_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_23, %mul_26), kwargs = {})
#   %mul_27 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt), kwargs = {})
#   %mul_28 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_48, %mul_6), kwargs = {})
#   %sum_6 : Tensor "f32[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_28, [0, 1]), kwargs = {})
#   %convert_element_type_51 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_27, torch.bfloat16), kwargs = {})
#   %add_8 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %convert_element_type_51), kwargs = {})
#   return %sum_5,%add_8
triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_slice_backward_view_5 = async_compile.triton('triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_slice_backward_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_slice_backward_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 0, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_slice_backward_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
    xnumel = 131072
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * RSPLIT_SIZE
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    accum0 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        x0 = xindex
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr3 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr5 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp23 = tl.load(in_out_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = r0_1
        tmp5 = tl.full([1, 1], 12, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr2 + (r0_1 + 12*x0), r0_mask & tmp6, other=0.0).to(tl.float32)
        tmp8 = tl.full([1, 1], 0.0, tl.float32)
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp11 = tmp9 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp12 * tmp15
        tmp17 = tmp3 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = tl.where(r0_mask, tmp18, 0)
        tmp21 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
        tmp24 = tmp22 + tmp23
        tmp25 = tl.full([1, 1], 0.0026041666666666665, tl.float32)
        tmp26 = tmp3 * tmp25
        tmp27 = tmp26 * tmp21
        tmp28 = tmp16 - tmp27
        tmp29 = tmp28 * tmp2
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp24 + tmp30
        tmp32 = tmp12 * tmp3
        tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp31, r0_mask)
        tmp33 = tl.sum(tmp32, 0)
        tmp34 = accum0 + tmp33
        accum0 = tmp34
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/nz/cnz4dl4kih4utqlfsafjwacpq4fzrk5aihtmfvzaizphkb2jp3uq.py
# Topologically Sorted Source Nodes: [full_default_1, to_5, sigmoid_1, getitem, mul_29, to_6, unsqueeze, sub, mul_31, linear, linear_1, sigmoid, to_4, mul_33, slice_8, add_10, mul_37, mul_38, mul_39, convert_element_type_62, convert_element_type_63, sub_5, mul_40, mul_41, convert_element_type_64], Original ATen: [aten.slice_backward, aten._to_copy, aten.sigmoid, aten.unsqueeze, aten.mul, aten.rsub, aten._unsafe_view, aten.view, aten.slice, aten.add, aten.sigmoid_backward]
# Source node to ATen node mapping:
#   add_10 => add_10
#   convert_element_type_62 => convert_element_type_62
#   convert_element_type_63 => convert_element_type_63
#   convert_element_type_64 => convert_element_type_64
#   full_default_1 => full_default_1
#   getitem => unsqueeze, unsqueeze_1
#   linear => view_1
#   linear_1 => view_3
#   mul_29 => mul_29
#   mul_31 => mul_31
#   mul_33 => mul_33
#   mul_37 => mul_37
#   mul_38 => mul_38
#   mul_39 => mul_39
#   mul_40 => mul_40
#   mul_41 => mul_41
#   sigmoid => sigmoid
#   sigmoid_1 => sigmoid_1
#   slice_8 => slice_8
#   sub => sub
#   sub_5 => sub_5
#   to_4 => convert_element_type_9
#   to_5 => convert_element_type_10
#   to_6 => convert_element_type_11
#   unsqueeze => unsqueeze_2
# Graph fragment:
#   %add_8 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1" = PlaceHolder[target=add_8]
#   %primals_8 : Tensor "f32[384][1]cuda:1" = PlaceHolder[target=primals_8]
#   %ne : Tensor "b8[128, 1024][1024, 1]cuda:1" = PlaceHolder[target=ne]
#   %primals_7 : Tensor "f32[][]cuda:1" = PlaceHolder[target=primals_7]
#   %mul_37 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1" = PlaceHolder[target=mul_37]
#   %mm : Tensor "bf16[131072, 384][384, 1]cuda:1" = PlaceHolder[target=mm]
#   %addmm : Tensor "bf16[131072, 384][384, 1]cuda:1" = PlaceHolder[target=addmm]
#   %full_default_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([128, 1024, 384], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %convert_element_type_10 : Tensor "bf16[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.bfloat16), kwargs = {})
#   %sigmoid_1 : Tensor "bf16[384][1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_10,), kwargs = {})
#   %unsqueeze : Tensor "bf16[1, 384][384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %mul_29 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %unsqueeze_1), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[128, 1024][1024, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ne, torch.bfloat16), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[128, 1024, 1][1024, 1, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_11, -1), kwargs = {})
#   %sub : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %unsqueeze_1), kwargs = {})
#   %mul_31 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %sub), kwargs = {})
#   %view_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 384]), kwargs = {})
#   %view_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [128, 1024, 384]), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[][]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_7, torch.bfloat16), kwargs = {})
#   %mul_33 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %unsqueeze_2), kwargs = {})
#   %slice_8 : Tensor "bf16[128, 1023, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_33, 1, 1, 1024), kwargs = {})
#   %slice_scatter_default_2 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_1, %slice_8, 1, 0, -1), kwargs = {})
#   %add_10 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %slice_scatter_default_2), kwargs = {})
#   %mul_37 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %convert_element_type_9), kwargs = {})
#   %mul_38 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %view_1), kwargs = {})
#   %mul_39 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %sigmoid), kwargs = {})
#   %convert_element_type_62 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_38, torch.float32), kwargs = {})
#   %convert_element_type_63 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid, torch.float32), kwargs = {})
#   %sub_5 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_63), kwargs = {})
#   %mul_40 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_63, %sub_5), kwargs = {})
#   %mul_41 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_62, %mul_40), kwargs = {})
#   %convert_element_type_64 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_41, torch.bfloat16), kwargs = {})
#   return %mul_37,%convert_element_type_64,%mul_39
triton_poi_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_sigmoid_backward_slice_slice_backward_unsqueeze_view_6 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_sigmoid_backward_slice_slice_backward_unsqueeze_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_sigmoid_backward_slice_slice_backward_unsqueeze_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 805307904}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_sigmoid_backward_slice_slice_backward_unsqueeze_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 1024)
    x4 = xindex // 384
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr3 + (0))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp27 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp30 = tl.load(in_ptr5 + (x3), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp4 - tmp3
    tmp6 = tmp0 * tmp5
    tmp7 = x1
    tmp8 = tl.full([1], 1023, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tl.load(in_ptr0 + (384 + x3), tmp9, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (x0), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp10 * tmp13
    tmp15 = tl.load(in_ptr2 + (1 + x4), tmp9, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp9, tmp17, tmp18)
    tmp20 = tl.full([1], 0.0, tl.float32)
    tmp21 = tl.where(tmp9, tmp19, tmp20)
    tmp22 = tmp6 + tmp21
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp22 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp31 = tl.sigmoid(tmp30)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp4 - tmp32
    tmp34 = tmp32 * tmp33
    tmp35 = tmp29 * tmp34
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp26 * tmp31
    tl.store(out_ptr1 + (x3), tmp36, None)
    tl.store(out_ptr2 + (x3), tmp37, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/h3/ch3qmvzmqqmn6cojv3giv4slyqkryyfe5366k5xlaguugshmcxgq.py
# Topologically Sorted Source Nodes: [full_default_1, to_5, sigmoid_1, getitem, mul_29, to_6, unsqueeze, mul_3, mul_30, sum_7, sub, mul_31, linear, linear_1, sigmoid, mul_1, to_4, mul_2, add, mul_32, sum_8, mul_33, slice_8, add_10, mul_36, sum_9, view_28, add_11], Original ATen: [aten.slice_backward, aten._to_copy, aten.sigmoid, aten.unsqueeze, aten.mul, aten.sum, aten.rsub, aten._unsafe_view, aten.view, aten.add, aten.slice]
# Source node to ATen node mapping:
#   add => add
#   add_10 => add_10
#   add_11 => add_11
#   full_default_1 => full_default_1
#   getitem => unsqueeze, unsqueeze_1
#   linear => view_1
#   linear_1 => view_3
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_29 => mul_29
#   mul_3 => mul_3
#   mul_30 => mul_30
#   mul_31 => mul_31
#   mul_32 => mul_32
#   mul_33 => mul_33
#   mul_36 => mul_36
#   sigmoid => sigmoid
#   sigmoid_1 => sigmoid_1
#   slice_8 => slice_8
#   sub => sub
#   sum_7 => sum_7
#   sum_8 => sum_8
#   sum_9 => sum_9
#   to_4 => convert_element_type_9
#   to_5 => convert_element_type_10
#   to_6 => convert_element_type_11
#   unsqueeze => unsqueeze_2
#   view_28 => view_28
# Graph fragment:
#   %add_8 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1" = PlaceHolder[target=add_8]
#   %primals_8 : Tensor "f32[384][1]cuda:1" = PlaceHolder[target=primals_8]
#   %ne : Tensor "b8[128, 1024][1024, 1]cuda:1" = PlaceHolder[target=ne]
#   %mm : Tensor "bf16[131072, 384][384, 1]cuda:1" = PlaceHolder[target=mm]
#   %addmm : Tensor "bf16[131072, 384][384, 1]cuda:1" = PlaceHolder[target=addmm]
#   %mm_10 : Tensor "bf16[131072, 384][384, 1]cuda:1" = PlaceHolder[target=mm_10]
#   %cat : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1" = PlaceHolder[target=cat]
#   %primals_6 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1" = PlaceHolder[target=primals_6]
#   %primals_7 : Tensor "f32[][]cuda:1" = PlaceHolder[target=primals_7]
#   %full_default_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([128, 1024, 384], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %convert_element_type_10 : Tensor "bf16[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.bfloat16), kwargs = {})
#   %sigmoid_1 : Tensor "bf16[384][1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_10,), kwargs = {})
#   %unsqueeze : Tensor "bf16[1, 384][384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %mul_29 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %unsqueeze_1), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[128, 1024][1024, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ne, torch.bfloat16), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[128, 1024, 1][1024, 1, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_11, -1), kwargs = {})
#   %mul_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_2), kwargs = {})
#   %mul_30 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %mul_3), kwargs = {})
#   %sum_7 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_30, [0, 1], True), kwargs = {dtype: torch.float32})
#   %sub : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %unsqueeze_1), kwargs = {})
#   %mul_31 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %sub), kwargs = {})
#   %view_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 384]), kwargs = {})
#   %view_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [128, 1024, 384]), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_3,), kwargs = {})
#   %mul_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %sigmoid), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[][]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_7, torch.bfloat16), kwargs = {})
#   %mul_2 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %convert_element_type_9), kwargs = {})
#   %add : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_6, %mul_2), kwargs = {})
#   %mul_32 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %add), kwargs = {})
#   %sum_8 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_32, [0, 1], True), kwargs = {dtype: torch.float32})
#   %mul_33 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %unsqueeze_2), kwargs = {})
#   %slice_8 : Tensor "bf16[128, 1023, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_33, 1, 1, 1024), kwargs = {})
#   %slice_scatter_default_2 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_1, %slice_8, 1, 0, -1), kwargs = {})
#   %add_10 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %slice_scatter_default_2), kwargs = {})
#   %mul_36 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %mul_1), kwargs = {})
#   %sum_9 : Tensor "f32[][]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_36,), kwargs = {dtype: torch.float32})
#   %view_28 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_10, [128, 1024, 384]), kwargs = {})
#   %add_11 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_28), kwargs = {})
#   return %buf32,%add_11
triton_per_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_slice_slice_backward_sum_unsqueeze_view_7 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_slice_slice_backward_sum_unsqueeze_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*i1', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_slice_slice_backward_sum_unsqueeze_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 12, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_slice_slice_backward_sum_unsqueeze_view_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
    xnumel = 131072
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * RSPLIT_SIZE
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    accum0 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    accum1 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        x3 = xindex
        x0 = (xindex % 1024)
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_2 + 384*x3), r0_mask, other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr3 + (r0_2 + 384*x3), r0_mask, other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr4 + (r0_2 + 384*x3), r0_mask, other=0.0).to(tl.float32)
        tmp33 = tl.load(in_out_ptr0 + (r0_2 + 384*x3), r0_mask, other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr5 + (r0_2 + 384*x3), r0_mask, other=0.0).to(tl.float32)
        tmp36 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last').to(tl.int1)
        tmp41 = tl.load(in_ptr6 + (r0_2 + 384*x3), r0_mask, other=0.0).to(tl.float32)
        tmp42 = tl.load(in_ptr7 + (0))
        tmp43 = tl.broadcast_to(tmp42, [1, 1])
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.sigmoid(tmp2)
        tmp4 = tl.full([1, 1], 1.0, tl.float32)
        tmp5 = tmp4 - tmp3
        tmp6 = tmp0 * tmp5
        tmp7 = x0
        tmp8 = tl.full([1, 1], 1023, tl.int64)
        tmp9 = tmp7 < tmp8
        tmp10 = tl.load(in_ptr0 + (384 + r0_2 + 384*x3), r0_mask & tmp9, other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.sigmoid(tmp12)
        tmp14 = tmp10 * tmp13
        tmp15 = tl.load(in_ptr2 + (tl.broadcast_to(1 + x3, [XBLOCK, R0_BLOCK])), r0_mask & tmp9, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 * tmp16
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp9, tmp17, tmp18)
        tmp20 = tl.full([1, 1], 0.0, tl.float32)
        tmp21 = tl.where(tmp9, tmp19, tmp20)
        tmp22 = tmp6 + tmp21
        tmp25 = tl.sigmoid(tmp24)
        tmp26 = tmp23 * tmp25
        tmp27 = tmp22 * tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
        tmp31 = tl.where(r0_mask, tmp29, 0)
        tmp32 = tl.sum(tmp31, 1)[:, None].to(tl.float32)
        tmp34 = tmp22 + tmp33
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 * tmp37
        tmp39 = tmp0 * tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp44 = tmp43.to(tl.float32)
        tmp45 = tmp26 * tmp44
        tmp46 = tmp41 + tmp45
        tmp47 = tmp0 * tmp46
        tmp48 = tmp47.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_2 + 384*x3), tmp34, r0_mask)
        tl.store(out_ptr0 + (x3), tmp32, None)
        tmp49 = tl.sum(tmp40, 0)
        tmp50 = accum0 + tmp49
        accum0 = tmp50
        tmp51 = tl.sum(tmp48, 0)
        tmp52 = accum1 + tmp51
        accum1 = tmp52
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/kn/cknxqfyivtz5wzt6ichd7eictubk6hrt7sa7dk7kpo2bonmi5zsa.py
# Topologically Sorted Source Nodes: [to_5, sigmoid_1, convert_element_type_54, convert_element_type_55, neg, add_9, squeeze_1, squeeze_2, convert_element_type_56, convert_element_type_57, sub_4, mul_34, mul_35], Original ATen: [aten._to_copy, aten.sigmoid, aten.neg, aten.add, aten.squeeze, aten.sigmoid_backward]
# Source node to ATen node mapping:
#   add_9 => add_9
#   convert_element_type_54 => convert_element_type_54
#   convert_element_type_55 => convert_element_type_55
#   convert_element_type_56 => convert_element_type_56
#   convert_element_type_57 => convert_element_type_57
#   mul_34 => mul_34
#   mul_35 => mul_35
#   neg => neg
#   sigmoid_1 => sigmoid_1
#   squeeze_1 => squeeze_1
#   squeeze_2 => squeeze_2
#   sub_4 => sub_4
#   to_5 => convert_element_type_10
# Graph fragment:
#   %sum_7 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:1" = PlaceHolder[target=sum_7]
#   %sum_8 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:1" = PlaceHolder[target=sum_8]
#   %primals_8 : Tensor "f32[384][1]cuda:1" = PlaceHolder[target=primals_8]
#   %convert_element_type_10 : Tensor "bf16[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.bfloat16), kwargs = {})
#   %sigmoid_1 : Tensor "bf16[384][1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_10,), kwargs = {})
#   %convert_element_type_54 : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_7, torch.bfloat16), kwargs = {})
#   %convert_element_type_55 : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_8, torch.bfloat16), kwargs = {})
#   %neg : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_55,), kwargs = {})
#   %add_9 : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_54, %neg), kwargs = {})
#   %squeeze_1 : Tensor "bf16[1, 384][384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%add_9, 1), kwargs = {})
#   %squeeze_2 : Tensor "bf16[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%squeeze_1, 0), kwargs = {})
#   %convert_element_type_56 : Tensor "f32[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%squeeze_2, torch.float32), kwargs = {})
#   %convert_element_type_57 : Tensor "f32[384][1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_1, torch.float32), kwargs = {})
#   %sub_4 : Tensor "f32[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_57), kwargs = {})
#   %mul_34 : Tensor "f32[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_57, %sub_4), kwargs = {})
#   %mul_35 : Tensor "f32[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_56, %mul_34), kwargs = {})
#   return %mul_35
triton_poi_fused__to_copy_add_neg_sigmoid_sigmoid_backward_squeeze_8 = async_compile.triton('triton_poi_fused__to_copy_add_neg_sigmoid_sigmoid_backward_squeeze_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_neg_sigmoid_sigmoid_backward_squeeze_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 7680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_neg_sigmoid_sigmoid_backward_squeeze_8(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr0 + (x0), xmask)
    tmp7 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = -tmp3
    tmp5 = tmp1 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.full([1], 1.0, tl.float32)
    tmp12 = tmp11 - tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp6 * tmp13
    tl.store(in_out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ze/czeieu2zmoufzzufv466kbp4wxedcjxleacgfvbiroxkkv4knw6r.py
# Topologically Sorted Source Nodes: [full_default_1, to_5, sigmoid_1, getitem, mul_29, to_6, unsqueeze, sub, mul_31, linear, linear_1, sigmoid, mul_1, mul_33, slice_8, add_10, mul_36, sum_9], Original ATen: [aten.slice_backward, aten._to_copy, aten.sigmoid, aten.unsqueeze, aten.mul, aten.rsub, aten._unsafe_view, aten.view, aten.slice, aten.add, aten.sum]
# Source node to ATen node mapping:
#   add_10 => add_10
#   full_default_1 => full_default_1
#   getitem => unsqueeze, unsqueeze_1
#   linear => view_1
#   linear_1 => view_3
#   mul_1 => mul_1
#   mul_29 => mul_29
#   mul_31 => mul_31
#   mul_33 => mul_33
#   mul_36 => mul_36
#   sigmoid => sigmoid
#   sigmoid_1 => sigmoid_1
#   slice_8 => slice_8
#   sub => sub
#   sum_9 => sum_9
#   to_5 => convert_element_type_10
#   to_6 => convert_element_type_11
#   unsqueeze => unsqueeze_2
# Graph fragment:
#   %buf32 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:1" = PlaceHolder[target=buf32]
#   %full_default_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([128, 1024, 384], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %convert_element_type_10 : Tensor "bf16[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.bfloat16), kwargs = {})
#   %sigmoid_1 : Tensor "bf16[384][1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_10,), kwargs = {})
#   %unsqueeze : Tensor "bf16[1, 384][384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %mul_29 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %unsqueeze_1), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[128, 1024][1024, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ne, torch.bfloat16), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[128, 1024, 1][1024, 1, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_11, -1), kwargs = {})
#   %sub : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %unsqueeze_1), kwargs = {})
#   %mul_31 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %sub), kwargs = {})
#   %view_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 384]), kwargs = {})
#   %view_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [128, 1024, 384]), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_3,), kwargs = {})
#   %mul_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %sigmoid), kwargs = {})
#   %mul_33 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %unsqueeze_2), kwargs = {})
#   %slice_8 : Tensor "bf16[128, 1023, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_33, 1, 1, 1024), kwargs = {})
#   %slice_scatter_default_2 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_1, %slice_8, 1, 0, -1), kwargs = {})
#   %add_10 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %slice_scatter_default_2), kwargs = {})
#   %mul_36 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %mul_1), kwargs = {})
#   %sum_9 : Tensor "f32[][]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_36,), kwargs = {dtype: torch.float32})
#   return %sum_9
triton_red_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_slice_slice_backward_sum_unsqueeze_view_9 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_slice_slice_backward_sum_unsqueeze_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_slice_slice_backward_sum_unsqueeze_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 524288}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_slice_slice_backward_sum_unsqueeze_view_9(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 131072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), None, eviction_policy='evict_first')
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tmp3
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp2, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/mb/cmbwkkmbbpx7aia7vjc6trssiil4223nezx62svfa4fx23ymsdro.py
# Topologically Sorted Source Nodes: [linear, linear_1, sigmoid, mul_38, convert_element_type_62, convert_element_type_63, sub_5, mul_40, mul_41, convert_element_type_64, view_26, sum_10], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.sigmoid_backward, aten.sum]
# Source node to ATen node mapping:
#   convert_element_type_62 => convert_element_type_62
#   convert_element_type_63 => convert_element_type_63
#   convert_element_type_64 => convert_element_type_64
#   linear => view_1
#   linear_1 => view_3
#   mul_38 => mul_38
#   mul_40 => mul_40
#   mul_41 => mul_41
#   sigmoid => sigmoid
#   sub_5 => sub_5
#   sum_10 => sum_10
#   view_26 => view_26
# Graph fragment:
#   %convert_element_type_64 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1" = PlaceHolder[target=convert_element_type_64]
#   %view_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 384]), kwargs = {})
#   %view_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [128, 1024, 384]), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_3,), kwargs = {})
#   %mul_38 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %view_1), kwargs = {})
#   %convert_element_type_62 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_38, torch.float32), kwargs = {})
#   %convert_element_type_63 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid, torch.float32), kwargs = {})
#   %sub_5 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_63), kwargs = {})
#   %mul_40 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_63, %sub_5), kwargs = {})
#   %mul_41 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_62, %mul_40), kwargs = {})
#   %convert_element_type_64 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_41, torch.bfloat16), kwargs = {})
#   %view_26 : Tensor "bf16[131072, 384][384, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_64, [131072, 384]), kwargs = {})
#   %sum_10 : Tensor "f32[1, 384][384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_26, [0], True), kwargs = {dtype: torch.float32})
#   return %buf37
triton_red_fused__unsafe_view_mul_sigmoid_sigmoid_backward_sum_view_10 = async_compile.triton('triton_red_fused__unsafe_view_mul_sigmoid_sigmoid_backward_sum_view_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_mul_sigmoid_sigmoid_backward_sum_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 101735424, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_mul_sigmoid_sigmoid_backward_sum_view_10(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 376*x1
        tmp1 = tl.full([1, 1], 131072, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 384*r0_2 + 144384*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/u7/cu7r3yotv5k2lkkaofnlingafuivovmgpc2lqz2v3mfcuwhrlge7.py
# Topologically Sorted Source Nodes: [linear, linear_1, sigmoid, mul_38, convert_element_type_62, convert_element_type_63, sub_5, mul_40, mul_41, convert_element_type_64, view_26, sum_10], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.sigmoid_backward, aten.sum]
# Source node to ATen node mapping:
#   convert_element_type_62 => convert_element_type_62
#   convert_element_type_63 => convert_element_type_63
#   convert_element_type_64 => convert_element_type_64
#   linear => view_1
#   linear_1 => view_3
#   mul_38 => mul_38
#   mul_40 => mul_40
#   mul_41 => mul_41
#   sigmoid => sigmoid
#   sub_5 => sub_5
#   sum_10 => sum_10
#   view_26 => view_26
# Graph fragment:
#   %buf37 : Tensor "f32[1, 384, 349][134016, 1, 384]cuda:1" = PlaceHolder[target=buf37]
#   %view_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 384]), kwargs = {})
#   %view_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [128, 1024, 384]), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_3,), kwargs = {})
#   %mul_38 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %view_1), kwargs = {})
#   %convert_element_type_62 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_38, torch.float32), kwargs = {})
#   %convert_element_type_63 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid, torch.float32), kwargs = {})
#   %sub_5 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_63), kwargs = {})
#   %mul_40 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_63, %sub_5), kwargs = {})
#   %mul_41 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_62, %mul_40), kwargs = {})
#   %convert_element_type_64 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_41, torch.bfloat16), kwargs = {})
#   %view_26 : Tensor "bf16[131072, 384][384, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_64, [131072, 384]), kwargs = {})
#   %sum_10 : Tensor "f32[1, 384][384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_26, [0], True), kwargs = {dtype: torch.float32})
#   return %sum_10
triton_red_fused__unsafe_view_mul_sigmoid_sigmoid_backward_sum_view_11 = async_compile.triton('triton_red_fused__unsafe_view_mul_sigmoid_sigmoid_backward_sum_view_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_mul_sigmoid_sigmoid_backward_sum_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 539136, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_mul_sigmoid_sigmoid_backward_sum_view_11(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/zr/czr722x4tkwx7moro6fpsfkqwqc65a574y2loife6n66agbmsyrd.py
# Topologically Sorted Source Nodes: [full_default_4], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   full_default_4 => full_default_4
# Graph fragment:
#   %full_default_4 : Tensor "f32[131072, 96][96, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([131072, 96], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:1, pin_memory: False})
#   return %index_put
triton_poi_fused_embedding_dense_backward_12 = async_compile.triton('triton_poi_fused_embedding_dense_backward_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_12(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ut/cut555z5melozmrf7mqpwyculmwi2djqfg6zx7obtgnqsed6qgl6.py
# Topologically Sorted Source Nodes: [view_30, convert_element_type_75, mul_42, unsqueeze_4, expand, convert_element_type_default, eq, unsqueeze_5, full_default_3, where, full_default_4, index_put], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.unsqueeze, aten.expand, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   convert_element_type_75 => convert_element_type_75
#   convert_element_type_default => convert_element_type_default
#   eq => eq
#   expand => expand
#   full_default_3 => full_default_3
#   full_default_4 => full_default_4
#   index_put => index_put
#   mul_42 => mul_42
#   unsqueeze_4 => unsqueeze_4
#   unsqueeze_5 => unsqueeze_5
#   view_30 => view_30
#   where => where
# Graph fragment:
#   %primals_1 : Tensor "i64[128, 1024, 8][8192, 8, 1]cuda:1" = PlaceHolder[target=primals_1]
#   %mm_13 : Tensor "bf16[131072, 96][96, 1]cuda:1" = PlaceHolder[target=mm_13]
#   %index_put : Tensor "f32[131072, 96][96, 1]cuda:1" = PlaceHolder[target=index_put]
#   %view_30 : Tensor "bf16[128, 1024, 96][98304, 96, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_13, [128, 1024, 96]), kwargs = {})
#   %convert_element_type_75 : Tensor "f32[128, 1024, 96][98304, 96, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_30, torch.float32), kwargs = {})
#   %mul_42 : Tensor "f32[128, 1024, 96][98304, 96, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_75, 0.3535533905932738), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[128, 1024, 1, 96][98304, 96, 96, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_42, -2), kwargs = {})
#   %expand : Tensor "f32[128, 1024, 8, 96][98304, 96, 0, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [128, 1024, 8, 96]), kwargs = {})
#   %convert_element_type_default : Tensor "f32[128, 1024, 8, 96][786432, 768, 96, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%expand, torch.float32), kwargs = {})
#   %eq : Tensor "b8[128, 1024, 8][8192, 8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "b8[128, 1024, 8, 1][8192, 8, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, -1), kwargs = {})
#   %full_default_3 : Tensor "f32[][]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %where : Tensor "f32[128, 1024, 8, 96][786432, 768, 96, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_5, %full_default_3, %convert_element_type_default), kwargs = {})
#   %full_default_4 : Tensor "f32[131072, 96][96, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([131072, 96], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %index_put : Tensor "f32[131072, 96][96, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_4, [%primals_1], %where, True), kwargs = {})
#   return %buf44
triton_poi_fused__to_copy_embedding_dense_backward_expand_mul_unsqueeze_view_13 = async_compile.triton('triton_poi_fused__to_copy_embedding_dense_backward_expand_mul_unsqueeze_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_embedding_dense_backward_expand_mul_unsqueeze_view_13', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': True, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 8388608, 'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_embedding_dense_backward_expand_mul_unsqueeze_view_13(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
    xnumel = 96
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y3 = yindex
    x2 = xindex
    y1 = yindex // 8
    tmp0 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x2 + 96*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.full([1, 1], 131072, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 131072)) | ~(ymask), "index out of bounds: 0 <= tmp4 < 131072")
    tmp6 = tl.full([1, 1], -1, tl.int64)
    tmp7 = tmp0 == tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.full([1, 1], 0.3535533905932738, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full([1, 1], 0.0, tl.float32)
    tmp13 = tl.where(tmp7, tmp12, tmp11)
    tl.atomic_add(out_ptr0 + (tl.broadcast_to(x2 + 96*tmp4, [YBLOCK, XBLOCK])), tmp13, xmask & ymask, sem='relaxed')
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/b2/cb2euzh6jnfxwtypufg5qbpgjsbb7auiqiz2n6w4qrd3mzvyxc45.py
# Topologically Sorted Source Nodes: [convert_element_type_80], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   convert_element_type_80 => convert_element_type_80
# Graph fragment:
#   %buf44 : Tensor "f32[131072, 96][96, 1]cuda:1" = PlaceHolder[target=buf44]
#   %convert_element_type_80 : Tensor "bf16[131072, 96][96, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%index_put, torch.bfloat16), kwargs = {})
#   return %convert_element_type_80
triton_poi_fused_embedding_dense_backward_14 = async_compile.triton('triton_poi_fused_embedding_dense_backward_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
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
        primals_1, primals_5, primals_6, primals_7, primals_8, primals_10, primals_11, primals_15, primals_16, view, mm, addmm, cat, ne, add_1, rsqrt, view_4, view_6, select_1, select_3, permute_5, permute_6, permute_7, getitem_2, getitem_3, getitem_8, getitem_9, view_7, mm_2, view_10, add_3, rsqrt_1, permute_17, permute_34, tangents_1, tangents_2, tangents_3 = args
        args.clear()
        assert_size_stride(primals_1, (128, 1024, 8), (8192, 8, 1))
        assert_size_stride(primals_5, (384, 384), (384, 1))
        assert_size_stride(primals_6, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(primals_7, (), ())
        assert_size_stride(primals_8, (384, ), (1, ))
        assert_size_stride(primals_10, (384, ), (1, ))
        assert_size_stride(primals_11, (768, 384), (384, 1))
        assert_size_stride(primals_15, (384, 384), (384, 1))
        assert_size_stride(primals_16, (384, ), (1, ))
        assert_size_stride(view, (131072, 96), (96, 1))
        assert_size_stride(mm, (131072, 384), (384, 1))
        assert_size_stride(addmm, (131072, 384), (384, 1))
        assert_size_stride(cat, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(ne, (128, 1024), (1024, 1))
        assert_size_stride(add_1, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(rsqrt, (128, 1024, 1), (1024, 1, 1))
        assert_size_stride(view_4, (131072, 384), (384, 1))
        assert_size_stride(view_6, (128, 1024, 16, 48), (786432, 768, 48, 1))
        assert_size_stride(select_1, (1024, 24), (24, 1))
        assert_size_stride(select_3, (1024, 24), (24, 1))
        assert_size_stride(permute_5, (128, 4, 1024, 48), (196608, 48, 192, 1))
        assert_size_stride(permute_6, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(permute_7, (128, 4, 1024, 48), (196608, 48, 192, 1))
        assert_size_stride(getitem_2, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(getitem_3, (128, 8, 1024), (8192, 1024, 1))
        assert_size_stride(getitem_8, (2, ), (1, ))
        assert_size_stride(getitem_9, (), ())
        assert_size_stride(view_7, (131072, 12), (12, 1))
        assert_size_stride(mm_2, (131072, 8), (8, 1))
        assert_size_stride(view_10, (131072, 384), (384, 1))
        assert_size_stride(add_3, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(rsqrt_1, (128, 1024, 1), (1024, 1, 1))
        assert_size_stride(permute_17, (8, 12), (12, 1))
        assert_size_stride(permute_34, (384, 96), (96, 1))
        assert_size_stride(tangents_1, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(tangents_2, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(tangents_3, (128, 1024, 384), (393216, 384, 1))
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            buf3 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [convert_element_type_25, to_11, convert_element_type_27, mul_12, rms_norm_1, mul_14, sum_2, div, mul_15, sub_1, mul_16, mul_17, sum_3, convert_element_type_28, add_5], Original ATen: [aten._fused_rms_norm_backward, aten._to_copy, aten._fused_rms_norm, aten.add]
            workspace_0 = empty_strided_cuda((786432, ), (1, ), torch.float32)
            stream1 = get_raw_stream(1)
            triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_0.run(add_3, rsqrt_1, tangents_1, primals_16, tangents_2, buf3, workspace_0, 131072, 384, stream=stream1)
            buf2 = workspace_0[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del workspace_0
            del add_3
            del primals_16
            del rsqrt_1
            del tangents_1
            del tangents_2
            buf4 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_13, permute_11, mm_4], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (384, 131072), (1, 384), 0), view_10, out=buf4)
            del view_10
            buf5 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_13, linear_4, permute_13, mm_5], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (131072, 384), (384, 1), 0), primals_15, out=buf5)
            del primals_15
            buf12 = empty_strided_cuda((128, 8, 1024, 48), (393216, 48, 384, 1), torch.bfloat16)
            buf7 = empty_strided_cuda((128, 1024, 8), (8192, 8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_14, view_15, transpose_3, mul_18, linear_3, mul_6, sigmoid_2, getitem_7, mul_19, sum_4, convert_element_type_35, squeeze, convert_element_type_36, convert_element_type_37, sub_2, mul_20, mul_21, convert_element_type_38, mul_22, permute_19, _scaled_dot_product_flash_attention_backward], Original ATen: [aten.view, aten.transpose, aten.mul, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten._scaled_dot_product_flash_attention_backward]
            stream1 = get_raw_stream(1)
            triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_transpose_unsqueeze_view_1.run(buf5, getitem_2, mm_2, buf12, buf7, 1048576, 48, stream=stream1)
            del buf5
            del mm_2
            buf8 = empty_strided_cuda((131072, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [constant_pad_nd_default], Original ATen: [aten.mm]
            stream1 = get_raw_stream(1)
            triton_poi_fused_mm_2.run(view_7, buf8, 2097152, stream=stream1)
            del view_7
            buf9 = empty_strided_cuda((8, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_3, mul_6, sigmoid_2, convert_element_type_35, squeeze, convert_element_type_36, convert_element_type_37, sub_2, mul_20, mul_21, convert_element_type_38, mul_22, view_16, permute_15, constant_pad_nd_default, mm_default], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf7, (8, 131072), (1, 8), 0), buf8, out=buf9)
            del buf8
            buf10 = empty_strided_cuda((131072, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_3, mul_6, sigmoid_2, convert_element_type_35, squeeze, convert_element_type_36, convert_element_type_37, sub_2, mul_20, mul_21, convert_element_type_38, mul_22, view_16, mm_7], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf7, (131072, 8), (8, 1), 0), permute_17, out=buf10)
            del buf7
            del permute_17
            buf11 = empty_strided_cuda((8, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [slice_tensor, convert_element_type_43], Original ATen: [aten.mm, aten._to_copy]
            stream1 = get_raw_stream(1)
            triton_poi_fused__to_copy_mm_3.run(buf9, buf11, 96, stream=stream1)
            del buf9
            # Topologically Sorted Source Nodes: [view_14, view_15, linear_3, mul_6, sigmoid_2, getitem_7, mul_19, permute_19, _scaled_dot_product_flash_attention_backward], Original ATen: [aten.view, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.transpose, aten._scaled_dot_product_flash_attention_backward]
            buf13 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(buf12, permute_6, permute_7, permute_5, getitem_2, getitem_3, None, None, 1024, 1024, 0.0, True, getitem_8, getitem_9, scale=0.14433756729740646)
            del getitem_2
            del getitem_3
            del getitem_8
            del getitem_9
            del permute_5
            del permute_6
            del permute_7
            buf14 = buf13[0]
            assert_size_stride(buf14, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf14, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf15 = buf13[1]
            assert_size_stride(buf15, (128, 4, 1024, 48), (196608, 48, 192, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf15, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf16 = buf13[2]
            assert_size_stride(buf16, (128, 4, 1024, 48), (196608, 48, 192, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf16, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            del buf13
            buf18 = empty_strided_cuda((128, 1024, 16, 48), (786432, 768, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_20, permute_21, permute_22, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream1 = get_raw_stream(1)
            triton_poi_fused_copy_slice_transpose_4.run(buf16, buf18, 100663296, stream=stream1)
            del buf16
            # Topologically Sorted Source Nodes: [permute_20, permute_21, permute_22, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream1 = get_raw_stream(1)
            _fused_qkv_postprocess_bwd_kernel_0.run(reinterpret_tensor(buf14, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), reinterpret_tensor(buf15, (128, 1024, 4, 48), (196608, 192, 48, 1), 0), view_6, buf18, select_1, select_3, 1572864, 1, 1, stream=stream1)
            del buf15
            del select_1
            del select_3
            del view_6
            buf20 = empty_strided_cuda((768, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_21, view_22, permute_24, mm_8], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf18, (768, 131072), (1, 768), 0), view_4, out=buf20)
            del view_4
            buf21 = reinterpret_tensor(buf14, (131072, 384), (384, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [view_21, view_22, linear_2, permute_26, mm_9], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf18, (131072, 768), (768, 1), 0), primals_11, out=buf21)
            del buf18
            del primals_11
            buf25 = buf3; del buf3  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [add_6, view_17, full_default_1, view_25, add_7, convert_element_type_48, to_7, convert_element_type_50, mul_23, rms_norm, mul_25, sum_5, div_1, mul_26, sub_3, mul_27, mul_28, sum_6, convert_element_type_51, add_8], Original ATen: [aten.add, aten.view, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten._fused_rms_norm]
            workspace_1 = empty_strided_cuda((786432, ), (1, ), torch.float32)
            stream1 = get_raw_stream(1)
            triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_slice_backward_view_5.run(buf25, add_1, rsqrt, buf10, buf21, primals_10, tangents_3, workspace_1, 131072, 384, stream=stream1)
            buf24 = workspace_1[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del workspace_1
            del add_1
            del buf10
            del primals_10
            del rsqrt
            del tangents_3
            buf34 = reinterpret_tensor(buf21, (128, 1024, 384), (393216, 384, 1), 0); del buf21  # reuse
            buf40 = reinterpret_tensor(buf12, (128, 1024, 384), (393216, 384, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [full_default_1, to_5, sigmoid_1, getitem, mul_29, to_6, unsqueeze, sub, mul_31, linear, linear_1, sigmoid, to_4, mul_33, slice_8, add_10, mul_37, mul_38, mul_39, convert_element_type_62, convert_element_type_63, sub_5, mul_40, mul_41, convert_element_type_64], Original ATen: [aten.slice_backward, aten._to_copy, aten.sigmoid, aten.unsqueeze, aten.mul, aten.rsub, aten._unsafe_view, aten.view, aten.slice, aten.add, aten.sigmoid_backward]
            stream1 = get_raw_stream(1)
            triton_poi_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_sigmoid_backward_slice_slice_backward_unsqueeze_view_6.run(buf25, primals_8, ne, primals_7, mm, addmm, buf34, buf40, 50331648, stream=stream1)
            buf35 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, linear_1, sigmoid, mul_38, convert_element_type_62, convert_element_type_63, sub_5, mul_40, mul_41, convert_element_type_64, view_26, permute_28, mm_10], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.sigmoid_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf34, (131072, 384), (384, 1), 0), primals_5, out=buf35)
            del primals_5
            buf32 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
            buf39 = reinterpret_tensor(buf35, (128, 1024, 384), (393216, 384, 1), 0); del buf35  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [full_default_1, to_5, sigmoid_1, getitem, mul_29, to_6, unsqueeze, mul_3, mul_30, sum_7, sub, mul_31, linear, linear_1, sigmoid, mul_1, to_4, mul_2, add, mul_32, sum_8, mul_33, slice_8, add_10, mul_36, sum_9, view_28, add_11], Original ATen: [aten.slice_backward, aten._to_copy, aten.sigmoid, aten.unsqueeze, aten.mul, aten.sum, aten.rsub, aten._unsafe_view, aten.view, aten.add, aten.slice]
            workspace_2 = empty_strided_cuda((1572864, ), (1, ), torch.float32)
            stream1 = get_raw_stream(1)
            triton_per_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_slice_slice_backward_sum_unsqueeze_view_7.run(buf39, buf25, primals_8, ne, mm, addmm, cat, primals_6, primals_7, buf32, workspace_2, 131072, 384, stream=stream1)
            buf27 = workspace_2[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            buf29 = workspace_2[1 * 2048 * 384 : (1 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del workspace_2
            del addmm
            del buf25
            del cat
            del mm
            del ne
            del primals_7
            buf30 = reinterpret_tensor(buf27, (384, ), (1, ), 0); del buf27  # reuse
            # Topologically Sorted Source Nodes: [to_5, sigmoid_1, convert_element_type_54, convert_element_type_55, neg, add_9, squeeze_1, squeeze_2, convert_element_type_56, convert_element_type_57, sub_4, mul_34, mul_35], Original ATen: [aten._to_copy, aten.sigmoid, aten.neg, aten.add, aten.squeeze, aten.sigmoid_backward]
            stream1 = get_raw_stream(1)
            triton_poi_fused__to_copy_add_neg_sigmoid_sigmoid_backward_squeeze_8.run(buf30, buf29, primals_8, 384, stream=stream1)
            del primals_8
            buf33 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [full_default_1, to_5, sigmoid_1, getitem, mul_29, to_6, unsqueeze, sub, mul_31, linear, linear_1, sigmoid, mul_1, mul_33, slice_8, add_10, mul_36, sum_9], Original ATen: [aten.slice_backward, aten._to_copy, aten.sigmoid, aten.unsqueeze, aten.mul, aten.rsub, aten._unsafe_view, aten.view, aten.slice, aten.add, aten.sum]
            stream1 = get_raw_stream(1)
            triton_red_fused__to_copy__unsafe_view_add_mul_rsub_sigmoid_slice_slice_backward_sum_unsqueeze_view_9.run(buf32, buf33, 1, 131072, stream=stream1)
            del buf32
            buf36 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, linear_1, sigmoid, mul_38, convert_element_type_62, convert_element_type_63, sub_5, mul_40, mul_41, convert_element_type_64, view_26, permute_29, mm_11], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.sigmoid_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf34, (384, 131072), (1, 384), 0), reinterpret_tensor(primals_6, (131072, 384), (384, 1), 0), out=buf36)
            del primals_6
            buf37 = empty_strided_cuda((1, 384, 349), (134016, 1, 384), torch.float32)
            # Topologically Sorted Source Nodes: [linear, linear_1, sigmoid, mul_38, convert_element_type_62, convert_element_type_63, sub_5, mul_40, mul_41, convert_element_type_64, view_26, sum_10], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.sigmoid_backward, aten.sum]
            stream1 = get_raw_stream(1)
            triton_red_fused__unsafe_view_mul_sigmoid_sigmoid_backward_sum_view_10.run(buf34, buf37, 134016, 376, stream=stream1)
            del buf34
            buf38 = reinterpret_tensor(buf29, (1, 384), (384, 1), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [linear, linear_1, sigmoid, mul_38, convert_element_type_62, convert_element_type_63, sub_5, mul_40, mul_41, convert_element_type_64, view_26, sum_10], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.sigmoid_backward, aten.sum]
            stream1 = get_raw_stream(1)
            triton_red_fused__unsafe_view_mul_sigmoid_sigmoid_backward_sum_view_11.run(buf37, buf38, 384, 349, stream=stream1)
            del buf37
            buf41 = empty_strided_cuda((384, 96), (96, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1, sigmoid, mul_39, view_29, permute_32, mm_12], Original ATen: [aten.view, aten.sigmoid, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (384, 131072), (1, 384), 0), view, out=buf41)
            del view
            buf42 = empty_strided_cuda((131072, 96), (96, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1, sigmoid, mul_39, view_29, mm_13], Original ATen: [aten.view, aten.sigmoid, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (131072, 384), (384, 1), 0), permute_34, out=buf42)
            del buf40
            del permute_34
            buf43 = empty_strided_cuda((131072, 96), (96, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_default_4], Original ATen: [aten.embedding_dense_backward]
            stream1 = get_raw_stream(1)
            triton_poi_fused_embedding_dense_backward_12.run(buf43, 12582912, stream=stream1)
            # Topologically Sorted Source Nodes: [view_30, convert_element_type_75, mul_42, unsqueeze_4, expand, convert_element_type_default, eq, unsqueeze_5, full_default_3, where, full_default_4, index_put], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.unsqueeze, aten.expand, aten.embedding_dense_backward]
            stream1 = get_raw_stream(1)
            triton_poi_fused__to_copy_embedding_dense_backward_expand_mul_unsqueeze_view_13.run(primals_1, buf42, buf43, 1048576, 96, stream=stream1)
            del primals_1
            buf45 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_80], Original ATen: [aten.embedding_dense_backward]
            stream1 = get_raw_stream(1)
            triton_poi_fused_embedding_dense_backward_14.run(buf43, buf45, 12582912, stream=stream1)
            del buf43
        return (None, buf45, buf41, reinterpret_tensor(buf38, (384, ), (1, ), 0), buf36, buf39, buf33, buf30, None, buf24, buf20, None, None, buf11, buf4, buf2, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((128, 1024, 8), (8192, 8, 1), device='cuda:1', dtype=torch.int64)
    primals_5 = rand_strided((384, 384), (384, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_6 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_7 = rand_strided((), (), device='cuda:1', dtype=torch.float32)
    primals_8 = rand_strided((384, ), (1, ), device='cuda:1', dtype=torch.float32)
    primals_10 = rand_strided((384, ), (1, ), device='cuda:1', dtype=torch.float32)
    primals_11 = rand_strided((768, 384), (384, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_15 = rand_strided((384, 384), (384, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_16 = rand_strided((384, ), (1, ), device='cuda:1', dtype=torch.float32)
    view = rand_strided((131072, 96), (96, 1), device='cuda:1', dtype=torch.bfloat16)
    mm = rand_strided((131072, 384), (384, 1), device='cuda:1', dtype=torch.bfloat16)
    addmm = rand_strided((131072, 384), (384, 1), device='cuda:1', dtype=torch.bfloat16)
    cat = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:1', dtype=torch.bfloat16)
    ne = rand_strided((128, 1024), (1024, 1), device='cuda:1', dtype=torch.bool)
    add_1 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:1', dtype=torch.bfloat16)
    rsqrt = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:1', dtype=torch.float32)
    view_4 = rand_strided((131072, 384), (384, 1), device='cuda:1', dtype=torch.bfloat16)
    view_6 = rand_strided((128, 1024, 16, 48), (786432, 768, 48, 1), device='cuda:1', dtype=torch.bfloat16)
    select_1 = rand_strided((1024, 24), (24, 1), device='cuda:1', dtype=torch.bfloat16)
    select_3 = rand_strided((1024, 24), (24, 1), device='cuda:1', dtype=torch.bfloat16)
    permute_5 = rand_strided((128, 4, 1024, 48), (196608, 48, 192, 1), device='cuda:1', dtype=torch.bfloat16)
    permute_6 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:1', dtype=torch.bfloat16)
    permute_7 = rand_strided((128, 4, 1024, 48), (196608, 48, 192, 1), device='cuda:1', dtype=torch.bfloat16)
    getitem_2 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:1', dtype=torch.bfloat16)
    getitem_3 = rand_strided((128, 8, 1024), (8192, 1024, 1), device='cuda:1', dtype=torch.float32)
    getitem_8 = rand_strided((2, ), (1, ), device='cuda:1', dtype=torch.uint64)
    getitem_9 = rand_strided((), (), device='cuda:1', dtype=torch.uint64)
    view_7 = rand_strided((131072, 12), (12, 1), device='cuda:1', dtype=torch.bfloat16)
    mm_2 = rand_strided((131072, 8), (8, 1), device='cuda:1', dtype=torch.bfloat16)
    view_10 = rand_strided((131072, 384), (384, 1), device='cuda:1', dtype=torch.bfloat16)
    add_3 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:1', dtype=torch.bfloat16)
    rsqrt_1 = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:1', dtype=torch.float32)
    permute_17 = rand_strided((8, 12), (12, 1), device='cuda:1', dtype=torch.bfloat16)
    permute_34 = rand_strided((384, 96), (96, 1), device='cuda:1', dtype=torch.bfloat16)
    tangents_1 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:1', dtype=torch.bfloat16)
    tangents_2 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:1', dtype=torch.bfloat16)
    tangents_3 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:1', dtype=torch.bfloat16)
    return [primals_1, primals_5, primals_6, primals_7, primals_8, primals_10, primals_11, primals_15, primals_16, view, mm, addmm, cat, ne, add_1, rsqrt, view_4, view_6, select_1, select_3, permute_5, permute_6, permute_7, getitem_2, getitem_3, getitem_8, getitem_9, view_7, mm_2, view_10, add_3, rsqrt_1, permute_17, permute_34, tangents_1, tangents_2, tangents_3]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
