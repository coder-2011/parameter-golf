# AOT ID: ['7_backward']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ox/coxpin5szmpxocegusxeusx7a53xyyoetaqcqoiqjgutdlhbqc4u.py
# Topologically Sorted Source Nodes: [constant_pad_nd_default_2], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   constant_pad_nd_default_2 => constant_pad_nd_default_2
# Graph fragment:
#   %view_27 : Tensor "bf16[131072, 12][12, 1]cuda:2" = PlaceHolder[target=view_27]
#   %constant_pad_nd_default_2 : Tensor "bf16[131072, 16][16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_27, [0, 4, 0, 0]), kwargs = {})
#   return %constant_pad_nd_default_2
triton_poi_fused_mm_0 = async_compile.triton('triton_poi_fused_mm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 11534336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/fx/cfxhi5dfarxps2znocxqlja5ksfj2gqolj5ick44ssfqjm22uxfr.py
# Topologically Sorted Source Nodes: [view_37, view_38, transpose_11, mul_47, linear_11, mul_19, sigmoid_2, getitem_30, mul_48, sum_3, convert_element_type_82, squeeze, convert_element_type_83, convert_element_type_84, sub_1, mul_49, mul_50, convert_element_type_85, mul_51, permute_44, _scaled_dot_product_flash_attention_backward], Original ATen: [aten.view, aten.transpose, aten.mul, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten._scaled_dot_product_flash_attention_backward]
# Source node to ATen node mapping:
#   _scaled_dot_product_flash_attention_backward => _scaled_dot_product_flash_attention_backward
#   convert_element_type_82 => convert_element_type_82
#   convert_element_type_83 => convert_element_type_83
#   convert_element_type_84 => convert_element_type_84
#   convert_element_type_85 => convert_element_type_85
#   getitem_30 => unsqueeze_22
#   linear_11 => view_28
#   mul_19 => mul_31
#   mul_47 => mul_47
#   mul_48 => mul_48
#   mul_49 => mul_49
#   mul_50 => mul_50
#   mul_51 => mul_51
#   permute_44 => permute_44
#   sigmoid_2 => sigmoid_2
#   squeeze => squeeze
#   sub_1 => sub_1
#   sum_3 => sum_3
#   transpose_11 => permute_28
#   view_37 => view_37
#   view_38 => view_38
# Graph fragment:
#   %mm_17 : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm_17]
#   %getitem_24 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:2" = PlaceHolder[target=getitem_24]
#   %mm_11 : Tensor "bf16[131072, 8][8, 1]cuda:2" = PlaceHolder[target=mm_11]
#   %sum_3 : Tensor "f32[128, 1024, 8, 1][8192, 8, 1, 1048576]cuda:2" = PlaceHolder[target=sum_3]
#   %view_37 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_17, [128, 1024, 384]), kwargs = {})
#   %view_38 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_37, [128, 1024, 8, 48]), kwargs = {})
#   %permute_28 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_24, [0, 2, 1, 3]), kwargs = {})
#   %mul_47 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_38, %permute_28), kwargs = {})
#   %view_28 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_11, [128, 1024, 8]), kwargs = {})
#   %mul_31 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_28, 0.5), kwargs = {})
#   %sigmoid_2 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_31,), kwargs = {})
#   %unsqueeze_22 : Tensor "bf16[128, 1024, 8, 1][8192, 8, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_2, 3), kwargs = {})
#   %mul_48 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_38, %unsqueeze_22), kwargs = {})
#   %sum_3 : Tensor "f32[128, 1024, 8, 1][8192, 8, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_47, [3], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_82 : Tensor "bf16[128, 1024, 8, 1][8192, 8, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_3, torch.bfloat16), kwargs = {})
#   %squeeze : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%convert_element_type_82, 3), kwargs = {})
#   %convert_element_type_83 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%squeeze, torch.float32), kwargs = {})
#   %convert_element_type_84 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_2, torch.float32), kwargs = {})
#   %sub_1 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_84), kwargs = {})
#   %mul_49 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_84, %sub_1), kwargs = {})
#   %mul_50 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_83, %mul_49), kwargs = {})
#   %convert_element_type_85 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_50, torch.bfloat16), kwargs = {})
#   %mul_51 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_85, 0.5), kwargs = {})
#   %permute_44 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_48, [0, 2, 1, 3]), kwargs = {})
#   %_scaled_dot_product_flash_attention_backward : [num_users=3] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention_backward.default](args = (%permute_44, %permute_26, %permute_27, %permute_25, %getitem_24, %getitem_25, None, None, 1024, 1024, 0.0, True, %getitem_30, %getitem_31), kwargs = {scale: 0.14433756729740646})
#   return %sum_3,%buf14,%mul_51
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ad/cadoje6onux73c4jkmk4tkjqx2zri6wybhejxbni5ssgzawoqltm.py
# Topologically Sorted Source Nodes: [slice_tensor_2, convert_element_type_91], Original ATen: [aten.mm, aten._to_copy]
# Source node to ATen node mapping:
#   convert_element_type_91 => convert_element_type_91
#   slice_tensor_2 => slice_tensor_2
# Graph fragment:
#   %mm_default_2 : Tensor "bf16[8, 16][16, 1]cuda:2" = PlaceHolder[target=mm_default_2]
#   %slice_tensor_2 : Tensor "bf16[8, 12][16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_default_2, 1, 0, -4), kwargs = {})
#   %convert_element_type_91 : Tensor "f32[8, 12][12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_tensor_2, torch.float32), kwargs = {})
#   return %convert_element_type_91
triton_poi_fused__to_copy_mm_2 = async_compile.triton('triton_poi_fused__to_copy_mm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mm_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/nw/cnwkl77qx5d2zqaho2jxc2whykmtfna762xlm5e5xdiwtvppemsg.py
# Topologically Sorted Source Nodes: [linear_13, leaky_relu_2, pow_10, mul_37, mul_38, mul_39, where_3, convert_element_type_70], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   convert_element_type_70 => convert_element_type_70
#   leaky_relu_2 => convert_element_type_64, gt_2, mul_36, where_2
#   linear_13 => view_33
#   mul_37 => mul_37
#   mul_38 => mul_38
#   mul_39 => mul_39
#   pow_10 => pow_10
#   where_3 => where_3
# Graph fragment:
#   %mm_13 : Tensor "bf16[131072, 1536][1536, 1]cuda:2" = PlaceHolder[target=mm_13]
#   %tangents_1 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2" = PlaceHolder[target=tangents_1]
#   %view_33 : Tensor "bf16[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_13, [128, 1024, 1536]), kwargs = {})
#   %convert_element_type_64 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_33, torch.float32), kwargs = {})
#   %gt_2 : Tensor "b8[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_64, 0), kwargs = {})
#   %mul_36 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_64, 0.5), kwargs = {})
#   %where_2 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %convert_element_type_64, %mul_36), kwargs = {})
#   %pow_10 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_2, 1.0), kwargs = {})
#   %mul_37 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_10, 2.0), kwargs = {})
#   %mul_38 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %mul_37), kwargs = {})
#   %mul_39 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, 0.5), kwargs = {})
#   %where_3 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %mul_38, %mul_39), kwargs = {})
#   %convert_element_type_70 : Tensor "bf16[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_3, torch.bfloat16), kwargs = {})
#   return %convert_element_type_70
triton_poi_fused__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_3 = async_compile.triton('triton_poi_fused__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2013265920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201326592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp4 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.0, tl.float32)
    tmp3 = tmp1 > tmp2
    tmp5 = tl.full([1], 0.5, tl.float32)
    tmp6 = tmp1 * tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp8 = tl.full([1], 2.0, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = tmp10 * tmp5
    tmp12 = tl.where(tmp3, tmp10, tmp11)
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ko/ckoiuc65ippzbxqrmb7eafe24j3bfapxanlyg7lsztyj5zx7sq7n.py
# Topologically Sorted Source Nodes: [permute_45, permute_46, permute_47, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
# Source node to ATen node mapping:
#   copy_ => copy
#   getitem => slice_7
#   permute_45 => permute_45
#   permute_46 => permute_46
#   permute_47 => permute_47
#   triton_kernel_wrapper_mutation => triton_kernel_wrapper_mutation_2
# Graph fragment:
#   %getitem_35 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:2" = PlaceHolder[target=getitem_35]
#   %permute_45 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_35, [0, 2, 1, 3]), kwargs = {})
#   %permute_46 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_34, [0, 2, 1, 3]), kwargs = {})
#   %permute_47 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_33, [0, 2, 1, 3]), kwargs = {})
#   %slice_7 : Tensor "bf16[128, 1024, 8, 48][1179648, 1152, 48, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%empty_6, 2, 16, 9223372036854775807), kwargs = {})
#   %copy : Tensor "bf16[128, 1024, 8, 48][1179648, 1152, 48, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_7, %permute_45), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[128, 1024, 24, 48][1179648, 1152, 48, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty_6, %copy, 2, 16, 9223372036854775807), kwargs = {})
#   %triton_kernel_wrapper_mutation_2 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 23, constant_args_idx: 20, grid: [(2097152, 1, 1)], tma_descriptor_metadata: {}, kwargs: {DQ: %permute_47, DK: %permute_46, QKV: %view_26, DQKV: %slice_scatter_default_1, COS: %select_3, SIN: %select_5}})
#   return %buf20
triton_poi_fused_copy_slice_transpose_4 = async_compile.triton('triton_poi_fused_copy_slice_transpose_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_transpose_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 704643072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_slice_transpose_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150994944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 48) % 24)
    x2 = xindex // 1152
    x3 = (xindex % 1152)
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 16, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-768) + x3 + 384*x2), tmp2, other=0.0).to(tl.float32)
    tmp4 = tl.full([1], float("nan"), tl.float32)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x4), tmp5, None)
''', device_str='cuda')


# Original path: /workspace/parameter-golf/train_gpt_parcae.py:4479
_fused_qkv_postprocess_bwd_kernel_0 = async_compile.triton('_fused_qkv_postprocess_bwd_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 2, 'num_stages': 3}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_fused_qkv_postprocess_bwd_kernel_0', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'DQ': '*bf16', 'DK': '*bf16', 'QKV': '*bf16', 'DQKV': '*bf16', 'COS': '*bf16', 'SIN': '*bf16', 'TOTAL_ROWS': 'constexpr', 'SEQLEN': 'constexpr', 'H_Q': 'constexpr', 'H_KV': 'constexpr', 'HEAD_DIM': 'constexpr', 'ROPE_DIMS': 'constexpr', 'DO_QK_NORM': 'constexpr', 'ROPE_INTERLEAVED': 'constexpr', 'RMS_EPS': 'constexpr', 'BLOCK_D': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'TOTAL_ROWS': 131072, 'SEQLEN': 1024, 'H_Q': 8, 'H_KV': 8, 'HEAD_DIM': 48, 'ROPE_DIMS': 48, 'DO_QK_NORM': True, 'ROPE_INTERLEAVED': False, 'RMS_EPS': 1e-06, 'BLOCK_D': 64}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/k6/ck6tsadodfctkaub4n75rks3nhsxjbsvqwr3gzekneq5u3jwf3ix.py
# Topologically Sorted Source Nodes: [view_35, convert_element_type_75, mul_40, mul_41, rms_norm_4, mul_43, sum_1, div, mul_44, sub, mul_45, mul_46, sum_2, add_13, view_40, convert_element_type_90, full_default, view_48, convert_element_type_96, add_14, mul_52, mul_53, mul_55, sum_4, mul_56, sub_2, mul_57, mul_58, sum_5, add_15, getitem_22, getitem_23, mul_60, getitem_21, mul_62, convert_element_type_99, getitem_20, mul_64, convert_element_type_100], Original ATen: [aten.view, aten._to_copy, aten.mul, aten._fused_rms_norm_backward, aten._fused_rms_norm, aten.add, aten.slice_backward, aten.select, aten.unsqueeze]
# Source node to ATen node mapping:
#   add_13 => add_13
#   add_14 => add_14
#   add_15 => add_15
#   convert_element_type_100 => convert_element_type_100
#   convert_element_type_75 => convert_element_type_75
#   convert_element_type_90 => convert_element_type_90
#   convert_element_type_96 => convert_element_type_96
#   convert_element_type_99 => convert_element_type_99
#   div => div
#   full_default => full_default
#   getitem_20 => unsqueeze_14, unsqueeze_15
#   getitem_21 => unsqueeze_16, unsqueeze_17
#   getitem_22 => select_12
#   getitem_23 => unsqueeze_18, unsqueeze_19
#   mul_40 => mul_40
#   mul_41 => mul_41
#   mul_43 => mul_43
#   mul_44 => mul_44
#   mul_45 => mul_45
#   mul_46 => mul_46
#   mul_52 => mul_52
#   mul_53 => mul_53
#   mul_55 => mul_55
#   mul_56 => mul_56
#   mul_57 => mul_57
#   mul_58 => mul_58
#   mul_60 => mul_60
#   mul_62 => mul_62
#   mul_64 => mul_64
#   rms_norm_4 => mul_28
#   sub => sub
#   sub_2 => sub_2
#   sum_1 => sum_1
#   sum_2 => sum_2
#   sum_4 => sum_4
#   sum_5 => sum_5
#   view_35 => view_35
#   view_40 => view_40
#   view_48 => view_48
# Graph fragment:
#   %add_10 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=add_10]
#   %rsqrt_4 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:2" = PlaceHolder[target=rsqrt_4]
#   %mm_15 : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm_15]
#   %primals_32 : Tensor "f32[384][1]cuda:2" = PlaceHolder[target=primals_32]
#   %mm_19 : Tensor "bf16[131072, 12][12, 1]cuda:2" = PlaceHolder[target=mm_19]
#   %mm_21 : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm_21]
#   %primals_26 : Tensor "f32[384][1]cuda:2" = PlaceHolder[target=primals_26]
#   %tangents_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=tangents_3]
#   %sum_1 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:2" = PlaceHolder[target=sum_1]
#   %sum_4 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:2" = PlaceHolder[target=sum_4]
#   %add_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=add_15]
#   %primals_25 : Tensor "f32[2, 384][384, 1]cuda:2" = PlaceHolder[target=primals_25]
#   %primals_21 : Tensor "f32[384][1]cuda:2" = PlaceHolder[target=primals_21]
#   %primals_20 : Tensor "f32[384][1]cuda:2" = PlaceHolder[target=primals_20]
#   %view_35 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_15, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_75 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_35, torch.float32), kwargs = {})
#   %mul_40 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_75, 0.5), kwargs = {})
#   %mul_41 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %primals_32), kwargs = {})
#   %mul_28 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %rsqrt_4), kwargs = {})
#   %mul_43 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %mul_41), kwargs = {})
#   %sum_1 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_43, [2], True), kwargs = {})
#   %div : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_28, 384), kwargs = {})
#   %mul_44 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sum_1), kwargs = {})
#   %sub : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_41, %mul_44), kwargs = {})
#   %mul_45 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt_4), kwargs = {})
#   %mul_46 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %mul_28), kwargs = {})
#   %sum_2 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_46, [0, 1]), kwargs = {})
#   %add_13 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_3, %mul_45), kwargs = {})
#   %view_40 : Tensor "bf16[128, 1024, 12][12288, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_19, [128, 1024, 12]), kwargs = {})
#   %convert_element_type_90 : Tensor "f32[128, 1024, 12][12288, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_40, torch.float32), kwargs = {})
#   %full_default : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([128, 1024, 384], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:2, pin_memory: False})
#   %slice_scatter_default : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %convert_element_type_90, 2, 0, 12), kwargs = {})
#   %view_48 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_21, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_96 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_48, torch.float32), kwargs = {})
#   %add_14 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %convert_element_type_96), kwargs = {})
#   %mul_52 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, 0.5), kwargs = {})
#   %mul_53 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %primals_26), kwargs = {})
#   %mul_55 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %mul_53), kwargs = {})
#   %sum_4 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_55, [2], True), kwargs = {})
#   %mul_56 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sum_4), kwargs = {})
#   %sub_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_53, %mul_56), kwargs = {})
#   %mul_57 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_4), kwargs = {})
#   %mul_58 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %mul_28), kwargs = {})
#   %sum_5 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_58, [0, 1]), kwargs = {})
#   %add_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %mul_57), kwargs = {})
#   %select_12 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_25, 0, 0), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_12, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 1), kwargs = {})
#   %mul_60 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %unsqueeze_19), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_21, 0), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 1), kwargs = {})
#   %mul_62 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_60, %unsqueeze_17), kwargs = {})
#   %convert_element_type_99 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_62, torch.bfloat16), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_20, 0), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, 1), kwargs = {})
#   %mul_64 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_60, %unsqueeze_15), kwargs = {})
#   %convert_element_type_100 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_64, torch.bfloat16), kwargs = {})
#   return %sum_1,%sum_4,%add_15,%convert_element_type_99,%convert_element_type_100
triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_5 = async_compile.triton('triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 11, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, out_ptr4, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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
    accum1 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        x0 = xindex
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr7 + (r0_1 + 384*x0), r0_mask, other=0.0)
        tmp45 = tl.load(in_ptr8 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp47 = tl.load(in_ptr9 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp50 = tl.load(in_ptr10 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.full([1, 1], 0.5, tl.float32)
        tmp6 = tmp4 * tmp5
        tmp8 = tmp6 * tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = tl.where(r0_mask, tmp10, 0)
        tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
        tmp14 = r0_1
        tmp15 = tl.full([1, 1], 12, tl.int64)
        tmp16 = tmp14 < tmp15
        tmp17 = tl.load(in_ptr4 + (r0_1 + 12*x0), r0_mask & tmp16, other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
        tmp20 = tl.where(tmp16, tmp18, tmp19)
        tmp21 = tl.full([1, 1], 0.0, tl.float32)
        tmp22 = tl.where(tmp16, tmp20, tmp21)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp22 + tmp24
        tmp26 = tmp25 * tmp5
        tmp28 = tmp26 * tmp27
        tmp29 = tmp2 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
        tmp32 = tl.where(r0_mask, tmp30, 0)
        tmp33 = tl.sum(tmp32, 1)[:, None].to(tl.float32)
        tmp35 = tl.full([1, 1], 0.0026041666666666665, tl.float32)
        tmp36 = tmp2 * tmp35
        tmp37 = tmp36 * tmp13
        tmp38 = tmp8 - tmp37
        tmp39 = tmp38 * tmp1
        tmp40 = tmp34 + tmp39
        tmp41 = tmp36 * tmp33
        tmp42 = tmp28 - tmp41
        tmp43 = tmp42 * tmp1
        tmp44 = tmp40 + tmp43
        tmp46 = tmp44 * tmp45
        tmp48 = tmp46 * tmp47
        tmp49 = tmp48.to(tl.float32)
        tmp51 = tmp46 * tmp50
        tmp52 = tmp51.to(tl.float32)
        tmp53 = tmp6 * tmp2
        tmp54 = tmp26 * tmp2
        tl.store(out_ptr2 + (r0_1 + 384*x0), tmp44, r0_mask)
        tl.store(out_ptr3 + (r0_1 + 384*x0), tmp49, r0_mask)
        tl.store(out_ptr4 + (r0_1 + 384*x0), tmp52, r0_mask)
        tmp55 = tl.sum(tmp53, 0)
        tmp56 = accum0 + tmp55
        accum0 = tmp56
        tmp57 = tl.sum(tmp54, 0)
        tmp58 = accum1 + tmp57
        accum1 = tmp58
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/hy/chyc36q6wrkkz57d36tny5e74vyhti7xxwm5uoiqhkyqukrbwshx.py
# Topologically Sorted Source Nodes: [view_50, convert_element_type_105, linear_8, leaky_relu_1, pow_11, mul_66, mul_67, mul_68, where_4, convert_element_type_111], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   convert_element_type_105 => convert_element_type_105
#   convert_element_type_111 => convert_element_type_111
#   leaky_relu_1 => convert_element_type_40, gt_1, mul_23, where_1
#   linear_8 => view_21
#   mul_66 => mul_66
#   mul_67 => mul_67
#   mul_68 => mul_68
#   pow_11 => pow_11
#   view_50 => view_50
#   where_4 => where_4
# Graph fragment:
#   %mm_8 : Tensor "bf16[131072, 1536][1536, 1]cuda:2" = PlaceHolder[target=mm_8]
#   %mm_23 : Tensor "bf16[131072, 1536][1536, 1]cuda:2" = PlaceHolder[target=mm_23]
#   %view_50 : Tensor "bf16[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_23, [128, 1024, 1536]), kwargs = {})
#   %convert_element_type_105 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_50, torch.float32), kwargs = {})
#   %view_21 : Tensor "bf16[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [128, 1024, 1536]), kwargs = {})
#   %convert_element_type_40 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_21, torch.float32), kwargs = {})
#   %gt_1 : Tensor "b8[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_40, 0), kwargs = {})
#   %mul_23 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_40, 0.5), kwargs = {})
#   %where_1 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convert_element_type_40, %mul_23), kwargs = {})
#   %pow_11 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_1, 1.0), kwargs = {})
#   %mul_66 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_11, 2.0), kwargs = {})
#   %mul_67 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_105, %mul_66), kwargs = {})
#   %mul_68 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, 0.5), kwargs = {})
#   %where_4 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %mul_67, %mul_68), kwargs = {})
#   %convert_element_type_111 : Tensor "bf16[128, 1024, 1536][1572864, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_4, torch.bfloat16), kwargs = {})
#   return %convert_element_type_111
triton_poi_fused__to_copy__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_view_6 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1610612736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_view_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201326592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp4 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.0, tl.float32)
    tmp3 = tmp1 > tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.full([1], 0.5, tl.float32)
    tmp7 = tmp1 * tmp6
    tmp8 = tl.where(tmp3, tmp1, tmp7)
    tmp9 = tl.full([1], 2.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 * tmp10
    tmp12 = tmp11 * tmp6
    tmp13 = tl.where(tmp3, tmp11, tmp12)
    tmp14 = tmp13.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/5b/c5bsd5wuxterd23i4lj4tke7dqmkgw5kmywsge4s6k64bza2mfas.py
# Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_8, getitem_13, getitem_14, mul_9, add_3, rms_norm_2, getitem, getitem_1, mul, getitem_2, getitem_3, mul_1, add, rms_norm], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._fused_rms_norm]
# Source node to ATen node mapping:
#   add => add
#   add_3 => add_5
#   getitem => select
#   getitem_1 => unsqueeze, unsqueeze_1
#   getitem_11 => select_6
#   getitem_12 => unsqueeze_10, unsqueeze_9
#   getitem_13 => select_7
#   getitem_14 => unsqueeze_11, unsqueeze_12
#   getitem_2 => select_1
#   getitem_3 => unsqueeze_2, unsqueeze_3
#   mul => mul
#   mul_1 => mul_1
#   mul_8 => mul_13
#   mul_9 => mul_14
#   rms_norm => mul_2
#   rms_norm_2 => mul_15
# Graph fragment:
#   %primals_15 : Tensor "f32[2, 384][384, 1]cuda:2" = PlaceHolder[target=primals_15]
#   %add_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=add_4]
#   %primals_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=primals_3]
#   %rsqrt_2 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:2" = PlaceHolder[target=rsqrt_2]
#   %primals_1 : Tensor "f32[2, 384][384, 1]cuda:2" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=primals_2]
#   %rsqrt : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:2" = PlaceHolder[target=rsqrt]
#   %select_6 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_6, 0), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 1), kwargs = {})
#   %mul_13 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_10, %add_4), kwargs = {})
#   %select_7 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_7, 0), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_11, 1), kwargs = {})
#   %mul_14 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_12, %primals_3), kwargs = {})
#   %add_5 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %mul_14), kwargs = {})
#   %mul_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %rsqrt_2), kwargs = {})
#   %select : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %mul : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %primals_2), kwargs = {})
#   %select_1 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_1, 0, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 0), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 1), kwargs = {})
#   %mul_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %primals_3), kwargs = {})
#   %add : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %mul_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt), kwargs = {})
#   return %mul_15,%mul_2
triton_poi_fused__fused_rms_norm_add_mul_select_unsqueeze_7 = async_compile.triton('triton_poi_fused__fused_rms_norm_add_mul_select_unsqueeze_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__fused_rms_norm_add_mul_select_unsqueeze_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1308628992}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__fused_rms_norm_add_mul_select_unsqueeze_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 384)
    x2 = xindex
    x1 = xindex // 384
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr0 + (384 + x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), None)
    tmp13 = tl.load(in_ptr4 + (384 + x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tmp2 + tmp6
    tmp9 = tmp7 * tmp8
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp5
    tmp15 = tmp12 + tmp14
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp9, None)
    tl.store(out_ptr1 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/x3/cx3gauq23lszwulgrvnaapyb2ouig4ilwsd7lexfxgqsrbzqvx4g.py
# Topologically Sorted Source Nodes: [full_default, getitem_22, getitem_23, mul_60, view_52, convert_element_type_116, mul_69, mul_70, getitem_11, getitem_12, mul_72, sum_10, div_2, mul_73, sub_3, mul_74, mul_75, sum_11, add_17, view_57, convert_element_type_131, view_65, convert_element_type_137, add_18, mul_81, mul_82, mul_84, sum_13, mul_85, sub_5, mul_86, mul_87, sum_14, add_19, mul_89, getitem_10, mul_91, convert_element_type_140, getitem_9, mul_93, convert_element_type_141], Original ATen: [aten.slice_backward, aten.select, aten.unsqueeze, aten.mul, aten.view, aten._to_copy, aten._fused_rms_norm_backward, aten.add]
# Source node to ATen node mapping:
#   add_17 => add_17
#   add_18 => add_18
#   add_19 => add_19
#   convert_element_type_116 => convert_element_type_116
#   convert_element_type_131 => convert_element_type_131
#   convert_element_type_137 => convert_element_type_137
#   convert_element_type_140 => convert_element_type_140
#   convert_element_type_141 => convert_element_type_141
#   div_2 => div_2
#   full_default => full_default
#   getitem_10 => unsqueeze_7, unsqueeze_8
#   getitem_11 => select_6
#   getitem_12 => unsqueeze_10, unsqueeze_9
#   getitem_22 => select_12
#   getitem_23 => unsqueeze_18, unsqueeze_19
#   getitem_9 => unsqueeze_5, unsqueeze_6
#   mul_60 => mul_60
#   mul_69 => mul_69
#   mul_70 => mul_70
#   mul_72 => mul_72
#   mul_73 => mul_73
#   mul_74 => mul_74
#   mul_75 => mul_75
#   mul_81 => mul_81
#   mul_82 => mul_82
#   mul_84 => mul_84
#   mul_85 => mul_85
#   mul_86 => mul_86
#   mul_87 => mul_87
#   mul_89 => mul_89
#   mul_91 => mul_91
#   mul_93 => mul_93
#   sub_3 => sub_3
#   sub_5 => sub_5
#   sum_10 => sum_10
#   sum_11 => sum_11
#   sum_13 => sum_13
#   sum_14 => sum_14
#   view_52 => view_52
#   view_57 => view_57
#   view_65 => view_65
# Graph fragment:
#   %mul_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=mul_15]
#   %mm_25 : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm_25]
#   %primals_22 : Tensor "f32[384][1]cuda:2" = PlaceHolder[target=primals_22]
#   %mm_29 : Tensor "bf16[131072, 12][12, 1]cuda:2" = PlaceHolder[target=mm_29]
#   %mm_31 : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm_31]
#   %primals_16 : Tensor "f32[384][1]cuda:2" = PlaceHolder[target=primals_16]
#   %add_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=add_15]
#   %primals_25 : Tensor "f32[2, 384][384, 1]cuda:2" = PlaceHolder[target=primals_25]
#   %sum_10 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:2" = PlaceHolder[target=sum_10]
#   %rsqrt_2 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:2" = PlaceHolder[target=rsqrt_2]
#   %sum_13 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:2" = PlaceHolder[target=sum_13]
#   %add_19 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=add_19]
#   %primals_15 : Tensor "f32[2, 384][384, 1]cuda:2" = PlaceHolder[target=primals_15]
#   %primals_11 : Tensor "f32[384][1]cuda:2" = PlaceHolder[target=primals_11]
#   %primals_10 : Tensor "f32[384][1]cuda:2" = PlaceHolder[target=primals_10]
#   %full_default : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([128, 1024, 384], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:2, pin_memory: False})
#   %select_12 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_25, 0, 0), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_12, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 1), kwargs = {})
#   %mul_60 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %unsqueeze_19), kwargs = {})
#   %view_52 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_25, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_116 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_52, torch.float32), kwargs = {})
#   %mul_69 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_116, 0.5773502691896258), kwargs = {})
#   %mul_70 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %primals_22), kwargs = {})
#   %select_6 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_6, 0), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 1), kwargs = {})
#   %mul_72 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %mul_70), kwargs = {})
#   %sum_10 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_72, [2], True), kwargs = {})
#   %div_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_15, 384), kwargs = {})
#   %mul_73 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sum_10), kwargs = {})
#   %sub_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_70, %mul_73), kwargs = {})
#   %mul_74 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_75 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %mul_15), kwargs = {})
#   %sum_11 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_75, [0, 1]), kwargs = {})
#   %add_17 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_60, %mul_74), kwargs = {})
#   %view_57 : Tensor "bf16[128, 1024, 12][12288, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_29, [128, 1024, 12]), kwargs = {})
#   %convert_element_type_131 : Tensor "f32[128, 1024, 12][12288, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_57, torch.float32), kwargs = {})
#   %slice_scatter_default_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %convert_element_type_131, 2, 0, 12), kwargs = {})
#   %view_65 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_31, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_137 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_65, torch.float32), kwargs = {})
#   %add_18 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_2, %convert_element_type_137), kwargs = {})
#   %mul_81 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, 0.5773502691896258), kwargs = {})
#   %mul_82 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_81, %primals_16), kwargs = {})
#   %mul_84 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %mul_82), kwargs = {})
#   %sum_13 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_84, [2], True), kwargs = {})
#   %mul_85 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sum_13), kwargs = {})
#   %sub_5 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_82, %mul_85), kwargs = {})
#   %mul_86 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_2), kwargs = {})
#   %mul_87 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_81, %mul_15), kwargs = {})
#   %sum_14 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_87, [0, 1]), kwargs = {})
#   %add_19 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %mul_86), kwargs = {})
#   %mul_89 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %unsqueeze_10), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_11, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 1), kwargs = {})
#   %mul_91 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, %unsqueeze_8), kwargs = {})
#   %convert_element_type_140 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_91, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_10, 0), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 1), kwargs = {})
#   %mul_93 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, %unsqueeze_6), kwargs = {})
#   %convert_element_type_141 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_93, torch.bfloat16), kwargs = {})
#   return %sum_10,%sum_13,%add_19,%convert_element_type_140,%convert_element_type_141
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_8 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 12, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr2, out_ptr3, out_ptr4, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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
    accum1 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        x0 = xindex
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr6 + (r0_1 + 384*x0), r0_mask, other=0.0)
        tmp33 = tl.load(in_ptr7 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp39 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
        tmp46 = tl.load(in_ptr9 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp48 = tl.load(in_ptr10 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.load(in_ptr11 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.full([1, 1], 0.5773502691896258, tl.float32)
        tmp4 = tmp2 * tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = tl.where(r0_mask, tmp8, 0)
        tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
        tmp12 = r0_1
        tmp13 = tl.full([1, 1], 12, tl.int64)
        tmp14 = tmp12 < tmp13
        tmp15 = tl.load(in_ptr3 + (r0_1 + 12*x0), r0_mask & tmp14, other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp14, tmp16, tmp17)
        tmp19 = tl.full([1, 1], 0.0, tl.float32)
        tmp20 = tl.where(tmp14, tmp18, tmp19)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 + tmp22
        tmp24 = tmp23 * tmp3
        tmp26 = tmp24 * tmp25
        tmp27 = tmp0 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
        tmp30 = tl.where(r0_mask, tmp28, 0)
        tmp31 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
        tmp34 = tmp32 * tmp33
        tmp35 = tl.full([1, 1], 0.0026041666666666665, tl.float32)
        tmp36 = tmp0 * tmp35
        tmp37 = tmp36 * tmp11
        tmp38 = tmp6 - tmp37
        tmp40 = tmp38 * tmp39
        tmp41 = tmp34 + tmp40
        tmp42 = tmp36 * tmp31
        tmp43 = tmp26 - tmp42
        tmp44 = tmp43 * tmp39
        tmp45 = tmp41 + tmp44
        tmp47 = tmp45 * tmp46
        tmp49 = tmp47 * tmp48
        tmp50 = tmp49.to(tl.float32)
        tmp52 = tmp47 * tmp51
        tmp53 = tmp52.to(tl.float32)
        tmp54 = tmp4 * tmp0
        tmp55 = tmp24 * tmp0
        tl.store(out_ptr2 + (r0_1 + 384*x0), tmp45, r0_mask)
        tl.store(out_ptr3 + (r0_1 + 384*x0), tmp50, r0_mask)
        tl.store(out_ptr4 + (r0_1 + 384*x0), tmp53, r0_mask)
        tmp56 = tl.sum(tmp54, 0)
        tmp57 = accum0 + tmp56
        accum0 = tmp57
        tmp58 = tl.sum(tmp55, 0)
        tmp59 = accum1 + tmp58
        accum1 = tmp59
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/3c/c3cu5n2eapfqjwuytsezqcgncwujz5xrpd35kjvt3rzjzenisg4x.py
# Topologically Sorted Source Nodes: [mul_59, sum_6, getitem_22, getitem_23, mul_60, mul_61, sum_7, linear_9, mul_63, sum_8, linear_7, mul_65, sum_9, getitem_11, getitem_12, mul_88, sum_15, mul_89, mul_90, sum_16, linear_4, mul_92, sum_17, linear_2, mul_94, sum_18], Original ATen: [aten.mul, aten.sum, aten.select, aten.unsqueeze, aten._unsafe_view]
# Source node to ATen node mapping:
#   getitem_11 => select_6
#   getitem_12 => unsqueeze_10, unsqueeze_9
#   getitem_22 => select_12
#   getitem_23 => unsqueeze_18, unsqueeze_19
#   linear_2 => view_7
#   linear_4 => view_11
#   linear_7 => view_19
#   linear_9 => view_23
#   mul_59 => mul_59
#   mul_60 => mul_60
#   mul_61 => mul_61
#   mul_63 => mul_63
#   mul_65 => mul_65
#   mul_88 => mul_88
#   mul_89 => mul_89
#   mul_90 => mul_90
#   mul_92 => mul_92
#   mul_94 => mul_94
#   sum_15 => sum_15
#   sum_16 => sum_16
#   sum_17 => sum_17
#   sum_18 => sum_18
#   sum_6 => sum_6
#   sum_7 => sum_7
#   sum_8 => sum_8
#   sum_9 => sum_9
# Graph fragment:
#   %add_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=add_15]
#   %primals_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=primals_3]
#   %add_9 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=add_9]
#   %primals_25 : Tensor "f32[2, 384][384, 1]cuda:2" = PlaceHolder[target=primals_25]
#   %mm_9 : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm_9]
#   %mm_7 : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm_7]
#   %add_19 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=add_19]
#   %add_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=add_4]
#   %primals_15 : Tensor "f32[2, 384][384, 1]cuda:2" = PlaceHolder[target=primals_15]
#   %mm_4 : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm_4]
#   %mm_2 : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm_2]
#   %mul_59 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %primals_3), kwargs = {})
#   %sum_6 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_59, [0, 1], True), kwargs = {dtype: torch.float32})
#   %select_12 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_25, 0, 0), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_12, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 1), kwargs = {})
#   %mul_60 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %unsqueeze_19), kwargs = {})
#   %mul_61 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %add_9), kwargs = {})
#   %sum_7 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_61, [0, 1], True), kwargs = {dtype: torch.float32})
#   %view_23 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [128, 1024, 384]), kwargs = {})
#   %mul_63 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_60, %view_23), kwargs = {})
#   %sum_8 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_63, [0, 1], True), kwargs = {dtype: torch.float32})
#   %view_19 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [128, 1024, 384]), kwargs = {})
#   %mul_65 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_60, %view_19), kwargs = {})
#   %sum_9 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_65, [0, 1], True), kwargs = {dtype: torch.float32})
#   %select_6 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_6, 0), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 1), kwargs = {})
#   %mul_88 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %primals_3), kwargs = {})
#   %sum_15 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_88, [0, 1], True), kwargs = {dtype: torch.float32})
#   %mul_89 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %unsqueeze_10), kwargs = {})
#   %mul_90 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %add_4), kwargs = {})
#   %sum_16 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_90, [0, 1], True), kwargs = {dtype: torch.float32})
#   %view_11 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [128, 1024, 384]), kwargs = {})
#   %mul_92 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, %view_11), kwargs = {})
#   %sum_17 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_92, [0, 1], True), kwargs = {dtype: torch.float32})
#   %view_7 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [128, 1024, 384]), kwargs = {})
#   %mul_94 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, %view_7), kwargs = {})
#   %sum_18 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_94, [0, 1], True), kwargs = {dtype: torch.float32})
#   return %buf28,%buf30,%buf33,%buf35,%buf70,%buf72,%buf75,%buf77
triton_red_fused__unsafe_view_mul_select_sum_unsqueeze_9 = async_compile.triton('triton_red_fused__unsafe_view_mul_select_sum_unsqueeze_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_mul_select_sum_unsqueeze_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 11, 'num_store': 8, 'num_reduction': 8, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1317202944, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_mul_select_sum_unsqueeze_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp35 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp42 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp49 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp59 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp67 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
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
        tmp19 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp3 * tmp19
        tmp21 = tl.load(in_ptr4 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 * tmp22
        tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
        tmp25 = tl.where(tmp2, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
        tmp29 = tl.load(in_ptr5 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp20 * tmp30
        tmp32 = tl.full(tmp31.shape, 0, tmp31.dtype)
        tmp33 = tl.where(tmp2, tmp31, tmp32)
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
        tmp36 = _tmp35 + tmp34
        _tmp35 = tl.where(r0_mask & xmask, tmp36, _tmp35)
        tmp37 = tl.load(in_ptr6 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp38 = tmp37 * tmp5
        tmp39 = tl.full(tmp38.shape, 0, tmp38.dtype)
        tmp40 = tl.where(tmp2, tmp38, tmp39)
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(r0_mask & xmask, tmp43, _tmp42)
        tmp44 = tl.load(in_ptr7 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp45 = tmp37 * tmp44
        tmp46 = tl.full(tmp45.shape, 0, tmp45.dtype)
        tmp47 = tl.where(tmp2, tmp45, tmp46)
        tmp48 = tl.broadcast_to(tmp47, [XBLOCK, R0_BLOCK])
        tmp50 = _tmp49 + tmp48
        _tmp49 = tl.where(r0_mask & xmask, tmp50, _tmp49)
        tmp51 = tl.load(in_ptr8 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp52 = tmp37 * tmp51
        tmp53 = tl.load(in_ptr9 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp54 = tmp53.to(tl.float32)
        tmp55 = tmp52 * tmp54
        tmp56 = tl.full(tmp55.shape, 0, tmp55.dtype)
        tmp57 = tl.where(tmp2, tmp55, tmp56)
        tmp58 = tl.broadcast_to(tmp57, [XBLOCK, R0_BLOCK])
        tmp60 = _tmp59 + tmp58
        _tmp59 = tl.where(r0_mask & xmask, tmp60, _tmp59)
        tmp61 = tl.load(in_ptr10 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tmp52 * tmp62
        tmp64 = tl.full(tmp63.shape, 0, tmp63.dtype)
        tmp65 = tl.where(tmp2, tmp63, tmp64)
        tmp66 = tl.broadcast_to(tmp65, [XBLOCK, R0_BLOCK])
        tmp68 = _tmp67 + tmp66
        _tmp67 = tl.where(r0_mask & xmask, tmp68, _tmp67)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tmp35 = tl.sum(_tmp35, 1)[:, None]
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp49 = tl.sum(_tmp49, 1)[:, None]
    tmp59 = tl.sum(_tmp59, 1)[:, None]
    tmp67 = tl.sum(_tmp67, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
    tl.store(out_ptr2 + (x3), tmp27, xmask)
    tl.store(out_ptr3 + (x3), tmp35, xmask)
    tl.store(out_ptr4 + (x3), tmp42, xmask)
    tl.store(out_ptr5 + (x3), tmp49, xmask)
    tl.store(out_ptr6 + (x3), tmp59, xmask)
    tl.store(out_ptr7 + (x3), tmp67, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/hq/chqtslr6ahy3lfm2atazfej736ukkn6dkayqgmm44z3yb5yzeasq.py
# Topologically Sorted Source Nodes: [mul_59, sum_6], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_59 => mul_59
#   sum_6 => sum_6
# Graph fragment:
#   %buf28 : Tensor "f32[1, 1, 384, 349][134016, 134016, 1, 384]cuda:2" = PlaceHolder[target=buf28]
#   %mul_59 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %primals_3), kwargs = {})
#   %sum_6 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_59, [0, 1], True), kwargs = {dtype: torch.float32})
#   return %sum_6
triton_red_fused_mul_sum_10 = async_compile.triton('triton_red_fused_mul_sum_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 539136, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_mul_sum_10(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/gi/cgiwrked4gqc7tos6a4xlut5rzx4k5pginxle3wb5gxoh2leglww.py
# Topologically Sorted Source Nodes: [squeeze_1, squeeze_2, full_default_1, squeeze_3, squeeze_4, add_16], Original ATen: [aten.squeeze, aten.select_backward, aten.add]
# Source node to ATen node mapping:
#   add_16 => add_16
#   full_default_1 => full_default_1
#   squeeze_1 => squeeze_1
#   squeeze_2 => squeeze_2
#   squeeze_3 => squeeze_3
#   squeeze_4 => squeeze_4
# Graph fragment:
#   %sum_6 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2" = PlaceHolder[target=sum_6]
#   %sum_7 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2" = PlaceHolder[target=sum_7]
#   %squeeze_1 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%sum_6, 1), kwargs = {})
#   %squeeze_2 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%squeeze_1, 0), kwargs = {})
#   %full_default_1 : Tensor "f32[2, 384][384, 1]cuda:2"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([2, 384], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:2, pin_memory: False})
#   %select_scatter_default : Tensor "f32[2, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %squeeze_2, 0, 1), kwargs = {})
#   %squeeze_3 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%sum_7, 1), kwargs = {})
#   %squeeze_4 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%squeeze_3, 0), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[2, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %squeeze_4, 0, 0), kwargs = {})
#   %add_16 : Tensor "f32[2, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   return %add_16
triton_poi_fused_add_select_backward_squeeze_11 = async_compile.triton('triton_poi_fused_add_select_backward_squeeze_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_select_backward_squeeze_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_select_backward_squeeze_11(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 384
    x0 = (xindex % 384)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 0.0, tl.float32)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = tmp0 == tmp6
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp5 + tmp9
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/yw/cywyhlepxmu3e6fukrbjtsp5hvtqpb6n2vrzpk3ektqlot2enfbw.py
# Topologically Sorted Source Nodes: [full_default, getitem_11, getitem_12, mul_89, view_69, convert_element_type_157, mul_98, mul_99, getitem, getitem_1, mul_101, sum_19, div_4, mul_102, sub_6, mul_103, mul_104, sum_20, add_21, view_74, convert_element_type_172, view_82, convert_element_type_178, add_22, mul_110, mul_111, mul_113, sum_22, mul_114, sub_8, mul_115, mul_116, sum_23, add_23, mul_118], Original ATen: [aten.slice_backward, aten.select, aten.unsqueeze, aten.mul, aten.view, aten._to_copy, aten._fused_rms_norm_backward, aten.add]
# Source node to ATen node mapping:
#   add_21 => add_21
#   add_22 => add_22
#   add_23 => add_23
#   convert_element_type_157 => convert_element_type_157
#   convert_element_type_172 => convert_element_type_172
#   convert_element_type_178 => convert_element_type_178
#   div_4 => div_4
#   full_default => full_default
#   getitem => select
#   getitem_1 => unsqueeze, unsqueeze_1
#   getitem_11 => select_6
#   getitem_12 => unsqueeze_10, unsqueeze_9
#   mul_101 => mul_101
#   mul_102 => mul_102
#   mul_103 => mul_103
#   mul_104 => mul_104
#   mul_110 => mul_110
#   mul_111 => mul_111
#   mul_113 => mul_113
#   mul_114 => mul_114
#   mul_115 => mul_115
#   mul_116 => mul_116
#   mul_118 => mul_118
#   mul_89 => mul_89
#   mul_98 => mul_98
#   mul_99 => mul_99
#   sub_6 => sub_6
#   sub_8 => sub_8
#   sum_19 => sum_19
#   sum_20 => sum_20
#   sum_22 => sum_22
#   sum_23 => sum_23
#   view_69 => view_69
#   view_74 => view_74
#   view_82 => view_82
# Graph fragment:
#   %mul_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=mul_2]
#   %mm_35 : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm_35]
#   %primals_12 : Tensor "f32[384][1]cuda:2" = PlaceHolder[target=primals_12]
#   %mm_39 : Tensor "bf16[131072, 12][12, 1]cuda:2" = PlaceHolder[target=mm_39]
#   %mm_41 : Tensor "bf16[131072, 384][384, 1]cuda:2" = PlaceHolder[target=mm_41]
#   %primals_4 : Tensor "f32[384][1]cuda:2" = PlaceHolder[target=primals_4]
#   %add_19 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=add_19]
#   %primals_15 : Tensor "f32[2, 384][384, 1]cuda:2" = PlaceHolder[target=primals_15]
#   %sum_19 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:2" = PlaceHolder[target=sum_19]
#   %rsqrt : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:2" = PlaceHolder[target=rsqrt]
#   %sum_22 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:2" = PlaceHolder[target=sum_22]
#   %add_23 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=add_23]
#   %primals_1 : Tensor "f32[2, 384][384, 1]cuda:2" = PlaceHolder[target=primals_1]
#   %full_default : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([128, 1024, 384], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:2, pin_memory: False})
#   %select_6 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_6, 0), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 1), kwargs = {})
#   %mul_89 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %unsqueeze_10), kwargs = {})
#   %view_69 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_35, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_157 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_69, torch.float32), kwargs = {})
#   %mul_98 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_157, 0.7071067811865475), kwargs = {})
#   %mul_99 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_98, %primals_12), kwargs = {})
#   %select : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %mul_101 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %mul_99), kwargs = {})
#   %sum_19 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_101, [2], True), kwargs = {})
#   %div_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, 384), kwargs = {})
#   %mul_102 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sum_19), kwargs = {})
#   %sub_6 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_99, %mul_102), kwargs = {})
#   %mul_103 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt), kwargs = {})
#   %mul_104 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_98, %mul_2), kwargs = {})
#   %sum_20 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_104, [0, 1]), kwargs = {})
#   %add_21 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %mul_103), kwargs = {})
#   %view_74 : Tensor "bf16[128, 1024, 12][12288, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_39, [128, 1024, 12]), kwargs = {})
#   %convert_element_type_172 : Tensor "f32[128, 1024, 12][12288, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_74, torch.float32), kwargs = {})
#   %slice_scatter_default_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %convert_element_type_172, 2, 0, 12), kwargs = {})
#   %view_82 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_41, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_178 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_82, torch.float32), kwargs = {})
#   %add_22 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_4, %convert_element_type_178), kwargs = {})
#   %mul_110 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 0.7071067811865475), kwargs = {})
#   %mul_111 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_110, %primals_4), kwargs = {})
#   %mul_113 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %mul_111), kwargs = {})
#   %sum_22 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_113, [2], True), kwargs = {})
#   %mul_114 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sum_22), kwargs = {})
#   %sub_8 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_111, %mul_114), kwargs = {})
#   %mul_115 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt), kwargs = {})
#   %mul_116 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_110, %mul_2), kwargs = {})
#   %sum_23 : Tensor "f32[384][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_116, [0, 1]), kwargs = {})
#   %add_23 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %mul_115), kwargs = {})
#   %mul_118 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, %unsqueeze_1), kwargs = {})
#   return %sum_19,%sum_22,%add_23,%mul_118
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_12 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr2': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 10, 'num_store': 0, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr2, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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
    accum1 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        x0 = xindex
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_out_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0)
        tmp33 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp39 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
        tmp46 = tl.load(in_ptr8 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.full([1, 1], 0.7071067811865475, tl.float32)
        tmp4 = tmp2 * tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = tl.where(r0_mask, tmp8, 0)
        tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
        tmp12 = r0_1
        tmp13 = tl.full([1, 1], 12, tl.int64)
        tmp14 = tmp12 < tmp13
        tmp15 = tl.load(in_ptr3 + (r0_1 + 12*x0), r0_mask & tmp14, other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp14, tmp16, tmp17)
        tmp19 = tl.full([1, 1], 0.0, tl.float32)
        tmp20 = tl.where(tmp14, tmp18, tmp19)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 + tmp22
        tmp24 = tmp23 * tmp3
        tmp26 = tmp24 * tmp25
        tmp27 = tmp0 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
        tmp30 = tl.where(r0_mask, tmp28, 0)
        tmp31 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
        tmp34 = tmp32 * tmp33
        tmp35 = tl.full([1, 1], 0.0026041666666666665, tl.float32)
        tmp36 = tmp0 * tmp35
        tmp37 = tmp36 * tmp11
        tmp38 = tmp6 - tmp37
        tmp40 = tmp38 * tmp39
        tmp41 = tmp34 + tmp40
        tmp42 = tmp36 * tmp31
        tmp43 = tmp26 - tmp42
        tmp44 = tmp43 * tmp39
        tmp45 = tmp41 + tmp44
        tmp47 = tmp45 * tmp46
        tmp48 = tmp4 * tmp0
        tmp49 = tmp24 * tmp0
        tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp45, r0_mask)
        tl.store(out_ptr2 + (r0_1 + 384*x0), tmp47, r0_mask)
        tmp50 = tl.sum(tmp48, 0)
        tmp51 = accum0 + tmp50
        accum0 = tmp51
        tmp52 = tl.sum(tmp49, 0)
        tmp53 = accum1 + tmp52
        accum1 = tmp53
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/rb/crbtd4yuazeqdnrjjo5wkbonuesgy7gpiryxedjlr7q4eyvmumlx.py
# Topologically Sorted Source Nodes: [mul_117, sum_24, mul_119, sum_25], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_117 => mul_117
#   mul_119 => mul_119
#   sum_24 => sum_24
#   sum_25 => sum_25
# Graph fragment:
#   %add_23 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=add_23]
#   %primals_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=primals_3]
#   %primals_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2" = PlaceHolder[target=primals_2]
#   %mul_117 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, %primals_3), kwargs = {})
#   %sum_24 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_117, [0, 1], True), kwargs = {dtype: torch.float32})
#   %mul_119 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, %primals_2), kwargs = {})
#   %sum_25 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_119, [0, 1], True), kwargs = {dtype: torch.float32})
#   return %buf112,%buf115
triton_red_fused_mul_sum_13 = async_compile.triton('triton_red_fused_mul_sum_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 505460736, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_mul_sum_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        primals_1, primals_2, primals_3, primals_4, primals_9, primals_10, primals_11, primals_12, primals_15, primals_16, primals_19, primals_20, primals_21, primals_22, primals_25, primals_26, primals_29, primals_32, rsqrt, view, view_2, select_3, select_5, permute_3, permute_4, permute_5, getitem_2, getitem_3, getitem_8, getitem_9, view_3, mm_1, view_6, mm_2, view_8, mm_3, view_10, mm_4, add_4, rsqrt_2, view_12, view_14, permute_14, permute_15, permute_16, getitem_13, getitem_14, getitem_19, getitem_20, view_15, mm_6, view_18, mm_7, view_20, mm_8, view_22, mm_9, add_9, add_10, rsqrt_4, view_24, view_26, permute_25, permute_26, permute_27, getitem_24, getitem_25, getitem_30, getitem_31, view_27, mm_11, view_30, view_32, mm_13, permute_34, permute_42, permute_51, permute_55, permute_59, permute_67, permute_76, permute_80, permute_84, permute_92, permute_101, tangents_1, tangents_2, tangents_3 = args
        args.clear()
        assert_size_stride(primals_1, (2, 384), (384, 1))
        assert_size_stride(primals_2, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(primals_3, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(primals_4, (384, ), (1, ))
        assert_size_stride(primals_9, (384, 384), (384, 1))
        assert_size_stride(primals_10, (384, ), (1, ))
        assert_size_stride(primals_11, (384, ), (1, ))
        assert_size_stride(primals_12, (384, ), (1, ))
        assert_size_stride(primals_15, (2, 384), (384, 1))
        assert_size_stride(primals_16, (384, ), (1, ))
        assert_size_stride(primals_19, (384, 384), (384, 1))
        assert_size_stride(primals_20, (384, ), (1, ))
        assert_size_stride(primals_21, (384, ), (1, ))
        assert_size_stride(primals_22, (384, ), (1, ))
        assert_size_stride(primals_25, (2, 384), (384, 1))
        assert_size_stride(primals_26, (384, ), (1, ))
        assert_size_stride(primals_29, (384, 384), (384, 1))
        assert_size_stride(primals_32, (384, ), (1, ))
        assert_size_stride(rsqrt, (128, 1024, 1), (1024, 1, 1))
        assert_size_stride(view, (131072, 384), (384, 1))
        assert_size_stride(view_2, (128, 1024, 24, 48), (1179648, 1152, 48, 1))
        assert_size_stride(select_3, (1024, 24), (24, 1))
        assert_size_stride(select_5, (1024, 24), (24, 1))
        assert_size_stride(permute_3, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(permute_4, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(permute_5, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(getitem_2, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(getitem_3, (128, 8, 1024), (8192, 1024, 1))
        assert_size_stride(getitem_8, (2, ), (1, ))
        assert_size_stride(getitem_9, (), ())
        assert_size_stride(view_3, (131072, 12), (12, 1))
        assert_size_stride(mm_1, (131072, 8), (8, 1))
        assert_size_stride(view_6, (131072, 384), (384, 1))
        assert_size_stride(mm_2, (131072, 384), (384, 1))
        assert_size_stride(view_8, (131072, 384), (384, 1))
        assert_size_stride(mm_3, (131072, 1536), (1536, 1))
        assert_size_stride(view_10, (131072, 1536), (1536, 1))
        assert_size_stride(mm_4, (131072, 384), (384, 1))
        assert_size_stride(add_4, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(rsqrt_2, (128, 1024, 1), (1024, 1, 1))
        assert_size_stride(view_12, (131072, 384), (384, 1))
        assert_size_stride(view_14, (128, 1024, 24, 48), (1179648, 1152, 48, 1))
        assert_size_stride(permute_14, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(permute_15, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(permute_16, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(getitem_13, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(getitem_14, (128, 8, 1024), (8192, 1024, 1))
        assert_size_stride(getitem_19, (2, ), (1, ))
        assert_size_stride(getitem_20, (), ())
        assert_size_stride(view_15, (131072, 12), (12, 1))
        assert_size_stride(mm_6, (131072, 8), (8, 1))
        assert_size_stride(view_18, (131072, 384), (384, 1))
        assert_size_stride(mm_7, (131072, 384), (384, 1))
        assert_size_stride(view_20, (131072, 384), (384, 1))
        assert_size_stride(mm_8, (131072, 1536), (1536, 1))
        assert_size_stride(view_22, (131072, 1536), (1536, 1))
        assert_size_stride(mm_9, (131072, 384), (384, 1))
        assert_size_stride(add_9, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(add_10, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(rsqrt_4, (128, 1024, 1), (1024, 1, 1))
        assert_size_stride(view_24, (131072, 384), (384, 1))
        assert_size_stride(view_26, (128, 1024, 24, 48), (1179648, 1152, 48, 1))
        assert_size_stride(permute_25, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(permute_26, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(permute_27, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(getitem_24, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(getitem_25, (128, 8, 1024), (8192, 1024, 1))
        assert_size_stride(getitem_30, (2, ), (1, ))
        assert_size_stride(getitem_31, (), ())
        assert_size_stride(view_27, (131072, 12), (12, 1))
        assert_size_stride(mm_11, (131072, 8), (8, 1))
        assert_size_stride(view_30, (131072, 384), (384, 1))
        assert_size_stride(view_32, (131072, 384), (384, 1))
        assert_size_stride(mm_13, (131072, 1536), (1536, 1))
        assert_size_stride(permute_34, (1536, 384), (384, 1))
        assert_size_stride(permute_42, (8, 12), (12, 1))
        assert_size_stride(permute_51, (1152, 384), (384, 1))
        assert_size_stride(permute_55, (384, 1536), (1536, 1))
        assert_size_stride(permute_59, (1536, 384), (384, 1))
        assert_size_stride(permute_67, (8, 12), (12, 1))
        assert_size_stride(permute_76, (1152, 384), (384, 1))
        assert_size_stride(permute_80, (384, 1536), (1536, 1))
        assert_size_stride(permute_84, (1536, 384), (384, 1))
        assert_size_stride(permute_92, (8, 12), (12, 1))
        assert_size_stride(permute_101, (1152, 384), (384, 1))
        assert_size_stride(tangents_1, (128, 1024, 1536), (1572864, 1536, 1))
        assert_size_stride(tangents_2, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(tangents_3, (128, 1024, 384), (393216, 384, 1))
        with torch.cuda._DeviceGuard(2):
            torch.cuda.set_device(2)
            buf6 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_36, permute_36, mm_16], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(tangents_2, (384, 131072), (1, 384), 0), view_30, out=buf6)
            del view_30
            buf10 = empty_strided_cuda((131072, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [constant_pad_nd_default_2], Original ATen: [aten.mm]
            stream2 = get_raw_stream(2)
            triton_poi_fused_mm_0.run(view_27, buf10, 2097152, stream=stream2)
            del view_27
            buf52 = empty_strided_cuda((131072, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [constant_pad_nd_default_1], Original ATen: [aten.mm]
            stream2 = get_raw_stream(2)
            triton_poi_fused_mm_0.run(view_15, buf52, 2097152, stream=stream2)
            del view_15
            buf94 = empty_strided_cuda((131072, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [constant_pad_nd_default], Original ATen: [aten.mm]
            stream2 = get_raw_stream(2)
            triton_poi_fused_mm_0.run(view_3, buf94, 2097152, stream=stream2)
            del view_3
            buf7 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_36, linear_12, permute_38, mm_17], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(tangents_2, (131072, 384), (384, 1), 0), primals_29, out=buf7)
            del primals_29
            del tangents_2
            buf14 = empty_strided_cuda((128, 8, 1024, 48), (393216, 48, 384, 1), torch.bfloat16)
            buf9 = empty_strided_cuda((128, 1024, 8), (8192, 8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_37, view_38, transpose_11, mul_47, linear_11, mul_19, sigmoid_2, getitem_30, mul_48, sum_3, convert_element_type_82, squeeze, convert_element_type_83, convert_element_type_84, sub_1, mul_49, mul_50, convert_element_type_85, mul_51, permute_44, _scaled_dot_product_flash_attention_backward], Original ATen: [aten.view, aten.transpose, aten.mul, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten._scaled_dot_product_flash_attention_backward]
            stream2 = get_raw_stream(2)
            triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_transpose_unsqueeze_view_1.run(buf7, getitem_24, mm_11, buf14, buf9, 1048576, 48, stream=stream2)
            del buf7
            del mm_11
            buf11 = empty_strided_cuda((8, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_11, mul_19, sigmoid_2, convert_element_type_82, squeeze, convert_element_type_83, convert_element_type_84, sub_1, mul_49, mul_50, convert_element_type_85, mul_51, view_39, permute_40, constant_pad_nd_default_2, mm_default_2], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf9, (8, 131072), (1, 8), 0), buf10, out=buf11)
            del buf10
            buf13 = empty_strided_cuda((8, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [slice_tensor_2, convert_element_type_91], Original ATen: [aten.mm, aten._to_copy]
            stream2 = get_raw_stream(2)
            triton_poi_fused__to_copy_mm_2.run(buf11, buf13, 96, stream=stream2)
            del buf11
            buf12 = empty_strided_cuda((131072, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_11, mul_19, sigmoid_2, convert_element_type_82, squeeze, convert_element_type_83, convert_element_type_84, sub_1, mul_49, mul_50, convert_element_type_85, mul_51, view_39, mm_19], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf9, (131072, 8), (8, 1), 0), permute_42, out=buf12)
            del buf9
            del permute_42
            # Topologically Sorted Source Nodes: [view_37, view_38, linear_11, mul_19, sigmoid_2, getitem_30, mul_48, permute_44, _scaled_dot_product_flash_attention_backward], Original ATen: [aten.view, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.transpose, aten._scaled_dot_product_flash_attention_backward]
            buf15 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(buf14, permute_26, permute_27, permute_25, getitem_24, getitem_25, None, None, 1024, 1024, 0.0, True, getitem_30, getitem_31, scale=0.14433756729740646)
            del getitem_24
            del getitem_25
            del getitem_30
            del getitem_31
            del permute_25
            del permute_26
            del permute_27
            buf0 = reinterpret_tensor(mm_13, (128, 1024, 1536), (1572864, 1536, 1), 0); del mm_13  # reuse
            # Topologically Sorted Source Nodes: [linear_13, leaky_relu_2, pow_10, mul_37, mul_38, mul_39, where_3, convert_element_type_70], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward]
            stream2 = get_raw_stream(2)
            triton_poi_fused__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_3.run(buf0, tangents_1, 201326592, stream=stream2)
            del tangents_1
            buf1 = empty_strided_cuda((1536, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_13, leaky_relu_2, pow_10, mul_37, mul_38, mul_39, where_3, convert_element_type_70, view_34, permute_32, mm_14], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf0, (1536, 131072), (1, 1536), 0), view_32, out=buf1)
            del view_32
            buf2 = reinterpret_tensor(buf14, (131072, 384), (384, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [linear_13, leaky_relu_2, pow_10, mul_37, mul_38, mul_39, where_3, convert_element_type_70, view_34, mm_15], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf0, (131072, 1536), (1536, 1), 0), permute_34, out=buf2)
            del permute_34
            buf16 = buf15[0]
            assert_size_stride(buf16, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf16, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf17 = buf15[1]
            assert_size_stride(buf17, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf17, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf18 = buf15[2]
            assert_size_stride(buf18, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf18, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            del buf15
            buf20 = empty_strided_cuda((128, 1024, 24, 48), (1179648, 1152, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_45, permute_46, permute_47, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream2 = get_raw_stream(2)
            triton_poi_fused_copy_slice_transpose_4.run(buf18, buf20, 150994944, stream=stream2)
            # Topologically Sorted Source Nodes: [permute_45, permute_46, permute_47, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream2 = get_raw_stream(2)
            _fused_qkv_postprocess_bwd_kernel_0.run(reinterpret_tensor(buf16, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), reinterpret_tensor(buf17, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), view_26, buf20, select_3, select_5, 2097152, 1, 1, stream=stream2)
            del view_26
            buf22 = empty_strided_cuda((1152, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_44, view_45, permute_49, mm_20], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf20, (1152, 131072), (1, 1152), 0), view_24, out=buf22)
            del view_24
            buf23 = reinterpret_tensor(buf17, (131072, 384), (384, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [view_44, view_45, mm_21], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf20, (131072, 1152), (1152, 1), 0), permute_51, out=buf23)
            del permute_51
            buf27 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            buf37 = reinterpret_tensor(buf16, (128, 1024, 384), (393216, 384, 1), 0); del buf16  # reuse
            buf47 = reinterpret_tensor(buf18, (128, 1024, 384), (393216, 384, 1), 0); del buf18  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_35, convert_element_type_75, mul_40, mul_41, rms_norm_4, mul_43, sum_1, div, mul_44, sub, mul_45, mul_46, sum_2, add_13, view_40, convert_element_type_90, full_default, view_48, convert_element_type_96, add_14, mul_52, mul_53, mul_55, sum_4, mul_56, sub_2, mul_57, mul_58, sum_5, add_15, getitem_22, getitem_23, mul_60, getitem_21, mul_62, convert_element_type_99, getitem_20, mul_64, convert_element_type_100], Original ATen: [aten.view, aten._to_copy, aten.mul, aten._fused_rms_norm_backward, aten._fused_rms_norm, aten.add, aten.slice_backward, aten.select, aten.unsqueeze]
            workspace_0 = empty_strided_cuda((1572864, ), (1, ), torch.float32)
            stream2 = get_raw_stream(2)
            triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_5.run(add_10, rsqrt_4, buf2, primals_32, buf12, buf23, primals_26, tangents_3, primals_25, primals_21, primals_20, buf27, buf37, buf47, workspace_0, 131072, 384, stream=stream2)
            buf5 = workspace_0[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            buf26 = workspace_0[1 * 2048 * 384 : (1 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del add_10
            del buf2
            del primals_20
            del primals_21
            del primals_26
            del primals_32
            del rsqrt_4
            del tangents_3
            buf38 = empty_strided_cuda((384, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_22, getitem_23, mul_60, getitem_21, mul_62, convert_element_type_99, view_49, permute_53, mm_22], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf37, (384, 131072), (1, 384), 0), view_22, out=buf38)
            del view_22
            buf48 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_22, getitem_23, mul_60, getitem_20, mul_64, convert_element_type_100, view_53, permute_61, mm_26], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf47, (384, 131072), (1, 384), 0), view_18, out=buf48)
            del view_18
            buf49 = buf23; del buf23  # reuse
            # Topologically Sorted Source Nodes: [getitem_22, getitem_23, mul_60, getitem_20, mul_64, convert_element_type_100, view_53, linear_7, permute_63, mm_27], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf47, (131072, 384), (384, 1), 0), primals_19, out=buf49)
            del primals_19
            buf56 = reinterpret_tensor(buf47, (128, 8, 1024, 48), (393216, 48, 384, 1), 0); del buf47  # reuse
            buf51 = empty_strided_cuda((128, 1024, 8), (8192, 8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_54, view_55, transpose_7, mul_76, linear_6, mul_11, sigmoid_1, getitem_19, mul_77, sum_12, convert_element_type_123, squeeze_9, convert_element_type_124, convert_element_type_125, sub_4, mul_78, mul_79, convert_element_type_126, mul_80, permute_69, _scaled_dot_product_flash_attention_backward_1], Original ATen: [aten.view, aten.transpose, aten.mul, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten._scaled_dot_product_flash_attention_backward]
            stream2 = get_raw_stream(2)
            triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_transpose_unsqueeze_view_1.run(buf49, getitem_13, mm_6, buf56, buf51, 1048576, 48, stream=stream2)
            del buf49
            del mm_6
            # Topologically Sorted Source Nodes: [view_54, view_55, linear_6, mul_11, sigmoid_1, getitem_19, mul_77, permute_69, _scaled_dot_product_flash_attention_backward_1], Original ATen: [aten.view, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.transpose, aten._scaled_dot_product_flash_attention_backward]
            buf57 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(buf56, permute_15, permute_16, permute_14, getitem_13, getitem_14, None, None, 1024, 1024, 0.0, True, getitem_19, getitem_20, scale=0.14433756729740646)
            del buf56
            del getitem_13
            del getitem_14
            del getitem_19
            del getitem_20
            del permute_14
            del permute_15
            del permute_16
            buf53 = empty_strided_cuda((8, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_6, mul_11, sigmoid_1, convert_element_type_123, squeeze_9, convert_element_type_124, convert_element_type_125, sub_4, mul_78, mul_79, convert_element_type_126, mul_80, view_56, permute_65, constant_pad_nd_default_1, mm_default_1], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf51, (8, 131072), (1, 8), 0), buf52, out=buf53)
            del buf52
            buf58 = buf57[0]
            assert_size_stride(buf58, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf58, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf59 = buf57[1]
            assert_size_stride(buf59, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf59, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf60 = buf57[2]
            assert_size_stride(buf60, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf60, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            del buf57
            buf55 = empty_strided_cuda((8, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [slice_tensor_1, convert_element_type_132], Original ATen: [aten.mm, aten._to_copy]
            stream2 = get_raw_stream(2)
            triton_poi_fused__to_copy_mm_2.run(buf53, buf55, 96, stream=stream2)
            buf54 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [linear_6, mul_11, sigmoid_1, convert_element_type_123, squeeze_9, convert_element_type_124, convert_element_type_125, sub_4, mul_78, mul_79, convert_element_type_126, mul_80, view_56, mm_29], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf51, (131072, 8), (8, 1), 0), permute_67, out=buf54)
            del permute_67
            buf62 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [permute_70, permute_71, permute_72, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream2 = get_raw_stream(2)
            triton_poi_fused_copy_slice_transpose_4.run(buf60, buf62, 150994944, stream=stream2)
            # Topologically Sorted Source Nodes: [permute_70, permute_71, permute_72, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream2 = get_raw_stream(2)
            _fused_qkv_postprocess_bwd_kernel_0.run(reinterpret_tensor(buf58, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), reinterpret_tensor(buf59, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), view_14, buf62, select_3, select_5, 2097152, 1, 1, stream=stream2)
            del view_14
            buf64 = empty_strided_cuda((1152, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_61, view_62, permute_74, mm_30], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf62, (1152, 131072), (1, 1152), 0), view_12, out=buf64)
            del view_12
            buf65 = reinterpret_tensor(buf59, (131072, 384), (384, 1), 0); del buf59  # reuse
            # Topologically Sorted Source Nodes: [view_61, view_62, mm_31], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf62, (131072, 1152), (1152, 1), 0), permute_76, out=buf65)
            del permute_76
            buf39 = reinterpret_tensor(buf0, (131072, 1536), (1536, 1), 0); del buf0  # reuse
            # Topologically Sorted Source Nodes: [getitem_22, getitem_23, mul_60, getitem_21, mul_62, convert_element_type_99, view_49, mm_23], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf37, (131072, 384), (384, 1), 0), permute_55, out=buf39)
            del permute_55
            buf40 = reinterpret_tensor(mm_8, (128, 1024, 1536), (1572864, 1536, 1), 0); del mm_8  # reuse
            # Topologically Sorted Source Nodes: [view_50, convert_element_type_105, linear_8, leaky_relu_1, pow_11, mul_66, mul_67, mul_68, where_4, convert_element_type_111], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward]
            stream2 = get_raw_stream(2)
            triton_poi_fused__to_copy__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_view_6.run(buf40, buf39, 201326592, stream=stream2)
            del buf39
            buf41 = empty_strided_cuda((1536, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_50, convert_element_type_105, linear_8, leaky_relu_1, pow_11, mul_66, mul_67, mul_68, where_4, convert_element_type_111, view_51, permute_57, mm_24], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (1536, 131072), (1, 1536), 0), view_20, out=buf41)
            del view_20
            buf42 = reinterpret_tensor(buf37, (131072, 384), (384, 1), 0); del buf37  # reuse
            # Topologically Sorted Source Nodes: [view_50, convert_element_type_105, linear_8, leaky_relu_1, pow_11, mul_66, mul_67, mul_68, where_4, convert_element_type_111, view_51, mm_25], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (131072, 1536), (1536, 1), 0), permute_59, out=buf42)
            del permute_59
            buf43 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            buf85 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_8, getitem_13, getitem_14, mul_9, add_3, rms_norm_2, getitem, getitem_1, mul, getitem_2, getitem_3, mul_1, add, rms_norm], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._fused_rms_norm]
            stream2 = get_raw_stream(2)
            triton_poi_fused__fused_rms_norm_add_mul_select_unsqueeze_7.run(primals_15, add_4, primals_3, rsqrt_2, primals_1, primals_2, rsqrt, buf43, buf85, 50331648, stream=stream2)
            buf69 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            buf79 = reinterpret_tensor(buf58, (128, 1024, 384), (393216, 384, 1), 0); del buf58  # reuse
            buf89 = reinterpret_tensor(buf60, (128, 1024, 384), (393216, 384, 1), 0); del buf60  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [full_default, getitem_22, getitem_23, mul_60, view_52, convert_element_type_116, mul_69, mul_70, getitem_11, getitem_12, mul_72, sum_10, div_2, mul_73, sub_3, mul_74, mul_75, sum_11, add_17, view_57, convert_element_type_131, view_65, convert_element_type_137, add_18, mul_81, mul_82, mul_84, sum_13, mul_85, sub_5, mul_86, mul_87, sum_14, add_19, mul_89, getitem_10, mul_91, convert_element_type_140, getitem_9, mul_93, convert_element_type_141], Original ATen: [aten.slice_backward, aten.select, aten.unsqueeze, aten.mul, aten.view, aten._to_copy, aten._fused_rms_norm_backward, aten.add]
            workspace_1 = workspace_0; del workspace_0  # reuse
            stream2 = get_raw_stream(2)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_8.run(buf43, buf42, primals_22, buf54, buf65, primals_16, buf27, primals_25, rsqrt_2, primals_15, primals_11, primals_10, buf69, buf79, buf89, workspace_1, 131072, 384, stream=stream2)
            buf46 = workspace_1[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            buf68 = workspace_1[1 * 2048 * 384 : (1 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del buf42
            del buf43
            del primals_10
            del primals_11
            del primals_16
            del primals_22
            del rsqrt_2
            buf28 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf30 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf33 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf35 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf70 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf72 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf75 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf77 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            # Topologically Sorted Source Nodes: [mul_59, sum_6, getitem_22, getitem_23, mul_60, mul_61, sum_7, linear_9, mul_63, sum_8, linear_7, mul_65, sum_9, getitem_11, getitem_12, mul_88, sum_15, mul_89, mul_90, sum_16, linear_4, mul_92, sum_17, linear_2, mul_94, sum_18], Original ATen: [aten.mul, aten.sum, aten.select, aten.unsqueeze, aten._unsafe_view]
            stream2 = get_raw_stream(2)
            triton_red_fused__unsafe_view_mul_select_sum_unsqueeze_9.run(buf27, primals_3, add_9, primals_25, mm_9, mm_7, buf69, add_4, primals_15, mm_4, mm_2, buf28, buf30, buf33, buf35, buf70, buf72, buf75, buf77, 134016, 376, stream=stream2)
            del add_4
            del add_9
            del mm_2
            del mm_4
            del mm_7
            del mm_9
            del primals_25
            buf80 = empty_strided_cuda((384, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_89, getitem_10, mul_91, convert_element_type_140, view_66, permute_78, mm_32], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf79, (384, 131072), (1, 384), 0), view_10, out=buf80)
            del view_10
            buf90 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_89, getitem_9, mul_93, convert_element_type_141, view_70, permute_86, mm_36], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf89, (384, 131072), (1, 384), 0), view_6, out=buf90)
            del view_6
            buf29 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_59, sum_6], Original ATen: [aten.mul, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused_mul_sum_10.run(buf28, buf29, 384, 349, stream=stream2)
            del buf28
            buf31 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_61, sum_7], Original ATen: [aten.mul, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused_mul_sum_10.run(buf30, buf31, 384, 349, stream=stream2)
            del buf30
            buf34 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_22, getitem_23, mul_60, linear_9, mul_63, sum_8], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._unsafe_view, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused_mul_sum_10.run(buf33, buf34, 384, 349, stream=stream2)
            del buf33
            buf36 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_22, getitem_23, mul_60, linear_7, mul_65, sum_9], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._unsafe_view, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused_mul_sum_10.run(buf35, buf36, 384, 349, stream=stream2)
            del buf35
            buf71 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_88, sum_15], Original ATen: [aten.mul, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused_mul_sum_10.run(buf70, buf71, 384, 349, stream=stream2)
            del buf70
            buf73 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_90, sum_16], Original ATen: [aten.mul, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused_mul_sum_10.run(buf72, buf73, 384, 349, stream=stream2)
            del buf72
            buf76 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_89, linear_4, mul_92, sum_17], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._unsafe_view, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused_mul_sum_10.run(buf75, buf76, 384, 349, stream=stream2)
            buf78 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_89, linear_2, mul_94, sum_18], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._unsafe_view, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused_mul_sum_10.run(buf77, buf78, 384, 349, stream=stream2)
            buf32 = empty_strided_cuda((2, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [squeeze_1, squeeze_2, full_default_1, squeeze_3, squeeze_4, add_16], Original ATen: [aten.squeeze, aten.select_backward, aten.add]
            stream2 = get_raw_stream(2)
            triton_poi_fused_add_select_backward_squeeze_11.run(buf29, buf31, buf32, 768, stream=stream2)
            del buf29
            del buf31
            buf74 = empty_strided_cuda((2, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_default_1, squeeze_10, squeeze_11, squeeze_12, squeeze_13, add_20], Original ATen: [aten.select_backward, aten.squeeze, aten.add]
            stream2 = get_raw_stream(2)
            triton_poi_fused_add_select_backward_squeeze_11.run(buf71, buf73, buf74, 768, stream=stream2)
            buf91 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_89, getitem_9, mul_93, convert_element_type_141, view_70, linear_2, permute_88, mm_37], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf89, (131072, 384), (384, 1), 0), primals_9, out=buf91)
            del primals_9
            buf98 = reinterpret_tensor(buf89, (128, 8, 1024, 48), (393216, 48, 384, 1), 0); del buf89  # reuse
            buf93 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [view_71, view_72, transpose_3, mul_105, linear_1, mul_3, sigmoid, getitem_8, mul_106, sum_21, convert_element_type_164, squeeze_18, convert_element_type_165, convert_element_type_166, sub_7, mul_107, mul_108, convert_element_type_167, mul_109, permute_94, _scaled_dot_product_flash_attention_backward_2], Original ATen: [aten.view, aten.transpose, aten.mul, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten._scaled_dot_product_flash_attention_backward]
            stream2 = get_raw_stream(2)
            triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_transpose_unsqueeze_view_1.run(buf91, getitem_2, mm_1, buf98, buf93, 1048576, 48, stream=stream2)
            del buf91
            del mm_1
            # Topologically Sorted Source Nodes: [view_71, view_72, linear_1, mul_3, sigmoid, getitem_8, mul_106, permute_94, _scaled_dot_product_flash_attention_backward_2], Original ATen: [aten.view, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.transpose, aten._scaled_dot_product_flash_attention_backward]
            buf99 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(buf98, permute_4, permute_5, permute_3, getitem_2, getitem_3, None, None, 1024, 1024, 0.0, True, getitem_8, getitem_9, scale=0.14433756729740646)
            del buf98
            del getitem_2
            del getitem_3
            del getitem_8
            del getitem_9
            del permute_3
            del permute_4
            del permute_5
            buf95 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [linear_1, mul_3, sigmoid, convert_element_type_164, squeeze_18, convert_element_type_165, convert_element_type_166, sub_7, mul_107, mul_108, convert_element_type_167, mul_109, view_73, permute_90, constant_pad_nd_default, mm_default], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf93, (8, 131072), (1, 8), 0), buf94, out=buf95)
            del buf94
            buf100 = buf99[0]
            assert_size_stride(buf100, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf100, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf101 = buf99[1]
            assert_size_stride(buf101, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf101, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf102 = buf99[2]
            assert_size_stride(buf102, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf102, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            del buf99
            buf97 = empty_strided_cuda((8, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [slice_tensor, convert_element_type_173], Original ATen: [aten.mm, aten._to_copy]
            stream2 = get_raw_stream(2)
            triton_poi_fused__to_copy_mm_2.run(buf95, buf97, 96, stream=stream2)
            del buf95
            buf96 = buf54; del buf54  # reuse
            # Topologically Sorted Source Nodes: [linear_1, mul_3, sigmoid, convert_element_type_164, squeeze_18, convert_element_type_165, convert_element_type_166, sub_7, mul_107, mul_108, convert_element_type_167, mul_109, view_73, mm_39], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf93, (131072, 8), (8, 1), 0), permute_92, out=buf96)
            del buf93
            del permute_92
            buf104 = buf62; del buf62  # reuse
            # Topologically Sorted Source Nodes: [permute_95, permute_96, permute_97, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream2 = get_raw_stream(2)
            triton_poi_fused_copy_slice_transpose_4.run(buf102, buf104, 150994944, stream=stream2)
            del buf102
            # Topologically Sorted Source Nodes: [permute_95, permute_96, permute_97, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream2 = get_raw_stream(2)
            _fused_qkv_postprocess_bwd_kernel_0.run(reinterpret_tensor(buf100, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), reinterpret_tensor(buf101, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), view_2, buf104, select_3, select_5, 2097152, 1, 1, stream=stream2)
            del buf100
            del select_3
            del select_5
            del view_2
            buf106 = empty_strided_cuda((1152, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_78, view_79, permute_99, mm_40], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf104, (1152, 131072), (1, 1152), 0), view, out=buf106)
            del view
            buf107 = reinterpret_tensor(buf101, (131072, 384), (384, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [view_78, view_79, mm_41], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf104, (131072, 1152), (1152, 1), 0), permute_101, out=buf107)
            del buf104
            del permute_101
            buf81 = reinterpret_tensor(buf40, (131072, 1536), (1536, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_89, getitem_10, mul_91, convert_element_type_140, view_66, mm_33], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf79, (131072, 384), (384, 1), 0), permute_80, out=buf81)
            del permute_80
            buf82 = reinterpret_tensor(mm_3, (128, 1024, 1536), (1572864, 1536, 1), 0); del mm_3  # reuse
            # Topologically Sorted Source Nodes: [view_67, convert_element_type_146, linear_3, leaky_relu, pow_12, mul_95, mul_96, mul_97, where_5, convert_element_type_152], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward]
            stream2 = get_raw_stream(2)
            triton_poi_fused__to_copy__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_view_6.run(buf82, buf81, 201326592, stream=stream2)
            del buf81
            buf83 = empty_strided_cuda((1536, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_67, convert_element_type_146, linear_3, leaky_relu, pow_12, mul_95, mul_96, mul_97, where_5, convert_element_type_152, view_68, permute_82, mm_34], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf82, (1536, 131072), (1, 1536), 0), view_8, out=buf83)
            del view_8
            buf84 = reinterpret_tensor(buf79, (131072, 384), (384, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [view_67, convert_element_type_146, linear_3, leaky_relu, pow_12, mul_95, mul_96, mul_97, where_5, convert_element_type_152, view_68, mm_35], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf82, (131072, 1536), (1536, 1), 0), permute_84, out=buf84)
            del buf82
            del permute_84
            buf111 = buf69; del buf69  # reuse
            buf114 = buf27; del buf27  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [full_default, getitem_11, getitem_12, mul_89, view_69, convert_element_type_157, mul_98, mul_99, getitem, getitem_1, mul_101, sum_19, div_4, mul_102, sub_6, mul_103, mul_104, sum_20, add_21, view_74, convert_element_type_172, view_82, convert_element_type_178, add_22, mul_110, mul_111, mul_113, sum_22, mul_114, sub_8, mul_115, mul_116, sum_23, add_23, mul_118], Original ATen: [aten.slice_backward, aten.select, aten.unsqueeze, aten.mul, aten.view, aten._to_copy, aten._fused_rms_norm_backward, aten.add]
            workspace_2 = workspace_1; del workspace_1  # reuse
            stream2 = get_raw_stream(2)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_12.run(buf111, buf85, buf84, primals_12, buf96, buf107, primals_4, primals_15, rsqrt, primals_1, buf114, workspace_2, 131072, 384, stream=stream2)
            buf88 = workspace_2[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            buf110 = workspace_2[1 * 2048 * 384 : (1 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del workspace_2
            del buf107
            del buf84
            del buf85
            del buf96
            del primals_1
            del primals_12
            del primals_15
            del primals_4
            del rsqrt
            buf112 = buf77; del buf77  # reuse
            buf115 = buf75; del buf75  # reuse
            # Topologically Sorted Source Nodes: [mul_117, sum_24, mul_119, sum_25], Original ATen: [aten.mul, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused_mul_sum_13.run(buf111, primals_3, primals_2, buf112, buf115, 134016, 376, stream=stream2)
            del buf111
            del primals_2
            del primals_3
            buf113 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [mul_117, sum_24], Original ATen: [aten.mul, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused_mul_sum_10.run(buf112, buf113, 384, 349, stream=stream2)
            del buf112
            buf116 = buf71; del buf71  # reuse
            # Topologically Sorted Source Nodes: [mul_119, sum_25], Original ATen: [aten.mul, aten.sum]
            stream2 = get_raw_stream(2)
            triton_red_fused_mul_sum_10.run(buf115, buf116, 384, 349, stream=stream2)
            del buf115
            buf117 = empty_strided_cuda((2, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_default_1, squeeze_19, squeeze_20, squeeze_21, squeeze_22, add_24], Original ATen: [aten.select_backward, aten.squeeze, aten.add]
            stream2 = get_raw_stream(2)
            triton_poi_fused_add_select_backward_squeeze_11.run(buf113, buf116, buf117, 768, stream=stream2)
            del buf113
            del buf116
        return (buf117, buf114, None, buf110, buf106, None, None, buf97, buf90, reinterpret_tensor(buf78, (384, ), (1, ), 0), reinterpret_tensor(buf76, (384, ), (1, ), 0), buf88, buf83, buf80, buf74, buf68, buf64, buf55, buf48, reinterpret_tensor(buf36, (384, ), (1, ), 0), reinterpret_tensor(buf34, (384, ), (1, ), 0), buf46, buf41, buf38, buf32, buf26, buf22, buf13, buf6, None, None, buf5, buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((2, 384), (384, 1), device='cuda:2', dtype=torch.float32)
    primals_2 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.float32)
    primals_3 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_4 = rand_strided((384, ), (1, ), device='cuda:2', dtype=torch.float32)
    primals_9 = rand_strided((384, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_10 = rand_strided((384, ), (1, ), device='cuda:2', dtype=torch.float32)
    primals_11 = rand_strided((384, ), (1, ), device='cuda:2', dtype=torch.float32)
    primals_12 = rand_strided((384, ), (1, ), device='cuda:2', dtype=torch.float32)
    primals_15 = rand_strided((2, 384), (384, 1), device='cuda:2', dtype=torch.float32)
    primals_16 = rand_strided((384, ), (1, ), device='cuda:2', dtype=torch.float32)
    primals_19 = rand_strided((384, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_20 = rand_strided((384, ), (1, ), device='cuda:2', dtype=torch.float32)
    primals_21 = rand_strided((384, ), (1, ), device='cuda:2', dtype=torch.float32)
    primals_22 = rand_strided((384, ), (1, ), device='cuda:2', dtype=torch.float32)
    primals_25 = rand_strided((2, 384), (384, 1), device='cuda:2', dtype=torch.float32)
    primals_26 = rand_strided((384, ), (1, ), device='cuda:2', dtype=torch.float32)
    primals_29 = rand_strided((384, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_32 = rand_strided((384, ), (1, ), device='cuda:2', dtype=torch.float32)
    rsqrt = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:2', dtype=torch.float32)
    view = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    view_2 = rand_strided((128, 1024, 24, 48), (1179648, 1152, 48, 1), device='cuda:2', dtype=torch.bfloat16)
    select_3 = rand_strided((1024, 24), (24, 1), device='cuda:2', dtype=torch.bfloat16)
    select_5 = rand_strided((1024, 24), (24, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_3 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_4 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_5 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    getitem_2 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    getitem_3 = rand_strided((128, 8, 1024), (8192, 1024, 1), device='cuda:2', dtype=torch.float32)
    getitem_8 = rand_strided((2, ), (1, ), device='cuda:2', dtype=torch.uint64)
    getitem_9 = rand_strided((), (), device='cuda:2', dtype=torch.uint64)
    view_3 = rand_strided((131072, 12), (12, 1), device='cuda:2', dtype=torch.bfloat16)
    mm_1 = rand_strided((131072, 8), (8, 1), device='cuda:2', dtype=torch.bfloat16)
    view_6 = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    mm_2 = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    view_8 = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    mm_3 = rand_strided((131072, 1536), (1536, 1), device='cuda:2', dtype=torch.bfloat16)
    view_10 = rand_strided((131072, 1536), (1536, 1), device='cuda:2', dtype=torch.bfloat16)
    mm_4 = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    add_4 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.float32)
    rsqrt_2 = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:2', dtype=torch.float32)
    view_12 = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    view_14 = rand_strided((128, 1024, 24, 48), (1179648, 1152, 48, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_14 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_15 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_16 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    getitem_13 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    getitem_14 = rand_strided((128, 8, 1024), (8192, 1024, 1), device='cuda:2', dtype=torch.float32)
    getitem_19 = rand_strided((2, ), (1, ), device='cuda:2', dtype=torch.uint64)
    getitem_20 = rand_strided((), (), device='cuda:2', dtype=torch.uint64)
    view_15 = rand_strided((131072, 12), (12, 1), device='cuda:2', dtype=torch.bfloat16)
    mm_6 = rand_strided((131072, 8), (8, 1), device='cuda:2', dtype=torch.bfloat16)
    view_18 = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    mm_7 = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    view_20 = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    mm_8 = rand_strided((131072, 1536), (1536, 1), device='cuda:2', dtype=torch.bfloat16)
    view_22 = rand_strided((131072, 1536), (1536, 1), device='cuda:2', dtype=torch.bfloat16)
    mm_9 = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    add_9 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.float32)
    add_10 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.float32)
    rsqrt_4 = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:2', dtype=torch.float32)
    view_24 = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    view_26 = rand_strided((128, 1024, 24, 48), (1179648, 1152, 48, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_25 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_26 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_27 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    getitem_24 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    getitem_25 = rand_strided((128, 8, 1024), (8192, 1024, 1), device='cuda:2', dtype=torch.float32)
    getitem_30 = rand_strided((2, ), (1, ), device='cuda:2', dtype=torch.uint64)
    getitem_31 = rand_strided((), (), device='cuda:2', dtype=torch.uint64)
    view_27 = rand_strided((131072, 12), (12, 1), device='cuda:2', dtype=torch.bfloat16)
    mm_11 = rand_strided((131072, 8), (8, 1), device='cuda:2', dtype=torch.bfloat16)
    view_30 = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    view_32 = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    mm_13 = rand_strided((131072, 1536), (1536, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_34 = rand_strided((1536, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_42 = rand_strided((8, 12), (12, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_51 = rand_strided((1152, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_55 = rand_strided((384, 1536), (1536, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_59 = rand_strided((1536, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_67 = rand_strided((8, 12), (12, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_76 = rand_strided((1152, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_80 = rand_strided((384, 1536), (1536, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_84 = rand_strided((1536, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_92 = rand_strided((8, 12), (12, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_101 = rand_strided((1152, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    tangents_1 = rand_strided((128, 1024, 1536), (1572864, 1536, 1), device='cuda:2', dtype=torch.float32)
    tangents_2 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.bfloat16)
    tangents_3 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:2', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_9, primals_10, primals_11, primals_12, primals_15, primals_16, primals_19, primals_20, primals_21, primals_22, primals_25, primals_26, primals_29, primals_32, rsqrt, view, view_2, select_3, select_5, permute_3, permute_4, permute_5, getitem_2, getitem_3, getitem_8, getitem_9, view_3, mm_1, view_6, mm_2, view_8, mm_3, view_10, mm_4, add_4, rsqrt_2, view_12, view_14, permute_14, permute_15, permute_16, getitem_13, getitem_14, getitem_19, getitem_20, view_15, mm_6, view_18, mm_7, view_20, mm_8, view_22, mm_9, add_9, add_10, rsqrt_4, view_24, view_26, permute_25, permute_26, permute_27, getitem_24, getitem_25, getitem_30, getitem_31, view_27, mm_11, view_30, view_32, mm_13, permute_34, permute_42, permute_51, permute_55, permute_59, permute_67, permute_76, permute_80, permute_84, permute_92, permute_101, tangents_1, tangents_2, tangents_3]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
