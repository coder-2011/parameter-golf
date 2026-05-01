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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/4h/c4h5ep6optstwzjxnxykmcdv57w6hv2rp2w4psmvw2bi6igbiznj.py
# Topologically Sorted Source Nodes: [constant_pad_nd_default_2], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   constant_pad_nd_default_2 => constant_pad_nd_default_2
# Graph fragment:
#   %view_27 : Tensor "bf16[131072, 12][12, 1]cuda:5" = PlaceHolder[target=view_27]
#   %constant_pad_nd_default_2 : Tensor "bf16[131072, 16][16, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_27, [0, 4, 0, 0]), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 11534336}},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/gw/cgwd7umyxuqqaklvjurctadj5z6737wa47lin5qo3gxbyaef2qxg.py
# Topologically Sorted Source Nodes: [view_33, view_34, transpose_11, mul_43, linear_11, mul_19, sigmoid_2, getitem_30, mul_44, sum_3, convert_element_type_63, squeeze, convert_element_type_64, convert_element_type_65, sub_1, mul_45, mul_46, convert_element_type_66, mul_47, permute_39, _scaled_dot_product_flash_attention_backward], Original ATen: [aten.view, aten.transpose, aten.mul, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten._scaled_dot_product_flash_attention_backward]
# Source node to ATen node mapping:
#   _scaled_dot_product_flash_attention_backward => _scaled_dot_product_flash_attention_backward
#   convert_element_type_63 => convert_element_type_63
#   convert_element_type_64 => convert_element_type_64
#   convert_element_type_65 => convert_element_type_65
#   convert_element_type_66 => convert_element_type_66
#   getitem_30 => unsqueeze_22
#   linear_11 => view_28
#   mul_19 => mul_31
#   mul_43 => mul_43
#   mul_44 => mul_44
#   mul_45 => mul_45
#   mul_46 => mul_46
#   mul_47 => mul_47
#   permute_39 => permute_39
#   sigmoid_2 => sigmoid_2
#   squeeze => squeeze
#   sub_1 => sub_1
#   sum_3 => sum_3
#   transpose_11 => permute_28
#   view_33 => view_33
#   view_34 => view_34
# Graph fragment:
#   %mm_14 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_14]
#   %getitem_24 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:5" = PlaceHolder[target=getitem_24]
#   %mm_11 : Tensor "bf16[131072, 8][8, 1]cuda:5" = PlaceHolder[target=mm_11]
#   %sum_3 : Tensor "f32[128, 1024, 8, 1][8192, 8, 1, 1048576]cuda:5" = PlaceHolder[target=sum_3]
#   %view_33 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_14, [128, 1024, 384]), kwargs = {})
#   %view_34 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_33, [128, 1024, 8, 48]), kwargs = {})
#   %permute_28 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_24, [0, 2, 1, 3]), kwargs = {})
#   %mul_43 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_34, %permute_28), kwargs = {})
#   %view_28 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_11, [128, 1024, 8]), kwargs = {})
#   %mul_31 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_28, 0.5), kwargs = {})
#   %sigmoid_2 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_31,), kwargs = {})
#   %unsqueeze_22 : Tensor "bf16[128, 1024, 8, 1][8192, 8, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_2, 3), kwargs = {})
#   %mul_44 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_34, %unsqueeze_22), kwargs = {})
#   %sum_3 : Tensor "f32[128, 1024, 8, 1][8192, 8, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_43, [3], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_63 : Tensor "bf16[128, 1024, 8, 1][8192, 8, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_3, torch.bfloat16), kwargs = {})
#   %squeeze : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%convert_element_type_63, 3), kwargs = {})
#   %convert_element_type_64 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%squeeze, torch.float32), kwargs = {})
#   %convert_element_type_65 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_2, torch.float32), kwargs = {})
#   %sub_1 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_65), kwargs = {})
#   %mul_45 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_65, %sub_1), kwargs = {})
#   %mul_46 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_64, %mul_45), kwargs = {})
#   %convert_element_type_66 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_46, torch.bfloat16), kwargs = {})
#   %mul_47 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_66, 0.5), kwargs = {})
#   %permute_39 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_44, [0, 2, 1, 3]), kwargs = {})
#   %_scaled_dot_product_flash_attention_backward : [num_users=3] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention_backward.default](args = (%permute_39, %permute_26, %permute_27, %permute_25, %getitem_24, %getitem_25, None, None, 1024, 1024, 0.0, True, %getitem_30, %getitem_31), kwargs = {scale: 0.14433756729740646})
#   return %sum_3,%buf11,%mul_47
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_transpose_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6291456, 'r0_': 402653184}}
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ob/cobgzo6ve2fttkkotsqfn3dx3hbrj5jvpyfennph4ockisga7och.py
# Topologically Sorted Source Nodes: [slice_tensor_2, convert_element_type_72], Original ATen: [aten.mm, aten._to_copy]
# Source node to ATen node mapping:
#   convert_element_type_72 => convert_element_type_72
#   slice_tensor_2 => slice_tensor_2
# Graph fragment:
#   %mm_default_2 : Tensor "bf16[8, 16][16, 1]cuda:5" = PlaceHolder[target=mm_default_2]
#   %slice_tensor_2 : Tensor "bf16[8, 12][16, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_default_2, 1, 0, -4), kwargs = {})
#   %convert_element_type_72 : Tensor "f32[8, 12][12, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_tensor_2, torch.float32), kwargs = {})
#   return %convert_element_type_72
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 960}},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/gf/cgfaesgwplfiq7lwcs56slmnneajzfj7h536penph33onsdtmyf4.py
# Topologically Sorted Source Nodes: [permute_40, permute_41, permute_42, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
# Source node to ATen node mapping:
#   copy_ => copy
#   getitem => slice_7
#   permute_40 => permute_40
#   permute_41 => permute_41
#   permute_42 => permute_42
#   triton_kernel_wrapper_mutation => triton_kernel_wrapper_mutation_2
# Graph fragment:
#   %getitem_35 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:5" = PlaceHolder[target=getitem_35]
#   %permute_40 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_35, [0, 2, 1, 3]), kwargs = {})
#   %permute_41 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_34, [0, 2, 1, 3]), kwargs = {})
#   %permute_42 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_33, [0, 2, 1, 3]), kwargs = {})
#   %slice_7 : Tensor "bf16[128, 1024, 8, 48][1179648, 1152, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%empty_6, 2, 16, 9223372036854775807), kwargs = {})
#   %copy : Tensor "bf16[128, 1024, 8, 48][1179648, 1152, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_7, %permute_40), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[128, 1024, 24, 48][1179648, 1152, 48, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty_6, %copy, 2, 16, 9223372036854775807), kwargs = {})
#   %triton_kernel_wrapper_mutation_2 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 23, constant_args_idx: 20, grid: [(2097152, 1, 1)], tma_descriptor_metadata: {}, kwargs: {DQ: %permute_42, DK: %permute_41, QKV: %view_26, DQKV: %slice_scatter_default_1, COS: %select_3, SIN: %select_5}})
#   return %buf17
triton_poi_fused_copy_slice_transpose_3 = async_compile.triton('triton_poi_fused_copy_slice_transpose_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_transpose_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 704643072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_slice_transpose_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# Original path: /workspace/parameter-golf/train_gpt_parcae.py:4480
_fused_qkv_postprocess_bwd_kernel_0 = async_compile.triton('_fused_qkv_postprocess_bwd_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 2, 'num_stages': 3}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_fused_qkv_postprocess_bwd_kernel_0', 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'DQ': '*bf16', 'DK': '*bf16', 'QKV': '*bf16', 'DQKV': '*bf16', 'COS': '*bf16', 'SIN': '*bf16', 'TOTAL_ROWS': 'constexpr', 'SEQLEN': 'constexpr', 'H_Q': 'constexpr', 'H_KV': 'constexpr', 'HEAD_DIM': 'constexpr', 'ROPE_DIMS': 'constexpr', 'DO_QK_NORM': 'constexpr', 'ROPE_INTERLEAVED': 'constexpr', 'RMS_EPS': 'constexpr', 'BLOCK_D': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'TOTAL_ROWS': 131072, 'SEQLEN': 1024, 'H_Q': 8, 'H_KV': 8, 'HEAD_DIM': 48, 'ROPE_DIMS': 48, 'DO_QK_NORM': True, 'ROPE_INTERLEAVED': False, 'RMS_EPS': 1e-06, 'BLOCK_D': 64}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/wp/cwpulpnyvzlnv7ffwc4oxt77nuwapoati5i7nawjv7kzp2swbr55.py
# Topologically Sorted Source Nodes: [mul_36, mul_37, rms_norm_4, mul_39, sum_1, div, mul_40, sub, mul_41, mul_42, sum_2, add_13, view_36, convert_element_type_71, full_default, view_44, convert_element_type_77, add_14, mul_48, mul_49, mul_51, sum_4, mul_52, sub_2, mul_53, mul_54, sum_5, add_15, getitem_22, getitem_23, mul_56, getitem_21, mul_58, convert_element_type_80, getitem_20, mul_60, convert_element_type_81], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._fused_rms_norm, aten.add, aten.view, aten._to_copy, aten.slice_backward, aten.select, aten.unsqueeze]
# Source node to ATen node mapping:
#   add_13 => add_13
#   add_14 => add_14
#   add_15 => add_15
#   convert_element_type_71 => convert_element_type_71
#   convert_element_type_77 => convert_element_type_77
#   convert_element_type_80 => convert_element_type_80
#   convert_element_type_81 => convert_element_type_81
#   div => div
#   full_default => full_default
#   getitem_20 => unsqueeze_14, unsqueeze_15
#   getitem_21 => unsqueeze_16, unsqueeze_17
#   getitem_22 => select_12
#   getitem_23 => unsqueeze_18, unsqueeze_19
#   mul_36 => mul_36
#   mul_37 => mul_37
#   mul_39 => mul_39
#   mul_40 => mul_40
#   mul_41 => mul_41
#   mul_42 => mul_42
#   mul_48 => mul_48
#   mul_49 => mul_49
#   mul_51 => mul_51
#   mul_52 => mul_52
#   mul_53 => mul_53
#   mul_54 => mul_54
#   mul_56 => mul_56
#   mul_58 => mul_58
#   mul_60 => mul_60
#   rms_norm_4 => mul_28
#   sub => sub
#   sub_2 => sub_2
#   sum_1 => sum_1
#   sum_2 => sum_2
#   sum_4 => sum_4
#   sum_5 => sum_5
#   view_36 => view_36
#   view_44 => view_44
# Graph fragment:
#   %add_10 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_10]
#   %rsqrt_4 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5" = PlaceHolder[target=rsqrt_4]
#   %tangents_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=tangents_1]
#   %primals_32 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_32]
#   %mm_16 : Tensor "bf16[131072, 12][12, 1]cuda:5" = PlaceHolder[target=mm_16]
#   %mm_18 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_18]
#   %primals_26 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_26]
#   %tangents_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=tangents_3]
#   %sum_1 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:5" = PlaceHolder[target=sum_1]
#   %sum_4 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:5" = PlaceHolder[target=sum_4]
#   %add_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_15]
#   %primals_25 : Tensor "f32[2, 384][384, 1]cuda:5" = PlaceHolder[target=primals_25]
#   %primals_21 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_21]
#   %primals_20 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_20]
#   %mul_36 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, 0.5), kwargs = {})
#   %mul_37 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %primals_32), kwargs = {})
#   %mul_28 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %rsqrt_4), kwargs = {})
#   %mul_39 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %mul_37), kwargs = {})
#   %sum_1 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_39, [2], True), kwargs = {})
#   %div : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_28, 384), kwargs = {})
#   %mul_40 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sum_1), kwargs = {})
#   %sub : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_37, %mul_40), kwargs = {})
#   %mul_41 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt_4), kwargs = {})
#   %mul_42 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %mul_28), kwargs = {})
#   %sum_2 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_42, [0, 1]), kwargs = {})
#   %add_13 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_3, %mul_41), kwargs = {})
#   %view_36 : Tensor "bf16[128, 1024, 12][12288, 12, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_16, [128, 1024, 12]), kwargs = {})
#   %convert_element_type_71 : Tensor "f32[128, 1024, 12][12288, 12, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_36, torch.float32), kwargs = {})
#   %full_default : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([128, 1024, 384], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:5, pin_memory: False})
#   %slice_scatter_default : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %convert_element_type_71, 2, 0, 12), kwargs = {})
#   %view_44 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_18, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_77 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_44, torch.float32), kwargs = {})
#   %add_14 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %convert_element_type_77), kwargs = {})
#   %mul_48 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, 0.5), kwargs = {})
#   %mul_49 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %primals_26), kwargs = {})
#   %mul_51 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %mul_49), kwargs = {})
#   %sum_4 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_51, [2], True), kwargs = {})
#   %mul_52 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sum_4), kwargs = {})
#   %sub_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_49, %mul_52), kwargs = {})
#   %mul_53 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_4), kwargs = {})
#   %mul_54 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %mul_28), kwargs = {})
#   %sum_5 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_54, [0, 1]), kwargs = {})
#   %add_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %mul_53), kwargs = {})
#   %select_12 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_25, 0, 0), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_12, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 1), kwargs = {})
#   %mul_56 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %unsqueeze_19), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_21, 0), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 1), kwargs = {})
#   %mul_58 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_56, %unsqueeze_17), kwargs = {})
#   %convert_element_type_80 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_58, torch.bfloat16), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_20, 0), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, 1), kwargs = {})
#   %mul_60 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_56, %unsqueeze_15), kwargs = {})
#   %convert_element_type_81 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_60, torch.bfloat16), kwargs = {})
#   return %sum_1,%sum_4,%add_15,%convert_element_type_80,%convert_element_type_81
triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_4 = async_compile.triton('triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 11, 'num_store': 1, 'num_reduction': 2, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, out_ptr4, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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
        tmp3 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, other=0.0)
        tmp6 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr5 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr7 + (r0_1 + 384*x0), r0_mask, other=0.0)
        tmp44 = tl.load(in_ptr8 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp46 = tl.load(in_ptr9 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp49 = tl.load(in_ptr10 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tl.full([1, 1], 0.5, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = tl.where(r0_mask, tmp9, 0)
        tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
        tmp13 = r0_1
        tmp14 = tl.full([1, 1], 12, tl.int64)
        tmp15 = tmp13 < tmp14
        tmp16 = tl.load(in_ptr4 + (r0_1 + 12*x0), r0_mask & tmp15, other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp15, tmp17, tmp18)
        tmp20 = tl.full([1, 1], 0.0, tl.float32)
        tmp21 = tl.where(tmp15, tmp19, tmp20)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 + tmp23
        tmp25 = tmp24 * tmp4
        tmp27 = tmp25 * tmp26
        tmp28 = tmp2 * tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
        tmp31 = tl.where(r0_mask, tmp29, 0)
        tmp32 = tl.sum(tmp31, 1)[:, None].to(tl.float32)
        tmp34 = tl.full([1, 1], 0.0026041666666666665, tl.float32)
        tmp35 = tmp2 * tmp34
        tmp36 = tmp35 * tmp12
        tmp37 = tmp7 - tmp36
        tmp38 = tmp37 * tmp1
        tmp39 = tmp33 + tmp38
        tmp40 = tmp35 * tmp32
        tmp41 = tmp27 - tmp40
        tmp42 = tmp41 * tmp1
        tmp43 = tmp39 + tmp42
        tmp45 = tmp43 * tmp44
        tmp47 = tmp45 * tmp46
        tmp48 = tmp47.to(tl.float32)
        tmp50 = tmp45 * tmp49
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp5 * tmp2
        tmp53 = tmp25 * tmp2
        tl.store(out_ptr2 + (r0_1 + 384*x0), tmp43, r0_mask)
        tl.store(out_ptr3 + (r0_1 + 384*x0), tmp48, r0_mask)
        tl.store(out_ptr4 + (r0_1 + 384*x0), tmp51, r0_mask)
        tmp54 = tl.sum(tmp52, 0)
        tmp55 = accum0 + tmp54
        accum0 = tmp55
        tmp56 = tl.sum(tmp53, 0)
        tmp57 = accum1 + tmp56
        accum1 = tmp57
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/jg/cjgfaddssmqjwxnjwco6dcfs247m6uf5at7xlh5hiclaqcgp4ssn.py
# Topologically Sorted Source Nodes: [view_46, convert_element_type_86, linear_8, leaky_relu_1, pow_9, mul_62, mul_63, mul_64, where_2, convert_element_type_92], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   convert_element_type_86 => convert_element_type_86
#   convert_element_type_92 => convert_element_type_92
#   leaky_relu_1 => convert_element_type_40, gt_1, mul_23, where_1
#   linear_8 => view_21
#   mul_62 => mul_62
#   mul_63 => mul_63
#   mul_64 => mul_64
#   pow_9 => pow_9
#   view_46 => view_46
#   where_2 => where_2
# Graph fragment:
#   %mm_8 : Tensor "bf16[131072, 1024][1024, 1]cuda:5" = PlaceHolder[target=mm_8]
#   %mm_20 : Tensor "bf16[131072, 1024][1024, 1]cuda:5" = PlaceHolder[target=mm_20]
#   %view_46 : Tensor "bf16[128, 1024, 1024][1048576, 1024, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_20, [128, 1024, 1024]), kwargs = {})
#   %convert_element_type_86 : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_46, torch.float32), kwargs = {})
#   %view_21 : Tensor "bf16[128, 1024, 1024][1048576, 1024, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [128, 1024, 1024]), kwargs = {})
#   %convert_element_type_40 : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:5"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_21, torch.float32), kwargs = {})
#   %gt_1 : Tensor "b8[128, 1024, 1024][1048576, 1024, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_40, 0), kwargs = {})
#   %mul_23 : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_40, 0.5), kwargs = {})
#   %where_1 : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convert_element_type_40, %mul_23), kwargs = {})
#   %pow_9 : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_1, 1.0), kwargs = {})
#   %mul_62 : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_9, 2.0), kwargs = {})
#   %mul_63 : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_86, %mul_62), kwargs = {})
#   %mul_64 : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_63, 0.5), kwargs = {})
#   %where_2 : Tensor "f32[128, 1024, 1024][1048576, 1024, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %mul_63, %mul_64), kwargs = {})
#   %convert_element_type_92 : Tensor "bf16[128, 1024, 1024][1048576, 1024, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_2, torch.bfloat16), kwargs = {})
#   return %convert_element_type_92
triton_poi_fused__to_copy__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_view_5 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1073741824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_view_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/xu/cxuwgp5mads774e6uksrgdz4cqfsloahrz7mhjtf2xw43hyo7kbn.py
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
#   %primals_15 : Tensor "f32[2, 384][384, 1]cuda:5" = PlaceHolder[target=primals_15]
#   %add_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_4]
#   %primals_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=primals_3]
#   %rsqrt_2 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5" = PlaceHolder[target=rsqrt_2]
#   %primals_1 : Tensor "f32[2, 384][384, 1]cuda:5" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=primals_2]
#   %rsqrt : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5" = PlaceHolder[target=rsqrt]
#   %select_6 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_6, 0), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 1), kwargs = {})
#   %mul_13 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_10, %add_4), kwargs = {})
#   %select_7 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_7, 0), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_11, 1), kwargs = {})
#   %mul_14 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_12, %primals_3), kwargs = {})
#   %add_5 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %mul_14), kwargs = {})
#   %mul_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %rsqrt_2), kwargs = {})
#   %select : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %mul : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %primals_2), kwargs = {})
#   %select_1 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_1, 0, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 0), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 1), kwargs = {})
#   %mul_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %primals_3), kwargs = {})
#   %add : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %mul_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt), kwargs = {})
#   return %mul_15,%mul_2
triton_poi_fused__fused_rms_norm_add_mul_select_unsqueeze_6 = async_compile.triton('triton_poi_fused__fused_rms_norm_add_mul_select_unsqueeze_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__fused_rms_norm_add_mul_select_unsqueeze_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1308628992}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__fused_rms_norm_add_mul_select_unsqueeze_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/fj/cfjktsrzzswemod254ybv3asd5aknsn5sv3ce5psuimyejp4g2ql.py
# Topologically Sorted Source Nodes: [full_default, getitem_22, getitem_23, mul_56, view_48, convert_element_type_97, mul_65, mul_66, getitem_11, getitem_12, mul_68, sum_10, div_2, mul_69, sub_3, mul_70, mul_71, sum_11, add_17, view_53, convert_element_type_112, view_61, convert_element_type_118, add_18, mul_77, mul_78, mul_80, sum_13, mul_81, sub_5, mul_82, mul_83, sum_14, add_19, mul_85, getitem_10, mul_87, convert_element_type_121, getitem_9, mul_89, convert_element_type_122], Original ATen: [aten.slice_backward, aten.select, aten.unsqueeze, aten.mul, aten.view, aten._to_copy, aten._fused_rms_norm_backward, aten.add]
# Source node to ATen node mapping:
#   add_17 => add_17
#   add_18 => add_18
#   add_19 => add_19
#   convert_element_type_112 => convert_element_type_112
#   convert_element_type_118 => convert_element_type_118
#   convert_element_type_121 => convert_element_type_121
#   convert_element_type_122 => convert_element_type_122
#   convert_element_type_97 => convert_element_type_97
#   div_2 => div_2
#   full_default => full_default
#   getitem_10 => unsqueeze_7, unsqueeze_8
#   getitem_11 => select_6
#   getitem_12 => unsqueeze_10, unsqueeze_9
#   getitem_22 => select_12
#   getitem_23 => unsqueeze_18, unsqueeze_19
#   getitem_9 => unsqueeze_5, unsqueeze_6
#   mul_56 => mul_56
#   mul_65 => mul_65
#   mul_66 => mul_66
#   mul_68 => mul_68
#   mul_69 => mul_69
#   mul_70 => mul_70
#   mul_71 => mul_71
#   mul_77 => mul_77
#   mul_78 => mul_78
#   mul_80 => mul_80
#   mul_81 => mul_81
#   mul_82 => mul_82
#   mul_83 => mul_83
#   mul_85 => mul_85
#   mul_87 => mul_87
#   mul_89 => mul_89
#   sub_3 => sub_3
#   sub_5 => sub_5
#   sum_10 => sum_10
#   sum_11 => sum_11
#   sum_13 => sum_13
#   sum_14 => sum_14
#   view_48 => view_48
#   view_53 => view_53
#   view_61 => view_61
# Graph fragment:
#   %mul_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=mul_15]
#   %mm_22 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_22]
#   %primals_22 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_22]
#   %mm_26 : Tensor "bf16[131072, 12][12, 1]cuda:5" = PlaceHolder[target=mm_26]
#   %mm_28 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_28]
#   %primals_16 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_16]
#   %add_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_15]
#   %primals_25 : Tensor "f32[2, 384][384, 1]cuda:5" = PlaceHolder[target=primals_25]
#   %sum_10 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:5" = PlaceHolder[target=sum_10]
#   %rsqrt_2 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5" = PlaceHolder[target=rsqrt_2]
#   %sum_13 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:5" = PlaceHolder[target=sum_13]
#   %add_19 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_19]
#   %primals_15 : Tensor "f32[2, 384][384, 1]cuda:5" = PlaceHolder[target=primals_15]
#   %primals_11 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_11]
#   %primals_10 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_10]
#   %full_default : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([128, 1024, 384], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:5, pin_memory: False})
#   %select_12 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_25, 0, 0), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_12, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 1), kwargs = {})
#   %mul_56 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %unsqueeze_19), kwargs = {})
#   %view_48 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_22, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_97 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_48, torch.float32), kwargs = {})
#   %mul_65 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_97, 0.5773502691896258), kwargs = {})
#   %mul_66 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_65, %primals_22), kwargs = {})
#   %select_6 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_6, 0), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 1), kwargs = {})
#   %mul_68 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %mul_66), kwargs = {})
#   %sum_10 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_68, [2], True), kwargs = {})
#   %div_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_15, 384), kwargs = {})
#   %mul_69 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sum_10), kwargs = {})
#   %sub_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_66, %mul_69), kwargs = {})
#   %mul_70 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_71 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_65, %mul_15), kwargs = {})
#   %sum_11 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_71, [0, 1]), kwargs = {})
#   %add_17 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %mul_70), kwargs = {})
#   %view_53 : Tensor "bf16[128, 1024, 12][12288, 12, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_26, [128, 1024, 12]), kwargs = {})
#   %convert_element_type_112 : Tensor "f32[128, 1024, 12][12288, 12, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_53, torch.float32), kwargs = {})
#   %slice_scatter_default_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %convert_element_type_112, 2, 0, 12), kwargs = {})
#   %view_61 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_28, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_118 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_61, torch.float32), kwargs = {})
#   %add_18 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_2, %convert_element_type_118), kwargs = {})
#   %mul_77 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, 0.5773502691896258), kwargs = {})
#   %mul_78 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_77, %primals_16), kwargs = {})
#   %mul_80 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %mul_78), kwargs = {})
#   %sum_13 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_80, [2], True), kwargs = {})
#   %mul_81 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sum_13), kwargs = {})
#   %sub_5 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_78, %mul_81), kwargs = {})
#   %mul_82 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_2), kwargs = {})
#   %mul_83 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_77, %mul_15), kwargs = {})
#   %sum_14 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_83, [0, 1]), kwargs = {})
#   %add_19 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %mul_82), kwargs = {})
#   %mul_85 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %unsqueeze_10), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_11, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 1), kwargs = {})
#   %mul_87 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_8), kwargs = {})
#   %convert_element_type_121 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_87, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_10, 0), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 1), kwargs = {})
#   %mul_89 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_6), kwargs = {})
#   %convert_element_type_122 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_89, torch.bfloat16), kwargs = {})
#   return %sum_10,%sum_13,%add_19,%convert_element_type_121,%convert_element_type_122
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_7 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 12, 'num_store': 1, 'num_reduction': 2, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr2, out_ptr3, out_ptr4, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/fh/cfhuxv26mdqxk4bipfkdpd54qfhrabq3vlpyl6dsiswatplpybjq.py
# Topologically Sorted Source Nodes: [mul_55, sum_6, getitem_22, getitem_23, mul_56, mul_57, sum_7, linear_9, mul_59, sum_8, linear_7, mul_61, sum_9, getitem_11, getitem_12, mul_84, sum_15, mul_85, mul_86, sum_16, linear_4, mul_88, sum_17, linear_2, mul_90, sum_18], Original ATen: [aten.mul, aten.sum, aten.select, aten.unsqueeze, aten._unsafe_view]
# Source node to ATen node mapping:
#   getitem_11 => select_6
#   getitem_12 => unsqueeze_10, unsqueeze_9
#   getitem_22 => select_12
#   getitem_23 => unsqueeze_18, unsqueeze_19
#   linear_2 => view_7
#   linear_4 => view_11
#   linear_7 => view_19
#   linear_9 => view_23
#   mul_55 => mul_55
#   mul_56 => mul_56
#   mul_57 => mul_57
#   mul_59 => mul_59
#   mul_61 => mul_61
#   mul_84 => mul_84
#   mul_85 => mul_85
#   mul_86 => mul_86
#   mul_88 => mul_88
#   mul_90 => mul_90
#   sum_15 => sum_15
#   sum_16 => sum_16
#   sum_17 => sum_17
#   sum_18 => sum_18
#   sum_6 => sum_6
#   sum_7 => sum_7
#   sum_8 => sum_8
#   sum_9 => sum_9
# Graph fragment:
#   %add_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_15]
#   %primals_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=primals_3]
#   %add_9 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_9]
#   %primals_25 : Tensor "f32[2, 384][384, 1]cuda:5" = PlaceHolder[target=primals_25]
#   %mm_9 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_9]
#   %mm_7 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_7]
#   %add_19 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_19]
#   %add_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_4]
#   %primals_15 : Tensor "f32[2, 384][384, 1]cuda:5" = PlaceHolder[target=primals_15]
#   %mm_4 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_4]
#   %mm_2 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_2]
#   %mul_55 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %primals_3), kwargs = {})
#   %sum_6 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_55, [0, 1], True), kwargs = {dtype: torch.float32})
#   %select_12 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_25, 0, 0), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_12, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 1), kwargs = {})
#   %mul_56 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %unsqueeze_19), kwargs = {})
#   %mul_57 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %add_9), kwargs = {})
#   %sum_7 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_57, [0, 1], True), kwargs = {dtype: torch.float32})
#   %view_23 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [128, 1024, 384]), kwargs = {})
#   %mul_59 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_56, %view_23), kwargs = {})
#   %sum_8 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_59, [0, 1], True), kwargs = {dtype: torch.float32})
#   %view_19 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [128, 1024, 384]), kwargs = {})
#   %mul_61 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_56, %view_19), kwargs = {})
#   %sum_9 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_61, [0, 1], True), kwargs = {dtype: torch.float32})
#   %select_6 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_6, 0), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 1), kwargs = {})
#   %mul_84 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %primals_3), kwargs = {})
#   %sum_15 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_84, [0, 1], True), kwargs = {dtype: torch.float32})
#   %mul_85 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %unsqueeze_10), kwargs = {})
#   %mul_86 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %add_4), kwargs = {})
#   %sum_16 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_86, [0, 1], True), kwargs = {dtype: torch.float32})
#   %view_11 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [128, 1024, 384]), kwargs = {})
#   %mul_88 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %view_11), kwargs = {})
#   %sum_17 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_88, [0, 1], True), kwargs = {dtype: torch.float32})
#   %view_7 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [128, 1024, 384]), kwargs = {})
#   %mul_90 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %view_7), kwargs = {})
#   %sum_18 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_90, [0, 1], True), kwargs = {dtype: torch.float32})
#   return %buf25,%buf27,%buf30,%buf32,%buf67,%buf69,%buf72,%buf74
triton_red_fused__unsafe_view_mul_select_sum_unsqueeze_8 = async_compile.triton('triton_red_fused__unsafe_view_mul_select_sum_unsqueeze_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_mul_select_sum_unsqueeze_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 11, 'num_store': 8, 'num_reduction': 8, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1317202944, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_mul_select_sum_unsqueeze_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/sd/csd5bv4zvo5wk6zsaoemets6nn46ctbwhdai5wcwypsxohjsvs43.py
# Topologically Sorted Source Nodes: [mul_55, sum_6], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_55 => mul_55
#   sum_6 => sum_6
# Graph fragment:
#   %buf25 : Tensor "f32[1, 1, 384, 349][134016, 134016, 1, 384]cuda:5" = PlaceHolder[target=buf25]
#   %mul_55 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %primals_3), kwargs = {})
#   %sum_6 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_55, [0, 1], True), kwargs = {dtype: torch.float32})
#   return %sum_6
triton_red_fused_mul_sum_9 = async_compile.triton('triton_red_fused_mul_sum_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 539136, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_mul_sum_9(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/43/c43hpce7t5ilzn5uomdg4afvd3myrqowgivr4qtqnvzbphr4jxna.py
# Topologically Sorted Source Nodes: [squeeze_1, squeeze_2, full_default_1, squeeze_3, squeeze_4, add_16], Original ATen: [aten.squeeze, aten.select_backward, aten.add]
# Source node to ATen node mapping:
#   add_16 => add_16
#   full_default_1 => full_default_1
#   squeeze_1 => squeeze_1
#   squeeze_2 => squeeze_2
#   squeeze_3 => squeeze_3
#   squeeze_4 => squeeze_4
# Graph fragment:
#   %sum_6 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5" = PlaceHolder[target=sum_6]
#   %sum_7 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5" = PlaceHolder[target=sum_7]
#   %squeeze_1 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%sum_6, 1), kwargs = {})
#   %squeeze_2 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%squeeze_1, 0), kwargs = {})
#   %full_default_1 : Tensor "f32[2, 384][384, 1]cuda:5"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([2, 384], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:5, pin_memory: False})
#   %select_scatter_default : Tensor "f32[2, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %squeeze_2, 0, 1), kwargs = {})
#   %squeeze_3 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%sum_7, 1), kwargs = {})
#   %squeeze_4 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%squeeze_3, 0), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[2, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %squeeze_4, 0, 0), kwargs = {})
#   %add_16 : Tensor "f32[2, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   return %add_16
triton_poi_fused_add_select_backward_squeeze_10 = async_compile.triton('triton_poi_fused_add_select_backward_squeeze_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_select_backward_squeeze_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_select_backward_squeeze_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/dg/cdggwtqbbixorwuoesb2rvuxhvbtp6ihyk7dqh52525yp3ovbzli.py
# Topologically Sorted Source Nodes: [full_default, getitem_11, getitem_12, mul_85, view_65, convert_element_type_138, mul_94, mul_95, getitem, getitem_1, mul_97, sum_19, div_4, mul_98, sub_6, mul_99, mul_100, sum_20, add_21, view_70, convert_element_type_153, view_78, convert_element_type_159, add_22, mul_106, mul_107, mul_109, sum_22, mul_110, sub_8, mul_111, mul_112, sum_23, add_23, mul_114], Original ATen: [aten.slice_backward, aten.select, aten.unsqueeze, aten.mul, aten.view, aten._to_copy, aten._fused_rms_norm_backward, aten.add]
# Source node to ATen node mapping:
#   add_21 => add_21
#   add_22 => add_22
#   add_23 => add_23
#   convert_element_type_138 => convert_element_type_138
#   convert_element_type_153 => convert_element_type_153
#   convert_element_type_159 => convert_element_type_159
#   div_4 => div_4
#   full_default => full_default
#   getitem => select
#   getitem_1 => unsqueeze, unsqueeze_1
#   getitem_11 => select_6
#   getitem_12 => unsqueeze_10, unsqueeze_9
#   mul_100 => mul_100
#   mul_106 => mul_106
#   mul_107 => mul_107
#   mul_109 => mul_109
#   mul_110 => mul_110
#   mul_111 => mul_111
#   mul_112 => mul_112
#   mul_114 => mul_114
#   mul_85 => mul_85
#   mul_94 => mul_94
#   mul_95 => mul_95
#   mul_97 => mul_97
#   mul_98 => mul_98
#   mul_99 => mul_99
#   sub_6 => sub_6
#   sub_8 => sub_8
#   sum_19 => sum_19
#   sum_20 => sum_20
#   sum_22 => sum_22
#   sum_23 => sum_23
#   view_65 => view_65
#   view_70 => view_70
#   view_78 => view_78
# Graph fragment:
#   %mul_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=mul_2]
#   %mm_32 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_32]
#   %primals_12 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_12]
#   %mm_36 : Tensor "bf16[131072, 12][12, 1]cuda:5" = PlaceHolder[target=mm_36]
#   %mm_38 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_38]
#   %primals_4 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_4]
#   %add_19 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_19]
#   %primals_15 : Tensor "f32[2, 384][384, 1]cuda:5" = PlaceHolder[target=primals_15]
#   %sum_19 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:5" = PlaceHolder[target=sum_19]
#   %rsqrt : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5" = PlaceHolder[target=rsqrt]
#   %sum_22 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:5" = PlaceHolder[target=sum_22]
#   %add_23 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_23]
#   %primals_1 : Tensor "f32[2, 384][384, 1]cuda:5" = PlaceHolder[target=primals_1]
#   %full_default : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([128, 1024, 384], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:5, pin_memory: False})
#   %select_6 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_6, 0), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 1), kwargs = {})
#   %mul_85 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %unsqueeze_10), kwargs = {})
#   %view_65 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_32, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_138 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_65, torch.float32), kwargs = {})
#   %mul_94 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_138, 0.7071067811865475), kwargs = {})
#   %mul_95 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_94, %primals_12), kwargs = {})
#   %select : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %mul_97 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %mul_95), kwargs = {})
#   %sum_19 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_97, [2], True), kwargs = {})
#   %div_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, 384), kwargs = {})
#   %mul_98 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sum_19), kwargs = {})
#   %sub_6 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_95, %mul_98), kwargs = {})
#   %mul_99 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt), kwargs = {})
#   %mul_100 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_94, %mul_2), kwargs = {})
#   %sum_20 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_100, [0, 1]), kwargs = {})
#   %add_21 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_85, %mul_99), kwargs = {})
#   %view_70 : Tensor "bf16[128, 1024, 12][12288, 12, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_36, [128, 1024, 12]), kwargs = {})
#   %convert_element_type_153 : Tensor "f32[128, 1024, 12][12288, 12, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_70, torch.float32), kwargs = {})
#   %slice_scatter_default_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %convert_element_type_153, 2, 0, 12), kwargs = {})
#   %view_78 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_38, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_159 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_78, torch.float32), kwargs = {})
#   %add_22 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_4, %convert_element_type_159), kwargs = {})
#   %mul_106 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 0.7071067811865475), kwargs = {})
#   %mul_107 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, %primals_4), kwargs = {})
#   %mul_109 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %mul_107), kwargs = {})
#   %sum_22 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_109, [2], True), kwargs = {})
#   %mul_110 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sum_22), kwargs = {})
#   %sub_8 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_107, %mul_110), kwargs = {})
#   %mul_111 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt), kwargs = {})
#   %mul_112 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, %mul_2), kwargs = {})
#   %sum_23 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_112, [0, 1]), kwargs = {})
#   %add_23 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %mul_111), kwargs = {})
#   %mul_114 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, %unsqueeze_1), kwargs = {})
#   return %sum_19,%sum_22,%add_23,%mul_114
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_11 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr2': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 10, 'num_store': 0, 'num_reduction': 2, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr2, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ct/cctbw6ee7xor6dtcqihhjb63qa6bjr33rmzc7hfijjntqznorfj3.py
# Topologically Sorted Source Nodes: [mul_113, sum_24, mul_115, sum_25], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_113 => mul_113
#   mul_115 => mul_115
#   sum_24 => sum_24
#   sum_25 => sum_25
# Graph fragment:
#   %add_23 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_23]
#   %primals_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=primals_3]
#   %primals_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=primals_2]
#   %mul_113 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, %primals_3), kwargs = {})
#   %sum_24 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_113, [0, 1], True), kwargs = {dtype: torch.float32})
#   %mul_115 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, %primals_2), kwargs = {})
#   %sum_25 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_115, [0, 1], True), kwargs = {dtype: torch.float32})
#   return %buf109,%buf112
triton_red_fused_mul_sum_12 = async_compile.triton('triton_red_fused_mul_sum_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 505460736, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_mul_sum_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        primals_1, primals_2, primals_3, primals_4, primals_9, primals_10, primals_11, primals_12, primals_15, primals_16, primals_19, primals_20, primals_21, primals_22, primals_25, primals_26, primals_29, primals_32, rsqrt, view, view_2, select_3, select_5, permute_3, permute_4, permute_5, getitem_2, getitem_3, getitem_8, getitem_9, view_3, mm_1, view_6, mm_2, view_8, mm_3, view_10, mm_4, add_4, rsqrt_2, view_12, view_14, permute_14, permute_15, permute_16, getitem_13, getitem_14, getitem_19, getitem_20, view_15, mm_6, view_18, mm_7, view_20, mm_8, view_22, mm_9, add_9, add_10, rsqrt_4, view_24, view_26, permute_25, permute_26, permute_27, getitem_24, getitem_25, getitem_30, getitem_31, view_27, mm_11, view_30, permute_37, permute_46, permute_50, permute_54, permute_62, permute_71, permute_75, permute_79, permute_87, permute_96, tangents_1, tangents_2, tangents_3 = args
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
        assert_size_stride(mm_3, (131072, 1024), (1024, 1))
        assert_size_stride(view_10, (131072, 1024), (1024, 1))
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
        assert_size_stride(mm_8, (131072, 1024), (1024, 1))
        assert_size_stride(view_22, (131072, 1024), (1024, 1))
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
        assert_size_stride(permute_37, (8, 12), (12, 1))
        assert_size_stride(permute_46, (1152, 384), (384, 1))
        assert_size_stride(permute_50, (384, 1024), (1024, 1))
        assert_size_stride(permute_54, (1024, 384), (384, 1))
        assert_size_stride(permute_62, (8, 12), (12, 1))
        assert_size_stride(permute_71, (1152, 384), (384, 1))
        assert_size_stride(permute_75, (384, 1024), (1024, 1))
        assert_size_stride(permute_79, (1024, 384), (384, 1))
        assert_size_stride(permute_87, (8, 12), (12, 1))
        assert_size_stride(permute_96, (1152, 384), (384, 1))
        assert_size_stride(tangents_1, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(tangents_2, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(tangents_3, (128, 1024, 384), (393216, 384, 1))
        with torch.cuda._DeviceGuard(5):
            torch.cuda.set_device(5)
            buf3 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_32, permute_31, mm_13], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(tangents_2, (384, 131072), (1, 384), 0), view_30, out=buf3)
            del view_30
            buf7 = empty_strided_cuda((131072, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [constant_pad_nd_default_2], Original ATen: [aten.mm]
            stream5 = get_raw_stream(5)
            triton_poi_fused_mm_0.run(view_27, buf7, 2097152, stream=stream5)
            del view_27
            buf49 = empty_strided_cuda((131072, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [constant_pad_nd_default_1], Original ATen: [aten.mm]
            stream5 = get_raw_stream(5)
            triton_poi_fused_mm_0.run(view_15, buf49, 2097152, stream=stream5)
            del view_15
            buf91 = empty_strided_cuda((131072, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [constant_pad_nd_default], Original ATen: [aten.mm]
            stream5 = get_raw_stream(5)
            triton_poi_fused_mm_0.run(view_3, buf91, 2097152, stream=stream5)
            del view_3
            buf4 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_32, linear_12, permute_33, mm_14], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(tangents_2, (131072, 384), (384, 1), 0), primals_29, out=buf4)
            del primals_29
            del tangents_2
            buf11 = empty_strided_cuda((128, 8, 1024, 48), (393216, 48, 384, 1), torch.bfloat16)
            buf6 = empty_strided_cuda((128, 1024, 8), (8192, 8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_33, view_34, transpose_11, mul_43, linear_11, mul_19, sigmoid_2, getitem_30, mul_44, sum_3, convert_element_type_63, squeeze, convert_element_type_64, convert_element_type_65, sub_1, mul_45, mul_46, convert_element_type_66, mul_47, permute_39, _scaled_dot_product_flash_attention_backward], Original ATen: [aten.view, aten.transpose, aten.mul, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten._scaled_dot_product_flash_attention_backward]
            stream5 = get_raw_stream(5)
            triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_transpose_unsqueeze_view_1.run(buf4, getitem_24, mm_11, buf11, buf6, 1048576, 48, stream=stream5)
            del buf4
            del mm_11
            buf8 = empty_strided_cuda((8, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_11, mul_19, sigmoid_2, convert_element_type_63, squeeze, convert_element_type_64, convert_element_type_65, sub_1, mul_45, mul_46, convert_element_type_66, mul_47, view_35, permute_35, constant_pad_nd_default_2, mm_default_2], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf6, (8, 131072), (1, 8), 0), buf7, out=buf8)
            del buf7
            buf10 = empty_strided_cuda((8, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [slice_tensor_2, convert_element_type_72], Original ATen: [aten.mm, aten._to_copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused__to_copy_mm_2.run(buf8, buf10, 96, stream=stream5)
            del buf8
            buf9 = empty_strided_cuda((131072, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_11, mul_19, sigmoid_2, convert_element_type_63, squeeze, convert_element_type_64, convert_element_type_65, sub_1, mul_45, mul_46, convert_element_type_66, mul_47, view_35, mm_16], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf6, (131072, 8), (8, 1), 0), permute_37, out=buf9)
            del buf6
            del permute_37
            # Topologically Sorted Source Nodes: [view_33, view_34, linear_11, mul_19, sigmoid_2, getitem_30, mul_44, permute_39, _scaled_dot_product_flash_attention_backward], Original ATen: [aten.view, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.transpose, aten._scaled_dot_product_flash_attention_backward]
            buf12 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(buf11, permute_26, permute_27, permute_25, getitem_24, getitem_25, None, None, 1024, 1024, 0.0, True, getitem_30, getitem_31, scale=0.14433756729740646)
            del buf11
            del getitem_24
            del getitem_25
            del getitem_30
            del getitem_31
            del permute_25
            del permute_26
            del permute_27
            buf13 = buf12[0]
            assert_size_stride(buf13, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf13, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf14 = buf12[1]
            assert_size_stride(buf14, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf14, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf15 = buf12[2]
            assert_size_stride(buf15, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf15, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            del buf12
            buf17 = empty_strided_cuda((128, 1024, 24, 48), (1179648, 1152, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_40, permute_41, permute_42, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused_copy_slice_transpose_3.run(buf15, buf17, 150994944, stream=stream5)
            # Topologically Sorted Source Nodes: [permute_40, permute_41, permute_42, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream5 = get_raw_stream(5)
            _fused_qkv_postprocess_bwd_kernel_0.run(reinterpret_tensor(buf13, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), reinterpret_tensor(buf14, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), view_26, buf17, select_3, select_5, 2097152, 1, 1, stream=stream5)
            del view_26
            buf19 = empty_strided_cuda((1152, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_40, view_41, permute_44, mm_17], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf17, (1152, 131072), (1, 1152), 0), view_24, out=buf19)
            del view_24
            buf20 = reinterpret_tensor(buf14, (131072, 384), (384, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [view_40, view_41, mm_18], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf17, (131072, 1152), (1152, 1), 0), permute_46, out=buf20)
            del permute_46
            buf24 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            buf34 = reinterpret_tensor(buf13, (128, 1024, 384), (393216, 384, 1), 0); del buf13  # reuse
            buf44 = reinterpret_tensor(buf15, (128, 1024, 384), (393216, 384, 1), 0); del buf15  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [mul_36, mul_37, rms_norm_4, mul_39, sum_1, div, mul_40, sub, mul_41, mul_42, sum_2, add_13, view_36, convert_element_type_71, full_default, view_44, convert_element_type_77, add_14, mul_48, mul_49, mul_51, sum_4, mul_52, sub_2, mul_53, mul_54, sum_5, add_15, getitem_22, getitem_23, mul_56, getitem_21, mul_58, convert_element_type_80, getitem_20, mul_60, convert_element_type_81], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._fused_rms_norm, aten.add, aten.view, aten._to_copy, aten.slice_backward, aten.select, aten.unsqueeze]
            workspace_0 = empty_strided_cuda((1572864, ), (1, ), torch.float32)
            stream5 = get_raw_stream(5)
            triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_4.run(add_10, rsqrt_4, tangents_1, primals_32, buf9, buf20, primals_26, tangents_3, primals_25, primals_21, primals_20, buf24, buf34, buf44, workspace_0, 131072, 384, stream=stream5)
            buf2 = workspace_0[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            buf23 = workspace_0[1 * 2048 * 384 : (1 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del add_10
            del primals_20
            del primals_21
            del primals_26
            del primals_32
            del rsqrt_4
            del tangents_1
            del tangents_3
            buf35 = empty_strided_cuda((384, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_22, getitem_23, mul_56, getitem_21, mul_58, convert_element_type_80, view_45, permute_48, mm_19], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf34, (384, 131072), (1, 384), 0), view_22, out=buf35)
            del view_22
            buf45 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_22, getitem_23, mul_56, getitem_20, mul_60, convert_element_type_81, view_49, permute_56, mm_23], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf44, (384, 131072), (1, 384), 0), view_18, out=buf45)
            del view_18
            buf46 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [getitem_22, getitem_23, mul_56, getitem_20, mul_60, convert_element_type_81, view_49, linear_7, permute_58, mm_24], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf44, (131072, 384), (384, 1), 0), primals_19, out=buf46)
            del primals_19
            buf53 = reinterpret_tensor(buf44, (128, 8, 1024, 48), (393216, 48, 384, 1), 0); del buf44  # reuse
            buf48 = empty_strided_cuda((128, 1024, 8), (8192, 8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_50, view_51, transpose_7, mul_72, linear_6, mul_11, sigmoid_1, getitem_19, mul_73, sum_12, convert_element_type_104, squeeze_9, convert_element_type_105, convert_element_type_106, sub_4, mul_74, mul_75, convert_element_type_107, mul_76, permute_64, _scaled_dot_product_flash_attention_backward_1], Original ATen: [aten.view, aten.transpose, aten.mul, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten._scaled_dot_product_flash_attention_backward]
            stream5 = get_raw_stream(5)
            triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_transpose_unsqueeze_view_1.run(buf46, getitem_13, mm_6, buf53, buf48, 1048576, 48, stream=stream5)
            del buf46
            del mm_6
            # Topologically Sorted Source Nodes: [view_50, view_51, linear_6, mul_11, sigmoid_1, getitem_19, mul_73, permute_64, _scaled_dot_product_flash_attention_backward_1], Original ATen: [aten.view, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.transpose, aten._scaled_dot_product_flash_attention_backward]
            buf54 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(buf53, permute_15, permute_16, permute_14, getitem_13, getitem_14, None, None, 1024, 1024, 0.0, True, getitem_19, getitem_20, scale=0.14433756729740646)
            del buf53
            del getitem_13
            del getitem_14
            del getitem_19
            del getitem_20
            del permute_14
            del permute_15
            del permute_16
            buf50 = empty_strided_cuda((8, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_6, mul_11, sigmoid_1, convert_element_type_104, squeeze_9, convert_element_type_105, convert_element_type_106, sub_4, mul_74, mul_75, convert_element_type_107, mul_76, view_52, permute_60, constant_pad_nd_default_1, mm_default_1], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf48, (8, 131072), (1, 8), 0), buf49, out=buf50)
            del buf49
            buf55 = buf54[0]
            assert_size_stride(buf55, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf55, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf56 = buf54[1]
            assert_size_stride(buf56, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf56, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf57 = buf54[2]
            assert_size_stride(buf57, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf57, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            del buf54
            buf52 = empty_strided_cuda((8, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [slice_tensor_1, convert_element_type_113], Original ATen: [aten.mm, aten._to_copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused__to_copy_mm_2.run(buf50, buf52, 96, stream=stream5)
            buf51 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [linear_6, mul_11, sigmoid_1, convert_element_type_104, squeeze_9, convert_element_type_105, convert_element_type_106, sub_4, mul_74, mul_75, convert_element_type_107, mul_76, view_52, mm_26], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf48, (131072, 8), (8, 1), 0), permute_62, out=buf51)
            del permute_62
            buf36 = empty_strided_cuda((131072, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_22, getitem_23, mul_56, getitem_21, mul_58, convert_element_type_80, view_45, mm_20], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf34, (131072, 384), (384, 1), 0), permute_50, out=buf36)
            del permute_50
            buf37 = reinterpret_tensor(mm_8, (128, 1024, 1024), (1048576, 1024, 1), 0); del mm_8  # reuse
            # Topologically Sorted Source Nodes: [view_46, convert_element_type_86, linear_8, leaky_relu_1, pow_9, mul_62, mul_63, mul_64, where_2, convert_element_type_92], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward]
            stream5 = get_raw_stream(5)
            triton_poi_fused__to_copy__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_view_5.run(buf37, buf36, 134217728, stream=stream5)
            del buf36
            buf38 = empty_strided_cuda((1024, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_46, convert_element_type_86, linear_8, leaky_relu_1, pow_9, mul_62, mul_63, mul_64, where_2, convert_element_type_92, view_47, permute_52, mm_21], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf37, (1024, 131072), (1, 1024), 0), view_20, out=buf38)
            del view_20
            buf39 = reinterpret_tensor(buf34, (131072, 384), (384, 1), 0); del buf34  # reuse
            # Topologically Sorted Source Nodes: [view_46, convert_element_type_86, linear_8, leaky_relu_1, pow_9, mul_62, mul_63, mul_64, where_2, convert_element_type_92, view_47, mm_22], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf37, (131072, 1024), (1024, 1), 0), permute_54, out=buf39)
            del permute_54
            buf59 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [permute_65, permute_66, permute_67, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused_copy_slice_transpose_3.run(buf57, buf59, 150994944, stream=stream5)
            # Topologically Sorted Source Nodes: [permute_65, permute_66, permute_67, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream5 = get_raw_stream(5)
            _fused_qkv_postprocess_bwd_kernel_0.run(reinterpret_tensor(buf55, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), reinterpret_tensor(buf56, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), view_14, buf59, select_3, select_5, 2097152, 1, 1, stream=stream5)
            del view_14
            buf61 = empty_strided_cuda((1152, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_57, view_58, permute_69, mm_27], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf59, (1152, 131072), (1, 1152), 0), view_12, out=buf61)
            del view_12
            buf62 = reinterpret_tensor(buf56, (131072, 384), (384, 1), 0); del buf56  # reuse
            # Topologically Sorted Source Nodes: [view_57, view_58, mm_28], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf59, (131072, 1152), (1152, 1), 0), permute_71, out=buf62)
            del permute_71
            buf40 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            buf82 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_8, getitem_13, getitem_14, mul_9, add_3, rms_norm_2, getitem, getitem_1, mul, getitem_2, getitem_3, mul_1, add, rms_norm], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._fused_rms_norm]
            stream5 = get_raw_stream(5)
            triton_poi_fused__fused_rms_norm_add_mul_select_unsqueeze_6.run(primals_15, add_4, primals_3, rsqrt_2, primals_1, primals_2, rsqrt, buf40, buf82, 50331648, stream=stream5)
            buf66 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            buf76 = reinterpret_tensor(buf55, (128, 1024, 384), (393216, 384, 1), 0); del buf55  # reuse
            buf86 = reinterpret_tensor(buf57, (128, 1024, 384), (393216, 384, 1), 0); del buf57  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [full_default, getitem_22, getitem_23, mul_56, view_48, convert_element_type_97, mul_65, mul_66, getitem_11, getitem_12, mul_68, sum_10, div_2, mul_69, sub_3, mul_70, mul_71, sum_11, add_17, view_53, convert_element_type_112, view_61, convert_element_type_118, add_18, mul_77, mul_78, mul_80, sum_13, mul_81, sub_5, mul_82, mul_83, sum_14, add_19, mul_85, getitem_10, mul_87, convert_element_type_121, getitem_9, mul_89, convert_element_type_122], Original ATen: [aten.slice_backward, aten.select, aten.unsqueeze, aten.mul, aten.view, aten._to_copy, aten._fused_rms_norm_backward, aten.add]
            workspace_1 = workspace_0; del workspace_0  # reuse
            stream5 = get_raw_stream(5)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_7.run(buf40, buf39, primals_22, buf51, buf62, primals_16, buf24, primals_25, rsqrt_2, primals_15, primals_11, primals_10, buf66, buf76, buf86, workspace_1, 131072, 384, stream=stream5)
            buf43 = workspace_1[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            buf65 = workspace_1[1 * 2048 * 384 : (1 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del buf39
            del buf40
            del primals_10
            del primals_11
            del primals_16
            del primals_22
            del rsqrt_2
            buf25 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf27 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf30 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf32 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf67 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf69 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf72 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            buf74 = empty_strided_cuda((1, 1, 384, 349), (134016, 134016, 1, 384), torch.float32)
            # Topologically Sorted Source Nodes: [mul_55, sum_6, getitem_22, getitem_23, mul_56, mul_57, sum_7, linear_9, mul_59, sum_8, linear_7, mul_61, sum_9, getitem_11, getitem_12, mul_84, sum_15, mul_85, mul_86, sum_16, linear_4, mul_88, sum_17, linear_2, mul_90, sum_18], Original ATen: [aten.mul, aten.sum, aten.select, aten.unsqueeze, aten._unsafe_view]
            stream5 = get_raw_stream(5)
            triton_red_fused__unsafe_view_mul_select_sum_unsqueeze_8.run(buf24, primals_3, add_9, primals_25, mm_9, mm_7, buf66, add_4, primals_15, mm_4, mm_2, buf25, buf27, buf30, buf32, buf67, buf69, buf72, buf74, 134016, 376, stream=stream5)
            del add_4
            del add_9
            del mm_2
            del mm_4
            del mm_7
            del mm_9
            del primals_25
            buf77 = empty_strided_cuda((384, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_85, getitem_10, mul_87, convert_element_type_121, view_62, permute_73, mm_29], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf76, (384, 131072), (1, 384), 0), view_10, out=buf77)
            del view_10
            buf87 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_85, getitem_9, mul_89, convert_element_type_122, view_66, permute_81, mm_33], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf86, (384, 131072), (1, 384), 0), view_6, out=buf87)
            del view_6
            buf26 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_55, sum_6], Original ATen: [aten.mul, aten.sum]
            stream5 = get_raw_stream(5)
            triton_red_fused_mul_sum_9.run(buf25, buf26, 384, 349, stream=stream5)
            del buf25
            buf28 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_57, sum_7], Original ATen: [aten.mul, aten.sum]
            stream5 = get_raw_stream(5)
            triton_red_fused_mul_sum_9.run(buf27, buf28, 384, 349, stream=stream5)
            del buf27
            buf31 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_22, getitem_23, mul_56, linear_9, mul_59, sum_8], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._unsafe_view, aten.sum]
            stream5 = get_raw_stream(5)
            triton_red_fused_mul_sum_9.run(buf30, buf31, 384, 349, stream=stream5)
            del buf30
            buf33 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_22, getitem_23, mul_56, linear_7, mul_61, sum_9], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._unsafe_view, aten.sum]
            stream5 = get_raw_stream(5)
            triton_red_fused_mul_sum_9.run(buf32, buf33, 384, 349, stream=stream5)
            del buf32
            buf68 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_84, sum_15], Original ATen: [aten.mul, aten.sum]
            stream5 = get_raw_stream(5)
            triton_red_fused_mul_sum_9.run(buf67, buf68, 384, 349, stream=stream5)
            del buf67
            buf70 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_86, sum_16], Original ATen: [aten.mul, aten.sum]
            stream5 = get_raw_stream(5)
            triton_red_fused_mul_sum_9.run(buf69, buf70, 384, 349, stream=stream5)
            del buf69
            buf73 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_85, linear_4, mul_88, sum_17], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._unsafe_view, aten.sum]
            stream5 = get_raw_stream(5)
            triton_red_fused_mul_sum_9.run(buf72, buf73, 384, 349, stream=stream5)
            buf75 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_85, linear_2, mul_90, sum_18], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._unsafe_view, aten.sum]
            stream5 = get_raw_stream(5)
            triton_red_fused_mul_sum_9.run(buf74, buf75, 384, 349, stream=stream5)
            buf29 = empty_strided_cuda((2, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [squeeze_1, squeeze_2, full_default_1, squeeze_3, squeeze_4, add_16], Original ATen: [aten.squeeze, aten.select_backward, aten.add]
            stream5 = get_raw_stream(5)
            triton_poi_fused_add_select_backward_squeeze_10.run(buf26, buf28, buf29, 768, stream=stream5)
            del buf26
            del buf28
            buf71 = empty_strided_cuda((2, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_default_1, squeeze_10, squeeze_11, squeeze_12, squeeze_13, add_20], Original ATen: [aten.select_backward, aten.squeeze, aten.add]
            stream5 = get_raw_stream(5)
            triton_poi_fused_add_select_backward_squeeze_10.run(buf68, buf70, buf71, 768, stream=stream5)
            buf88 = buf62; del buf62  # reuse
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_85, getitem_9, mul_89, convert_element_type_122, view_66, linear_2, permute_83, mm_34], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf86, (131072, 384), (384, 1), 0), primals_9, out=buf88)
            del primals_9
            buf95 = reinterpret_tensor(buf86, (128, 8, 1024, 48), (393216, 48, 384, 1), 0); del buf86  # reuse
            buf90 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [view_67, view_68, transpose_3, mul_101, linear_1, mul_3, sigmoid, getitem_8, mul_102, sum_21, convert_element_type_145, squeeze_18, convert_element_type_146, convert_element_type_147, sub_7, mul_103, mul_104, convert_element_type_148, mul_105, permute_89, _scaled_dot_product_flash_attention_backward_2], Original ATen: [aten.view, aten.transpose, aten.mul, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten._scaled_dot_product_flash_attention_backward]
            stream5 = get_raw_stream(5)
            triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_transpose_unsqueeze_view_1.run(buf88, getitem_2, mm_1, buf95, buf90, 1048576, 48, stream=stream5)
            del buf88
            del mm_1
            # Topologically Sorted Source Nodes: [view_67, view_68, linear_1, mul_3, sigmoid, getitem_8, mul_102, permute_89, _scaled_dot_product_flash_attention_backward_2], Original ATen: [aten.view, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.transpose, aten._scaled_dot_product_flash_attention_backward]
            buf96 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(buf95, permute_4, permute_5, permute_3, getitem_2, getitem_3, None, None, 1024, 1024, 0.0, True, getitem_8, getitem_9, scale=0.14433756729740646)
            del buf95
            del getitem_2
            del getitem_3
            del getitem_8
            del getitem_9
            del permute_3
            del permute_4
            del permute_5
            buf92 = buf50; del buf50  # reuse
            # Topologically Sorted Source Nodes: [linear_1, mul_3, sigmoid, convert_element_type_145, squeeze_18, convert_element_type_146, convert_element_type_147, sub_7, mul_103, mul_104, convert_element_type_148, mul_105, view_69, permute_85, constant_pad_nd_default, mm_default], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf90, (8, 131072), (1, 8), 0), buf91, out=buf92)
            del buf91
            buf99 = buf96[2]
            assert_size_stride(buf99, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf99, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf97 = buf96[0]
            assert_size_stride(buf97, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf97, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf98 = buf96[1]
            assert_size_stride(buf98, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf98, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            del buf96
            buf94 = empty_strided_cuda((8, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [slice_tensor, convert_element_type_154], Original ATen: [aten.mm, aten._to_copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused__to_copy_mm_2.run(buf92, buf94, 96, stream=stream5)
            del buf92
            buf93 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [linear_1, mul_3, sigmoid, convert_element_type_145, squeeze_18, convert_element_type_146, convert_element_type_147, sub_7, mul_103, mul_104, convert_element_type_148, mul_105, view_69, mm_36], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf90, (131072, 8), (8, 1), 0), permute_87, out=buf93)
            del buf90
            del permute_87
            buf78 = reinterpret_tensor(buf37, (131072, 1024), (1024, 1), 0); del buf37  # reuse
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_85, getitem_10, mul_87, convert_element_type_121, view_62, mm_30], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf76, (131072, 384), (384, 1), 0), permute_75, out=buf78)
            del permute_75
            buf79 = reinterpret_tensor(mm_3, (128, 1024, 1024), (1048576, 1024, 1), 0); del mm_3  # reuse
            # Topologically Sorted Source Nodes: [view_63, convert_element_type_127, linear_3, leaky_relu, pow_10, mul_91, mul_92, mul_93, where_3, convert_element_type_133], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward]
            stream5 = get_raw_stream(5)
            triton_poi_fused__to_copy__unsafe_view_leaky_relu_leaky_relu_backward_mul_pow_view_5.run(buf79, buf78, 134217728, stream=stream5)
            del buf78
            buf80 = empty_strided_cuda((1024, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_63, convert_element_type_127, linear_3, leaky_relu, pow_10, mul_91, mul_92, mul_93, where_3, convert_element_type_133, view_64, permute_77, mm_31], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf79, (1024, 131072), (1, 1024), 0), view_8, out=buf80)
            del view_8
            buf81 = reinterpret_tensor(buf76, (131072, 384), (384, 1), 0); del buf76  # reuse
            # Topologically Sorted Source Nodes: [view_63, convert_element_type_127, linear_3, leaky_relu, pow_10, mul_91, mul_92, mul_93, where_3, convert_element_type_133, view_64, mm_32], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.leaky_relu, aten.pow, aten.mul, aten.leaky_relu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf79, (131072, 1024), (1024, 1), 0), permute_79, out=buf81)
            del buf79
            del permute_79
            buf101 = buf59; del buf59  # reuse
            # Topologically Sorted Source Nodes: [permute_90, permute_91, permute_92, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused_copy_slice_transpose_3.run(buf99, buf101, 150994944, stream=stream5)
            del buf99
            # Topologically Sorted Source Nodes: [permute_90, permute_91, permute_92, getitem, copy_, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose, aten.slice, aten.copy]
            stream5 = get_raw_stream(5)
            _fused_qkv_postprocess_bwd_kernel_0.run(reinterpret_tensor(buf97, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), reinterpret_tensor(buf98, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), view_2, buf101, select_3, select_5, 2097152, 1, 1, stream=stream5)
            del buf97
            del select_3
            del select_5
            del view_2
            buf103 = empty_strided_cuda((1152, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_74, view_75, permute_94, mm_37], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf101, (1152, 131072), (1, 1152), 0), view, out=buf103)
            del view
            buf104 = reinterpret_tensor(buf98, (131072, 384), (384, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [view_74, view_75, mm_38], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf101, (131072, 1152), (1152, 1), 0), permute_96, out=buf104)
            del buf101
            del permute_96
            buf108 = buf66; del buf66  # reuse
            buf111 = buf24; del buf24  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [full_default, getitem_11, getitem_12, mul_85, view_65, convert_element_type_138, mul_94, mul_95, getitem, getitem_1, mul_97, sum_19, div_4, mul_98, sub_6, mul_99, mul_100, sum_20, add_21, view_70, convert_element_type_153, view_78, convert_element_type_159, add_22, mul_106, mul_107, mul_109, sum_22, mul_110, sub_8, mul_111, mul_112, sum_23, add_23, mul_114], Original ATen: [aten.slice_backward, aten.select, aten.unsqueeze, aten.mul, aten.view, aten._to_copy, aten._fused_rms_norm_backward, aten.add]
            workspace_2 = workspace_1; del workspace_1  # reuse
            stream5 = get_raw_stream(5)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_11.run(buf108, buf82, buf81, primals_12, buf93, buf104, primals_4, primals_15, rsqrt, primals_1, buf111, workspace_2, 131072, 384, stream=stream5)
            buf85 = workspace_2[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            buf107 = workspace_2[1 * 2048 * 384 : (1 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del workspace_2
            del buf104
            del buf81
            del buf82
            del buf93
            del primals_1
            del primals_12
            del primals_15
            del primals_4
            del rsqrt
            buf109 = buf74; del buf74  # reuse
            buf112 = buf72; del buf72  # reuse
            # Topologically Sorted Source Nodes: [mul_113, sum_24, mul_115, sum_25], Original ATen: [aten.mul, aten.sum]
            stream5 = get_raw_stream(5)
            triton_red_fused_mul_sum_12.run(buf108, primals_3, primals_2, buf109, buf112, 134016, 376, stream=stream5)
            del buf108
            del primals_2
            del primals_3
            buf110 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [mul_113, sum_24], Original ATen: [aten.mul, aten.sum]
            stream5 = get_raw_stream(5)
            triton_red_fused_mul_sum_9.run(buf109, buf110, 384, 349, stream=stream5)
            del buf109
            buf113 = buf68; del buf68  # reuse
            # Topologically Sorted Source Nodes: [mul_115, sum_25], Original ATen: [aten.mul, aten.sum]
            stream5 = get_raw_stream(5)
            triton_red_fused_mul_sum_9.run(buf112, buf113, 384, 349, stream=stream5)
            del buf112
            buf114 = empty_strided_cuda((2, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_default_1, squeeze_19, squeeze_20, squeeze_21, squeeze_22, add_24], Original ATen: [aten.select_backward, aten.squeeze, aten.add]
            stream5 = get_raw_stream(5)
            triton_poi_fused_add_select_backward_squeeze_10.run(buf110, buf113, buf114, 768, stream=stream5)
            del buf110
            del buf113
        return (buf114, buf111, None, buf107, buf103, None, None, buf94, buf87, reinterpret_tensor(buf75, (384, ), (1, ), 0), reinterpret_tensor(buf73, (384, ), (1, ), 0), buf85, buf80, buf77, buf71, buf65, buf61, buf52, buf45, reinterpret_tensor(buf33, (384, ), (1, ), 0), reinterpret_tensor(buf31, (384, ), (1, ), 0), buf43, buf38, buf35, buf29, buf23, buf19, buf10, buf3, None, None, buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((2, 384), (384, 1), device='cuda:5', dtype=torch.float32)
    primals_2 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.float32)
    primals_3 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    primals_4 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_9 = rand_strided((384, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    primals_10 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_11 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_12 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_15 = rand_strided((2, 384), (384, 1), device='cuda:5', dtype=torch.float32)
    primals_16 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_19 = rand_strided((384, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    primals_20 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_21 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_22 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_25 = rand_strided((2, 384), (384, 1), device='cuda:5', dtype=torch.float32)
    primals_26 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_29 = rand_strided((384, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    primals_32 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    rsqrt = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:5', dtype=torch.float32)
    view = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    view_2 = rand_strided((128, 1024, 24, 48), (1179648, 1152, 48, 1), device='cuda:5', dtype=torch.bfloat16)
    select_3 = rand_strided((1024, 24), (24, 1), device='cuda:5', dtype=torch.bfloat16)
    select_5 = rand_strided((1024, 24), (24, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_3 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_4 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_5 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_2 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_3 = rand_strided((128, 8, 1024), (8192, 1024, 1), device='cuda:5', dtype=torch.float32)
    getitem_8 = rand_strided((2, ), (1, ), device='cuda:5', dtype=torch.uint64)
    getitem_9 = rand_strided((), (), device='cuda:5', dtype=torch.uint64)
    view_3 = rand_strided((131072, 12), (12, 1), device='cuda:5', dtype=torch.bfloat16)
    mm_1 = rand_strided((131072, 8), (8, 1), device='cuda:5', dtype=torch.bfloat16)
    view_6 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    mm_2 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    view_8 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    mm_3 = rand_strided((131072, 1024), (1024, 1), device='cuda:5', dtype=torch.bfloat16)
    view_10 = rand_strided((131072, 1024), (1024, 1), device='cuda:5', dtype=torch.bfloat16)
    mm_4 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    add_4 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.float32)
    rsqrt_2 = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:5', dtype=torch.float32)
    view_12 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    view_14 = rand_strided((128, 1024, 24, 48), (1179648, 1152, 48, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_14 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_15 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_16 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_13 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_14 = rand_strided((128, 8, 1024), (8192, 1024, 1), device='cuda:5', dtype=torch.float32)
    getitem_19 = rand_strided((2, ), (1, ), device='cuda:5', dtype=torch.uint64)
    getitem_20 = rand_strided((), (), device='cuda:5', dtype=torch.uint64)
    view_15 = rand_strided((131072, 12), (12, 1), device='cuda:5', dtype=torch.bfloat16)
    mm_6 = rand_strided((131072, 8), (8, 1), device='cuda:5', dtype=torch.bfloat16)
    view_18 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    mm_7 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    view_20 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    mm_8 = rand_strided((131072, 1024), (1024, 1), device='cuda:5', dtype=torch.bfloat16)
    view_22 = rand_strided((131072, 1024), (1024, 1), device='cuda:5', dtype=torch.bfloat16)
    mm_9 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    add_9 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.float32)
    add_10 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.float32)
    rsqrt_4 = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:5', dtype=torch.float32)
    view_24 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    view_26 = rand_strided((128, 1024, 24, 48), (1179648, 1152, 48, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_25 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_26 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_27 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_24 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_25 = rand_strided((128, 8, 1024), (8192, 1024, 1), device='cuda:5', dtype=torch.float32)
    getitem_30 = rand_strided((2, ), (1, ), device='cuda:5', dtype=torch.uint64)
    getitem_31 = rand_strided((), (), device='cuda:5', dtype=torch.uint64)
    view_27 = rand_strided((131072, 12), (12, 1), device='cuda:5', dtype=torch.bfloat16)
    mm_11 = rand_strided((131072, 8), (8, 1), device='cuda:5', dtype=torch.bfloat16)
    view_30 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_37 = rand_strided((8, 12), (12, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_46 = rand_strided((1152, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_50 = rand_strided((384, 1024), (1024, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_54 = rand_strided((1024, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_62 = rand_strided((8, 12), (12, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_71 = rand_strided((1152, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_75 = rand_strided((384, 1024), (1024, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_79 = rand_strided((1024, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_87 = rand_strided((8, 12), (12, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_96 = rand_strided((1152, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    tangents_1 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.float32)
    tangents_2 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    tangents_3 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_9, primals_10, primals_11, primals_12, primals_15, primals_16, primals_19, primals_20, primals_21, primals_22, primals_25, primals_26, primals_29, primals_32, rsqrt, view, view_2, select_3, select_5, permute_3, permute_4, permute_5, getitem_2, getitem_3, getitem_8, getitem_9, view_3, mm_1, view_6, mm_2, view_8, mm_3, view_10, mm_4, add_4, rsqrt_2, view_12, view_14, permute_14, permute_15, permute_16, getitem_13, getitem_14, getitem_19, getitem_20, view_15, mm_6, view_18, mm_7, view_20, mm_8, view_22, mm_9, add_9, add_10, rsqrt_4, view_24, view_26, permute_25, permute_26, permute_27, getitem_24, getitem_25, getitem_30, getitem_31, view_27, mm_11, view_30, permute_37, permute_46, permute_50, permute_54, permute_62, permute_71, permute_75, permute_79, permute_87, permute_96, tangents_1, tangents_2, tangents_3]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
