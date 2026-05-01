# AOT ID: ['31_inference']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/6s/c6stwfjeaa3emmw3pw4ckvehj6v6eeuig2nxu3ogjlvpnqczxf2h.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_2
# Graph fragment:
#   %arg1_1 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6" = PlaceHolder[target=arg1_1]
#   %convert_element_type_2 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1075200000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134400000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/jt/cjt3jxuer4qj6vzboyb45zpj5l6o3dky2ljdmh7gbpxaao5cf6wm.py
# Topologically Sorted Source Nodes: [linear, rms_norm, to_1], Original ATen: [aten._unsafe_view, aten._fused_rms_norm, aten._to_copy]
# Source node to ATen node mapping:
#   linear => view_1
#   rms_norm => add, convert_element_type_6, convert_element_type_7, mean, mul, mul_1, pow_1, rsqrt
#   to_1 => convert_element_type_5
# Graph fragment:
#   %mm : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm]
#   %buf2 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:6" = PlaceHolder[target=buf2]
#   %arg2_1 : Tensor "f32[384][1]cuda:6" = PlaceHolder[target=arg2_1]
#   %view_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [350, 1000, 384]), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_6, 2), kwargs = {})
#   %mean : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %rsqrt), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[384][1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.bfloat16), kwargs = {})
#   %mul_1 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %convert_element_type_5), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   return %buf2,%convert_element_type_7
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_1 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 806401536}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_1(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tl.full([1, 1], 384.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tmp9 = tl.full([1, 1], 1e-06, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp1 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp12 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp17, r0_mask & xmask)
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
    triton_meta={'signature': {'QKV': '*bf16', 'Q': '*bf16', 'K': '*bf16', 'COS': '*bf16', 'SIN': '*bf16', 'TOTAL_ROWS': 'constexpr', 'SEQLEN': 'constexpr', 'H_Q': 'constexpr', 'H_KV': 'constexpr', 'HEAD_DIM': 'constexpr', 'ROPE_DIMS': 'constexpr', 'DO_QK_NORM': 'constexpr', 'ROPE_INTERLEAVED': 'constexpr', 'RMS_EPS': 'constexpr', 'BLOCK_D': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'TOTAL_ROWS': 350000, 'SEQLEN': 1000, 'H_Q': 8, 'H_KV': 4, 'HEAD_DIM': 48, 'ROPE_DIMS': 48, 'DO_QK_NORM': True, 'ROPE_INTERLEAVED': False, 'RMS_EPS': 1e-06, 'BLOCK_D': 64}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/qn/cqn4av464o7ae3ljpjpfmga3xcjjadgq5xo735gaivsvviv7ii7v.py
# Topologically Sorted Source Nodes: [linear_1, view, scaled_dot_product_attention, getitem, contiguous, transpose_2, normalize], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   contiguous => clone
#   getitem => slice_1
#   linear_1 => view_3
#   normalize => convert_element_type_10, pow_2, sum_1
#   scaled_dot_product_attention => _scaled_dot_product_flash_attention, permute_5, permute_6
#   transpose_2 => permute_4
#   view => view_4
# Graph fragment:
#   %mm_1 : Tensor "bf16[350000, 768][768, 1]cuda:6" = PlaceHolder[target=mm_1]
#   %view_3 : Tensor "bf16[350, 1000, 768][768000, 768, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [350, 1000, 768]), kwargs = {})
#   %view_4 : Tensor "bf16[350, 1000, 16, 48][768000, 768, 48, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_3, [350, 1000, 16, 48]), kwargs = {})
#   %permute_5 : Tensor "bf16[350, 8, 1000, 48][384000, 48, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty, [0, 2, 1, 3]), kwargs = {})
#   %permute_6 : Tensor "bf16[350, 4, 1000, 48][192000, 48, 192, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty_1, [0, 2, 1, 3]), kwargs = {})
#   %slice_1 : Tensor "bf16[350, 1000, 4, 48][768000, 768, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_4, 2, 12, 9223372036854775807), kwargs = {})
#   %clone : Tensor "bf16[350, 1000, 4, 48][192000, 192, 48, 1]cuda:6"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%slice_1,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_4 : Tensor "bf16[350, 4, 1000, 48][192000, 48, 192, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%clone, [0, 2, 1, 3]), kwargs = {})
#   %_scaled_dot_product_flash_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%permute_5, %permute_6, %permute_4, 0.0, True), kwargs = {scale: 0.14433756729740646})
#   %convert_element_type_10 : Tensor "f32[350, 1000, 4, 48][192000, 192, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[350, 1000, 4, 48][192000, 192, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_10, 2.0), kwargs = {})
#   %sum_1 : Tensor "f32[350, 1000, 4, 1][4000, 4, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [-1], True), kwargs = {})
#   return %buf9,%sum_1
triton_per_fused__scaled_dot_product_flash_attention__unsafe_view_clone_linalg_vector_norm_slice_transpose_view_2 = async_compile.triton('triton_per_fused__scaled_dot_product_flash_attention__unsafe_view_clone_linalg_vector_norm_slice_transpose_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2097152, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_dot_product_flash_attention__unsafe_view_clone_linalg_vector_norm_slice_transpose_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 11200000, 'r0_': 403200000}}
)
@triton.jit
def triton_per_fused__scaled_dot_product_flash_attention__unsafe_view_clone_linalg_vector_norm_slice_transpose_view_2(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1400000
    r0_numel = 48
    R0_BLOCK: tl.constexpr = 64
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
    r0_2 = r0_index
    x0 = (xindex % 4)
    x1 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (576 + r0_2 + 48*x0 + 768*x1), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (r0_2 + 48*x3), tmp0, r0_mask & xmask)
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/vj/cvj37mpqilpwd3flqntww7bkweyxrbmbxkcitjdtkziualoud3he.py
# Topologically Sorted Source Nodes: [linear_1, view, getitem, contiguous, transpose_3, view_1, normalize, unsqueeze, mul, sum_1], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.unsqueeze, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   contiguous => clone
#   getitem => slice_1
#   linear_1 => view_3
#   mul => mul_2
#   normalize => clamp_min, div, expand, pow_3
#   sum_1 => sum_2
#   transpose_3 => permute_7
#   unsqueeze => unsqueeze
#   view => view_4
#   view_1 => view_5
# Graph fragment:
#   %getitem_2 : Tensor "bf16[350, 8, 1000, 48][384000, 48, 384, 1]cuda:6" = PlaceHolder[target=getitem_2]
#   %mm_1 : Tensor "bf16[350000, 768][768, 1]cuda:6" = PlaceHolder[target=mm_1]
#   %sum_1 : Tensor "f32[350, 1000, 4, 1][4000, 4, 1, 1400000]cuda:6" = PlaceHolder[target=sum_1]
#   %view_3 : Tensor "bf16[350, 1000, 768][768000, 768, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [350, 1000, 768]), kwargs = {})
#   %view_4 : Tensor "bf16[350, 1000, 16, 48][768000, 768, 48, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_3, [350, 1000, 16, 48]), kwargs = {})
#   %slice_1 : Tensor "bf16[350, 1000, 4, 48][768000, 768, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_4, 2, 12, 9223372036854775807), kwargs = {})
#   %clone : Tensor "bf16[350, 1000, 4, 48][192000, 192, 48, 1]cuda:6"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%slice_1,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_7 : Tensor "bf16[350, 1000, 8, 48][384000, 384, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [0, 2, 1, 3]), kwargs = {})
#   %view_5 : Tensor "bf16[350, 1000, 4, 2, 48][384000, 384, 96, 48, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_7, [350, 1000, 4, 2, 48]), kwargs = {})
#   %pow_3 : Tensor "f32[350, 1000, 4, 1][4000, 4, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min : Tensor "f32[350, 1000, 4, 1][4000, 4, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_3, 1e-12), kwargs = {})
#   %expand : Tensor "f32[350, 1000, 4, 48][4000, 4, 1, 0]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min, [350, 1000, 4, 48]), kwargs = {})
#   %div : Tensor "f32[350, 1000, 4, 48][192000, 192, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clone, %expand), kwargs = {})
#   %unsqueeze : Tensor "f32[350, 1000, 4, 1, 48][192000, 192, 48, 48, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%div, -2), kwargs = {})
#   %mul_2 : Tensor "f32[350, 1000, 4, 2, 48][384000, 384, 96, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %unsqueeze), kwargs = {})
#   %sum_2 : Tensor "f32[350, 1000, 4, 2, 1][8000, 8, 2, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [-1], True), kwargs = {dtype: torch.float32})
#   return %sum_2
triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_slice_sum_transpose_unsqueeze_view_3 = async_compile.triton('triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_slice_sum_transpose_unsqueeze_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4194304, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_slice_sum_transpose_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 22400000, 'r0_': 403200000}}
)
@triton.jit
def triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_slice_sum_transpose_unsqueeze_view_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2800000
    r0_numel = 48
    R0_BLOCK: tl.constexpr = 64
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
    r0_3 = r0_index
    x4 = xindex
    x1 = ((xindex // 2) % 4)
    x2 = xindex // 8
    x5 = xindex // 2
    tmp0 = tl.load(in_ptr0 + (r0_3 + 48*x4), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (576 + r0_3 + 48*x1 + 768*x2), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x5), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tl.sqrt_rn(tmp4)
    tmp6 = tl.full([1, 1], 1e-12, tl.float32)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = (tmp3 / tmp7)
    tmp9 = tmp1 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/2p/c2phjq3hyzwiktshgdoevx6o7jfmzv5e2e7zhgyb2vxyrg5u74tf.py
# Topologically Sorted Source Nodes: [getitem_3, contiguous_7], Original ATen: [aten.slice, aten.clone]
# Source node to ATen node mapping:
#   contiguous_7 => clone_1
#   getitem_3 => slice_2
# Graph fragment:
#   %convert_element_type_7 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6" = PlaceHolder[target=convert_element_type_7]
#   %slice_2 : Tensor "bf16[350, 1000, 12][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_7, 2, 0, 12), kwargs = {})
#   %clone_1 : Tensor "bf16[350, 1000, 12][12000, 12, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused_clone_slice_4 = async_compile.triton('triton_poi_fused_clone_slice_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_slice_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 25200000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_slice_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4200000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 12)
    x3 = xindex // 12
    x2 = xindex // 12000
    x4 = (xindex % 12000)
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x3), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x4 + 12032*x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/3x/c3x3qlfssvki5dhbnyftwvzolur265rkznif6gnhavcpaqvklfa3.py
# Topologically Sorted Source Nodes: [getitem_3, contiguous_7, linear_2, to_3], Original ATen: [aten.slice, aten.clone, aten.view, aten._to_copy, aten.t, aten.mm]
# Source node to ATen node mapping:
#   contiguous_7 => clone_1
#   getitem_3 => slice_2
#   linear_2 => mm_2, permute_8, view_7
#   to_3 => convert_element_type_11
# Graph fragment:
#   %clone_1 : Tensor "bf16[350, 1000, 12][12032, 12, 1]cuda:6" = PlaceHolder[target=clone_1]
#   %slice_2 : Tensor "bf16[350, 1000, 12][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_7, 2, 0, 12), kwargs = {})
#   %clone_1 : Tensor "bf16[350, 1000, 12][12000, 12, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_7 : Tensor "bf16[350000, 12][12, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_1, [350000, 12]), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[8, 12][12, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.bfloat16), kwargs = {})
#   %permute_8 : Tensor "bf16[12, 8][1, 12]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_11, [1, 0]), kwargs = {})
#   %mm_2 : Tensor "bf16[350000, 8][8, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_7, %permute_8), kwargs = {})
#   return %buf19
triton_poi_fused__to_copy_clone_mm_slice_t_view_5 = async_compile.triton('triton_poi_fused__to_copy_clone_mm_slice_t_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_mm_slice_t_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 25200000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_mm_slice_t_view_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/3t/c3t7jrx265zjid3h5pdy32wsu3wledxbfu7pzvek2fgucryysc7w.py
# Topologically Sorted Source Nodes: [to_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   to_3 => convert_element_type_11
# Graph fragment:
#   %arg6_1 : Tensor "f32[8, 12][12, 1]cuda:6" = PlaceHolder[target=arg6_1]
#   %convert_element_type_11 : Tensor "bf16[8, 12][12, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_11
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/cz/cczzukaey5cl4rloulj3wsap3r73krlcwifpcsph6eczl65yf5ef.py
# Topologically Sorted Source Nodes: [linear_1, view, getitem, contiguous, transpose_3, view_1, normalize, unsqueeze, mul_1, sub, reshape, linear_2, mul_2, sigmoid, getitem_4, mul_3, reshape_1, linear_3], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.unsqueeze, aten.mul, aten.sub, aten.sigmoid, aten._to_copy]
# Source node to ATen node mapping:
#   contiguous => clone
#   getitem => slice_1
#   getitem_4 => unsqueeze_1
#   linear_1 => view_3
#   linear_2 => view_8
#   linear_3 => convert_element_type_16
#   mul_1 => mul_3
#   mul_2 => mul_4
#   mul_3 => mul_5
#   normalize => clamp_min, div, expand, pow_3
#   reshape => view_6
#   reshape_1 => view_9
#   sigmoid => sigmoid
#   sub => sub
#   transpose_3 => permute_7
#   unsqueeze => unsqueeze
#   view => view_4
#   view_1 => view_5
# Graph fragment:
#   %getitem_2 : Tensor "bf16[350, 8, 1000, 48][384000, 48, 384, 1]cuda:6" = PlaceHolder[target=getitem_2]
#   %sum_2 : Tensor "f32[350, 1000, 4, 2, 1][8000, 8, 2, 1, 2800000]cuda:6" = PlaceHolder[target=sum_2]
#   %mm_1 : Tensor "bf16[350000, 768][768, 1]cuda:6" = PlaceHolder[target=mm_1]
#   %sum_1 : Tensor "f32[350, 1000, 4, 1][4000, 4, 1, 1400000]cuda:6" = PlaceHolder[target=sum_1]
#   %mm_2 : Tensor "bf16[350000, 8][8, 1]cuda:6" = PlaceHolder[target=mm_2]
#   %view_3 : Tensor "bf16[350, 1000, 768][768000, 768, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [350, 1000, 768]), kwargs = {})
#   %view_4 : Tensor "bf16[350, 1000, 16, 48][768000, 768, 48, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_3, [350, 1000, 16, 48]), kwargs = {})
#   %slice_1 : Tensor "bf16[350, 1000, 4, 48][768000, 768, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_4, 2, 12, 9223372036854775807), kwargs = {})
#   %clone : Tensor "bf16[350, 1000, 4, 48][192000, 192, 48, 1]cuda:6"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%slice_1,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_7 : Tensor "bf16[350, 1000, 8, 48][384000, 384, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [0, 2, 1, 3]), kwargs = {})
#   %view_5 : Tensor "bf16[350, 1000, 4, 2, 48][384000, 384, 96, 48, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_7, [350, 1000, 4, 2, 48]), kwargs = {})
#   %pow_3 : Tensor "f32[350, 1000, 4, 1][4000, 4, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min : Tensor "f32[350, 1000, 4, 1][4000, 4, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_3, 1e-12), kwargs = {})
#   %expand : Tensor "f32[350, 1000, 4, 48][4000, 4, 1, 0]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min, [350, 1000, 4, 48]), kwargs = {})
#   %div : Tensor "f32[350, 1000, 4, 48][192000, 192, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clone, %expand), kwargs = {})
#   %unsqueeze : Tensor "f32[350, 1000, 4, 1, 48][192000, 192, 48, 48, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%div, -2), kwargs = {})
#   %mul_3 : Tensor "f32[350, 1000, 4, 2, 48][384000, 384, 96, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, %unsqueeze), kwargs = {})
#   %sub : Tensor "f32[350, 1000, 4, 2, 48][384000, 384, 96, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %mul_3), kwargs = {})
#   %view_6 : Tensor "f32[350, 1000, 8, 48][384000, 384, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub, [350, 1000, 8, 48]), kwargs = {})
#   %view_8 : Tensor "bf16[350, 1000, 8][8000, 8, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [350, 1000, 8]), kwargs = {})
#   %mul_4 : Tensor "bf16[350, 1000, 8][8000, 8, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, 0.5), kwargs = {})
#   %sigmoid : Tensor "bf16[350, 1000, 8][8000, 8, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_4,), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[350, 1000, 8, 1][8000, 8, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid, 3), kwargs = {})
#   %mul_5 : Tensor "f32[350, 1000, 8, 48][384000, 384, 48, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %unsqueeze_1), kwargs = {})
#   %view_9 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_5, [350, 1000, 384]), kwargs = {})
#   %convert_element_type_16 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_9, torch.bfloat16), kwargs = {})
#   return %convert_element_type_16
triton_poi_fused__to_copy__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sigmoid_slice_sub_transpose_unsqueeze_view_7 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sigmoid_slice_sub_transpose_unsqueeze_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sigmoid_slice_sub_transpose_unsqueeze_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1097600000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sigmoid_slice_sub_transpose_unsqueeze_view_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134400000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 384)
    x1 = xindex // 384
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x2 // 48), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (576 + 48*(x0 // 96) + 768*x1 + ((x0 % 48))), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x2 // 96), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 // 48), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1e-12, tl.float32)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = (tmp4 / tmp8)
    tmp10 = tmp2 * tmp9
    tmp11 = tmp1 - tmp10
    tmp13 = tl.full([1], 0.5, tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp11 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/tz/ctzlxyu7oorhzvnzgm3ihladk3ehgizzurxfns4khmusraa5eoux.py
# Topologically Sorted Source Nodes: [linear, linear_3, add, rms_norm_1, to_5, reshape_3, triton_kernel_wrapper_mutation_1], Original ATen: [aten._unsafe_view, aten.add, aten._fused_rms_norm, aten._to_copy, aten.view]
# Source node to ATen node mapping:
#   add => add_1
#   linear => view_1
#   linear_3 => view_11
#   reshape_3 => view_13
#   rms_norm_1 => add_2, convert_element_type_20, convert_element_type_21, mean_1, mul_6, mul_7, pow_4, rsqrt_1
#   to_5 => convert_element_type_19
#   triton_kernel_wrapper_mutation_1 => triton_kernel_wrapper_mutation_4
# Graph fragment:
#   %mm_3 : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm_3]
#   %mm : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm]
#   %buf24 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:6" = PlaceHolder[target=buf24]
#   %arg8_1 : Tensor "f32[384][1]cuda:6" = PlaceHolder[target=arg8_1]
#   %view_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [350, 1000, 384]), kwargs = {})
#   %view_11 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [350, 1000, 384]), kwargs = {})
#   %add_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %view_1), kwargs = {})
#   %convert_element_type_20 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_4 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_20, 2), kwargs = {})
#   %mean_1 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [2], True), kwargs = {})
#   %add_2 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_6 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, %rsqrt_1), kwargs = {})
#   %convert_element_type_19 : Tensor "bf16[384][1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg8_1, torch.bfloat16), kwargs = {})
#   %mul_7 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %convert_element_type_19), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.bfloat16), kwargs = {})
#   %view_13 : Tensor "bf16[350000, 384][384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_21, [-1, 384]), kwargs = {})
#   %triton_kernel_wrapper_mutation_4 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 67, constant_args_idx: 64, grid: [(65628, 1, 1)], tma_descriptor_metadata: {}, kwargs: {A: %view_13, B: %arg9_1, C: %empty_2, AUX: %empty_3}})
#   return %buf24,%buf27
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_8 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 1075201536}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_8(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 384.0, tl.float32)
    tmp10 = (tmp8 / tmp9)
    tmp11 = tl.full([1, 1], 1e-06, tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp3 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp14 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp19, r0_mask & xmask)
''', device_str='cuda')


# Original path: /workspace/parameter-golf/train_gpt_parcae.py:4796
_leaky_relu_sq_matmul_kernel_1 = async_compile.triton('_leaky_relu_sq_matmul_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 4, 'num_stages': 4}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_leaky_relu_sq_matmul_kernel_1', 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'A': '*bf16', 'B': '*bf16', 'C': '*bf16', 'AUX': '*bf16', 'M': 'constexpr', 'N': 'constexpr', 'K': 'constexpr', 'stride_am': 'constexpr', 'stride_ak': 'constexpr', 'stride_bn': 'constexpr', 'stride_bk': 'constexpr', 'stride_cm': 'constexpr', 'stride_cn': 'constexpr', 'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr', 'BLOCK_K': 'constexpr', 'GROUP_M': 'constexpr', 'FORWARD': 'constexpr', 'NEGATIVE_SLOPE': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'M': 350000, 'N': 1536, 'K': 384, 'stride_am': 384, 'stride_ak': 1, 'stride_bn': 384, 'stride_bk': 1, 'stride_cm': 1536, 'stride_cn': 1, 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'FORWARD': True, 'NEGATIVE_SLOPE': 0.5}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/es/cestzxj5izajnjnzi7ntjedu6ahpooyr4jkidkr6plumv7fab7yj.py
# Topologically Sorted Source Nodes: [linear, linear_3, add, reshape_4, add_1, rms_norm_2, to_8], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten._fused_rms_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add => add_1
#   add_1 => add_3
#   linear => view_1
#   linear_3 => view_11
#   reshape_4 => view_14
#   rms_norm_2 => add_4, convert_element_type_25, convert_element_type_26, mean_2, mul_8, mul_9, pow_5, rsqrt_2
#   to_8 => convert_element_type_24
# Graph fragment:
#   %mm_4 : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm_4]
#   %mm_3 : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm_3]
#   %mm : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm]
#   %buf31 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:6" = PlaceHolder[target=buf31]
#   %arg11_1 : Tensor "f32[384][1]cuda:6" = PlaceHolder[target=arg11_1]
#   %view_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [350, 1000, 384]), kwargs = {})
#   %view_11 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [350, 1000, 384]), kwargs = {})
#   %add_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %view_1), kwargs = {})
#   %view_14 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [350, 1000, 384]), kwargs = {})
#   %add_3 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_14, %add_1), kwargs = {})
#   %convert_element_type_25 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float32), kwargs = {})
#   %pow_5 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_25, 2), kwargs = {})
#   %mean_2 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [2], True), kwargs = {})
#   %add_4 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_8 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_25, %rsqrt_2), kwargs = {})
#   %convert_element_type_24 : Tensor "bf16[384][1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg11_1, torch.bfloat16), kwargs = {})
#   %mul_9 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %convert_element_type_24), kwargs = {})
#   %convert_element_type_26 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.bfloat16), kwargs = {})
#   return %buf31,%convert_element_type_26
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_9 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 1344001536}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = tl.full([1, 1], 384.0, tl.float32)
    tmp12 = (tmp10 / tmp11)
    tmp13 = tl.full([1, 1], 1e-06, tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp5 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp16 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp21, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/zf/czfkxyuecrodjfjbzoecyu333sj5cz75bfsh2sfct2uqyr47n6oz.py
# Topologically Sorted Source Nodes: [linear, linear_3, add, reshape_4, add_1, linear_7, add_2, rms_norm_3, to_12, reshape_8, triton_kernel_wrapper_mutation_3], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten._fused_rms_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add => add_1
#   add_1 => add_3
#   add_2 => add_5
#   linear => view_1
#   linear_3 => view_11
#   linear_7 => view_24
#   reshape_4 => view_14
#   reshape_8 => view_26
#   rms_norm_3 => add_6, convert_element_type_39, convert_element_type_40, mean_3, mul_14, mul_15, pow_8, rsqrt_3
#   to_12 => convert_element_type_38
#   triton_kernel_wrapper_mutation_3 => triton_kernel_wrapper_mutation_2
# Graph fragment:
#   %mm_7 : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm_7]
#   %mm_4 : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm_4]
#   %mm_3 : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm_3]
#   %mm : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm]
#   %buf53 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:6" = PlaceHolder[target=buf53]
#   %arg15_1 : Tensor "f32[384][1]cuda:6" = PlaceHolder[target=arg15_1]
#   %view_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [350, 1000, 384]), kwargs = {})
#   %view_11 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [350, 1000, 384]), kwargs = {})
#   %add_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %view_1), kwargs = {})
#   %view_14 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [350, 1000, 384]), kwargs = {})
#   %add_3 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_14, %add_1), kwargs = {})
#   %view_24 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [350, 1000, 384]), kwargs = {})
#   %add_5 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_24, %add_3), kwargs = {})
#   %convert_element_type_39 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float32), kwargs = {})
#   %pow_8 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_39, 2), kwargs = {})
#   %mean_3 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [2], True), kwargs = {})
#   %add_6 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_3, 1e-06), kwargs = {})
#   %rsqrt_3 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_14 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_39, %rsqrt_3), kwargs = {})
#   %convert_element_type_38 : Tensor "bf16[384][1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg15_1, torch.bfloat16), kwargs = {})
#   %mul_15 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %convert_element_type_38), kwargs = {})
#   %convert_element_type_40 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
#   %view_26 : Tensor "bf16[350000, 384][384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_40, [-1, 384]), kwargs = {})
#   %triton_kernel_wrapper_mutation_2 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 69, constant_args_idx: 66, grid: [(65628, 1, 1)], tma_descriptor_metadata: {}, kwargs: {A: %view_26, B: %arg16_1, C: %empty_6, AUX: %empty_7}})
#   return %buf53,%buf56
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_10 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 1612801536}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr3 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp13 = tl.full([1, 1], 384.0, tl.float32)
    tmp14 = (tmp12 / tmp13)
    tmp15 = tl.full([1, 1], 1e-06, tl.float32)
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp7 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp18 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp23, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/s5/cs5xuznukyu24ly2tagtq5np4rsf7hvaoi5krxb576vbplxjjklx.py
# Topologically Sorted Source Nodes: [linear, linear_3, add, reshape_4, add_1, linear_7, add_2, reshape_9, add_3, rms_norm_4, to_15], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten._fused_rms_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add => add_1
#   add_1 => add_3
#   add_2 => add_5
#   add_3 => add_7
#   linear => view_1
#   linear_3 => view_11
#   linear_7 => view_24
#   reshape_4 => view_14
#   reshape_9 => view_27
#   rms_norm_4 => add_8, convert_element_type_44, convert_element_type_45, mean_4, mul_16, mul_17, pow_9, rsqrt_4
#   to_15 => convert_element_type_43
# Graph fragment:
#   %mm_8 : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm_8]
#   %mm_7 : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm_7]
#   %mm_4 : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm_4]
#   %mm_3 : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm_3]
#   %mm : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm]
#   %add_7 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6" = PlaceHolder[target=add_7]
#   %buf61 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:6" = PlaceHolder[target=buf61]
#   %arg18_1 : Tensor "f32[384][1]cuda:6" = PlaceHolder[target=arg18_1]
#   %view_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [350, 1000, 384]), kwargs = {})
#   %view_11 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [350, 1000, 384]), kwargs = {})
#   %add_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %view_1), kwargs = {})
#   %view_14 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [350, 1000, 384]), kwargs = {})
#   %add_3 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_14, %add_1), kwargs = {})
#   %view_24 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [350, 1000, 384]), kwargs = {})
#   %add_5 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_24, %add_3), kwargs = {})
#   %view_27 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [350, 1000, 384]), kwargs = {})
#   %add_7 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_27, %add_5), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_7, torch.float32), kwargs = {})
#   %pow_9 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_44, 2), kwargs = {})
#   %mean_4 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_9, [2], True), kwargs = {})
#   %add_8 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_4, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_16 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_44, %rsqrt_4), kwargs = {})
#   %convert_element_type_43 : Tensor "bf16[384][1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg18_1, torch.bfloat16), kwargs = {})
#   %mul_17 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %convert_element_type_43), kwargs = {})
#   %convert_element_type_45 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_17, torch.bfloat16), kwargs = {})
#   return %add_7,%buf61,%convert_element_type_45
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_11 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 2419201536}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 + tmp6
    tmp8 = tmp0 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp15 = tl.full([1, 1], 384.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp17 = tl.full([1, 1], 1e-06, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp9 * tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp8, r0_mask & xmask)
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp25, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/66/c66zbgbsu5ejlldonk6zgcytb3hv72b4ercujv4mu4bmeck24ruv.py
# Topologically Sorted Source Nodes: [linear_11, add_4, reshape_14, add_5, rms_norm_6, to_22], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten._fused_rms_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add_4 => add_9
#   add_5 => add_11
#   linear_11 => view_37
#   reshape_14 => view_40
#   rms_norm_6 => add_12, convert_element_type_63, convert_element_type_64, mean_6, mul_24, mul_25, pow_13, rsqrt_6
#   to_22 => convert_element_type_62
# Graph fragment:
#   %mm_12 : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm_12]
#   %mm_11 : Tensor "bf16[350000, 384][384, 1]cuda:6" = PlaceHolder[target=mm_11]
#   %add_7 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6" = PlaceHolder[target=add_7]
#   %buf90 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:6" = PlaceHolder[target=buf90]
#   %arg25_1 : Tensor "f32[384][1]cuda:6" = PlaceHolder[target=arg25_1]
#   %view_37 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_11, [350, 1000, 384]), kwargs = {})
#   %add_9 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_37, %add_7), kwargs = {})
#   %view_40 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_12, [350, 1000, 384]), kwargs = {})
#   %add_11 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_40, %add_9), kwargs = {})
#   %convert_element_type_63 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.float32), kwargs = {})
#   %pow_13 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_63, 2), kwargs = {})
#   %mean_6 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_13, [2], True), kwargs = {})
#   %add_12 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_6, 1e-06), kwargs = {})
#   %rsqrt_6 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_24 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_63, %rsqrt_6), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[384][1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg25_1, torch.bfloat16), kwargs = {})
#   %mul_25 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %convert_element_type_62), kwargs = {})
#   %convert_element_type_64 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_25, torch.bfloat16), kwargs = {})
#   return %buf90,%convert_element_type_64
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_12 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 1344001536}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = tl.full([1, 1], 384.0, tl.float32)
    tmp12 = (tmp10 / tmp11)
    tmp13 = tl.full([1, 1], 1e-06, tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp5 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp16 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp21, r0_mask & xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1 = args
        args.clear()
        assert_size_stride(arg0_1, (384, 384), (384, 1))
        assert_size_stride(arg1_1, (350, 1000, 384), (384000, 384, 1))
        assert_size_stride(arg2_1, (384, ), (1, ))
        assert_size_stride(arg3_1, (768, 384), (384, 1))
        assert_size_stride(arg4_1, (1, 1000, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg5_1, (1, 1000, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg6_1, (8, 12), (12, 1))
        assert_size_stride(arg7_1, (384, 384), (384, 1))
        assert_size_stride(arg8_1, (384, ), (1, ))
        assert_size_stride(arg9_1, (1536, 384), (384, 1))
        assert_size_stride(arg10_1, (384, 1536), (1536, 1))
        assert_size_stride(arg11_1, (384, ), (1, ))
        assert_size_stride(arg12_1, (768, 384), (384, 1))
        assert_size_stride(arg13_1, (8, 12), (12, 1))
        assert_size_stride(arg14_1, (384, 384), (384, 1))
        assert_size_stride(arg15_1, (384, ), (1, ))
        assert_size_stride(arg16_1, (1536, 384), (384, 1))
        assert_size_stride(arg17_1, (384, 1536), (1536, 1))
        assert_size_stride(arg18_1, (384, ), (1, ))
        assert_size_stride(arg19_1, (768, 384), (384, 1))
        assert_size_stride(arg20_1, (8, 12), (12, 1))
        assert_size_stride(arg21_1, (384, 384), (384, 1))
        assert_size_stride(arg22_1, (384, ), (1, ))
        assert_size_stride(arg23_1, (1536, 384), (384, 1))
        assert_size_stride(arg24_1, (384, 1536), (1536, 1))
        assert_size_stride(arg25_1, (384, ), (1, ))
        assert_size_stride(arg26_1, (350, 1000), (1000, 1))
        with torch.cuda._DeviceGuard(6):
            torch.cuda.set_device(6)
            buf0 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
            stream6 = get_raw_stream(6)
            triton_poi_fused__to_copy_0.run(arg1_1, buf0, 134400000, stream=stream6)
            del arg1_1
            buf1 = empty_strided_cuda((350000, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf0, (350000, 384), (384, 1), 0), reinterpret_tensor(arg0_1, (384, 384), (1, 384), 0), out=buf1)
            del arg0_1
            buf3 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [linear, rms_norm, to_1], Original ATen: [aten._unsafe_view, aten._fused_rms_norm, aten._to_copy]
            stream6 = get_raw_stream(6)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_1.run(buf1, arg2_1, buf3, 350000, 384, stream=stream6)
            del arg2_1
            buf4 = empty_strided_cuda((350000, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, rms_norm, to_1, linear_1], Original ATen: [aten._unsafe_view, aten._fused_rms_norm, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (350000, 384), (384, 1), 0), reinterpret_tensor(arg3_1, (384, 768), (1, 384), 0), out=buf4)
            del arg3_1
            buf5 = empty_strided_cuda((350, 1000, 8, 48), (384000, 384, 48, 1), torch.bfloat16)
            buf6 = empty_strided_cuda((350, 1000, 4, 48), (192000, 192, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1, view, getitem_1, getitem_2, triton_kernel_wrapper_mutation], Original ATen: [aten._unsafe_view, aten.view, aten.select]
            stream6 = get_raw_stream(6)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf4, (350, 1000, 16, 48), (768000, 768, 48, 1), 0), buf5, buf6, reinterpret_tensor(arg4_1, (1000, 24), (24, 1), 0), reinterpret_tensor(arg5_1, (1000, 24), (24, 1), 0), 4200000, 1, 1, stream=stream6)
            buf9 = empty_strided_cuda((350, 4, 1000, 48), (192000, 48, 192, 1), torch.bfloat16)
            buf16 = empty_strided_cuda((350, 1000, 4, 1), (4000, 4, 1, 1400000), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, view, scaled_dot_product_attention, getitem, contiguous, transpose_2, normalize], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention, aten.linalg_vector_norm]
            stream6 = get_raw_stream(6)
            triton_per_fused__scaled_dot_product_flash_attention__unsafe_view_clone_linalg_vector_norm_slice_transpose_view_2.run(buf4, buf9, buf16, 1400000, 48, stream=stream6)
            # Topologically Sorted Source Nodes: [linear_1, view, scaled_dot_product_attention, getitem, contiguous, transpose_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
            buf10 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf5, (350, 8, 1000, 48), (384000, 48, 384, 1), 0), reinterpret_tensor(buf6, (350, 4, 1000, 48), (192000, 48, 192, 1), 0), buf9, 0.0, True, scale=0.14433756729740646)
            buf11 = buf10[0]
            assert_size_stride(buf11, (350, 8, 1000, 48), (384000, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf11, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf10
            buf17 = empty_strided_cuda((350, 1000, 4, 2, 1), (8000, 8, 2, 1, 2800000), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, view, getitem, contiguous, transpose_3, view_1, normalize, unsqueeze, mul, sum_1], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.unsqueeze, aten.mul, aten.sum]
            stream6 = get_raw_stream(6)
            triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_slice_sum_transpose_unsqueeze_view_3.run(buf11, buf4, buf16, buf17, 2800000, 48, stream=stream6)
            buf18 = empty_strided_cuda((350, 1000, 12), (12032, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_3, contiguous_7], Original ATen: [aten.slice, aten.clone]
            stream6 = get_raw_stream(6)
            triton_poi_fused_clone_slice_4.run(buf3, buf18, 4200000, stream=stream6)
            buf19 = empty_strided_cuda((350000, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_3, contiguous_7, linear_2, to_3], Original ATen: [aten.slice, aten.clone, aten.view, aten._to_copy, aten.t, aten.mm]
            stream6 = get_raw_stream(6)
            triton_poi_fused__to_copy_clone_mm_slice_t_view_5.run(buf18, buf19, 4200000, stream=stream6)
            del buf18
            buf20 = empty_strided_cuda((8, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_3], Original ATen: [aten._to_copy]
            stream6 = get_raw_stream(6)
            triton_poi_fused__to_copy_6.run(arg6_1, buf20, 96, stream=stream6)
            del arg6_1
            buf21 = empty_strided_cuda((350000, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_3, contiguous_7, linear_2, to_3], Original ATen: [aten.slice, aten.clone, aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf19, reinterpret_tensor(buf20, (12, 8), (1, 12), 0), out=buf21)
            del buf19
            del buf20
            buf22 = reinterpret_tensor(buf11, (350, 1000, 384), (384000, 384, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [linear_1, view, getitem, contiguous, transpose_3, view_1, normalize, unsqueeze, mul_1, sub, reshape, linear_2, mul_2, sigmoid, getitem_4, mul_3, reshape_1, linear_3], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.unsqueeze, aten.mul, aten.sub, aten.sigmoid, aten._to_copy]
            stream6 = get_raw_stream(6)
            triton_poi_fused__to_copy__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sigmoid_slice_sub_transpose_unsqueeze_view_7.run(buf22, buf17, buf4, buf16, buf21, 134400000, stream=stream6)
            del buf16
            del buf17
            del buf21
            del buf4
            buf23 = reinterpret_tensor(buf3, (350000, 384), (384, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [linear_1, view, getitem, contiguous, transpose_3, view_1, normalize, unsqueeze, mul_1, sub, reshape, linear_2, mul_2, sigmoid, getitem_4, mul_3, reshape_1, linear_3], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.unsqueeze, aten.mul, aten.sub, aten.sigmoid, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf22, (350000, 384), (384, 1), 0), reinterpret_tensor(arg7_1, (384, 384), (1, 384), 0), out=buf23)
            del arg7_1
            buf27 = reinterpret_tensor(buf22, (350000, 384), (384, 1), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [linear, linear_3, add, rms_norm_1, to_5, reshape_3, triton_kernel_wrapper_mutation_1], Original ATen: [aten._unsafe_view, aten.add, aten._fused_rms_norm, aten._to_copy, aten.view]
            stream6 = get_raw_stream(6)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_8.run(buf23, buf1, arg8_1, buf27, 350000, 384, stream=stream6)
            del arg8_1
            buf25 = empty_strided_cuda((350000, 1536), (1536, 1), torch.bfloat16)
            buf26 = empty_strided_cuda((350000, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, linear_3, add, rms_norm_1, to_5, reshape_3, triton_kernel_wrapper_mutation_1], Original ATen: [aten._unsafe_view, aten.add, aten._fused_rms_norm, aten._to_copy, aten.view]
            stream6 = get_raw_stream(6)
            _leaky_relu_sq_matmul_kernel_1.run(buf27, arg9_1, buf25, buf26, 65628, 1, 1, stream=stream6)
            del arg9_1
            del buf25
            buf30 = buf27; del buf27  # reuse
            # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf26, reinterpret_tensor(arg10_1, (1536, 384), (1, 1536), 0), out=buf30)
            del arg10_1
            del buf26
            buf32 = reinterpret_tensor(buf5, (350, 1000, 384), (384000, 384, 1), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [linear, linear_3, add, reshape_4, add_1, rms_norm_2, to_8], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten._fused_rms_norm, aten._to_copy]
            stream6 = get_raw_stream(6)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_9.run(buf30, buf23, buf1, arg11_1, buf32, 350000, 384, stream=stream6)
            del arg11_1
            buf33 = empty_strided_cuda((350000, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf32, (350000, 384), (384, 1), 0), reinterpret_tensor(arg12_1, (384, 768), (1, 384), 0), out=buf33)
            del arg12_1
            buf34 = empty_strided_cuda((350, 1000, 8, 48), (384000, 384, 48, 1), torch.bfloat16)
            buf35 = reinterpret_tensor(buf9, (350, 1000, 4, 48), (192000, 192, 48, 1), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [linear_5, view_2, getitem_6, getitem_7, triton_kernel_wrapper_mutation_2], Original ATen: [aten._unsafe_view, aten.view, aten.select]
            stream6 = get_raw_stream(6)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf33, (350, 1000, 16, 48), (768000, 768, 48, 1), 0), buf34, buf35, reinterpret_tensor(arg4_1, (1000, 24), (24, 1), 0), reinterpret_tensor(arg5_1, (1000, 24), (24, 1), 0), 4200000, 1, 1, stream=stream6)
            buf38 = reinterpret_tensor(buf6, (350, 4, 1000, 48), (192000, 48, 192, 1), 0); del buf6  # reuse
            buf45 = empty_strided_cuda((350, 1000, 4, 1), (4000, 4, 1, 1400000), torch.float32)
            # Topologically Sorted Source Nodes: [linear_5, view_2, scaled_dot_product_attention_1, getitem_5, contiguous_12, transpose_6, normalize_1], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention, aten.linalg_vector_norm]
            stream6 = get_raw_stream(6)
            triton_per_fused__scaled_dot_product_flash_attention__unsafe_view_clone_linalg_vector_norm_slice_transpose_view_2.run(buf33, buf38, buf45, 1400000, 48, stream=stream6)
            # Topologically Sorted Source Nodes: [linear_5, view_2, scaled_dot_product_attention_1, getitem_5, contiguous_12, transpose_6], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
            buf39 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf34, (350, 8, 1000, 48), (384000, 48, 384, 1), 0), reinterpret_tensor(buf35, (350, 4, 1000, 48), (192000, 48, 192, 1), 0), buf38, 0.0, True, scale=0.14433756729740646)
            del buf34
            del buf35
            del buf38
            buf40 = buf39[0]
            assert_size_stride(buf40, (350, 8, 1000, 48), (384000, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf40, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf39
            buf46 = empty_strided_cuda((350, 1000, 4, 2, 1), (8000, 8, 2, 1, 2800000), torch.float32)
            # Topologically Sorted Source Nodes: [linear_5, view_2, getitem_5, contiguous_12, transpose_7, view_3, normalize_1, unsqueeze_1, mul_4, sum_2], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.unsqueeze, aten.mul, aten.sum]
            stream6 = get_raw_stream(6)
            triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_slice_sum_transpose_unsqueeze_view_3.run(buf40, buf33, buf45, buf46, 2800000, 48, stream=stream6)
            buf47 = empty_strided_cuda((350, 1000, 12), (12032, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_8, contiguous_19], Original ATen: [aten.slice, aten.clone]
            stream6 = get_raw_stream(6)
            triton_poi_fused_clone_slice_4.run(buf32, buf47, 4200000, stream=stream6)
            buf48 = empty_strided_cuda((350000, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_8, contiguous_19, linear_6, to_10], Original ATen: [aten.slice, aten.clone, aten.view, aten._to_copy, aten.t, aten.mm]
            stream6 = get_raw_stream(6)
            triton_poi_fused__to_copy_clone_mm_slice_t_view_5.run(buf47, buf48, 4200000, stream=stream6)
            del buf47
            buf49 = empty_strided_cuda((8, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_10], Original ATen: [aten._to_copy]
            stream6 = get_raw_stream(6)
            triton_poi_fused__to_copy_6.run(arg13_1, buf49, 96, stream=stream6)
            del arg13_1
            buf50 = empty_strided_cuda((350000, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_8, contiguous_19, linear_6, to_10], Original ATen: [aten.slice, aten.clone, aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf48, reinterpret_tensor(buf49, (12, 8), (1, 12), 0), out=buf50)
            del buf48
            del buf49
            buf51 = reinterpret_tensor(buf40, (350, 1000, 384), (384000, 384, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [linear_5, view_2, getitem_5, contiguous_12, transpose_7, view_3, normalize_1, unsqueeze_1, mul_5, sub_1, reshape_5, linear_6, mul_6, sigmoid_1, getitem_9, mul_7, reshape_6, linear_7], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.unsqueeze, aten.mul, aten.sub, aten.sigmoid, aten._to_copy]
            stream6 = get_raw_stream(6)
            triton_poi_fused__to_copy__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sigmoid_slice_sub_transpose_unsqueeze_view_7.run(buf51, buf46, buf33, buf45, buf50, 134400000, stream=stream6)
            del buf33
            del buf45
            del buf46
            del buf50
            buf52 = reinterpret_tensor(buf32, (350000, 384), (384, 1), 0); del buf32  # reuse
            # Topologically Sorted Source Nodes: [linear_5, view_2, getitem_5, contiguous_12, transpose_7, view_3, normalize_1, unsqueeze_1, mul_5, sub_1, reshape_5, linear_6, mul_6, sigmoid_1, getitem_9, mul_7, reshape_6, linear_7], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.unsqueeze, aten.mul, aten.sub, aten.sigmoid, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf51, (350000, 384), (384, 1), 0), reinterpret_tensor(arg14_1, (384, 384), (1, 384), 0), out=buf52)
            del arg14_1
            buf56 = reinterpret_tensor(buf51, (350000, 384), (384, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [linear, linear_3, add, reshape_4, add_1, linear_7, add_2, rms_norm_3, to_12, reshape_8, triton_kernel_wrapper_mutation_3], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten._fused_rms_norm, aten._to_copy]
            stream6 = get_raw_stream(6)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_10.run(buf52, buf30, buf23, buf1, arg15_1, buf56, 350000, 384, stream=stream6)
            del arg15_1
            buf54 = empty_strided_cuda((350000, 1536), (1536, 1), torch.bfloat16)
            buf55 = empty_strided_cuda((350000, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, linear_3, add, reshape_4, add_1, linear_7, add_2, rms_norm_3, to_12, reshape_8, triton_kernel_wrapper_mutation_3], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten._fused_rms_norm, aten._to_copy]
            stream6 = get_raw_stream(6)
            _leaky_relu_sq_matmul_kernel_1.run(buf56, arg16_1, buf54, buf55, 65628, 1, 1, stream=stream6)
            del arg16_1
            del buf54
            buf59 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf55, reinterpret_tensor(arg17_1, (1536, 384), (1, 1536), 0), out=buf59)
            del arg17_1
            buf60 = reinterpret_tensor(buf59, (350, 1000, 384), (384000, 384, 1), 0); del buf59  # reuse
            buf62 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, linear_3, add, reshape_4, add_1, linear_7, add_2, reshape_9, add_3, rms_norm_4, to_15], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten._fused_rms_norm, aten._to_copy]
            stream6 = get_raw_stream(6)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_11.run(buf60, buf52, buf30, buf23, buf1, arg18_1, buf62, 350000, 384, stream=stream6)
            del arg18_1
            del buf1
            del buf23
            del buf30
            buf63 = empty_strided_cuda((350000, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_4, to_15, linear_9], Original ATen: [aten._fused_rms_norm, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf62, (350000, 384), (384, 1), 0), reinterpret_tensor(arg19_1, (384, 768), (1, 384), 0), out=buf63)
            del arg19_1
            buf64 = reinterpret_tensor(buf52, (350, 1000, 8, 48), (384000, 384, 48, 1), 0); del buf52  # reuse
            buf65 = empty_strided_cuda((350, 1000, 4, 48), (192000, 192, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_9, view_4, getitem_11, getitem_12, triton_kernel_wrapper_mutation_4], Original ATen: [aten._unsafe_view, aten.view, aten.select]
            stream6 = get_raw_stream(6)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf63, (350, 1000, 16, 48), (768000, 768, 48, 1), 0), buf64, buf65, reinterpret_tensor(arg4_1, (1000, 24), (24, 1), 0), reinterpret_tensor(arg5_1, (1000, 24), (24, 1), 0), 4200000, 1, 1, stream=stream6)
            del arg4_1
            del arg5_1
            buf68 = empty_strided_cuda((350, 4, 1000, 48), (192000, 48, 192, 1), torch.bfloat16)
            buf75 = empty_strided_cuda((350, 1000, 4, 1), (4000, 4, 1, 1400000), torch.float32)
            # Topologically Sorted Source Nodes: [linear_9, view_4, scaled_dot_product_attention_2, getitem_10, contiguous_24, transpose_10, normalize_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention, aten.linalg_vector_norm]
            stream6 = get_raw_stream(6)
            triton_per_fused__scaled_dot_product_flash_attention__unsafe_view_clone_linalg_vector_norm_slice_transpose_view_2.run(buf63, buf68, buf75, 1400000, 48, stream=stream6)
            # Topologically Sorted Source Nodes: [linear_9, view_4, scaled_dot_product_attention_2, getitem_10, contiguous_24, transpose_10], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
            buf69 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf64, (350, 8, 1000, 48), (384000, 48, 384, 1), 0), reinterpret_tensor(buf65, (350, 4, 1000, 48), (192000, 48, 192, 1), 0), buf68, 0.0, True, scale=0.14433756729740646)
            del buf64
            del buf65
            del buf68
            buf70 = buf69[0]
            assert_size_stride(buf70, (350, 8, 1000, 48), (384000, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf70, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf69
            buf76 = empty_strided_cuda((350, 1000, 4, 2, 1), (8000, 8, 2, 1, 2800000), torch.float32)
            # Topologically Sorted Source Nodes: [linear_9, view_4, getitem_10, contiguous_24, transpose_11, view_5, normalize_2, unsqueeze_2, mul_8, sum_3], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.unsqueeze, aten.mul, aten.sum]
            stream6 = get_raw_stream(6)
            triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_slice_sum_transpose_unsqueeze_view_3.run(buf70, buf63, buf75, buf76, 2800000, 48, stream=stream6)
            buf77 = empty_strided_cuda((350, 1000, 12), (12032, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_13, contiguous_31], Original ATen: [aten.slice, aten.clone]
            stream6 = get_raw_stream(6)
            triton_poi_fused_clone_slice_4.run(buf62, buf77, 4200000, stream=stream6)
            buf78 = empty_strided_cuda((350000, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_13, contiguous_31, linear_10, to_17], Original ATen: [aten.slice, aten.clone, aten.view, aten._to_copy, aten.t, aten.mm]
            stream6 = get_raw_stream(6)
            triton_poi_fused__to_copy_clone_mm_slice_t_view_5.run(buf77, buf78, 4200000, stream=stream6)
            del buf77
            buf79 = empty_strided_cuda((8, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_17], Original ATen: [aten._to_copy]
            stream6 = get_raw_stream(6)
            triton_poi_fused__to_copy_6.run(arg20_1, buf79, 96, stream=stream6)
            del arg20_1
            buf80 = empty_strided_cuda((350000, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_13, contiguous_31, linear_10, to_17], Original ATen: [aten.slice, aten.clone, aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf78, reinterpret_tensor(buf79, (12, 8), (1, 12), 0), out=buf80)
            del buf78
            del buf79
            buf81 = reinterpret_tensor(buf70, (350, 1000, 384), (384000, 384, 1), 0); del buf70  # reuse
            # Topologically Sorted Source Nodes: [linear_9, view_4, getitem_10, contiguous_24, transpose_11, view_5, normalize_2, unsqueeze_2, mul_9, sub_2, reshape_10, linear_10, mul_10, sigmoid_2, getitem_14, mul_11, reshape_11, linear_11], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.unsqueeze, aten.mul, aten.sub, aten.sigmoid, aten._to_copy]
            stream6 = get_raw_stream(6)
            triton_poi_fused__to_copy__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sigmoid_slice_sub_transpose_unsqueeze_view_7.run(buf81, buf76, buf63, buf75, buf80, 134400000, stream=stream6)
            del buf63
            del buf75
            del buf76
            del buf80
            buf82 = reinterpret_tensor(buf62, (350000, 384), (384, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [linear_9, view_4, getitem_10, contiguous_24, transpose_11, view_5, normalize_2, unsqueeze_2, mul_9, sub_2, reshape_10, linear_10, mul_10, sigmoid_2, getitem_14, mul_11, reshape_11, linear_11], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.unsqueeze, aten.mul, aten.sub, aten.sigmoid, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf81, (350000, 384), (384, 1), 0), reinterpret_tensor(arg21_1, (384, 384), (1, 384), 0), out=buf82)
            del arg21_1
            buf86 = reinterpret_tensor(buf81, (350000, 384), (384, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [linear_11, add_4, rms_norm_5, to_19, reshape_13, triton_kernel_wrapper_mutation_5], Original ATen: [aten._unsafe_view, aten.add, aten._fused_rms_norm, aten._to_copy, aten.view]
            stream6 = get_raw_stream(6)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_8.run(buf82, buf60, arg22_1, buf86, 350000, 384, stream=stream6)
            del arg22_1
            buf84 = buf55; del buf55  # reuse
            buf85 = empty_strided_cuda((350000, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_11, add_4, rms_norm_5, to_19, reshape_13, triton_kernel_wrapper_mutation_5], Original ATen: [aten._unsafe_view, aten.add, aten._fused_rms_norm, aten._to_copy, aten.view]
            stream6 = get_raw_stream(6)
            _leaky_relu_sq_matmul_kernel_1.run(buf86, arg23_1, buf84, buf85, 65628, 1, 1, stream=stream6)
            del arg23_1
            del buf84
            buf89 = buf86; del buf86  # reuse
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf85, reinterpret_tensor(arg24_1, (1536, 384), (1, 1536), 0), out=buf89)
            del arg24_1
            del buf85
            buf91 = reinterpret_tensor(buf89, (350, 1000, 384), (384000, 384, 1), 0); del buf89  # reuse
            # Topologically Sorted Source Nodes: [linear_11, add_4, reshape_14, add_5, rms_norm_6, to_22], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten._fused_rms_norm, aten._to_copy]
            stream6 = get_raw_stream(6)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_view_12.run(buf91, buf82, buf60, arg25_1, 350000, 384, stream=stream6)
            del arg25_1
            del buf60
            del buf82
        return (reinterpret_tensor(buf91, (350000, 384), (384, 1), 0), reinterpret_tensor(arg26_1, (350000, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((384, 384), (384, 1), device='cuda:6', dtype=torch.bfloat16)
    arg1_1 = rand_strided((350, 1000, 384), (384000, 384, 1), device='cuda:6', dtype=torch.float32)
    arg2_1 = rand_strided((384, ), (1, ), device='cuda:6', dtype=torch.float32)
    arg3_1 = rand_strided((768, 384), (384, 1), device='cuda:6', dtype=torch.bfloat16)
    arg4_1 = rand_strided((1, 1000, 1, 24), (24576, 24, 24, 1), device='cuda:6', dtype=torch.bfloat16)
    arg5_1 = rand_strided((1, 1000, 1, 24), (24576, 24, 24, 1), device='cuda:6', dtype=torch.bfloat16)
    arg6_1 = rand_strided((8, 12), (12, 1), device='cuda:6', dtype=torch.float32)
    arg7_1 = rand_strided((384, 384), (384, 1), device='cuda:6', dtype=torch.bfloat16)
    arg8_1 = rand_strided((384, ), (1, ), device='cuda:6', dtype=torch.float32)
    arg9_1 = rand_strided((1536, 384), (384, 1), device='cuda:6', dtype=torch.bfloat16)
    arg10_1 = rand_strided((384, 1536), (1536, 1), device='cuda:6', dtype=torch.bfloat16)
    arg11_1 = rand_strided((384, ), (1, ), device='cuda:6', dtype=torch.float32)
    arg12_1 = rand_strided((768, 384), (384, 1), device='cuda:6', dtype=torch.bfloat16)
    arg13_1 = rand_strided((8, 12), (12, 1), device='cuda:6', dtype=torch.float32)
    arg14_1 = rand_strided((384, 384), (384, 1), device='cuda:6', dtype=torch.bfloat16)
    arg15_1 = rand_strided((384, ), (1, ), device='cuda:6', dtype=torch.float32)
    arg16_1 = rand_strided((1536, 384), (384, 1), device='cuda:6', dtype=torch.bfloat16)
    arg17_1 = rand_strided((384, 1536), (1536, 1), device='cuda:6', dtype=torch.bfloat16)
    arg18_1 = rand_strided((384, ), (1, ), device='cuda:6', dtype=torch.float32)
    arg19_1 = rand_strided((768, 384), (384, 1), device='cuda:6', dtype=torch.bfloat16)
    arg20_1 = rand_strided((8, 12), (12, 1), device='cuda:6', dtype=torch.float32)
    arg21_1 = rand_strided((384, 384), (384, 1), device='cuda:6', dtype=torch.bfloat16)
    arg22_1 = rand_strided((384, ), (1, ), device='cuda:6', dtype=torch.float32)
    arg23_1 = rand_strided((1536, 384), (384, 1), device='cuda:6', dtype=torch.bfloat16)
    arg24_1 = rand_strided((384, 1536), (1536, 1), device='cuda:6', dtype=torch.bfloat16)
    arg25_1 = rand_strided((384, ), (1, ), device='cuda:6', dtype=torch.float32)
    arg26_1 = rand_strided((350, 1000), (1000, 1), device='cuda:6', dtype=torch.int64)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
