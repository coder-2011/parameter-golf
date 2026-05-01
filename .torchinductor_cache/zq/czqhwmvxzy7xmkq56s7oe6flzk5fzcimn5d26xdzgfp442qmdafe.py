# AOT ID: ['7_forward']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/kk/ckkkyteiv3bdcgzfirnaibypv3ai22lma7j5zd5p3jfp2u5ebh2t.py
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
#   rms_norm_1 => mul_8
# Graph fragment:
#   %primals_1 : Tensor "f32[2, 384][384, 1]cuda:4" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=primals_2]
#   %primals_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=primals_3]
#   %buf0 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:4" = PlaceHolder[target=buf0]
#   %rsqrt : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4" = PlaceHolder[target=rsqrt]
#   %mul_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=mul_2]
#   %primals_4 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_4]
#   %primals_12 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_12]
#   %select : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %mul : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %primals_2), kwargs = {})
#   %select_1 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_1, 0, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 0), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 1), kwargs = {})
#   %mul_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %primals_3), kwargs = {})
#   %add : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %pow_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add_1 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt), kwargs = {})
#   %mul_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_4), kwargs = {})
#   %mul_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 0.7071067811865475), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   %mul_8 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_12), kwargs = {})
#   %mul_9 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, 0.7071067811865475), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.bfloat16), kwargs = {})
#   return %buf0,%rsqrt,%mul_2,%convert_element_type_2,%convert_element_type_13
triton_red_fused__fused_rms_norm__to_copy_add_mul_select_unsqueeze_0 = async_compile.triton('triton_red_fused__fused_rms_norm__to_copy_add_mul_select_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm__to_copy_add_mul_select_unsqueeze_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 4, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1048576, 'r0_': 1107302400}}
)
@triton.jit
def triton_red_fused__fused_rms_norm__to_copy_add_mul_select_unsqueeze_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
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
        tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tmp2 + tmp6
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp12 = tl.full([1, 1], 384.0, tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tl.full([1, 1], 1e-06, tl.float32)
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp16, None)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp17 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr0 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp17 * tmp18
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 * tmp22
        tmp24 = tmp19 + tmp23
        tmp25 = tmp24 * tmp16
        tmp27 = tmp25 * tmp26
        tmp28 = tl.full([1, 1], 0.7071067811865475, tl.float32)
        tmp29 = tmp27 * tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp32 = tmp25 * tmp31
        tmp33 = tmp32 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr0 + (r0_1 + 384*x0), tmp25, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 384*x0), tmp30, r0_mask)
        tl.store(out_ptr2 + (r0_1 + 384*x0), tmp34, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/wh/cwhjnyajrpmh3gbpfblc7drypmftqkdhx36p5rdwlauixys4ywnu.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear => convert_element_type_default_21, permute
# Graph fragment:
#   %primals_5 : Tensor "bf16[1152, 384][384, 1]cuda:4" = PlaceHolder[target=primals_5]
#   %convert_element_type_default_21 : Tensor "bf16[1152, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_5, torch.bfloat16), kwargs = {})
#   %permute : Tensor "bf16[384, 1152][1, 384]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_default_21, [1, 0]), kwargs = {})
#   return %permute
triton_poi_fused__to_copy_t_1 = async_compile.triton('triton_poi_fused__to_copy_t_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2654208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# Original path: /workspace/parameter-golf/train_gpt_parcae.py:4417
_fused_qkv_postprocess_fwd_kernel_0 = async_compile.triton('_fused_qkv_postprocess_fwd_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 2, 'num_stages': 3}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_fused_qkv_postprocess_fwd_kernel_0', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'QKV': '*bf16', 'Q': '*bf16', 'K': '*bf16', 'COS': '*bf16', 'SIN': '*bf16', 'TOTAL_ROWS': 'constexpr', 'SEQLEN': 'constexpr', 'H_Q': 'constexpr', 'H_KV': 'constexpr', 'HEAD_DIM': 'constexpr', 'ROPE_DIMS': 'constexpr', 'DO_QK_NORM': 'constexpr', 'ROPE_INTERLEAVED': 'constexpr', 'RMS_EPS': 'constexpr', 'BLOCK_D': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'TOTAL_ROWS': 131072, 'SEQLEN': 1024, 'H_Q': 8, 'H_KV': 8, 'HEAD_DIM': 48, 'ROPE_DIMS': 48, 'DO_QK_NORM': True, 'ROPE_INTERLEAVED': False, 'RMS_EPS': 1e-06, 'BLOCK_D': 64}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/yx/cyx2qsi7d34bcw4mhwkkxnxyk277grl5lvb367zqjr7i4klsu4wm.py
# Topologically Sorted Source Nodes: [linear, view, getitem, v], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone]
# Source node to ATen node mapping:
#   getitem => slice_1
#   linear => view_1
#   v => clone
#   view => view_2
# Graph fragment:
#   %mm : Tensor "bf16[131072, 1152][1152, 1]cuda:4" = PlaceHolder[target=mm]
#   %view_1 : Tensor "bf16[128, 1024, 1152][1179648, 1152, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 1152]), kwargs = {})
#   %view_2 : Tensor "bf16[128, 1024, 24, 48][1179648, 1152, 48, 1]cuda:4"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [128, 1024, 24, 48]), kwargs = {})
#   %slice_1 : Tensor "bf16[128, 1024, 8, 48][1179648, 1152, 48, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_2, 2, 16, 9223372036854775807), kwargs = {})
#   %clone : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused__unsafe_view_clone_slice_view_2 = async_compile.triton('triton_poi_fused__unsafe_view_clone_slice_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_slice_view_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 301989888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_slice_view_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 384)
    x1 = xindex // 384
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + 1152*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/sx/csxzhk6teik7tmhhzfut5g7zxk2qx57fvnxu3wmczjdez6bmmhpv.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_5, permute_7
# Graph fragment:
#   %primals_8 : Tensor "f32[8, 12][12, 1]cuda:4" = PlaceHolder[target=primals_8]
#   %convert_element_type_5 : Tensor "bf16[8, 12][12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.bfloat16), kwargs = {})
#   %permute_7 : Tensor "bf16[12, 8][1, 12]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_5, [1, 0]), kwargs = {})
#   return %permute_7
triton_poi_fused__to_copy_t_3 = async_compile.triton('triton_poi_fused__to_copy_t_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/c7/cc7vpo65t2swj3ntb2s3uedla5ua25nnl63cnbzqcr25lslqhmyd.py
# Topologically Sorted Source Nodes: [rms_norm, mul_2, getitem_7, contiguous_3, linear_1], Original ATen: [aten._fused_rms_norm, aten.mul, aten.slice, aten.clone, aten._to_copy]
# Source node to ATen node mapping:
#   contiguous_3 => clone_1
#   getitem_7 => slice_2
#   linear_1 => convert_element_type_6
#   mul_2 => mul_4
#   rms_norm => mul_3
# Graph fragment:
#   %mul_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=mul_2]
#   %primals_4 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_4]
#   %mul_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_4), kwargs = {})
#   %mul_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 0.7071067811865475), kwargs = {})
#   %slice_2 : Tensor "f32[128, 1024, 12][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_4, 2, 0, 12), kwargs = {})
#   %clone_1 : Tensor "f32[128, 1024, 12][12288, 12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_6 : Tensor "bf16[128, 1024, 12][12288, 12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_6
triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_4 = async_compile.triton('triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12582960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 12)
    x1 = xindex // 12
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1], 0.7071067811865475, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/gh/cghr76e2lejbvpxsdlg74zfoxsjsn7jdubuc6442yoz6lgtt4m7n.py
# Topologically Sorted Source Nodes: [transpose_3, linear_1, mul_3, sigmoid, getitem_8, mul_4], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
# Source node to ATen node mapping:
#   getitem_8 => unsqueeze_4
#   linear_1 => view_4
#   mul_3 => mul_5
#   mul_4 => mul_6
#   sigmoid => sigmoid
#   transpose_3 => permute_6
# Graph fragment:
#   %getitem_2 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:4" = PlaceHolder[target=getitem_2]
#   %mm_1 : Tensor "bf16[131072, 8][8, 1]cuda:4" = PlaceHolder[target=mm_1]
#   %permute_6 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [0, 2, 1, 3]), kwargs = {})
#   %view_4 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [128, 1024, 8]), kwargs = {})
#   %mul_5 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, 0.5), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_5,), kwargs = {})
#   %unsqueeze_4 : Tensor "bf16[128, 1024, 8, 1][8192, 8, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid, 3), kwargs = {})
#   %mul_6 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_6, %unsqueeze_4), kwargs = {})
#   return %mul_6
triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5 = async_compile.triton('triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 301989888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 48
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ix/cixemwt6c32owywzoi6wi3zyu4jmuro5f4en4f2ztnsqehgl2ijs.py
# Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_3 => convert_element_type_default_20, permute_9
# Graph fragment:
#   %primals_13 : Tensor "bf16[1536, 384][384, 1]cuda:4" = PlaceHolder[target=primals_13]
#   %convert_element_type_default_20 : Tensor "bf16[1536, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_13, torch.bfloat16), kwargs = {})
#   %permute_9 : Tensor "bf16[384, 1536][1, 384]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_default_20, [1, 0]), kwargs = {})
#   return %permute_9
triton_poi_fused__to_copy_t_6 = async_compile.triton('triton_poi_fused__to_copy_t_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3538944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ig/cigezpn5sx3dx2unkvwi5q2likcb25kxvvskib6d4x64t5wc7aqs.py
# Topologically Sorted Source Nodes: [linear_3, leaky_relu, square, linear_4], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy]
# Source node to ATen node mapping:
#   leaky_relu => convert_element_type_16, gt, mul_10, where
#   linear_3 => view_9
#   linear_4 => convert_element_type_21
#   square => pow_3
# Graph fragment:
#   %mm_3 : Tensor "bf16[131072, 1536][1536, 1]cuda:4" = PlaceHolder[target=mm_3]
#   %view_9 : Tensor "bf16[128, 1024, 1536][1572864, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [128, 1024, 1536]), kwargs = {})
#   %convert_element_type_16 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:4"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_9, torch.float32), kwargs = {})
#   %gt : Tensor "b8[128, 1024, 1536][1572864, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_16, 0), kwargs = {})
#   %mul_10 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_16, 0.5), kwargs = {})
#   %where : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convert_element_type_16, %mul_10), kwargs = {})
#   %pow_3 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where, 2), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[128, 1024, 1536][1572864, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_3, torch.bfloat16), kwargs = {})
#   return %convert_element_type_21
triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_7 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1207959552}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201326592
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/id/cidajdethtmzg7vso3n6r4zp26hcf2llhksqwm44vhyorlxyqdkz.py
# Topologically Sorted Source Nodes: [getitem, getitem_1, mul, getitem_2, getitem_3, mul_1, add, linear_2, getitem_9, getitem_10, linear_4, mul_6, add_1, mul_7, add_2, getitem_11, getitem_12, mul_8, getitem_13, getitem_14, mul_9, add_3, rms_norm_2, mul_10, linear_5, rms_norm_3, mul_13, linear_8], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._unsafe_view, aten._fused_rms_norm, aten._to_copy]
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
#   rms_norm_3 => mul_21
# Graph fragment:
#   %primals_1 : Tensor "f32[2, 384][384, 1]cuda:4" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=primals_2]
#   %primals_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=primals_3]
#   %primals_10 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_10]
#   %mm_2 : Tensor "bf16[131072, 384][384, 1]cuda:4" = PlaceHolder[target=mm_2]
#   %primals_11 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_11]
#   %mm_4 : Tensor "bf16[131072, 384][384, 1]cuda:4" = PlaceHolder[target=mm_4]
#   %primals_15 : Tensor "f32[2, 384][384, 1]cuda:4" = PlaceHolder[target=primals_15]
#   %add_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=add_4]
#   %buf29 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:4" = PlaceHolder[target=buf29]
#   %rsqrt_2 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4" = PlaceHolder[target=rsqrt_2]
#   %mul_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=mul_15]
#   %primals_16 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_16]
#   %primals_22 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_22]
#   %select : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %mul : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %primals_2), kwargs = {})
#   %select_1 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_1, 0, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 0), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 1), kwargs = {})
#   %mul_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %primals_3), kwargs = {})
#   %add : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %view_7 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [128, 1024, 384]), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_10, 0), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_11, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 1), kwargs = {})
#   %view_11 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [128, 1024, 384]), kwargs = {})
#   %mul_11 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_6, %view_7), kwargs = {})
#   %add_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %mul_11), kwargs = {})
#   %mul_12 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_8, %view_11), kwargs = {})
#   %add_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mul_12), kwargs = {})
#   %select_6 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_6, 0), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 1), kwargs = {})
#   %mul_13 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_10, %add_4), kwargs = {})
#   %select_7 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_7, 0), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_11, 1), kwargs = {})
#   %mul_14 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_12, %primals_3), kwargs = {})
#   %add_5 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %mul_14), kwargs = {})
#   %pow_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_5, 2), kwargs = {})
#   %mean_2 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [2], True), kwargs = {})
#   %add_6 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %rsqrt_2), kwargs = {})
#   %mul_16 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %primals_16), kwargs = {})
#   %mul_17 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, 0.5773502691896258), kwargs = {})
#   %convert_element_type_26 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_17, torch.bfloat16), kwargs = {})
#   %mul_21 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %primals_22), kwargs = {})
#   %mul_22 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, 0.5773502691896258), kwargs = {})
#   %convert_element_type_37 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_22, torch.bfloat16), kwargs = {})
#   return %add_4,%buf29,%rsqrt_2,%mul_15,%convert_element_type_26,%convert_element_type_37
triton_red_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_8 = async_compile.triton('triton_red_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 16, 'num_store': 5, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1048576, 'r0_': 1711288320}}
)
@triton.jit
def triton_red_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
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
        tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr0 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr4 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr6 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr7 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr7 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tmp2 + tmp6
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp8 * tmp10
        tmp12 = tmp7 + tmp11
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp13 * tmp15
        tmp17 = tmp12 + tmp16
        tmp19 = tmp18 * tmp17
        tmp21 = tmp20 * tmp5
        tmp22 = tmp19 + tmp21
        tmp23 = tmp22 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask, tmp26, _tmp25)
        tl.store(out_ptr0 + (r0_1 + 384*x0), tmp17, r0_mask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp27 = tl.full([1, 1], 384.0, tl.float32)
    tmp28 = (tmp25 / tmp27)
    tmp29 = tl.full([1, 1], 1e-06, tl.float32)
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp31, None)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp32 = tl.load(in_ptr7 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(out_ptr0 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp35 = tl.load(in_ptr7 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp36 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp41 = tl.load(in_ptr8 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp46 = tl.load(in_ptr9 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp34 = tmp32 * tmp33
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 * tmp37
        tmp39 = tmp34 + tmp38
        tmp40 = tmp39 * tmp31
        tmp42 = tmp40 * tmp41
        tmp43 = tl.full([1, 1], 0.5773502691896258, tl.float32)
        tmp44 = tmp42 * tmp43
        tmp45 = tmp44.to(tl.float32)
        tmp47 = tmp40 * tmp46
        tmp48 = tmp47 * tmp43
        tmp49 = tmp48.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 384*x0), tmp40, r0_mask)
        tl.store(out_ptr2 + (r0_1 + 384*x0), tmp45, r0_mask)
        tl.store(out_ptr3 + (r0_1 + 384*x0), tmp49, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/tj/ctjda6nva7te5edp7d32r6kdo4zuuigaeo4tlqflea4sh6n2lfu5.py
# Topologically Sorted Source Nodes: [rms_norm_2, mul_10, getitem_18, contiguous_8, linear_6], Original ATen: [aten._fused_rms_norm, aten.mul, aten.slice, aten.clone, aten._to_copy]
# Source node to ATen node mapping:
#   contiguous_8 => clone_3
#   getitem_18 => slice_4
#   linear_6 => convert_element_type_30
#   mul_10 => mul_17
#   rms_norm_2 => mul_16
# Graph fragment:
#   %mul_15 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=mul_15]
#   %primals_16 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_16]
#   %mul_16 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %primals_16), kwargs = {})
#   %mul_17 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, 0.5773502691896258), kwargs = {})
#   %slice_4 : Tensor "f32[128, 1024, 12][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_17, 2, 0, 12), kwargs = {})
#   %clone_3 : Tensor "f32[128, 1024, 12][12288, 12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_4,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_30 : Tensor "bf16[128, 1024, 12][12288, 12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_3, torch.bfloat16), kwargs = {})
#   return %convert_element_type_30
triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_9 = async_compile.triton('triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12582960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 12)
    x1 = xindex // 12
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1], 0.5773502691896258, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/pz/cpzyjivb6f2nykzjf7mccwnzssadovvgzhg5aw4b6skhqrrzfwfn.py
# Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_8, getitem_13, getitem_14, mul_9, add_3, linear_7, getitem_20, getitem_21, linear_9, mul_14, add_4, mul_15, add_5, getitem_22, getitem_23, mul_16, getitem_24, getitem_25, mul_17, add_6, rms_norm_4, mul_18, linear_10, rms_norm_5, mul_21, linear_13], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._unsafe_view, aten._fused_rms_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add_3 => add_5
#   add_4 => add_8
#   add_5 => add_9
#   add_6 => add_10
#   getitem_11 => select_6
#   getitem_12 => unsqueeze_10, unsqueeze_9
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
#   mul_8 => mul_13
#   mul_9 => mul_14
#   rms_norm_4 => add_11, mean_4, mul_28, mul_29, pow_7, rsqrt_4
#   rms_norm_5 => mul_34
# Graph fragment:
#   %primals_15 : Tensor "f32[2, 384][384, 1]cuda:4" = PlaceHolder[target=primals_15]
#   %add_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=add_4]
#   %primals_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=primals_3]
#   %primals_20 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_20]
#   %mm_7 : Tensor "bf16[131072, 384][384, 1]cuda:4" = PlaceHolder[target=mm_7]
#   %primals_21 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_21]
#   %mm_9 : Tensor "bf16[131072, 384][384, 1]cuda:4" = PlaceHolder[target=mm_9]
#   %primals_25 : Tensor "f32[2, 384][384, 1]cuda:4" = PlaceHolder[target=primals_25]
#   %add_9 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=add_9]
#   %add_10 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=add_10]
#   %buf59 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:4" = PlaceHolder[target=buf59]
#   %rsqrt_4 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4" = PlaceHolder[target=rsqrt_4]
#   %primals_26 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_26]
#   %primals_32 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_32]
#   %select_6 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_6, 0), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 1), kwargs = {})
#   %mul_13 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_10, %add_4), kwargs = {})
#   %select_7 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_15, 0, 1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_7, 0), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_11, 1), kwargs = {})
#   %mul_14 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_12, %primals_3), kwargs = {})
#   %add_5 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %mul_14), kwargs = {})
#   %view_19 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [128, 1024, 384]), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_20, 0), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, 1), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_21, 0), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 1), kwargs = {})
#   %view_23 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [128, 1024, 384]), kwargs = {})
#   %mul_24 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_15, %view_19), kwargs = {})
#   %add_8 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_24), kwargs = {})
#   %mul_25 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_17, %view_23), kwargs = {})
#   %add_9 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %mul_25), kwargs = {})
#   %select_12 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_25, 0, 0), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_12, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 1), kwargs = {})
#   %mul_26 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_19, %add_9), kwargs = {})
#   %select_13 : Tensor "f32[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_25, 0, 1), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_13, 0), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, 1), kwargs = {})
#   %mul_27 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_21, %primals_3), kwargs = {})
#   %add_10 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_27), kwargs = {})
#   %pow_7 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_10, 2), kwargs = {})
#   %mean_4 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [2], True), kwargs = {})
#   %add_11 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_4, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_28 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %rsqrt_4), kwargs = {})
#   %mul_29 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %primals_26), kwargs = {})
#   %mul_30 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, 0.5), kwargs = {})
#   %convert_element_type_50 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_30, torch.bfloat16), kwargs = {})
#   %mul_34 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %primals_32), kwargs = {})
#   %mul_35 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, 0.5), kwargs = {})
#   %convert_element_type_61 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_35, torch.bfloat16), kwargs = {})
#   return %add_9,%add_10,%buf59,%rsqrt_4,%convert_element_type_50,%convert_element_type_61
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_10 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 12, 'num_store': 5, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1048576, 'r0_': 1711288320}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr4 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr6 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr7 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr7 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr8 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_ptr9 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tmp2 + tmp6
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 * tmp10
    tmp12 = tmp7 + tmp11
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 * tmp15
    tmp17 = tmp12 + tmp16
    tmp19 = tmp18 * tmp17
    tmp21 = tmp20 * tmp5
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.where(r0_mask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
    tmp28 = tl.full([1, 1], 384.0, tl.float32)
    tmp29 = (tmp27 / tmp28)
    tmp30 = tl.full([1, 1], 1e-06, tl.float32)
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp22 * tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tl.full([1, 1], 0.5, tl.float32)
    tmp37 = tmp35 * tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp40 = tmp33 * tmp39
    tmp41 = tmp40 * tmp36
    tmp42 = tmp41.to(tl.float32)
    tl.store(out_ptr0 + (r0_1 + 384*x0), tmp17, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp22, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp32, None)
    tl.store(out_ptr2 + (r0_1 + 384*x0), tmp38, r0_mask)
    tl.store(out_ptr3 + (r0_1 + 384*x0), tmp42, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/xf/cxft7sldrflqzsethriwbu3xbfvt24b4xybdifxnueexe6bkmai6.py
# Topologically Sorted Source Nodes: [rms_norm_4, mul_18, getitem_29, contiguous_13, linear_11], Original ATen: [aten._fused_rms_norm, aten.mul, aten.slice, aten.clone, aten._to_copy]
# Source node to ATen node mapping:
#   contiguous_13 => clone_5
#   getitem_29 => slice_6
#   linear_11 => convert_element_type_54
#   mul_18 => mul_30
#   rms_norm_4 => mul_28, mul_29
# Graph fragment:
#   %add_10 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=add_10]
#   %rsqrt_4 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4" = PlaceHolder[target=rsqrt_4]
#   %primals_26 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_26]
#   %mul_28 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %rsqrt_4), kwargs = {})
#   %mul_29 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %primals_26), kwargs = {})
#   %mul_30 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, 0.5), kwargs = {})
#   %slice_6 : Tensor "f32[128, 1024, 12][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_30, 2, 0, 12), kwargs = {})
#   %clone_5 : Tensor "f32[128, 1024, 12][12288, 12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_6,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_54 : Tensor "bf16[128, 1024, 12][12288, 12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_5, torch.bfloat16), kwargs = {})
#   return %convert_element_type_54
triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_11 = async_compile.triton('triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 524288, 'x': 12582960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 12
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 384*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1, 1], 0.5, tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 12*y0), tmp7, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/tm/ctmca4r22bjlktuxed6uimwn6ywomhdm7asn4vv6uzs73wxmigp6.py
# Topologically Sorted Source Nodes: [linear_13, leaky_relu_2, square_2], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow]
# Source node to ATen node mapping:
#   leaky_relu_2 => convert_element_type_64, gt_2, mul_36, where_2
#   linear_13 => view_33
#   square_2 => pow_9
# Graph fragment:
#   %mm_13 : Tensor "bf16[131072, 1536][1536, 1]cuda:4" = PlaceHolder[target=mm_13]
#   %view_33 : Tensor "bf16[128, 1024, 1536][1572864, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_13, [128, 1024, 1536]), kwargs = {})
#   %convert_element_type_64 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:4"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_33, torch.float32), kwargs = {})
#   %gt_2 : Tensor "b8[128, 1024, 1536][1572864, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_64, 0), kwargs = {})
#   %mul_36 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_64, 0.5), kwargs = {})
#   %where_2 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %convert_element_type_64, %mul_36), kwargs = {})
#   %pow_9 : Tensor "f32[128, 1024, 1536][1572864, 1536, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_2, 2), kwargs = {})
#   return %pow_9
triton_poi_fused__unsafe_view_leaky_relu_pow_12 = async_compile.triton('triton_poi_fused__unsafe_view_leaky_relu_pow_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_leaky_relu_pow_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2013265920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_leaky_relu_pow_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201326592
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33 = args
        args.clear()
        assert_size_stride(primals_1, (2, 384), (384, 1))
        assert_size_stride(primals_2, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(primals_3, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(primals_4, (384, ), (1, ))
        assert_size_stride(primals_5, (1152, 384), (384, 1))
        assert_size_stride(primals_6, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(primals_7, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(primals_8, (8, 12), (12, 1))
        assert_size_stride(primals_9, (384, 384), (384, 1))
        assert_size_stride(primals_10, (384, ), (1, ))
        assert_size_stride(primals_11, (384, ), (1, ))
        assert_size_stride(primals_12, (384, ), (1, ))
        assert_size_stride(primals_13, (1536, 384), (384, 1))
        assert_size_stride(primals_14, (384, 1536), (1536, 1))
        assert_size_stride(primals_15, (2, 384), (384, 1))
        assert_size_stride(primals_16, (384, ), (1, ))
        assert_size_stride(primals_17, (1152, 384), (384, 1))
        assert_size_stride(primals_18, (8, 12), (12, 1))
        assert_size_stride(primals_19, (384, 384), (384, 1))
        assert_size_stride(primals_20, (384, ), (1, ))
        assert_size_stride(primals_21, (384, ), (1, ))
        assert_size_stride(primals_22, (384, ), (1, ))
        assert_size_stride(primals_23, (1536, 384), (384, 1))
        assert_size_stride(primals_24, (384, 1536), (1536, 1))
        assert_size_stride(primals_25, (2, 384), (384, 1))
        assert_size_stride(primals_26, (384, ), (1, ))
        assert_size_stride(primals_27, (1152, 384), (384, 1))
        assert_size_stride(primals_28, (8, 12), (12, 1))
        assert_size_stride(primals_29, (384, 384), (384, 1))
        assert_size_stride(primals_30, (384, ), (1, ))
        assert_size_stride(primals_31, (384, ), (1, ))
        assert_size_stride(primals_32, (384, ), (1, ))
        assert_size_stride(primals_33, (1536, 384), (384, 1))
        with torch.cuda._DeviceGuard(4):
            torch.cuda.set_device(4)
            buf0 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
            buf1 = reinterpret_tensor(buf0, (128, 1024, 1), (1024, 1, 1), 0); del buf0  # reuse
            buf2 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            buf4 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            buf23 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, getitem_1, mul, getitem_2, getitem_3, mul_1, add, rms_norm, mul_2, linear, rms_norm_1, mul_5, linear_3], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._fused_rms_norm, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_red_fused__fused_rms_norm__to_copy_add_mul_select_unsqueeze_0.run(buf1, primals_1, primals_2, primals_3, primals_4, primals_12, buf2, buf4, buf23, 131072, 384, stream=stream4)
            buf3 = empty_strided_cuda((384, 1152), (1, 384), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_t_1.run(primals_5, buf3, 442368, stream=stream4)
            del primals_5
            buf5 = empty_strided_cuda((131072, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm, mul_2, linear], Original ATen: [aten._fused_rms_norm, aten.mul, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf4, (131072, 384), (384, 1), 0), buf3, out=buf5)
            buf6 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            buf7 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, view, getitem_1, getitem_2, triton_kernel_wrapper_mutation], Original ATen: [aten._unsafe_view, aten.view, aten.select]
            stream4 = get_raw_stream(4)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf5, (128, 1024, 24, 48), (1179648, 1152, 48, 1), 0), buf6, buf7, reinterpret_tensor(primals_6, (1024, 24), (24, 1), 0), reinterpret_tensor(primals_7, (1024, 24), (24, 1), 0), 2097152, 1, 1, stream=stream4)
            buf10 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, view, getitem, v], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone]
            stream4 = get_raw_stream(4)
            triton_poi_fused__unsafe_view_clone_slice_view_2.run(buf5, buf10, 50331648, stream=stream4)
            # Topologically Sorted Source Nodes: [linear, view, getitem, v, transpose_2, scaled_dot_product_attention], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf11 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf6, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf7, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf10, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), 0.0, True, scale=0.14433756729740646)
            buf12 = buf11[0]
            assert_size_stride(buf12, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf12, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf13 = buf11[1]
            assert_size_stride(buf13, (128, 8, 1024), (8192, 1024, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf13, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf14 = buf11[6]
            assert_size_stride(buf14, (2, ), (1, ), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf14, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf15 = buf11[7]
            assert_size_stride(buf15, (), (), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf15, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf11
            buf17 = empty_strided_cuda((12, 8), (1, 12), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_t_3.run(primals_8, buf17, 96, stream=stream4)
            del primals_8
            buf18 = empty_strided_cuda((128, 1024, 12), (12288, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm, mul_2, getitem_7, contiguous_3, linear_1], Original ATen: [aten._fused_rms_norm, aten.mul, aten.slice, aten.clone, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_4.run(buf2, primals_4, buf18, 1572864, stream=stream4)
            del buf2
            buf19 = empty_strided_cuda((131072, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm, mul_2, getitem_7, contiguous_3, linear_1], Original ATen: [aten._fused_rms_norm, aten.mul, aten.slice, aten.clone, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf18, (131072, 12), (12, 1), 0), buf17, out=buf19)
            buf20 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_3, linear_1, mul_3, sigmoid, getitem_8, mul_4], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
            stream4 = get_raw_stream(4)
            triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5.run(buf12, buf19, buf20, 50331648, stream=stream4)
            buf21 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_3, linear_1, mul_3, sigmoid, getitem_8, mul_4, reshape, linear_2], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf20, (131072, 384), (384, 1), 0), reinterpret_tensor(primals_9, (384, 384), (1, 384), 0), out=buf21)
            buf22 = empty_strided_cuda((384, 1536), (1, 384), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy, aten.t]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_t_6.run(primals_13, buf22, 589824, stream=stream4)
            del primals_13
            buf24 = empty_strided_cuda((131072, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_1, mul_5, linear_3], Original ATen: [aten._fused_rms_norm, aten.mul, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf23, (131072, 384), (384, 1), 0), buf22, out=buf24)
            buf25 = empty_strided_cuda((1536, 384), (1, 1536), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy, aten.t]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_t_6.run(primals_14, buf25, 589824, stream=stream4)
            del primals_14
            buf26 = empty_strided_cuda((128, 1024, 1536), (1572864, 1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_3, leaky_relu, square, linear_4], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_7.run(buf24, buf26, 201326592, stream=stream4)
            buf27 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_3, leaky_relu, square, linear_4], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf26, (131072, 1536), (1536, 1), 0), buf25, out=buf27)
            buf28 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            buf29 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
            buf30 = reinterpret_tensor(buf29, (128, 1024, 1), (1024, 1, 1), 0); del buf29  # reuse
            buf31 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            buf33 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            buf52 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, getitem_1, mul, getitem_2, getitem_3, mul_1, add, linear_2, getitem_9, getitem_10, linear_4, mul_6, add_1, mul_7, add_2, getitem_11, getitem_12, mul_8, getitem_13, getitem_14, mul_9, add_3, rms_norm_2, mul_10, linear_5, rms_norm_3, mul_13, linear_8], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._unsafe_view, aten._fused_rms_norm, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_red_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_8.run(buf30, primals_1, primals_2, primals_3, primals_10, buf21, primals_11, buf27, primals_15, primals_16, primals_22, buf28, buf31, buf33, buf52, 131072, 384, stream=stream4)
            buf32 = empty_strided_cuda((384, 1152), (1, 384), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy, aten.t]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_t_1.run(primals_17, buf32, 442368, stream=stream4)
            del primals_17
            buf34 = empty_strided_cuda((131072, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_2, mul_10, linear_5], Original ATen: [aten._fused_rms_norm, aten.mul, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf33, (131072, 384), (384, 1), 0), buf32, out=buf34)
            buf35 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            buf36 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_1, getitem_2, linear_5, view_1, triton_kernel_wrapper_mutation], Original ATen: [aten.select, aten._unsafe_view, aten.view]
            stream4 = get_raw_stream(4)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf34, (128, 1024, 24, 48), (1179648, 1152, 48, 1), 0), buf35, buf36, reinterpret_tensor(primals_6, (1024, 24), (24, 1), 0), reinterpret_tensor(primals_7, (1024, 24), (24, 1), 0), 2097152, 1, 1, stream=stream4)
            buf39 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_5, view_1, getitem, v], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone]
            stream4 = get_raw_stream(4)
            triton_poi_fused__unsafe_view_clone_slice_view_2.run(buf34, buf39, 50331648, stream=stream4)
            # Topologically Sorted Source Nodes: [linear_5, view_1, getitem, v, transpose_6, scaled_dot_product_attention_1], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf40 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf35, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf36, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf39, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), 0.0, True, scale=0.14433756729740646)
            buf41 = buf40[0]
            assert_size_stride(buf41, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf41, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf42 = buf40[1]
            assert_size_stride(buf42, (128, 8, 1024), (8192, 1024, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf42, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf43 = buf40[6]
            assert_size_stride(buf43, (2, ), (1, ), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf43, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf44 = buf40[7]
            assert_size_stride(buf44, (), (), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf44, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf40
            buf46 = empty_strided_cuda((12, 8), (1, 12), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy, aten.t]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_t_3.run(primals_18, buf46, 96, stream=stream4)
            del primals_18
            buf47 = empty_strided_cuda((128, 1024, 12), (12288, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_2, mul_10, getitem_18, contiguous_8, linear_6], Original ATen: [aten._fused_rms_norm, aten.mul, aten.slice, aten.clone, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_9.run(buf31, primals_16, buf47, 1572864, stream=stream4)
            del buf31
            buf48 = empty_strided_cuda((131072, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_2, mul_10, getitem_18, contiguous_8, linear_6], Original ATen: [aten._fused_rms_norm, aten.mul, aten.slice, aten.clone, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf47, (131072, 12), (12, 1), 0), buf46, out=buf48)
            buf49 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_7, linear_6, mul_11, sigmoid_1, getitem_19, mul_12], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
            stream4 = get_raw_stream(4)
            triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5.run(buf41, buf48, buf49, 50331648, stream=stream4)
            buf50 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_7, linear_6, mul_11, sigmoid_1, getitem_19, mul_12, reshape_1, linear_7], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf49, (131072, 384), (384, 1), 0), reinterpret_tensor(primals_19, (384, 384), (1, 384), 0), out=buf50)
            buf51 = empty_strided_cuda((384, 1536), (1, 384), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy, aten.t]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_t_6.run(primals_23, buf51, 589824, stream=stream4)
            del primals_23
            buf53 = empty_strided_cuda((131072, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_3, mul_13, linear_8], Original ATen: [aten._fused_rms_norm, aten.mul, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf52, (131072, 384), (384, 1), 0), buf51, out=buf53)
            buf54 = empty_strided_cuda((1536, 384), (1, 1536), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten._to_copy, aten.t]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_t_6.run(primals_24, buf54, 589824, stream=stream4)
            del primals_24
            buf55 = empty_strided_cuda((128, 1024, 1536), (1572864, 1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_8, leaky_relu_1, square_1, linear_9], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy__unsafe_view_leaky_relu_pow_7.run(buf53, buf55, 201326592, stream=stream4)
            buf56 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_8, leaky_relu_1, square_1, linear_9], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf55, (131072, 1536), (1536, 1), 0), buf54, out=buf56)
            buf57 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            buf58 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            buf59 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
            buf60 = reinterpret_tensor(buf59, (128, 1024, 1), (1024, 1, 1), 0); del buf59  # reuse
            buf62 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            buf81 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_11, getitem_12, mul_8, getitem_13, getitem_14, mul_9, add_3, linear_7, getitem_20, getitem_21, linear_9, mul_14, add_4, mul_15, add_5, getitem_22, getitem_23, mul_16, getitem_24, getitem_25, mul_17, add_6, rms_norm_4, mul_18, linear_10, rms_norm_5, mul_21, linear_13], Original ATen: [aten.select, aten.unsqueeze, aten.mul, aten.add, aten._unsafe_view, aten._fused_rms_norm, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_10.run(buf60, primals_15, buf28, primals_3, primals_20, buf50, primals_21, buf56, primals_25, primals_26, primals_32, buf57, buf58, buf62, buf81, 131072, 384, stream=stream4)
            buf61 = empty_strided_cuda((384, 1152), (1, 384), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten._to_copy, aten.t]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_t_1.run(primals_27, buf61, 442368, stream=stream4)
            del primals_27
            buf63 = empty_strided_cuda((131072, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_4, mul_18, linear_10], Original ATen: [aten._fused_rms_norm, aten.mul, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf62, (131072, 384), (384, 1), 0), buf61, out=buf63)
            buf64 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            buf65 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_1, getitem_2, linear_10, view_2, triton_kernel_wrapper_mutation], Original ATen: [aten.select, aten._unsafe_view, aten.view]
            stream4 = get_raw_stream(4)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf63, (128, 1024, 24, 48), (1179648, 1152, 48, 1), 0), buf64, buf65, reinterpret_tensor(primals_6, (1024, 24), (24, 1), 0), reinterpret_tensor(primals_7, (1024, 24), (24, 1), 0), 2097152, 1, 1, stream=stream4)
            buf68 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_10, view_2, getitem, v], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone]
            stream4 = get_raw_stream(4)
            triton_poi_fused__unsafe_view_clone_slice_view_2.run(buf63, buf68, 50331648, stream=stream4)
            # Topologically Sorted Source Nodes: [linear_10, view_2, getitem, v, transpose_10, scaled_dot_product_attention_2], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf69 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf64, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf65, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf68, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), 0.0, True, scale=0.14433756729740646)
            buf70 = buf69[0]
            assert_size_stride(buf70, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf70, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf71 = buf69[1]
            assert_size_stride(buf71, (128, 8, 1024), (8192, 1024, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf71, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf72 = buf69[6]
            assert_size_stride(buf72, (2, ), (1, ), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf72, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf73 = buf69[7]
            assert_size_stride(buf73, (), (), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf73, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf69
            buf75 = empty_strided_cuda((12, 8), (1, 12), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten._to_copy, aten.t]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_t_3.run(primals_28, buf75, 96, stream=stream4)
            del primals_28
            buf76 = empty_strided_cuda((128, 1024, 12), (12288, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_4, mul_18, getitem_29, contiguous_13, linear_11], Original ATen: [aten._fused_rms_norm, aten.mul, aten.slice, aten.clone, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__fused_rms_norm__to_copy_clone_mul_slice_11.run(buf58, buf60, primals_26, buf76, 131072, 12, stream=stream4)
            buf77 = empty_strided_cuda((131072, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_4, mul_18, getitem_29, contiguous_13, linear_11], Original ATen: [aten._fused_rms_norm, aten.mul, aten.slice, aten.clone, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf76, (131072, 12), (12, 1), 0), buf75, out=buf77)
            buf78 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_11, linear_11, mul_19, sigmoid_2, getitem_30, mul_20], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
            stream4 = get_raw_stream(4)
            triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_5.run(buf70, buf77, buf78, 50331648, stream=stream4)
            buf79 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_11, linear_11, mul_19, sigmoid_2, getitem_30, mul_20, reshape_2, linear_12], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf78, (131072, 384), (384, 1), 0), reinterpret_tensor(primals_29, (384, 384), (1, 384), 0), out=buf79)
            buf80 = empty_strided_cuda((384, 1536), (1, 384), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten._to_copy, aten.t]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_t_6.run(primals_33, buf80, 589824, stream=stream4)
            del primals_33
            buf82 = empty_strided_cuda((131072, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_4, rms_norm_5, mul_21, linear_13], Original ATen: [aten._fused_rms_norm, aten.mul, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf81, (131072, 384), (384, 1), 0), buf80, out=buf82)
            buf83 = empty_strided_cuda((128, 1024, 1536), (1572864, 1536, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_13, leaky_relu_2, square_2], Original ATen: [aten._unsafe_view, aten.leaky_relu, aten.pow]
            stream4 = get_raw_stream(4)
            triton_poi_fused__unsafe_view_leaky_relu_pow_12.run(buf82, buf83, 201326592, stream=stream4)
        return (buf83, reinterpret_tensor(primals_30, (1, 1, 384), (384, 384, 1), 0), reinterpret_tensor(buf79, (128, 1024, 384), (393216, 384, 1), 0), buf58, reinterpret_tensor(primals_31, (1, 1, 384), (384, 384, 1), 0), primals_1, primals_2, primals_3, primals_4, primals_9, primals_10, primals_11, primals_12, primals_15, primals_16, primals_19, primals_20, primals_21, primals_22, primals_25, primals_26, primals_29, primals_32, buf1, reinterpret_tensor(buf4, (131072, 384), (384, 1), 0), reinterpret_tensor(buf5, (128, 1024, 24, 48), (1179648, 1152, 48, 1), 0), reinterpret_tensor(primals_6, (1024, 24), (24, 1), 0), reinterpret_tensor(primals_7, (1024, 24), (24, 1), 0), reinterpret_tensor(buf10, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf6, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf7, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), buf12, buf13, buf14, buf15, reinterpret_tensor(buf18, (131072, 12), (12, 1), 0), buf19, reinterpret_tensor(buf20, (131072, 384), (384, 1), 0), buf21, reinterpret_tensor(buf23, (131072, 384), (384, 1), 0), buf24, reinterpret_tensor(buf26, (131072, 1536), (1536, 1), 0), buf27, buf28, buf30, reinterpret_tensor(buf33, (131072, 384), (384, 1), 0), reinterpret_tensor(buf34, (128, 1024, 24, 48), (1179648, 1152, 48, 1), 0), reinterpret_tensor(buf39, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf35, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf36, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), buf41, buf42, buf43, buf44, reinterpret_tensor(buf47, (131072, 12), (12, 1), 0), buf48, reinterpret_tensor(buf49, (131072, 384), (384, 1), 0), buf50, reinterpret_tensor(buf52, (131072, 384), (384, 1), 0), buf53, reinterpret_tensor(buf55, (131072, 1536), (1536, 1), 0), buf56, buf57, buf58, buf60, reinterpret_tensor(buf62, (131072, 384), (384, 1), 0), reinterpret_tensor(buf63, (128, 1024, 24, 48), (1179648, 1152, 48, 1), 0), reinterpret_tensor(buf68, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf64, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf65, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), buf70, buf71, buf72, buf73, reinterpret_tensor(buf76, (131072, 12), (12, 1), 0), buf77, reinterpret_tensor(buf78, (131072, 384), (384, 1), 0), reinterpret_tensor(buf81, (131072, 384), (384, 1), 0), buf82, reinterpret_tensor(buf80, (1536, 384), (384, 1), 0), reinterpret_tensor(buf75, (8, 12), (12, 1), 0), reinterpret_tensor(buf61, (1152, 384), (384, 1), 0), reinterpret_tensor(buf54, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf51, (1536, 384), (384, 1), 0), reinterpret_tensor(buf46, (8, 12), (12, 1), 0), reinterpret_tensor(buf32, (1152, 384), (384, 1), 0), reinterpret_tensor(buf25, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf22, (1536, 384), (384, 1), 0), reinterpret_tensor(buf17, (8, 12), (12, 1), 0), reinterpret_tensor(buf3, (1152, 384), (384, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((2, 384), (384, 1), device='cuda:4', dtype=torch.float32)
    primals_2 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:4', dtype=torch.float32)
    primals_3 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_4 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_5 = rand_strided((1152, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_6 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_7 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_8 = rand_strided((8, 12), (12, 1), device='cuda:4', dtype=torch.float32)
    primals_9 = rand_strided((384, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_10 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_11 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_12 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_13 = rand_strided((1536, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_14 = rand_strided((384, 1536), (1536, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_15 = rand_strided((2, 384), (384, 1), device='cuda:4', dtype=torch.float32)
    primals_16 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_17 = rand_strided((1152, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_18 = rand_strided((8, 12), (12, 1), device='cuda:4', dtype=torch.float32)
    primals_19 = rand_strided((384, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_20 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_21 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_22 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_23 = rand_strided((1536, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_24 = rand_strided((384, 1536), (1536, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_25 = rand_strided((2, 384), (384, 1), device='cuda:4', dtype=torch.float32)
    primals_26 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_27 = rand_strided((1152, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_28 = rand_strided((8, 12), (12, 1), device='cuda:4', dtype=torch.float32)
    primals_29 = rand_strided((384, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_30 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_31 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_32 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_33 = rand_strided((1536, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
