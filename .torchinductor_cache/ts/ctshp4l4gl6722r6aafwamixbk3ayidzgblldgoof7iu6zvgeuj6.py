# AOT ID: ['9_backward']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/v2/cv2tvej2bqodp4cojck2tfijnrkkgxi6qf5pixinioz2lbhvlbez.py
# Topologically Sorted Source Nodes: [view_46, convert_element_type_65, to_22, convert_element_type_67, mul_26, rms_norm_6, mul_28, sum_7, div_3, mul_29, sub_3, mul_30, mul_31, sum_8, convert_element_type_68], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten._fused_rms_norm]
# Source node to ATen node mapping:
#   convert_element_type_65 => convert_element_type_65
#   convert_element_type_67 => convert_element_type_67
#   convert_element_type_68 => convert_element_type_68
#   div_3 => div_3
#   mul_26 => mul_26
#   mul_28 => mul_28
#   mul_29 => mul_29
#   mul_30 => mul_30
#   mul_31 => mul_31
#   rms_norm_6 => convert_element_type_63, mul_24
#   sub_3 => sub_3
#   sum_7 => sum_7
#   sum_8 => sum_8
#   to_22 => convert_element_type_62
#   view_46 => view_46
# Graph fragment:
#   %add_11 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_11]
#   %rsqrt_6 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5" = PlaceHolder[target=rsqrt_6]
#   %tangents_1 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=tangents_1]
#   %primals_26 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_26]
#   %sum_7 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:5" = PlaceHolder[target=sum_7]
#   %view_46 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%tangents_1, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_65 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_46, torch.float32), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_26, torch.bfloat16), kwargs = {})
#   %convert_element_type_67 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_62, torch.float32), kwargs = {})
#   %mul_26 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_65, %convert_element_type_67), kwargs = {})
#   %convert_element_type_63 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.float32), kwargs = {})
#   %mul_24 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_63, %rsqrt_6), kwargs = {})
#   %mul_28 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %mul_26), kwargs = {})
#   %sum_7 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_28, [2], True), kwargs = {})
#   %div_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_24, 384), kwargs = {})
#   %mul_29 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %sum_7), kwargs = {})
#   %sub_3 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_26, %mul_29), kwargs = {})
#   %mul_30 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_6), kwargs = {})
#   %mul_31 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_65, %mul_24), kwargs = {})
#   %sum_8 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_31, [0, 1]), kwargs = {})
#   %convert_element_type_68 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_30, torch.bfloat16), kwargs = {})
#   return %sum_7,%convert_element_type_68
triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_view_0 = async_compile.triton('triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_view_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr1': '*bf16', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 0, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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
        tmp15 = tl.full([1, 1], 0.0026041666666666665, tl.float32)
        tmp16 = tmp3 * tmp15
        tmp17 = tmp16 * tmp14
        tmp18 = tmp9 - tmp17
        tmp19 = tmp18 * tmp2
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp5 * tmp3
        tl.store(out_ptr1 + (r0_1 + 384*x0), tmp20, r0_mask)
        tmp22 = tl.sum(tmp21, 0)
        tmp23 = accum0 + tmp22
        accum0 = tmp23
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/xw/cxw4yt7mmnbs4fmqsqexf2klyyijv5trc3x3b2jr5dgfru5bot7w.py
# Topologically Sorted Source Nodes: [reshape, getattr_6, contiguous_2, triton_kernel_wrapper_mutation], Original ATen: [aten.view, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   contiguous_2 => clone_9
#   getattr_6 => permute_35
#   reshape => view_47
#   triton_kernel_wrapper_mutation => triton_kernel_wrapper_mutation_5
# Graph fragment:
#   %primals_25 : Tensor "bf16[384, 1536][1536, 1]cuda:5" = PlaceHolder[target=primals_25]
#   %view_47 : Tensor "bf16[131072, 384][384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_68, [-1, 384]), kwargs = {})
#   %permute_35 : Tensor "bf16[1536, 384][1, 1536]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_25, [1, 0]), kwargs = {})
#   %clone_9 : Tensor "bf16[1536, 384][384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_35,), kwargs = {memory_format: torch.contiguous_format})
#   %triton_kernel_wrapper_mutation_5 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 35, constant_args_idx: 32, grid: [(24576, 1, 1)], tma_descriptor_metadata: {}, kwargs: {A: %view_47, B: %clone_9, C: %empty_12, AUX: %getitem_37}})
#   return %buf6
triton_poi_fused_clone_permute_view_1 = async_compile.triton('triton_poi_fused_clone_permute_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_permute_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1179648, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_permute_view_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1536*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + 384*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# Original path: /workspace/parameter-golf/train_gpt_parcae.py:4795
_leaky_relu_sq_matmul_kernel_0 = async_compile.triton('_leaky_relu_sq_matmul_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 4, 'num_stages': 3}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_leaky_relu_sq_matmul_kernel_0', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'A': '*bf16', 'B': '*bf16', 'C': '*bf16', 'AUX': '*bf16', 'M': 'constexpr', 'N': 'constexpr', 'K': 'constexpr', 'stride_am': 'constexpr', 'stride_ak': 'constexpr', 'stride_bn': 'constexpr', 'stride_bk': 'constexpr', 'stride_cm': 'constexpr', 'stride_cn': 'constexpr', 'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr', 'BLOCK_K': 'constexpr', 'GROUP_M': 'constexpr', 'FORWARD': 'constexpr', 'NEGATIVE_SLOPE': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'M': 131072, 'N': 1536, 'K': 384, 'stride_am': 384, 'stride_ak': 1, 'stride_bn': 384, 'stride_bk': 1, 'stride_cm': 1536, 'stride_cn': 1, 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'FORWARD': False, 'NEGATIVE_SLOPE': 0.5}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/xd/cxdkq6cnueoktjq5uednytpk25olrfwjk4qodzsf7ft2v4s57onv.py
# Topologically Sorted Source Nodes: [reshape_as, convert_element_type_77, to_19, convert_element_type_79, mul_32, rms_norm_5, mul_34, sum_9, div_4, mul_35, sub_4, mul_36, mul_37, sum_10, convert_element_type_80, add_13], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten._fused_rms_norm, aten.add]
# Source node to ATen node mapping:
#   add_13 => add_13
#   convert_element_type_77 => convert_element_type_77
#   convert_element_type_79 => convert_element_type_79
#   convert_element_type_80 => convert_element_type_80
#   div_4 => div_4
#   mul_32 => mul_32
#   mul_34 => mul_34
#   mul_35 => mul_35
#   mul_36 => mul_36
#   mul_37 => mul_37
#   reshape_as => view_48
#   rms_norm_5 => convert_element_type_58, mul_22
#   sub_4 => sub_4
#   sum_10 => sum_10
#   sum_9 => sum_9
#   to_19 => convert_element_type_57
# Graph fragment:
#   %add_9 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_9]
#   %rsqrt_5 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5" = PlaceHolder[target=rsqrt_5]
#   %mm_15 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_15]
#   %primals_23 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_23]
#   %convert_element_type_68 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=convert_element_type_68]
#   %sum_9 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:5" = PlaceHolder[target=sum_9]
#   %view_48 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_15, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_77 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_48, torch.float32), kwargs = {})
#   %convert_element_type_57 : Tensor "bf16[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_23, torch.bfloat16), kwargs = {})
#   %convert_element_type_79 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_57, torch.float32), kwargs = {})
#   %mul_32 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_77, %convert_element_type_79), kwargs = {})
#   %convert_element_type_58 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.float32), kwargs = {})
#   %mul_22 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_58, %rsqrt_5), kwargs = {})
#   %mul_34 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %mul_32), kwargs = {})
#   %sum_9 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_34, [2], True), kwargs = {})
#   %div_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_22, 384), kwargs = {})
#   %mul_35 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sum_9), kwargs = {})
#   %sub_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_32, %mul_35), kwargs = {})
#   %mul_36 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_5), kwargs = {})
#   %mul_37 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_77, %mul_22), kwargs = {})
#   %sum_10 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_37, [0, 1]), kwargs = {})
#   %convert_element_type_80 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_36, torch.bfloat16), kwargs = {})
#   %add_13 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_68, %convert_element_type_80), kwargs = {})
#   return %sum_9,%add_13
triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_view_2 = async_compile.triton('triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_view_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 0, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_view_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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
        tmp15 = tl.load(in_out_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
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
        tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp22, r0_mask)
        tmp24 = tl.sum(tmp23, 0)
        tmp25 = accum0 + tmp24
        accum0 = tmp25
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/27/c27lxtzry6qtejnttzji7sgoiv6se5fugbmonbm5m2ktm5alr4n2.py
# Topologically Sorted Source Nodes: [view_50, convert_element_type_87, view_51, transpose_11, view_5, unsqueeze_2, mul_9, sub_2, reshape_6, mul_38, linear_8, mul_10, sigmoid_2, getitem_16, mul_39, sum_11, convert_element_type_90, squeeze, convert_element_type_91, convert_element_type_92, sub_5, mul_40, mul_41, convert_element_type_93, mul_42, view_54, neg, convert_element_type_99, mul_44, sum_12, expand_3, mul_46, convert_element_type_100, add_14, view_55, permute_48, transpose_10, _scaled_dot_product_flash_attention_backward], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.unsqueeze, aten.mul, aten.sub, aten._unsafe_view, aten.sigmoid, aten.sum, aten.squeeze, aten.sigmoid_backward, aten.neg, aten.expand, aten.add, aten._scaled_dot_product_flash_attention_backward]
# Source node to ATen node mapping:
#   _scaled_dot_product_flash_attention_backward => _scaled_dot_product_flash_attention_backward
#   add_14 => add_14
#   convert_element_type_100 => convert_element_type_100
#   convert_element_type_87 => convert_element_type_87
#   convert_element_type_90 => convert_element_type_90
#   convert_element_type_91 => convert_element_type_91
#   convert_element_type_92 => convert_element_type_92
#   convert_element_type_93 => convert_element_type_93
#   convert_element_type_99 => convert_element_type_99
#   expand_3 => expand_3
#   getitem_16 => unsqueeze_5
#   linear_8 => view_36
#   mul_10 => mul_20
#   mul_38 => mul_38
#   mul_39 => mul_39
#   mul_40 => mul_40
#   mul_41 => mul_41
#   mul_42 => mul_42
#   mul_44 => mul_44
#   mul_46 => mul_46
#   mul_9 => mul_19
#   neg => neg
#   permute_48 => permute_48
#   reshape_6 => view_34
#   sigmoid_2 => sigmoid_2
#   squeeze => squeeze
#   sub_2 => sub_2
#   sub_5 => sub_5
#   sum_11 => sum_11
#   sum_12 => sum_12
#   transpose_10 => permute_28
#   transpose_11 => permute_31
#   unsqueeze_2 => unsqueeze_4
#   view_5 => view_33
#   view_50 => view_50
#   view_51 => view_51
#   view_54 => view_54
#   view_55 => view_55
# Graph fragment:
#   %mm_17 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_17]
#   %mm_10 : Tensor "bf16[131072, 8][8, 1]cuda:5" = PlaceHolder[target=mm_10]
#   %div_2 : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5" = PlaceHolder[target=div_2]
#   %getitem_28 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:5" = PlaceHolder[target=getitem_28]
#   %sum_6 : Tensor "f32[128, 1024, 4, 2, 1][8192, 8, 2, 1, 1]cuda:5" = PlaceHolder[target=sum_6]
#   %sum_12 : Tensor "f32[128, 1024, 4, 2, 1][8192, 8, 2, 1, 1048576]cuda:5" = PlaceHolder[target=sum_12]
#   %sum_11 : Tensor "f32[128, 1024, 8, 1][8192, 8, 1, 1048576]cuda:5" = PlaceHolder[target=sum_11]
#   %view_50 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_17, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_87 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_50, torch.float32), kwargs = {})
#   %view_51 : Tensor "f32[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_87, [128, 1024, 8, 48]), kwargs = {})
#   %permute_31 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_28, [0, 2, 1, 3]), kwargs = {})
#   %view_33 : Tensor "bf16[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_31, [128, 1024, 4, 2, 48]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[128, 1024, 4, 1, 48][196608, 192, 48, 48, 1]cuda:5"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%div_2, -2), kwargs = {})
#   %mul_19 : Tensor "f32[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_6, %unsqueeze_4), kwargs = {})
#   %sub_2 : Tensor "f32[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_33, %mul_19), kwargs = {})
#   %view_34 : Tensor "f32[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_2, [128, 1024, 8, 48]), kwargs = {})
#   %mul_38 : Tensor "f32[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_51, %view_34), kwargs = {})
#   %view_36 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_10, [128, 1024, 8]), kwargs = {})
#   %mul_20 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_36, 0.5), kwargs = {})
#   %sigmoid_2 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_20,), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[128, 1024, 8, 1][8192, 8, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_2, 3), kwargs = {})
#   %mul_39 : Tensor "f32[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_51, %unsqueeze_5), kwargs = {})
#   %sum_11 : Tensor "f32[128, 1024, 8, 1][8192, 8, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_38, [3], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_90 : Tensor "bf16[128, 1024, 8, 1][8192, 8, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_11, torch.bfloat16), kwargs = {})
#   %squeeze : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%convert_element_type_90, 3), kwargs = {})
#   %convert_element_type_91 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%squeeze, torch.float32), kwargs = {})
#   %convert_element_type_92 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_2, torch.float32), kwargs = {})
#   %sub_5 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_92), kwargs = {})
#   %mul_40 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_92, %sub_5), kwargs = {})
#   %mul_41 : Tensor "f32[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_91, %mul_40), kwargs = {})
#   %convert_element_type_93 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_41, torch.bfloat16), kwargs = {})
#   %mul_42 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_93, 0.5), kwargs = {})
#   %view_54 : Tensor "f32[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_39, [128, 1024, 4, 2, 48]), kwargs = {})
#   %neg : Tensor "f32[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%view_54,), kwargs = {})
#   %convert_element_type_99 : Tensor "bf16[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_54, torch.bfloat16), kwargs = {})
#   %mul_44 : Tensor "f32[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %unsqueeze_4), kwargs = {})
#   %sum_12 : Tensor "f32[128, 1024, 4, 2, 1][8192, 8, 2, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_44, [4], True), kwargs = {dtype: torch.float32})
#   %expand_3 : Tensor "f32[128, 1024, 4, 2, 48][8192, 8, 2, 1, 0]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.expand.default](args = (%sum_12, [128, 1024, 4, 2, 48]), kwargs = {})
#   %mul_46 : Tensor "f32[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_3, %unsqueeze_4), kwargs = {})
#   %convert_element_type_100 : Tensor "bf16[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_46, torch.bfloat16), kwargs = {})
#   %add_14 : Tensor "bf16[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_99, %convert_element_type_100), kwargs = {})
#   %view_55 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_14, [128, 1024, 8, 48]), kwargs = {})
#   %permute_48 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_55, [0, 2, 1, 3]), kwargs = {})
#   %permute_28 : Tensor "bf16[128, 4, 1024, 48][196608, 48, 192, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%clone_4, [0, 2, 1, 3]), kwargs = {})
#   %_scaled_dot_product_flash_attention_backward : [num_users=3] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention_backward.default](args = (%permute_48, %permute_29, %permute_30, %permute_28, %getitem_28, %getitem_29, None, None, 1024, 1024, 0.0, True, %getitem_34, %getitem_35), kwargs = {scale: 0.14433756729740646})
#   return %sum_12,%sum_11,%buf25,%mul_42
triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_add_expand_mul_neg_sigmoid_sigmoid_backward_squeeze_sub_sum_transpose_unsqueeze_view_3 = async_compile.triton('triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_add_expand_mul_neg_sigmoid_sigmoid_backward_squeeze_sub_sum_transpose_unsqueeze_view_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_add_expand_mul_neg_sigmoid_sigmoid_backward_squeeze_sub_sum_transpose_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 3, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 14680064, 'r0_': 704643072}}
)
@triton.jit
def triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_add_expand_mul_neg_sigmoid_sigmoid_backward_squeeze_sub_sum_transpose_unsqueeze_view_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_2 = r0_index
    x5 = xindex
    x1 = xindex // 2
    x3 = (xindex % 8)
    x4 = xindex // 8
    tmp0 = tl.load(in_ptr0 + (r0_2 + 48*x5), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (r0_2 + 48*x1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + (r0_2 + 48*x5), r0_mask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (x5), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr2 + (r0_2 + 48*(x3 // 2) + 192*x4), r0_mask, other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1, 1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp1 * tmp6
    tmp8 = -tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 - tmp19
    tmp21 = tmp1 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(r0_mask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
    tmp26 = tmp7.to(tl.float32)
    tmp27 = tmp14 * tmp18
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 + tmp28
    tmp30 = tmp25.to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1, 1], 1.0, tl.float32)
    tmp33 = tmp32 - tmp6
    tmp34 = tmp6 * tmp33
    tmp35 = tmp31 * tmp34
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36 * tmp3
    tl.store(out_ptr2 + (r0_2 + 48*x5), tmp29, r0_mask)
    tl.store(out_ptr3 + (x5), tmp37, None)
    tl.store(out_ptr0 + (x5), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/k7/ck7b762cgjwq2ytbgg54lf74ab7yp5mkeikxyywjh24nco6aoojc.py
# Topologically Sorted Source Nodes: [constant_pad_nd_default_2], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   constant_pad_nd_default_2 => constant_pad_nd_default_2
# Graph fragment:
#   %view_35 : Tensor "bf16[131072, 12][12, 1]cuda:5" = PlaceHolder[target=view_35]
#   %constant_pad_nd_default_2 : Tensor "bf16[131072, 16][16, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_35, [0, 4, 0, 0]), kwargs = {})
#   return %constant_pad_nd_default_2
triton_poi_fused_mm_4 = async_compile.triton('triton_poi_fused_mm_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 11534336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/4k/c4kqvifp4wjr23lyhtqhtublkoghujwt2m5fsle2q3fknt2r3vx6.py
# Topologically Sorted Source Nodes: [slice_tensor_2, convert_element_type_98], Original ATen: [aten.mm, aten._to_copy]
# Source node to ATen node mapping:
#   convert_element_type_98 => convert_element_type_98
#   slice_tensor_2 => slice_tensor_2
# Graph fragment:
#   %mm_default_2 : Tensor "bf16[8, 16][16, 1]cuda:5" = PlaceHolder[target=mm_default_2]
#   %slice_tensor_2 : Tensor "bf16[8, 12][16, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_default_2, 1, 0, -4), kwargs = {})
#   %convert_element_type_98 : Tensor "f32[8, 12][12, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_tensor_2, torch.float32), kwargs = {})
#   return %convert_element_type_98
triton_poi_fused__to_copy_mm_5 = async_compile.triton('triton_poi_fused__to_copy_mm_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mm_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/hh/chhmk5qhwv5daocykwqfro3ufgwxcfimmnunrhgokiicsl5xizud.py
# Topologically Sorted Source Nodes: [view_50, convert_element_type_87, view_51, transpose_11, view_5, linear_8, mul_10, sigmoid_2, getitem_16, mul_39, view_54, neg, mul_43, sum_13, expand_3, mul_45, sum_14, add_15, squeeze_1, normalize_2, div_6, neg_1, mul_47, sum_15], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.neg, aten.sum, aten.expand, aten.add, aten.squeeze, aten.clamp_min, aten.div]
# Source node to ATen node mapping:
#   add_15 => add_15
#   convert_element_type_87 => convert_element_type_87
#   div_6 => div_6
#   expand_3 => expand_3
#   getitem_16 => unsqueeze_5
#   linear_8 => view_36
#   mul_10 => mul_20
#   mul_39 => mul_39
#   mul_43 => mul_43
#   mul_45 => mul_45
#   mul_47 => mul_47
#   neg => neg
#   neg_1 => neg_1
#   normalize_2 => clamp_min_2, expand_2
#   sigmoid_2 => sigmoid_2
#   squeeze_1 => squeeze_1
#   sum_13 => sum_13
#   sum_14 => sum_14
#   sum_15 => sum_15
#   transpose_11 => permute_31
#   view_5 => view_33
#   view_50 => view_50
#   view_51 => view_51
#   view_54 => view_54
# Graph fragment:
#   %mm_17 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_17]
#   %mm_10 : Tensor "bf16[131072, 8][8, 1]cuda:5" = PlaceHolder[target=mm_10]
#   %sum_6 : Tensor "f32[128, 1024, 4, 2, 1][8192, 8, 2, 1, 1]cuda:5" = PlaceHolder[target=sum_6]
#   %sum_12 : Tensor "f32[128, 1024, 4, 2, 1][8192, 8, 2, 1, 1048576]cuda:5" = PlaceHolder[target=sum_12]
#   %getitem_28 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:5" = PlaceHolder[target=getitem_28]
#   %add_15 : Tensor "f32[128, 1024, 4, 1, 48][196608, 192, 48, 48, 1]cuda:5" = PlaceHolder[target=add_15]
#   %div_2 : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5" = PlaceHolder[target=div_2]
#   %pow_11 : Tensor "f32[128, 1024, 4, 1][4096, 4, 1, 1]cuda:5" = PlaceHolder[target=pow_11]
#   %view_50 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_17, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_87 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_50, torch.float32), kwargs = {})
#   %view_51 : Tensor "f32[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_87, [128, 1024, 8, 48]), kwargs = {})
#   %permute_31 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_28, [0, 2, 1, 3]), kwargs = {})
#   %view_33 : Tensor "bf16[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_31, [128, 1024, 4, 2, 48]), kwargs = {})
#   %view_36 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_10, [128, 1024, 8]), kwargs = {})
#   %mul_20 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_36, 0.5), kwargs = {})
#   %sigmoid_2 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_20,), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[128, 1024, 8, 1][8192, 8, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_2, 3), kwargs = {})
#   %mul_39 : Tensor "f32[128, 1024, 8, 48][393216, 384, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_51, %unsqueeze_5), kwargs = {})
#   %view_54 : Tensor "f32[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_39, [128, 1024, 4, 2, 48]), kwargs = {})
#   %neg : Tensor "f32[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%view_54,), kwargs = {})
#   %mul_43 : Tensor "f32[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %sum_6), kwargs = {})
#   %sum_13 : Tensor "f32[128, 1024, 4, 1, 48][196608, 192, 48, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_43, [3], True), kwargs = {dtype: torch.float32})
#   %expand_3 : Tensor "f32[128, 1024, 4, 2, 48][8192, 8, 2, 1, 0]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.expand.default](args = (%sum_12, [128, 1024, 4, 2, 48]), kwargs = {})
#   %mul_45 : Tensor "f32[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_3, %view_33), kwargs = {})
#   %sum_14 : Tensor "f32[128, 1024, 4, 1, 48][196608, 192, 48, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_45, [3], True), kwargs = {dtype: torch.float32})
#   %add_15 : Tensor "f32[128, 1024, 4, 1, 48][196608, 192, 48, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_13, %sum_14), kwargs = {})
#   %squeeze_1 : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dim](args = (%add_15, -2), kwargs = {})
#   %clamp_min_2 : Tensor "f32[128, 1024, 4, 1][4096, 4, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_11, 1e-12), kwargs = {})
#   %expand_2 : Tensor "f32[128, 1024, 4, 48][4096, 4, 1, 0]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_2, [128, 1024, 4, 48]), kwargs = {})
#   %div_6 : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_2, %expand_2), kwargs = {})
#   %neg_1 : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_1,), kwargs = {})
#   %mul_47 : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_1, %div_6), kwargs = {})
#   %sum_15 : Tensor "f32[128, 1024, 4, 1][4096, 4, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_47, [3], True), kwargs = {dtype: torch.float32})
#   return %add_15,%sum_15
triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_expand_mul_neg_sigmoid_squeeze_sum_transpose_unsqueeze_view_6 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_expand_mul_neg_sigmoid_squeeze_sum_transpose_unsqueeze_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_expand_mul_neg_sigmoid_squeeze_sum_transpose_unsqueeze_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 12, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6291456, 'r0_': 503316480}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_expand_mul_neg_sigmoid_squeeze_sum_transpose_unsqueeze_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 524288
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 96*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (2*x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (48 + r0_1 + 96*x0), r0_mask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (1 + 2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp19 = tl.load(in_ptr2 + (1 + 2*x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (2*x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (r0_1 + 96*x0), r0_mask, other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr3 + (1 + 2*x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (48 + r0_1 + 96*x0), r0_mask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr5 + (r0_1 + 48*x0), r0_mask, other=0.0)
    tmp34 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1, 1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp1 * tmp6
    tmp8 = -tmp7
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp13 * tmp3
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp12 * tmp16
    tmp18 = -tmp17
    tmp20 = tmp18 * tmp19
    tmp21 = tmp10 + tmp20
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 * tmp24
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 * tmp28
    tmp30 = tmp25 + tmp29
    tmp31 = tmp21 + tmp30
    tmp32 = -tmp31
    tmp35 = tl.full([1, 1], 1e-12, tl.float32)
    tmp36 = triton_helpers.maximum(tmp34, tmp35)
    tmp37 = (tmp33 / tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.where(r0_mask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (r0_1 + 48*x0), tmp31, r0_mask)
    tl.store(out_ptr1 + (x0), tmp42, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/a5/ca57ct76ghdwzx2ikul5tog35zxvwh3kqmxywbeipnk7hdwtixb6.py
# Topologically Sorted Source Nodes: [squeeze_1, normalize_2, div_7, convert_element_type_101, full_default_1, ge, where, div_8, eq, where_1, mul_48, convert_element_type_102, add_16, permute_49, add_17, getitem, copy_], Original ATen: [aten.squeeze, aten.clamp_min, aten.expand, aten.div, aten._to_copy, aten.scalar_tensor, aten.ge, aten.where, aten.eq, aten.masked_fill, aten.mul, aten.add, aten.transpose, aten.slice, aten.copy]
# Source node to ATen node mapping:
#   add_16 => add_16
#   add_17 => add_17
#   convert_element_type_101 => convert_element_type_101
#   convert_element_type_102 => convert_element_type_102
#   copy_ => copy
#   div_7 => div_7
#   div_8 => div_8
#   eq => eq
#   full_default_1 => full_default_1
#   ge => ge
#   getitem => slice_7
#   mul_48 => mul_48
#   normalize_2 => clamp_min_2, expand_2
#   permute_49 => permute_49
#   squeeze_1 => squeeze_1
#   where => where
#   where_1 => where_1
# Graph fragment:
#   %add_15 : Tensor "f32[128, 1024, 4, 1, 48][196608, 192, 48, 48, 1]cuda:5" = PlaceHolder[target=add_15]
#   %pow_11 : Tensor "f32[128, 1024, 4, 1][4096, 4, 1, 1]cuda:5" = PlaceHolder[target=pow_11]
#   %sum_15 : Tensor "f32[128, 1024, 4, 1][4096, 4, 1, 524288]cuda:5" = PlaceHolder[target=sum_15]
#   %clone_4 : Tensor "bf16[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5" = PlaceHolder[target=clone_4]
#   %getitem_42 : Tensor "bf16[128, 4, 1024, 48][196608, 48, 192, 1]cuda:5" = PlaceHolder[target=getitem_42]
#   %squeeze_1 : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dim](args = (%add_15, -2), kwargs = {})
#   %clamp_min_2 : Tensor "f32[128, 1024, 4, 1][4096, 4, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_11, 1e-12), kwargs = {})
#   %expand_2 : Tensor "f32[128, 1024, 4, 48][4096, 4, 1, 0]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_2, [128, 1024, 4, 48]), kwargs = {})
#   %div_7 : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%squeeze_1, %expand_2), kwargs = {})
#   %convert_element_type_101 : Tensor "bf16[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_7, torch.bfloat16), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:5"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:5, pin_memory: False})
#   %ge : Tensor "b8[128, 1024, 4, 1][4096, 4, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%pow_11, 1e-12), kwargs = {})
#   %where : Tensor "f32[128, 1024, 4, 1][4096, 4, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %sum_15, %full_default_1), kwargs = {})
#   %div_8 : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clone_4, %pow_11), kwargs = {})
#   %eq : Tensor "b8[128, 1024, 4, 1][4096, 4, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_11, 0), kwargs = {})
#   %where_1 : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_1, %div_8), kwargs = {})
#   %mul_48 : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %where_1), kwargs = {})
#   %convert_element_type_102 : Tensor "bf16[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_48, torch.bfloat16), kwargs = {})
#   %add_16 : Tensor "bf16[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_101, %convert_element_type_102), kwargs = {})
#   %permute_49 : Tensor "bf16[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_42, [0, 2, 1, 3]), kwargs = {})
#   %add_17 : Tensor "bf16[128, 1024, 4, 48][196608, 192, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %permute_49), kwargs = {})
#   %slice_7 : Tensor "bf16[128, 1024, 4, 48][786432, 768, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%empty_13, 2, 12, 9223372036854775807), kwargs = {})
#   %copy : Tensor "bf16[128, 1024, 4, 48][786432, 768, 48, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_7, %add_17), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[128, 1024, 16, 48][786432, 768, 48, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty_13, %copy, 2, 12, 9223372036854775807), kwargs = {})
#   return %slice_scatter_default_1
triton_poi_fused__to_copy_add_clamp_min_copy_div_eq_expand_ge_masked_fill_mul_scalar_tensor_slice_squeeze_transpose_where_7 = async_compile.triton('triton_poi_fused__to_copy_add_clamp_min_copy_div_eq_expand_ge_masked_fill_mul_scalar_tensor_slice_squeeze_transpose_where_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_min_copy_div_eq_expand_ge_masked_fill_mul_scalar_tensor_slice_squeeze_transpose_where_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 603979776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clamp_min_copy_div_eq_expand_ge_masked_fill_mul_scalar_tensor_slice_squeeze_transpose_where_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr0 + ((-576) + x3 + 192*x2), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + ((-12) + x1 + 4*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.full([1], 1e-12, tl.float32)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = (tmp3 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp4 >= tmp5
    tmp10 = tl.load(in_ptr2 + ((-12) + x1 + 4*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full([1], 0.0, tl.float32)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp4 == tmp11
    tmp14 = tl.load(in_ptr3 + ((-576) + x3 + 192*x2), tmp2, other=0.0).to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = (tmp15 / tmp4)
    tmp17 = tl.where(tmp13, tmp11, tmp16)
    tmp18 = tmp12 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp8 + tmp19
    tmp21 = tl.load(in_ptr4 + ((-576) + x3 + 192*x2), tmp2, other=0.0).to(tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = tl.full([1], float("nan"), tl.float32)
    tmp26 = tl.where(tmp2, tmp24, tmp25)
    tl.store(out_ptr0 + (x4), tmp26, None)
''', device_str='cuda')


# Original path: /workspace/parameter-golf/train_gpt_parcae.py:4479
_fused_qkv_postprocess_bwd_kernel_1 = async_compile.triton('_fused_qkv_postprocess_bwd_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 2, 'num_stages': 3}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_fused_qkv_postprocess_bwd_kernel_1', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'DQ': '*bf16', 'DK': '*bf16', 'QKV': '*bf16', 'DQKV': '*bf16', 'COS': '*bf16', 'SIN': '*bf16', 'TOTAL_ROWS': 'constexpr', 'SEQLEN': 'constexpr', 'H_Q': 'constexpr', 'H_KV': 'constexpr', 'HEAD_DIM': 'constexpr', 'ROPE_DIMS': 'constexpr', 'DO_QK_NORM': 'constexpr', 'ROPE_INTERLEAVED': 'constexpr', 'RMS_EPS': 'constexpr', 'BLOCK_D': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'TOTAL_ROWS': 131072, 'SEQLEN': 1024, 'H_Q': 8, 'H_KV': 4, 'HEAD_DIM': 48, 'ROPE_DIMS': 48, 'DO_QK_NORM': True, 'ROPE_INTERLEAVED': False, 'RMS_EPS': 1e-06, 'BLOCK_D': 64}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/dh/cdhovxy3uu7oa4c4to2mxmb7dxvlec2gy5ok4b3urlxxuriuyz7g.py
# Topologically Sorted Source Nodes: [view_53, full_default, view_63, add_18, convert_element_type_107, to_15, convert_element_type_109, mul_49, rms_norm_4, mul_51, sum_16, div_9, mul_52, sub_6, mul_53, mul_54, sum_17, convert_element_type_110, add_19], Original ATen: [aten.view, aten.slice_backward, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten._fused_rms_norm]
# Source node to ATen node mapping:
#   add_18 => add_18
#   add_19 => add_19
#   convert_element_type_107 => convert_element_type_107
#   convert_element_type_109 => convert_element_type_109
#   convert_element_type_110 => convert_element_type_110
#   div_9 => div_9
#   full_default => full_default
#   mul_49 => mul_49
#   mul_51 => mul_51
#   mul_52 => mul_52
#   mul_53 => mul_53
#   mul_54 => mul_54
#   rms_norm_4 => convert_element_type_44, mul_16
#   sub_6 => sub_6
#   sum_16 => sum_16
#   sum_17 => sum_17
#   to_15 => convert_element_type_43
#   view_53 => view_53
#   view_63 => view_63
# Graph fragment:
#   %add_7 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_7]
#   %rsqrt_4 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5" = PlaceHolder[target=rsqrt_4]
#   %mm_19 : Tensor "bf16[131072, 12][12, 1]cuda:5" = PlaceHolder[target=mm_19]
#   %mm_21 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_21]
#   %primals_19 : Tensor "f32[384][1]cuda:5" = PlaceHolder[target=primals_19]
#   %add_13 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5" = PlaceHolder[target=add_13]
#   %sum_16 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:5" = PlaceHolder[target=sum_16]
#   %view_53 : Tensor "bf16[128, 1024, 12][12288, 12, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_19, [128, 1024, 12]), kwargs = {})
#   %full_default : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([128, 1024, 384], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:5, pin_memory: False})
#   %slice_scatter_default : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %view_53, 2, 0, 12), kwargs = {})
#   %view_63 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_21, [128, 1024, 384]), kwargs = {})
#   %add_18 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %view_63), kwargs = {})
#   %convert_element_type_107 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_18, torch.float32), kwargs = {})
#   %convert_element_type_43 : Tensor "bf16[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_19, torch.bfloat16), kwargs = {})
#   %convert_element_type_109 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_43, torch.float32), kwargs = {})
#   %mul_49 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_107, %convert_element_type_109), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_7, torch.float32), kwargs = {})
#   %mul_16 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_44, %rsqrt_4), kwargs = {})
#   %mul_51 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %mul_49), kwargs = {})
#   %sum_16 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_51, [2], True), kwargs = {})
#   %div_9 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_16, 384), kwargs = {})
#   %mul_52 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %sum_16), kwargs = {})
#   %sub_6 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_49, %mul_52), kwargs = {})
#   %mul_53 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_4), kwargs = {})
#   %mul_54 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_107, %mul_16), kwargs = {})
#   %sum_17 : Tensor "f32[384][1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_54, [0, 1]), kwargs = {})
#   %convert_element_type_110 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_53, torch.bfloat16), kwargs = {})
#   %add_19 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %convert_element_type_110), kwargs = {})
#   return %sum_16,%add_19
triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_slice_backward_view_8 = async_compile.triton('triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_slice_backward_view_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_slice_backward_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 0, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_slice_backward_view_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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
        tmp22 = tl.load(in_out_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
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
        tmp23 = tl.full([1, 1], 0.0026041666666666665, tl.float32)
        tmp24 = tmp3 * tmp23
        tmp25 = tmp24 * tmp21
        tmp26 = tmp16 - tmp25
        tmp27 = tmp26 * tmp2
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp22 + tmp28
        tmp30 = tmp12 * tmp3
        tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp29, r0_mask)
        tmp31 = tl.sum(tmp30, 0)
        tmp32 = accum0 + tmp31
        accum0 = tmp32
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/wp/cwpydktugbby65erzengpqk32cjh3eskjwgn55ed3ygnq3s4yegt.py
# Topologically Sorted Source Nodes: [view_99, convert_element_type_201], Original ATen: [aten.view, aten._to_copy]
# Source node to ATen node mapping:
#   convert_element_type_201 => convert_element_type_201
#   view_99 => view_99
# Graph fragment:
#   %mm_41 : Tensor "bf16[131072, 384][384, 1]cuda:5" = PlaceHolder[target=mm_41]
#   %view_99 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_41, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_201 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:5"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_99, torch.float32), kwargs = {})
#   return %convert_element_type_201
triton_poi_fused__to_copy_view_9 = async_compile.triton('triton_poi_fused__to_copy_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 503316480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_view_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
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
        primals_3, primals_4, primals_9, primals_10, primals_11, primals_12, primals_13, primals_16, primals_17, primals_18, primals_19, primals_20, primals_23, primals_24, primals_25, primals_26, view, mm, rsqrt, view_2, view_4, clone, select_1, select_3, permute_5, permute_6, getitem_2, getitem_3, getitem_8, getitem_9, pow_3, div, sum_2, view_7, mm_2, view_10, add_1, rsqrt_1, view_13, getitem_11, getitem_12, add_3, rsqrt_2, view_16, view_18, clone_2, permute_17, permute_18, getitem_15, getitem_16, getitem_21, getitem_22, pow_7, div_1, sum_4, view_21, mm_6, view_24, add_5, rsqrt_3, view_27, getitem_24, getitem_25, add_7, rsqrt_4, view_30, view_32, clone_4, permute_29, permute_30, getitem_28, getitem_29, getitem_34, getitem_35, pow_11, div_2, sum_6, view_35, mm_10, view_38, add_9, rsqrt_5, view_41, getitem_37, getitem_38, add_11, rsqrt_6, permute_42, permute_46, permute_62, permute_66, permute_82, permute_86, permute_99, tangents_1 = args
        args.clear()
        assert_size_stride(primals_3, (384, ), (1, ))
        assert_size_stride(primals_4, (768, 384), (384, 1))
        assert_size_stride(primals_9, (384, ), (1, ))
        assert_size_stride(primals_10, (1536, 384), (384, 1))
        assert_size_stride(primals_11, (384, 1536), (1536, 1))
        assert_size_stride(primals_12, (384, ), (1, ))
        assert_size_stride(primals_13, (768, 384), (384, 1))
        assert_size_stride(primals_16, (384, ), (1, ))
        assert_size_stride(primals_17, (1536, 384), (384, 1))
        assert_size_stride(primals_18, (384, 1536), (1536, 1))
        assert_size_stride(primals_19, (384, ), (1, ))
        assert_size_stride(primals_20, (768, 384), (384, 1))
        assert_size_stride(primals_23, (384, ), (1, ))
        assert_size_stride(primals_24, (1536, 384), (384, 1))
        assert_size_stride(primals_25, (384, 1536), (1536, 1))
        assert_size_stride(primals_26, (384, ), (1, ))
        assert_size_stride(view, (131072, 384), (384, 1))
        assert_size_stride(mm, (131072, 384), (384, 1))
        assert_size_stride(rsqrt, (128, 1024, 1), (1024, 1, 1))
        assert_size_stride(view_2, (131072, 384), (384, 1))
        assert_size_stride(view_4, (128, 1024, 16, 48), (786432, 768, 48, 1))
        assert_size_stride(clone, (128, 1024, 4, 48), (196608, 192, 48, 1))
        assert_size_stride(select_1, (1024, 24), (24, 1))
        assert_size_stride(select_3, (1024, 24), (24, 1))
        assert_size_stride(permute_5, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(permute_6, (128, 4, 1024, 48), (196608, 48, 192, 1))
        assert_size_stride(getitem_2, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(getitem_3, (128, 8, 1024), (8192, 1024, 1))
        assert_size_stride(getitem_8, (2, ), (1, ))
        assert_size_stride(getitem_9, (), ())
        assert_size_stride(pow_3, (128, 1024, 4, 1), (4096, 4, 1, 1))
        assert_size_stride(div, (128, 1024, 4, 48), (196608, 192, 48, 1))
        assert_size_stride(sum_2, (128, 1024, 4, 2, 1), (8192, 8, 2, 1, 1))
        assert_size_stride(view_7, (131072, 12), (12, 1))
        assert_size_stride(mm_2, (131072, 8), (8, 1))
        assert_size_stride(view_10, (131072, 384), (384, 1))
        assert_size_stride(add_1, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(rsqrt_1, (128, 1024, 1), (1024, 1, 1))
        assert_size_stride(view_13, (131072, 384), (384, 1))
        assert_size_stride(getitem_11, (131072, 1536), (1536, 1))
        assert_size_stride(getitem_12, (131072, 1536), (1536, 1))
        assert_size_stride(add_3, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(rsqrt_2, (128, 1024, 1), (1024, 1, 1))
        assert_size_stride(view_16, (131072, 384), (384, 1))
        assert_size_stride(view_18, (128, 1024, 16, 48), (786432, 768, 48, 1))
        assert_size_stride(clone_2, (128, 1024, 4, 48), (196608, 192, 48, 1))
        assert_size_stride(permute_17, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(permute_18, (128, 4, 1024, 48), (196608, 48, 192, 1))
        assert_size_stride(getitem_15, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(getitem_16, (128, 8, 1024), (8192, 1024, 1))
        assert_size_stride(getitem_21, (2, ), (1, ))
        assert_size_stride(getitem_22, (), ())
        assert_size_stride(pow_7, (128, 1024, 4, 1), (4096, 4, 1, 1))
        assert_size_stride(div_1, (128, 1024, 4, 48), (196608, 192, 48, 1))
        assert_size_stride(sum_4, (128, 1024, 4, 2, 1), (8192, 8, 2, 1, 1))
        assert_size_stride(view_21, (131072, 12), (12, 1))
        assert_size_stride(mm_6, (131072, 8), (8, 1))
        assert_size_stride(view_24, (131072, 384), (384, 1))
        assert_size_stride(add_5, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(rsqrt_3, (128, 1024, 1), (1024, 1, 1))
        assert_size_stride(view_27, (131072, 384), (384, 1))
        assert_size_stride(getitem_24, (131072, 1536), (1536, 1))
        assert_size_stride(getitem_25, (131072, 1536), (1536, 1))
        assert_size_stride(add_7, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(rsqrt_4, (128, 1024, 1), (1024, 1, 1))
        assert_size_stride(view_30, (131072, 384), (384, 1))
        assert_size_stride(view_32, (128, 1024, 16, 48), (786432, 768, 48, 1))
        assert_size_stride(clone_4, (128, 1024, 4, 48), (196608, 192, 48, 1))
        assert_size_stride(permute_29, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(permute_30, (128, 4, 1024, 48), (196608, 48, 192, 1))
        assert_size_stride(getitem_28, (128, 8, 1024, 48), (393216, 48, 384, 1))
        assert_size_stride(getitem_29, (128, 8, 1024), (8192, 1024, 1))
        assert_size_stride(getitem_34, (2, ), (1, ))
        assert_size_stride(getitem_35, (), ())
        assert_size_stride(pow_11, (128, 1024, 4, 1), (4096, 4, 1, 1))
        assert_size_stride(div_2, (128, 1024, 4, 48), (196608, 192, 48, 1))
        assert_size_stride(sum_6, (128, 1024, 4, 2, 1), (8192, 8, 2, 1, 1))
        assert_size_stride(view_35, (131072, 12), (12, 1))
        assert_size_stride(mm_10, (131072, 8), (8, 1))
        assert_size_stride(view_38, (131072, 384), (384, 1))
        assert_size_stride(add_9, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(rsqrt_5, (128, 1024, 1), (1024, 1, 1))
        assert_size_stride(view_41, (131072, 384), (384, 1))
        assert_size_stride(getitem_37, (131072, 1536), (1536, 1))
        assert_size_stride(getitem_38, (131072, 1536), (1536, 1))
        assert_size_stride(add_11, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(rsqrt_6, (128, 1024, 1), (1024, 1, 1))
        assert_size_stride(permute_42, (384, 384), (384, 1))
        assert_size_stride(permute_46, (8, 12), (12, 1))
        assert_size_stride(permute_62, (384, 384), (384, 1))
        assert_size_stride(permute_66, (8, 12), (12, 1))
        assert_size_stride(permute_82, (384, 384), (384, 1))
        assert_size_stride(permute_86, (8, 12), (12, 1))
        assert_size_stride(permute_99, (384, 384), (384, 1))
        assert_size_stride(tangents_1, (131072, 384), (384, 1))
        with torch.cuda._DeviceGuard(5):
            torch.cuda.set_device(5)
            buf3 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_46, convert_element_type_65, to_22, convert_element_type_67, mul_26, rms_norm_6, mul_28, sum_7, div_3, mul_29, sub_3, mul_30, mul_31, sum_8, convert_element_type_68], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten._fused_rms_norm]
            workspace_0 = empty_strided_cuda((786432, ), (1, ), torch.float32)
            stream5 = get_raw_stream(5)
            triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_view_0.run(add_11, rsqrt_6, tangents_1, primals_26, buf3, workspace_0, 131072, 384, stream=stream5)
            buf2 = workspace_0[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del add_11
            del primals_26
            del rsqrt_6
            del tangents_1
            buf4 = empty_strided_cuda((384, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [reshape, getattr_1, d_down_w], Original ATen: [aten.view, aten.permute, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (384, 131072), (1, 384), 0), getitem_38, out=buf4)
            del getitem_38
            buf5 = empty_strided_cuda((131072, 1536), (1536, 1), torch.bfloat16)
            buf6 = empty_strided_cuda((1536, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [reshape, getattr_6, contiguous_2, triton_kernel_wrapper_mutation], Original ATen: [aten.view, aten.permute, aten.clone]
            stream5 = get_raw_stream(5)
            triton_poi_fused_clone_permute_view_1.run(primals_25, buf6, 1536, 384, stream=stream5)
            del primals_25
            # Topologically Sorted Source Nodes: [reshape, getattr_6, contiguous_2, triton_kernel_wrapper_mutation], Original ATen: [aten.view, aten.permute, aten.clone]
            stream5 = get_raw_stream(5)
            _leaky_relu_sq_matmul_kernel_0.run(reinterpret_tensor(buf3, (131072, 384), (384, 1), 0), buf6, buf5, getitem_37, 24576, 1, 1, stream=stream5)
            del getitem_37
            buf8 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [d_up_w], Original ATen: [aten.permute, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf5, (1536, 131072), (1, 1536), 0), view_41, out=buf8)
            del view_41
            buf9 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dx], Original ATen: [aten.mm]
            extern_kernels.mm(buf5, primals_24, out=buf9)
            del primals_24
            buf13 = buf3; del buf3  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [reshape_as, convert_element_type_77, to_19, convert_element_type_79, mul_32, rms_norm_5, mul_34, sum_9, div_4, mul_35, sub_4, mul_36, mul_37, sum_10, convert_element_type_80, add_13], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten._fused_rms_norm, aten.add]
            workspace_1 = workspace_0; del workspace_0  # reuse
            stream5 = get_raw_stream(5)
            triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_view_2.run(buf13, add_9, rsqrt_5, buf9, primals_23, workspace_1, 131072, 384, stream=stream5)
            buf12 = workspace_1[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del add_9
            del primals_23
            del rsqrt_5
            buf14 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_49, permute_40, mm_16], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf13, (384, 131072), (1, 384), 0), view_38, out=buf14)
            del view_38
            buf15 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [view_49, mm_17], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf13, (131072, 384), (384, 1), 0), permute_42, out=buf15)
            del permute_42
            buf22 = empty_strided_cuda((128, 1024, 4, 2, 1), (8192, 8, 2, 1, 1048576), torch.float32)
            buf25 = empty_strided_cuda((128, 8, 1024, 48), (393216, 48, 384, 1), torch.bfloat16)
            buf17 = empty_strided_cuda((128, 1024, 8), (8192, 8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_50, convert_element_type_87, view_51, transpose_11, view_5, unsqueeze_2, mul_9, sub_2, reshape_6, mul_38, linear_8, mul_10, sigmoid_2, getitem_16, mul_39, sum_11, convert_element_type_90, squeeze, convert_element_type_91, convert_element_type_92, sub_5, mul_40, mul_41, convert_element_type_93, mul_42, view_54, neg, convert_element_type_99, mul_44, sum_12, expand_3, mul_46, convert_element_type_100, add_14, view_55, permute_48, transpose_10, _scaled_dot_product_flash_attention_backward], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.unsqueeze, aten.mul, aten.sub, aten._unsafe_view, aten.sigmoid, aten.sum, aten.squeeze, aten.sigmoid_backward, aten.neg, aten.expand, aten.add, aten._scaled_dot_product_flash_attention_backward]
            stream5 = get_raw_stream(5)
            triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_add_expand_mul_neg_sigmoid_sigmoid_backward_squeeze_sub_sum_transpose_unsqueeze_view_3.run(buf15, mm_10, div_2, getitem_28, sum_6, buf22, buf25, buf17, 1048576, 48, stream=stream5)
            buf18 = empty_strided_cuda((131072, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [constant_pad_nd_default_2], Original ATen: [aten.mm]
            stream5 = get_raw_stream(5)
            triton_poi_fused_mm_4.run(view_35, buf18, 2097152, stream=stream5)
            del view_35
            buf19 = empty_strided_cuda((8, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_8, mul_10, sigmoid_2, convert_element_type_90, squeeze, convert_element_type_91, convert_element_type_92, sub_5, mul_40, mul_41, convert_element_type_93, mul_42, view_52, permute_44, constant_pad_nd_default_2, mm_default_2], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf17, (8, 131072), (1, 8), 0), buf18, out=buf19)
            buf20 = empty_strided_cuda((131072, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_8, mul_10, sigmoid_2, convert_element_type_90, squeeze, convert_element_type_91, convert_element_type_92, sub_5, mul_40, mul_41, convert_element_type_93, mul_42, view_52, mm_19], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf17, (131072, 8), (8, 1), 0), permute_46, out=buf20)
            del permute_46
            buf21 = empty_strided_cuda((8, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [slice_tensor_2, convert_element_type_98], Original ATen: [aten.mm, aten._to_copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused__to_copy_mm_5.run(buf19, buf21, 96, stream=stream5)
            buf23 = empty_strided_cuda((128, 1024, 4, 1, 48), (196608, 192, 48, 48, 1), torch.float32)
            buf24 = empty_strided_cuda((128, 1024, 4, 1), (4096, 4, 1, 524288), torch.float32)
            # Topologically Sorted Source Nodes: [view_50, convert_element_type_87, view_51, transpose_11, view_5, linear_8, mul_10, sigmoid_2, getitem_16, mul_39, view_54, neg, mul_43, sum_13, expand_3, mul_45, sum_14, add_15, squeeze_1, normalize_2, div_6, neg_1, mul_47, sum_15], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.neg, aten.sum, aten.expand, aten.add, aten.squeeze, aten.clamp_min, aten.div]
            stream5 = get_raw_stream(5)
            triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_expand_mul_neg_sigmoid_squeeze_sum_transpose_unsqueeze_view_6.run(buf15, mm_10, sum_6, buf22, getitem_28, div_2, pow_11, buf23, buf24, 524288, 48, stream=stream5)
            del buf15
            del div_2
            del mm_10
            del sum_6
            # Topologically Sorted Source Nodes: [view_50, convert_element_type_87, view_51, unsqueeze_2, linear_8, mul_10, sigmoid_2, getitem_16, mul_39, view_54, convert_element_type_99, expand_3, mul_46, convert_element_type_100, add_14, view_55, permute_48, transpose_10, _scaled_dot_product_flash_attention_backward], Original ATen: [aten.view, aten._to_copy, aten.unsqueeze, aten._unsafe_view, aten.mul, aten.sigmoid, aten.expand, aten.add, aten.transpose, aten._scaled_dot_product_flash_attention_backward]
            buf26 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(buf25, permute_29, permute_30, reinterpret_tensor(clone_4, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), getitem_28, getitem_29, None, None, 1024, 1024, 0.0, True, getitem_34, getitem_35, scale=0.14433756729740646)
            del getitem_28
            del getitem_29
            del getitem_34
            del getitem_35
            del permute_29
            del permute_30
            buf27 = buf26[0]
            assert_size_stride(buf27, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf27, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf28 = buf26[1]
            assert_size_stride(buf28, (128, 4, 1024, 48), (196608, 48, 192, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf28, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf29 = buf26[2]
            assert_size_stride(buf29, (128, 4, 1024, 48), (196608, 48, 192, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf29, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            del buf26
            buf31 = empty_strided_cuda((128, 1024, 16, 48), (786432, 768, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [squeeze_1, normalize_2, div_7, convert_element_type_101, full_default_1, ge, where, div_8, eq, where_1, mul_48, convert_element_type_102, add_16, permute_49, add_17, getitem, copy_], Original ATen: [aten.squeeze, aten.clamp_min, aten.expand, aten.div, aten._to_copy, aten.scalar_tensor, aten.ge, aten.where, aten.eq, aten.masked_fill, aten.mul, aten.add, aten.transpose, aten.slice, aten.copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused__to_copy_add_clamp_min_copy_div_eq_expand_ge_masked_fill_mul_scalar_tensor_slice_squeeze_transpose_where_7.run(buf23, pow_11, buf24, clone_4, buf29, buf31, 100663296, stream=stream5)
            del buf29
            del clone_4
            del pow_11
            # Topologically Sorted Source Nodes: [permute_50, permute_51, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose]
            stream5 = get_raw_stream(5)
            _fused_qkv_postprocess_bwd_kernel_1.run(reinterpret_tensor(buf27, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), reinterpret_tensor(buf28, (128, 1024, 4, 48), (196608, 192, 48, 1), 0), view_32, buf31, select_1, select_3, 1572864, 1, 1, stream=stream5)
            del buf28
            del view_32
            buf33 = empty_strided_cuda((768, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_59, view_60, permute_53, mm_20], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf31, (768, 131072), (1, 768), 0), view_30, out=buf33)
            del view_30
            buf34 = reinterpret_tensor(buf27, (131072, 384), (384, 1), 0); del buf27  # reuse
            # Topologically Sorted Source Nodes: [view_59, view_60, linear_7, permute_55, mm_21], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf31, (131072, 768), (768, 1), 0), primals_20, out=buf34)
            del primals_20
            buf38 = buf13; del buf13  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_53, full_default, view_63, add_18, convert_element_type_107, to_15, convert_element_type_109, mul_49, rms_norm_4, mul_51, sum_16, div_9, mul_52, sub_6, mul_53, mul_54, sum_17, convert_element_type_110, add_19], Original ATen: [aten.view, aten.slice_backward, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten._fused_rms_norm]
            workspace_2 = workspace_1; del workspace_1  # reuse
            stream5 = get_raw_stream(5)
            triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_slice_backward_view_8.run(buf38, add_7, rsqrt_4, buf20, buf34, primals_19, workspace_2, 131072, 384, stream=stream5)
            buf37 = workspace_2[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del add_7
            del primals_19
            del rsqrt_4
            buf39 = empty_strided_cuda((384, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [reshape, getattr_1, d_down_w], Original ATen: [aten.view, aten.permute, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf38, (384, 131072), (1, 384), 0), getitem_25, out=buf39)
            del getitem_25
            buf40 = buf5; del buf5  # reuse
            buf41 = empty_strided_cuda((1536, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [reshape, getattr_4, contiguous_2, triton_kernel_wrapper_mutation], Original ATen: [aten.view, aten.permute, aten.clone]
            stream5 = get_raw_stream(5)
            triton_poi_fused_clone_permute_view_1.run(primals_18, buf41, 1536, 384, stream=stream5)
            del primals_18
            # Topologically Sorted Source Nodes: [reshape, getattr_4, contiguous_2, triton_kernel_wrapper_mutation], Original ATen: [aten.view, aten.permute, aten.clone]
            stream5 = get_raw_stream(5)
            _leaky_relu_sq_matmul_kernel_0.run(reinterpret_tensor(buf38, (131072, 384), (384, 1), 0), buf41, buf40, getitem_24, 24576, 1, 1, stream=stream5)
            del getitem_24
            buf43 = buf41; del buf41  # reuse
            # Topologically Sorted Source Nodes: [d_up_w], Original ATen: [aten.permute, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (1536, 131072), (1, 1536), 0), view_27, out=buf43)
            del view_27
            buf44 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [dx], Original ATen: [aten.mm]
            extern_kernels.mm(buf40, primals_17, out=buf44)
            del primals_17
            buf48 = buf38; del buf38  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [reshape_as, convert_element_type_119, to_12, convert_element_type_121, mul_55, rms_norm_3, mul_57, sum_18, div_10, mul_58, sub_7, mul_59, mul_60, sum_19, convert_element_type_122, add_20], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten._fused_rms_norm, aten.add]
            workspace_3 = workspace_2; del workspace_2  # reuse
            stream5 = get_raw_stream(5)
            triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_view_2.run(buf48, add_5, rsqrt_3, buf44, primals_16, workspace_3, 131072, 384, stream=stream5)
            buf47 = workspace_3[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del add_5
            del primals_16
            del rsqrt_3
            buf49 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_66, permute_60, mm_25], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf48, (384, 131072), (1, 384), 0), view_24, out=buf49)
            del view_24
            buf50 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [view_66, mm_26], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf48, (131072, 384), (384, 1), 0), permute_62, out=buf50)
            del permute_62
            buf57 = buf22; del buf22  # reuse
            buf60 = buf25; del buf25  # reuse
            buf52 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [view_67, convert_element_type_129, view_68, transpose_7, view_3, unsqueeze_1, mul_5, sub_1, reshape_3, mul_61, linear_5, mul_6, sigmoid_1, getitem_10, mul_62, sum_20, convert_element_type_132, squeeze_2, convert_element_type_133, convert_element_type_134, sub_8, mul_63, mul_64, convert_element_type_135, mul_65, view_71, neg_2, convert_element_type_141, mul_67, sum_21, expand_4, mul_69, convert_element_type_142, add_21, view_72, permute_68, transpose_6, _scaled_dot_product_flash_attention_backward_1], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.unsqueeze, aten.mul, aten.sub, aten._unsafe_view, aten.sigmoid, aten.sum, aten.squeeze, aten.sigmoid_backward, aten.neg, aten.expand, aten.add, aten._scaled_dot_product_flash_attention_backward]
            stream5 = get_raw_stream(5)
            triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_add_expand_mul_neg_sigmoid_sigmoid_backward_squeeze_sub_sum_transpose_unsqueeze_view_3.run(buf50, mm_6, div_1, getitem_15, sum_4, buf57, buf60, buf52, 1048576, 48, stream=stream5)
            buf53 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [constant_pad_nd_default_1], Original ATen: [aten.mm]
            stream5 = get_raw_stream(5)
            triton_poi_fused_mm_4.run(view_21, buf53, 2097152, stream=stream5)
            del view_21
            buf54 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [linear_5, mul_6, sigmoid_1, convert_element_type_132, squeeze_2, convert_element_type_133, convert_element_type_134, sub_8, mul_63, mul_64, convert_element_type_135, mul_65, view_69, permute_64, constant_pad_nd_default_1, mm_default_1], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf52, (8, 131072), (1, 8), 0), buf53, out=buf54)
            buf55 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [linear_5, mul_6, sigmoid_1, convert_element_type_132, squeeze_2, convert_element_type_133, convert_element_type_134, sub_8, mul_63, mul_64, convert_element_type_135, mul_65, view_69, mm_28], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf52, (131072, 8), (8, 1), 0), permute_66, out=buf55)
            del permute_66
            buf56 = empty_strided_cuda((8, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [slice_tensor_1, convert_element_type_140], Original ATen: [aten.mm, aten._to_copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused__to_copy_mm_5.run(buf54, buf56, 96, stream=stream5)
            buf58 = buf23; del buf23  # reuse
            buf59 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [view_67, convert_element_type_129, view_68, transpose_7, view_3, linear_5, mul_6, sigmoid_1, getitem_10, mul_62, view_71, neg_2, mul_66, sum_22, expand_4, mul_68, sum_23, add_22, squeeze_3, normalize_1, div_12, neg_3, mul_70, sum_24], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.neg, aten.sum, aten.expand, aten.add, aten.squeeze, aten.clamp_min, aten.div]
            stream5 = get_raw_stream(5)
            triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_expand_mul_neg_sigmoid_squeeze_sum_transpose_unsqueeze_view_6.run(buf50, mm_6, sum_4, buf57, getitem_15, div_1, pow_7, buf58, buf59, 524288, 48, stream=stream5)
            del buf50
            del div_1
            del mm_6
            del sum_4
            # Topologically Sorted Source Nodes: [view_67, convert_element_type_129, view_68, unsqueeze_1, linear_5, mul_6, sigmoid_1, getitem_10, mul_62, view_71, convert_element_type_141, expand_4, mul_69, convert_element_type_142, add_21, view_72, permute_68, transpose_6, _scaled_dot_product_flash_attention_backward_1], Original ATen: [aten.view, aten._to_copy, aten.unsqueeze, aten._unsafe_view, aten.mul, aten.sigmoid, aten.expand, aten.add, aten.transpose, aten._scaled_dot_product_flash_attention_backward]
            buf61 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(buf60, permute_17, permute_18, reinterpret_tensor(clone_2, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), getitem_15, getitem_16, None, None, 1024, 1024, 0.0, True, getitem_21, getitem_22, scale=0.14433756729740646)
            del getitem_15
            del getitem_16
            del getitem_21
            del getitem_22
            del permute_17
            del permute_18
            buf62 = buf61[0]
            assert_size_stride(buf62, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf62, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf63 = buf61[1]
            assert_size_stride(buf63, (128, 4, 1024, 48), (196608, 48, 192, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf63, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf64 = buf61[2]
            assert_size_stride(buf64, (128, 4, 1024, 48), (196608, 48, 192, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf64, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            del buf61
            buf66 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [full_default_1, squeeze_3, normalize_1, div_13, convert_element_type_143, ge_1, where_2, div_14, eq_1, where_3, mul_71, convert_element_type_144, add_23, permute_69, add_24, getitem, copy_], Original ATen: [aten.scalar_tensor, aten.squeeze, aten.clamp_min, aten.expand, aten.div, aten._to_copy, aten.ge, aten.where, aten.eq, aten.masked_fill, aten.mul, aten.add, aten.transpose, aten.slice, aten.copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused__to_copy_add_clamp_min_copy_div_eq_expand_ge_masked_fill_mul_scalar_tensor_slice_squeeze_transpose_where_7.run(buf58, pow_7, buf59, clone_2, buf64, buf66, 100663296, stream=stream5)
            del buf64
            del clone_2
            del pow_7
            # Topologically Sorted Source Nodes: [permute_70, permute_71, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose]
            stream5 = get_raw_stream(5)
            _fused_qkv_postprocess_bwd_kernel_1.run(reinterpret_tensor(buf62, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), reinterpret_tensor(buf63, (128, 1024, 4, 48), (196608, 192, 48, 1), 0), view_18, buf66, select_1, select_3, 1572864, 1, 1, stream=stream5)
            del buf63
            del view_18
            buf68 = empty_strided_cuda((768, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_76, view_77, permute_73, mm_29], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf66, (768, 131072), (1, 768), 0), view_16, out=buf68)
            del view_16
            buf69 = reinterpret_tensor(buf62, (131072, 384), (384, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [view_76, view_77, linear_4, permute_75, mm_30], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf66, (131072, 768), (768, 1), 0), primals_13, out=buf69)
            del primals_13
            buf73 = buf48; del buf48  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [full_default, view_70, view_80, add_25, convert_element_type_149, to_8, convert_element_type_151, mul_72, rms_norm_2, mul_74, sum_25, div_15, mul_75, sub_9, mul_76, mul_77, sum_26, convert_element_type_152, add_26], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten._fused_rms_norm]
            workspace_4 = workspace_3; del workspace_3  # reuse
            stream5 = get_raw_stream(5)
            triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_slice_backward_view_8.run(buf73, add_3, rsqrt_2, buf55, buf69, primals_12, workspace_4, 131072, 384, stream=stream5)
            buf72 = workspace_4[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del add_3
            del primals_12
            del rsqrt_2
            buf74 = empty_strided_cuda((384, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [reshape, getattr_1, d_down_w], Original ATen: [aten.view, aten.permute, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf73, (384, 131072), (1, 384), 0), getitem_12, out=buf74)
            del getitem_12
            buf75 = buf40; del buf40  # reuse
            buf76 = empty_strided_cuda((1536, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [reshape, getattr_2, contiguous_2, triton_kernel_wrapper_mutation], Original ATen: [aten.view, aten.permute, aten.clone]
            stream5 = get_raw_stream(5)
            triton_poi_fused_clone_permute_view_1.run(primals_11, buf76, 1536, 384, stream=stream5)
            del primals_11
            # Topologically Sorted Source Nodes: [reshape, getattr_2, contiguous_2, triton_kernel_wrapper_mutation], Original ATen: [aten.view, aten.permute, aten.clone]
            stream5 = get_raw_stream(5)
            _leaky_relu_sq_matmul_kernel_0.run(reinterpret_tensor(buf73, (131072, 384), (384, 1), 0), buf76, buf75, getitem_11, 24576, 1, 1, stream=stream5)
            del getitem_11
            buf78 = buf76; del buf76  # reuse
            # Topologically Sorted Source Nodes: [d_up_w], Original ATen: [aten.permute, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf75, (1536, 131072), (1, 1536), 0), view_13, out=buf78)
            del view_13
            buf79 = buf69; del buf69  # reuse
            # Topologically Sorted Source Nodes: [dx], Original ATen: [aten.mm]
            extern_kernels.mm(buf75, primals_10, out=buf79)
            del buf75
            del primals_10
            buf83 = buf73; del buf73  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [reshape_as, convert_element_type_161, to_5, convert_element_type_163, mul_78, rms_norm_1, mul_80, sum_27, div_16, mul_81, sub_10, mul_82, mul_83, sum_28, convert_element_type_164, add_27], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten._fused_rms_norm, aten.add]
            workspace_5 = workspace_4; del workspace_4  # reuse
            stream5 = get_raw_stream(5)
            triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_view_2.run(buf83, add_1, rsqrt_1, buf79, primals_9, workspace_5, 131072, 384, stream=stream5)
            buf82 = workspace_5[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del add_1
            del primals_9
            del rsqrt_1
            buf84 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_83, permute_80, mm_34], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf83, (384, 131072), (1, 384), 0), view_10, out=buf84)
            del view_10
            buf85 = buf79; del buf79  # reuse
            # Topologically Sorted Source Nodes: [view_83, mm_35], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf83, (131072, 384), (384, 1), 0), permute_82, out=buf85)
            del permute_82
            buf92 = buf57; del buf57  # reuse
            buf95 = buf60; del buf60  # reuse
            buf87 = buf52; del buf52  # reuse
            # Topologically Sorted Source Nodes: [view_84, convert_element_type_171, view_85, transpose_3, view_1, unsqueeze, mul_1, sub, reshape, mul_84, linear_2, mul_2, sigmoid, getitem_4, mul_85, sum_29, convert_element_type_174, squeeze_4, convert_element_type_175, convert_element_type_176, sub_11, mul_86, mul_87, convert_element_type_177, mul_88, view_88, neg_4, convert_element_type_183, mul_90, sum_30, expand_5, mul_92, convert_element_type_184, add_28, view_89, permute_88, transpose_2, _scaled_dot_product_flash_attention_backward_2], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.unsqueeze, aten.mul, aten.sub, aten._unsafe_view, aten.sigmoid, aten.sum, aten.squeeze, aten.sigmoid_backward, aten.neg, aten.expand, aten.add, aten._scaled_dot_product_flash_attention_backward]
            stream5 = get_raw_stream(5)
            triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy__unsafe_view_add_expand_mul_neg_sigmoid_sigmoid_backward_squeeze_sub_sum_transpose_unsqueeze_view_3.run(buf85, mm_2, div, getitem_2, sum_2, buf92, buf95, buf87, 1048576, 48, stream=stream5)
            buf88 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [constant_pad_nd_default], Original ATen: [aten.mm]
            stream5 = get_raw_stream(5)
            triton_poi_fused_mm_4.run(view_7, buf88, 2097152, stream=stream5)
            del view_7
            buf89 = buf54; del buf54  # reuse
            # Topologically Sorted Source Nodes: [linear_2, mul_2, sigmoid, convert_element_type_174, squeeze_4, convert_element_type_175, convert_element_type_176, sub_11, mul_86, mul_87, convert_element_type_177, mul_88, view_86, permute_84, constant_pad_nd_default, mm_default], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf87, (8, 131072), (1, 8), 0), buf88, out=buf89)
            del buf88
            buf90 = buf55; del buf55  # reuse
            # Topologically Sorted Source Nodes: [linear_2, mul_2, sigmoid, convert_element_type_174, squeeze_4, convert_element_type_175, convert_element_type_176, sub_11, mul_86, mul_87, convert_element_type_177, mul_88, view_86, mm_37], Original ATen: [aten._unsafe_view, aten.mul, aten.sigmoid, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf87, (131072, 8), (8, 1), 0), permute_86, out=buf90)
            del buf87
            del permute_86
            buf91 = empty_strided_cuda((8, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [slice_tensor, convert_element_type_182], Original ATen: [aten.mm, aten._to_copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused__to_copy_mm_5.run(buf89, buf91, 96, stream=stream5)
            del buf89
            buf93 = buf58; del buf58  # reuse
            buf94 = buf59; del buf59  # reuse
            # Topologically Sorted Source Nodes: [view_84, convert_element_type_171, view_85, transpose_3, view_1, linear_2, mul_2, sigmoid, getitem_4, mul_85, view_88, neg_4, mul_89, sum_31, expand_5, mul_91, sum_32, add_29, squeeze_5, normalize, div_18, neg_5, mul_93, sum_33], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.neg, aten.sum, aten.expand, aten.add, aten.squeeze, aten.clamp_min, aten.div]
            stream5 = get_raw_stream(5)
            triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_expand_mul_neg_sigmoid_squeeze_sum_transpose_unsqueeze_view_6.run(buf85, mm_2, sum_2, buf92, getitem_2, div, pow_3, buf93, buf94, 524288, 48, stream=stream5)
            del buf85
            del buf92
            del div
            del mm_2
            del sum_2
            # Topologically Sorted Source Nodes: [view_84, convert_element_type_171, view_85, unsqueeze, linear_2, mul_2, sigmoid, getitem_4, mul_85, view_88, convert_element_type_183, expand_5, mul_92, convert_element_type_184, add_28, view_89, permute_88, transpose_2, _scaled_dot_product_flash_attention_backward_2], Original ATen: [aten.view, aten._to_copy, aten.unsqueeze, aten._unsafe_view, aten.mul, aten.sigmoid, aten.expand, aten.add, aten.transpose, aten._scaled_dot_product_flash_attention_backward]
            buf96 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(buf95, permute_5, permute_6, reinterpret_tensor(clone, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), getitem_2, getitem_3, None, None, 1024, 1024, 0.0, True, getitem_8, getitem_9, scale=0.14433756729740646)
            del buf95
            del getitem_2
            del getitem_3
            del getitem_8
            del getitem_9
            del permute_5
            del permute_6
            buf97 = buf96[0]
            assert_size_stride(buf97, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf97, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf98 = buf96[1]
            assert_size_stride(buf98, (128, 4, 1024, 48), (196608, 48, 192, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf98, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            buf99 = buf96[2]
            assert_size_stride(buf99, (128, 4, 1024, 48), (196608, 48, 192, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            assert_alignment(buf99, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
            del buf96
            buf101 = buf66; del buf66  # reuse
            # Topologically Sorted Source Nodes: [full_default_1, squeeze_5, normalize, div_19, convert_element_type_185, ge_2, where_4, div_20, eq_2, where_5, mul_94, convert_element_type_186, add_30, permute_89, add_31, getitem, copy_], Original ATen: [aten.scalar_tensor, aten.squeeze, aten.clamp_min, aten.expand, aten.div, aten._to_copy, aten.ge, aten.where, aten.eq, aten.masked_fill, aten.mul, aten.add, aten.transpose, aten.slice, aten.copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused__to_copy_add_clamp_min_copy_div_eq_expand_ge_masked_fill_mul_scalar_tensor_slice_squeeze_transpose_where_7.run(buf93, pow_3, buf94, clone, buf99, buf101, 100663296, stream=stream5)
            del buf93
            del buf94
            del buf99
            del clone
            del pow_3
            # Topologically Sorted Source Nodes: [permute_90, permute_91, triton_kernel_wrapper_mutation], Original ATen: [aten.transpose]
            stream5 = get_raw_stream(5)
            _fused_qkv_postprocess_bwd_kernel_1.run(reinterpret_tensor(buf97, (128, 1024, 8, 48), (393216, 384, 48, 1), 0), reinterpret_tensor(buf98, (128, 1024, 4, 48), (196608, 192, 48, 1), 0), view_4, buf101, select_1, select_3, 1572864, 1, 1, stream=stream5)
            del buf98
            del select_1
            del select_3
            del view_4
            buf103 = empty_strided_cuda((768, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_93, view_94, permute_93, mm_38], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf101, (768, 131072), (1, 768), 0), view_2, out=buf103)
            del view_2
            buf104 = reinterpret_tensor(buf97, (131072, 384), (384, 1), 0); del buf97  # reuse
            # Topologically Sorted Source Nodes: [view_93, view_94, linear_1, permute_95, mm_39], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf101, (131072, 768), (768, 1), 0), primals_4, out=buf104)
            del buf101
            del primals_4
            buf108 = buf83; del buf83  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [full_default, view_87, view_97, add_32, convert_element_type_191, to_1, convert_element_type_193, mul_95, linear, rms_norm, mul_97, sum_34, div_21, mul_98, sub_12, mul_99, mul_100, sum_35, convert_element_type_194, add_33], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten._unsafe_view, aten._fused_rms_norm]
            workspace_6 = workspace_5; del workspace_5  # reuse
            stream5 = get_raw_stream(5)
            triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_slice_backward_view_8.run(buf108, mm, rsqrt, buf90, buf104, primals_3, workspace_6, 131072, 384, stream=stream5)
            buf107 = workspace_6[0 * 2048 * 384 : (0 + 1) * 2048 * 384].view(2048, 384).sum(dim=0)
            del workspace_6
            del buf90
            del mm
            del primals_3
            del rsqrt
            buf109 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [full_default, view_87, view_97, add_32, convert_element_type_191, to_1, convert_element_type_193, mul_95, linear, rms_norm, div_21, mul_98, sub_12, mul_99, convert_element_type_194, add_33, view_98, permute_97, mm_40], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten._unsafe_view, aten._fused_rms_norm, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf108, (384, 131072), (1, 384), 0), view, out=buf109)
            del view
            buf110 = buf104; del buf104  # reuse
            # Topologically Sorted Source Nodes: [full_default, view_87, view_97, add_32, convert_element_type_191, to_1, convert_element_type_193, mul_95, linear, rms_norm, div_21, mul_98, sub_12, mul_99, convert_element_type_194, add_33, view_98, mm_41], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten._unsafe_view, aten._fused_rms_norm, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf108, (131072, 384), (384, 1), 0), permute_99, out=buf110)
            del buf108
            del permute_99
            buf111 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_99, convert_element_type_201], Original ATen: [aten.view, aten._to_copy]
            stream5 = get_raw_stream(5)
            triton_poi_fused__to_copy_view_9.run(buf110, buf111, 50331648, stream=stream5)
            del buf110
        return (buf109, buf111, buf107, buf103, None, None, buf91, buf84, buf82, buf78, buf74, buf72, buf68, buf56, buf49, buf47, buf43, buf39, buf37, buf33, buf21, buf14, buf12, buf8, buf4, buf2, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_3 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_4 = rand_strided((768, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    primals_9 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_10 = rand_strided((1536, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    primals_11 = rand_strided((384, 1536), (1536, 1), device='cuda:5', dtype=torch.bfloat16)
    primals_12 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_13 = rand_strided((768, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    primals_16 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_17 = rand_strided((1536, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    primals_18 = rand_strided((384, 1536), (1536, 1), device='cuda:5', dtype=torch.bfloat16)
    primals_19 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_20 = rand_strided((768, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    primals_23 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    primals_24 = rand_strided((1536, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    primals_25 = rand_strided((384, 1536), (1536, 1), device='cuda:5', dtype=torch.bfloat16)
    primals_26 = rand_strided((384, ), (1, ), device='cuda:5', dtype=torch.float32)
    view = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    mm = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    rsqrt = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:5', dtype=torch.float32)
    view_2 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    view_4 = rand_strided((128, 1024, 16, 48), (786432, 768, 48, 1), device='cuda:5', dtype=torch.bfloat16)
    clone = rand_strided((128, 1024, 4, 48), (196608, 192, 48, 1), device='cuda:5', dtype=torch.bfloat16)
    select_1 = rand_strided((1024, 24), (24, 1), device='cuda:5', dtype=torch.bfloat16)
    select_3 = rand_strided((1024, 24), (24, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_5 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_6 = rand_strided((128, 4, 1024, 48), (196608, 48, 192, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_2 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_3 = rand_strided((128, 8, 1024), (8192, 1024, 1), device='cuda:5', dtype=torch.float32)
    getitem_8 = rand_strided((2, ), (1, ), device='cuda:5', dtype=torch.uint64)
    getitem_9 = rand_strided((), (), device='cuda:5', dtype=torch.uint64)
    pow_3 = rand_strided((128, 1024, 4, 1), (4096, 4, 1, 1), device='cuda:5', dtype=torch.float32)
    div = rand_strided((128, 1024, 4, 48), (196608, 192, 48, 1), device='cuda:5', dtype=torch.float32)
    sum_2 = rand_strided((128, 1024, 4, 2, 1), (8192, 8, 2, 1, 1), device='cuda:5', dtype=torch.float32)
    view_7 = rand_strided((131072, 12), (12, 1), device='cuda:5', dtype=torch.bfloat16)
    mm_2 = rand_strided((131072, 8), (8, 1), device='cuda:5', dtype=torch.bfloat16)
    view_10 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    add_1 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    rsqrt_1 = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:5', dtype=torch.float32)
    view_13 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_11 = rand_strided((131072, 1536), (1536, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_12 = rand_strided((131072, 1536), (1536, 1), device='cuda:5', dtype=torch.bfloat16)
    add_3 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    rsqrt_2 = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:5', dtype=torch.float32)
    view_16 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    view_18 = rand_strided((128, 1024, 16, 48), (786432, 768, 48, 1), device='cuda:5', dtype=torch.bfloat16)
    clone_2 = rand_strided((128, 1024, 4, 48), (196608, 192, 48, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_17 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_18 = rand_strided((128, 4, 1024, 48), (196608, 48, 192, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_15 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_16 = rand_strided((128, 8, 1024), (8192, 1024, 1), device='cuda:5', dtype=torch.float32)
    getitem_21 = rand_strided((2, ), (1, ), device='cuda:5', dtype=torch.uint64)
    getitem_22 = rand_strided((), (), device='cuda:5', dtype=torch.uint64)
    pow_7 = rand_strided((128, 1024, 4, 1), (4096, 4, 1, 1), device='cuda:5', dtype=torch.float32)
    div_1 = rand_strided((128, 1024, 4, 48), (196608, 192, 48, 1), device='cuda:5', dtype=torch.float32)
    sum_4 = rand_strided((128, 1024, 4, 2, 1), (8192, 8, 2, 1, 1), device='cuda:5', dtype=torch.float32)
    view_21 = rand_strided((131072, 12), (12, 1), device='cuda:5', dtype=torch.bfloat16)
    mm_6 = rand_strided((131072, 8), (8, 1), device='cuda:5', dtype=torch.bfloat16)
    view_24 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    add_5 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    rsqrt_3 = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:5', dtype=torch.float32)
    view_27 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_24 = rand_strided((131072, 1536), (1536, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_25 = rand_strided((131072, 1536), (1536, 1), device='cuda:5', dtype=torch.bfloat16)
    add_7 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    rsqrt_4 = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:5', dtype=torch.float32)
    view_30 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    view_32 = rand_strided((128, 1024, 16, 48), (786432, 768, 48, 1), device='cuda:5', dtype=torch.bfloat16)
    clone_4 = rand_strided((128, 1024, 4, 48), (196608, 192, 48, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_29 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_30 = rand_strided((128, 4, 1024, 48), (196608, 48, 192, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_28 = rand_strided((128, 8, 1024, 48), (393216, 48, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_29 = rand_strided((128, 8, 1024), (8192, 1024, 1), device='cuda:5', dtype=torch.float32)
    getitem_34 = rand_strided((2, ), (1, ), device='cuda:5', dtype=torch.uint64)
    getitem_35 = rand_strided((), (), device='cuda:5', dtype=torch.uint64)
    pow_11 = rand_strided((128, 1024, 4, 1), (4096, 4, 1, 1), device='cuda:5', dtype=torch.float32)
    div_2 = rand_strided((128, 1024, 4, 48), (196608, 192, 48, 1), device='cuda:5', dtype=torch.float32)
    sum_6 = rand_strided((128, 1024, 4, 2, 1), (8192, 8, 2, 1, 1), device='cuda:5', dtype=torch.float32)
    view_35 = rand_strided((131072, 12), (12, 1), device='cuda:5', dtype=torch.bfloat16)
    mm_10 = rand_strided((131072, 8), (8, 1), device='cuda:5', dtype=torch.bfloat16)
    view_38 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    add_9 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    rsqrt_5 = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:5', dtype=torch.float32)
    view_41 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_37 = rand_strided((131072, 1536), (1536, 1), device='cuda:5', dtype=torch.bfloat16)
    getitem_38 = rand_strided((131072, 1536), (1536, 1), device='cuda:5', dtype=torch.bfloat16)
    add_11 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:5', dtype=torch.bfloat16)
    rsqrt_6 = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:5', dtype=torch.float32)
    permute_42 = rand_strided((384, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_46 = rand_strided((8, 12), (12, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_62 = rand_strided((384, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_66 = rand_strided((8, 12), (12, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_82 = rand_strided((384, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_86 = rand_strided((8, 12), (12, 1), device='cuda:5', dtype=torch.bfloat16)
    permute_99 = rand_strided((384, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    tangents_1 = rand_strided((131072, 384), (384, 1), device='cuda:5', dtype=torch.bfloat16)
    return [primals_3, primals_4, primals_9, primals_10, primals_11, primals_12, primals_13, primals_16, primals_17, primals_18, primals_19, primals_20, primals_23, primals_24, primals_25, primals_26, view, mm, rsqrt, view_2, view_4, clone, select_1, select_3, permute_5, permute_6, getitem_2, getitem_3, getitem_8, getitem_9, pow_3, div, sum_2, view_7, mm_2, view_10, add_1, rsqrt_1, view_13, getitem_11, getitem_12, add_3, rsqrt_2, view_16, view_18, clone_2, permute_17, permute_18, getitem_15, getitem_16, getitem_21, getitem_22, pow_7, div_1, sum_4, view_21, mm_6, view_24, add_5, rsqrt_3, view_27, getitem_24, getitem_25, add_7, rsqrt_4, view_30, view_32, clone_4, permute_29, permute_30, getitem_28, getitem_29, getitem_34, getitem_35, pow_11, div_2, sum_6, view_35, mm_10, view_38, add_9, rsqrt_5, view_41, getitem_37, getitem_38, add_11, rsqrt_6, permute_42, permute_46, permute_62, permute_66, permute_82, permute_86, permute_99, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
