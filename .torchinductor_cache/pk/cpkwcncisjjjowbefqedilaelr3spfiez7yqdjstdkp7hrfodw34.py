# AOT ID: ['9_forward']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ee/ceermlqippjauth5nefvdinkyxcxfl4ql66voqnfvlmrjjivcb4f.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear => convert_element_type_default_14, permute
# Graph fragment:
#   %primals_1 : Tensor "bf16[384, 384][384, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %convert_element_type_default_14 : Tensor "bf16[384, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.bfloat16), kwargs = {})
#   %permute : Tensor "bf16[384, 384][1, 384]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_default_14, [1, 0]), kwargs = {})
#   return %permute
triton_poi_fused__to_copy_t_0 = async_compile.triton('triton_poi_fused__to_copy_t_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 884736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ct/ccthzojzt4big5xzmdpk62njjcjd27nvzxep64lnai7sc5nt2spx.py
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 402653184}},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/dh/cdhfhmulmqhdum2dwsqvyzs2dqnvllvrdrddbg6ygollwn4xidlq.py
# Topologically Sorted Source Nodes: [linear, to_1, rms_norm], Original ATen: [aten._unsafe_view, aten._to_copy, aten._fused_rms_norm]
# Source node to ATen node mapping:
#   linear => view_1
#   rms_norm => add, convert_element_type_6, convert_element_type_7, mean, mul, mul_1, pow_1, rsqrt
#   to_1 => convert_element_type_5
# Graph fragment:
#   %mm : Tensor "bf16[131072, 384][384, 1]cuda:0" = PlaceHolder[target=mm]
#   %buf3 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:0" = PlaceHolder[target=buf3]
#   %rsqrt : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %primals_3 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=primals_3]
#   %view_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.bfloat16), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_6, 2), kwargs = {})
#   %mean : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %convert_element_type_5), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   return %buf3,%rsqrt,%convert_element_type_7
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_2 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1048576, 'r0_': 301991424}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask, tmp3, 0)
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr0 + (r0_1 + 384*x0), tmp17, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/of/cofpyuthvkgnusergdccojf23npyzmwdczvkaaqmt3nkk6oajyjn.py
# Topologically Sorted Source Nodes: [linear_1, view, getitem, v, normalize], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div]
# Source node to ATen node mapping:
#   getitem => slice_1
#   linear_1 => view_3
#   normalize => clamp_min, convert_element_type_10, div, expand, pow_2, pow_3, sum_1
#   v => clone
#   view => view_4
# Graph fragment:
#   %mm_1 : Tensor "bf16[131072, 768][768, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %clone : Tensor "bf16[128, 1024, 4, 48][196608, 192, 48, 1]cuda:0" = PlaceHolder[target=clone]
#   %sum_1 : Tensor "f32[128, 1024, 4, 1][4096, 4, 1, 524288]cuda:0" = PlaceHolder[target=sum_1]
#   %pow_3 : Tensor "f32[128, 1024, 4, 1][4096, 4, 1, 1]cuda:0" = PlaceHolder[target=pow_3]
#   %view_3 : Tensor "bf16[128, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [128, 1024, 768]), kwargs = {})
#   %view_4 : Tensor "bf16[128, 1024, 16, 48][786432, 768, 48, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%view_3, [128, 1024, 16, 48]), kwargs = {})
#   %slice_1 : Tensor "bf16[128, 1024, 4, 48][786432, 768, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_4, 2, 12, 9223372036854775807), kwargs = {})
#   %clone : Tensor "bf16[128, 1024, 4, 48][196608, 192, 48, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.clone.default](args = (%slice_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_10 : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_10, 2.0), kwargs = {})
#   %sum_1 : Tensor "f32[128, 1024, 4, 1][4096, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [-1], True), kwargs = {})
#   %pow_3 : Tensor "f32[128, 1024, 4, 1][4096, 4, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min : Tensor "f32[128, 1024, 4, 1][4096, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_3, 1e-12), kwargs = {})
#   %expand : Tensor "f32[128, 1024, 4, 48][4096, 4, 1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min, [128, 1024, 4, 48]), kwargs = {})
#   %div : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%clone, %expand), kwargs = {})
#   return %clone,%sum_1,%pow_3,%div
triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_slice_view_3 = async_compile.triton('triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_slice_view_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_slice_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4194304, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_slice_view_3(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_2 = r0_index
    x0 = (xindex % 4)
    x1 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (576 + r0_2 + 48*x0 + 768*x1), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tl.sqrt_rn(tmp6)
    tmp8 = tl.full([1, 1], 1e-12, tl.float32)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = (tmp1 / tmp9)
    tl.store(out_ptr0 + (r0_2 + 48*x3), tmp0, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (r0_2 + 48*x3), tmp10, r0_mask)
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
    triton_meta={'signature': {'QKV': '*bf16', 'Q': '*bf16', 'K': '*bf16', 'COS': '*bf16', 'SIN': '*bf16', 'TOTAL_ROWS': 'constexpr', 'SEQLEN': 'constexpr', 'H_Q': 'constexpr', 'H_KV': 'constexpr', 'HEAD_DIM': 'constexpr', 'ROPE_DIMS': 'constexpr', 'DO_QK_NORM': 'constexpr', 'ROPE_INTERLEAVED': 'constexpr', 'RMS_EPS': 'constexpr', 'BLOCK_D': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'TOTAL_ROWS': 131072, 'SEQLEN': 1024, 'H_Q': 8, 'H_KV': 4, 'HEAD_DIM': 48, 'ROPE_DIMS': 48, 'DO_QK_NORM': True, 'ROPE_INTERLEAVED': False, 'RMS_EPS': 1e-06, 'BLOCK_D': 64}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/4z/c4zaoayj4hurkqep45kyh72ufabs7hh3fvnin6oly6c4oub77q7a.py
# Topologically Sorted Source Nodes: [transpose_3, view_1, unsqueeze, mul, sum_1], Original ATen: [aten.transpose, aten.view, aten.unsqueeze, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul => mul_2
#   sum_1 => sum_2
#   transpose_3 => permute_7
#   unsqueeze => unsqueeze
#   view_1 => view_5
# Graph fragment:
#   %getitem_2 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:0" = PlaceHolder[target=getitem_2]
#   %div : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:0" = PlaceHolder[target=div]
#   %permute_7 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [0, 2, 1, 3]), kwargs = {})
#   %view_5 : Tensor "bf16[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_7, [128, 1024, 4, 2, 48]), kwargs = {})
#   %unsqueeze : Tensor "f32[128, 1024, 4, 1, 48][196608, 192, 48, 48, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%div, -2), kwargs = {})
#   %mul_2 : Tensor "f32[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %unsqueeze), kwargs = {})
#   %sum_2 : Tensor "f32[128, 1024, 4, 2, 1][8192, 8, 2, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [-1], True), kwargs = {dtype: torch.float32})
#   return %sum_2
triton_per_fused_mul_sum_transpose_unsqueeze_view_4 = async_compile.triton('triton_per_fused_mul_sum_transpose_unsqueeze_view_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_transpose_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8388608, 'r0_': 201326592}}
)
@triton.jit
def triton_per_fused_mul_sum_transpose_unsqueeze_view_4(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x3 = xindex
    x1 = xindex // 2
    tmp0 = tl.load(in_ptr0 + (r0_2 + 48*x3), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_2 + 48*x1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/bg/cbg2gkgjfcgcu7eiauolmahw4y3wnml5irkd2ju5ackpbgcfwdnj.py
# Topologically Sorted Source Nodes: [to_3, linear_2], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_2 => permute_8
#   to_3 => convert_element_type_11
# Graph fragment:
#   %primals_7 : Tensor "f32[8, 12][12, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %convert_element_type_11 : Tensor "bf16[8, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_7, torch.bfloat16), kwargs = {})
#   %permute_8 : Tensor "bf16[12, 8][1, 12]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_11, [1, 0]), kwargs = {})
#   return %permute_8
triton_poi_fused__to_copy_t_5 = async_compile.triton('triton_poi_fused__to_copy_t_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/po/cpo7frrlb6q5kvms5rtc2uyox2vevub44pz2hutgl4ku6qfegy6g.py
# Topologically Sorted Source Nodes: [getitem_3, contiguous_4], Original ATen: [aten.slice, aten.clone]
# Source node to ATen node mapping:
#   contiguous_4 => clone_1
#   getitem_3 => slice_2
# Graph fragment:
#   %convert_element_type_7 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:0" = PlaceHolder[target=convert_element_type_7]
#   %slice_2 : Tensor "bf16[128, 1024, 12][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_7, 2, 0, 12), kwargs = {})
#   %clone_1 : Tensor "bf16[128, 1024, 12][12288, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused_clone_slice_6 = async_compile.triton('triton_poi_fused_clone_slice_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_slice_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_slice_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 12)
    x1 = xindex // 12
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/mu/cmu2miem6eqhrkjk2vfgdjdibmxfnue6ydz6c3v5igtfvrfc46nb.py
# Topologically Sorted Source Nodes: [transpose_3, view_1, unsqueeze, mul_1, sub, reshape, linear_2, mul_2, sigmoid, getitem_4, mul_3, reshape_1, linear_3], Original ATen: [aten.transpose, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten._unsafe_view, aten.sigmoid, aten._to_copy]
# Source node to ATen node mapping:
#   getitem_4 => unsqueeze_1
#   linear_2 => view_8
#   linear_3 => convert_element_type_16
#   mul_1 => mul_3
#   mul_2 => mul_4
#   mul_3 => mul_5
#   reshape => view_6
#   reshape_1 => view_9
#   sigmoid => sigmoid
#   sub => sub
#   transpose_3 => permute_7
#   unsqueeze => unsqueeze
#   view_1 => view_5
# Graph fragment:
#   %getitem_2 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:0" = PlaceHolder[target=getitem_2]
#   %sum_2 : Tensor "f32[128, 1024, 4, 2, 1][8192, 8, 2, 1, 1]cuda:0" = PlaceHolder[target=sum_2]
#   %div : Tensor "f32[128, 1024, 4, 48][196608, 192, 48, 1]cuda:0" = PlaceHolder[target=div]
#   %mm_2 : Tensor "bf16[131072, 8][8, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %permute_7 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [0, 2, 1, 3]), kwargs = {})
#   %view_5 : Tensor "bf16[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_7, [128, 1024, 4, 2, 48]), kwargs = {})
#   %unsqueeze : Tensor "f32[128, 1024, 4, 1, 48][196608, 192, 48, 48, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%div, -2), kwargs = {})
#   %mul_3 : Tensor "f32[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, %unsqueeze), kwargs = {})
#   %sub : Tensor "f32[128, 1024, 4, 2, 48][393216, 384, 96, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %mul_3), kwargs = {})
#   %view_6 : Tensor "f32[128, 1024, 8, 48][393216, 384, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub, [128, 1024, 8, 48]), kwargs = {})
#   %view_8 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [128, 1024, 8]), kwargs = {})
#   %mul_4 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, 0.5), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_4,), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[128, 1024, 8, 1][8192, 8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid, 3), kwargs = {})
#   %mul_5 : Tensor "f32[128, 1024, 8, 48][393216, 384, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %unsqueeze_1), kwargs = {})
#   %view_9 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_5, [128, 1024, 384]), kwargs = {})
#   %convert_element_type_16 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_9, torch.bfloat16), kwargs = {})
#   return %convert_element_type_16
triton_poi_fused__to_copy__unsafe_view_mul_sigmoid_sub_transpose_unsqueeze_view_7 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_mul_sigmoid_sub_transpose_unsqueeze_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_mul_sigmoid_sub_transpose_unsqueeze_view_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 408944640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_mul_sigmoid_sub_transpose_unsqueeze_view_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 384)
    x1 = xindex // 384
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2 // 48), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (48*(x0 // 96) + 192*x1 + ((x0 % 48))), None)
    tmp6 = tl.load(in_ptr3 + (x2 // 48), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tl.full([1], 0.5, tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp5 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/x5/cx5tppa7npgqwh7y6u73cyr624xmvbhob3hcjwxyfkm7jbl3qxh7.py
# Topologically Sorted Source Nodes: [linear, linear_3, add, to_5, rms_norm_1], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten._fused_rms_norm]
# Source node to ATen node mapping:
#   add => add_1
#   linear => view_1
#   linear_3 => view_11
#   rms_norm_1 => add_2, convert_element_type_20, convert_element_type_21, mean_1, mul_6, mul_7, pow_4, rsqrt_1
#   to_5 => convert_element_type_19
# Graph fragment:
#   %mm_3 : Tensor "bf16[131072, 384][384, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %mm : Tensor "bf16[131072, 384][384, 1]cuda:0" = PlaceHolder[target=mm]
#   %add_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:0" = PlaceHolder[target=add_1]
#   %buf29 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:0" = PlaceHolder[target=buf29]
#   %rsqrt_1 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %primals_9 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=primals_9]
#   %view_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 384]), kwargs = {})
#   %view_11 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [128, 1024, 384]), kwargs = {})
#   %add_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %view_1), kwargs = {})
#   %convert_element_type_19 : Tensor "bf16[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_9, torch.bfloat16), kwargs = {})
#   %convert_element_type_20 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_4 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_20, 2), kwargs = {})
#   %mean_1 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [2], True), kwargs = {})
#   %add_2 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_6 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, %rsqrt_1), kwargs = {})
#   %mul_7 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %convert_element_type_19), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.bfloat16), kwargs = {})
#   return %add_1,%buf29,%rsqrt_1,%convert_element_type_21
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1048576, 'r0_': 603981312}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask, tmp5, 0)
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
    tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp13, None)
    tl.store(out_ptr0 + (r0_1 + 384*x0), tmp19, r0_mask)
''', device_str='cuda')


# Original path: /workspace/parameter-golf/train_gpt_parcae.py:4795
_leaky_relu_sq_matmul_kernel_1 = async_compile.triton('_leaky_relu_sq_matmul_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 4, 'num_stages': 4}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_leaky_relu_sq_matmul_kernel_1', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'A': '*bf16', 'B': '*bf16', 'C': '*bf16', 'AUX': '*bf16', 'M': 'constexpr', 'N': 'constexpr', 'K': 'constexpr', 'stride_am': 'constexpr', 'stride_ak': 'constexpr', 'stride_bn': 'constexpr', 'stride_bk': 'constexpr', 'stride_cm': 'constexpr', 'stride_cn': 'constexpr', 'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr', 'BLOCK_K': 'constexpr', 'GROUP_M': 'constexpr', 'FORWARD': 'constexpr', 'NEGATIVE_SLOPE': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'M': 131072, 'N': 1536, 'K': 384, 'stride_am': 384, 'stride_ak': 1, 'stride_bn': 384, 'stride_bk': 1, 'stride_cm': 1536, 'stride_cn': 1, 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'FORWARD': True, 'NEGATIVE_SLOPE': 0.5}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27 = args
        args.clear()
        assert_size_stride(primals_1, (384, 384), (384, 1))
        assert_size_stride(primals_2, (128, 1024, 384), (393216, 384, 1))
        assert_size_stride(primals_3, (384, ), (1, ))
        assert_size_stride(primals_4, (768, 384), (384, 1))
        assert_size_stride(primals_5, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(primals_6, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(primals_7, (8, 12), (12, 1))
        assert_size_stride(primals_8, (384, 384), (384, 1))
        assert_size_stride(primals_9, (384, ), (1, ))
        assert_size_stride(primals_10, (1536, 384), (384, 1))
        assert_size_stride(primals_11, (384, 1536), (1536, 1))
        assert_size_stride(primals_12, (384, ), (1, ))
        assert_size_stride(primals_13, (768, 384), (384, 1))
        assert_size_stride(primals_14, (8, 12), (12, 1))
        assert_size_stride(primals_15, (384, 384), (384, 1))
        assert_size_stride(primals_16, (384, ), (1, ))
        assert_size_stride(primals_17, (1536, 384), (384, 1))
        assert_size_stride(primals_18, (384, 1536), (1536, 1))
        assert_size_stride(primals_19, (384, ), (1, ))
        assert_size_stride(primals_20, (768, 384), (384, 1))
        assert_size_stride(primals_21, (8, 12), (12, 1))
        assert_size_stride(primals_22, (384, 384), (384, 1))
        assert_size_stride(primals_23, (384, ), (1, ))
        assert_size_stride(primals_24, (1536, 384), (384, 1))
        assert_size_stride(primals_25, (384, 1536), (1536, 1))
        assert_size_stride(primals_26, (384, ), (1, ))
        assert_size_stride(primals_27, (128, 1024), (1024, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((384, 384), (1, 384), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_0.run(primals_1, buf0, 147456, stream=stream0)
            del primals_1
            buf1 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_2, buf1, 50331648, stream=stream0)
            del primals_2
            buf2 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1, (131072, 384), (384, 1), 0), buf0, out=buf2)
            buf3 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
            buf4 = reinterpret_tensor(buf3, (128, 1024, 1), (1024, 1, 1), 0); del buf3  # reuse
            buf5 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, to_1, rms_norm], Original ATen: [aten._unsafe_view, aten._to_copy, aten._fused_rms_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_2.run(buf4, buf2, primals_3, buf5, 131072, 384, stream=stream0)
            buf6 = empty_strided_cuda((131072, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, to_1, rms_norm, linear_1], Original ATen: [aten._unsafe_view, aten._to_copy, aten._fused_rms_norm, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf5, (131072, 384), (384, 1), 0), reinterpret_tensor(primals_4, (384, 768), (1, 384), 0), out=buf6)
            buf7 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            buf8 = empty_strided_cuda((128, 1024, 4, 48), (196608, 192, 48, 1), torch.bfloat16)
            buf9 = empty_strided_cuda((128, 1024, 4, 48), (196608, 192, 48, 1), torch.bfloat16)
            buf18 = empty_strided_cuda((128, 1024, 4, 1), (4096, 4, 1, 524288), torch.float32)
            buf19 = reinterpret_tensor(buf18, (128, 1024, 4, 1), (4096, 4, 1, 1), 0); del buf18  # reuse
            buf20 = empty_strided_cuda((128, 1024, 4, 48), (196608, 192, 48, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, view, getitem, v, normalize], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_slice_view_3.run(buf19, buf6, buf9, buf20, 524288, 48, stream=stream0)
            # Topologically Sorted Source Nodes: [linear_1, view, getitem_1, getitem_2, triton_kernel_wrapper_mutation], Original ATen: [aten._unsafe_view, aten.view, aten.select]
            stream0 = get_raw_stream(0)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf6, (128, 1024, 16, 48), (786432, 768, 48, 1), 0), buf7, buf8, reinterpret_tensor(primals_5, (1024, 24), (24, 1), 0), reinterpret_tensor(primals_6, (1024, 24), (24, 1), 0), 1572864, 1, 1, stream=stream0)
            # Topologically Sorted Source Nodes: [transpose_2, scaled_dot_product_attention], Original ATen: [aten.transpose, aten._scaled_dot_product_flash_attention]
            buf12 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf7, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf8, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), reinterpret_tensor(buf9, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), 0.0, True, scale=0.14433756729740646)
            buf13 = buf12[0]
            assert_size_stride(buf13, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf13, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf14 = buf12[1]
            assert_size_stride(buf14, (128, 8, 1024), (8192, 1024, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf14, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf15 = buf12[6]
            assert_size_stride(buf15, (2, ), (1, ), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf15, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf16 = buf12[7]
            assert_size_stride(buf16, (), (), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf16, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf12
            buf21 = empty_strided_cuda((128, 1024, 4, 2, 1), (8192, 8, 2, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [transpose_3, view_1, unsqueeze, mul, sum_1], Original ATen: [aten.transpose, aten.view, aten.unsqueeze, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused_mul_sum_transpose_unsqueeze_view_4.run(buf13, buf20, buf21, 1048576, 48, stream=stream0)
            buf22 = empty_strided_cuda((12, 8), (1, 12), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_3, linear_2], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_7, buf22, 96, stream=stream0)
            del primals_7
            buf23 = empty_strided_cuda((128, 1024, 12), (12288, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_3, contiguous_4], Original ATen: [aten.slice, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_slice_6.run(buf5, buf23, 1572864, stream=stream0)
            buf24 = empty_strided_cuda((131072, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_3, contiguous_4, linear_2], Original ATen: [aten.slice, aten.clone, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf23, (131072, 12), (12, 1), 0), buf22, out=buf24)
            buf25 = empty_strided_cuda((384, 384), (1, 384), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_0.run(primals_8, buf25, 147456, stream=stream0)
            del primals_8
            buf26 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_3, view_1, unsqueeze, mul_1, sub, reshape, linear_2, mul_2, sigmoid, getitem_4, mul_3, reshape_1, linear_3], Original ATen: [aten.transpose, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten._unsafe_view, aten.sigmoid, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_sigmoid_sub_transpose_unsqueeze_view_7.run(buf13, buf21, buf20, buf24, buf26, 50331648, stream=stream0)
            buf27 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_3, view_1, unsqueeze, mul_1, sub, reshape, linear_2, mul_2, sigmoid, getitem_4, mul_3, reshape_1, linear_3], Original ATen: [aten.transpose, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf26, (131072, 384), (384, 1), 0), buf25, out=buf27)
            buf28 = reinterpret_tensor(buf27, (128, 1024, 384), (393216, 384, 1), 0); del buf27  # reuse
            buf29 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
            buf30 = reinterpret_tensor(buf29, (128, 1024, 1), (1024, 1, 1), 0); del buf29  # reuse
            buf31 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, linear_3, add, to_5, rms_norm_1], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten._fused_rms_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8.run(buf28, buf30, buf2, primals_9, buf31, 131072, 384, stream=stream0)
            buf32 = empty_strided_cuda((131072, 1536), (1536, 1), torch.bfloat16)
            buf33 = empty_strided_cuda((131072, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_5, rms_norm_1, reshape, triton_kernel_wrapper_mutation], Original ATen: [aten._to_copy, aten._fused_rms_norm, aten.view]
            stream0 = get_raw_stream(0)
            _leaky_relu_sq_matmul_kernel_1.run(reinterpret_tensor(buf31, (131072, 384), (384, 1), 0), primals_10, buf32, buf33, 24576, 1, 1, stream=stream0)
            buf36 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getattr_2, out], Original ATen: [aten.permute, aten.mm]
            extern_kernels.mm(buf33, reinterpret_tensor(primals_11, (1536, 384), (1, 1536), 0), out=buf36)
            buf37 = reinterpret_tensor(buf36, (128, 1024, 384), (393216, 384, 1), 0); del buf36  # reuse
            buf38 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
            buf39 = reinterpret_tensor(buf38, (128, 1024, 1), (1024, 1, 1), 0); del buf38  # reuse
            buf40 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [outs, add_1, to_8, rms_norm_2], Original ATen: [aten.view, aten.add, aten._to_copy, aten._fused_rms_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8.run(buf37, buf39, buf28, primals_12, buf40, 131072, 384, stream=stream0)
            buf41 = empty_strided_cuda((131072, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_8, rms_norm_2, linear_4], Original ATen: [aten._to_copy, aten._fused_rms_norm, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (131072, 384), (384, 1), 0), reinterpret_tensor(primals_13, (384, 768), (1, 384), 0), out=buf41)
            buf42 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            buf43 = empty_strided_cuda((128, 1024, 4, 48), (196608, 192, 48, 1), torch.bfloat16)
            buf44 = empty_strided_cuda((128, 1024, 4, 48), (196608, 192, 48, 1), torch.bfloat16)
            buf53 = empty_strided_cuda((128, 1024, 4, 1), (4096, 4, 1, 524288), torch.float32)
            buf54 = reinterpret_tensor(buf53, (128, 1024, 4, 1), (4096, 4, 1, 1), 0); del buf53  # reuse
            buf55 = empty_strided_cuda((128, 1024, 4, 48), (196608, 192, 48, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_4, view_2, getitem, v, normalize_1], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_slice_view_3.run(buf54, buf41, buf44, buf55, 524288, 48, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_1, getitem_2, linear_4, view_2, triton_kernel_wrapper_mutation], Original ATen: [aten.select, aten._unsafe_view, aten.view]
            stream0 = get_raw_stream(0)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf41, (128, 1024, 16, 48), (786432, 768, 48, 1), 0), buf42, buf43, reinterpret_tensor(primals_5, (1024, 24), (24, 1), 0), reinterpret_tensor(primals_6, (1024, 24), (24, 1), 0), 1572864, 1, 1, stream=stream0)
            # Topologically Sorted Source Nodes: [transpose_6, scaled_dot_product_attention_1], Original ATen: [aten.transpose, aten._scaled_dot_product_flash_attention]
            buf47 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf42, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf43, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), reinterpret_tensor(buf44, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), 0.0, True, scale=0.14433756729740646)
            buf48 = buf47[0]
            assert_size_stride(buf48, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf48, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf49 = buf47[1]
            assert_size_stride(buf49, (128, 8, 1024), (8192, 1024, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf49, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf50 = buf47[6]
            assert_size_stride(buf50, (2, ), (1, ), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf50, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf51 = buf47[7]
            assert_size_stride(buf51, (), (), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf51, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf47
            buf56 = empty_strided_cuda((128, 1024, 4, 2, 1), (8192, 8, 2, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [transpose_7, view_3, unsqueeze_1, mul_4, sum_2], Original ATen: [aten.transpose, aten.view, aten.unsqueeze, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused_mul_sum_transpose_unsqueeze_view_4.run(buf48, buf55, buf56, 1048576, 48, stream=stream0)
            buf57 = empty_strided_cuda((12, 8), (1, 12), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_10, linear_5], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_14, buf57, 96, stream=stream0)
            del primals_14
            buf58 = empty_strided_cuda((128, 1024, 12), (12288, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_9, contiguous_10], Original ATen: [aten.slice, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_slice_6.run(buf40, buf58, 1572864, stream=stream0)
            buf59 = empty_strided_cuda((131072, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_9, contiguous_10, linear_5], Original ATen: [aten.slice, aten.clone, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf58, (131072, 12), (12, 1), 0), buf57, out=buf59)
            buf60 = empty_strided_cuda((384, 384), (1, 384), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_0.run(primals_15, buf60, 147456, stream=stream0)
            del primals_15
            buf61 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_7, view_3, unsqueeze_1, mul_5, sub_1, reshape_3, linear_5, mul_6, sigmoid_1, getitem_10, mul_7, reshape_4, linear_6], Original ATen: [aten.transpose, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten._unsafe_view, aten.sigmoid, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_sigmoid_sub_transpose_unsqueeze_view_7.run(buf48, buf56, buf55, buf59, buf61, 50331648, stream=stream0)
            buf62 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_7, view_3, unsqueeze_1, mul_5, sub_1, reshape_3, linear_5, mul_6, sigmoid_1, getitem_10, mul_7, reshape_4, linear_6], Original ATen: [aten.transpose, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf61, (131072, 384), (384, 1), 0), buf60, out=buf62)
            buf63 = reinterpret_tensor(buf62, (128, 1024, 384), (393216, 384, 1), 0); del buf62  # reuse
            buf64 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
            buf65 = reinterpret_tensor(buf64, (128, 1024, 1), (1024, 1, 1), 0); del buf64  # reuse
            buf66 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_6, add_2, to_12, rms_norm_3], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten._fused_rms_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8.run(buf63, buf65, buf37, primals_16, buf66, 131072, 384, stream=stream0)
            buf67 = empty_strided_cuda((131072, 1536), (1536, 1), torch.bfloat16)
            buf68 = empty_strided_cuda((131072, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_12, rms_norm_3, reshape, triton_kernel_wrapper_mutation], Original ATen: [aten._to_copy, aten._fused_rms_norm, aten.view]
            stream0 = get_raw_stream(0)
            _leaky_relu_sq_matmul_kernel_1.run(reinterpret_tensor(buf66, (131072, 384), (384, 1), 0), primals_17, buf67, buf68, 24576, 1, 1, stream=stream0)
            buf71 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getattr_4, out], Original ATen: [aten.permute, aten.mm]
            extern_kernels.mm(buf68, reinterpret_tensor(primals_18, (1536, 384), (1, 1536), 0), out=buf71)
            buf72 = reinterpret_tensor(buf71, (128, 1024, 384), (393216, 384, 1), 0); del buf71  # reuse
            buf73 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
            buf74 = reinterpret_tensor(buf73, (128, 1024, 1), (1024, 1, 1), 0); del buf73  # reuse
            buf75 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [outs, add_3, to_15, rms_norm_4], Original ATen: [aten.view, aten.add, aten._to_copy, aten._fused_rms_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8.run(buf72, buf74, buf63, primals_19, buf75, 131072, 384, stream=stream0)
            buf76 = empty_strided_cuda((131072, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_15, rms_norm_4, linear_7], Original ATen: [aten._to_copy, aten._fused_rms_norm, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf75, (131072, 384), (384, 1), 0), reinterpret_tensor(primals_20, (384, 768), (1, 384), 0), out=buf76)
            buf77 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            buf78 = empty_strided_cuda((128, 1024, 4, 48), (196608, 192, 48, 1), torch.bfloat16)
            buf79 = empty_strided_cuda((128, 1024, 4, 48), (196608, 192, 48, 1), torch.bfloat16)
            buf88 = empty_strided_cuda((128, 1024, 4, 1), (4096, 4, 1, 524288), torch.float32)
            buf89 = reinterpret_tensor(buf88, (128, 1024, 4, 1), (4096, 4, 1, 1), 0); del buf88  # reuse
            buf90 = empty_strided_cuda((128, 1024, 4, 48), (196608, 192, 48, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7, view_4, getitem, v, normalize_2], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_slice_view_3.run(buf89, buf76, buf79, buf90, 524288, 48, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_1, getitem_2, linear_7, view_4, triton_kernel_wrapper_mutation], Original ATen: [aten.select, aten._unsafe_view, aten.view]
            stream0 = get_raw_stream(0)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf76, (128, 1024, 16, 48), (786432, 768, 48, 1), 0), buf77, buf78, reinterpret_tensor(primals_5, (1024, 24), (24, 1), 0), reinterpret_tensor(primals_6, (1024, 24), (24, 1), 0), 1572864, 1, 1, stream=stream0)
            # Topologically Sorted Source Nodes: [transpose_10, scaled_dot_product_attention_2], Original ATen: [aten.transpose, aten._scaled_dot_product_flash_attention]
            buf82 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf77, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf78, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), reinterpret_tensor(buf79, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), 0.0, True, scale=0.14433756729740646)
            buf83 = buf82[0]
            assert_size_stride(buf83, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf83, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf84 = buf82[1]
            assert_size_stride(buf84, (128, 8, 1024), (8192, 1024, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf84, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf85 = buf82[6]
            assert_size_stride(buf85, (2, ), (1, ), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf85, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf86 = buf82[7]
            assert_size_stride(buf86, (), (), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf86, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf82
            buf91 = empty_strided_cuda((128, 1024, 4, 2, 1), (8192, 8, 2, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [transpose_11, view_5, unsqueeze_2, mul_8, sum_3], Original ATen: [aten.transpose, aten.view, aten.unsqueeze, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused_mul_sum_transpose_unsqueeze_view_4.run(buf83, buf90, buf91, 1048576, 48, stream=stream0)
            buf92 = empty_strided_cuda((12, 8), (1, 12), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_17, linear_8], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_21, buf92, 96, stream=stream0)
            del primals_21
            buf93 = empty_strided_cuda((128, 1024, 12), (12288, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_15, contiguous_16], Original ATen: [aten.slice, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_slice_6.run(buf75, buf93, 1572864, stream=stream0)
            buf94 = empty_strided_cuda((131072, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_15, contiguous_16, linear_8], Original ATen: [aten.slice, aten.clone, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf93, (131072, 12), (12, 1), 0), buf92, out=buf94)
            buf95 = empty_strided_cuda((384, 384), (1, 384), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_0.run(primals_22, buf95, 147456, stream=stream0)
            del primals_22
            buf96 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_11, view_5, unsqueeze_2, mul_9, sub_2, reshape_6, linear_8, mul_10, sigmoid_2, getitem_16, mul_11, reshape_7, linear_9], Original ATen: [aten.transpose, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten._unsafe_view, aten.sigmoid, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_sigmoid_sub_transpose_unsqueeze_view_7.run(buf83, buf91, buf90, buf94, buf96, 50331648, stream=stream0)
            buf97 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_11, view_5, unsqueeze_2, mul_9, sub_2, reshape_6, linear_8, mul_10, sigmoid_2, getitem_16, mul_11, reshape_7, linear_9], Original ATen: [aten.transpose, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf96, (131072, 384), (384, 1), 0), buf95, out=buf97)
            buf98 = reinterpret_tensor(buf97, (128, 1024, 384), (393216, 384, 1), 0); del buf97  # reuse
            buf99 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
            buf100 = reinterpret_tensor(buf99, (128, 1024, 1), (1024, 1, 1), 0); del buf99  # reuse
            buf101 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_9, add_4, to_19, rms_norm_5], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten._fused_rms_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8.run(buf98, buf100, buf72, primals_23, buf101, 131072, 384, stream=stream0)
            buf102 = empty_strided_cuda((131072, 1536), (1536, 1), torch.bfloat16)
            buf103 = empty_strided_cuda((131072, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_19, rms_norm_5, reshape, triton_kernel_wrapper_mutation], Original ATen: [aten._to_copy, aten._fused_rms_norm, aten.view]
            stream0 = get_raw_stream(0)
            _leaky_relu_sq_matmul_kernel_1.run(reinterpret_tensor(buf101, (131072, 384), (384, 1), 0), primals_24, buf102, buf103, 24576, 1, 1, stream=stream0)
            buf106 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getattr_6, out], Original ATen: [aten.permute, aten.mm]
            extern_kernels.mm(buf103, reinterpret_tensor(primals_25, (1536, 384), (1, 1536), 0), out=buf106)
            buf107 = reinterpret_tensor(buf106, (128, 1024, 384), (393216, 384, 1), 0); del buf106  # reuse
            buf108 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
            buf109 = reinterpret_tensor(buf108, (128, 1024, 1), (1024, 1, 1), 0); del buf108  # reuse
            buf110 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [outs, add_5, to_22, rms_norm_6], Original ATen: [aten.view, aten.add, aten._to_copy, aten._fused_rms_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8.run(buf107, buf109, buf98, primals_26, buf110, 131072, 384, stream=stream0)
        return (reinterpret_tensor(buf110, (131072, 384), (384, 1), 0), reinterpret_tensor(primals_27, (131072, ), (1, ), 0), primals_3, primals_4, primals_9, primals_10, primals_11, primals_12, primals_13, primals_16, primals_17, primals_18, primals_19, primals_20, primals_23, primals_24, primals_25, primals_26, reinterpret_tensor(buf1, (131072, 384), (384, 1), 0), buf2, buf4, reinterpret_tensor(buf5, (131072, 384), (384, 1), 0), reinterpret_tensor(buf6, (128, 1024, 16, 48), (786432, 768, 48, 1), 0), buf9, reinterpret_tensor(primals_5, (1024, 24), (24, 1), 0), reinterpret_tensor(primals_6, (1024, 24), (24, 1), 0), reinterpret_tensor(buf7, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf8, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), buf13, buf14, buf15, buf16, buf19, buf20, buf21, reinterpret_tensor(buf23, (131072, 12), (12, 1), 0), buf24, reinterpret_tensor(buf26, (131072, 384), (384, 1), 0), buf28, buf30, reinterpret_tensor(buf31, (131072, 384), (384, 1), 0), buf32, buf33, buf37, buf39, reinterpret_tensor(buf40, (131072, 384), (384, 1), 0), reinterpret_tensor(buf41, (128, 1024, 16, 48), (786432, 768, 48, 1), 0), buf44, reinterpret_tensor(buf42, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf43, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), buf48, buf49, buf50, buf51, buf54, buf55, buf56, reinterpret_tensor(buf58, (131072, 12), (12, 1), 0), buf59, reinterpret_tensor(buf61, (131072, 384), (384, 1), 0), buf63, buf65, reinterpret_tensor(buf66, (131072, 384), (384, 1), 0), buf67, buf68, buf72, buf74, reinterpret_tensor(buf75, (131072, 384), (384, 1), 0), reinterpret_tensor(buf76, (128, 1024, 16, 48), (786432, 768, 48, 1), 0), buf79, reinterpret_tensor(buf77, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf78, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), buf83, buf84, buf85, buf86, buf89, buf90, buf91, reinterpret_tensor(buf93, (131072, 12), (12, 1), 0), buf94, reinterpret_tensor(buf96, (131072, 384), (384, 1), 0), buf98, buf100, reinterpret_tensor(buf101, (131072, 384), (384, 1), 0), buf102, buf103, buf107, buf109, reinterpret_tensor(buf95, (384, 384), (384, 1), 0), reinterpret_tensor(buf92, (8, 12), (12, 1), 0), reinterpret_tensor(buf60, (384, 384), (384, 1), 0), reinterpret_tensor(buf57, (8, 12), (12, 1), 0), reinterpret_tensor(buf25, (384, 384), (384, 1), 0), reinterpret_tensor(buf22, (8, 12), (12, 1), 0), reinterpret_tensor(buf0, (384, 384), (384, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((8, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_11 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_12 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_14 = rand_strided((8, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_16 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_18 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_19 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_21 = rand_strided((8, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_23 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_25 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_26 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
