# AOT ID: ['10_forward']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/bl/cblkx7muujjjlohfaarnmvdbduzlnjckvzn72en7t6wqhztgc4st.py
# Topologically Sorted Source Nodes: [mul, linear], Original ATen: [aten.mul, aten.t]
# Source node to ATen node mapping:
#   linear => permute
#   mul => mul
# Graph fragment:
#   %primals_1 : Tensor "bf16[8192, 384][384, 1]cuda:2" = PlaceHolder[target=primals_1]
#   %mul : Tensor "bf16[8192, 384][384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, 1.0), kwargs = {})
#   %permute : Tensor "bf16[384, 8192][1, 384]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [1, 0]), kwargs = {})
#   return %permute
triton_poi_fused_mul_t_0 = async_compile.triton('triton_poi_fused_mul_t_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_t_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_t_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 1.0, tl.float32)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/aa/caamh6ldymp65knudtuqbtysebebyjxff55bujsbsxol7xbswvjq.py
# Topologically Sorted Source Nodes: [float_1, truediv, tanh, cross_entropy], Original ATen: [aten._to_copy, aten.div, aten.tanh, aten.mul, aten.amax, aten.sub, aten._log_softmax]
# Source node to ATen node mapping:
#   cross_entropy => exp, log, sum_1
#   float_1 => convert_element_type_2
#   tanh => tanh
#   truediv => div
# Graph fragment:
#   %mm : Tensor "bf16[131072, 8192][8192, 1]cuda:2" = PlaceHolder[target=mm]
#   %amax_default : Tensor "f32[131072, 1][1, 1]cuda:2" = PlaceHolder[target=amax_default]
#   %sum_1 : Tensor "f32[131072, 1][1, 131072]cuda:2" = PlaceHolder[target=sum_1]
#   %convert_element_type_2 : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %div : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_2, 30.0), kwargs = {})
#   %tanh : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_tensor : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 1), kwargs = {})
#   %amax_default : Tensor "f32[131072, 1][1, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor, 30.0), kwargs = {})
#   %exp : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_1,), kwargs = {})
#   %sum_1 : Tensor "f32[131072, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : Tensor "f32[131072, 1][1, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   return %amax_default,%sum_1,%log
triton_red_fused__log_softmax__to_copy_amax_div_mul_sub_tanh_1 = async_compile.triton('triton_red_fused__log_softmax__to_copy_amax_div_mul_sub_tanh_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_amax_div_mul_sub_tanh_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 2097152, 'r0_': 2147483648}}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy_amax_div_mul_sub_tanh_1(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.full([1, 1], 0.03333333333333333, tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = libdevice.tanh(tmp3)
        tmp5 = tl.full([1, 1], 1.0, tl.float32)
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, None)
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 0.03333333333333333, tl.float32)
        tmp13 = tmp11 * tmp12
        tmp14 = libdevice.tanh(tmp13)
        tmp15 = tl.full([1, 1], 1.0, tl.float32)
        tmp16 = tmp14 * tmp15
        tmp17 = tmp16 - tmp8
        tmp18 = tl.full([1, 1], 30.0, tl.float32)
        tmp19 = tmp17 * tmp18
        tmp20 = libdevice.exp(tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp24 = tl_math.log(tmp22)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp24, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/bc/cbcq3l2zlxpbhcnqztyrcceqnhf6ysi426sufv22rgkuycpnucod.py
# Topologically Sorted Source Nodes: [float_1, truediv, tanh, cross_entropy], Original ATen: [aten._to_copy, aten.div, aten.tanh, aten.mul, aten.sub, aten._log_softmax, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   cross_entropy => full_default, full_default_1, gather, ne, neg, squeeze, sub_1, sum_2, sum_3, unsqueeze, where, where_1
#   float_1 => convert_element_type_2
#   tanh => tanh
#   truediv => div
# Graph fragment:
#   %primals_3 : Tensor "i64[131072][1]cuda:2" = PlaceHolder[target=primals_3]
#   %mm : Tensor "bf16[131072, 8192][8192, 1]cuda:2" = PlaceHolder[target=mm]
#   %amax_default : Tensor "f32[131072, 1][1, 1]cuda:2" = PlaceHolder[target=amax_default]
#   %log : Tensor "f32[131072, 1][1, 1]cuda:2" = PlaceHolder[target=log]
#   %convert_element_type_2 : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %div : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_2, 30.0), kwargs = {})
#   %tanh : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_tensor : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 1), kwargs = {})
#   %sub_tensor : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor, 30.0), kwargs = {})
#   %sub_1 : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %log), kwargs = {})
#   %ne : Tensor "b8[131072][1]cuda:2"[num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%primals_3, -100), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:2, pin_memory: False})
#   %where : Tensor "i64[131072][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %primals_3, %full_default), kwargs = {})
#   %unsqueeze : Tensor "i64[131072, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 1), kwargs = {})
#   %gather : Tensor "f32[131072, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_1, 1, %unsqueeze), kwargs = {})
#   %squeeze : Tensor "f32[131072][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg : Tensor "f32[131072][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:2, pin_memory: False})
#   %where_1 : Tensor "f32[131072][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg, %full_default_1), kwargs = {})
#   %sum_2 : Tensor "i64[][]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne,), kwargs = {})
#   %sum_3 : Tensor "f32[][]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   return %buf6,%buf10
triton_red_fused__log_softmax__to_copy_div_mul_nll_loss_forward_sub_tanh_2 = async_compile.triton('triton_red_fused__log_softmax__to_copy_div_mul_nll_loss_forward_sub_tanh_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*i64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_div_mul_nll_loss_forward_sub_tanh_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 384, 'r0_': 2097152}}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy_div_mul_nll_loss_forward_sub_tanh_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    _tmp31 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr2 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.load(in_ptr3 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tmp2.to(tl.int64)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
        tmp7 = tl.full([1, 1], 0, tl.int64)
        tmp8 = tl.where(tmp2, tmp0, tmp7)
        tmp9 = tl.full([1, 1], 8192, tl.int32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp8 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp8)
        tl.device_assert(((0 <= tmp12) & (tmp12 < 8192)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp12 < 8192")
        tmp14 = tl.load(in_ptr1 + (tmp12 + 8192*r0_1 + 67108864*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1, 1], 0.03333333333333333, tl.float32)
        tmp17 = tmp15 * tmp16
        tmp18 = libdevice.tanh(tmp17)
        tmp19 = tl.full([1, 1], 1.0, tl.float32)
        tmp20 = tmp18 * tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = tl.full([1, 1], 30.0, tl.float32)
        tmp24 = tmp22 * tmp23
        tmp26 = tmp24 - tmp25
        tmp27 = -tmp26
        tmp28 = tl.full([1, 1], 0.0, tl.float32)
        tmp29 = tl.where(tmp2, tmp27, tmp28)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(r0_mask & xmask, tmp32, _tmp31)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr1 + (x0), tmp31, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ri/cri2tigjmj2zvq6i2wj5mz3ybwdxso4gwj7navhw5y2vodqfu7i4.py
# Topologically Sorted Source Nodes: [float_1, truediv, tanh, cross_entropy], Original ATen: [aten._to_copy, aten.div, aten.tanh, aten.mul, aten.sub, aten._log_softmax, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   cross_entropy => convert_element_type_3, div_1, full_default, full_default_1, gather, ne, neg, squeeze, sub_1, sum_2, sum_3, unsqueeze, where, where_1
#   float_1 => convert_element_type_2
#   tanh => tanh
#   truediv => div
# Graph fragment:
#   %buf10 : Tensor "f32[16][1]cuda:2" = PlaceHolder[target=buf10]
#   %buf6 : Tensor "i64[16][1]cuda:2" = PlaceHolder[target=buf6]
#   %sum_2 : Tensor "i64[][]cuda:2" = PlaceHolder[target=sum_2]
#   %sum_3 : Tensor "f32[][]cuda:2" = PlaceHolder[target=sum_3]
#   %convert_element_type_3 : Tensor "f32[][]cuda:2" = PlaceHolder[target=convert_element_type_3]
#   %convert_element_type_2 : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %div : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_2, 30.0), kwargs = {})
#   %tanh : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_tensor : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 1), kwargs = {})
#   %sub_tensor : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor, 30.0), kwargs = {})
#   %sub_1 : Tensor "f32[131072, 8192][8192, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %log), kwargs = {})
#   %ne : Tensor "b8[131072][1]cuda:2"[num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%primals_3, -100), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:2, pin_memory: False})
#   %where : Tensor "i64[131072][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %primals_3, %full_default), kwargs = {})
#   %unsqueeze : Tensor "i64[131072, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 1), kwargs = {})
#   %gather : Tensor "f32[131072, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_1, 1, %unsqueeze), kwargs = {})
#   %squeeze : Tensor "f32[131072][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg : Tensor "f32[131072][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:2, pin_memory: False})
#   %where_1 : Tensor "f32[131072][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg, %full_default_1), kwargs = {})
#   %sum_2 : Tensor "i64[][]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne,), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[][]cuda:2"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %sum_3 : Tensor "f32[][]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %div_1 : Tensor "f32[][]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type_3), kwargs = {})
#   return %sum_3,%sum_2,%convert_element_type_3,%div_1
triton_per_fused__log_softmax__to_copy_div_mul_nll_loss_forward_sub_tanh_3 = async_compile.triton('triton_per_fused__log_softmax__to_copy_div_mul_nll_loss_forward_sub_tanh_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__to_copy_div_mul_nll_loss_forward_sub_tanh_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 192}}
)
@triton.jit
def triton_per_fused__log_softmax__to_copy_div_mul_nll_loss_forward_sub_tanh_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp4 = tl.load(in_ptr1 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None].to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None].to(tl.int64)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = (tmp3 / tmp8)
    tl.store(out_ptr1 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp8, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp9, None)
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
        primals_1, primals_2, primals_3 = args
        args.clear()
        assert_size_stride(primals_1, (8192, 384), (384, 1))
        assert_size_stride(primals_2, (131072, 384), (384, 1))
        assert_size_stride(primals_3, (131072, ), (1, ))
        with torch.cuda._DeviceGuard(2):
            torch.cuda.set_device(2)
            buf0 = empty_strided_cuda((384, 8192), (1, 384), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul, linear], Original ATen: [aten.mul, aten.t]
            stream2 = get_raw_stream(2)
            triton_poi_fused_mul_t_0.run(primals_1, buf0, 3145728, stream=stream2)
            del primals_1
            buf1 = empty_strided_cuda((131072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
            extern_kernels.mm(primals_2, buf0, out=buf1)
            buf2 = empty_strided_cuda((131072, 1), (1, 1), torch.float32)
            buf3 = empty_strided_cuda((131072, 1), (1, 131072), torch.float32)
            buf4 = reinterpret_tensor(buf3, (131072, 1), (1, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [float_1, truediv, tanh, cross_entropy], Original ATen: [aten._to_copy, aten.div, aten.tanh, aten.mul, aten.amax, aten.sub, aten._log_softmax]
            stream2 = get_raw_stream(2)
            triton_red_fused__log_softmax__to_copy_amax_div_mul_sub_tanh_1.run(buf4, buf1, buf2, 131072, 8192, stream=stream2)
            buf6 = empty_strided_cuda((16, ), (1, ), torch.int64)
            buf10 = empty_strided_cuda((16, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [float_1, truediv, tanh, cross_entropy], Original ATen: [aten._to_copy, aten.div, aten.tanh, aten.mul, aten.sub, aten._log_softmax, aten.nll_loss_forward]
            stream2 = get_raw_stream(2)
            triton_red_fused__log_softmax__to_copy_div_mul_nll_loss_forward_sub_tanh_2.run(primals_3, buf1, buf2, buf4, buf6, buf10, 16, 8192, stream=stream2)
            buf11 = empty_strided_cuda((), (), torch.float32)
            buf8 = empty_strided_cuda((), (), torch.float32)
            buf12 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [float_1, truediv, tanh, cross_entropy], Original ATen: [aten._to_copy, aten.div, aten.tanh, aten.mul, aten.sub, aten._log_softmax, aten.nll_loss_forward]
            stream2 = get_raw_stream(2)
            triton_per_fused__log_softmax__to_copy_div_mul_nll_loss_forward_sub_tanh_3.run(buf12, buf10, buf6, buf8, 1, 16, stream=stream2)
            del buf10
            del buf6
        return (buf12, primals_2, primals_3, buf1, buf2, buf4, buf8, reinterpret_tensor(buf0, (8192, 384), (384, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((8192, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_2 = rand_strided((131072, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_3 = rand_strided((131072, ), (1, ), device='cuda:2', dtype=torch.int64)
    return [primals_1, primals_2, primals_3]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
