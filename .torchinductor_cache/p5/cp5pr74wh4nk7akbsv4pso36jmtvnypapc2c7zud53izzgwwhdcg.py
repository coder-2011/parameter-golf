# AOT ID: ['22_inference']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/il/cil573acdjx5rm7znlg5fro2lldrdc4svpioz7hgelvhyhmfejfl.py
# Topologically Sorted Source Nodes: [getitem_4, getitem_6, tensor, mul, getitem_5, getitem_7, tensor_1, mul_1, bitwise_xor], Original ATen: [aten.slice, aten.unsqueeze, aten.lift_fresh, aten.mul, aten.bitwise_xor]
# Source node to ATen node mapping:
#   bitwise_xor => bitwise_xor
#   getitem_4 => slice_5
#   getitem_5 => slice_6
#   getitem_6 => unsqueeze
#   getitem_7 => unsqueeze_1
#   mul => mul
#   mul_1 => mul_1
#   tensor => lift_fresh_copy
#   tensor_1 => lift_fresh_copy_1
# Graph fragment:
#   %arg4_1 : Tensor "i64[350, 1000][1000, 1]cuda:3" = PlaceHolder[target=arg4_1]
#   %slice_5 : Tensor "i64[350, 999][1000, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg4_1, 1, 1, 9223372036854775807), kwargs = {})
#   %unsqueeze : Tensor "i64[350, 999, 1][1000, 1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_5, 2), kwargs = {})
#   %lift_fresh_copy : Tensor "i64[7][1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant0,), kwargs = {})
#   %mul : Tensor "i64[350, 999, 7][6993, 7, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %lift_fresh_copy), kwargs = {})
#   %slice_6 : Tensor "i64[350, 999][1000, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg4_1, 1, 0, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[350, 999, 1][1000, 1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_6, 2), kwargs = {})
#   %lift_fresh_copy_1 : Tensor "i64[7][1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant1,), kwargs = {})
#   %mul_1 : Tensor "i64[350, 999, 7][6993, 7, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %lift_fresh_copy_1), kwargs = {})
#   %bitwise_xor : Tensor "i64[350, 999, 7][6993, 7, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.bitwise_xor.Tensor](args = (%mul, %mul_1), kwargs = {})
#   return %bitwise_xor
triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0 = async_compile.triton('triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 39160800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2447550
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 7) % 999)
    x2 = xindex // 6993
    x0 = (xindex % 7)
    x3 = (xindex % 6993)
    tmp0 = tl.load(in_ptr0 + (1 + x1 + 1000*x2), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (x1 + 1000*x2), xmask, eviction_policy='evict_last')
    tmp1 = x0
    tmp2 = tl.full([1], 3, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp1 < tmp6
    tmp8 = tl.full([1], 17491, tl.int64)
    tmp9 = tl.full([1], 52973, tl.int64)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tl.full([1], 36313, tl.int64)
    tmp12 = tl.where(tmp5, tmp11, tmp10)
    tmp13 = tl.full([1], 5, tl.int64)
    tmp14 = tmp1 < tmp13
    tmp15 = tl.full([1], 4, tl.int64)
    tmp16 = tmp1 < tmp15
    tmp17 = tl.full([1], 29837, tl.int64)
    tmp18 = tl.full([1], 44497, tl.int64)
    tmp19 = tl.where(tmp16, tmp17, tmp18)
    tmp20 = tl.full([1], 6, tl.int64)
    tmp21 = tmp1 < tmp20
    tmp22 = tl.full([1], 62137, tl.int64)
    tmp23 = tl.full([1], 24071, tl.int64)
    tmp24 = tl.where(tmp21, tmp22, tmp23)
    tmp25 = tl.where(tmp14, tmp19, tmp24)
    tmp26 = tl.where(tmp3, tmp12, tmp25)
    tmp27 = tmp0 * tmp26
    tmp29 = tl.full([1], 43889, tl.int64)
    tmp30 = tl.full([1], 19937, tl.int64)
    tmp31 = tl.where(tmp7, tmp29, tmp30)
    tmp32 = tl.full([1], 27191, tl.int64)
    tmp33 = tl.where(tmp5, tmp32, tmp31)
    tmp34 = tl.full([1], 60271, tl.int64)
    tmp35 = tl.full([1], 34583, tl.int64)
    tmp36 = tl.where(tmp16, tmp34, tmp35)
    tmp37 = tl.full([1], 49331, tl.int64)
    tmp38 = tl.full([1], 15461, tl.int64)
    tmp39 = tl.where(tmp21, tmp37, tmp38)
    tmp40 = tl.where(tmp14, tmp36, tmp39)
    tmp41 = tl.where(tmp3, tmp33, tmp40)
    tmp42 = tmp28 * tmp41
    tmp43 = tmp27 ^ tmp42
    tl.store(out_ptr0 + (x3 + 7008*x2), tmp43, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/uq/cuqcomjtvn6asqw4gkxrd6fd6wjl33tr4ecpvfda3px3a7bsdd2k.py
# Topologically Sorted Source Nodes: [setitem, setitem_1, remainder, add, arange, mul_2, add_1, embedding_1, sum_1, mul_3, linear], Original ATen: [aten.select, aten.lift_fresh, aten.fill, aten.slice, aten.remainder, aten.add, aten.copy, aten.arange, aten.mul, aten.embedding, aten.sum, aten._to_copy]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   arange => iota
#   embedding_1 => embedding_1
#   linear => convert_element_type_2
#   mul_2 => mul_2
#   mul_3 => mul_3
#   remainder => remainder
#   setitem => copy, full_default, select
#   setitem_1 => copy_1, slice_8
#   sum_1 => sum_1
# Graph fragment:
#   %bitwise_xor : Tensor "i64[350, 999, 7][7008, 7, 1]cuda:3" = PlaceHolder[target=bitwise_xor]
#   %arg6_1 : Tensor "bf16[57344, 128][128, 1]cuda:3" = PlaceHolder[target=arg6_1]
#   %sum_1 : Tensor "f32[350, 1000, 128][128000, 128, 1]cuda:3" = PlaceHolder[target=sum_1]
#   %select : Tensor "i64[350, 7][7000, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%empty, 1, 0), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:3, pin_memory: False})
#   %copy : Tensor "i64[350, 7][7000, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select, %full_default), kwargs = {})
#   %select_scatter_default : Tensor "i64[350, 1000, 7][7000, 7, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%empty, %copy, 1, 0), kwargs = {})
#   %slice_8 : Tensor "i64[350, 999, 7][7000, 7, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%select_scatter_default, 1, 1, 9223372036854775807), kwargs = {})
#   %remainder : Tensor "i64[350, 999, 7][6993, 7, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%bitwise_xor, 8191), kwargs = {})
#   %add : Tensor "i64[350, 999, 7][6993, 7, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%remainder, 1), kwargs = {})
#   %copy_1 : Tensor "i64[350, 999, 7][7000, 7, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_8, %add), kwargs = {})
#   %slice_scatter_default : Tensor "i64[350, 1000, 7][7000, 7, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_scatter_default, %copy_1, 1, 1, 9223372036854775807), kwargs = {})
#   %iota : Tensor "i64[7][1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (7,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:3, requires_grad: False})
#   %mul_2 : Tensor "i64[7][1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 8192), kwargs = {})
#   %add_1 : Tensor "i64[350, 1000, 7][7000, 7, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %mul_2), kwargs = {})
#   %embedding_1 : Tensor "bf16[350, 1000, 7, 128][896000, 896, 128, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg6_1, %add_1), kwargs = {})
#   %sum_1 : Tensor "f32[350, 1000, 128][128000, 128, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_1, [-2]), kwargs = {dtype: torch.float32})
#   %mul_3 : Tensor "f32[350, 1000, 128][128000, 128, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 0.37796447300922725), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[350, 1000, 128][128000, 128, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   return %sum_1,%convert_element_type_2
triton_poi_fused__to_copy_add_arange_copy_embedding_fill_lift_fresh_mul_remainder_select_slice_sum_1 = async_compile.triton('triton_poi_fused__to_copy_add_arange_copy_embedding_fill_lift_fresh_mul_remainder_select_slice_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_copy_embedding_fill_lift_fresh_mul_remainder_select_slice_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 179200000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_copy_embedding_fill_lift_fresh_mul_remainder_select_slice_sum_1(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 44800000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 128) % 1000)
    x2 = xindex // 128000
    x0 = (xindex % 128)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-7) + 7*x1 + 7008*x2), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full([1], 8191, tl.int64)
    tmp5 = (tmp3 % tmp4)
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = tmp5 != tmp6
    tmp8 = (libdevice.signbit(tmp5) != 0) if (tmp5).dtype is tl.float32 else tmp5 < 0
    tmp9 = (libdevice.signbit(tmp4) != 0) if (tmp4).dtype is tl.float32 else tmp4 < 0
    tmp10 = tmp8 != tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tmp5 + tmp4
    tmp13 = tl.where(tmp11, tmp12, tmp5)
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = tmp0 == tmp18
    tmp20 = tl.full([1], 0, tl.int64)
    tmp21 = tl.where(tmp19, tmp20, tmp20)
    tmp22 = tl.where(tmp2, tmp17, tmp21)
    tmp23 = tmp22 + tmp20
    tmp24 = tl.full([XBLOCK], 57344, tl.int32)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp23 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp23)
    tl.device_assert(((0 <= tmp27) & (tmp27 < 57344)) | ~(xmask), "index out of bounds: 0 <= tmp27 < 57344")
    tmp29 = tl.load(in_ptr1 + (x0 + 128*tmp27), xmask).to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tl.load(in_ptr0 + ((-6) + 7*x1 + 7008*x2), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = (tmp31 % tmp4)
    tmp33 = tmp32 != tmp6
    tmp34 = (libdevice.signbit(tmp32) != 0) if (tmp32).dtype is tl.float32 else tmp32 < 0
    tmp35 = tmp34 != tmp9
    tmp36 = tmp33 & tmp35
    tmp37 = tmp32 + tmp4
    tmp38 = tl.where(tmp36, tmp37, tmp32)
    tmp39 = tmp38 + tmp14
    tmp40 = tl.full(tmp39.shape, 0, tmp39.dtype)
    tmp41 = tl.where(tmp2, tmp39, tmp40)
    tmp42 = tl.where(tmp2, tmp41, tmp21)
    tmp43 = tl.full([1], 8192, tl.int64)
    tmp44 = tmp42 + tmp43
    tmp45 = tmp44 + tmp24
    tmp46 = tmp44 < 0
    tmp47 = tl.where(tmp46, tmp45, tmp44)
    tl.device_assert(((0 <= tmp47) & (tmp47 < 57344)) | ~(xmask), "index out of bounds: 0 <= tmp47 < 57344")
    tmp49 = tl.load(in_ptr1 + (x0 + 128*tmp47), xmask).to(tl.float32)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp30 + tmp50
    tmp52 = tl.load(in_ptr0 + ((-5) + 7*x1 + 7008*x2), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp53 = (tmp52 % tmp4)
    tmp54 = tmp53 != tmp6
    tmp55 = (libdevice.signbit(tmp53) != 0) if (tmp53).dtype is tl.float32 else tmp53 < 0
    tmp56 = tmp55 != tmp9
    tmp57 = tmp54 & tmp56
    tmp58 = tmp53 + tmp4
    tmp59 = tl.where(tmp57, tmp58, tmp53)
    tmp60 = tmp59 + tmp14
    tmp61 = tl.full(tmp60.shape, 0, tmp60.dtype)
    tmp62 = tl.where(tmp2, tmp60, tmp61)
    tmp63 = tl.where(tmp2, tmp62, tmp21)
    tmp64 = tl.full([1], 16384, tl.int64)
    tmp65 = tmp63 + tmp64
    tmp66 = tmp65 + tmp24
    tmp67 = tmp65 < 0
    tmp68 = tl.where(tmp67, tmp66, tmp65)
    tl.device_assert(((0 <= tmp68) & (tmp68 < 57344)) | ~(xmask), "index out of bounds: 0 <= tmp68 < 57344")
    tmp70 = tl.load(in_ptr1 + (x0 + 128*tmp68), xmask).to(tl.float32)
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tmp51 + tmp71
    tmp73 = tl.load(in_ptr0 + ((-4) + 7*x1 + 7008*x2), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp74 = (tmp73 % tmp4)
    tmp75 = tmp74 != tmp6
    tmp76 = (libdevice.signbit(tmp74) != 0) if (tmp74).dtype is tl.float32 else tmp74 < 0
    tmp77 = tmp76 != tmp9
    tmp78 = tmp75 & tmp77
    tmp79 = tmp74 + tmp4
    tmp80 = tl.where(tmp78, tmp79, tmp74)
    tmp81 = tmp80 + tmp14
    tmp82 = tl.full(tmp81.shape, 0, tmp81.dtype)
    tmp83 = tl.where(tmp2, tmp81, tmp82)
    tmp84 = tl.where(tmp2, tmp83, tmp21)
    tmp85 = tl.full([1], 24576, tl.int64)
    tmp86 = tmp84 + tmp85
    tmp87 = tmp86 + tmp24
    tmp88 = tmp86 < 0
    tmp89 = tl.where(tmp88, tmp87, tmp86)
    tl.device_assert(((0 <= tmp89) & (tmp89 < 57344)) | ~(xmask), "index out of bounds: 0 <= tmp89 < 57344")
    tmp91 = tl.load(in_ptr1 + (x0 + 128*tmp89), xmask).to(tl.float32)
    tmp92 = tmp91.to(tl.float32)
    tmp93 = tmp72 + tmp92
    tmp94 = tl.load(in_ptr0 + ((-3) + 7*x1 + 7008*x2), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp95 = (tmp94 % tmp4)
    tmp96 = tmp95 != tmp6
    tmp97 = (libdevice.signbit(tmp95) != 0) if (tmp95).dtype is tl.float32 else tmp95 < 0
    tmp98 = tmp97 != tmp9
    tmp99 = tmp96 & tmp98
    tmp100 = tmp95 + tmp4
    tmp101 = tl.where(tmp99, tmp100, tmp95)
    tmp102 = tmp101 + tmp14
    tmp103 = tl.full(tmp102.shape, 0, tmp102.dtype)
    tmp104 = tl.where(tmp2, tmp102, tmp103)
    tmp105 = tl.where(tmp2, tmp104, tmp21)
    tmp106 = tl.full([1], 32768, tl.int64)
    tmp107 = tmp105 + tmp106
    tmp108 = tmp107 + tmp24
    tmp109 = tmp107 < 0
    tmp110 = tl.where(tmp109, tmp108, tmp107)
    tl.device_assert(((0 <= tmp110) & (tmp110 < 57344)) | ~(xmask), "index out of bounds: 0 <= tmp110 < 57344")
    tmp112 = tl.load(in_ptr1 + (x0 + 128*tmp110), xmask).to(tl.float32)
    tmp113 = tmp112.to(tl.float32)
    tmp114 = tmp93 + tmp113
    tmp115 = tl.load(in_ptr0 + ((-2) + 7*x1 + 7008*x2), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp116 = (tmp115 % tmp4)
    tmp117 = tmp116 != tmp6
    tmp118 = (libdevice.signbit(tmp116) != 0) if (tmp116).dtype is tl.float32 else tmp116 < 0
    tmp119 = tmp118 != tmp9
    tmp120 = tmp117 & tmp119
    tmp121 = tmp116 + tmp4
    tmp122 = tl.where(tmp120, tmp121, tmp116)
    tmp123 = tmp122 + tmp14
    tmp124 = tl.full(tmp123.shape, 0, tmp123.dtype)
    tmp125 = tl.where(tmp2, tmp123, tmp124)
    tmp126 = tl.where(tmp2, tmp125, tmp21)
    tmp127 = tl.full([1], 40960, tl.int64)
    tmp128 = tmp126 + tmp127
    tmp129 = tmp128 + tmp24
    tmp130 = tmp128 < 0
    tmp131 = tl.where(tmp130, tmp129, tmp128)
    tl.device_assert(((0 <= tmp131) & (tmp131 < 57344)) | ~(xmask), "index out of bounds: 0 <= tmp131 < 57344")
    tmp133 = tl.load(in_ptr1 + (x0 + 128*tmp131), xmask).to(tl.float32)
    tmp134 = tmp133.to(tl.float32)
    tmp135 = tmp114 + tmp134
    tmp136 = tl.load(in_ptr0 + ((-1) + 7*x1 + 7008*x2), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp137 = (tmp136 % tmp4)
    tmp138 = tmp137 != tmp6
    tmp139 = (libdevice.signbit(tmp137) != 0) if (tmp137).dtype is tl.float32 else tmp137 < 0
    tmp140 = tmp139 != tmp9
    tmp141 = tmp138 & tmp140
    tmp142 = tmp137 + tmp4
    tmp143 = tl.where(tmp141, tmp142, tmp137)
    tmp144 = tmp143 + tmp14
    tmp145 = tl.full(tmp144.shape, 0, tmp144.dtype)
    tmp146 = tl.where(tmp2, tmp144, tmp145)
    tmp147 = tl.where(tmp2, tmp146, tmp21)
    tmp148 = tl.full([1], 49152, tl.int64)
    tmp149 = tmp147 + tmp148
    tmp150 = tmp149 + tmp24
    tmp151 = tmp149 < 0
    tmp152 = tl.where(tmp151, tmp150, tmp149)
    tl.device_assert(((0 <= tmp152) & (tmp152 < 57344)) | ~(xmask), "index out of bounds: 0 <= tmp152 < 57344")
    tmp154 = tl.load(in_ptr1 + (x0 + 128*tmp152), xmask).to(tl.float32)
    tmp155 = tmp154.to(tl.float32)
    tmp156 = tmp135 + tmp155
    tmp157 = tl.full([1], 0.37796447300922725, tl.float32)
    tmp158 = tmp156 * tmp157
    tmp159 = tmp158.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp159, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/37/c37qninb2wkmzscuacwmtquqc3epzhblj4vs4d33d2hhxkzgeloo.py
# Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   embedding => embedding
# Graph fragment:
#   %arg4_1 : Tensor "i64[350, 1000][1000, 1]cuda:3" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "bf16[8192, 384][384, 1]cuda:3" = PlaceHolder[target=arg5_1]
#   %embedding : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg5_1, %arg4_1), kwargs = {})
#   return %embedding
triton_poi_fused_embedding_2 = async_compile.triton('triton_poi_fused_embedding_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 537600000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134400000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 384
    x0 = (xindex % 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8192, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 8192)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 8192")
    tmp6 = tl.load(in_ptr1 + (x0 + 384*tmp4), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/xo/cxo2tur5eymysr2oh3gmtrl7ldt2bdwe4s5nk72h743lhyz4lywj.py
# Topologically Sorted Source Nodes: [to_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   to_2 => convert_element_type_5
# Graph fragment:
#   %arg8_1 : Tensor "f32[384][1]cuda:3" = PlaceHolder[target=arg8_1]
#   %convert_element_type_5 : Tensor "bf16[384][1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg8_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_5
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/uh/cuho7wrppogezotrwcmyhpjeevakiyxso3qbg3bxktmyorw3voop.py
# Topologically Sorted Source Nodes: [to_6, sigmoid_1, getitem_8, sub, linear, linear_1, sigmoid, mul_4, to_5, mul_5, add_2, mul_7, zeros_like, getitem_10, cat, ne, to_7, unsqueeze, mul_6, mul_8, add_3, rms_norm, to_8], Original ATen: [aten._to_copy, aten.sigmoid, aten.unsqueeze, aten.rsub, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.zeros_like, aten.slice, aten.cat, aten.ne, aten._fused_rms_norm]
# Source node to ATen node mapping:
#   add_2 => add_2
#   add_3 => add_3
#   cat => cat
#   getitem_10 => slice_11
#   getitem_8 => unsqueeze_2, unsqueeze_3
#   linear => view_1
#   linear_1 => view_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   ne => ne
#   rms_norm => add_4, convert_element_type_13, convert_element_type_14, mean, mul_10, mul_9, pow_1, rsqrt
#   sigmoid => sigmoid
#   sigmoid_1 => sigmoid_1
#   sub => sub
#   to_5 => convert_element_type_9
#   to_6 => convert_element_type_10
#   to_7 => convert_element_type_11
#   to_8 => convert_element_type_12
#   unsqueeze => unsqueeze_4
#   zeros_like => full_default_1
# Graph fragment:
#   %arg11_1 : Tensor "f32[384][1]cuda:3" = PlaceHolder[target=arg11_1]
#   %embedding : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3" = PlaceHolder[target=embedding]
#   %mm : Tensor "bf16[350000, 384][384, 1]cuda:3" = PlaceHolder[target=mm]
#   %addmm : Tensor "bf16[350000, 384][384, 1]cuda:3" = PlaceHolder[target=addmm]
#   %arg10_1 : Tensor "f32[][]cuda:3" = PlaceHolder[target=arg10_1]
#   %arg4_1 : Tensor "i64[350, 1000][1000, 1]cuda:3" = PlaceHolder[target=arg4_1]
#   %add_3 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3" = PlaceHolder[target=add_3]
#   %buf9 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:3" = PlaceHolder[target=buf9]
#   %arg12_1 : Tensor "f32[384][1]cuda:3" = PlaceHolder[target=arg12_1]
#   %convert_element_type_10 : Tensor "bf16[384][1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg11_1, torch.bfloat16), kwargs = {})
#   %sigmoid_1 : Tensor "bf16[384][1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_10,), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[1, 384][384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_1, 0), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 1), kwargs = {})
#   %sub : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %unsqueeze_3), kwargs = {})
#   %view_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [350, 1000, 384]), kwargs = {})
#   %view_3 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [350, 1000, 384]), kwargs = {})
#   %sigmoid : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_3,), kwargs = {})
#   %mul_4 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %sigmoid), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[][]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg10_1, torch.bfloat16), kwargs = {})
#   %mul_5 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %convert_element_type_9), kwargs = {})
#   %add_2 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %mul_5), kwargs = {})
#   %mul_7 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %add_2), kwargs = {})
#   %full_default_1 : Tensor "bf16[350, 1, 384][384, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([350, 1, 384], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:3, pin_memory: False})
#   %slice_11 : Tensor "bf16[350, 999, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_2, 1, 0, -1), kwargs = {})
#   %cat : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%full_default_1, %slice_11], 1), kwargs = {})
#   %ne : Tensor "b8[350, 1000][1000, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%arg4_1, 1), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[350, 1000][1000, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ne, torch.bfloat16), kwargs = {})
#   %unsqueeze_4 : Tensor "bf16[350, 1000, 1][1000, 1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_11, -1), kwargs = {})
#   %mul_6 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_4), kwargs = {})
#   %mul_8 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %mul_6), kwargs = {})
#   %add_3 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %mul_8), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_13, 2), kwargs = {})
#   %mean : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add_4 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_9 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_13, %rsqrt), kwargs = {})
#   %convert_element_type_12 : Tensor "bf16[384][1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg12_1, torch.bfloat16), kwargs = {})
#   %mul_10 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %convert_element_type_12), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
#   return %add_3,%buf9,%convert_element_type_14
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_4 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 11, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2800000, 'r0_': 2688003072}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 1000)
    x1 = xindex // 1000
    tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr1 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [1, 1])
    tmp41 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr6 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tl.full([1, 1], 1.0, tl.float32)
    tmp4 = tmp3 - tmp2
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp6 * tmp8
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp9 * tmp12
    tmp14 = tmp5 + tmp13
    tmp15 = tmp4 * tmp14
    tmp16 = x0
    tmp17 = tl.full([1, 1], 0, tl.int64)
    tmp18 = tmp16 >= tmp17
    tmp19 = tl.full([1, 1], 1, tl.int64)
    tmp20 = tmp16 < tmp19
    tmp21 = tl.full([1, 1], 0.0, tl.float32)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp20, tmp21, tmp22)
    tmp24 = tmp16 >= tmp19
    tmp25 = tl.full([1, 1], 1000, tl.int64)
    tmp26 = tmp16 < tmp25
    tmp27 = tl.load(in_ptr1 + (r0_2 + 384*((-1) + x0) + 384000*x1), r0_mask & tmp24 & xmask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr2 + (r0_2 + 384*((-1) + x0) + 384000*x1), r0_mask & tmp24 & xmask, other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr3 + (r0_2 + 384*((-1) + x0) + 384000*x1), r0_mask & tmp24 & xmask, other=0.0).to(tl.float32)
    tmp30 = tl.sigmoid(tmp29)
    tmp31 = tmp28 * tmp30
    tmp32 = tl.load(in_ptr4 + (0))
    tmp33 = tl.broadcast_to(tmp32, [1, 1])
    tmp34 = tl.where(tmp24, tmp33, 0.0)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp31 * tmp35
    tmp37 = tmp27 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp24, tmp37, tmp38)
    tmp40 = tl.where(tmp20, tmp23, tmp39)
    tmp42 = tmp41 != tmp19
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp40 * tmp43
    tmp45 = tmp2 * tmp44
    tmp46 = tmp15 + tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp47 * tmp47
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, R0_BLOCK])
    tmp51 = tl.where(r0_mask & xmask, tmp49, 0)
    tmp52 = tl.sum(tmp51, 1)[:, None].to(tl.float32)
    tmp53 = tl.full([1, 1], 384.0, tl.float32)
    tmp54 = (tmp52 / tmp53)
    tmp55 = tl.full([1, 1], 1e-06, tl.float32)
    tmp56 = tmp54 + tmp55
    tmp57 = libdevice.rsqrt(tmp56)
    tmp58 = tmp47 * tmp57
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tmp58 * tmp61
    tmp63 = tmp62.to(tl.float32)
    tl.store(out_ptr0 + (r0_2 + 384*x3), tmp46, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_2 + 384*x3), tmp63, r0_mask & xmask)
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
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_fused_qkv_postprocess_fwd_kernel_0', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'QKV': '*bf16', 'Q': '*bf16', 'K': '*bf16', 'COS': '*bf16', 'SIN': '*bf16', 'TOTAL_ROWS': 'constexpr', 'SEQLEN': 'constexpr', 'H_Q': 'constexpr', 'H_KV': 'constexpr', 'HEAD_DIM': 'constexpr', 'ROPE_DIMS': 'constexpr', 'DO_QK_NORM': 'constexpr', 'ROPE_INTERLEAVED': 'constexpr', 'RMS_EPS': 'constexpr', 'BLOCK_D': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'TOTAL_ROWS': 350000, 'SEQLEN': 1000, 'H_Q': 8, 'H_KV': 4, 'HEAD_DIM': 48, 'ROPE_DIMS': 48, 'DO_QK_NORM': True, 'ROPE_INTERLEAVED': False, 'RMS_EPS': 1e-06, 'BLOCK_D': 64}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/tw/ctwyn237tpjk2vygczixgqu3skwip2n4s4edhqxtfnlwq6iwu3oi.py
# Topologically Sorted Source Nodes: [linear_2, view, scaled_dot_product_attention, getitem_11, contiguous, transpose_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
# Source node to ATen node mapping:
#   contiguous => clone
#   getitem_11 => slice_12
#   linear_2 => view_5
#   scaled_dot_product_attention => _scaled_dot_product_flash_attention, permute_6, permute_7
#   transpose_2 => permute_5
#   view => view_6
# Graph fragment:
#   %mm_1 : Tensor "bf16[350000, 768][768, 1]cuda:3" = PlaceHolder[target=mm_1]
#   %view_5 : Tensor "bf16[350, 1000, 768][768000, 768, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [350, 1000, 768]), kwargs = {})
#   %view_6 : Tensor "bf16[350, 1000, 16, 48][768000, 768, 48, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [350, 1000, 16, 48]), kwargs = {})
#   %permute_6 : Tensor "bf16[350, 8, 1000, 48][384000, 48, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty_1, [0, 2, 1, 3]), kwargs = {})
#   %permute_7 : Tensor "bf16[350, 4, 1000, 48][192000, 48, 192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty_2, [0, 2, 1, 3]), kwargs = {})
#   %slice_12 : Tensor "bf16[350, 1000, 4, 48][768000, 768, 48, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_6, 2, 12, 9223372036854775807), kwargs = {})
#   %clone : Tensor "bf16[350, 1000, 4, 48][192000, 192, 48, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_12,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_5 : Tensor "bf16[350, 4, 1000, 48][192000, 48, 192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%clone, [0, 2, 1, 3]), kwargs = {})
#   %_scaled_dot_product_flash_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%permute_6, %permute_7, %permute_5, 0.0, True), kwargs = {scale: 0.14433756729740646})
#   return %buf16
triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_5 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 403200000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67200000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = xindex // 192
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (576 + x0 + 768*x1), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ko/cko6i3tkwq6y4xcebn67han25j5rfajsudlvcgjf3icrsyeuqzr7.py
# Topologically Sorted Source Nodes: [getitem_14, contiguous_6], Original ATen: [aten.slice, aten.clone]
# Source node to ATen node mapping:
#   contiguous_6 => clone_1
#   getitem_14 => slice_13
# Graph fragment:
#   %convert_element_type_14 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3" = PlaceHolder[target=convert_element_type_14]
#   %slice_13 : Tensor "bf16[350, 1000, 12][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_14, 2, 0, 12), kwargs = {})
#   %clone_1 : Tensor "bf16[350, 1000, 12][12000, 12, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_13,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused_clone_slice_6 = async_compile.triton('triton_poi_fused_clone_slice_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_slice_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 25200000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_slice_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/d7/cd7bcipyundfuofvgoousvmc3d7qbo5dwywkddk6minjji5tjgs7.py
# Topologically Sorted Source Nodes: [getitem_14, contiguous_6, linear_3, to_10], Original ATen: [aten.slice, aten.clone, aten.view, aten._to_copy, aten.t, aten.mm]
# Source node to ATen node mapping:
#   contiguous_6 => clone_1
#   getitem_14 => slice_13
#   linear_3 => mm_2, permute_9, view_7
#   to_10 => convert_element_type_17
# Graph fragment:
#   %clone_1 : Tensor "bf16[350, 1000, 12][12032, 12, 1]cuda:3" = PlaceHolder[target=clone_1]
#   %slice_13 : Tensor "bf16[350, 1000, 12][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_14, 2, 0, 12), kwargs = {})
#   %clone_1 : Tensor "bf16[350, 1000, 12][12000, 12, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_13,), kwargs = {memory_format: torch.contiguous_format})
#   %view_7 : Tensor "bf16[350000, 12][12, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_1, [350000, 12]), kwargs = {})
#   %convert_element_type_17 : Tensor "bf16[8, 12][12, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg14_1, torch.bfloat16), kwargs = {})
#   %permute_9 : Tensor "bf16[12, 8][1, 12]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_17, [1, 0]), kwargs = {})
#   %mm_2 : Tensor "bf16[350000, 8][8, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_7, %permute_9), kwargs = {})
#   return %buf24
triton_poi_fused__to_copy_clone_mm_slice_t_view_7 = async_compile.triton('triton_poi_fused__to_copy_clone_mm_slice_t_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_mm_slice_t_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 25200000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_mm_slice_t_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/fg/cfgrbacx7xne7zgcxplf4wvrnwu5t4pdqqtlj5jcj5qe7caymdut.py
# Topologically Sorted Source Nodes: [to_10], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   to_10 => convert_element_type_17
# Graph fragment:
#   %arg14_1 : Tensor "f32[8, 12][12, 1]cuda:3" = PlaceHolder[target=arg14_1]
#   %convert_element_type_17 : Tensor "bf16[8, 12][12, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg14_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_17
triton_poi_fused__to_copy_8 = async_compile.triton('triton_poi_fused__to_copy_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/h6/ch6fcmvirdbcnj4x27k3kxori4krcuexd7khs6sssnt6fhh2ncnp.py
# Topologically Sorted Source Nodes: [transpose_3, linear_3, mul_9, sigmoid_2, getitem_15, mul_10], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
# Source node to ATen node mapping:
#   getitem_15 => unsqueeze_5
#   linear_3 => view_8
#   mul_10 => mul_12
#   mul_9 => mul_11
#   sigmoid_2 => sigmoid_2
#   transpose_3 => permute_8
# Graph fragment:
#   %getitem_2 : Tensor "bf16[350, 8, 1000, 48][384000, 48, 384, 1]cuda:3" = PlaceHolder[target=getitem_2]
#   %mm_2 : Tensor "bf16[350000, 8][8, 1]cuda:3" = PlaceHolder[target=mm_2]
#   %permute_8 : Tensor "bf16[350, 1000, 8, 48][384000, 384, 48, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [0, 2, 1, 3]), kwargs = {})
#   %view_8 : Tensor "bf16[350, 1000, 8][8000, 8, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [350, 1000, 8]), kwargs = {})
#   %mul_11 : Tensor "bf16[350, 1000, 8][8000, 8, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, 0.5), kwargs = {})
#   %sigmoid_2 : Tensor "bf16[350, 1000, 8][8000, 8, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_11,), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[350, 1000, 8, 1][8000, 8, 1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_2, 3), kwargs = {})
#   %mul_12 : Tensor "bf16[350, 1000, 8, 48][384000, 384, 48, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_8, %unsqueeze_5), kwargs = {})
#   return %mul_12
triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_9 = async_compile.triton('triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 806400000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/vl/cvln47wn5xcaqr4q7sambl42njo5g3iiqyrg4bjlb6tk57sz4r4r.py
# Topologically Sorted Source Nodes: [linear_4, add_4, rms_norm_1, to_12], Original ATen: [aten._unsafe_view, aten.add, aten._fused_rms_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add_4 => add_5
#   linear_4 => view_11
#   rms_norm_1 => add_6, convert_element_type_23, convert_element_type_24, mean_1, mul_13, mul_14, pow_2, rsqrt_1
#   to_12 => convert_element_type_22
# Graph fragment:
#   %mm_3 : Tensor "bf16[350000, 384][384, 1]cuda:3" = PlaceHolder[target=mm_3]
#   %add_3 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3" = PlaceHolder[target=add_3]
#   %add_5 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3" = PlaceHolder[target=add_5]
#   %buf30 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:3" = PlaceHolder[target=buf30]
#   %arg16_1 : Tensor "f32[384][1]cuda:3" = PlaceHolder[target=arg16_1]
#   %view_11 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [350, 1000, 384]), kwargs = {})
#   %add_5 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %add_3), kwargs = {})
#   %convert_element_type_23 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_23, 2), kwargs = {})
#   %mean_1 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_6 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_13 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_23, %rsqrt_1), kwargs = {})
#   %convert_element_type_22 : Tensor "bf16[384][1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg16_1, torch.bfloat16), kwargs = {})
#   %mul_14 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %convert_element_type_22), kwargs = {})
#   %convert_element_type_24 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14, torch.bfloat16), kwargs = {})
#   return %add_5,%buf30,%convert_element_type_24
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_10 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 1612801536}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_10(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp15 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
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
    tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp2, r0_mask & xmask)
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp19, r0_mask & xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg1_1, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg2_1, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg3_1, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg4_1, (350, 1000), (1000, 1))
        assert_size_stride(arg5_1, (8192, 384), (384, 1))
        assert_size_stride(arg6_1, (57344, 128), (128, 1))
        assert_size_stride(arg7_1, (384, 128), (128, 1))
        assert_size_stride(arg8_1, (384, ), (1, ))
        assert_size_stride(arg9_1, (384, 384), (384, 1))
        assert_size_stride(arg10_1, (), ())
        assert_size_stride(arg11_1, (384, ), (1, ))
        assert_size_stride(arg12_1, (384, ), (1, ))
        assert_size_stride(arg13_1, (768, 384), (384, 1))
        assert_size_stride(arg14_1, (8, 12), (12, 1))
        assert_size_stride(arg15_1, (384, 384), (384, 1))
        assert_size_stride(arg16_1, (384, ), (1, ))
        assert_size_stride(arg17_1, (1536, 384), (384, 1))
        with torch.cuda._DeviceGuard(3):
            torch.cuda.set_device(3)
            buf1 = empty_strided_cuda((350, 999, 7), (7008, 7, 1), torch.int64)
            # Topologically Sorted Source Nodes: [getitem_4, getitem_6, tensor, mul, getitem_5, getitem_7, tensor_1, mul_1, bitwise_xor], Original ATen: [aten.slice, aten.unsqueeze, aten.lift_fresh, aten.mul, aten.bitwise_xor]
            stream3 = get_raw_stream(3)
            triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0.run(arg4_1, buf1, 2447550, stream=stream3)
            buf3 = empty_strided_cuda((350, 1000, 128), (128000, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [setitem, setitem_1, remainder, add, arange, mul_2, add_1, embedding_1, sum_1, mul_3, linear], Original ATen: [aten.select, aten.lift_fresh, aten.fill, aten.slice, aten.remainder, aten.add, aten.copy, aten.arange, aten.mul, aten.embedding, aten.sum, aten._to_copy]
            stream3 = get_raw_stream(3)
            triton_poi_fused__to_copy_add_arange_copy_embedding_fill_lift_fresh_mul_remainder_select_slice_sum_1.run(buf1, arg6_1, buf3, 44800000, stream=stream3)
            del arg6_1
            del buf1
            buf4 = empty_strided_cuda((350000, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_3, linear], Original ATen: [aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (350000, 128), (128, 1), 0), reinterpret_tensor(arg7_1, (128, 384), (1, 128), 0), out=buf4)
            del arg7_1
            del buf3
            buf5 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
            stream3 = get_raw_stream(3)
            triton_poi_fused_embedding_2.run(arg4_1, arg5_1, buf5, 134400000, stream=stream3)
            del arg5_1
            buf6 = empty_strided_cuda((384, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_2], Original ATen: [aten._to_copy]
            stream3 = get_raw_stream(3)
            triton_poi_fused__to_copy_3.run(arg8_1, buf6, 384, stream=stream3)
            del arg8_1
            buf7 = empty_strided_cuda((350000, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_2, embedding, linear_1], Original ATen: [aten._to_copy, aten.embedding, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(buf6, reinterpret_tensor(buf5, (350000, 384), (384, 1), 0), reinterpret_tensor(arg9_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf7)
            del arg9_1
            del buf6
            buf8 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            buf10 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_6, sigmoid_1, getitem_8, sub, linear, linear_1, sigmoid, mul_4, to_5, mul_5, add_2, mul_7, zeros_like, getitem_10, cat, ne, to_7, unsqueeze, mul_6, mul_8, add_3, rms_norm, to_8], Original ATen: [aten._to_copy, aten.sigmoid, aten.unsqueeze, aten.rsub, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.zeros_like, aten.slice, aten.cat, aten.ne, aten._fused_rms_norm]
            stream3 = get_raw_stream(3)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_4.run(arg11_1, buf5, buf4, buf7, arg10_1, arg4_1, arg12_1, buf8, buf10, 350000, 384, stream=stream3)
            del arg10_1
            del arg11_1
            del arg12_1
            del arg4_1
            del buf4
            del buf5
            buf11 = empty_strided_cuda((350000, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm, to_8, linear_2], Original ATen: [aten._fused_rms_norm, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf10, (350000, 384), (384, 1), 0), reinterpret_tensor(arg13_1, (384, 768), (1, 384), 0), out=buf11)
            del arg13_1
            buf12 = reinterpret_tensor(buf7, (350, 1000, 8, 48), (384000, 384, 48, 1), 0); del buf7  # reuse
            buf13 = empty_strided_cuda((350, 1000, 4, 48), (192000, 192, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_2, view, getitem, getitem_12, getitem_1, getitem_13, triton_kernel_wrapper_mutation], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.select]
            stream3 = get_raw_stream(3)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf11, (350, 1000, 16, 48), (768000, 768, 48, 1), 0), buf12, buf13, reinterpret_tensor(arg0_1, (1000, 24), (24, 1), 0), reinterpret_tensor(arg1_1, (1000, 24), (24, 1), 0), 4200000, 1, 1, stream=stream3)
            buf16 = empty_strided_cuda((350, 4, 1000, 48), (192000, 48, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_2, view, scaled_dot_product_attention, getitem_11, contiguous, transpose_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
            stream3 = get_raw_stream(3)
            triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_5.run(buf11, buf16, 67200000, stream=stream3)
            del buf11
            # Topologically Sorted Source Nodes: [linear_2, view, scaled_dot_product_attention, getitem_11, contiguous, transpose_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
            buf17 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf12, (350, 8, 1000, 48), (384000, 48, 384, 1), 0), reinterpret_tensor(buf13, (350, 4, 1000, 48), (192000, 48, 192, 1), 0), buf16, 0.0, True, scale=0.14433756729740646)
            del buf12
            del buf13
            del buf16
            buf18 = buf17[0]
            assert_size_stride(buf18, (350, 8, 1000, 48), (384000, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf18, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf17
            buf23 = empty_strided_cuda((350, 1000, 12), (12032, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_14, contiguous_6], Original ATen: [aten.slice, aten.clone]
            stream3 = get_raw_stream(3)
            triton_poi_fused_clone_slice_6.run(buf10, buf23, 4200000, stream=stream3)
            buf24 = empty_strided_cuda((350000, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_14, contiguous_6, linear_3, to_10], Original ATen: [aten.slice, aten.clone, aten.view, aten._to_copy, aten.t, aten.mm]
            stream3 = get_raw_stream(3)
            triton_poi_fused__to_copy_clone_mm_slice_t_view_7.run(buf23, buf24, 4200000, stream=stream3)
            del buf23
            buf25 = empty_strided_cuda((8, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_10], Original ATen: [aten._to_copy]
            stream3 = get_raw_stream(3)
            triton_poi_fused__to_copy_8.run(arg14_1, buf25, 96, stream=stream3)
            del arg14_1
            buf26 = empty_strided_cuda((350000, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_14, contiguous_6, linear_3, to_10], Original ATen: [aten.slice, aten.clone, aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf24, reinterpret_tensor(buf25, (12, 8), (1, 12), 0), out=buf26)
            del buf24
            del buf25
            buf27 = reinterpret_tensor(buf18, (350, 1000, 8, 48), (384000, 384, 48, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, linear_3, mul_9, sigmoid_2, getitem_15, mul_10], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
            stream3 = get_raw_stream(3)
            triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_9.run(buf27, buf26, 134400000, stream=stream3)
            del buf26
            buf28 = reinterpret_tensor(buf10, (350000, 384), (384, 1), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, linear_3, mul_9, sigmoid_2, getitem_15, mul_10, reshape, linear_4], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf27, (350000, 384), (384, 1), 0), reinterpret_tensor(arg15_1, (384, 384), (1, 384), 0), out=buf28)
            del arg15_1
            buf29 = reinterpret_tensor(buf28, (350, 1000, 384), (384000, 384, 1), 0); del buf28  # reuse
            buf31 = reinterpret_tensor(buf27, (350, 1000, 384), (384000, 384, 1), 0); del buf27  # reuse
            # Topologically Sorted Source Nodes: [linear_4, add_4, rms_norm_1, to_12], Original ATen: [aten._unsafe_view, aten.add, aten._fused_rms_norm, aten._to_copy]
            stream3 = get_raw_stream(3)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_10.run(buf29, buf8, arg16_1, buf31, 350000, 384, stream=stream3)
            del arg16_1
        return (buf31, arg17_1, buf29, reinterpret_tensor(arg2_1, (1, 1000, 1, 24), (24576, 24, 24, 1), 0), reinterpret_tensor(arg3_1, (1, 1000, 1, 24), (24576, 24, 24, 1), 0), reinterpret_tensor(arg0_1, (1, 1000, 1, 24), (24576, 24, 24, 1), 0), reinterpret_tensor(arg1_1, (1, 1000, 1, 24), (24576, 24, 24, 1), 0), buf8, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:3', dtype=torch.bfloat16)
    arg1_1 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:3', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:3', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:3', dtype=torch.bfloat16)
    arg4_1 = rand_strided((350, 1000), (1000, 1), device='cuda:3', dtype=torch.int64)
    arg5_1 = rand_strided((8192, 384), (384, 1), device='cuda:3', dtype=torch.bfloat16)
    arg6_1 = rand_strided((57344, 128), (128, 1), device='cuda:3', dtype=torch.bfloat16)
    arg7_1 = rand_strided((384, 128), (128, 1), device='cuda:3', dtype=torch.bfloat16)
    arg8_1 = rand_strided((384, ), (1, ), device='cuda:3', dtype=torch.float32)
    arg9_1 = rand_strided((384, 384), (384, 1), device='cuda:3', dtype=torch.bfloat16)
    arg10_1 = rand_strided((), (), device='cuda:3', dtype=torch.float32)
    arg11_1 = rand_strided((384, ), (1, ), device='cuda:3', dtype=torch.float32)
    arg12_1 = rand_strided((384, ), (1, ), device='cuda:3', dtype=torch.float32)
    arg13_1 = rand_strided((768, 384), (384, 1), device='cuda:3', dtype=torch.bfloat16)
    arg14_1 = rand_strided((8, 12), (12, 1), device='cuda:3', dtype=torch.float32)
    arg15_1 = rand_strided((384, 384), (384, 1), device='cuda:3', dtype=torch.bfloat16)
    arg16_1 = rand_strided((384, ), (1, ), device='cuda:3', dtype=torch.float32)
    arg17_1 = rand_strided((1536, 384), (384, 1), device='cuda:3', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
