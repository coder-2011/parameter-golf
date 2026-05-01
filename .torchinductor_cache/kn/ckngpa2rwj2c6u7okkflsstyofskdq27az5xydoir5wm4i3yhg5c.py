# AOT ID: ['0_forward']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/tl/ctliy2xd3ahtfwidx5qtsy3eqorliujmcygotenq4xk23jx7dmin.py
# Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   embedding => embedding
# Graph fragment:
#   %primals_5 : Tensor "i64[128, 1024][1024, 1]cuda:4" = PlaceHolder[target=primals_5]
#   %primals_6 : Tensor "bf16[8192, 384][384, 1]cuda:4" = PlaceHolder[target=primals_6]
#   %embedding : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_6, %primals_5), kwargs = {})
#   return %embedding
triton_poi_fused_embedding_0 = async_compile.triton('triton_poi_fused_embedding_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1048576, 'x': 201326592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 384
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 8192, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 8192)) | ~(ymask), "index out of bounds: 0 <= tmp4 < 8192")
    tmp6 = tl.load(in_ptr1 + (x1 + 384*tmp4), xmask & ymask).to(tl.float32)
    tl.store(out_ptr0 + (x1 + 384*y0), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ox/coxw2ljrafpmnoejcje2a7kkh3hrb2v7up2dkqkmkoin7tlq45uz.py
# Topologically Sorted Source Nodes: [getitem_4, getitem_5, tensor, tensor_1, getitem_6, mul, getitem_7, mul_1, bitwise_xor], Original ATen: [aten.slice, aten.lift_fresh, aten.unsqueeze, aten.mul, aten.bitwise_xor]
# Source node to ATen node mapping:
#   bitwise_xor => bitwise_xor
#   getitem_4 => slice_1
#   getitem_5 => slice_2
#   getitem_6 => unsqueeze
#   getitem_7 => unsqueeze_1
#   mul => mul
#   mul_1 => mul_1
#   tensor => lift_fresh_copy
#   tensor_1 => lift_fresh_copy_1
# Graph fragment:
#   %primals_5 : Tensor "i64[128, 1024][1024, 1]cuda:4" = PlaceHolder[target=primals_5]
#   %slice_1 : Tensor "i64[128, 1023][1024, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_5, 1, 1, 9223372036854775807), kwargs = {})
#   %slice_2 : Tensor "i64[128, 1023][1024, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_5, 1, 0, -1), kwargs = {})
#   %lift_fresh_copy : Tensor "i64[7][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant0,), kwargs = {})
#   %lift_fresh_copy_1 : Tensor "i64[7][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant1,), kwargs = {})
#   %unsqueeze : Tensor "i64[128, 1023, 1][1024, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_1, 2), kwargs = {})
#   %mul : Tensor "i64[128, 1023, 7][7161, 7, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %lift_fresh_copy), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[128, 1023, 1][1024, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_2, 2), kwargs = {})
#   %mul_1 : Tensor "i64[128, 1023, 7][7161, 7, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %lift_fresh_copy_1), kwargs = {})
#   %bitwise_xor : Tensor "i64[128, 1023, 7][7161, 7, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.bitwise_xor.Tensor](args = (%mul, %mul_1), kwargs = {})
#   return %bitwise_xor
triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_1 = async_compile.triton('triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 14665728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 916608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 7) % 1023)
    x2 = xindex // 7161
    x0 = (xindex % 7)
    x3 = (xindex % 7161)
    tmp0 = tl.load(in_ptr0 + (1 + x1 + 1024*x2), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (x1 + 1024*x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + 7168*x2), tmp43, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/fx/cfxm3ezserghh72qh6zzxpi4qqmukurozsrjhssztcip6nwn6a7n.py
# Topologically Sorted Source Nodes: [setitem, remainder, add, setitem_1, arange, mul_2, add_1], Original ATen: [aten.lift_fresh, aten.select, aten.fill, aten.remainder, aten.add, aten.slice, aten.copy, aten.arange, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   arange => iota
#   mul_2 => mul_2
#   remainder => remainder
#   setitem => copy, full_default, select
#   setitem_1 => copy_1, slice_4
# Graph fragment:
#   %bitwise_xor : Tensor "i64[128, 1023, 7][7168, 7, 1]cuda:4" = PlaceHolder[target=bitwise_xor]
#   %full_default : Tensor "i64[][]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:4, pin_memory: False})
#   %select : Tensor "i64[128, 7][7168, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%empty, 1, 0), kwargs = {})
#   %copy : Tensor "i64[128, 7][7168, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select, %full_default), kwargs = {})
#   %select_scatter_default : Tensor "i64[128, 1024, 7][7168, 7, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%empty, %copy, 1, 0), kwargs = {})
#   %remainder : Tensor "i64[128, 1023, 7][7161, 7, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%bitwise_xor, 8191), kwargs = {})
#   %add : Tensor "i64[128, 1023, 7][7161, 7, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%remainder, 1), kwargs = {})
#   %slice_4 : Tensor "i64[128, 1023, 7][7168, 7, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%select_scatter_default, 1, 1, 9223372036854775807), kwargs = {})
#   %copy_1 : Tensor "i64[128, 1023, 7][7168, 7, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_4, %add), kwargs = {})
#   %slice_scatter_default : Tensor "i64[128, 1024, 7][7168, 7, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_scatter_default, %copy_1, 1, 1, 9223372036854775807), kwargs = {})
#   %iota : Tensor "i64[7][1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (7,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:4, requires_grad: False})
#   %mul_2 : Tensor "i64[7][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 8192), kwargs = {})
#   %add_1 : Tensor "i64[128, 1024, 7][7168, 7, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %mul_2), kwargs = {})
#   return %add_1
triton_poi_fused_add_arange_copy_fill_lift_fresh_mul_remainder_select_slice_2 = async_compile.triton('triton_poi_fused_add_arange_copy_fill_lift_fresh_mul_remainder_select_slice_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_arange_copy_fill_lift_fresh_mul_remainder_select_slice_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 22012928}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_copy_fill_lift_fresh_mul_remainder_select_slice_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 917504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 7) % 1024)
    x3 = xindex
    x0 = (xindex % 7)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-7) + x3), tmp2, other=0.0)
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
    tmp23 = 8192*x0
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr0 + (x3), tmp24, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/5a/c5asre6lzcnq42rz33vbjglcjb67zlrimcg2o7z73f77lcj6inw6.py
# Topologically Sorted Source Nodes: [embedding_1, sum_1, mul_3, linear], Original ATen: [aten.embedding, aten.sum, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   embedding_1 => embedding_1
#   linear => convert_element_type_2
#   mul_3 => mul_3
#   sum_1 => sum_1
# Graph fragment:
#   %add_1 : Tensor "i64[128, 1024, 7][7168, 7, 1]cuda:4" = PlaceHolder[target=add_1]
#   %primals_7 : Tensor "bf16[57344, 128][128, 1]cuda:4" = PlaceHolder[target=primals_7]
#   %sum_1 : Tensor "f32[128, 1024, 128][131072, 128, 1]cuda:4" = PlaceHolder[target=sum_1]
#   %embedding_1 : Tensor "bf16[128, 1024, 7, 128][917504, 896, 128, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_7, %add_1), kwargs = {})
#   %sum_1 : Tensor "f32[128, 1024, 128][131072, 128, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_1, [-2]), kwargs = {dtype: torch.float32})
#   %mul_3 : Tensor "f32[128, 1024, 128][131072, 128, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 0.37796447300922725), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[128, 1024, 128][131072, 128, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   return %sum_1,%convert_element_type_2
triton_poi_fused__to_copy_embedding_mul_sum_3 = async_compile.triton('triton_poi_fused__to_copy_embedding_mul_sum_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_embedding_mul_sum_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 67108864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_embedding_mul_sum_3(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 128
    x0 = (xindex % 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (7*x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 7*x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (2 + 7*x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (3 + 7*x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (4 + 7*x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (5 + 7*x1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr0 + (6 + 7*x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 57344, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 57344), "index out of bounds: 0 <= tmp4 < 57344")
    tmp6 = tl.load(in_ptr1 + (x0 + 128*tmp4), None).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp8 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp8)
    tl.device_assert((0 <= tmp11) & (tmp11 < 57344), "index out of bounds: 0 <= tmp11 < 57344")
    tmp13 = tl.load(in_ptr1 + (x0 + 128*tmp11), None).to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp7 + tmp14
    tmp17 = tmp16 + tmp1
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tl.device_assert((0 <= tmp19) & (tmp19 < 57344), "index out of bounds: 0 <= tmp19 < 57344")
    tmp21 = tl.load(in_ptr1 + (x0 + 128*tmp19), None).to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp15 + tmp22
    tmp25 = tmp24 + tmp1
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tl.device_assert((0 <= tmp27) & (tmp27 < 57344), "index out of bounds: 0 <= tmp27 < 57344")
    tmp29 = tl.load(in_ptr1 + (x0 + 128*tmp27), None).to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp23 + tmp30
    tmp33 = tmp32 + tmp1
    tmp34 = tmp32 < 0
    tmp35 = tl.where(tmp34, tmp33, tmp32)
    tl.device_assert((0 <= tmp35) & (tmp35 < 57344), "index out of bounds: 0 <= tmp35 < 57344")
    tmp37 = tl.load(in_ptr1 + (x0 + 128*tmp35), None).to(tl.float32)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp31 + tmp38
    tmp41 = tmp40 + tmp1
    tmp42 = tmp40 < 0
    tmp43 = tl.where(tmp42, tmp41, tmp40)
    tl.device_assert((0 <= tmp43) & (tmp43 < 57344), "index out of bounds: 0 <= tmp43 < 57344")
    tmp45 = tl.load(in_ptr1 + (x0 + 128*tmp43), None).to(tl.float32)
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp39 + tmp46
    tmp49 = tmp48 + tmp1
    tmp50 = tmp48 < 0
    tmp51 = tl.where(tmp50, tmp49, tmp48)
    tl.device_assert((0 <= tmp51) & (tmp51 < 57344), "index out of bounds: 0 <= tmp51 < 57344")
    tmp53 = tl.load(in_ptr1 + (x0 + 128*tmp51), None).to(tl.float32)
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp47 + tmp54
    tmp56 = tl.full([1], 0.37796447300922725, tl.float32)
    tmp57 = tmp55 * tmp56
    tmp58 = tmp57.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp58, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/hh/chhzeysbyxfslgf77ztgurabb3hpluxjmwyltbs7owakq4gomzzp.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear => convert_element_type_default_7, permute
# Graph fragment:
#   %primals_8 : Tensor "bf16[384, 128][128, 1]cuda:4" = PlaceHolder[target=primals_8]
#   %convert_element_type_default_7 : Tensor "bf16[384, 128][128, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.bfloat16), kwargs = {})
#   %permute : Tensor "bf16[128, 384][1, 128]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_default_7, [1, 0]), kwargs = {})
#   return %permute
triton_poi_fused__to_copy_t_4 = async_compile.triton('triton_poi_fused__to_copy_t_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 294912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/7g/c7g72gggjpw7jioyaxoljl2gkb32y5kccmkfuvngmt7nwmgpbr7p.py
# Topologically Sorted Source Nodes: [to_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   to_2 => convert_element_type_5
# Graph fragment:
#   %primals_9 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_9]
#   %convert_element_type_5 : Tensor "bf16[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_9, torch.bfloat16), kwargs = {})
#   return %convert_element_type_5
triton_poi_fused__to_copy_5 = async_compile.triton('triton_poi_fused__to_copy_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/gx/cgxbqskpy3wxq5qn43okrmt7ch5zkebs4f4i7ada45onafshuol5.py
# Topologically Sorted Source Nodes: [linear, linear_1, sigmoid, mul_4, to_5, mul_5, add_2, to_6, sigmoid_1, getitem_8, zeros_like, getitem_10, cat, ne, to_7, unsqueeze, mul_6, sub, mul_7, mul_8, add_3, to_8, rms_norm], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten._to_copy, aten.add, aten.unsqueeze, aten.zeros_like, aten.slice, aten.cat, aten.ne, aten.rsub, aten._fused_rms_norm]
# Source node to ATen node mapping:
#   add_2 => add_2
#   add_3 => add_3
#   cat => cat
#   getitem_10 => slice_7
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
#   %primals_5 : Tensor "i64[128, 1024][1024, 1]cuda:4" = PlaceHolder[target=primals_5]
#   %embedding : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=embedding]
#   %mm : Tensor "bf16[131072, 384][384, 1]cuda:4" = PlaceHolder[target=mm]
#   %addmm : Tensor "bf16[131072, 384][384, 1]cuda:4" = PlaceHolder[target=addmm]
#   %primals_11 : Tensor "f32[][]cuda:4" = PlaceHolder[target=primals_11]
#   %primals_12 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_12]
#   %cat : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=cat]
#   %ne : Tensor "b8[128, 1024][1024, 1]cuda:4" = PlaceHolder[target=ne]
#   %add_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=add_3]
#   %buf13 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:4" = PlaceHolder[target=buf13]
#   %rsqrt : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4" = PlaceHolder[target=rsqrt]
#   %primals_13 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_13]
#   %view_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 1024, 384]), kwargs = {})
#   %view_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [128, 1024, 384]), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_3,), kwargs = {})
#   %mul_4 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %sigmoid), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[][]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_11, torch.bfloat16), kwargs = {})
#   %mul_5 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %convert_element_type_9), kwargs = {})
#   %add_2 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %mul_5), kwargs = {})
#   %convert_element_type_10 : Tensor "bf16[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_12, torch.bfloat16), kwargs = {})
#   %sigmoid_1 : Tensor "bf16[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_10,), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[1, 384][384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_1, 0), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 1), kwargs = {})
#   %full_default_1 : Tensor "bf16[128, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 1, 384], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:4, pin_memory: False})
#   %slice_7 : Tensor "bf16[128, 1023, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_2, 1, 0, -1), kwargs = {})
#   %cat : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%full_default_1, %slice_7], 1), kwargs = {})
#   %ne : Tensor "b8[128, 1024][1024, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%primals_5, 1), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[128, 1024][1024, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ne, torch.bfloat16), kwargs = {})
#   %unsqueeze_4 : Tensor "bf16[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_11, -1), kwargs = {})
#   %mul_6 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_4), kwargs = {})
#   %sub : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %unsqueeze_3), kwargs = {})
#   %mul_7 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %add_2), kwargs = {})
#   %mul_8 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %mul_6), kwargs = {})
#   %add_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %mul_8), kwargs = {})
#   %convert_element_type_12 : Tensor "bf16[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_13, torch.bfloat16), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_13, 2), kwargs = {})
#   %mean : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add_4 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_9 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_13, %rsqrt), kwargs = {})
#   %mul_10 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %convert_element_type_12), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
#   return %ne,%cat,%add_3,%buf13,%rsqrt,%convert_element_type_14
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_6 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 11, 'num_store': 5, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2359296, 'r0_': 1207962624}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x0 = xindex
    x1 = (xindex % 1024)
    r0_3 = r0_index
    x2 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr1 + (r0_3 + 384*x0), r0_mask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr2 + (r0_3 + 384*x0), r0_mask, other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr3 + (r0_3 + 384*x0), r0_mask, other=0.0).to(tl.float32)
    tmp37 = tl.load(in_ptr4 + (0))
    tmp38 = tl.broadcast_to(tmp37, [1, 1])
    tmp59 = tl.load(in_ptr6 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = x1
    tmp4 = tl.full([1, 1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp3 < tmp1
    tmp7 = tl.full([1, 1], 0.0, tl.float32)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tmp3 >= tmp1
    tmp11 = tl.full([1, 1], 1024, tl.int64)
    tmp12 = tmp3 < tmp11
    tmp13 = tl.load(in_ptr1 + (r0_3 + 384*((-1) + x1) + 393216*x2), r0_mask & tmp10, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (r0_3 + 384*((-1) + x1) + 393216*x2), r0_mask & tmp10, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (r0_3 + 384*((-1) + x1) + 393216*x2), r0_mask & tmp10, other=0.0).to(tl.float32)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp14 * tmp16
    tmp18 = tl.load(in_ptr4 + (0))
    tmp19 = tl.broadcast_to(tmp18, [1, 1])
    tmp20 = tl.where(tmp10, tmp19, 0.0)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp17 * tmp21
    tmp23 = tmp13 + tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tl.where(tmp6, tmp9, tmp25)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tl.full([1, 1], 1.0, tl.float32)
    tmp31 = tmp30 - tmp29
    tmp35 = tl.sigmoid(tmp34)
    tmp36 = tmp33 * tmp35
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp36 * tmp39
    tmp41 = tmp32 + tmp40
    tmp42 = tmp31 * tmp41
    tmp43 = tmp2.to(tl.float32)
    tmp44 = tmp26 * tmp43
    tmp45 = tmp29 * tmp44
    tmp46 = tmp42 + tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp47 * tmp47
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, R0_BLOCK])
    tmp51 = tl.where(r0_mask, tmp49, 0)
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
    tl.store(out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr1 + (r0_3 + 384*x0), tmp26, r0_mask)
    tl.store(out_ptr2 + (r0_3 + 384*x0), tmp46, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp57, None)
    tl.store(out_ptr3 + (r0_3 + 384*x0), tmp63, r0_mask)
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
    triton_meta={'signature': {'QKV': '*bf16', 'Q': '*bf16', 'K': '*bf16', 'COS': '*bf16', 'SIN': '*bf16', 'TOTAL_ROWS': 'constexpr', 'SEQLEN': 'constexpr', 'H_Q': 'constexpr', 'H_KV': 'constexpr', 'HEAD_DIM': 'constexpr', 'ROPE_DIMS': 'constexpr', 'DO_QK_NORM': 'constexpr', 'ROPE_INTERLEAVED': 'constexpr', 'RMS_EPS': 'constexpr', 'BLOCK_D': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'TOTAL_ROWS': 131072, 'SEQLEN': 1024, 'H_Q': 8, 'H_KV': 4, 'HEAD_DIM': 48, 'ROPE_DIMS': 48, 'DO_QK_NORM': True, 'ROPE_INTERLEAVED': False, 'RMS_EPS': 1e-06, 'BLOCK_D': 64}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/xr/cxrhrfq2iyt73wj3svsfypighiv4pluyni6dgaa5odblrvxxecbu.py
# Topologically Sorted Source Nodes: [linear_2, view, getitem, v], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone]
# Source node to ATen node mapping:
#   getitem => slice_8
#   linear_2 => view_5
#   v => clone
#   view => view_6
# Graph fragment:
#   %mm_1 : Tensor "bf16[131072, 768][768, 1]cuda:4" = PlaceHolder[target=mm_1]
#   %view_5 : Tensor "bf16[128, 1024, 768][786432, 768, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [128, 1024, 768]), kwargs = {})
#   %view_6 : Tensor "bf16[128, 1024, 16, 48][786432, 768, 48, 1]cuda:4"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [128, 1024, 16, 48]), kwargs = {})
#   %slice_8 : Tensor "bf16[128, 1024, 4, 48][786432, 768, 48, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_6, 2, 12, 9223372036854775807), kwargs = {})
#   %clone : Tensor "bf16[128, 1024, 4, 48][196608, 192, 48, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_8,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused__unsafe_view_clone_slice_view_7 = async_compile.triton('triton_poi_fused__unsafe_view_clone_slice_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_slice_view_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 150994944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_slice_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 192)
    x1 = xindex // 192
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (576 + x0 + 768*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/fu/cfuz7wif2buplgejh33dt7gqzbxqnnr6qyu7otresfko2jdvnsif.py
# Topologically Sorted Source Nodes: [to_10, linear_3], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_3 => permute_9
#   to_10 => convert_element_type_17
# Graph fragment:
#   %primals_15 : Tensor "f32[8, 12][12, 1]cuda:4" = PlaceHolder[target=primals_15]
#   %convert_element_type_17 : Tensor "bf16[8, 12][12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_15, torch.bfloat16), kwargs = {})
#   %permute_9 : Tensor "bf16[12, 8][1, 12]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_17, [1, 0]), kwargs = {})
#   return %permute_9
triton_poi_fused__to_copy_t_8 = async_compile.triton('triton_poi_fused__to_copy_t_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/xs/cxsd62elafl4ad2ylxqj7z2nvnaml5226qbtt76qkzdl4lzi3vnq.py
# Topologically Sorted Source Nodes: [getitem_14, contiguous_3], Original ATen: [aten.slice, aten.clone]
# Source node to ATen node mapping:
#   contiguous_3 => clone_1
#   getitem_14 => slice_9
# Graph fragment:
#   %convert_element_type_14 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=convert_element_type_14]
#   %slice_9 : Tensor "bf16[128, 1024, 12][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_14, 2, 0, 12), kwargs = {})
#   %clone_1 : Tensor "bf16[128, 1024, 12][12288, 12, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_9,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused_clone_slice_9 = async_compile.triton('triton_poi_fused_clone_slice_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_slice_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_slice_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/5w/c5whisqqdlwonxru6b6q2nvocuti3f4w4nh7u2jxua3cspfcfwqo.py
# Topologically Sorted Source Nodes: [transpose_3, linear_3, mul_9, sigmoid_2, getitem_15, mul_10], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
# Source node to ATen node mapping:
#   getitem_15 => unsqueeze_5
#   linear_3 => view_8
#   mul_10 => mul_12
#   mul_9 => mul_11
#   sigmoid_2 => sigmoid_2
#   transpose_3 => permute_8
# Graph fragment:
#   %getitem_2 : Tensor "bf16[128, 8, 1024, 48][393216, 48, 384, 1]cuda:4" = PlaceHolder[target=getitem_2]
#   %mm_2 : Tensor "bf16[131072, 8][8, 1]cuda:4" = PlaceHolder[target=mm_2]
#   %permute_8 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [0, 2, 1, 3]), kwargs = {})
#   %view_8 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [128, 1024, 8]), kwargs = {})
#   %mul_11 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, 0.5), kwargs = {})
#   %sigmoid_2 : Tensor "bf16[128, 1024, 8][8192, 8, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_11,), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[128, 1024, 8, 1][8192, 8, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_2, 3), kwargs = {})
#   %mul_12 : Tensor "bf16[128, 1024, 8, 48][393216, 384, 48, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_8, %unsqueeze_5), kwargs = {})
#   return %mul_12
triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_10 = async_compile.triton('triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 301989888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/dk/cdklxkufcon3fvzaonnprkl4u5rwnlksqddeecxnyyyucihb72dy.py
# Topologically Sorted Source Nodes: [linear_4, add_4, to_12, rms_norm_1], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten._fused_rms_norm]
# Source node to ATen node mapping:
#   add_4 => add_5
#   linear_4 => view_11
#   rms_norm_1 => add_6, convert_element_type_23, convert_element_type_24, mean_1, mul_13, mul_14, pow_2, rsqrt_1
#   to_12 => convert_element_type_22
# Graph fragment:
#   %mm_3 : Tensor "bf16[131072, 384][384, 1]cuda:4" = PlaceHolder[target=mm_3]
#   %add_3 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=add_3]
#   %add_5 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4" = PlaceHolder[target=add_5]
#   %buf34 : Tensor "f32[128, 1024, 1][1024, 1, 131072]cuda:4" = PlaceHolder[target=buf34]
#   %rsqrt_1 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4" = PlaceHolder[target=rsqrt_1]
#   %primals_17 : Tensor "f32[384][1]cuda:4" = PlaceHolder[target=primals_17]
#   %view_11 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [128, 1024, 384]), kwargs = {})
#   %add_5 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %add_3), kwargs = {})
#   %convert_element_type_22 : Tensor "bf16[384][1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_17, torch.bfloat16), kwargs = {})
#   %convert_element_type_23 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_23, 2), kwargs = {})
#   %mean_1 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_6 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[128, 1024, 1][1024, 1, 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_13 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_23, %rsqrt_1), kwargs = {})
#   %mul_14 : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %convert_element_type_22), kwargs = {})
#   %convert_element_type_24 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14, torch.bfloat16), kwargs = {})
#   return %add_5,%buf34,%rsqrt_1,%convert_element_type_24
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_11 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_11', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1048576, 'r0_': 603981312}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18 = args
        args.clear()
        assert_size_stride(primals_1, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(primals_2, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(primals_3, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(primals_4, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(primals_5, (128, 1024), (1024, 1))
        assert_size_stride(primals_6, (8192, 384), (384, 1))
        assert_size_stride(primals_7, (57344, 128), (128, 1))
        assert_size_stride(primals_8, (384, 128), (128, 1))
        assert_size_stride(primals_9, (384, ), (1, ))
        assert_size_stride(primals_10, (384, 384), (384, 1))
        assert_size_stride(primals_11, (), ())
        assert_size_stride(primals_12, (384, ), (1, ))
        assert_size_stride(primals_13, (384, ), (1, ))
        assert_size_stride(primals_14, (768, 384), (384, 1))
        assert_size_stride(primals_15, (8, 12), (12, 1))
        assert_size_stride(primals_16, (384, 384), (384, 1))
        assert_size_stride(primals_17, (384, ), (1, ))
        assert_size_stride(primals_18, (1536, 384), (384, 1))
        with torch.cuda._DeviceGuard(4):
            torch.cuda.set_device(4)
            buf0 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
            stream4 = get_raw_stream(4)
            triton_poi_fused_embedding_0.run(primals_5, primals_6, buf0, 131072, 384, stream=stream4)
            del primals_6
            buf2 = empty_strided_cuda((128, 1023, 7), (7168, 7, 1), torch.int64)
            # Topologically Sorted Source Nodes: [getitem_4, getitem_5, tensor, tensor_1, getitem_6, mul, getitem_7, mul_1, bitwise_xor], Original ATen: [aten.slice, aten.lift_fresh, aten.unsqueeze, aten.mul, aten.bitwise_xor]
            stream4 = get_raw_stream(4)
            triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_1.run(primals_5, buf2, 916608, stream=stream4)
            buf3 = empty_strided_cuda((128, 1024, 7), (7168, 7, 1), torch.int64)
            # Topologically Sorted Source Nodes: [setitem, remainder, add, setitem_1, arange, mul_2, add_1], Original ATen: [aten.lift_fresh, aten.select, aten.fill, aten.remainder, aten.add, aten.slice, aten.copy, aten.arange, aten.mul]
            stream4 = get_raw_stream(4)
            triton_poi_fused_add_arange_copy_fill_lift_fresh_mul_remainder_select_slice_2.run(buf2, buf3, 917504, stream=stream4)
            del buf2
            buf6 = empty_strided_cuda((128, 1024, 128), (131072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [embedding_1, sum_1, mul_3, linear], Original ATen: [aten.embedding, aten.sum, aten.mul, aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_embedding_mul_sum_3.run(buf3, primals_7, buf6, 16777216, stream=stream4)
            del primals_7
            buf5 = empty_strided_cuda((128, 384), (1, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_t_4.run(primals_8, buf5, 49152, stream=stream4)
            del primals_8
            buf7 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_3, linear], Original ATen: [aten.mul, aten._to_copy, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf6, (131072, 128), (128, 1), 0), buf5, out=buf7)
            buf8 = empty_strided_cuda((384, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_2], Original ATen: [aten._to_copy]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_5.run(primals_9, buf8, 384, stream=stream4)
            del primals_9
            buf9 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_2, linear_1], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(buf8, reinterpret_tensor(buf0, (131072, 384), (384, 1), 0), reinterpret_tensor(primals_10, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf9)
            del buf8
            buf11 = empty_strided_cuda((128, 1024), (1024, 1), torch.bool)
            buf10 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            buf12 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            buf13 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
            buf14 = reinterpret_tensor(buf13, (128, 1024, 1), (1024, 1, 1), 0); del buf13  # reuse
            buf15 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, linear_1, sigmoid, mul_4, to_5, mul_5, add_2, to_6, sigmoid_1, getitem_8, zeros_like, getitem_10, cat, ne, to_7, unsqueeze, mul_6, sub, mul_7, mul_8, add_3, to_8, rms_norm], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten._to_copy, aten.add, aten.unsqueeze, aten.zeros_like, aten.slice, aten.cat, aten.ne, aten.rsub, aten._fused_rms_norm]
            stream4 = get_raw_stream(4)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_6.run(buf14, primals_5, buf0, buf7, buf9, primals_11, primals_12, primals_13, buf11, buf10, buf12, buf15, 131072, 384, stream=stream4)
            buf16 = empty_strided_cuda((131072, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_8, rms_norm, linear_2], Original ATen: [aten._to_copy, aten._fused_rms_norm, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf15, (131072, 384), (384, 1), 0), reinterpret_tensor(primals_14, (384, 768), (1, 384), 0), out=buf16)
            buf17 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            buf18 = empty_strided_cuda((128, 1024, 4, 48), (196608, 192, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_2, view, getitem_1, getitem_2, triton_kernel_wrapper_mutation], Original ATen: [aten._unsafe_view, aten.view, aten.select]
            stream4 = get_raw_stream(4)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf16, (128, 1024, 16, 48), (786432, 768, 48, 1), 0), buf17, buf18, reinterpret_tensor(primals_1, (1024, 24), (24, 1), 0), reinterpret_tensor(primals_2, (1024, 24), (24, 1), 0), 1572864, 1, 1, stream=stream4)
            buf21 = empty_strided_cuda((128, 1024, 4, 48), (196608, 192, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_2, view, getitem, v], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone]
            stream4 = get_raw_stream(4)
            triton_poi_fused__unsafe_view_clone_slice_view_7.run(buf16, buf21, 25165824, stream=stream4)
            # Topologically Sorted Source Nodes: [linear_2, view, getitem, v, transpose_2, scaled_dot_product_attention], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.clone, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf22 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf17, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf18, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), reinterpret_tensor(buf21, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), 0.0, True, scale=0.14433756729740646)
            buf23 = buf22[0]
            assert_size_stride(buf23, (128, 8, 1024, 48), (393216, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf23, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf24 = buf22[1]
            assert_size_stride(buf24, (128, 8, 1024), (8192, 1024, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf24, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf25 = buf22[6]
            assert_size_stride(buf25, (2, ), (1, ), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf25, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            buf26 = buf22[7]
            assert_size_stride(buf26, (), (), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf26, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf22
            buf28 = empty_strided_cuda((12, 8), (1, 12), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_10, linear_3], Original ATen: [aten._to_copy, aten.t]
            stream4 = get_raw_stream(4)
            triton_poi_fused__to_copy_t_8.run(primals_15, buf28, 96, stream=stream4)
            del primals_15
            buf29 = empty_strided_cuda((128, 1024, 12), (12288, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_14, contiguous_3], Original ATen: [aten.slice, aten.clone]
            stream4 = get_raw_stream(4)
            triton_poi_fused_clone_slice_9.run(buf15, buf29, 1572864, stream=stream4)
            buf30 = empty_strided_cuda((131072, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_14, contiguous_3, linear_3], Original ATen: [aten.slice, aten.clone, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf29, (131072, 12), (12, 1), 0), buf28, out=buf30)
            buf31 = empty_strided_cuda((128, 1024, 8, 48), (393216, 384, 48, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_3, linear_3, mul_9, sigmoid_2, getitem_15, mul_10], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
            stream4 = get_raw_stream(4)
            triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_10.run(buf23, buf30, buf31, 50331648, stream=stream4)
            buf32 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_3, linear_3, mul_9, sigmoid_2, getitem_15, mul_10, reshape, linear_4], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf31, (131072, 384), (384, 1), 0), reinterpret_tensor(primals_16, (384, 384), (1, 384), 0), out=buf32)
            buf33 = reinterpret_tensor(buf32, (128, 1024, 384), (393216, 384, 1), 0); del buf32  # reuse
            buf34 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
            buf35 = reinterpret_tensor(buf34, (128, 1024, 1), (1024, 1, 1), 0); del buf34  # reuse
            buf36 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_4, add_4, to_12, rms_norm_1], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten._fused_rms_norm]
            stream4 = get_raw_stream(4)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_11.run(buf33, buf35, buf12, primals_17, buf36, 131072, 384, stream=stream4)
        return (buf36, primals_18, buf33, primals_3, primals_4, primals_1, primals_2, buf12, primals_1, primals_2, primals_5, primals_10, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, buf0, buf3, reinterpret_tensor(buf6, (131072, 128), (128, 1), 0), buf7, buf9, buf10, buf11, buf12, buf14, reinterpret_tensor(buf15, (131072, 384), (384, 1), 0), reinterpret_tensor(buf16, (128, 1024, 16, 48), (786432, 768, 48, 1), 0), reinterpret_tensor(buf21, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), reinterpret_tensor(buf17, (128, 8, 1024, 48), (393216, 48, 384, 1), 0), reinterpret_tensor(buf18, (128, 4, 1024, 48), (196608, 48, 192, 1), 0), buf23, buf24, buf25, buf26, reinterpret_tensor(buf29, (131072, 12), (12, 1), 0), buf30, reinterpret_tensor(buf31, (131072, 384), (384, 1), 0), buf33, buf35, reinterpret_tensor(buf28, (8, 12), (12, 1), 0), reinterpret_tensor(buf5, (384, 128), (128, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_2 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_3 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_4 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_5 = rand_strided((128, 1024), (1024, 1), device='cuda:4', dtype=torch.int64)
    primals_6 = rand_strided((8192, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_7 = rand_strided((57344, 128), (128, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_8 = rand_strided((384, 128), (128, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_9 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_10 = rand_strided((384, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_11 = rand_strided((), (), device='cuda:4', dtype=torch.float32)
    primals_12 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_13 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_14 = rand_strided((768, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_15 = rand_strided((8, 12), (12, 1), device='cuda:4', dtype=torch.float32)
    primals_16 = rand_strided((384, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    primals_17 = rand_strided((384, ), (1, ), device='cuda:4', dtype=torch.float32)
    primals_18 = rand_strided((1536, 384), (384, 1), device='cuda:4', dtype=torch.bfloat16)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
