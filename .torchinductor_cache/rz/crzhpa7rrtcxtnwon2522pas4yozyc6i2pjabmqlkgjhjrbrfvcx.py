# AOT ID: ['24_inference']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/nc/cnc6sjv2nydbdsq4jjk4b4ksjuyzclthytkfy4qy5q2op5k4wyf5.py
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
#   %lift_fresh_copy : Tensor "i64[8][1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant0,), kwargs = {})
#   %mul : Tensor "i64[350, 999, 8][7992, 8, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %lift_fresh_copy), kwargs = {})
#   %slice_6 : Tensor "i64[350, 999][1000, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg4_1, 1, 0, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[350, 999, 1][1000, 1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_6, 2), kwargs = {})
#   %lift_fresh_copy_1 : Tensor "i64[8][1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant1,), kwargs = {})
#   %mul_1 : Tensor "i64[350, 999, 8][7992, 8, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %lift_fresh_copy_1), kwargs = {})
#   %bitwise_xor : Tensor "i64[350, 999, 8][7992, 8, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.bitwise_xor.Tensor](args = (%mul, %mul_1), kwargs = {})
#   return %bitwise_xor
triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0 = async_compile.triton('triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 5594400, 'x': 44755200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 349650
    xnumel = 8
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = (yindex % 999)
    y1 = yindex // 999
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + y0 + 1000*y1), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (y0 + 1000*y1), ymask, eviction_policy='evict_last')
    tmp1 = x2
    tmp2 = tl.full([1, 1], 4, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.full([1, 1], 2, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.full([1, 1], 1, tl.int64)
    tmp7 = tmp1 < tmp6
    tmp8 = tl.full([1, 1], 36313, tl.int64)
    tmp9 = tl.full([1, 1], 17491, tl.int64)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tl.full([1, 1], 3, tl.int64)
    tmp12 = tmp1 < tmp11
    tmp13 = tl.full([1, 1], 52973, tl.int64)
    tmp14 = tl.full([1, 1], 29837, tl.int64)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tl.where(tmp5, tmp10, tmp15)
    tmp17 = tl.full([1, 1], 6, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tl.full([1, 1], 5, tl.int64)
    tmp20 = tmp1 < tmp19
    tmp21 = tl.full([1, 1], 44497, tl.int64)
    tmp22 = tl.full([1, 1], 62137, tl.int64)
    tmp23 = tl.where(tmp20, tmp21, tmp22)
    tmp24 = tl.full([1, 1], 7, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.full([1, 1], 24071, tl.int64)
    tmp27 = tl.full([1, 1], 57223, tl.int64)
    tmp28 = tl.where(tmp25, tmp26, tmp27)
    tmp29 = tl.where(tmp18, tmp23, tmp28)
    tmp30 = tl.where(tmp3, tmp16, tmp29)
    tmp31 = tmp0 * tmp30
    tmp33 = tl.full([1, 1], 27191, tl.int64)
    tmp34 = tl.full([1, 1], 43889, tl.int64)
    tmp35 = tl.where(tmp7, tmp33, tmp34)
    tmp36 = tl.full([1, 1], 19937, tl.int64)
    tmp37 = tl.full([1, 1], 60271, tl.int64)
    tmp38 = tl.where(tmp12, tmp36, tmp37)
    tmp39 = tl.where(tmp5, tmp35, tmp38)
    tmp40 = tl.full([1, 1], 34583, tl.int64)
    tmp41 = tl.full([1, 1], 49331, tl.int64)
    tmp42 = tl.where(tmp20, tmp40, tmp41)
    tmp43 = tl.full([1, 1], 15461, tl.int64)
    tmp44 = tl.full([1, 1], 41077, tl.int64)
    tmp45 = tl.where(tmp25, tmp43, tmp44)
    tmp46 = tl.where(tmp18, tmp42, tmp45)
    tmp47 = tl.where(tmp3, tmp39, tmp46)
    tmp48 = tmp32 * tmp47
    tmp49 = tmp31 ^ tmp48
    tl.store(out_ptr0 + (x2 + 8*y0 + 8000*y1), tmp49, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/pv/cpvrswkqdeq3jm6ejwmz3i3umca3dss2ofuc7mr5jtkkefc5ao5g.py
# Topologically Sorted Source Nodes: [setitem, setitem_1, remainder, add, arange, mul_2, add_1], Original ATen: [aten.select, aten.lift_fresh, aten.fill, aten.slice, aten.remainder, aten.add, aten.copy, aten.arange, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   arange => iota
#   mul_2 => mul_2
#   remainder => remainder
#   setitem => copy, full_default, select
#   setitem_1 => copy_1, slice_8
# Graph fragment:
#   %bitwise_xor : Tensor "i64[350, 999, 8][8000, 8, 1]cuda:3" = PlaceHolder[target=bitwise_xor]
#   %select : Tensor "i64[350, 8][8000, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%empty, 1, 0), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:3, pin_memory: False})
#   %copy : Tensor "i64[350, 8][8000, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select, %full_default), kwargs = {})
#   %select_scatter_default : Tensor "i64[350, 1000, 8][8000, 8, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%empty, %copy, 1, 0), kwargs = {})
#   %slice_8 : Tensor "i64[350, 999, 8][8000, 8, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%select_scatter_default, 1, 1, 9223372036854775807), kwargs = {})
#   %remainder : Tensor "i64[350, 999, 8][7992, 8, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%bitwise_xor, 16383), kwargs = {})
#   %add : Tensor "i64[350, 999, 8][7992, 8, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%remainder, 1), kwargs = {})
#   %copy_1 : Tensor "i64[350, 999, 8][8000, 8, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_8, %add), kwargs = {})
#   %slice_scatter_default : Tensor "i64[350, 1000, 8][8000, 8, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_scatter_default, %copy_1, 1, 1, 9223372036854775807), kwargs = {})
#   %iota : Tensor "i64[8][1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:3, requires_grad: False})
#   %mul_2 : Tensor "i64[8][1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 16384), kwargs = {})
#   %add_1 : Tensor "i64[350, 1000, 8][8000, 8, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %mul_2), kwargs = {})
#   return %add_1
triton_poi_fused_add_arange_copy_fill_lift_fresh_mul_remainder_select_slice_1 = async_compile.triton('triton_poi_fused_add_arange_copy_fill_lift_fresh_mul_remainder_select_slice_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_arange_copy_fill_lift_fresh_mul_remainder_select_slice_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 67177600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_copy_fill_lift_fresh_mul_remainder_select_slice_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2800000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 1000)
    x3 = xindex
    x0 = (xindex % 8)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-8) + x3), tmp2 & xmask, other=0.0)
    tmp4 = tl.full([1], 16383, tl.int64)
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
    tmp23 = 16384*x0
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr0 + (x3), tmp24, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/37/c37qninb2wkmzscuacwmtquqc3epzhblj4vs4d33d2hhxkzgeloo.py
# Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   embedding => embedding
# Graph fragment:
#   %arg4_1 : Tensor "i64[350, 1000][1000, 1]cuda:3" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "bf16[8192, 384][384, 1]cuda:3" = PlaceHolder[target=arg5_1]
#   %embedding : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg5_1, %arg4_1), kwargs = {})
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg1_1, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg2_1, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg3_1, (1, 1024, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg4_1, (350, 1000), (1000, 1))
        assert_size_stride(arg5_1, (8192, 384), (384, 1))
        with torch.cuda._DeviceGuard(3):
            torch.cuda.set_device(3)
            buf1 = empty_strided_cuda((350, 999, 8), (8000, 8, 1), torch.int64)
            # Topologically Sorted Source Nodes: [getitem_4, getitem_6, tensor, mul, getitem_5, getitem_7, tensor_1, mul_1, bitwise_xor], Original ATen: [aten.slice, aten.unsqueeze, aten.lift_fresh, aten.mul, aten.bitwise_xor]
            stream3 = get_raw_stream(3)
            triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0.run(arg4_1, buf1, 349650, 8, stream=stream3)
            buf2 = empty_strided_cuda((350, 1000, 8), (8000, 8, 1), torch.int64)
            # Topologically Sorted Source Nodes: [setitem, setitem_1, remainder, add, arange, mul_2, add_1], Original ATen: [aten.select, aten.lift_fresh, aten.fill, aten.slice, aten.remainder, aten.add, aten.copy, aten.arange, aten.mul]
            stream3 = get_raw_stream(3)
            triton_poi_fused_add_arange_copy_fill_lift_fresh_mul_remainder_select_slice_1.run(buf1, buf2, 2800000, stream=stream3)
            del buf1
            buf3 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
            stream3 = get_raw_stream(3)
            triton_poi_fused_embedding_2.run(arg4_1, arg5_1, buf3, 134400000, stream=stream3)
            del arg4_1
            del arg5_1
        return (buf2, buf3, reinterpret_tensor(arg0_1, (1, 1000, 1, 24), (24576, 24, 24, 1), 0), reinterpret_tensor(arg1_1, (1, 1000, 1, 24), (24576, 24, 24, 1), 0), reinterpret_tensor(arg2_1, (1, 1000, 1, 24), (24576, 24, 24, 1), 0), reinterpret_tensor(arg3_1, (1, 1000, 1, 24), (24576, 24, 24, 1), 0), )

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
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
