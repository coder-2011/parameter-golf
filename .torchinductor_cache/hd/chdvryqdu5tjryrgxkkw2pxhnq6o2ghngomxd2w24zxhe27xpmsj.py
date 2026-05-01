# AOT ID: ['12_inference']
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
_tensor_constant0 = None  # device(type='cuda', index=2) torch.int64 (12,) (1,) 7b42592dbde0
_tensor_constant1 = None  # device(type='cuda', index=2) torch.int64 (12,) (1,) 7b42592dfd40


# kernel path: /workspace/parameter-golf/.torchinductor_cache/vp/cvptctmepwgkxdzhpdjz52eolwidwiv6j4drbqxgvstto7uad5kf.py
# Topologically Sorted Source Nodes: [setitem, setitem_1, getitem_4, getitem_6, tensor, mul, getitem_5, getitem_7, tensor_1, mul_1, bitwise_xor, remainder, add, arange, mul_2, add_1], Original ATen: [aten.select, aten.lift_fresh, aten.fill, aten.slice, aten.unsqueeze, aten.mul, aten.bitwise_xor, aten.remainder, aten.add, aten.copy, aten.arange]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   arange => iota
#   bitwise_xor => bitwise_xor
#   getitem_4 => slice_5
#   getitem_5 => slice_6
#   getitem_6 => unsqueeze
#   getitem_7 => unsqueeze_1
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   remainder => remainder
#   setitem => copy, full_default, select
#   setitem_1 => copy_1, slice_8
#   tensor => lift_fresh_copy
#   tensor_1 => lift_fresh_copy_1
# Graph fragment:
#   %arg4_1 : Tensor "i64[512, 1000][1000, 1]cuda:2" = PlaceHolder[target=arg4_1]
#   %_tensor_constant0 : Tensor "i64[12][1]cuda:2" = PlaceHolder[target=_tensor_constant0]
#   %_tensor_constant1 : Tensor "i64[12][1]cuda:2" = PlaceHolder[target=_tensor_constant1]
#   %select : Tensor "i64[512, 12][12000, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%empty, 1, 0), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:2, pin_memory: False})
#   %copy : Tensor "i64[512, 12][12000, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select, %full_default), kwargs = {})
#   %select_scatter_default : Tensor "i64[512, 1000, 12][12000, 12, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%empty, %copy, 1, 0), kwargs = {})
#   %slice_8 : Tensor "i64[512, 999, 12][12000, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%select_scatter_default, 1, 1, 9223372036854775807), kwargs = {})
#   %slice_5 : Tensor "i64[512, 999][1000, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg4_1, 1, 1, 9223372036854775807), kwargs = {})
#   %unsqueeze : Tensor "i64[512, 999, 1][1000, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_5, 2), kwargs = {})
#   %lift_fresh_copy : Tensor "i64[12][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant0,), kwargs = {})
#   %mul : Tensor "i64[512, 999, 12][11988, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %lift_fresh_copy), kwargs = {})
#   %slice_6 : Tensor "i64[512, 999][1000, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg4_1, 1, 0, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[512, 999, 1][1000, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_6, 2), kwargs = {})
#   %lift_fresh_copy_1 : Tensor "i64[12][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant1,), kwargs = {})
#   %mul_1 : Tensor "i64[512, 999, 12][11988, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %lift_fresh_copy_1), kwargs = {})
#   %bitwise_xor : Tensor "i64[512, 999, 12][11988, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.bitwise_xor.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %remainder : Tensor "i64[512, 999, 12][11988, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%bitwise_xor, 32767), kwargs = {})
#   %add : Tensor "i64[512, 999, 12][11988, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%remainder, 1), kwargs = {})
#   %copy_1 : Tensor "i64[512, 999, 12][12000, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_8, %add), kwargs = {})
#   %slice_scatter_default : Tensor "i64[512, 1000, 12][12000, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_scatter_default, %copy_1, 1, 1, 9223372036854775807), kwargs = {})
#   %iota : Tensor "i64[12][1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (12,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:2, requires_grad: False})
#   %mul_2 : Tensor "i64[12][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 32768), kwargs = {})
#   %add_1 : Tensor "i64[512, 1000, 12][12000, 12, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %mul_2), kwargs = {})
#   return %add_1
triton_poi_fused_add_arange_bitwise_xor_copy_fill_lift_fresh_mul_remainder_select_slice_unsqueeze_0 = async_compile.triton('triton_poi_fused_add_arange_bitwise_xor_copy_fill_lift_fresh_mul_remainder_select_slice_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'out_ptr0': '*i64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_arange_bitwise_xor_copy_fill_lift_fresh_mul_remainder_select_slice_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 8192000, 'x': 98304192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_bitwise_xor_copy_fill_lift_fresh_mul_remainder_select_slice_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512000
    xnumel = 12
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = (yindex % 1000)
    y3 = yindex
    x2 = xindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (tl.broadcast_to(y3, [YBLOCK, XBLOCK])), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x2, [YBLOCK, XBLOCK])), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + y3, [YBLOCK, XBLOCK])), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(x2, [YBLOCK, XBLOCK])), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 ^ tmp8
    tmp10 = tl.full([1, 1], 32767, tl.int64)
    tmp11 = (tmp9 % tmp10)
    tmp12 = tl.full([1, 1], 0, tl.int32)
    tmp13 = tmp11 != tmp12
    tmp14 = (libdevice.signbit(tmp11) != 0) if (tmp11).dtype is tl.float32 else tmp11 < 0
    tmp15 = (libdevice.signbit(tmp10) != 0) if (tmp10).dtype is tl.float32 else tmp10 < 0
    tmp16 = tmp14 != tmp15
    tmp17 = tmp13 & tmp16
    tmp18 = tmp11 + tmp10
    tmp19 = tl.where(tmp17, tmp18, tmp11)
    tmp20 = tl.full([1, 1], 1, tl.int64)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
    tmp23 = tl.where(tmp2, tmp21, tmp22)
    tmp24 = tl.full([1, 1], 0, tl.int32)
    tmp25 = tmp0 == tmp24
    tmp26 = tl.full([1, 1], 0, tl.int64)
    tmp27 = tl.where(tmp25, tmp26, tmp26)
    tmp28 = tl.where(tmp2, tmp23, tmp27)
    tmp29 = 32768*x2
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr0 + (x2 + 12*y3), tmp30, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ax/caxy6wprjuaqqsqy3myv2ll4nhddvodctm3ehwp3m4vpnnelas4p.py
# Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   embedding => embedding
# Graph fragment:
#   %arg4_1 : Tensor "i64[512, 1000][1000, 1]cuda:2" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "bf16[8192, 384][384, 1]cuda:2" = PlaceHolder[target=arg5_1]
#   %embedding : Tensor "bf16[512, 1000, 384][384000, 384, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg5_1, %arg4_1), kwargs = {})
#   return %embedding
triton_poi_fused_embedding_1 = async_compile.triton('triton_poi_fused_embedding_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4096000, 'x': 786432000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_1(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512000
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
        assert_size_stride(arg4_1, (512, 1000), (1000, 1))
        assert_size_stride(arg5_1, (8192, 384), (384, 1))
        with torch.cuda._DeviceGuard(2):
            torch.cuda.set_device(2)
            buf1 = empty_strided_cuda((512, 1000, 12), (12000, 12, 1), torch.int64)
            # Topologically Sorted Source Nodes: [setitem, setitem_1, getitem_4, getitem_6, tensor, mul, getitem_5, getitem_7, tensor_1, mul_1, bitwise_xor, remainder, add, arange, mul_2, add_1], Original ATen: [aten.select, aten.lift_fresh, aten.fill, aten.slice, aten.unsqueeze, aten.mul, aten.bitwise_xor, aten.remainder, aten.add, aten.copy, aten.arange]
            stream2 = get_raw_stream(2)
            triton_poi_fused_add_arange_bitwise_xor_copy_fill_lift_fresh_mul_remainder_select_slice_unsqueeze_0.run(arg4_1, _tensor_constant0, _tensor_constant1, buf1, 512000, 12, stream=stream2)
            buf2 = empty_strided_cuda((512, 1000, 384), (384000, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
            stream2 = get_raw_stream(2)
            triton_poi_fused_embedding_1.run(arg4_1, arg5_1, buf2, 512000, 384, stream=stream2)
            del arg4_1
            del arg5_1
        return (buf1, buf2, reinterpret_tensor(arg0_1, (1, 1000, 1, 24), (24576, 24, 24, 1), 0), reinterpret_tensor(arg1_1, (1, 1000, 1, 24), (24576, 24, 24, 1), 0), reinterpret_tensor(arg2_1, (1, 1000, 1, 24), (24576, 24, 24, 1), 0), reinterpret_tensor(arg3_1, (1, 1000, 1, 24), (24576, 24, 24, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    global _tensor_constant0
    _tensor_constant0 = rand_strided((12, ), (1, ), device='cuda:2', dtype=torch.int64)
    global _tensor_constant1
    _tensor_constant1 = rand_strided((12, ), (1, ), device='cuda:2', dtype=torch.int64)
    arg0_1 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:2', dtype=torch.bfloat16)
    arg1_1 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:2', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:2', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1, 1024, 1, 24), (24576, 24, 24, 1), device='cuda:2', dtype=torch.bfloat16)
    arg4_1 = rand_strided((512, 1000), (1000, 1), device='cuda:2', dtype=torch.int64)
    arg5_1 = rand_strided((8192, 384), (384, 1), device='cuda:2', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
