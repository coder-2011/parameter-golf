# AOT ID: ['0_backward']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/t4/ct47jfe5zbk53uot2nmijshrknivo5pny5t4g6ww6rg3pjzm7hws.py
# Topologically Sorted Source Nodes: [full_default_2], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   full_default_2 => full_default_2
# Graph fragment:
#   %full_default_2 : Tensor "f32[8192, 384][384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8192, 384], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:6, pin_memory: False})
#   return %index_put
triton_poi_fused_embedding_dense_backward_0 = async_compile.triton('triton_poi_fused_embedding_dense_backward_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/eu/ceuiveik35zd3a6ybvttkgeba4q6ccuddfevoi5fw7vpe4wrvvc7.py
# Topologically Sorted Source Nodes: [convert_element_type, eq, unsqueeze_2, full_default_1, where, full_default_2, index_put], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   convert_element_type => convert_element_type
#   eq => eq
#   full_default_1 => full_default_1
#   full_default_2 => full_default_2
#   index_put => index_put
#   unsqueeze_2 => unsqueeze_2
#   where => where
# Graph fragment:
#   %primals_5 : Tensor "i64[128, 1024][1024, 1]cuda:6" = PlaceHolder[target=primals_5]
#   %tangents_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:6" = PlaceHolder[target=tangents_1]
#   %index_put : Tensor "f32[8192, 384][384, 1]cuda:6" = PlaceHolder[target=index_put]
#   %convert_element_type : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.float32), kwargs = {})
#   %eq : Tensor "b8[128, 1024][1024, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_5, -1), kwargs = {})
#   %unsqueeze_2 : Tensor "b8[128, 1024, 1][1024, 1, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, -1), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:6, pin_memory: False})
#   %where : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_2, %full_default_1, %convert_element_type), kwargs = {})
#   %full_default_2 : Tensor "f32[8192, 384][384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8192, 384], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:6, pin_memory: False})
#   %index_put : Tensor "f32[8192, 384][384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_2, [%primals_5], %where, True), kwargs = {})
#   return %buf1
triton_poi_fused_embedding_dense_backward_1 = async_compile.triton('triton_poi_fused_embedding_dense_backward_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_1', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': True, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1048576, 'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_1(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp8 = tl.load(in_ptr1 + (x1 + 384*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.full([1, 1], 8192, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 8192)) | ~(ymask), "index out of bounds: 0 <= tmp4 < 8192")
    tmp6 = tl.full([1, 1], -1, tl.int64)
    tmp7 = tmp0 == tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.full([1, 1], 0.0, tl.float32)
    tmp11 = tl.where(tmp7, tmp10, tmp9)
    tl.atomic_add(out_ptr0 + (tl.broadcast_to(x1 + 384*tmp4, [YBLOCK, XBLOCK])), tmp11, xmask & ymask, sem='relaxed')
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/sk/csksrpzzm33scbwb2hflmvvgzxefeucq3ssxefl5ayxry3wzgxqk.py
# Topologically Sorted Source Nodes: [convert_element_type_1], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   convert_element_type_1 => convert_element_type_1
# Graph fragment:
#   %buf1 : Tensor "f32[8192, 384][384, 1]cuda:6" = PlaceHolder[target=buf1]
#   %convert_element_type_1 : Tensor "bf16[8192, 384][384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%index_put, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused_embedding_dense_backward_2 = async_compile.triton('triton_poi_fused_embedding_dense_backward_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12582912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
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
        primals_5, tangents_1 = args
        args.clear()
        assert_size_stride(primals_5, (128, 1024), (1024, 1))
        assert_size_stride(tangents_1, (128, 1024, 384), (393216, 384, 1))
        with torch.cuda._DeviceGuard(6):
            torch.cuda.set_device(6)
            buf0 = empty_strided_cuda((8192, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_default_2], Original ATen: [aten.embedding_dense_backward]
            stream6 = get_raw_stream(6)
            triton_poi_fused_embedding_dense_backward_0.run(buf0, 3145728, stream=stream6)
            # Topologically Sorted Source Nodes: [convert_element_type, eq, unsqueeze_2, full_default_1, where, full_default_2, index_put], Original ATen: [aten.embedding_dense_backward]
            stream6 = get_raw_stream(6)
            triton_poi_fused_embedding_dense_backward_1.run(primals_5, tangents_1, buf0, 131072, 384, stream=stream6)
            del primals_5
            del tangents_1
            buf2 = empty_strided_cuda((8192, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [convert_element_type_1], Original ATen: [aten.embedding_dense_backward]
            stream6 = get_raw_stream(6)
            triton_poi_fused_embedding_dense_backward_2.run(buf0, buf2, 3145728, stream=stream6)
            del buf0
        return (None, None, None, None, None, buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_5 = rand_strided((128, 1024), (1024, 1), device='cuda:6', dtype=torch.int64)
    tangents_1 = rand_strided((128, 1024, 384), (393216, 384, 1), device='cuda:6', dtype=torch.bfloat16)
    return [primals_5, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
