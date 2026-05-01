# AOT ID: ['2_inference']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/rv/crvgwi7ag2swjuk2ksjj73fsf6jupn4el346luj7pmsq4klbb5vy.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.randn]
# Source node to ATen node mapping:
#   x => convert_element_type_default, inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_seeds_default : Tensor "i64[1][1]cuda:6" = PlaceHolder[target=inductor_seeds_default]
#   %inductor_random_default : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:6" = PlaceHolder[target=inductor_random_default]
#   %inductor_lookup_seed_default : Tensor "i64[][]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([128, 1024, 384], %inductor_lookup_seed_default, randn), kwargs = {})
#   %convert_element_type_default : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_random_default, torch.bfloat16), kwargs = {})
#   return %inductor_random_default,%convert_element_type_default
triton_poi_fused_randn_0 = async_compile.triton('triton_poi_fused_randn_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr1': '*bf16', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_randn_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 201326592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_randn_0(in_ptr0, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ez/cezhtdgvtjx2yfoil4il5nmpoaiulvjzxp2tawnfdqwb6yquuysd.py
# Topologically Sorted Source Nodes: [erfinv_, mul_, add_, clamp_], Original ATen: [aten.erfinv, aten.mul, aten.add, aten.clamp]
# Source node to ATen node mapping:
#   add_ => add
#   clamp_ => clamp_max, clamp_min, convert_element_type, convert_element_type_1
#   erfinv_ => erfinv
#   mul_ => mul
# Graph fragment:
#   %uniform : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:6" = PlaceHolder[target=uniform]
#   %erfinv : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.erfinv.default](args = (%uniform,), kwargs = {})
#   %mul : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%erfinv, 0.007071067811865476), kwargs = {})
#   %add : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 0.0), kwargs = {})
#   %convert_element_type : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %clamp_min : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type, -0.015), kwargs = {})
#   %clamp_max : Tensor "f32[128, 1024, 384][393216, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 0.015), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 1024, 384][393216, 384, 1]cuda:6"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused_add_clamp_erfinv_mul_1 = async_compile.triton('triton_poi_fused_add_clamp_erfinv_mul_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_erfinv_mul_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 301989888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_erfinv_mul_1(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = libdevice.erfinv(tmp0)
    tmp2 = tl.full([1], 0.007071067811865476, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.full([1], 0.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full([1], -0.015, tl.float32)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tl.full([1], 0.015, tl.float32)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp11, None)
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
        with torch.cuda._DeviceGuard(6):
            torch.cuda.set_device(6)
            buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [], Original ATen: []
            aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
            buf2 = empty_strided_cuda((128, 1024, 384), (393216, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.randn]
            stream6 = get_raw_stream(6)
            triton_poi_fused_randn_0.run(buf0, buf2, 0, 50331648, stream=stream6)
            del buf0
            # Topologically Sorted Source Nodes: [x, uniform_], Original ATen: [aten.randn, aten.uniform]
            buf3 = torch.ops.aten.uniform.default(buf2, -0.9973002039367398, 0.9973002039367398)
            del buf2
            buf4 = buf3
            assert_size_stride(buf4, (128, 1024, 384), (393216, 384, 1), 'torch.ops.aten.uniform.default')
            assert_alignment(buf4, 16, 'torch.ops.aten.uniform.default')
            del buf3
            buf5 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [erfinv_, mul_, add_, clamp_], Original ATen: [aten.erfinv, aten.mul, aten.add, aten.clamp]
            stream6 = get_raw_stream(6)
            triton_poi_fused_add_clamp_erfinv_mul_1.run(buf5, 50331648, stream=stream6)
        return (buf5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    return []


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
