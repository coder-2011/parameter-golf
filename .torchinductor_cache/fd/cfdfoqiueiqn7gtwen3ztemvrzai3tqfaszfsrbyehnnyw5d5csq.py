# AOT ID: ['25_inference']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/6v/c6vbgjmjninsr6mc2qmuj5ecwcayfetky3d7qf6rmwy5aor56d64.py
# Topologically Sorted Source Nodes: [to_9], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   to_9 => convert_element_type_17
# Graph fragment:
#   %arg13_1 : Tensor "f32[8, 12][12, 1]cuda:1" = PlaceHolder[target=arg13_1]
#   %convert_element_type_17 : Tensor "bf16[8, 12][12, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg13_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_17
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/y6/cy6cclo4xofyowdkofsljzw5xo2iggfzuodl5pufyikfnf5xpb3a.py
# Topologically Sorted Source Nodes: [to_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   to_1 => convert_element_type_5
# Graph fragment:
#   %arg3_1 : Tensor "f32[384][1]cuda:1" = PlaceHolder[target=arg3_1]
#   %convert_element_type_5 : Tensor "bf16[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_5
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/jg/cjgvgnjhjkwdcrkrbc6v43q6ua7mtmckqzhmkwtmu5mvl52bmz75.py
# Topologically Sorted Source Nodes: [embedding, sum_1, mul, linear], Original ATen: [aten.embedding, aten.sum, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   embedding => embedding
#   linear => convert_element_type_2
#   mul => mul
#   sum_1 => sum_1
# Graph fragment:
#   %arg0_1 : Tensor "i64[350, 1000, 8][8000, 8, 1]cuda:1" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[131072, 128][128, 1]cuda:1" = PlaceHolder[target=arg1_1]
#   %sum_1 : Tensor "f32[350, 1000, 128][128000, 128, 1]cuda:1" = PlaceHolder[target=sum_1]
#   %embedding : Tensor "bf16[350, 1000, 8, 128][1024000, 1024, 128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %sum_1 : Tensor "f32[350, 1000, 128][128000, 128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding, [-2]), kwargs = {dtype: torch.float32})
#   %mul : Tensor "f32[350, 1000, 128][128000, 128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 0.3535533905932738), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[350, 1000, 128][128000, 128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   return %sum_1,%convert_element_type_2
triton_per_fused__to_copy_embedding_mul_sum_2 = async_compile.triton('triton_per_fused__to_copy_embedding_mul_sum_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 67108864, 'r0_': 8},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_embedding_mul_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 179200000, 'r0_': 22400000}}
)
@triton.jit
def triton_per_fused__to_copy_embedding_mul_sum_2(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 44800000
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x1 = xindex // 128
    x0 = (xindex % 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 8*x1), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1, 1], 131072, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 131072)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 131072")
    tmp6 = tl.load(in_ptr1 + (x0 + 128*tmp4), xmask).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 0.3535533905932738, tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/st/cstzn4x6czvuy2qxykdqlwdpgsdw6kx3klrbugg4iriei6hs35f7.py
# Topologically Sorted Source Nodes: [to_5, sigmoid_1, getitem, sub, linear, linear_1, sigmoid, mul_1, to_4, mul_2, add, mul_4, zeros_like, getitem_2, cat, ne, to_6, unsqueeze, mul_3, mul_5, add_2, rms_norm, to_7], Original ATen: [aten._to_copy, aten.sigmoid, aten.unsqueeze, aten.rsub, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.zeros_like, aten.slice, aten.cat, aten.ne, aten._fused_rms_norm]
# Source node to ATen node mapping:
#   add => add
#   add_2 => add_1
#   cat => cat
#   getitem => unsqueeze, unsqueeze_1
#   getitem_2 => slice_2
#   linear => view_1
#   linear_1 => view_3
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   ne => ne
#   rms_norm => add_2, convert_element_type_13, convert_element_type_14, mean, mul_6, mul_7, pow_1, rsqrt
#   sigmoid => sigmoid
#   sigmoid_1 => sigmoid_1
#   sub => sub
#   to_4 => convert_element_type_9
#   to_5 => convert_element_type_10
#   to_6 => convert_element_type_11
#   to_7 => convert_element_type_12
#   unsqueeze => unsqueeze_2
#   zeros_like => full_default
# Graph fragment:
#   %arg7_1 : Tensor "f32[384][1]cuda:1" = PlaceHolder[target=arg7_1]
#   %arg5_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1" = PlaceHolder[target=arg5_1]
#   %mm : Tensor "bf16[350000, 384][384, 1]cuda:1" = PlaceHolder[target=mm]
#   %addmm : Tensor "bf16[350000, 384][384, 1]cuda:1" = PlaceHolder[target=addmm]
#   %arg6_1 : Tensor "f32[][]cuda:1" = PlaceHolder[target=arg6_1]
#   %arg8_1 : Tensor "i64[350, 1000][1000, 1]cuda:1" = PlaceHolder[target=arg8_1]
#   %add_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1" = PlaceHolder[target=add_1]
#   %buf6 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:1" = PlaceHolder[target=buf6]
#   %arg9_1 : Tensor "f32[384][1]cuda:1" = PlaceHolder[target=arg9_1]
#   %convert_element_type_10 : Tensor "bf16[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg7_1, torch.bfloat16), kwargs = {})
#   %sigmoid_1 : Tensor "bf16[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_10,), kwargs = {})
#   %unsqueeze : Tensor "bf16[1, 384][384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %sub : Tensor "bf16[1, 1, 384][384, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %unsqueeze_1), kwargs = {})
#   %view_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [350, 1000, 384]), kwargs = {})
#   %view_3 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [350, 1000, 384]), kwargs = {})
#   %sigmoid : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_3,), kwargs = {})
#   %mul_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %sigmoid), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[][]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.bfloat16), kwargs = {})
#   %mul_2 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %convert_element_type_9), kwargs = {})
#   %add : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg5_1, %mul_2), kwargs = {})
#   %mul_4 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %add), kwargs = {})
#   %full_default : Tensor "bf16[350, 1, 384][384, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([350, 1, 384], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %slice_2 : Tensor "bf16[350, 999, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add, 1, 0, -1), kwargs = {})
#   %cat : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%full_default, %slice_2], 1), kwargs = {})
#   %ne : Tensor "b8[350, 1000][1000, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%arg8_1, 1), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[350, 1000][1000, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ne, torch.bfloat16), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[350, 1000, 1][1000, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_11, -1), kwargs = {})
#   %mul_3 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_2), kwargs = {})
#   %mul_5 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %mul_3), kwargs = {})
#   %add_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_13, 2), kwargs = {})
#   %mean : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add_2 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_6 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_13, %rsqrt), kwargs = {})
#   %convert_element_type_12 : Tensor "bf16[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.bfloat16), kwargs = {})
#   %mul_7 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %convert_element_type_12), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.bfloat16), kwargs = {})
#   return %add_1,%buf6,%convert_element_type_14
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_3 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 11, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2800000, 'r0_': 2688003072}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ac/cacarf62jtia7ockwbaaiqmkwnha4iahndrjpstliwz4mqa5dlgv.py
# Topologically Sorted Source Nodes: [getitem_6, contiguous_6], Original ATen: [aten.slice, aten.clone]
# Source node to ATen node mapping:
#   contiguous_6 => clone_1
#   getitem_6 => slice_4
# Graph fragment:
#   %convert_element_type_14 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1" = PlaceHolder[target=convert_element_type_14]
#   %slice_4 : Tensor "bf16[350, 1000, 12][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_14, 2, 0, 12), kwargs = {})
#   %clone_1 : Tensor "bf16[350, 1000, 12][12000, 12, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_4,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused_clone_slice_4 = async_compile.triton('triton_poi_fused_clone_slice_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_slice_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 25200000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_slice_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/un/cunvzlbsc4s4qkuf2qri3xcxifvflniubesornx4t5ukf6shh65l.py
# Topologically Sorted Source Nodes: [getitem_6, contiguous_6, linear_3, to_9], Original ATen: [aten.slice, aten.clone, aten.view, aten._to_copy, aten.t, aten.mm]
# Source node to ATen node mapping:
#   contiguous_6 => clone_1
#   getitem_6 => slice_4
#   linear_3 => mm_2, permute_9, view_7
#   to_9 => convert_element_type_17
# Graph fragment:
#   %clone_1 : Tensor "bf16[350, 1000, 12][12032, 12, 1]cuda:1" = PlaceHolder[target=clone_1]
#   %slice_4 : Tensor "bf16[350, 1000, 12][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_14, 2, 0, 12), kwargs = {})
#   %clone_1 : Tensor "bf16[350, 1000, 12][12000, 12, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_7 : Tensor "bf16[350000, 12][12, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_1, [350000, 12]), kwargs = {})
#   %convert_element_type_17 : Tensor "bf16[8, 12][12, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg13_1, torch.bfloat16), kwargs = {})
#   %permute_9 : Tensor "bf16[12, 8][1, 12]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_17, [1, 0]), kwargs = {})
#   %mm_2 : Tensor "bf16[350000, 8][8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_7, %permute_9), kwargs = {})
#   return %buf21
triton_poi_fused__to_copy_clone_mm_slice_t_view_5 = async_compile.triton('triton_poi_fused__to_copy_clone_mm_slice_t_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_mm_slice_t_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 25200000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_mm_slice_t_view_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    triton_meta={'signature': {'QKV': '*bf16', 'Q': '*bf16', 'K': '*bf16', 'COS': '*bf16', 'SIN': '*bf16', 'TOTAL_ROWS': 'constexpr', 'SEQLEN': 'constexpr', 'H_Q': 'constexpr', 'H_KV': 'constexpr', 'HEAD_DIM': 'constexpr', 'ROPE_DIMS': 'constexpr', 'DO_QK_NORM': 'constexpr', 'ROPE_INTERLEAVED': 'constexpr', 'RMS_EPS': 'constexpr', 'BLOCK_D': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'TOTAL_ROWS': 350000, 'SEQLEN': 1000, 'H_Q': 8, 'H_KV': 4, 'HEAD_DIM': 48, 'ROPE_DIMS': 48, 'DO_QK_NORM': True, 'ROPE_INTERLEAVED': False, 'RMS_EPS': 1e-06, 'BLOCK_D': 64}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/w7/cw73vo4h7ndyppvyahjwirotjknvapnofhcg24oiuamijg4vtlgc.py
# Topologically Sorted Source Nodes: [linear_2, view, scaled_dot_product_attention, getitem_3, contiguous, transpose_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
# Source node to ATen node mapping:
#   contiguous => clone
#   getitem_3 => slice_3
#   linear_2 => view_5
#   scaled_dot_product_attention => _scaled_dot_product_flash_attention, permute_6, permute_7
#   transpose_2 => permute_5
#   view => view_6
# Graph fragment:
#   %mm_1 : Tensor "bf16[350000, 768][768, 1]cuda:1" = PlaceHolder[target=mm_1]
#   %view_5 : Tensor "bf16[350, 1000, 768][768000, 768, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [350, 1000, 768]), kwargs = {})
#   %view_6 : Tensor "bf16[350, 1000, 16, 48][768000, 768, 48, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [350, 1000, 16, 48]), kwargs = {})
#   %permute_6 : Tensor "bf16[350, 8, 1000, 48][384000, 48, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty, [0, 2, 1, 3]), kwargs = {})
#   %permute_7 : Tensor "bf16[350, 4, 1000, 48][192000, 48, 192, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty_1, [0, 2, 1, 3]), kwargs = {})
#   %slice_3 : Tensor "bf16[350, 1000, 4, 48][768000, 768, 48, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_6, 2, 12, 9223372036854775807), kwargs = {})
#   %clone : Tensor "bf16[350, 1000, 4, 48][192000, 192, 48, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_3,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_5 : Tensor "bf16[350, 4, 1000, 48][192000, 48, 192, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%clone, [0, 2, 1, 3]), kwargs = {})
#   %_scaled_dot_product_flash_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%permute_6, %permute_7, %permute_5, 0.0, True), kwargs = {scale: 0.14433756729740646})
#   return %buf13
triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_6 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 403200000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/l6/cl6zuinhz5bzw3r76cgej6ssxh25lazxkd5a4djmh2b7wmnljn3v.py
# Topologically Sorted Source Nodes: [transpose_3, linear_3, mul_6, sigmoid_2, getitem_7, mul_7], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
# Source node to ATen node mapping:
#   getitem_7 => unsqueeze_3
#   linear_3 => view_8
#   mul_6 => mul_8
#   mul_7 => mul_9
#   sigmoid_2 => sigmoid_2
#   transpose_3 => permute_8
# Graph fragment:
#   %getitem_2 : Tensor "bf16[350, 8, 1000, 48][384000, 48, 384, 1]cuda:1" = PlaceHolder[target=getitem_2]
#   %mm_2 : Tensor "bf16[350000, 8][8, 1]cuda:1" = PlaceHolder[target=mm_2]
#   %permute_8 : Tensor "bf16[350, 1000, 8, 48][384000, 384, 48, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [0, 2, 1, 3]), kwargs = {})
#   %view_8 : Tensor "bf16[350, 1000, 8][8000, 8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [350, 1000, 8]), kwargs = {})
#   %mul_8 : Tensor "bf16[350, 1000, 8][8000, 8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, 0.5), kwargs = {})
#   %sigmoid_2 : Tensor "bf16[350, 1000, 8][8000, 8, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_8,), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[350, 1000, 8, 1][8000, 8, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_2, 3), kwargs = {})
#   %mul_9 : Tensor "bf16[350, 1000, 8, 48][384000, 384, 48, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_8, %unsqueeze_3), kwargs = {})
#   return %mul_9
triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_7 = async_compile.triton('triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 806400000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/ob/cob2gsabrn5qz747v7riak32rctopjdqonngklzzmjah67iwvpmb.py
# Topologically Sorted Source Nodes: [linear_4, add_3, rms_norm_1, to_11], Original ATen: [aten._unsafe_view, aten.add, aten._fused_rms_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add_3 => add_3
#   linear_4 => view_11
#   rms_norm_1 => add_4, convert_element_type_23, convert_element_type_24, mean_1, mul_10, mul_11, pow_2, rsqrt_1
#   to_11 => convert_element_type_22
# Graph fragment:
#   %mm_3 : Tensor "bf16[350000, 384][384, 1]cuda:1" = PlaceHolder[target=mm_3]
#   %add_1 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1" = PlaceHolder[target=add_1]
#   %add_3 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1" = PlaceHolder[target=add_3]
#   %buf27 : Tensor "f32[350, 1000, 1][1000, 1, 350016]cuda:1" = PlaceHolder[target=buf27]
#   %arg15_1 : Tensor "f32[384][1]cuda:1" = PlaceHolder[target=arg15_1]
#   %view_11 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [350, 1000, 384]), kwargs = {})
#   %add_3 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %add_1), kwargs = {})
#   %convert_element_type_23 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_23, 2), kwargs = {})
#   %mean_1 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_4 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[350, 1000, 1][1000, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_10 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_23, %rsqrt_1), kwargs = {})
#   %convert_element_type_22 : Tensor "bf16[384][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg15_1, torch.bfloat16), kwargs = {})
#   %mul_11 : Tensor "f32[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %convert_element_type_22), kwargs = {})
#   %convert_element_type_24 : Tensor "bf16[350, 1000, 384][384000, 384, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.bfloat16), kwargs = {})
#   return %add_3,%buf27,%convert_element_type_24
triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 1612801536}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1 = args
        args.clear()
        assert_size_stride(arg0_1, (350, 1000, 8), (8000, 8, 1))
        assert_size_stride(arg1_1, (131072, 128), (128, 1))
        assert_size_stride(arg2_1, (384, 128), (128, 1))
        assert_size_stride(arg3_1, (384, ), (1, ))
        assert_size_stride(arg4_1, (384, 384), (384, 1))
        assert_size_stride(arg5_1, (350, 1000, 384), (384000, 384, 1))
        assert_size_stride(arg6_1, (), ())
        assert_size_stride(arg7_1, (384, ), (1, ))
        assert_size_stride(arg8_1, (350, 1000), (1000, 1))
        assert_size_stride(arg9_1, (384, ), (1, ))
        assert_size_stride(arg10_1, (768, 384), (384, 1))
        assert_size_stride(arg11_1, (1, 1000, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg12_1, (1, 1000, 1, 24), (24576, 24, 24, 1))
        assert_size_stride(arg13_1, (8, 12), (12, 1))
        assert_size_stride(arg14_1, (384, 384), (384, 1))
        assert_size_stride(arg15_1, (384, ), (1, ))
        assert_size_stride(arg16_1, (1536, 384), (384, 1))
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            buf22 = empty_strided_cuda((8, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_9], Original ATen: [aten._to_copy]
            stream1 = get_raw_stream(1)
            triton_poi_fused__to_copy_0.run(arg13_1, buf22, 96, stream=stream1)
            del arg13_1
            buf3 = empty_strided_cuda((384, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_1], Original ATen: [aten._to_copy]
            stream1 = get_raw_stream(1)
            triton_poi_fused__to_copy_1.run(arg3_1, buf3, 384, stream=stream1)
            del arg3_1
            buf1 = empty_strided_cuda((350, 1000, 128), (128000, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [embedding, sum_1, mul, linear], Original ATen: [aten.embedding, aten.sum, aten.mul, aten._to_copy]
            stream1 = get_raw_stream(1)
            triton_per_fused__to_copy_embedding_mul_sum_2.run(arg0_1, arg1_1, buf1, 44800000, 8, stream=stream1)
            del arg0_1
            del arg1_1
            buf10 = empty_strided_cuda((350, 1000, 4, 48), (192000, 192, 48, 1), torch.bfloat16)
            buf2 = empty_strided_cuda((350000, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul, linear], Original ATen: [aten.mul, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1, (350000, 128), (128, 1), 0), reinterpret_tensor(arg2_1, (128, 384), (1, 128), 0), out=buf2)
            del arg2_1
            del buf1
            buf4 = empty_strided_cuda((350000, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_1, linear_1], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(buf3, reinterpret_tensor(arg5_1, (350000, 384), (384, 1), 0), reinterpret_tensor(arg4_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf4)
            del arg4_1
            del buf3
            buf9 = empty_strided_cuda((350, 1000, 8, 48), (384000, 384, 48, 1), torch.bfloat16)
            buf5 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            buf7 = empty_strided_cuda((350, 1000, 384), (384000, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_5, sigmoid_1, getitem, sub, linear, linear_1, sigmoid, mul_1, to_4, mul_2, add, mul_4, zeros_like, getitem_2, cat, ne, to_6, unsqueeze, mul_3, mul_5, add_2, rms_norm, to_7], Original ATen: [aten._to_copy, aten.sigmoid, aten.unsqueeze, aten.rsub, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.zeros_like, aten.slice, aten.cat, aten.ne, aten._fused_rms_norm]
            stream1 = get_raw_stream(1)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_3.run(arg7_1, arg5_1, buf2, buf4, arg6_1, arg8_1, arg9_1, buf5, buf7, 350000, 384, stream=stream1)
            del arg5_1
            del arg6_1
            del arg7_1
            del arg8_1
            del arg9_1
            del buf2
            del buf4
            buf20 = empty_strided_cuda((350, 1000, 12), (12032, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_6, contiguous_6], Original ATen: [aten.slice, aten.clone]
            stream1 = get_raw_stream(1)
            triton_poi_fused_clone_slice_4.run(buf7, buf20, 4200000, stream=stream1)
            buf21 = empty_strided_cuda((350000, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_6, contiguous_6, linear_3, to_9], Original ATen: [aten.slice, aten.clone, aten.view, aten._to_copy, aten.t, aten.mm]
            stream1 = get_raw_stream(1)
            triton_poi_fused__to_copy_clone_mm_slice_t_view_5.run(buf20, buf21, 4200000, stream=stream1)
            del buf20
            buf23 = empty_strided_cuda((350000, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_6, contiguous_6, linear_3, to_9], Original ATen: [aten.slice, aten.clone, aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf21, reinterpret_tensor(buf22, (12, 8), (1, 12), 0), out=buf23)
            del buf21
            del buf22
            buf8 = empty_strided_cuda((350000, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm, to_7, linear_2], Original ATen: [aten._fused_rms_norm, aten._to_copy, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf7, (350000, 384), (384, 1), 0), reinterpret_tensor(arg10_1, (384, 768), (1, 384), 0), out=buf8)
            del arg10_1
            del buf7
            # Topologically Sorted Source Nodes: [linear_2, view, getitem_4, getitem_5, triton_kernel_wrapper_mutation], Original ATen: [aten._unsafe_view, aten.view, aten.select]
            stream1 = get_raw_stream(1)
            _fused_qkv_postprocess_fwd_kernel_0.run(reinterpret_tensor(buf8, (350, 1000, 16, 48), (768000, 768, 48, 1), 0), buf9, buf10, reinterpret_tensor(arg11_1, (1000, 24), (24, 1), 0), reinterpret_tensor(arg12_1, (1000, 24), (24, 1), 0), 4200000, 1, 1, stream=stream1)
            del arg11_1
            del arg12_1
            buf13 = empty_strided_cuda((350, 4, 1000, 48), (192000, 48, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_2, view, scaled_dot_product_attention, getitem_3, contiguous, transpose_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
            stream1 = get_raw_stream(1)
            triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_slice_transpose_view_6.run(buf8, buf13, 67200000, stream=stream1)
            del buf8
            # Topologically Sorted Source Nodes: [linear_2, view, scaled_dot_product_attention, getitem_3, contiguous, transpose_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.slice, aten.clone, aten._scaled_dot_product_flash_attention]
            buf14 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf9, (350, 8, 1000, 48), (384000, 48, 384, 1), 0), reinterpret_tensor(buf10, (350, 4, 1000, 48), (192000, 48, 192, 1), 0), buf13, 0.0, True, scale=0.14433756729740646)
            del buf10
            del buf13
            buf15 = buf14[0]
            assert_size_stride(buf15, (350, 8, 1000, 48), (384000, 48, 384, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf15, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf14
            buf24 = reinterpret_tensor(buf15, (350, 1000, 8, 48), (384000, 384, 48, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, linear_3, mul_6, sigmoid_2, getitem_7, mul_7], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze]
            stream1 = get_raw_stream(1)
            triton_poi_fused__unsafe_view_mul_sigmoid_transpose_unsqueeze_7.run(buf24, buf23, 134400000, stream=stream1)
            del buf23
            buf25 = reinterpret_tensor(buf9, (350000, 384), (384, 1), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, linear_3, mul_6, sigmoid_2, getitem_7, mul_7, reshape, linear_4], Original ATen: [aten.transpose, aten._unsafe_view, aten.mul, aten.sigmoid, aten.unsqueeze, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf24, (350000, 384), (384, 1), 0), reinterpret_tensor(arg14_1, (384, 384), (1, 384), 0), out=buf25)
            del arg14_1
            buf26 = reinterpret_tensor(buf25, (350, 1000, 384), (384000, 384, 1), 0); del buf25  # reuse
            buf28 = reinterpret_tensor(buf24, (350, 1000, 384), (384000, 384, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [linear_4, add_3, rms_norm_1, to_11], Original ATen: [aten._unsafe_view, aten.add, aten._fused_rms_norm, aten._to_copy]
            stream1 = get_raw_stream(1)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_8.run(buf26, buf5, arg15_1, buf28, 350000, 384, stream=stream1)
            del arg15_1
        return (buf28, arg16_1, buf26, buf5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((350, 1000, 8), (8000, 8, 1), device='cuda:1', dtype=torch.int64)
    arg1_1 = rand_strided((131072, 128), (128, 1), device='cuda:1', dtype=torch.bfloat16)
    arg2_1 = rand_strided((384, 128), (128, 1), device='cuda:1', dtype=torch.bfloat16)
    arg3_1 = rand_strided((384, ), (1, ), device='cuda:1', dtype=torch.float32)
    arg4_1 = rand_strided((384, 384), (384, 1), device='cuda:1', dtype=torch.bfloat16)
    arg5_1 = rand_strided((350, 1000, 384), (384000, 384, 1), device='cuda:1', dtype=torch.bfloat16)
    arg6_1 = rand_strided((), (), device='cuda:1', dtype=torch.float32)
    arg7_1 = rand_strided((384, ), (1, ), device='cuda:1', dtype=torch.float32)
    arg8_1 = rand_strided((350, 1000), (1000, 1), device='cuda:1', dtype=torch.int64)
    arg9_1 = rand_strided((384, ), (1, ), device='cuda:1', dtype=torch.float32)
    arg10_1 = rand_strided((768, 384), (384, 1), device='cuda:1', dtype=torch.bfloat16)
    arg11_1 = rand_strided((1, 1000, 1, 24), (24576, 24, 24, 1), device='cuda:1', dtype=torch.bfloat16)
    arg12_1 = rand_strided((1, 1000, 1, 24), (24576, 24, 24, 1), device='cuda:1', dtype=torch.bfloat16)
    arg13_1 = rand_strided((8, 12), (12, 1), device='cuda:1', dtype=torch.float32)
    arg14_1 = rand_strided((384, 384), (384, 1), device='cuda:1', dtype=torch.bfloat16)
    arg15_1 = rand_strided((384, ), (1, ), device='cuda:1', dtype=torch.float32)
    arg16_1 = rand_strided((1536, 384), (384, 1), device='cuda:1', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
