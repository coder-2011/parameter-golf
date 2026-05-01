# AOT ID: ['10_backward']
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


# kernel path: /workspace/parameter-golf/.torchinductor_cache/3q/c3qmma75bawbe32waxqpycyeb57kytkbgofc7tnq2yuryl5wpii4.py
# Topologically Sorted Source Nodes: [div_2, unsqueeze_1, ne_3, cross_entropy, where_2, where_3, mul_2, float_1, truediv, tanh, exp_1, sum_4, mul_3, sub_2, mul_4, mul_5, sub_3, mul_6, div_3, convert_element_type_4], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward, aten.arange, aten.view, aten.expand, aten.eq, aten.scalar_tensor, aten.where, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.sub, aten._log_softmax, aten._log_softmax_backward_data, aten.tanh_backward]
# Source node to ATen node mapping:
#   convert_element_type_4 => convert_element_type_4
#   cross_entropy => full_default, full_default_1, sub_1
#   div_2 => div_2
#   div_3 => div_3
#   exp_1 => exp_1
#   float_1 => convert_element_type_2
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   ne_3 => ne_3
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sum_4 => sum_4
#   tanh => tanh
#   truediv => div
#   unsqueeze_1 => unsqueeze_1
#   where_2 => where_2
#   where_3 => where_3
# Graph fragment:
#   %primals_3 : Tensor "i64[131072][1]cuda:3" = PlaceHolder[target=primals_3]
#   %tangents_1 : Tensor "f32[][]cuda:3" = PlaceHolder[target=tangents_1]
#   %convert_element_type_3 : Tensor "f32[][]cuda:3" = PlaceHolder[target=convert_element_type_3]
#   %mm : Tensor "bf16[131072, 8192][8192, 1]cuda:3" = PlaceHolder[target=mm]
#   %amax_default : Tensor "f32[131072, 1][1, 1]cuda:3" = PlaceHolder[target=amax_default]
#   %log : Tensor "f32[131072, 1][1, 1]cuda:3" = PlaceHolder[target=log]
#   %sum_4 : Tensor "f32[131072, 1][1, 131072]cuda:3" = PlaceHolder[target=sum_4]
#   %sub_2 : Tensor "f32[131072, 8192][8192, 1]cuda:3" = PlaceHolder[target=sub_2]
#   %div_2 : Tensor "f32[][]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_3), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[131072, 1][1, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_3, 1), kwargs = {})
#   %ne_3 : Tensor "b8[131072, 1][1, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_1, -100), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:3, pin_memory: False})
#   %where_2 : Tensor "i64[131072, 1][1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %unsqueeze_1, %full_default), kwargs = {})
#   %iota_default : Tensor "i64[8192][1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8192,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:3, requires_grad: False})
#   %view_default : Tensor "i64[1, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_default, [1, 8192]), kwargs = {})
#   %expand_default : Tensor "i64[131072, 8192][1, 0]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_2, [131072, 8192]), kwargs = {})
#   %eq_tensor : Tensor "b8[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%expand_default, %view_default), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:3})
#   %scalar_tensor_default_1 : Tensor "f32[][]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (-1.0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:3})
#   %where_self : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_tensor, %scalar_tensor_default_1, %scalar_tensor_default), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:3, pin_memory: False})
#   %where_3 : Tensor "f32[131072, 1][1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %div_2, %full_default_1), kwargs = {})
#   %mul_2 : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self, %where_3), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %div : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_2, 30.0), kwargs = {})
#   %tanh : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_tensor : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 1), kwargs = {})
#   %sub_tensor : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor, 30.0), kwargs = {})
#   %sub_1 : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %log), kwargs = {})
#   %exp_1 : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_4 : Tensor "f32[131072, 1][1, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [1], True), kwargs = {})
#   %mul_3 : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, %sum_4), kwargs = {})
#   %sub_2 : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2, %mul_3), kwargs = {})
#   %mul_4 : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, 30.0), kwargs = {})
#   %mul_5 : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %tanh), kwargs = {})
#   %sub_3 : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_5), kwargs = {})
#   %mul_6 : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %sub_3), kwargs = {})
#   %div_3 : Tensor "f32[131072, 8192][8192, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_6, 30.0), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[131072, 8192][8192, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_3, torch.bfloat16), kwargs = {})
#   return %sum_4,%sub_2,%convert_element_type_4
triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_arange_div_eq_expand_mul_nll_loss_backward_nll_loss_forward_scalar_tensor_sub_tanh_tanh_backward_view_where_0 = async_compile.triton('triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_arange_div_eq_expand_mul_nll_loss_backward_nll_loss_forward_scalar_tensor_sub_tanh_tanh_backward_view_where_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_arange_div_eq_expand_mul_nll_loss_backward_nll_loss_forward_scalar_tensor_sub_tanh_tanh_backward_view_where_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2097152, 'r0_': 6442450944}}
)
@triton.jit
def triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_arange_div_eq_expand_mul_nll_loss_backward_nll_loss_forward_scalar_tensor_sub_tanh_tanh_backward_view_where_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (0))
    tmp11 = tl.broadcast_to(tmp10, [1, 1])
    tmp12 = tl.load(in_ptr2 + (0))
    tmp13 = tl.broadcast_to(tmp12, [1, 1])
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = r0_1
        tmp6 = tmp4 == tmp5
        tmp7 = tl.full([1, 1], -1.0, tl.float32)
        tmp8 = tl.full([1, 1], 0.0, tl.float32)
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp14 = (tmp11 / tmp13)
        tmp15 = tl.where(tmp2, tmp14, tmp8)
        tmp16 = tmp9 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp29 = tl.load(in_ptr1 + (0))
    tmp30 = tl.broadcast_to(tmp29, [1, 1])
    tmp31 = tl.load(in_ptr2 + (0))
    tmp32 = tl.broadcast_to(tmp31, [1, 1])
    tmp43 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp36 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.full([1, 1], -100, tl.int64)
        tmp21 = tmp0 != tmp20
        tmp22 = tl.full([1, 1], 0, tl.int64)
        tmp23 = tl.where(tmp21, tmp0, tmp22)
        tmp24 = r0_1
        tmp25 = tmp23 == tmp24
        tmp26 = tl.full([1, 1], -1.0, tl.float32)
        tmp27 = tl.full([1, 1], 0.0, tl.float32)
        tmp28 = tl.where(tmp25, tmp26, tmp27)
        tmp33 = (tmp30 / tmp32)
        tmp34 = tl.where(tmp21, tmp33, tmp27)
        tmp35 = tmp28 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tl.full([1, 1], 0.03333333333333333, tl.float32)
        tmp39 = tmp37 * tmp38
        tmp40 = libdevice.tanh(tmp39)
        tmp41 = tl.full([1, 1], 1.0, tl.float32)
        tmp42 = tmp40 * tmp41
        tmp44 = tmp42 - tmp43
        tmp45 = tl.full([1, 1], 30.0, tl.float32)
        tmp46 = tmp44 * tmp45
        tmp48 = tmp46 - tmp47
        tmp49 = libdevice.exp(tmp48)
        tmp50 = tmp49 * tmp18
        tmp51 = tmp35 - tmp50
        tmp52 = tmp51 * tmp45
        tmp53 = tmp40 * tmp40
        tmp54 = tmp41 - tmp53
        tmp55 = tmp52 * tmp54
        tmp56 = tmp55 * tmp38
        tmp57 = tmp56.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp57, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/parameter-golf/.torchinductor_cache/b5/cb55vjoclhq6dn7ttm6q5as47vvyrgnri4tn3m475yoiejcijy5q.py
# Topologically Sorted Source Nodes: [mul_7], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_7 => mul_7
# Graph fragment:
#   %mm_1 : Tensor "bf16[8192, 384][384, 1]cuda:3" = PlaceHolder[target=mm_1]
#   %mul_7 : Tensor "bf16[8192, 384][384, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_1, 1.0), kwargs = {})
#   return %mul_7
triton_poi_fused_mul_1 = async_compile.triton('triton_poi_fused_mul_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 1.0, tl.float32)
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
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
        primals_2, primals_3, mm, amax_default, log, convert_element_type_3, permute_3, tangents_1 = args
        args.clear()
        assert_size_stride(primals_2, (131072, 384), (384, 1))
        assert_size_stride(primals_3, (131072, ), (1, ))
        assert_size_stride(mm, (131072, 8192), (8192, 1))
        assert_size_stride(amax_default, (131072, 1), (1, 1))
        assert_size_stride(log, (131072, 1), (1, 1))
        assert_size_stride(convert_element_type_3, (), ())
        assert_size_stride(permute_3, (8192, 384), (384, 1))
        assert_size_stride(tangents_1, (), ())
        with torch.cuda._DeviceGuard(3):
            torch.cuda.set_device(3)
            buf2 = mm; del mm  # reuse
            # Topologically Sorted Source Nodes: [div_2, unsqueeze_1, ne_3, cross_entropy, where_2, where_3, mul_2, float_1, truediv, tanh, exp_1, sum_4, mul_3, sub_2, mul_4, mul_5, sub_3, mul_6, div_3, convert_element_type_4], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward, aten.arange, aten.view, aten.expand, aten.eq, aten.scalar_tensor, aten.where, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.sub, aten._log_softmax, aten._log_softmax_backward_data, aten.tanh_backward]
            stream3 = get_raw_stream(3)
            triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_arange_div_eq_expand_mul_nll_loss_backward_nll_loss_forward_scalar_tensor_sub_tanh_tanh_backward_view_where_0.run(buf2, primals_3, tangents_1, convert_element_type_3, amax_default, log, 131072, 8192, stream=stream3)
            del amax_default
            del convert_element_type_3
            del log
            del primals_3
            del tangents_1
            buf3 = empty_strided_cuda((8192, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_1, mm_1], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf2, (8192, 131072), (1, 8192), 0), primals_2, out=buf3)
            del primals_2
            buf4 = empty_strided_cuda((131072, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mm_2], Original ATen: [aten.mm]
            extern_kernels.mm(buf2, permute_3, out=buf4)
            del buf2
            del permute_3
            buf5 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [mul_7], Original ATen: [aten.mul]
            stream3 = get_raw_stream(3)
            triton_poi_fused_mul_1.run(buf5, 3145728, stream=stream3)
        return (buf5, buf4, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_2 = rand_strided((131072, 384), (384, 1), device='cuda:3', dtype=torch.bfloat16)
    primals_3 = rand_strided((131072, ), (1, ), device='cuda:3', dtype=torch.int64)
    mm = rand_strided((131072, 8192), (8192, 1), device='cuda:3', dtype=torch.bfloat16)
    amax_default = rand_strided((131072, 1), (1, 1), device='cuda:3', dtype=torch.float32)
    log = rand_strided((131072, 1), (1, 1), device='cuda:3', dtype=torch.float32)
    convert_element_type_3 = rand_strided((), (), device='cuda:3', dtype=torch.float32)
    permute_3 = rand_strided((8192, 384), (384, 1), device='cuda:3', dtype=torch.bfloat16)
    tangents_1 = rand_strided((), (), device='cuda:3', dtype=torch.float32)
    return [primals_2, primals_3, mm, amax_default, log, convert_element_type_3, permute_3, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
