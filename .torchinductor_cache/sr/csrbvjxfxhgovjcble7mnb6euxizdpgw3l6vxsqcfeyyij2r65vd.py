
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
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
