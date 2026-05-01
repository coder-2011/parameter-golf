
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 524288, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 15, 'num_store': 4, 'num_reduction': 2, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 4569612288}}
)
@triton.jit
def triton_red_fused__fused_rms_norm__to_copy__unsafe_view_add_mul_select_unsqueeze_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 350000
    r0_numel = 384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr5 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr7 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr0 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 * tmp2
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 * tmp6
        tmp8 = tmp3 + tmp7
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp9 * tmp11
        tmp13 = tmp8 + tmp12
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 * tmp16
        tmp18 = tmp13 + tmp17
        tmp19 = tmp0 * tmp18
        tmp21 = tmp20 * tmp6
        tmp22 = tmp19 + tmp21
        tmp23 = tmp22 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask & xmask, tmp26, _tmp25)
        tl.store(out_ptr0 + (r0_1 + 384*x0), tmp19, r0_mask & xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp27 = tl.load(out_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr0 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.load(in_ptr3 + (r0_1 + 384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp39 = tl.load(in_ptr8 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp43 = tl.load(in_ptr9 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp28 * tmp30
        tmp32 = tmp27 + tmp31
        tmp33 = tl.full([1, 1], 384.0, tl.float32)
        tmp34 = (tmp25 / tmp33)
        tmp35 = tl.full([1, 1], 1e-06, tl.float32)
        tmp36 = tmp34 + tmp35
        tmp37 = libdevice.rsqrt(tmp36)
        tmp38 = tmp32 * tmp37
        tmp40 = tmp38 * tmp39
        tmp41 = tl.full([1, 1], 0.5773502691896258, tl.float32)
        tmp42 = tmp40 * tmp41
        tmp44 = tmp38 * tmp43
        tmp45 = tmp44 * tmp41
        tmp46 = tmp45.to(tl.float32)
        tmp47 = tmp42.to(tl.float32)
        tl.store(out_ptr3 + (r0_1 + 384*x0), tmp42, r0_mask & xmask)
        tl.store(out_ptr4 + (r0_1 + 384*x0), tmp46, r0_mask & xmask)
        tl.store(out_ptr5 + (r0_1 + 384*x0), tmp47, r0_mask & xmask)
