
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_mul_select_sum_unsqueeze_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 11, 'num_store': 8, 'num_reduction': 8, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1317202944, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_mul_select_sum_unsqueeze_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 134016
    r0_numel = 376
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 384
    x0 = (xindex % 384)
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp35 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp42 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp49 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp59 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp67 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 376*x1
        tmp1 = tl.full([1, 1], 131072, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp12 = tl.load(in_ptr2 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp3 * tmp12
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask & xmask, tmp18, _tmp17)
        tmp19 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp3 * tmp19
        tmp21 = tl.load(in_ptr4 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 * tmp22
        tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
        tmp25 = tl.where(tmp2, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
        tmp29 = tl.load(in_ptr5 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp20 * tmp30
        tmp32 = tl.full(tmp31.shape, 0, tmp31.dtype)
        tmp33 = tl.where(tmp2, tmp31, tmp32)
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
        tmp36 = _tmp35 + tmp34
        _tmp35 = tl.where(r0_mask & xmask, tmp36, _tmp35)
        tmp37 = tl.load(in_ptr6 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp38 = tmp37 * tmp5
        tmp39 = tl.full(tmp38.shape, 0, tmp38.dtype)
        tmp40 = tl.where(tmp2, tmp38, tmp39)
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(r0_mask & xmask, tmp43, _tmp42)
        tmp44 = tl.load(in_ptr7 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp45 = tmp37 * tmp44
        tmp46 = tl.full(tmp45.shape, 0, tmp45.dtype)
        tmp47 = tl.where(tmp2, tmp45, tmp46)
        tmp48 = tl.broadcast_to(tmp47, [XBLOCK, R0_BLOCK])
        tmp50 = _tmp49 + tmp48
        _tmp49 = tl.where(r0_mask & xmask, tmp50, _tmp49)
        tmp51 = tl.load(in_ptr8 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp52 = tmp37 * tmp51
        tmp53 = tl.load(in_ptr9 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp54 = tmp53.to(tl.float32)
        tmp55 = tmp52 * tmp54
        tmp56 = tl.full(tmp55.shape, 0, tmp55.dtype)
        tmp57 = tl.where(tmp2, tmp55, tmp56)
        tmp58 = tl.broadcast_to(tmp57, [XBLOCK, R0_BLOCK])
        tmp60 = _tmp59 + tmp58
        _tmp59 = tl.where(r0_mask & xmask, tmp60, _tmp59)
        tmp61 = tl.load(in_ptr10 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tmp52 * tmp62
        tmp64 = tl.full(tmp63.shape, 0, tmp63.dtype)
        tmp65 = tl.where(tmp2, tmp63, tmp64)
        tmp66 = tl.broadcast_to(tmp65, [XBLOCK, R0_BLOCK])
        tmp68 = _tmp67 + tmp66
        _tmp67 = tl.where(r0_mask & xmask, tmp68, _tmp67)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tmp35 = tl.sum(_tmp35, 1)[:, None]
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp49 = tl.sum(_tmp49, 1)[:, None]
    tmp59 = tl.sum(_tmp59, 1)[:, None]
    tmp67 = tl.sum(_tmp67, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
    tl.store(out_ptr2 + (x3), tmp27, xmask)
    tl.store(out_ptr3 + (x3), tmp35, xmask)
    tl.store(out_ptr4 + (x3), tmp42, xmask)
    tl.store(out_ptr5 + (x3), tmp49, xmask)
    tl.store(out_ptr6 + (x3), tmp59, xmask)
    tl.store(out_ptr7 + (x3), tmp67, xmask)
