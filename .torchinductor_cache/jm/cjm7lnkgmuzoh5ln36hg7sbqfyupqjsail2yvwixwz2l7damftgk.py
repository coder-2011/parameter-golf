
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 11, 'num_store': 1, 'num_reduction': 2, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, out_ptr4, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
    xnumel = 131072
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * RSPLIT_SIZE
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    accum0 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    accum1 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        x0 = xindex
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, other=0.0)
        tmp6 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr5 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr7 + (r0_1 + 384*x0), r0_mask, other=0.0)
        tmp44 = tl.load(in_ptr8 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp46 = tl.load(in_ptr9 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp49 = tl.load(in_ptr10 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tl.full([1, 1], 0.5, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = tl.where(r0_mask, tmp9, 0)
        tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
        tmp13 = r0_1
        tmp14 = tl.full([1, 1], 12, tl.int64)
        tmp15 = tmp13 < tmp14
        tmp16 = tl.load(in_ptr4 + (r0_1 + 12*x0), r0_mask & tmp15, other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp15, tmp17, tmp18)
        tmp20 = tl.full([1, 1], 0.0, tl.float32)
        tmp21 = tl.where(tmp15, tmp19, tmp20)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 + tmp23
        tmp25 = tmp24 * tmp4
        tmp27 = tmp25 * tmp26
        tmp28 = tmp2 * tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
        tmp31 = tl.where(r0_mask, tmp29, 0)
        tmp32 = tl.sum(tmp31, 1)[:, None].to(tl.float32)
        tmp34 = tl.full([1, 1], 0.0026041666666666665, tl.float32)
        tmp35 = tmp2 * tmp34
        tmp36 = tmp35 * tmp12
        tmp37 = tmp7 - tmp36
        tmp38 = tmp37 * tmp1
        tmp39 = tmp33 + tmp38
        tmp40 = tmp35 * tmp32
        tmp41 = tmp27 - tmp40
        tmp42 = tmp41 * tmp1
        tmp43 = tmp39 + tmp42
        tmp45 = tmp43 * tmp44
        tmp47 = tmp45 * tmp46
        tmp48 = tmp47.to(tl.float32)
        tmp50 = tmp45 * tmp49
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp5 * tmp2
        tmp53 = tmp25 * tmp2
        tl.store(out_ptr2 + (r0_1 + 384*x0), tmp43, r0_mask)
        tl.store(out_ptr3 + (r0_1 + 384*x0), tmp48, r0_mask)
        tl.store(out_ptr4 + (r0_1 + 384*x0), tmp51, r0_mask)
        tmp54 = tl.sum(tmp52, 0)
        tmp55 = accum0 + tmp54
        accum0 = tmp55
        tmp56 = tl.sum(tmp53, 0)
        tmp57 = accum1 + tmp56
        accum1 = tmp57
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
