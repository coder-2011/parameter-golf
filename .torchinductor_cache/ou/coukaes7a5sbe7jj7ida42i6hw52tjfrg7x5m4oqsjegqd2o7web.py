
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr2': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=7, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 10, 'num_store': 0, 'num_reduction': 2, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_unsqueeze_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr2, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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
        tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r0_1 + 384*x0), r0_mask, other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_out_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0)
        tmp33 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp39 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
        tmp46 = tl.load(in_ptr8 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.full([1, 1], 0.7071067811865475, tl.float32)
        tmp4 = tmp2 * tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = tl.where(r0_mask, tmp8, 0)
        tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
        tmp12 = r0_1
        tmp13 = tl.full([1, 1], 12, tl.int64)
        tmp14 = tmp12 < tmp13
        tmp15 = tl.load(in_ptr3 + (r0_1 + 12*x0), r0_mask & tmp14, other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp14, tmp16, tmp17)
        tmp19 = tl.full([1, 1], 0.0, tl.float32)
        tmp20 = tl.where(tmp14, tmp18, tmp19)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 + tmp22
        tmp24 = tmp23 * tmp3
        tmp26 = tmp24 * tmp25
        tmp27 = tmp0 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
        tmp30 = tl.where(r0_mask, tmp28, 0)
        tmp31 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
        tmp34 = tmp32 * tmp33
        tmp35 = tl.full([1, 1], 0.0026041666666666665, tl.float32)
        tmp36 = tmp0 * tmp35
        tmp37 = tmp36 * tmp11
        tmp38 = tmp6 - tmp37
        tmp40 = tmp38 * tmp39
        tmp41 = tmp34 + tmp40
        tmp42 = tmp36 * tmp31
        tmp43 = tmp26 - tmp42
        tmp44 = tmp43 * tmp39
        tmp45 = tmp41 + tmp44
        tmp47 = tmp45 * tmp46
        tmp48 = tmp4 * tmp0
        tmp49 = tmp24 * tmp0
        tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp45, r0_mask)
        tl.store(out_ptr2 + (r0_1 + 384*x0), tmp47, r0_mask)
        tmp50 = tl.sum(tmp48, 0)
        tmp51 = accum0 + tmp50
        accum0 = tmp51
        tmp52 = tl.sum(tmp49, 0)
        tmp53 = accum1 + tmp52
        accum1 = tmp53
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
