
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_expand_mul_neg_sigmoid_squeeze_sum_transpose_unsqueeze_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 12, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6291456, 'r0_': 503316480}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_expand_mul_neg_sigmoid_squeeze_sum_transpose_unsqueeze_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 524288
    r0_numel = 48
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 96*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (2*x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (48 + r0_1 + 96*x0), r0_mask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (1 + 2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp19 = tl.load(in_ptr2 + (1 + 2*x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (2*x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (r0_1 + 96*x0), r0_mask, other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr3 + (1 + 2*x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (48 + r0_1 + 96*x0), r0_mask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr5 + (r0_1 + 48*x0), r0_mask, other=0.0)
    tmp34 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1, 1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp1 * tmp6
    tmp8 = -tmp7
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp13 * tmp3
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp12 * tmp16
    tmp18 = -tmp17
    tmp20 = tmp18 * tmp19
    tmp21 = tmp10 + tmp20
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 * tmp24
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 * tmp28
    tmp30 = tmp25 + tmp29
    tmp31 = tmp21 + tmp30
    tmp32 = -tmp31
    tmp35 = tl.full([1, 1], 1e-12, tl.float32)
    tmp36 = triton_helpers.maximum(tmp34, tmp35)
    tmp37 = (tmp33 / tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.where(r0_mask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (r0_1 + 48*x0), tmp31, r0_mask)
    tl.store(out_ptr1 + (x0), tmp42, None)
