
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm__to_copy_add_mul_select_unsqueeze_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 4, 'num_reduction': 1, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1048576, 'r0_': 1107302400}}
)
@triton.jit
def triton_red_fused__fused_rms_norm__to_copy_add_mul_select_unsqueeze_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tmp2 + tmp6
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp12 = tl.full([1, 1], 384.0, tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tl.full([1, 1], 1e-06, tl.float32)
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp16, None)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp17 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr0 + (384 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp17 * tmp18
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 * tmp22
        tmp24 = tmp19 + tmp23
        tmp25 = tmp24 * tmp16
        tmp27 = tmp25 * tmp26
        tmp28 = tl.full([1, 1], 0.7071067811865475, tl.float32)
        tmp29 = tmp27 * tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp32 = tmp25 * tmp31
        tmp33 = tmp32 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr0 + (r0_1 + 384*x0), tmp25, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 384*x0), tmp30, r0_mask)
        tl.store(out_ptr2 + (r0_1 + 384*x0), tmp34, r0_mask)
