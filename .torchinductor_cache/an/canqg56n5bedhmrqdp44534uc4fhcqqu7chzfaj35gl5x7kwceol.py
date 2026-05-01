
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_mul_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 404797440, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_mul_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
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
        tmp12 = tl.load(in_ptr2 + (x0 + 384*(((r0_2 + 376*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp3 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask & xmask, tmp19, _tmp18)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp18, xmask)
