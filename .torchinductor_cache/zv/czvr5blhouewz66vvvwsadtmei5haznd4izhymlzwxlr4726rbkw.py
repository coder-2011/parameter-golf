
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'i64', 'r0_numel': 'i64', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_div_mul_nll_loss_forward_sub_tanh_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1032, 'r0_': 5600000}}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy_div_mul_nll_loss_forward_sub_tanh_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 43
    r0_numel = 8140
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    rbase = r0_base
    x0 = xindex
    _tmp33 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp39 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + 8140*x0
        tmp1 = tl.full([1, 1], 350000, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r0_1 + 8140*x0), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full([1, 1], -100, tl.int64)
        tmp5 = tmp3 != tmp4
        tmp6 = tl.full([1, 1], 0, tl.int64)
        tmp7 = tl.where(tmp5, tmp3, tmp6)
        tmp8 = tl.full([1, 1], 8192, tl.int32)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp7 < 0
        tmp11 = tl.where(tmp10, tmp9, tmp7)
        tl.device_assert(((0 <= tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK]) < 8192)) | ~(r0_mask & tmp2 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK]) < 8192")
        tmp13 = tl.load(in_ptr1 + (tmp11 + 8192*r0_1 + 66682880*x0), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.full([1, 1], 0.03333333333333333, tl.float32)
        tmp16 = tmp14 * tmp15
        tmp17 = libdevice.tanh(tmp16)
        tmp18 = tl.full([1, 1], 1.0, tl.float32)
        tmp19 = tmp17 * tmp18
        tmp20 = tl.load(in_ptr2 + (r0_1 + 8140*x0), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = tl.full([1, 1], 30.0, tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = tl.load(in_ptr3 + (r0_1 + 8140*x0), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl_math.log(tmp24)
        tmp26 = tmp23 - tmp25
        tmp27 = -tmp26
        tmp28 = tl.full([1, 1], 0.0, tl.float32)
        tmp29 = tl.where(tmp5, tmp27, tmp28)
        tmp30 = tl.full(tmp29.shape, 0, tmp29.dtype)
        tmp31 = tl.where(tmp2, tmp29, tmp30)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(r0_mask & xmask, tmp34, _tmp33)
        tmp35 = tmp5.to(tl.int64)
        tmp36 = tl.full(tmp35.shape, 0, tmp35.dtype)
        tmp37 = tl.where(tmp2, tmp35, tmp36)
        tmp38 = tl.broadcast_to(tmp37, [XBLOCK, R0_BLOCK])
        tmp40 = _tmp39 + tmp38
        _tmp39 = tl.where(r0_mask & xmask, tmp40, _tmp39)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tmp39 = tl.sum(_tmp39, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp33, xmask)
    tl.store(out_ptr1 + (x0), tmp39, xmask)
