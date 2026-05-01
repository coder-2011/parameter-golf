
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=7, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 11, 'num_store': 5, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2359296, 'r0_': 1207962624}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_view_add_cat_mul_ne_rsub_sigmoid_slice_unsqueeze_view_zeros_like_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    x0 = xindex
    x1 = (xindex % 1024)
    r0_3 = r0_index
    x2 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr1 + (r0_3 + 384*x0), r0_mask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr2 + (r0_3 + 384*x0), r0_mask, other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr3 + (r0_3 + 384*x0), r0_mask, other=0.0).to(tl.float32)
    tmp37 = tl.load(in_ptr4 + (0))
    tmp38 = tl.broadcast_to(tmp37, [1, 1])
    tmp59 = tl.load(in_ptr6 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = x1
    tmp4 = tl.full([1, 1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp3 < tmp1
    tmp7 = tl.full([1, 1], 0.0, tl.float32)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tmp3 >= tmp1
    tmp11 = tl.full([1, 1], 1024, tl.int64)
    tmp12 = tmp3 < tmp11
    tmp13 = tl.load(in_ptr1 + (r0_3 + 384*((-1) + x1) + 393216*x2), r0_mask & tmp10, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (r0_3 + 384*((-1) + x1) + 393216*x2), r0_mask & tmp10, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (r0_3 + 384*((-1) + x1) + 393216*x2), r0_mask & tmp10, other=0.0).to(tl.float32)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp14 * tmp16
    tmp18 = tl.load(in_ptr4 + (0))
    tmp19 = tl.broadcast_to(tmp18, [1, 1])
    tmp20 = tl.where(tmp10, tmp19, 0.0)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp17 * tmp21
    tmp23 = tmp13 + tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tl.where(tmp6, tmp9, tmp25)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tl.full([1, 1], 1.0, tl.float32)
    tmp31 = tmp30 - tmp29
    tmp35 = tl.sigmoid(tmp34)
    tmp36 = tmp33 * tmp35
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp36 * tmp39
    tmp41 = tmp32 + tmp40
    tmp42 = tmp31 * tmp41
    tmp43 = tmp2.to(tl.float32)
    tmp44 = tmp26 * tmp43
    tmp45 = tmp29 * tmp44
    tmp46 = tmp42 + tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp47 * tmp47
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, R0_BLOCK])
    tmp51 = tl.where(r0_mask, tmp49, 0)
    tmp52 = tl.sum(tmp51, 1)[:, None].to(tl.float32)
    tmp53 = tl.full([1, 1], 384.0, tl.float32)
    tmp54 = (tmp52 / tmp53)
    tmp55 = tl.full([1, 1], 1e-06, tl.float32)
    tmp56 = tmp54 + tmp55
    tmp57 = libdevice.rsqrt(tmp56)
    tmp58 = tmp47 * tmp57
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tmp58 * tmp61
    tmp63 = tmp62.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr1 + (r0_3 + 384*x0), tmp26, r0_mask)
    tl.store(out_ptr2 + (r0_3 + 384*x0), tmp46, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp57, None)
    tl.store(out_ptr3 + (r0_3 + 384*x0), tmp63, r0_mask)
