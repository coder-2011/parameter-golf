
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*i1', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*i64', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_embedding_dense_backward_mul_rsub_sigmoid_slice_slice_backward_sum_unsqueeze_view_8', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': True, 'num_load': 13, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_embedding_dense_backward_mul_rsub_sigmoid_slice_slice_backward_sum_unsqueeze_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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
    r0_2 = r0_index
    accum0 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    accum1 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        x3 = xindex
        x0 = (xindex % 1024)
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_2 + 384*x3), r0_mask, other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr3 + (r0_2 + 384*x3), r0_mask, other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr4 + (r0_2 + 384*x3), r0_mask, other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
        tmp41 = tl.load(in_ptr6 + (r0_2 + 384*x3), r0_mask, other=0.0).to(tl.float32)
        tmp45 = tl.load(in_ptr7 + (r0_2 + 384*x3), r0_mask, other=0.0).to(tl.float32)
        tmp46 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last').to(tl.int1)
        tmp51 = tl.load(in_ptr8 + (r0_2 + 384*x3), r0_mask, other=0.0).to(tl.float32)
        tmp52 = tl.load(in_ptr9 + (0))
        tmp53 = tl.broadcast_to(tmp52, [1, 1])
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.sigmoid(tmp2)
        tmp4 = tl.full([1, 1], 1.0, tl.float32)
        tmp5 = tmp4 - tmp3
        tmp6 = tmp0 * tmp5
        tmp7 = x0
        tmp8 = tl.full([1, 1], 1023, tl.int64)
        tmp9 = tmp7 < tmp8
        tmp10 = tl.load(in_ptr0 + (384 + r0_2 + 384*x3), r0_mask & tmp9, other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.sigmoid(tmp12)
        tmp14 = tmp10 * tmp13
        tmp15 = tl.load(in_ptr2 + (tl.broadcast_to(1 + x3, [XBLOCK, R0_BLOCK])), r0_mask & tmp9, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 * tmp16
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp9, tmp17, tmp18)
        tmp20 = tl.full([1, 1], 0.0, tl.float32)
        tmp21 = tl.where(tmp9, tmp19, tmp20)
        tmp22 = tmp6 + tmp21
        tmp25 = tl.sigmoid(tmp24)
        tmp26 = tmp23 * tmp25
        tmp27 = tmp22 * tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
        tmp31 = tl.where(r0_mask, tmp29, 0)
        tmp32 = tl.sum(tmp31, 1)[:, None].to(tl.float32)
        tmp34 = tl.full([1, 1], 8192, tl.int32)
        tmp35 = tmp33 + tmp34
        tmp36 = tmp33 < 0
        tmp37 = tl.where(tmp36, tmp35, tmp33)
        tl.device_assert((0 <= tmp37) & (tmp37 < 8192), "index out of bounds: 0 <= tmp37 < 8192")
        tmp39 = tl.full([1, 1], -1, tl.int64)
        tmp40 = tmp33 == tmp39
        tmp42 = tmp22 + tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tl.where(tmp40, tmp20, tmp43)
        tmp47 = tmp46.to(tl.float32)
        tmp48 = tmp45 * tmp47
        tmp49 = tmp0 * tmp48
        tmp50 = tmp49.to(tl.float32)
        tmp54 = tmp53.to(tl.float32)
        tmp55 = tmp26 * tmp54
        tmp56 = tmp51 + tmp55
        tmp57 = tmp0 * tmp56
        tmp58 = tmp57.to(tl.float32)
        tl.atomic_add(out_ptr1 + (tl.broadcast_to(r0_2 + 384*tmp37, [XBLOCK, R0_BLOCK])), tmp44, r0_mask, sem='relaxed')
        tl.store(out_ptr0 + (x3), tmp32, None)
        tmp59 = tl.sum(tmp50, 0)
        tmp60 = accum0 + tmp59
        accum0 = tmp60
        tmp61 = tl.sum(tmp58, 0)
        tmp62 = accum1 + tmp61
        accum1 = tmp62
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
