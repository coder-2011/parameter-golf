
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'out_ptr0': '*i64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_arange_bitwise_xor_copy_fill_lift_fresh_mul_remainder_select_slice_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 2097152, 'x': 25166016}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_bitwise_xor_copy_fill_lift_fresh_mul_remainder_select_slice_unsqueeze_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 12
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = (yindex % 1024)
    y3 = yindex
    x2 = xindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (tl.broadcast_to(y3, [YBLOCK, XBLOCK])), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x2, [YBLOCK, XBLOCK])), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + y3, [YBLOCK, XBLOCK])), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(x2, [YBLOCK, XBLOCK])), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 ^ tmp8
    tmp10 = tl.full([1, 1], 16383, tl.int64)
    tmp11 = (tmp9 % tmp10)
    tmp12 = tl.full([1, 1], 0, tl.int32)
    tmp13 = tmp11 != tmp12
    tmp14 = (libdevice.signbit(tmp11) != 0) if (tmp11).dtype is tl.float32 else tmp11 < 0
    tmp15 = (libdevice.signbit(tmp10) != 0) if (tmp10).dtype is tl.float32 else tmp10 < 0
    tmp16 = tmp14 != tmp15
    tmp17 = tmp13 & tmp16
    tmp18 = tmp11 + tmp10
    tmp19 = tl.where(tmp17, tmp18, tmp11)
    tmp20 = tl.full([1, 1], 1, tl.int64)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
    tmp23 = tl.where(tmp2, tmp21, tmp22)
    tmp24 = tl.full([1, 1], 0, tl.int32)
    tmp25 = tmp0 == tmp24
    tmp26 = tl.full([1, 1], 0, tl.int64)
    tmp27 = tl.where(tmp25, tmp26, tmp26)
    tmp28 = tl.where(tmp2, tmp23, tmp27)
    tmp29 = 16384*x2
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr0 + (x2 + 12*y3), tmp30, xmask & ymask)
