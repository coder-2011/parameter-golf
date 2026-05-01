
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=7, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 5594400, 'x': 44755200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 349650
    xnumel = 8
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = (yindex % 999)
    y1 = yindex // 999
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + y0 + 1000*y1), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (y0 + 1000*y1), ymask, eviction_policy='evict_last')
    tmp1 = x2
    tmp2 = tl.full([1, 1], 4, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.full([1, 1], 2, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.full([1, 1], 1, tl.int64)
    tmp7 = tmp1 < tmp6
    tmp8 = tl.full([1, 1], 36313, tl.int64)
    tmp9 = tl.full([1, 1], 17491, tl.int64)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tl.full([1, 1], 3, tl.int64)
    tmp12 = tmp1 < tmp11
    tmp13 = tl.full([1, 1], 52973, tl.int64)
    tmp14 = tl.full([1, 1], 29837, tl.int64)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tl.where(tmp5, tmp10, tmp15)
    tmp17 = tl.full([1, 1], 6, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tl.full([1, 1], 5, tl.int64)
    tmp20 = tmp1 < tmp19
    tmp21 = tl.full([1, 1], 44497, tl.int64)
    tmp22 = tl.full([1, 1], 62137, tl.int64)
    tmp23 = tl.where(tmp20, tmp21, tmp22)
    tmp24 = tl.full([1, 1], 7, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.full([1, 1], 24071, tl.int64)
    tmp27 = tl.full([1, 1], 57223, tl.int64)
    tmp28 = tl.where(tmp25, tmp26, tmp27)
    tmp29 = tl.where(tmp18, tmp23, tmp28)
    tmp30 = tl.where(tmp3, tmp16, tmp29)
    tmp31 = tmp0 * tmp30
    tmp33 = tl.full([1, 1], 27191, tl.int64)
    tmp34 = tl.full([1, 1], 43889, tl.int64)
    tmp35 = tl.where(tmp7, tmp33, tmp34)
    tmp36 = tl.full([1, 1], 19937, tl.int64)
    tmp37 = tl.full([1, 1], 60271, tl.int64)
    tmp38 = tl.where(tmp12, tmp36, tmp37)
    tmp39 = tl.where(tmp5, tmp35, tmp38)
    tmp40 = tl.full([1, 1], 34583, tl.int64)
    tmp41 = tl.full([1, 1], 49331, tl.int64)
    tmp42 = tl.where(tmp20, tmp40, tmp41)
    tmp43 = tl.full([1, 1], 15461, tl.int64)
    tmp44 = tl.full([1, 1], 41077, tl.int64)
    tmp45 = tl.where(tmp25, tmp43, tmp44)
    tmp46 = tl.where(tmp18, tmp42, tmp45)
    tmp47 = tl.where(tmp3, tmp39, tmp46)
    tmp48 = tmp32 * tmp47
    tmp49 = tmp31 ^ tmp48
    tl.store(out_ptr0 + (x2 + 8*y0 + 8000*y1), tmp49, xmask & ymask)
