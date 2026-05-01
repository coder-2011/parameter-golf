
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 57286656}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_xor_lift_fresh_mul_slice_unsqueeze_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3580416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 7) % 999)
    x2 = xindex // 6993
    x0 = (xindex % 7)
    x3 = (xindex % 6993)
    tmp0 = tl.load(in_ptr0 + (1 + x1 + 1000*x2), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (x1 + 1000*x2), xmask, eviction_policy='evict_last')
    tmp1 = x0
    tmp2 = tl.full([1], 3, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp1 < tmp6
    tmp8 = tl.full([1], 17491, tl.int64)
    tmp9 = tl.full([1], 52973, tl.int64)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tl.full([1], 36313, tl.int64)
    tmp12 = tl.where(tmp5, tmp11, tmp10)
    tmp13 = tl.full([1], 5, tl.int64)
    tmp14 = tmp1 < tmp13
    tmp15 = tl.full([1], 4, tl.int64)
    tmp16 = tmp1 < tmp15
    tmp17 = tl.full([1], 29837, tl.int64)
    tmp18 = tl.full([1], 44497, tl.int64)
    tmp19 = tl.where(tmp16, tmp17, tmp18)
    tmp20 = tl.full([1], 6, tl.int64)
    tmp21 = tmp1 < tmp20
    tmp22 = tl.full([1], 62137, tl.int64)
    tmp23 = tl.full([1], 24071, tl.int64)
    tmp24 = tl.where(tmp21, tmp22, tmp23)
    tmp25 = tl.where(tmp14, tmp19, tmp24)
    tmp26 = tl.where(tmp3, tmp12, tmp25)
    tmp27 = tmp0 * tmp26
    tmp29 = tl.full([1], 43889, tl.int64)
    tmp30 = tl.full([1], 19937, tl.int64)
    tmp31 = tl.where(tmp7, tmp29, tmp30)
    tmp32 = tl.full([1], 27191, tl.int64)
    tmp33 = tl.where(tmp5, tmp32, tmp31)
    tmp34 = tl.full([1], 60271, tl.int64)
    tmp35 = tl.full([1], 34583, tl.int64)
    tmp36 = tl.where(tmp16, tmp34, tmp35)
    tmp37 = tl.full([1], 49331, tl.int64)
    tmp38 = tl.full([1], 15461, tl.int64)
    tmp39 = tl.where(tmp21, tmp37, tmp38)
    tmp40 = tl.where(tmp14, tmp36, tmp39)
    tmp41 = tl.where(tmp3, tmp33, tmp40)
    tmp42 = tmp28 * tmp41
    tmp43 = tmp27 ^ tmp42
    tl.store(out_ptr0 + (x3 + 7008*x2), tmp43, xmask)
