
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=4, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_bitwise_xor_copy_fill_lift_fresh_mul_remainder_select_slice_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32768000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_bitwise_xor_copy_fill_lift_fresh_mul_remainder_select_slice_unsqueeze_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 4) % 1000)
    x3 = xindex // 4
    x0 = (xindex % 4)
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x3), tmp2, eviction_policy='evict_last', other=0.0)
    tmp4 = x0
    tmp5 = tl.full([1], 2, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = tl.full([1], 36313, tl.int64)
    tmp10 = tl.full([1], 17491, tl.int64)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.full([1], 3, tl.int64)
    tmp13 = tmp4 < tmp12
    tmp14 = tl.full([1], 52973, tl.int64)
    tmp15 = tl.full([1], 29837, tl.int64)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tl.where(tmp6, tmp11, tmp16)
    tmp18 = tmp3 * tmp17
    tmp19 = tl.load(in_ptr0 + ((-1) + x3), tmp2, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full([1], 27191, tl.int64)
    tmp21 = tl.full([1], 43889, tl.int64)
    tmp22 = tl.where(tmp8, tmp20, tmp21)
    tmp23 = tl.full([1], 19937, tl.int64)
    tmp24 = tl.full([1], 60271, tl.int64)
    tmp25 = tl.where(tmp13, tmp23, tmp24)
    tmp26 = tl.where(tmp6, tmp22, tmp25)
    tmp27 = tmp19 * tmp26
    tmp28 = tmp18 ^ tmp27
    tmp29 = tl.full([1], 8191, tl.int64)
    tmp30 = (tmp28 % tmp29)
    tmp31 = tl.full([1], 0, tl.int32)
    tmp32 = tmp30 != tmp31
    tmp33 = (libdevice.signbit(tmp30) != 0) if (tmp30).dtype is tl.float32 else tmp30 < 0
    tmp34 = (libdevice.signbit(tmp29) != 0) if (tmp29).dtype is tl.float32 else tmp29 < 0
    tmp35 = tmp33 != tmp34
    tmp36 = tmp32 & tmp35
    tmp37 = tmp30 + tmp29
    tmp38 = tl.where(tmp36, tmp37, tmp30)
    tmp39 = tmp38 + tmp7
    tmp40 = tl.full(tmp39.shape, 0, tmp39.dtype)
    tmp41 = tl.where(tmp2, tmp39, tmp40)
    tmp42 = tl.full([1], 0, tl.int32)
    tmp43 = tmp0 == tmp42
    tmp44 = tl.full([1], 0, tl.int64)
    tmp45 = tl.where(tmp43, tmp44, tmp44)
    tmp46 = tl.where(tmp2, tmp41, tmp45)
    tl.store(out_ptr0 + (x4), tmp46, None)
