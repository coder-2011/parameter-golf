
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_copy_embedding_fill_lift_fresh_mul_remainder_select_slice_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 262144000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_copy_embedding_fill_lift_fresh_mul_remainder_select_slice_sum_1(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 128) % 1000)
    x2 = xindex // 128000
    x0 = (xindex % 128)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-7) + 7*x1 + 7008*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full([1], 8191, tl.int64)
    tmp5 = (tmp3 % tmp4)
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = tmp5 != tmp6
    tmp8 = (libdevice.signbit(tmp5) != 0) if (tmp5).dtype is tl.float32 else tmp5 < 0
    tmp9 = (libdevice.signbit(tmp4) != 0) if (tmp4).dtype is tl.float32 else tmp4 < 0
    tmp10 = tmp8 != tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tmp5 + tmp4
    tmp13 = tl.where(tmp11, tmp12, tmp5)
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = tmp0 == tmp18
    tmp20 = tl.full([1], 0, tl.int64)
    tmp21 = tl.where(tmp19, tmp20, tmp20)
    tmp22 = tl.where(tmp2, tmp17, tmp21)
    tmp23 = tmp22 + tmp20
    tmp24 = tl.full([XBLOCK], 57344, tl.int32)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp23 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp23)
    tl.device_assert((0 <= tmp27) & (tmp27 < 57344), "index out of bounds: 0 <= tmp27 < 57344")
    tmp29 = tl.load(in_ptr1 + (x0 + 128*tmp27), None).to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tl.load(in_ptr0 + ((-6) + 7*x1 + 7008*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp32 = (tmp31 % tmp4)
    tmp33 = tmp32 != tmp6
    tmp34 = (libdevice.signbit(tmp32) != 0) if (tmp32).dtype is tl.float32 else tmp32 < 0
    tmp35 = tmp34 != tmp9
    tmp36 = tmp33 & tmp35
    tmp37 = tmp32 + tmp4
    tmp38 = tl.where(tmp36, tmp37, tmp32)
    tmp39 = tmp38 + tmp14
    tmp40 = tl.full(tmp39.shape, 0, tmp39.dtype)
    tmp41 = tl.where(tmp2, tmp39, tmp40)
    tmp42 = tl.where(tmp2, tmp41, tmp21)
    tmp43 = tl.full([1], 8192, tl.int64)
    tmp44 = tmp42 + tmp43
    tmp45 = tmp44 + tmp24
    tmp46 = tmp44 < 0
    tmp47 = tl.where(tmp46, tmp45, tmp44)
    tl.device_assert((0 <= tmp47) & (tmp47 < 57344), "index out of bounds: 0 <= tmp47 < 57344")
    tmp49 = tl.load(in_ptr1 + (x0 + 128*tmp47), None).to(tl.float32)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp30 + tmp50
    tmp52 = tl.load(in_ptr0 + ((-5) + 7*x1 + 7008*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp53 = (tmp52 % tmp4)
    tmp54 = tmp53 != tmp6
    tmp55 = (libdevice.signbit(tmp53) != 0) if (tmp53).dtype is tl.float32 else tmp53 < 0
    tmp56 = tmp55 != tmp9
    tmp57 = tmp54 & tmp56
    tmp58 = tmp53 + tmp4
    tmp59 = tl.where(tmp57, tmp58, tmp53)
    tmp60 = tmp59 + tmp14
    tmp61 = tl.full(tmp60.shape, 0, tmp60.dtype)
    tmp62 = tl.where(tmp2, tmp60, tmp61)
    tmp63 = tl.where(tmp2, tmp62, tmp21)
    tmp64 = tl.full([1], 16384, tl.int64)
    tmp65 = tmp63 + tmp64
    tmp66 = tmp65 + tmp24
    tmp67 = tmp65 < 0
    tmp68 = tl.where(tmp67, tmp66, tmp65)
    tl.device_assert((0 <= tmp68) & (tmp68 < 57344), "index out of bounds: 0 <= tmp68 < 57344")
    tmp70 = tl.load(in_ptr1 + (x0 + 128*tmp68), None).to(tl.float32)
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tmp51 + tmp71
    tmp73 = tl.load(in_ptr0 + ((-4) + 7*x1 + 7008*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp74 = (tmp73 % tmp4)
    tmp75 = tmp74 != tmp6
    tmp76 = (libdevice.signbit(tmp74) != 0) if (tmp74).dtype is tl.float32 else tmp74 < 0
    tmp77 = tmp76 != tmp9
    tmp78 = tmp75 & tmp77
    tmp79 = tmp74 + tmp4
    tmp80 = tl.where(tmp78, tmp79, tmp74)
    tmp81 = tmp80 + tmp14
    tmp82 = tl.full(tmp81.shape, 0, tmp81.dtype)
    tmp83 = tl.where(tmp2, tmp81, tmp82)
    tmp84 = tl.where(tmp2, tmp83, tmp21)
    tmp85 = tl.full([1], 24576, tl.int64)
    tmp86 = tmp84 + tmp85
    tmp87 = tmp86 + tmp24
    tmp88 = tmp86 < 0
    tmp89 = tl.where(tmp88, tmp87, tmp86)
    tl.device_assert((0 <= tmp89) & (tmp89 < 57344), "index out of bounds: 0 <= tmp89 < 57344")
    tmp91 = tl.load(in_ptr1 + (x0 + 128*tmp89), None).to(tl.float32)
    tmp92 = tmp91.to(tl.float32)
    tmp93 = tmp72 + tmp92
    tmp94 = tl.load(in_ptr0 + ((-3) + 7*x1 + 7008*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp95 = (tmp94 % tmp4)
    tmp96 = tmp95 != tmp6
    tmp97 = (libdevice.signbit(tmp95) != 0) if (tmp95).dtype is tl.float32 else tmp95 < 0
    tmp98 = tmp97 != tmp9
    tmp99 = tmp96 & tmp98
    tmp100 = tmp95 + tmp4
    tmp101 = tl.where(tmp99, tmp100, tmp95)
    tmp102 = tmp101 + tmp14
    tmp103 = tl.full(tmp102.shape, 0, tmp102.dtype)
    tmp104 = tl.where(tmp2, tmp102, tmp103)
    tmp105 = tl.where(tmp2, tmp104, tmp21)
    tmp106 = tl.full([1], 32768, tl.int64)
    tmp107 = tmp105 + tmp106
    tmp108 = tmp107 + tmp24
    tmp109 = tmp107 < 0
    tmp110 = tl.where(tmp109, tmp108, tmp107)
    tl.device_assert((0 <= tmp110) & (tmp110 < 57344), "index out of bounds: 0 <= tmp110 < 57344")
    tmp112 = tl.load(in_ptr1 + (x0 + 128*tmp110), None).to(tl.float32)
    tmp113 = tmp112.to(tl.float32)
    tmp114 = tmp93 + tmp113
    tmp115 = tl.load(in_ptr0 + ((-2) + 7*x1 + 7008*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp116 = (tmp115 % tmp4)
    tmp117 = tmp116 != tmp6
    tmp118 = (libdevice.signbit(tmp116) != 0) if (tmp116).dtype is tl.float32 else tmp116 < 0
    tmp119 = tmp118 != tmp9
    tmp120 = tmp117 & tmp119
    tmp121 = tmp116 + tmp4
    tmp122 = tl.where(tmp120, tmp121, tmp116)
    tmp123 = tmp122 + tmp14
    tmp124 = tl.full(tmp123.shape, 0, tmp123.dtype)
    tmp125 = tl.where(tmp2, tmp123, tmp124)
    tmp126 = tl.where(tmp2, tmp125, tmp21)
    tmp127 = tl.full([1], 40960, tl.int64)
    tmp128 = tmp126 + tmp127
    tmp129 = tmp128 + tmp24
    tmp130 = tmp128 < 0
    tmp131 = tl.where(tmp130, tmp129, tmp128)
    tl.device_assert((0 <= tmp131) & (tmp131 < 57344), "index out of bounds: 0 <= tmp131 < 57344")
    tmp133 = tl.load(in_ptr1 + (x0 + 128*tmp131), None).to(tl.float32)
    tmp134 = tmp133.to(tl.float32)
    tmp135 = tmp114 + tmp134
    tmp136 = tl.load(in_ptr0 + ((-1) + 7*x1 + 7008*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp137 = (tmp136 % tmp4)
    tmp138 = tmp137 != tmp6
    tmp139 = (libdevice.signbit(tmp137) != 0) if (tmp137).dtype is tl.float32 else tmp137 < 0
    tmp140 = tmp139 != tmp9
    tmp141 = tmp138 & tmp140
    tmp142 = tmp137 + tmp4
    tmp143 = tl.where(tmp141, tmp142, tmp137)
    tmp144 = tmp143 + tmp14
    tmp145 = tl.full(tmp144.shape, 0, tmp144.dtype)
    tmp146 = tl.where(tmp2, tmp144, tmp145)
    tmp147 = tl.where(tmp2, tmp146, tmp21)
    tmp148 = tl.full([1], 49152, tl.int64)
    tmp149 = tmp147 + tmp148
    tmp150 = tmp149 + tmp24
    tmp151 = tmp149 < 0
    tmp152 = tl.where(tmp151, tmp150, tmp149)
    tl.device_assert((0 <= tmp152) & (tmp152 < 57344), "index out of bounds: 0 <= tmp152 < 57344")
    tmp154 = tl.load(in_ptr1 + (x0 + 128*tmp152), None).to(tl.float32)
    tmp155 = tmp154.to(tl.float32)
    tmp156 = tmp135 + tmp155
    tmp157 = tl.full([1], 0.37796447300922725, tl.float32)
    tmp158 = tmp156 * tmp157
    tmp159 = tmp158.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp159, None)
