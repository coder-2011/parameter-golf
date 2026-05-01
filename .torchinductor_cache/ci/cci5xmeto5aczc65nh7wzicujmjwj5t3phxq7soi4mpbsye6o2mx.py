
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_min_copy_div_eq_expand_ge_masked_fill_mul_scalar_tensor_slice_squeeze_transpose_where_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 603979776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clamp_min_copy_div_eq_expand_ge_masked_fill_mul_scalar_tensor_slice_squeeze_transpose_where_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100663296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 48) % 16)
    x2 = xindex // 768
    x3 = (xindex % 768)
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 12, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-576) + x3 + 192*x2), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + ((-12) + x1 + 4*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.full([1], 1e-12, tl.float32)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = (tmp3 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp4 >= tmp5
    tmp10 = tl.load(in_ptr2 + ((-12) + x1 + 4*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full([1], 0.0, tl.float32)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp4 == tmp11
    tmp14 = tl.load(in_ptr3 + ((-576) + x3 + 192*x2), tmp2, other=0.0).to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = (tmp15 / tmp4)
    tmp17 = tl.where(tmp13, tmp11, tmp16)
    tmp18 = tmp12 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp8 + tmp19
    tmp21 = tl.load(in_ptr4 + ((-576) + x3 + 192*x2), tmp2, other=0.0).to(tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = tl.full([1], float("nan"), tl.float32)
    tmp26 = tl.where(tmp2, tmp24, tmp25)
    tl.store(out_ptr0 + (x4), tmp26, None)
