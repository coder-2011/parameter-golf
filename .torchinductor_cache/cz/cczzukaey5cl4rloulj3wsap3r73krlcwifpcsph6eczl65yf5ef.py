
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=6, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sigmoid_slice_sub_transpose_unsqueeze_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1097600000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sigmoid_slice_sub_transpose_unsqueeze_view_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134400000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 384)
    x1 = xindex // 384
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x2 // 48), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (576 + 48*(x0 // 96) + 768*x1 + ((x0 % 48))), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x2 // 96), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 // 48), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1e-12, tl.float32)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = (tmp4 / tmp8)
    tmp10 = tmp2 * tmp9
    tmp11 = tmp1 - tmp10
    tmp13 = tl.full([1], 0.5, tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp11 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp18, xmask)
