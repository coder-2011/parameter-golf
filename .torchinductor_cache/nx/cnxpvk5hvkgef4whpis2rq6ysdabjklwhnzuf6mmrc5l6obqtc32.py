
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=5, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_embedding_mul_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D4ED500704FBBDF1079C9014EF6AD77B088C60B861A53457972D5398CA90C866', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 262144000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_embedding_mul_sum_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 128
    x0 = (xindex % 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + 4*x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (2 + 4*x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (3 + 4*x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([XBLOCK], 32768, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert((0 <= tmp6) & (tmp6 < 32768), "index out of bounds: 0 <= tmp6 < 32768")
    tmp8 = tl.load(in_ptr1 + (x0 + 128*tmp6), None).to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tl.full([1], 8192, tl.int64)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp3
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.device_assert((0 <= tmp15) & (tmp15 < 32768), "index out of bounds: 0 <= tmp15 < 32768")
    tmp17 = tl.load(in_ptr1 + (x0 + 128*tmp15), None).to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp9 + tmp18
    tmp21 = tl.full([1], 16384, tl.int64)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22 + tmp3
    tmp24 = tmp22 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp22)
    tl.device_assert((0 <= tmp25) & (tmp25 < 32768), "index out of bounds: 0 <= tmp25 < 32768")
    tmp27 = tl.load(in_ptr1 + (x0 + 128*tmp25), None).to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp19 + tmp28
    tmp31 = tl.full([1], 24576, tl.int64)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp32 + tmp3
    tmp34 = tmp32 < 0
    tmp35 = tl.where(tmp34, tmp33, tmp32)
    tl.device_assert((0 <= tmp35) & (tmp35 < 32768), "index out of bounds: 0 <= tmp35 < 32768")
    tmp37 = tl.load(in_ptr1 + (x0 + 128*tmp35), None).to(tl.float32)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp29 + tmp38
    tmp40 = tl.full([1], 0.5, tl.float32)
    tmp41 = tmp39 * tmp40
    tmp42 = tmp41.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp42, None)
