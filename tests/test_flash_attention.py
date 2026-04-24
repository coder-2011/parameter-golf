import torch
import torch.nn.functional as F

import utils.flash_attention as flash_attention_backend


def test_flash_attention_sdpa_fallback_matches_causal_gqa_reference():
    flash_attention_backend.set_backend_override("sdpa")
    q = torch.randn(2, 5, 4, 8)
    k = torch.randn(2, 5, 2, 8)
    v = torch.randn(2, 5, 2, 8)

    out = flash_attention_backend.flash_attn.flash_attn_func(q, k, v, causal=True)
    k_ref = k.transpose(1, 2).repeat_interleave(2, dim=1)
    v_ref = v.transpose(1, 2).repeat_interleave(2, dim=1)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k_ref,
        v_ref,
        is_causal=True,
    ).transpose(1, 2)

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)
    flash_attention_backend.set_backend_override(None)
