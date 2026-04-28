import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_gpt_parcae as pg


def test_liger_ce_cpu_falls_back_to_torch_cross_entropy():
    torch.manual_seed(123)
    logits = torch.randn(11, 23, requires_grad=True)
    target = torch.randint(0, 23, (11,))
    ref_logits = logits.detach().clone().requires_grad_(True)

    loss = pg.liger_cross_entropy(logits, target, assume_no_ignore=True)
    ref = F.cross_entropy(ref_logits, target)
    loss.backward()
    ref.backward()

    assert torch.allclose(loss, ref)
    assert torch.allclose(logits.grad, ref_logits.grad)


def test_liger_ce_cpu_fallback_applies_softcap_before_cross_entropy():
    torch.manual_seed(123)
    logits = (torch.randn(11, 23) * 2.0).requires_grad_(True)
    target = torch.randint(0, 23, (11,))
    ref_logits = logits.detach().clone().requires_grad_(True)
    softcap = 7.0

    loss = pg.liger_cross_entropy(logits, target, assume_no_ignore=True, softcap=softcap)
    ref = F.cross_entropy(softcap * torch.tanh(ref_logits / softcap), target)
    loss.backward()
    ref.backward()

    assert torch.allclose(loss, ref)
    assert torch.allclose(logits.grad, ref_logits.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton CE kernel")
def test_liger_ce_cuda_matches_torch_cross_entropy_with_ignore_index():
    torch.manual_seed(123)
    logits = torch.randn(19, 17, device="cuda", dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, 17, (19,), device="cuda")
    target[::5] = -100
    ref_logits = logits.detach().clone().requires_grad_(True)

    loss = pg.liger_cross_entropy(logits, target, ignore_index=-100)
    ref = F.cross_entropy(ref_logits, target, ignore_index=-100)
    loss.backward()
    ref.backward()
    torch.cuda.synchronize()

    assert torch.allclose(loss, ref, atol=3e-4, rtol=1e-4)
    assert torch.allclose(logits.grad, ref_logits.grad, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available() or pg.LigerCrossEntropyLoss is None,
    reason="CUDA and real Liger CE required",
)
def test_liger_ce_cuda_softcap_matches_torch_cross_entropy():
    torch.manual_seed(123)
    logits = (torch.randn(19, 17, device="cuda", dtype=torch.float32) * 2.0).requires_grad_(True)
    target = torch.randint(0, 17, (19,), device="cuda")
    ref_logits = logits.detach().clone().requires_grad_(True)
    softcap = 7.0

    loss = pg.liger_cross_entropy(logits, target, assume_no_ignore=True, softcap=softcap)
    ref = F.cross_entropy(softcap * torch.tanh(ref_logits / softcap), target)
    loss.backward()
    ref.backward()
    torch.cuda.synchronize()

    assert torch.allclose(loss, ref, atol=3e-4, rtol=1e-4)
    assert torch.allclose(logits.grad, ref_logits.grad, atol=1e-5, rtol=1e-4)
