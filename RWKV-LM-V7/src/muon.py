from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz zeroth-power approximation used by Muon."""
    if G.ndim < 2:
        raise ValueError(f"Muon expects matrix-like gradients, got shape={tuple(G.shape)}")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if transposed else X


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float,
    ns_steps: int,
    nesterov: bool = True,
    row_normalize: bool = False,
) -> torch.Tensor:
    momentum.lerp_(grad, 1.0 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    if row_normalize:
        row_norms = update.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
        update = update / row_norms.to(update.dtype)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    """Matrix-only Muon optimizer ported from train_gpt_parcae.py."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        row_normalize: bool = False,
    ):
        params = list(params)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
                row_normalize=row_normalize,
                is_muon=True,
                schedule_weight_decay=False,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank:
                    if p.ndim != 2:
                        raise ValueError(f"Muon parameter must be 2D, got shape={tuple(p.shape)}")
                    g = p.grad if p.grad is not None else torch.zeros_like(p)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    g = muon_update(
                        g,
                        state["momentum_buffer"],
                        beta=momentum,
                        ns_steps=backend_steps,
                        nesterov=nesterov,
                        row_normalize=group.get("row_normalize", False),
                    )
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


def classify_rwkv_muon_parameters(named_parameters):
    groups = {
        "embed": [],
        "head": [],
        "matrix": [],
        "scalar": [],
        "excluded_2d": [],
    }
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if name == "emb.weight":
            groups["embed"].append((name, param))
        elif name == "head.weight":
            groups["head"].append((name, param))
        elif name.endswith(".att.r_k"):
            groups["scalar"].append((name, param))
            groups["excluded_2d"].append(name)
        elif param.ndim == 2:
            groups["matrix"].append((name, param))
        else:
            groups["scalar"].append((name, param))
    return groups
