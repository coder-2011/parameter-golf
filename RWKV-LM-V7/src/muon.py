"""PyTorch-backed Muon helpers for RWKV-v7.

Muon is only used for hidden 2D matrix parameters. Embeddings, output
heads, biases, vectors, and gain-like parameters remain on AdamW.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


def zeropower_via_newtonschulz5(*_args, **_kwargs):
    raise RuntimeError("RWKV Muon now delegates updates to torch.optim.Muon")


@dataclass
class MuonParamGroups:
    embed: list[torch.nn.Parameter]
    matrix: list[torch.nn.Parameter]
    scalar: list[torch.nn.Parameter]
    head: list[torch.nn.Parameter]
    excluded_2d_names: list[str]


class Muon(torch.optim.Muon):
    """Compatibility wrapper around ``torch.optim.Muon``.

    The local training code was wired against a small custom Muon class with
    ``backend_steps`` and ``row_normalize`` arguments. Keep that constructor
    stable, but delegate the optimizer step itself to PyTorch's implementation.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        backend_steps: int = 5,
        row_normalize: bool = False,
        nesterov: bool = True,
        compile_backend: bool = False,
        compile_muon_backend: bool = False,
    ) -> None:
        if row_normalize:
            raise ValueError("torch.optim.Muon does not support row-normalized updates")
        if compile_backend or compile_muon_backend:
            raise ValueError("torch.optim.Muon does not expose the local compile backend knob")

        sorted_params = sorted(list(params), key=lambda p: p.numel(), reverse=True)
        super().__init__(
            sorted_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=backend_steps,
            adjust_lr_fn="original",
        )
        for group in self.param_groups:
            group["is_muon"] = True
            group["backend_steps"] = backend_steps
            group["row_normalize"] = False
            group.setdefault("base_lr", group["lr"])


def _is_embedding_param(name: str) -> bool:
    return name == "emb.weight" or name.endswith(".emb.weight")


def _is_head_param(name: str) -> bool:
    return name == "head.weight" or name.endswith(".head.weight")


def _is_gain_like_2d_param(name: str) -> bool:
    return name == "att.r_k" or name.endswith(".att.r_k")


def split_muon_parameters(model: torch.nn.Module) -> MuonParamGroups:
    """Split RWKV parameters into Muon and AdamW-compatible groups."""
    embed: list[torch.nn.Parameter] = []
    matrix: list[torch.nn.Parameter] = []
    scalar: list[torch.nn.Parameter] = []
    head: list[torch.nn.Parameter] = []
    excluded_2d_names: list[str] = []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)

        if _is_embedding_param(name):
            embed.append(param)
        elif _is_head_param(name):
            head.append(param)
        elif param.ndim == 2 and _is_gain_like_2d_param(name):
            scalar.append(param)
            excluded_2d_names.append(name)
        elif param.ndim == 2:
            matrix.append(param)
        else:
            scalar.append(param)

    matrix.sort(key=lambda p: p.numel(), reverse=True)
    return MuonParamGroups(
        embed=embed,
        matrix=matrix,
        scalar=scalar,
        head=head,
        excluded_2d_names=excluded_2d_names,
    )


def classify_rwkv_muon_parameters(named_parameters):
    groups = {
        "embed": [],
        "head": [],
        "matrix": [],
        "scalar": [],
        "excluded_2d": [],
    }
    seen: set[int] = set()
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)

        if _is_embedding_param(name):
            groups["embed"].append((name, param))
        elif _is_head_param(name):
            groups["head"].append((name, param))
        elif param.ndim == 2 and _is_gain_like_2d_param(name):
            groups["scalar"].append((name, param))
            groups["excluded_2d"].append(name)
        elif param.ndim == 2:
            groups["matrix"].append((name, param))
        else:
            groups["scalar"].append((name, param))
    groups["matrix"].sort(key=lambda item: item[1].numel(), reverse=True)
    return groups
