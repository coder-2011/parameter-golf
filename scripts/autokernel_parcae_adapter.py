#!/usr/bin/env python3
"""AutoKernel model adapter for profiling Parcae without editing train_gpt_parcae.py.

AutoKernel's profiler instantiates a no-argument model class and calls
``model(input_ids=...)``.  The Parcae training model expects both input and
target token IDs, so this wrapper supplies shifted labels and keeps all tuning
configuration in environment variables.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _setdefault_env() -> None:
    # Keep the default profile small enough to run quickly while still hitting
    # the same Parcae module paths as training. Callers can override any knob.
    defaults = {
        "VOCAB_SIZE": "1024",
        "TRAIN_SEQ_LEN": "128",
        "EVAL_SEQ_LEN": "128",
        "MODEL_DIM": "256",
        "RECURRENT_DIM": "256",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "2",
        "MLP_MULT": "3",
        "N_LAYERS_IN_PRELUDE": "1",
        "N_LAYERS_IN_RECURRENT_BLOCK": "2",
        "N_LAYERS_IN_CODA": "1",
        "RECURRENT_NUM_HEADS": "4",
        "RECURRENT_INTERMEDIATE_DIM": "768",
        "MEAN_RECURRENCE": "2",
        "MEAN_BACKPROP_DEPTH": "1",
        "STATE_INIT": "like-init",
        "QK_NORM": "1",
        "ROPE_DIMS": "16",
        "TRIDAO_PACKED_ROPE": "1",
        "LIGER_CE": "1",
        "LIGER_FUSED_CE": "1",
        "QAT_BITS": "0",
        "BIGRAM_HASH_BUCKETS": "4096",
        "BIGRAM_HASH_DIM": "128",
        "BIGRAM_HASH_HEADS": "2",
        "BIGRAM_HASH_GATE": "1",
        "COMPILE_MODEL": "0",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


_setdefault_env()


def _prefer_stdlib_profile_module() -> None:
    # AutoKernel's entrypoint is named profile.py. When train_gpt_parcae imports
    # torch._dynamo, cProfile imports stdlib profile; without this guard Python
    # can resolve that import back to AutoKernel's script.
    stdlib_profile = Path(sysconfig.get_path("stdlib")) / "profile.py"
    current = sys.modules.get("profile")
    if current is not None and Path(getattr(current, "__file__", "")).resolve() == stdlib_profile:
        return
    import importlib.util

    spec = importlib.util.spec_from_file_location("profile", stdlib_profile)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["profile"] = module
    spec.loader.exec_module(module)


_prefer_stdlib_profile_module()

from train_gpt_parcae import (  # noqa: E402
    GPT,
    Hyperparameters,
    prepare_torchao_qat,
    restore_low_dim_params_to_fp32,
)


class ParcaeAutoKernelModel(nn.Module):
    """No-argument Parcae wrapper matching AutoKernel's profiler contract."""

    def __init__(self) -> None:
        super().__init__()
        self.h = Hyperparameters()
        self.vocab_size = self.h.vocab_size
        self.mode = os.environ.get("AUTOKERNEL_PARCAE_MODE", "loss").strip().lower()
        if self.mode not in {"loss", "logits"}:
            raise ValueError("AUTOKERNEL_PARCAE_MODE must be 'loss' or 'logits'")
        self.compile_forward = bool(int(os.environ.get("AUTOKERNEL_PARCAE_COMPILE", "0")))
        self.model = GPT(self.h)
        restore_low_dim_params_to_fp32(self.model)
        prepare_torchao_qat(self.model, self.h)
        qat_step = int(os.environ.get("AUTOKERNEL_PARCAE_QAT_STEP", "0"))
        self.model.set_training_step(qat_step)
        self._compiled_model: nn.Module | None = None

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        restore_low_dim_params_to_fp32(self.model)
        return result

    def _active_model(self) -> nn.Module:
        if not self.compile_forward:
            return self.model
        if self._compiled_model is None:
            self._compiled_model = torch.compile(self.model, dynamic=False)
        return self._compiled_model

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.shape[1] > self.h.train_seq_len:
            raise ValueError(
                f"AutoKernel input sequence length {input_ids.shape[1]} exceeds "
                f"TRAIN_SEQ_LEN={self.h.train_seq_len}; set TRAIN_SEQ_LEN to match "
                "AUTOKERNEL_INPUT_SHAPE."
            )
        input_ids = input_ids.to(dtype=torch.long).remainder(self.vocab_size)
        with torch.autocast(
            device_type=input_ids.device.type,
            dtype=torch.bfloat16,
            enabled=input_ids.device.type == "cuda",
        ):
            if self.mode == "logits":
                return self._active_model().forward_logits(input_ids)

            labels = torch.empty_like(input_ids)
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = input_ids[:, -1]
            return self._active_model()(input_ids, labels)
