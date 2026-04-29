#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_data_shard(path: Path) -> torch.Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"unexpected shard header for {path}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * np.dtype("<u2").itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(f"shard size mismatch for {path}: expected {expected_size}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"short read for {path}")
    return torch.from_numpy(tokens.astype(np.int64, copy=False))


def first_train_shard(data_path: Path) -> Path:
    shards = sorted(data_path.glob("fineweb_train_*.bin"))
    if not shards:
        raise FileNotFoundError(f"no fineweb_train_*.bin shards found in {data_path}")
    return shards[0]


class HedgehogFeatureMap(nn.Module):
    def __init__(self, head_dim: int, activation: str):
        super().__init__()
        self.activation = activation
        self.proj = nn.Linear(head_dim, head_dim)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.activation == "exp":
            return torch.cat((torch.exp(x), torch.exp(-x)), dim=-1)
        if self.activation == "softmax":
            return torch.cat((F.softmax(x, dim=-1), F.softmax(-x, dim=-1)), dim=-1)
        raise ValueError(f"unsupported Hedgehog activation: {self.activation}")


class SoftmaxCausalAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        y = torch.matmul(F.softmax(scores, dim=-1), v)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(y)


class HedgehogCausalAttention(nn.Module):
    def __init__(self, dim: int, heads: int, feature_activation: str):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_map = HedgehogFeatureMap(self.head_dim, feature_activation)
        self.k_map = HedgehogFeatureMap(self.head_dim, feature_activation)
        self.out = nn.Linear(dim, dim, bias=False)

    def qkv_heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        q, k, v = self.qkv_heads(x)
        q = self.q_map(q)
        k = self.k_map(k)
        k_prefix = k.cumsum(dim=2)
        kv_prefix = torch.einsum("bhtf,bhtd->bhtfd", k, v).cumsum(dim=2)
        numer = torch.einsum("bhtf,bhtfd->bhtd", q, kv_prefix)
        denom = torch.einsum("bhtf,bhtf->bht", q, k_prefix).unsqueeze(-1).clamp_min(1e-6)
        y = (numer / denom).transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(y)

    def attention_mimicry_loss(self, x: torch.Tensor) -> torch.Tensor:
        q, k, _ = self.qkv_heads(x)
        true_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        seq_len = x.shape[1]
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        true_log_weights = F.log_softmax(true_scores.masked_fill(~mask, float("-inf")), dim=-1)
        true_weights = true_log_weights.exp().masked_fill(~mask, 0.0)

        q_features = self.q_map(q)
        k_features = self.k_map(k)
        pred_scores = torch.matmul(q_features, k_features.transpose(-2, -1))
        pred_log_weights = F.log_softmax(pred_scores.masked_fill(~mask, float("-inf")), dim=-1)
        pred_log_weights = pred_log_weights.masked_fill(~mask, 0.0)
        return -(true_weights * pred_log_weights).sum(dim=-1).mean()


class TinyHedgehogLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        heads: int,
        depth: int,
        mlp_mult: int,
        attention: str,
        feature_activation: str,
    ):
        super().__init__()
        attn_cls = SoftmaxCausalAttention if attention == "softmax" else HedgehogCausalAttention
        self.emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            attn = attn_cls(dim, heads) if attention == "softmax" else attn_cls(dim, heads, feature_activation)
            self.blocks.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        attn,
                        nn.LayerNorm(dim),
                        nn.Sequential(
                            nn.Linear(dim, mlp_mult * dim),
                            nn.GELU(),
                            nn.Linear(mlp_mult * dim, dim),
                        ),
                    ]
                )
            )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx)
        for norm1, attn, norm2, mlp in self.blocks:
            x = x + attn(norm1(x))
            x = x + mlp(norm2(x))
        return self.head(self.norm(x))

    def attention_mimicry_loss(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx)
        losses = []
        for norm1, attn, norm2, mlp in self.blocks:
            attn_in = norm1(x)
            if isinstance(attn, HedgehogCausalAttention):
                losses.append(attn.attention_mimicry_loss(attn_in))
            x = x + attn(attn_in)
            x = x + mlp(norm2(x))
        if not losses:
            return x.new_zeros(())
        return torch.stack(losses).mean()


def next_batch(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    span = batch_size * (seq_len + 1)
    if tokens.numel() <= span:
        raise ValueError(f"need more than {span} tokens, found {tokens.numel()}")
    start = (step * span) % (tokens.numel() - span)
    chunk = tokens[start : start + span].view(batch_size, seq_len + 1)
    return chunk[:, :-1], chunk[:, 1:]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Hedgehog attention on local FineWeb token shards.")
    parser.add_argument("--data-path", type=Path, default=Path("data/datasets/fineweb10B_sp1024"))
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--attention", choices=("hedgehog", "softmax"), default="hedgehog")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--mlp-mult", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature-activation", choices=("softmax", "exp"), default="softmax")
    parser.add_argument("--mimicry-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError(f"--steps must be >= 1, got {args.steps}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    shard = first_train_shard(args.data_path)
    tokens = load_data_shard(shard)
    if int(tokens.max()) >= args.vocab_size:
        raise ValueError(f"token id {int(tokens.max())} exceeds vocab_size={args.vocab_size}")

    model = TinyHedgehogLM(
        vocab_size=args.vocab_size,
        dim=args.dim,
        heads=args.heads,
        depth=args.depth,
        mlp_mult=args.mlp_mult,
        attention=args.attention,
        feature_activation=args.feature_activation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    last_loss = math.nan
    last_grad_norm = math.nan
    for step in range(args.steps):
        x_cpu, y_cpu = next_batch(tokens, args.batch_size, args.seq_len, step)
        x = x_cpu.to(device)
        y = y_cpu.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        lm_loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))
        mimicry_loss = model.attention_mimicry_loss(x) if args.mimicry_weight else logits.new_zeros(())
        loss = lm_loss + args.mimicry_weight * mimicry_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
        last_grad_norm = float(grad_norm.detach())

    finite = bool(torch.isfinite(logits).all().item())
    print(
        f"attention={args.attention} feature_activation={args.feature_activation} "
        f"mimicry_weight={args.mimicry_weight:g} device={device.type} torch={torch.__version__} "
        f"shard={shard} tokens={tuple(x.shape)}"
    )
    print(
        f"logits={tuple(logits.shape)} finite_logits={finite} "
        f"loss={last_loss:.6f} lm_loss={float(lm_loss.detach()):.6f} "
        f"mimicry_loss={float(mimicry_loss.detach()):.6f} grad_norm={last_grad_norm:.6f}"
    )
    print(f"params={sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
