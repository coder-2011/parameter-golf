#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_INTS = 256
SHARD_HEADER_BYTES = SHARD_HEADER_INTS * np.dtype("<i4").itemsize


@dataclass(frozen=True)
class TokenizerLuts:
    base_bytes: np.ndarray
    has_leading_space: np.ndarray
    is_boundary_token: np.ndarray


@dataclass(frozen=True)
class ScoreSpan:
    window_start: int
    window_end: int
    score_start: int
    score_end: int
    rel_start: int
    rel_end: int


def build_sentencepiece_luts(tokenizer_path: str | Path, vocab_size: int) -> TokenizerLuts:
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    sp_vocab_size = int(sp.vocab_size())
    if sp_vocab_size != vocab_size:
        raise ValueError(f"VOCAB_SIZE={vocab_size} does not match tokenizer vocab_size={sp_vocab_size}")

    base_bytes = np.zeros((vocab_size,), dtype=np.int16)
    has_leading_space = np.zeros((vocab_size,), dtype=np.bool_)
    is_boundary_token = np.ones((vocab_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token[token_id] = False
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))

    return TokenizerLuts(base_bytes, has_leading_space, is_boundary_token)


def token_byte_sum(prev_ids: np.ndarray, tgt_ids: np.ndarray, luts: TokenizerLuts) -> int:
    prev = np.asarray(prev_ids, dtype=np.int64)
    tgt = np.asarray(tgt_ids, dtype=np.int64)
    token_bytes = luts.base_bytes[tgt].astype(np.int64, copy=True)
    # SentencePiece "▁" contributes one real space byte only after a non-boundary token.
    token_bytes += (luts.has_leading_space[tgt] & ~luts.is_boundary_token[prev]).astype(np.int64)
    return int(token_bytes.sum(dtype=np.int64))


def bpb_from_sums(loss_sum: float, token_count: int, byte_count: int) -> tuple[float, float]:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if byte_count <= 0:
        raise ValueError("byte_count must be positive")
    val_loss = float(loss_sum) / float(token_count)
    val_bpb = float(loss_sum) / (math.log(2.0) * float(byte_count))
    return val_loss, val_bpb


def resolve_val_files(data_file: str | Path) -> list[Path]:
    path = Path(data_file)
    if path.is_dir():
        files = sorted(path.glob("fineweb_val_*.bin"))
        if not files:
            raise FileNotFoundError(f"No fineweb_val_*.bin shards found in {path}")
        return files
    files = [Path(p) for p in sorted(glob.glob(str(data_file)))]
    if files:
        return files
    if path.exists():
        return [path]
    raise FileNotFoundError(f"No validation shards found for {data_file}")


def load_u16_tokens(files: list[Path]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for file in files:
        header = np.fromfile(file, dtype="<i4", count=SHARD_HEADER_INTS)
        if header.size != SHARD_HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
            raise ValueError(f"Unexpected FineWeb shard header for {file}")
        num_tokens = int(header[2])
        expected_size = SHARD_HEADER_BYTES + num_tokens * np.dtype("<u2").itemsize
        if file.stat().st_size != expected_size:
            raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
        chunk = np.fromfile(file, dtype="<u2", count=num_tokens, offset=SHARD_HEADER_BYTES)
        if chunk.size != num_tokens:
            raise ValueError(f"Short read for {file}")
        chunks.append(chunk)
    return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


def iter_score_spans(total_targets: int, ctx_len: int, stride: int) -> list[ScoreSpan]:
    if total_targets <= 0:
        raise ValueError("total_targets must be positive")
    if ctx_len <= 0:
        raise ValueError("ctx_len must be positive")
    if stride <= 0 or stride > ctx_len:
        raise ValueError("stride must be in 1..ctx_len")

    spans: list[ScoreSpan] = []
    window_len = min(ctx_len, total_targets)
    for score_start in range(0, total_targets, stride):
        score_end = min(score_start + stride, total_targets)
        if total_targets <= ctx_len or score_end <= ctx_len:
            window_start = 0
            window_end = window_len
        else:
            window_end = score_end
            window_start = window_end - ctx_len
        rel_start = score_start - window_start
        rel_end = score_end - window_start
        if not (0 <= rel_start < rel_end <= window_end - window_start):
            raise AssertionError("invalid validation score span")
        spans.append(ScoreSpan(window_start, window_end, score_start, score_end, rel_start, rel_end))
    return spans


def _make_batch(tokens: np.ndarray, spans: list[ScoreSpan]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(span.window_end - span.window_start for span in spans)
    x_np = np.zeros((len(spans), max_len), dtype=np.int64)
    y_np = np.zeros((len(spans), max_len), dtype=np.int64)
    for row, span in enumerate(spans):
        window = tokens[span.window_start : span.window_end + 1].astype(np.int64, copy=False)
        x_np[row, : window.size - 1] = window[:-1]
        y_np[row, : window.size - 1] = window[1:]
    return torch.from_numpy(x_np), torch.from_numpy(y_np)


def load_rwkv_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(args.head_size)
    os.environ["RWKV_COMPILE_ON"] = "1" if args.compile else "0"
    os.environ["RWKV_FLOAT_MODE"] = args.precision

    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        multiplier = 4 if args.my_testing == "x070" else 3.5
        args.dim_ffn = int((args.n_embd * multiplier) // 32 * 32)

    from src.model import RWKV

    model_args = SimpleNamespace(
        vocab_size=args.vocab_size,
        ctx_len=args.ctx_len,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        dim_att=args.dim_att,
        dim_ffn=args.dim_ffn,
        head_size=args.head_size,
        rope_mode=args.rope_mode,
        rope_theta=args.rope_theta,
        rope_dims=args.rope_dims,
        learned_shift_state=args.learned_shift_state,
        attn_every=args.attn_every,
        attn_offset=args.attn_offset if args.attn_offset > 0 else args.attn_every,
        attn_heads=args.attn_heads,
        attn_dim=args.attn_dim,
        attn_dropout=0.0,
        attn_rope=args.attn_rope,
        norm_type=args.norm_type,
        tie_embeddings=args.tie_embeddings,
        my_testing=args.my_testing,
        grad_cp=0,
        strategy="auto",
        lr_init=0.0,
        betas=(0.9, 0.99),
        adam_eps=1e-18,
        weight_decay=0.0,
        accelerator="GPU" if device.type == "cuda" else "CPU",
    )
    model = RWKV(model_args)
    if args.load_model.endswith(".ptz"):
        from src.quant import load_quantized_state_dict

        state = load_quantized_state_dict(args.load_model)
    else:
        state = torch.load(args.load_model, map_location="cpu")
    for key in list(state.keys()):
        if key.startswith("_forward_module."):
            state[key.replace("_forward_module.", "")] = state.pop(key)
    model_state = model.state_dict()
    for key, value in list(state.items()):
        if key in model_state and value.shape != model_state[key].shape and value.numel() == model_state[key].numel():
            state[key] = value.reshape(model_state[key].shape)
    model.load_state_dict(state, strict=True)
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "tf32": torch.float32,
    }[args.precision]
    return model.to(device=device, dtype=dtype).eval()


def evaluate(args: argparse.Namespace) -> tuple[float, float, int, int]:
    data_path = Path(args.data_file)
    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}. "
            "For the default SP1892 path, build data_sp1892 with "
            "data/download_hf_docs_and_tokenize.py as described in README.md."
        )
    if not data_path.exists() and not glob.glob(str(args.data_file)):
        raise FileNotFoundError(
            f"Validation data not found: {data_path}. "
            "For the default SP1892 path, build the matched data_sp1892 export before scoring BPB."
        )

    luts = build_sentencepiece_luts(tokenizer_path, args.vocab_size)
    tokens = load_u16_tokens(resolve_val_files(args.data_file))
    total_targets = ((tokens.size - 1) // args.ctx_len) * args.ctx_len
    if args.max_tokens > 0:
        total_targets = min(total_targets, (args.max_tokens // args.ctx_len) * args.ctx_len)
    if total_targets <= 0:
        raise ValueError(f"Validation split is too short for ctx_len={args.ctx_len}")

    stride = args.stride if args.stride > 0 else args.ctx_len
    spans = iter_score_spans(total_targets, args.ctx_len, stride)
    if args.max_spans > 0:
        spans = spans[: args.max_spans]

    device = torch.device(args.device)
    model = load_rwkv_model(args, device)
    loss_sum = 0.0
    token_count = 0
    byte_count = 0

    with torch.inference_mode():
        for start in range(0, len(spans), args.micro_bsz):
            batch_spans = spans[start : start + args.micro_bsz]
            x_cpu, y_cpu = _make_batch(tokens, batch_spans)
            x = x_cpu.to(device=device, non_blocking=True)
            y = y_cpu.to(device=device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
            for row, span in enumerate(batch_spans):
                row_logits = logits[row, span.rel_start : span.rel_end].float()
                row_targets = y[row, span.rel_start : span.rel_end]
                loss_sum += float(F.cross_entropy(row_logits, row_targets, reduction="sum").item())
                token_count += int(row_targets.numel())
                byte_count += token_byte_sum(
                    tokens[span.score_start : span.score_end],
                    tokens[span.score_start + 1 : span.score_end + 1],
                    luts,
                )

    val_loss, val_bpb = bpb_from_sums(loss_sum, token_count, byte_count)
    return val_loss, val_bpb, token_count, byte_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RWKV FineWeb validation BPB with SP1892 byte accounting.")
    parser.add_argument("--load_model", required=True, help="Path to rwkv-*.pth or rwkv-final.pth")
    parser.add_argument("--data_file", default="../data_sp1892/datasets/fineweb10B_sp1892")
    parser.add_argument("--tokenizer_path", default="../data_sp1892/tokenizers/fineweb_1892_bpe.model")
    parser.add_argument("--vocab_size", type=int, default=1892)
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=0, help="Score stride; 0 means non-overlap ctx_len scoring")
    parser.add_argument("--micro_bsz", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32", "tf32", "fp16"])
    parser.add_argument("--compile", type=int, default=0)
    parser.add_argument("--my_testing", default="x070")
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--dim_att", type=int, default=0)
    parser.add_argument("--dim_ffn", type=int, default=0)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--rope_mode", default="none", choices=["none", "rk"])
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--rope_dims", type=int, default=0)
    parser.add_argument("--learned_shift_state", type=int, default=0)
    parser.add_argument("--attn_every", type=int, default=0)
    parser.add_argument("--attn_offset", type=int, default=0)
    parser.add_argument("--attn_heads", type=int, default=0)
    parser.add_argument("--attn_dim", type=int, default=0)
    parser.add_argument("--attn_rope", type=int, default=1)
    parser.add_argument(
        "--norm_type", default="layernorm", choices=["layernorm", "rmsnorm"]
    )
    parser.add_argument("--tie_embeddings", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=0, help="Optional ctx_len-rounded target-token cap for smoke checks")
    parser.add_argument("--max_spans", type=int, default=0, help="Optional score-span cap for smoke checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_loss, val_bpb, scored_tokens, scored_bytes = evaluate(args)
    stride = args.stride if args.stride > 0 else args.ctx_len
    print(
        f"val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} "
        f"scored_tokens:{scored_tokens} scored_bytes:{scored_bytes} "
        f"ctx_len:{args.ctx_len} stride:{stride} vocab_size:{args.vocab_size} "
        f"rope_mode:{args.rope_mode} learned_shift_state:{args.learned_shift_state} "
        f"norm_type:{args.norm_type} "
        f"attn_every:{args.attn_every} attn_offset:{args.attn_offset if args.attn_offset > 0 else args.attn_every}"
    )


if __name__ == "__main__":
    main()
