import argparse
import glob
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_data_shard(path: Path) -> torch.Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if path.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {path}: expected {expected_size} bytes")
    data = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if data.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return torch.from_numpy(data.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def take(self, n: int) -> torch.Tensor:
        chunks = []
        remaining = n
        while remaining:
            available = self.tokens.numel() - self.pos
            if available <= 0:
                self.file_idx = (self.file_idx + 1) % len(self.files)
                self.tokens = load_data_shard(self.files[self.file_idx])
                self.pos = 0
                continue
            k = min(available, remaining)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def next_batch(stream: TokenStream, batch_size: int, seq_len: int, device: torch.device):
    raw = stream.take(batch_size * seq_len + 1).to(device=device, dtype=torch.long)
    return raw[:-1].reshape(batch_size, seq_len), raw[1:].reshape(batch_size, seq_len)


def rosa_qkv_ref(qqq, kkk, vvv):
    n = len(qqq)
    y = [-1] * n
    size = 2 * n + 1
    trans = [None] * size
    link = [-1] * size
    length = [0] * size
    right = [-1] * size
    trans[0] = {}
    last = 0
    matched_len = 0
    used = 1
    root = 0
    assert n == len(kkk) == len(vvv)
    for i, (q, k) in enumerate(zip(qqq, kkk)):
        p, x = last, matched_len
        while p != -1 and q not in trans[p]:
            x = max(x, length[p])
            p = link[p]
        p, x = (trans[p][q], x + 1) if p != -1 else (0, 0)
        v = p
        while link[v] != -1 and length[link[v]] >= x:
            v = link[v]
        while v != -1 and (length[v] <= 0 or right[v] < 0):
            v = link[v]
        y[i] = vvv[right[v] + 1] if v != -1 else -1
        last, matched_len = p, x

        cur = used
        used += 1
        trans[cur] = {}
        length[cur] = length[root] + 1
        p = root
        while p != -1 and k not in trans[p]:
            trans[p][k] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q_state = trans[p][k]
            if length[p] + 1 == length[q_state]:
                link[cur] = q_state
            else:
                clone = used
                used += 1
                trans[clone] = trans[q_state].copy()
                length[clone] = length[p] + 1
                link[clone] = link[q_state]
                right[clone] = right[q_state]
                link[q_state] = link[cur] = clone
                while p != -1 and trans[p][k] == q_state:
                    trans[p][k] = clone
                    p = link[p]
        v = root = cur
        while v != -1 and right[v] < i:
            right[v] = i
            v = link[v]
    return [max(0, yy) for yy in y]


def rosa_qkv_batch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert q.dtype == k.dtype == v.dtype == torch.uint8
    qc = q.detach().contiguous().cpu()
    kc = k.detach().contiguous().cpu()
    vc = v.detach().contiguous().cpu()
    out = [
        torch.as_tensor(rosa_qkv_ref(qq.tolist(), kk.tolist(), vv.tolist()), dtype=q.dtype)
        for qq, kk, vv in zip(qc, kc, vc)
    ]
    return torch.stack(out).to(q.device)


class RosaQkv1Bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, emb):
        b, t, c = q.shape
        qb = (q > 0).to(torch.uint8).transpose(1, 2).reshape(-1, t).contiguous()
        kb = (k > 0).to(torch.uint8).transpose(1, 2).reshape(-1, t).contiguous()
        vb = (v > 0).to(torch.uint8).transpose(1, 2).reshape(-1, t).contiguous()
        idx = rosa_qkv_batch_ref(qb, kb, vb).view(b, c, t).transpose(1, 2).contiguous()
        return (2.0 * idx.to(q.dtype) - 1.0) * emb


class RosaLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_q = nn.Parameter(torch.zeros(1, 1, dim))
        self.x_k = nn.Parameter(torch.zeros(1, 1, dim))
        self.x_v = nn.Parameter(torch.zeros(1, 1, dim))
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.emb = nn.Parameter(torch.ones(1, 1, dim))
        self.o = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x) - x
        q = x + xx * self.x_q
        k = x + xx * self.x_k
        v = x + xx * self.x_v
        y = RosaQkv1Bit.apply(self.q(q), self.k(k), self.v(v), self.emb)
        return self.o(y)


class ChannelMix(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(1, 1, dim))
        self.key = nn.Linear(dim, dim * 4, bias=False)
        self.value = nn.Linear(dim * 4, dim, bias=False)
        nn.init.orthogonal_(self.key.weight, gain=2.0)
        nn.init.zeros_(self.value.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        return self.value(torch.relu(self.key(k)) ** 2)


class Block(nn.Module):
    def __init__(self, dim: int, use_rosa: bool):
        super().__init__()
        self.use_rosa = use_rosa
        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.rosa = RosaLayer(dim)
        self.ffn = ChannelMix(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln0(x)
        if self.use_rosa:
            x = x + self.rosa(self.ln1(x)).detach()
        return x + self.ffn(self.ln2(x))


class Rwkv8RosaLm(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int, n_embd: int, use_rosa: bool):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, use_rosa) for _ in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_out(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss


def sentencepiece_byte_lut(tokenizer_path: str, vocab_size: int, device: torch.device) -> torch.Tensor:
    try:
        import sentencepiece as spm
    except ModuleNotFoundError:
        print("warning: sentencepiece is not installed; val_bpb uses 1 byte per token as a rough proxy")
        return torch.ones((vocab_size,), dtype=torch.float64, device=device)

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    table = np.ones((vocab_size,), dtype=np.int64)
    for token_id in range(min(vocab_size, int(sp.vocab_size()))):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            table[token_id] = 0
        elif sp.is_byte(token_id):
            table[token_id] = 1
        else:
            piece = sp.id_to_piece(token_id)
            if piece.startswith("▁"):
                piece = piece[1:]
                table[token_id] = len(piece.encode("utf-8")) + 1
            else:
                table[token_id] = len(piece.encode("utf-8"))
    return torch.tensor(table, dtype=torch.float64, device=device)


@torch.inference_mode()
def evaluate(model, val_tokens, args, byte_lut, device):
    model.eval()
    loss_sum = 0.0
    token_count = 0
    byte_count = 0.0
    max_tokens = min(args.val_tokens, val_tokens.numel() - 1)
    usable = (max_tokens // args.seq_len) * args.seq_len
    for start in range(0, usable, args.eval_batch_size * args.seq_len):
        end = min(start + args.eval_batch_size * args.seq_len, usable)
        local = val_tokens[start : end + 1].to(device=device, dtype=torch.long)
        x = local[:-1].reshape(-1, args.seq_len)
        y = local[1:].reshape(-1, args.seq_len)
        loss = model(x, y)
        loss_sum += float(loss.item()) * y.numel()
        token_count += y.numel()
        byte_count += float(byte_lut[y].sum().item())
    val_loss = loss_sum / max(1, token_count)
    val_bpb = val_loss / math.log(2) * token_count / max(1.0, byte_count)
    model.train()
    return val_loss, val_bpb


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny RWKV-8 ROSA-style LM on local FineWeb shards.")
    parser.add_argument("--data-path", default="data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-every", type=int, default=25)
    parser.add_argument("--val-tokens", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-rosa", action="store_true")
    parser.add_argument("--save", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    root = Path(__file__).resolve().parent
    data_path = (root / args.data_path).resolve()
    tokenizer_path = (root / args.tokenizer_path).resolve()
    train_pattern = str(data_path / "fineweb_train_*.bin")
    val_pattern = str(data_path / "fineweb_val_*.bin")

    stream = TokenStream(train_pattern)
    val_files = [Path(p) for p in sorted(glob.glob(val_pattern))]
    if not val_files:
        raise FileNotFoundError(f"No validation shards found for pattern: {val_pattern}")
    val_tokens = torch.cat([load_data_shard(p) for p in val_files])
    byte_lut = sentencepiece_byte_lut(str(tokenizer_path), args.vocab_size, device)

    model = Rwkv8RosaLm(args.vocab_size, args.n_layer, args.n_embd, args.use_rosa).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"device:{device} params:{n_params} train_files:{len(stream.files)} val_tokens:{val_tokens.numel()}")
    print(f"shape:L{args.n_layer}-D{args.n_embd} seq_len:{args.seq_len} batch_size:{args.batch_size} use_rosa:{args.use_rosa}")

    start_time = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = next_batch(stream, args.batch_size, args.seq_len, device)
        loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.val_every == 0 or step == args.steps:
            val_loss, val_bpb = evaluate(model, val_tokens, args, byte_lut, device)
            elapsed = time.time() - start_time
            print(
                f"step:{step}/{args.steps} train_loss:{loss.item():.4f} "
                f"val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} elapsed:{elapsed:.1f}s"
            )

    if args.save:
        torch.save({"model": model.state_dict(), "args": vars(args)}, args.save)
        print(f"saved:{args.save}")


if __name__ == "__main__":
    main()
