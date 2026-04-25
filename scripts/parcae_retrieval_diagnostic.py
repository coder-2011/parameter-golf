from __future__ import annotations

import json
import math
import os
import sys
import types
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_gpt_parcae as tgp


def main() -> None:
    args = tgp.Hyperparameters()
    model_path = Path(os.environ.get("MODEL_PATH", "final_model.pt"))
    nseq = int(os.environ.get("DIAG_NSEQ", "64"))
    out_path = os.environ.get("DIAG_OUT", "")
    device = torch.device(os.environ.get("DEVICE", "cuda:0"))
    torch.cuda.set_device(device)

    model = tgp.GPT(args).to(device).bfloat16()
    tgp.restore_low_dim_params_to_fp32(model)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.eval()

    val_tokens = tgp.load_validation_tokens(args.val_files, args.train_seq_len)
    nseq = min(nseq, (val_tokens.numel() - 1) // args.train_seq_len)
    local = val_tokens[: nseq * args.train_seq_len + 1].to(device=device, dtype=torch.long)
    x = local[:-1].reshape(nseq, args.train_seq_len)
    y = local[1:].reshape(nseq, args.train_seq_len)

    attention_stats: dict[str, dict[str, float | int]] = {}

    def instrument(name: str, mod: tgp.ParcaeCausalSelfAttention) -> None:
        def forward(self, x_in, freqs_cis, mask=None, **kwargs):
            bsz, seqlen, dim = x_in.shape
            q = self.c_q(x_in).view(bsz, seqlen, self.n_head, self.head_dim)
            k = self.c_k(x_in).view(bsz, seqlen, self.n_kv_head, self.head_dim)
            v = self.c_v(x_in).view(bsz, seqlen, self.n_kv_head, self.head_dim)
            ve = kwargs.get("ve")
            if ve is not None and self.ve_gate is not None:
                ve = ve.view(bsz, seqlen, self.n_kv_head, self.head_dim)
                gate = 2 * torch.sigmoid(self.ve_gate(x_in[..., : self.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve
            if self.clip_qkv is not None:
                q = q.clamp(min=-self.clip_qkv, max=self.clip_qkv)
                k = k.clamp(min=-self.clip_qkv, max=self.clip_qkv)
                v = v.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            if self.qk_bias is not None:
                q_bias, k_bias = self.qk_bias.split(1, dim=0)
                q = (q + q_bias).to(q.dtype)
                k = (k + k_bias).to(k.dtype)
            q, k = tgp.apply_rotary_emb_complex_like(q, k, freqs_cis, self.rope_dims)
            if self.qk_norm:
                q = F.rms_norm(q, (q.size(-1),))
                k = F.rms_norm(k, (k.size(-1),))

            qh = q.transpose(1, 2).float()
            kh = k.transpose(1, 2).float()
            vh = v.transpose(1, 2)
            if self.n_kv_head != self.n_head:
                repeats = self.n_head // self.n_kv_head
                kh = kh.repeat_interleave(repeats, dim=1)
                vh = vh.repeat_interleave(repeats, dim=1)
            scores = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(self.head_dim)
            causal = torch.ones(seqlen, seqlen, device=device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
            attn = torch.softmax(scores, dim=-1)

            with torch.no_grad():
                idx = torch.arange(seqlen, device=device)
                dist = (idx[:, None] - idx[None, :]).clamp_min(0)
                target_match = x[:, None, :, None].eq(y[:, None, None, :]).permute(0, 1, 3, 2)
                target_match = target_match & causal[None, None]
                target_match = target_match & ~torch.eye(seqlen, device=device, dtype=torch.bool)[None, None]
                recent_target_match = target_match & (dist[None, None] <= 32)
                any_target = target_match.any(-1)
                any_recent_target = recent_target_match.any(-1)
                p = attn.detach()
                entropy = -(p.clamp_min(1e-30) * p.clamp_min(1e-30).log()).sum(-1)
                entropy_denom = torch.log(torch.arange(1, seqlen + 1, device=device).float()).clamp_min(1)
                stat = {
                    "calls": 1,
                    "entropy_norm": float((entropy / entropy_denom[None, None, :]).mean().item()),
                    "max_prob": float(p.max(-1).values.mean().item()),
                    "expected_distance": float((p * dist.float()).sum(-1).mean().item()),
                    "mass_self": float((p * (dist == 0)[None, None]).sum(-1).mean().item()),
                    "mass_prev1": float((p * (dist == 1)[None, None]).sum(-1).mean().item()),
                    "mass_2_8": float((p * ((dist >= 2) & (dist <= 8))[None, None]).sum(-1).mean().item()),
                    "mass_9_32": float((p * ((dist >= 9) & (dist <= 32))[None, None]).sum(-1).mean().item()),
                    "mass_33_128": float((p * ((dist >= 33) & (dist <= 128))[None, None]).sum(-1).mean().item()),
                    "target_seen_prev_frac": float(any_target.float().mean().item()),
                    "mass_to_prev_target": float((p * target_match.float()).sum(-1)[any_target.expand_as(p[..., 0])].mean().item())
                    if any_target.any()
                    else 0.0,
                    "mass_to_recent32_target": float(
                        (p * recent_target_match.float()).sum(-1)[any_recent_target.expand_as(p[..., 0])].mean().item()
                    )
                    if any_recent_target.any()
                    else 0.0,
                }
                old = attention_stats.get(name)
                if old is None:
                    attention_stats[name] = stat
                else:
                    calls = int(old["calls"])
                    for key, value in stat.items():
                        old[key] = calls + 1 if key == "calls" else (float(old[key]) * calls + float(value)) / (calls + 1)

            y_attn = torch.matmul(attn.to(vh.dtype), vh)
            return self.c_proj(y_attn.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))

        mod.forward = types.MethodType(forward, mod)

    for name, mod in model.named_modules():
        if isinstance(mod, tgp.ParcaeCausalSelfAttention):
            instrument(name, mod)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model.forward_model(x, y, return_logits=True)["logits"].float()
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(
        nseq, args.train_seq_len
    )
    pred = logits.argmax(-1)

    seen = torch.zeros_like(y, dtype=torch.bool)
    seen32 = torch.zeros_like(y, dtype=torch.bool)
    for pos in range(args.train_seq_len):
        seen[:, pos] = (x[:, : pos + 1] == y[:, pos : pos + 1]).any(1)
        seen32[:, pos] = (x[:, max(0, pos - 31) : pos + 1] == y[:, pos : pos + 1]).any(1)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    _, leading, _ = tgp.build_sentencepiece_luts(sp, args.vocab_size, device)
    leading_mask = leading[y.reshape(-1)].view_as(y)

    def masked_mean(t: torch.Tensor, mask: torch.Tensor) -> float:
        return float(t[mask].mean().item()) if mask.any() else float("nan")

    result = {
        "model_path": str(model_path),
        "nseq": nseq,
        "loss": float(loss.mean().item()),
        "accuracy": float((pred == y).float().mean().item()),
        "pos_loss": {
            "0_16": float(loss[:, :16].mean().item()),
            "16_64": float(loss[:, 16:64].mean().item()),
            "64_128": float(loss[:, 64:128].mean().item()),
            "128_256": float(loss[:, 128:256].mean().item()),
        },
        "leading_space_loss": masked_mean(loss, leading_mask),
        "nonleading_loss": masked_mean(loss, ~leading_mask),
        "seen_prev": {
            "fraction": float(seen.float().mean().item()),
            "loss": masked_mean(loss, seen),
            "accuracy": float((pred[seen] == y[seen]).float().mean().item()) if seen.any() else float("nan"),
        },
        "seen_last32": {
            "fraction": float(seen32.float().mean().item()),
            "loss": masked_mean(loss, seen32),
            "accuracy": float((pred[seen32] == y[seen32]).float().mean().item()) if seen32.any() else float("nan"),
        },
        "not_seen": {
            "fraction": float((~seen).float().mean().item()),
            "loss": masked_mean(loss, ~seen),
            "accuracy": float((pred[~seen] == y[~seen]).float().mean().item()) if (~seen).any() else float("nan"),
        },
        "attention": attention_stats,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
