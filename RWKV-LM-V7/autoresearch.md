# Autoresearch: Minimize FineWeb SP1892 Val BPB under 5-min / 16-MB INT6 cap

## Objective
Train RWKV-7 on FineWeb SP1892 and minimize validation bits-per-byte (BPB).
Runs are capped to wall-clock ~300 s (5 min) locally; challenge target is 10 min.
The model must serialize to ≤16 MB as an INT6 quantized checkpoint (`.int6.ptz`).

## Metrics
- **Primary**: `val_bpb` (unitless, lower is better)
- **Secondary**:
  - `train_steps` — steps completed in the time window
  - `train_avg_loss` — final epoch-smoothed training loss
  - `int6_size_mb` — size of the quantized `.int6.ptz` checkpoint

## How to Run
```bash
./autoresearch.sh
```
The script trains for `MY_EXIT_SECONDS`, saves `rwkv-final.pth`, quantizes to
`rwkv-final.int6.ptz`, then evaluates BPB and prints structured metrics.

All runs share:
- Data: `../data_sp1892/datasets/fineweb10B_sp1892` (100 M token train, 53 M token val)
- Tokenizer: SP1892, vocab 1892
- Precision: bf16
- Compile: enabled
- Strategy: deepspeed_stage_2
- Time cap: 300 s
- LR: 4e-4 → 4e-5, warmup 50 steps
- Weight decay: 0.001

## Files in Scope
| File | Role |
|------|------|
| `train.py` | Training entrypoint, argument parser |
| `src/model.py` | RWKV model, hybrid attention, RoPE, norms, init |
| `src/trainer.py` | Callbacks, LR schedule, logging, quantization save |
| `src/dataset.py` | FineWeb uint16 shard loader |
| `src/quant.py` | INT6 quantization/dequantization |
| `eval_fineweb_bpb.py` | Validation BPB scorer with SP1892 byte accounting |
| `run-fineweb-10min.sh` | High-level run script wrapper |

## Off Limits
- Do not change tokenizer or data paths.
- Do not change BPB accounting logic in `eval_fineweb_bpb.py`.
- Do not change the `RWKV_Tmix_x070` core CUDA kernel interface.
- Do not increase model size beyond what quantizes to ≤16 MB INT6.

## Constraints
- 5-minute wall-clock runs for local testing.
- INT6 checkpoint must be ≤16 MB.
- All evals must match args used during training (e.g., `rope_mode`, `norm_type`, `tie_embeddings`).
- Use SP1892 FineWeb data only for comparability.

## What's Been Tried (pre-autoresearch)
| Config | val_bpb | Notes |
|--------|---------|-------|
| L8-D512 (baseline) | 1.51916871 | Best so far; 29.5M params, int6=12.85 MB |
| RMSNorm L8-D512 | 1.52069558 | Slightly worse than LayerNorm |
| RoPE full rk L8-D512 | 1.53729809 | Worse than baseline |
| RoPE partial d16 L8-D512 | 1.53697412 | Slightly better than full RoPE, still worse |
| Muon L8-D512 | 1.69464814 | Worse; also step-count inflation issue |
| Deep/narrow L16-D384 | 1.59235983 | Worse than L8-D512 baseline |
| Hybrid attention attn_every=4 L8-D512 | 1.50430779 | **Best so far**; one SDPA anchor at layer 4 |

## Architecture Ideas Todo
- Vary `attn_every` / `attn_offset` with the L8-D512 baseline.
- Try multiple attention layers (e.g., layers 2 and 6).
- Test `tie_embeddings=1` with baseline and hybrid configs.
- Try `rope_mode=rk` combined with hybrid attention.
- Test partial attention RoPE (`rope_dims`) on hybrid layers.
- Deeper/narrower shapes that still quantize under 16 MB (e.g., L10-D384).
