# Autoresearch: Parcae BPB Optimization (Local 1x RTX Pro 4500)

## Objective
Optimize the validation BPB (bits-per-byte) of `train_gpt_parcae.py` on a single RTX Pro 4500 Blackwell GPU with a ~5-minute wall-clock training budget. The real leaderboard runs on 8×H100 SXM; this session targets the best possible local score as a proxy for architecture and hyperparameter quality.

Any tokenizer is fair game so long as BPB is calculated correctly. We default to **SP1024** for clean iteration (no byte-count override needed), but we will also explore **Scylla** and **SP1892** when the hypothesis is tokenizer-specific. The constraint is correct BPB accounting, not tokenizer choice.

## Metrics
- **Primary**: `val_bpb` (unitless, **lower is better**) — exact roundtrip BPB from `final_int8_zlib_roundtrip_exact`
- **Secondary**:
  - `steps` — training steps completed within wall-clock cap
  - `step_avg_ms` — average step time (throughput indicator)
  - `submission_bytes` — total artifact size (code + compressed model)
  - `train_loss` — final training loss before eval
  - `peak_memory_mb` — peak GPU memory allocated

## How to Run
```bash
./autoresearch.sh
```
Outputs `METRIC val_bpb=...` lines to stdout. The script runs `train_gpt_parcae.py` with a fixed 300s wall-clock cap, then parses `logs/<run_id>.txt` for the exact roundtrip BPB.

## Files in Scope
- **`train_gpt_parcae.py`** — Main training script. May be modified for architecture, optimizer, quantization, or evaluation changes.
- **`utils/flash_attention.py`** — Flash attention backend wrapper. May be modified for attention optimizations.
- **`scripts/run_parcae_scylla_current_best.sh`** — Reference launcher. May be modified for new default env vars.
- **`autoresearch.sh`** — Benchmark runner. May be updated to add instrumentation.

## Off Limits
- **Data files** (`data/`, `data_scylla/`, `data_sp1892/`) — fixed challenge data
- **Record submissions** (`records/`) — read-only reference
- **`train_gpt.py`** and **`train_gpt_mlx.py`** — keep as canonical baselines for newcomers
- **`.venv/`** — do not modify Python environment or dependencies
- **Model artifacts** (`final_model.pt`, `final_model.int8.ptz`) — overwritten each run; do not commit

## Constraints
- Must run on 1× GPU (no DDP). `WORLD_SIZE=1` equivalent.
- Must complete within ~600s total (including warmup + eval).
- Must not break existing tokenizer/evaluation contracts.
- Must keep code reviewable and minimal per the repo's AGENTS.md guidelines.
- Do not add heavy new dependencies.

## Available Tokenizers & Data

| Tokenizer | Vocab | Data path | Byte override | Best known local BPB | Notes |
|-----------|------:|-----------|---------------|---------------------:|-------|
| **SP1024** | 1024 | `data/datasets/fineweb10B_sp1024/` | None | **1.762** | Canonical; simplest BPB accounting |
| **SP1892** | 1892 | `data_sp1892/datasets/fineweb10B_sp1892/` | None | — | Larger vocab; more embed params |
| **Scylla** | 998 | `data_scylla/fineweb_scylla/` | `151080363` | **1.691** | TokenMonster; requires override |

Switching tokenizers changes the embedding table size and the bytes-per-token ratio, so **never compare BPB across tokenizers directly**. Compare only within the same tokenizer family.

## Baseline Config (SP1024, 300s)
The following config produced **val_bpb ≈ 1.762** in prior local runs on SP1024:

```bash
MODEL_DIM=256
RECURRENT_DIM=256
NUM_HEADS=4
NUM_KV_HEADS=2
RECURRENT_NUM_HEADS=4
N_LAYERS_IN_PRELUDE=1
N_LAYERS_IN_RECURRENT_BLOCK=2
N_LAYERS_IN_CODA=1
MLP_MULT=3
TRAIN_SEQ_LEN=512
TRAIN_BATCH_TOKENS=16384
ITERATIONS=1000000
MAX_WALLCLOCK_SECONDS=300
WARMUP_STEPS=500
TRAIN_LOG_EVERY=100
VAL_LOSS_EVERY=0
MEAN_RECURRENCE=2
MEAN_BACKPROP_DEPTH=1
ROPE_DIMS=16
QK_NORM=1
USE_VALUE_EMBEDDINGS=0
BIGRAM_HASH_BUCKETS=4096
BIGRAM_HASH_DIM=128
BIGRAM_HASH_HEADS=2
BIGRAM_HASH_GATE=1
COMPILE_MODEL=0
COMPILE_MUON_BACKEND=0
POE_NUM_EXPERTS=1
SEED=1337
```

## What's Been Tried (High-Level from EXPERIMENTS.md)

### Confirmed wins on this local setup
- **Recurrence 2/1** (2 forward iterations, 1 backprop depth) beats 2/2 and no-loop on the current stack.
- **No value embeddings** improves both speed and quality.
- **KV heads = 2** helps vs KV=1.
- **QK norm** helps with no-value + KV2.
- **Partial RoPE (16 dims)** is the sweet spot (beats 8 and 32).
- **BigramHash 4096×128, 2 heads, gated** improves BPB vs no hash.
- **Seq512 + MLP3** is better than shorter seq / smaller MLP.

### Confirmed neutral/negative on this local setup
- **Parallel residuals** in recurrent core hurt.
- **XSA** (cross-slot attention) hurt.
- **Coda-only MoE** (TopK) hurt.
- **SwiGLU-add injection** at scale 0.1 hurt.
- **Hyperloop / mHC** (multi-stream hyperconnections) hurt or was too complex for the gain.
- **Core LaUREL** rank 8/16: slight BPB improvement on one run but not cleanly matched to baseline.
- **PLE (per-layer embeddings)** coda-only dim 32: not yet conclusively tested.
- **TTT** improves BPB but eval takes too long (>10 min) for competition viability.

### Untested / promising directions
- **Compile on Blackwell**: `COMPILE_MODEL=1` or `COMPILE_MUON_BACKEND=1` may improve step times significantly on RTX Pro 4500.
- **EMA / SWA**: The current working tree has EMA/SWA code added but not yet proven locally.
- **PoE (Product-of-Experts) output heads**: Huge win on Scylla (1.691 BPB) but not yet tested on SP1024.
- **BigramHash bucket sweep on SP1024**: 4096 is the current default; 8192 might help if params allow.
- **Recurrent dim scaling**: Current is 256→256 (same as outer). Different ratios untested.
- **AdamW vs Muon WD**: Muon WD currently 0.095; lower values or AdamW might help.
- **Warmup / LR schedule**: Currently 500-step warmup. Shorter/longer might affect final quality.
- **Gradient clipping**: Currently off. Some record submissions use clip=0.3.
