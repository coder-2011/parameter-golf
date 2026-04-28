# Autoresearch: Parcae BPB Optimization (Local 1x RTX Pro 4500)

## Objective
Optimize the validation BPB (bits-per-byte) of `train_gpt_parcae.py` on a single
RTX Pro 4500 Blackwell GPU with a ~300s wall-clock training budget. The real
leaderboard runs on 8×H100 SXM; this session targets the best possible local
score as a proxy for architecture and hyperparameter quality.

Any tokenizer is fair game so long as BPB is calculated correctly. We default to
**SP1024** for clean iteration (no byte-count override needed), but we also
explore **Scylla** and **SP1892** when the hypothesis is tokenizer-specific.
The constraint is correct BPB accounting, not tokenizer choice.

## Metrics
- **Primary**: `val_bpb` (unitless, **lower is better**) — exact roundtrip BPB
  from `final_int8_zlib_roundtrip_exact`
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
Outputs `METRIC val_bpb=...` lines to stdout. The script runs
`train_gpt_parcae.py` with a fixed 300s wall-clock cap, then parses
`logs/<run_id>.txt` for the exact roundtrip BPB.

## Files in Scope
- **`train_gpt_parcae.py`** — Main training script. May be modified for
  architecture, optimizer, quantization, or evaluation changes.
- **`utils/flash_attention.py`** — Flash attention backend wrapper.
- **`scripts/run_parcae_scylla_current_best.sh`** — Reference launcher.
- **`autoresearch.sh`** — Benchmark runner. May be updated to add instrumentation.

## Off Limits
- **Data files** (`data/`, `data_scylla/`, `data_sp1892/`) — fixed challenge data
- **Record submissions** (`records/`) — read-only reference
- **`train_gpt.py`** and **`train_gpt_mlx.py`** — keep as canonical baselines
- **`.venv/`** — do not modify Python environment or dependencies
- **Model artifacts** (`final_model.pt`, `final_model.int8.ptz`) — overwritten each run

## Constraints
- Must run on 1× GPU (no DDP). `WORLD_SIZE=1` equivalent.
- Must complete within ~600s total (including warmup + eval).
- Must not break existing tokenizer/evaluation contracts.
- Must keep code reviewable and minimal per AGENTS.md guidelines.

## Available Tokenizers & Data

| Tokenizer | Vocab | Data path | Byte override | Best known local BPB | Notes |
|-----------|------:|-----------|---------------|---------------------:|-------|
| **SP1024** | 1024 | `data/datasets/fineweb10B_sp1024/` | None | **1.414** (batch 131k) | Canonical; simplest BPB accounting |
| **SP1892** | 1892 | `data_sp1892/datasets/fineweb10B_sp1892/` | None | — | Larger vocab; more embed params |
| **Scylla** | 998 | `data_scylla/fineweb_scylla/` | `151080363` | **1.694** | TokenMonster; requires override |

## Current Best Config (SP1024, 300s)
```bash
MODEL_DIM=384
RECURRENT_DIM=384
NUM_HEADS=4
NUM_KV_HEADS=2
RECURRENT_NUM_HEADS=4
N_LAYERS_IN_PRELUDE=1
N_LAYERS_IN_RECURRENT_BLOCK=3
N_LAYERS_IN_CODA=2
MLP_MULT=4
TRAIN_SEQ_LEN=1024
TRAIN_BATCH_TOKENS=131072
ITERATIONS=1000000
MAX_WALLCLOCK_SECONDS=300
WARMUP_STEPS=500
TRAIN_LOG_EVERY=100
VAL_LOSS_EVERY=0
MEAN_RECURRENCE=2
MEAN_BACKPROP_DEPTH=1
ROPE_DIMS=32
QK_NORM=1
USE_VALUE_EMBEDDINGS=0
BIGRAM_HASH_BUCKETS=8192
BIGRAM_HASH_DIM=128
BIGRAM_HASH_HEADS=2
BIGRAM_HASH_GATE=1
COMPILE_MODEL=1
COMPILE_MUON_BACKEND=0
POE_NUM_EXPERTS=1
GRAD_CLIP_NORM=0.3
SWA_ENABLED=1
SWA_START_FRAC=0.2
SWA_EVERY=50
SEED=1337
```

## New & Notable Features in train_gpt_parcae.py

### Validation-only eval accelerators
These features run **after** the standard neural roundtrip and do not change
the saved artifact. They are controlled by env vars and can mix with each other.

| Feature | Env vars | Description |
|---------|----------|-------------|
| **Sliding window eval** | `SLIDING_WINDOW_ENABLED=1`, `SLIDING_COMPILE_LOGITS=0/1` | Quantized roundtrip sliding-window BPB scorer. Evaluates with strided context windows. |
| **PPM mixer** | `PPM_ENABLED=1`, `PPM_ORDER=5`, `PPM_SUBSET_TOKENS=5000000`, `PPM_USE_META_MIX=0/1` | Byte-level online PPM mixing on scored sliding prefix. Reconstructs tokenizer bytes separately and logs `final_int*_zlib_roundtrip_sliding_ppm_exact`. |
| **Hashed n-gram mixer** | `NGRAM_EVAL_ORDER>=2`, `NGRAM_EVAL_ALPHA=0.30`, `NGRAM_CHUNK_TOKENS=1048576`, `NGRAM_MIX_MODE=linear` | Score-first chunks with deterministic count-table updates, entropy-adaptive alpha, optional `NGRAM_ORDER_MULTS` and `NGRAM_CUBRIC_CADENCE`. `linear` mode helped; `expert` mode (residual/product) hurt in ablation. |
| **TTT (score-first SGD)** | `TTT_ENABLED=1`, `TTT_LR=0.005`, `TTT_EPOCHS=3` | Scores validation chunks first, then trains on each chunk via SGD before scoring the next. Improves BPB but eval takes 10+ min — too slow for competition unless optimized. |

### Architecture variants

| Feature | Env vars | Description |
|---------|----------|-------------|
| **DeepSeek MoE coda** | `DEEPSEEK_MOE_NUM_BASE_EXPERTS>0`, `DEEPSEEK_MOE_ACTIVE_EXPERTS`, `DEEPSEEK_MOE_SHARED_EXPERTS` | Replaces coda MLPs with DeepSeek-style routed + shared experts. Default-off. |
| **LAuReL low-rank residual** | `LAUREL_SCOPE=prelude\|core\|coda\|all`, `LAUREL_RANK>0`, `LAUREL_SCALE_INIT=0.01` | Adds `scale * RMSNorm(right(left(block_input)))` to block output. Default-off. |
| **Parallel residual** | `RESIDUAL_MODE=parallel`, `PARALLEL_RESIDUAL_SCOPE=core` | Recurrent core blocks compute `x + attn(norm1(x)) + mlp(norm2(x))`. Default `sequential`. |
| **SwiGLU reinjection** | `INJECTION_TYPE=swiglu-add`, `INJECTION_SWIGLU_SCALE=0.1` | Nonlinear token/BigramHash reinjection into recurrent state. At scale `0.1` it hurt; zero scale preserves old behavior. |
| **RoPE variants** | `ROPE_DIMS=32`, `OUTER_ROPE_DIMS`, `RECURRENT_ROPE_DIMS` | Partial RoPE in different scopes. `ROPE_DIMS=32` is best tested for seq1024. |
| **QK norm** | `QK_NORM=1` | RMSNorm on Q/K before attention. Helps with no-value-embed + KV2. |
| **Quantization-aware training** | `QAT_BITS>0`, `QAT_START_STEP=500`, `QAT_LINEAR=1`, `QAT_TIED_OUTPUT=1` | Ste fake-quant during training for lower-bit artifacts. |
| **GPTQ** | `GPTQ_ENABLED=1`, `GPTQ_CALIBRATION_BATCHES=32`, `GPTQ_BLOCKSIZE=128` | Post-training GPTQ quantization. |

## What's Been Tried (from autoresearch.jsonl)

### Confirmed wins
- **COMPILE_MODEL=1** on Blackwell: +57% steps, -0.018 BPB
- **TRAIN_SEQ_LEN=1024** (was 512): -0.028 BPB, comparable steps
- **MODEL_DIM scaling**: 256→320→384 each gave ~-0.02 BPB. 384→448 was marginal.
- **MLP_MULT=4** (was 3): -0.013 BPB
- **N_LAYERS_IN_RECURRENT_BLOCK=3** (was 2): -0.007 BPB
- **N_LAYERS_IN_CODA=2** (was 1): -0.006 BPB  
- **BIGRAM_HASH_BUCKETS=8192** (was 4096): -0.002 BPB
- **TRAIN_BATCH_TOKENS=131072** (was 16384): MASSIVE improvement, -0.17 BPB total from batch scaling. Best at 131072.
- **SWA_ENABLED=1, SWA_START_FRAC=0.2**: Marginal -0.001 BPB
- **ROPE_DIMS=32** (was 16): Marginal -0.002 BPB for seq1024
- **GRAD_CLIP_NORM=0.3**: Marginal -0.001 BPB

### Confirmed neutral/negative
- **COMPILE_MUON_BACKEND=1**: Worse than compile_model alone
- **MEAN_RECURRENCE=3** (was 2): Marginal BPB gain, 11% fewer steps
- **MLP_MULT=5** (was 4): Marginal -0.001 BPB but much larger artifact
- **WARMUP_STEPS=300** (was 500): Slightly worse
- **SWA_START_FRAC=0.1** (was 0.2): Worse
- **Batch sizes 114688, 122880**: Worse than 131072
- **TTT**: Improves BPB but eval >10 min
- **PPM / NGRAM / Sliding**: Eval-only paths; need testing for absolute BPB impact

### Untested / promising directions
- **PoE on SP1024**: PoE=3 was huge on Scylla (1.694 BPB) but never tested on SP1024
- **Scylla with large batch + compile**: Current best Scylla is 1.694, but batch 131k was never tried
- **QAT / GPTQ**: QAT and GPTQ paths are code-complete but not yet proven on SP1024
- **NGRAM_EVAL_ORDER>=2 on SP1024**: Linear n-gram mixing helped on other tokenizers
- **DeepSeek MoE coda**: Code-complete, zero BPB result yet
- **LAuReL-LR**: Not yet promising in tested placements, but code-complete
- **Longer training**: Could try MAX_WALLCLOCK_SECONDS=600 with same config
