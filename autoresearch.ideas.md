# Deferred Optimization Ideas (pruned 2026-04-28)

Context: strict 300s wall-clock, SP1024, dim=384 batch=131k compile=1 SWA=1. Best = **1.42353 (GPTQ)**.

## High priority — untested

### A. QAT@6 + QUANT_BITS=6 + larger model
- Current artifact is 12.6MB at int8. Dropping to int6 packs to ~9MB, freeing ~3MB.
- Plan: `QAT_BITS=6 QUANT_BITS=6 QAT_START_STEP=500 GPTQ_ENABLED=1`, then scale `MODEL_DIM` to fill the freed space.
- Last attempt (QAT_BITS=6 alone, run 33) exported at int8 anyway because autoresearch.sh hardcoded `QUANT_BITS=8`. Fixed now via `${QUANT_BITS:-${QAT_BITS}}` fallback.

### B. Gradient checkpointing with larger model
- `GRADIENT_CHECKPOINTING=1 ACTIVATION_CHECKPOINT_IMPL=per-iteration` may allow dim=448 or more recurrent layers. Memory was 1.6GB headroom, but the real win is fitting larger active model during backward.
- Trade-off: ~30% step-time increase; may eat into the 300s budget.

### C. XSA on last N layers
- Multiple top leaderboard entries (PR #1019, #549, #287, #198) use XSA (cross-slot attention). `XSA_LAST_N=3` or `=4` on the deepest layers is the typical pattern.
- Low risk, code-complete.

### D. Different optimizer settings
- `MUON_WD=0.04` (vs default 0.095) is used by multiple leaderboard entries (jfprincz, thwu1, aruniyer).
- `MUON_MOMENTUM=0.90` worth a quick test.

### E. NGRAM eval mixer
- `NGRAM_EVAL_ORDER=5 NGRAM_MIX_MODE=expert` — eval-only, doesn't change trained model. Separate metric to watch: `final_..._ngram..._exact`.

## Medium priority

### G. Scylla tokenizer with large batch
- Never tested with optimized config. Requires `VAL_BYTE_COUNT_OVERRIDE=151080363`. Separate eval path because BPB accounting differs.

### H. SwiGLU reinjection at smaller scale
- Tested at scale=0.1 (hurt); never tested at scale=0.02 or 0.05.

### I. Partial RoPE (outer vs recurrent split)
- `ROPE_DIMS=32` fixed; never ablated `OUTER_ROPE_DIMS` vs `RECURRENT_ROPE_DIMS`.
- Leaderboard jfprincz used partial RoPE 16/64.

### J. MoE coda (DeepSeek-style)
- `DEEPSEEK_MOE_NUM_BASE_EXPERTS=4 DEEPSEEK_MOE_ACTIVE_EXPERTS=2`. Code complete.

## Already tried / ruled out

| Idea | Result | Run |
|------|--------|-----|
| COMPILE_MUON_BACKEND=1 | Worse than compile alone | 3 |
| MEAN_RECURRENCE=3 | Marginal, 11% fewer steps | 10 |
| MLP_MULT=5 | Negligible, +1.5MB artifact | 15 |
| Batches 114688, 122880 | Worse than 131072 | 20-21 |
| SWA_START_FRAC=0.1 | Worse than 0.2 | 23 |
| WARMUP_STEPS=300 | Worse than 500 | 24 |
| PoE (all variants) | No effect on SP1024 | 26, 28 |
| EMA decay=0.997 | Worse than SWA | 30 |
| EMA decay=0.999 | Underfits at 1400 steps | 31 |
| QAT@6 alone (int8 export) | No-op | 33 |
| Longer wall-clock 600s | Challenge cap is 10min on 8xH100, locally ~300s per spec |
| TTT | >10 min eval — exceeds eval budget |

## Code-change ideas (requires patches)

- **MTP auxiliary loss** — not wired; needs architecture change.
- **Tensor.item() graph break fix** — saw warnings; converting `step` from Tensor→int would eliminate graph breaks and likely speed compile. A previous uncommitted attempt existed but was reverted. Worth isolating as a pure perf win.
- **Learning adapters on random linear maps** — from README "requests for PRs".
