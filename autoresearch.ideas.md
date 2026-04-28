# Deferred Optimization Ideas

These are promising directions that haven't been conclusively tested or that we
decided not to pursue immediately. Revisit when current axes saturate.

## Highest Priority Untested Ideas

### 1. PoE (Product-of-Experts) on SP1024
- **Hypothesis**: `POE_NUM_EXPERTS=3` with tuned `POE_HEAD_LR` gave 1.694 on Scylla
- **Why deferred**: All autoresearch was on SP1024 batch-scaling; never tried PoE
- **Test plan**: `POE_NUM_EXPERTS=3 POE_HEAD_LR=0.005` on best SP1024 config (dim=384, batch=131k, compile=1)

### 2. Scylla with large-batch + compile
- **Hypothesis**: Current Scylla best is 1.694 at small batch. Batch=131072 + compile could be transformative
- **Why deferred**: Focused on SP1024 for clean iteration
- **Test plan**: Switch tokenizer to Scylla, keep best config (dim=384, batch=131k, compile=1), apply VAL_BYTE_COUNT_OVERRIDE

### 3. QAT (Quantization-Aware Training)
- **Hypothesis**: QAT_BITS=6 or 4 during training could reduce quantization error and enable lower-bit artifacts
- **Why deferred**: Complex to tune; need baseline before QAT
- **Test plan**: `QAT_BITS=6 QAT_START_STEP=1000` on current best config

### 4. NGRAM eval on SP1024
- **Hypothesis**: NGRAM_EVAL_ORDER=5 with linear mixing gave strong results on other tokenizers
- **Why deferred**: Eval-only path; need to test if absolute BPB improves neural-only roundtrip
- **Test plan**: `NGRAM_EVAL_ORDER=5 NGRAM_MIX_MODE=linear` on current best

### 5. Longer wall-clock budget
- **Hypothesis**: Diminishing returns from batch size suggest we're compute-limited. 600s could show continued improvement
- **Why deferred**: Stuck to 300s for fast iteration; 600s as next step
- **Test plan**: `MAX_WALLCLOCK_SECONDS=600` with current best config

## Medium Priority Ideas

### 6. Gradient checkpointing with larger model
- **Hypothesis**: Could fit larger model (dim=448 or more recurrent layers) if trading compute for memory
- **Why deferred**: GRADIENT_CHECKPOINTING=1 was never tested; step time increase unknown
- **Test plan**: `GRADIENT_CHECKPOINTING=1 MODEL_DIM=448` with batch=131072

### 7. Curriculum learning / sampling schemes
- **Hypothesis**: `SAMPLING_SCHEME=fixed` is baseline; adaptive schemes might help
- **Why deferred**: Complex to tune; no clear evidence from records
- **Test plan**: Try `SAMPLING_SCHEME=adaptive` with matched config

### 8. Different optimizer settings
- **Hypothesis**: `MUON_WD=0.095` is default; other values or AdamW fallback could help
- **Why deferred**: Muon matrix optimizer works well; small LR tweaks tried and marginal
- **Test plan**: `MUON_WD=0.05` or `MUON_MOMENTUM=0.90` on best config

### 9. Multi-token prediction (MTP) auxiliary loss
- **Hypothesis**: MTP auxiliary loss could improve representation quality
- **Why deferred**: Not wired as a BPB path; eval only scores next-token CE
- **Test plan**: Would need code change to enable MTP head

### 10. Different tokenizers
- **SP1892**: Larger vocab but more embed params. Never tested with optimized batch/compile.
- **Custom SP2048 or larger**: Could explore if embed params fit budget.

## Low Priority / Long Shot

### 11. DeepSeek MoE coda
- Code-complete but zero BPB result. Could try `DEEPSEEK_MOE_NUM_BASE_EXPERTS=8` with `ACTIVE_EXPERTS=4`.

### 12. LAuReL-LR in different scope
- Tested coda and core with negative results. Could try `LAUREL_SCOPE=all LAUREL_RANK=16`.

### 13. Attention mechanism in Parcae (non-recurrent)
- Not available in Parcae architecture. Would need core rewrite.

### 14. EMA (Exponential Moving Average)
- `EMA_ENABLED` env var referenced in autoresearch.sh but no EMA code in train_gpt_parcae.py. Code would need to be added.

## Already Tested & Ruled Out for Now

| Idea | Result | Notes |
|------|--------|-------|
| Parallel residuals in core | Hurt BPB | Recorded in EXPERIMENTS.md |
| XSA (cross-slot attention) | Hurt BPB | Recorded in EXPERIMENTS.md |
| Coda-only MoE (TopK) | Hurt BPB | Recorded in EXPERIMENTS.md |
| SwiGLU-add injection at 0.1 | Hurt BPB | Scale was too high |
| Hyperloop / mHC | Hurt or too complex | Recorded in EXPERIMENTS.md |
| TTT as default | Too slow (>10 min eval) | Keep as eval-only opt-in |
| MLP_MULT=5 | Marginal gain, larger artifact | Not worth it |
| MODEL_DIM=448 | Marginal vs 384 | Diminishing returns |
| MEAN_RECURRENCE=3 | Marginal gain, fewer steps | Stick to 2 |
| COMPILE_MUON_BACKEND=1 | Worse than compile alone | Revert |
| RMSNorm in RWKV | Slightly worse | Not applicable to Parcae but noted |
