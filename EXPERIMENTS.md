# Experiment Notes

## 2026-04-26 Parcae implementation log

This section records code changes made around the current Parcae/Scylla training script. These are implementation notes, not proof of quality improvements unless a run result is listed separately.

### Score-first TTT validation path

Implemented an opt-in score-first test-time training path in `train_gpt_parcae.py`.

Added knobs:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `TTT_ENABLED` | `0` | Enables the extra post-roundtrip TTT validation pass |
| `EVAL_STRIDE` | `64` | Sliding-window stride for TTT scoring |
| `TTT_CHUNK_TOKENS` | `32768` | Validation chunk size |
| `TTT_EPOCHS` | `3` | SGD epochs after each chunk is scored |
| `TTT_LR` | `0.005` | SGD learning rate before chunk-wise cosine decay |
| `TTT_MOMENTUM` | `0.9` | SGD momentum |
| `TTT_BATCH_SEQS` | `32` | Number of full sequences per TTT batch |
| `TTT_GRAD_CLIP` | `1.0` | Gradient clip norm |

Implementation behavior:

- TTT runs after quantized model roundtrip load and does not change the saved artifact.
- Validation tokens are assigned to `TTT_CHUNK_TOKENS` chunks by the first target token scored by each sliding window.
- Each chunk is scored first under `torch.no_grad()`, then the model is trained on full `TRAIN_SEQ_LEN` sequences from that same chunk.
- The final chunk is not trained because all validation loss has already been scored.
- Multi-GPU TTT manually all-reduces gradients across active ranks before clipping and `optimizer.step()`.
- TTT uses the same byte accounting and `VAL_BYTE_COUNT_OVERRIDE` path as standard validation.

Focused checks run:

- `python -m py_compile train_gpt_parcae.py`
- Synthetic CUDA one-epoch TTT smoke passed.
- Synthetic `TTT_EPOCHS=0` equivalence check matched a direct bf16 sliding-window reference exactly (`diff=0` for loss and BPB).

Current caveats:

- No full Scylla TTT result has been established yet.
- Quantized-weight SGD TTT is experimental and may hurt; the previous record-submission code warned that SGD TTT on quantized weights was unfavorable in that setup.
- Manual full-parameter gradient all-reduce is correctness-oriented but may be expensive.
- The latest cleanup reduced duplicated expressions but added helper surface; `_validation_result` and `_token_byte_sum` are worth keeping for metric safety, while `_window_batch` is debatable if minimizing line count becomes the priority.

### SwiGLU recurrent input injection

Implemented an opt-in `INJECTION_TYPE=swiglu-add` recurrent adapter in `train_gpt_parcae.py`.

Added knob:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `INJECTION_SWIGLU_SCALE` | `0.0` | Initial scalar multiplier for the SwiGLU input-injection branch |

Implementation behavior:

- The adapter projects original input embeddings to `2 * recurrent_dim`, splits into `gate` and `value`, then injects `scale * silu(gate) * value` into the recurrent state.
- Default scale is `0.0`, so the branch is opt-in and starts as a no-op unless a run explicitly sets a nonzero scale.
- Validation rejects negative `INJECTION_SWIGLU_SCALE`.

Current caveats:

- No completed result proves this injection helps.
- The default was changed from an earlier local `0.1` to `0.0` to keep the new branch inactive by default.

## 2026-04-25 Scylla / XSA / MoE / PoE session

This section is the central record for the Scylla tokenizer/scoring work, Parcae-recipe checks, XSA analysis, coda-only MoE attempt, and Product-of-Experts output-head tuning.

### Current best from this session

The best result from this session is the Scylla recurrent stack with 3 total categorical PoE experts and the larger BigramHash cache.

| Field | Value |
| --- | --- |
| Exact final BPB | **1.69172418** |
| Exact final loss | **2.90430866** |
| Run log | `logs/exp_scylla_poe3_lr0p002_bigram8192_2000.txt` |
| Key delta | `POE_NUM_EXPERTS=3 POE_HEAD_LR=0.002` |
| Base launcher | `scripts/run_parcae_scylla_current_best.sh` |
| Scylla data | `data_scylla/fineweb_scylla` |
| Tokenizer | `data_scylla/tokenizers/scylla/candidate.vocab` |
| Tokenizer metadata | `data_scylla/tokenizers/scylla/candidate.meta.npz` |
| Corrected val byte count | `VAL_BYTE_COUNT_OVERRIDE=151080363` |
| Total int8+zlib submission size | 5,348,437 bytes |

Current Scylla leaderboard from this session:

| Rank | Variant | Exact / corrected BPB | Notes |
| ---: | --- | ---: | --- |
| 1 | PoE3, `POE_HEAD_LR=0.002`, `BIGRAM_HASH_BUCKETS=8192` | **1.69172418** | Current best |
| 2 | PoE3, `POE_HEAD_LR=0.002`, `BIGRAM_HASH_BUCKETS=4096` | **1.69408507** | Clean rerun with final roundtrip |
| 3 | Dense Scylla baseline, corrected | about **1.69637493** | Log was pre-byte-override, corrected from exact val loss |
| 4 | PoE3, `POE_HEAD_LR=0.0025` | 1.69787138 | Best of second PoE sweep |
| 5 | PoE3, `POE_HEAD_LR=0.001` | 1.69919329 | Worse than `0.002` |
| 6 | PoE3, `POE_HEAD_LR=0.008` | 1.69923407 | Initial PoE try, too aggressive |
| 7 | PoE2, `POE_HEAD_LR=0.004` | 1.70085542 | Worse |
| 8 | XSA4 corrected | about 1.70250566 | XSA implementation is correct but hurt here |
| 9 | Eff8 prelude/coda + XSA3 | 1.70473223 | More effective layers plus XSA hurt |
| 10 | Coda MoE4 top-1 | 1.71156502 | Worse |

### BigramHash bucket-size sweep on current PoE best

A focused sweep was run on the best current-scaffold PoE config (`POE_NUM_EXPERTS=3`, `POE_HEAD_LR=0.002`) with varying `BIGRAM_HASH_BUCKETS`.

| Run | `BIGRAM_HASH_BUCKETS` | Final exact BPB | Final exact loss | Train time (ms) | Step avg (ms) | int8+zlib bytes | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `logs/exp_scylla_poe3_lr0p002_bigram4096_clean_2000.txt` | 4096 | 1.69408507 | 2.90836178 | 412858 | 206.43 | 5,357,060 | Worse than 8192 |
| `logs/exp_scylla_poe3_lr0p002_bigram8192_2000.txt` | 8192 | 1.69172418 | 2.90430866 | 397444 | 198.72 | 6,368,860 | **Current best** |
| `logs/exp_scylla_poe3_lr0p002_bigram16384_2000.txt` | 16384 | 1.71050917 | 2.93655825 | 423607 | 211.80 | 8,403,272 | Worse |
| `logs/exp_scylla_poe3_lr0p002_bigram32768_2000.txt` | 32768 | 1.70532632 | 2.92766048 | 404159 | 202.08 | 12,469,815 | Worse |

Findings from this sweep:
- `BIGRAM_HASH_BUCKETS=8192` improved validation quality by ~0.00236 BPB over 4096 and was not slower in this concurrent launch.
- Larger caches did not continue improving; 16384/32768 were clearly worse on BPB and significantly increased artifact size.
- The 8192 result remains a concurrent-run result sharing global artifact filenames during execution; for official reporting, a solo rerun is still recommended.

### Current-best Scylla base config

Unless otherwise noted, the Scylla experiments used the current recurrent stack:

| Setting | Value |
| --- | --- |
| `DATA_PATH` | `/workspace/parameter-golf/data_scylla/fineweb_scylla` |
| `TOKENIZER_PATH` | `/workspace/parameter-golf/data_scylla/tokenizers/scylla/candidate.vocab` |
| `TOKENIZER_META_PATH` | `/workspace/parameter-golf/data_scylla/tokenizers/scylla/candidate.meta.npz` |
| `VOCAB_SIZE` | `998` |
| `VAL_BYTE_COUNT_OVERRIDE` | `151080363` |
| `MODEL_DIM` / `RECURRENT_DIM` | `256` / `256` |
| `NUM_HEADS` / `NUM_KV_HEADS` | `4` / `2` |
| `RECURRENT_NUM_HEADS` | `4` |
| `N_LAYERS_IN_PRELUDE` / core / coda | `1` / `2` / `1` |
| `MEAN_RECURRENCE` / `MEAN_BACKPROP_DEPTH` | `2` / `1` |
| `TRAIN_SEQ_LEN` / `TRAIN_BATCH_TOKENS` | `512` / `16384` |
| `MLP_MULT` | `3` |
| `QK_NORM` | `1` |
| `USE_VALUE_EMBEDDINGS` | `0` |
| BigramHash | `BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=128 BIGRAM_HASH_HEADS=2 BIGRAM_HASH_GATE=1` |

### Scylla tokenizer and byte-count work

Implemented Scylla / TokenMonster metadata support in `train_gpt_parcae.py` and wired defaults into `scripts/run_parcae_scylla_current_best.sh`.

Code support added:

| Item | Purpose |
| --- | --- |
| `TOKENIZER_META_PATH` | Direct path for tokenizer metadata `.npz` |
| `TOKENIZER_META_VALIDATE` | Optional metadata validation path |
| Non-`.model` metadata discovery | Allows `candidate.vocab` to find `candidate.meta.npz` |
| Strict metadata vocab-size check | Prevents silent tokenizer/model mismatch |
| `VAL_BYTE_COUNT_OVERRIDE` | Uses a corrected validation-byte denominator for TokenMonster capcode |

Scylla files placed and checked:

| File | Purpose |
| --- | --- |
| `data_scylla/fineweb_scylla/fineweb_train_000000.bin` | Scylla training shard |
| `data_scylla/fineweb_scylla/fineweb_val_000000.bin` | Scylla validation shard |
| `data_scylla/tokenizers/scylla/candidate.vocab` | TokenMonster tokenizer vocab |
| `data_scylla/tokenizers/scylla/candidate.meta.npz` | Tokenizer byte metadata |

Checks:

- SHA256 checksums matched `data_scylla/SHA256SUMS.txt`.
- `candidate.vocab` loads with TokenMonster and reports vocab size 998.
- `candidate.meta.npz` has `tokenizer_kind=tokenmonster`, vocab size 998, and arrays of shape `(998,)`.
- Metadata per-token decoded byte lengths match `id_to_token_decoded(...).encode("utf-8")`.
- Full validation byte accounting does not equal the summed per-token metadata because TokenMonster capcode is context-dependent.

Byte-count measurements:

| Measurement | Value |
| --- | ---: |
| Scylla metadata-summed validation byte count | 157,319,833 |
| TokenMonster decoded Scylla validation bytes | about 151,073,500 |
| SP1024 full validation bytes | 151,080,891 |
| SP1024 seq512 scored validation bytes used for comparable BPB | 151,080,363 |

Conclusion: Scylla is usable in this codebase, but BPB must use the corrected denominator for the current seq512 evaluation setup: `VAL_BYTE_COUNT_OVERRIDE=151080363`. Old Scylla logs before this override have metadata BPB that is too optimistic and must be corrected from exact `val_loss`.

### Official Parcae recipe comparison and small recipe checks

Official Parcae-style settings we identified as missing or inactive in the local current-best runs:

| Item | Local status at the time | Official-style setting |
| --- | --- | --- |
| Prelude norm | off by default | `PRELUDE_NORM=1` |
| Gradient clipping | off by default | `GRAD_CLIP_NORM=1.0` |
| Recurrence sampling | fixed per-batch | per-sequence, poisson-truncated-full |
| Injection optimizer treatment | `B` still goes through matrix optimizer/WD path | no weight decay for injection params |
| Init recipe | close but not exact | scaled-zero + orthogonal details |
| Partial-depth eval / recurrence diagnostics | lighter | richer official diagnostics |

Focused checks run around the recurrent `2/1` SP1024 baseline:

| Run | Log | Config delta | Final exact BPB | Notes |
| --- | --- | --- | ---: | --- |
| Prelude norm 1930-step | `runs/exp_recur2_bptt1_preludenorm_1930/logs/exp_recur2_bptt1_preludenorm_1930.txt` | `PRELUDE_NORM=1` | 1.71984287 | Worse |
| Grad clip 1930-step | `runs/exp_recur2_bptt1_clip_1930/logs/exp_recur2_bptt1_clip_1930.txt` | `GRAD_CLIP_NORM=1.0` | 1.71223512 | Worse than best, better than prelude norm alone |
| Prelude norm + grad clip solo | `runs/exp_recur2_bptt1_preludenorm_clip_solo/logs/exp_recur2_bptt1_preludenorm_clip_solo.txt` | `PRELUDE_NORM=1 GRAD_CLIP_NORM=1.0` | 1.70760220 | Did not beat `1.70106946` |
| QK gain 5.25 solo | `runs/exp_recur2_bptt1_qkgain525_solo/logs/exp_recur2_bptt1_qkgain525_solo.txt` | `QK_GAIN_INIT=5.25` | 1.71530074 | Worse |
| No-loop current stack solo | `runs/exp_noloop_currentstack_solo/logs/exp_noloop_currentstack_solo.txt` | no recurrence with current stack | 1.71097365 | Faster but did not beat recurrence |

Conclusion: for the local short-budget setup, simply adding prelude norm, grad clipping, or higher QK gain did not improve the recurrent best. Recurrence still helped versus the no-loop current-stack rerun.

### XSA implementation profile and experiments

XSA behavior was checked directly before interpreting experiment results.

Implementation checks:

- `_xsa_efficient` matched an explicit `repeat_interleave` reference exactly in fp32.
- Post-XSA value-aligned projection was near zero in fp32 and small in bf16.
- Zero-value inputs did not produce NaNs.
- Gradients remained finite.
- Routing checks matched expected active layers: XSA4 on depth 6 activated `[2, 3, 4, 5]`; XSA3 on effective depth 8 activated `[5, 6, 7]`; XSA0 made zero calls.

Experiment results:

| Run | Config | Exact / corrected BPB | Notes |
| --- | --- | ---: | --- |
| `logs/exp_scylla_xsa4_full_2000.txt` | current Scylla stack, `XSA_LAST_N=4` | about 1.70250566 corrected | Worse than dense |
| `logs/exp_scylla_eff8_p2c2_xsa3_full_2000.txt` | prelude 2, core 2, coda 2, `XSA_LAST_N=3` | 1.70473223 | Worse |

Interpretation: XSA appears correctly implemented, but it likely removes useful value-aligned state in late recurrent/coda layers. If revisited, test weaker or more targeted variants: last-1 only, coda-only, fractional subtraction, or scheduled XSA.

### Coda-only MoE implementation and experiment

Implemented configurable coda-only MoE in `train_gpt_parcae.py`.

Added knobs:

| Variable | Meaning |
| --- | --- |
| `CODA_MOE_NUM_EXPERTS` | Enables MoE only in coda blocks when greater than zero |
| `CODA_MOE_TOP_K` | Sparse top-k routing count |
| `CODA_MOE_MLP_MULT` | Optional expert MLP width multiplier override |

Implementation notes:

- Prelude and recurrent core remain dense.
- Coda blocks can use `TopKMoE` instead of the dense MLP.
- Router gradient handling was fixed to gather full-softmax probabilities over selected experts. The first top-k-only softmax form collapses router gradients for top-1 routing.
- Expert fc/proj/router init is handled in `_init_weights`.

Experiment:

| Run | Config | Exact BPB | Notes |
| --- | --- | ---: | --- |
| `logs/exp_scylla_coda_moe4_top1_full_2000.txt` | `CODA_MOE_NUM_EXPERTS=4 CODA_MOE_TOP_K=1` | 1.71156502 | Worse than dense and PoE |

Conclusion: coda-only top-1 MoE did not help in this tested form. If revisited, change routing dynamics rather than simply adding more experts: smaller router LR, top-2, router temperature, delayed MoE activation, or router warmup.

### Product-of-Experts output heads

Implemented token-level categorical Product of Experts output heads in `train_gpt_parcae.py`.

Added knobs:

| Variable | Meaning |
| --- | --- |
| `POE_NUM_EXPERTS` | Total categorical experts, including the base tied/untied output head |
| `POE_HEAD_LR` | Adam LR for extra PoE output heads |

Implementation notes:

- Final logits are the sum of base logits plus extra expert logits.
- This implements categorical PoE because `softmax(sum_i logits_i)` is proportional to the product of the expert categorical distributions.
- Extra PoE heads are zero-init, so `POE_NUM_EXPERTS>1` starts as a no-op.
- Extra heads are excluded from Muon/scalar groups and optimized with the head Adam path.
- `POE_NUM_EXPERTS=1` preserves baseline behavior.

Initial PoE result:

| Run | Config | Exact BPB | Notes |
| --- | --- | ---: | --- |
| `logs/exp_scylla_poe3_full_2000.txt` | `POE_NUM_EXPERTS=3 POE_HEAD_LR=0.008` | 1.69923407 | Close but worse than corrected dense baseline |

PoE sweep 1:

| Run | Config | Exact BPB | Notes |
| --- | --- | ---: | --- |
| `logs/exp_scylla_poe2_lr0p004_full_2000.txt` | 2 experts, lr `0.004` | 1.70085542 | Worse |
| `logs/exp_scylla_poe2_lr0p002_full_2000.txt` | 2 experts, lr `0.002` | 1.70887737 | Worse |
| `logs/exp_scylla_poe3_lr0p004_full_2000.txt` | 3 experts, lr `0.004` | 1.70408370 | Worse |
| `logs/exp_scylla_poe3_lr0p002_full_2000.txt` | 3 experts, lr `0.002` | **1.69388607** | Best |

PoE sweep 2:

| Run | Config | Exact BPB | Notes |
| --- | --- | ---: | --- |
| `logs/exp_scylla_poe3_lr0p0010_full_2000.txt` | 3 experts, lr `0.001` | 1.69919329 | Worse |
| `logs/exp_scylla_poe3_lr0p0015_full_2000.txt` | 3 experts, lr `0.0015` | 1.70530234 | Worse |
| `logs/exp_scylla_poe3_lr0p0025_full_2000.txt` | 3 experts, lr `0.0025` | 1.69787138 | Best of sweep 2, still worse than lr `0.002` |
| `logs/exp_scylla_poe4_lr0p0015_full_2000.txt` | 4 experts, lr `0.0015` | 1.70043171 | Worse and larger artifact |

Conclusion: PoE is now the best tested path, but the simple count/LR response is sharp. `POE_NUM_EXPERTS=3 POE_HEAD_LR=0.002` is the current best. The next PoE work should be structural, not another small scalar LR sweep: delayed PoE-head activation, extra-head LR warmup, extra-head logit scaling, expert dropout/gating, or a clean solo rerun of the best setting.

### Scale test status

An attempted larger Scylla scale run was started as `logs/exp_scylla_scale448_full_2000.txt` with roughly 10.6M params, but it stopped during or near warmup and has no final comparable result. There is currently no proof that PoE scales to the 16MB-ish setting; a paired dense-vs-PoE scale test is still required.

### Caveats from this session

- Several Scylla baseline/XSA logs were produced before `VAL_BYTE_COUNT_OVERRIDE`; their exact logged BPB is metadata-denominator BPB and must be corrected from exact val loss.
- Some sweeps ran concurrently. The validation lines are usable, but concurrent final export/eval can race on shared `final_model*` filenames. If a result is promoted as official, rerun it alone.
- The dense Scylla baseline should be rerun with the current override for a clean paired official comparison against PoE.

## Current Recurrent Working Model

This is the best recurrent Parcae model so far and the current working baseline for recurrent tuning.
The best clean result now keeps recurrence at 2 but backpropagates only the final recurrent step.

| Field | Value |
| --- | --- |
| Exact final BPB | **1.70106946** |
| Exact final loss | **2.87218453** |
| Run log | `runs/exp_recur2_bptt1_solo/logs/exp_recur2_bptt1_solo.txt` |
| Retrieval diagnostic | not run yet |
| Exported artifacts | `runs/exp_recur2_bptt1_solo/final_model.pt` and `runs/exp_recur2_bptt1_solo/final_model.int8.ptz` |
| Launch mode | Direct `.venv/bin/python train_gpt_parcae.py` with no one-rank DDP wrapper |
| Key config | `MODEL_DIM=256 RECURRENT_DIM=256 NUM_HEADS=4 RECURRENT_NUM_HEADS=4 N_LAYERS_IN_RECURRENT_BLOCK=2 BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=128 BIGRAM_HASH_HEADS=2 BIGRAM_HASH_GATE=1 USE_VALUE_EMBEDDINGS=0 NUM_KV_HEADS=2 ROPE_DIMS=16 QK_NORM=1 MLP_MULT=3 TRAIN_SEQ_LEN=512 MEAN_RECURRENCE=2 MEAN_BACKPROP_DEPTH=1` |
| Steps / measured train time | 2042 steps / 300015 ms |
| Params | 4,295,873 |
| Total int8+zlib submission size | 4,820,852 bytes |

Command shape used for the current best:

```bash
RUN_ID=parcae_min_5min_recur2_seq512_mlp3_bigram4096x128h2_gate_no_value_kv2_qknorm_rope16_20260425 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MODEL_DIM=256 \
NUM_HEADS=4 \
NUM_KV_HEADS=2 \
MLP_MULT=3 \
N_LAYERS_IN_PRELUDE=1 \
N_LAYERS_IN_RECURRENT_BLOCK=2 \
N_LAYERS_IN_CODA=1 \
RECURRENT_DIM=256 \
RECURRENT_NUM_HEADS=4 \
MEAN_RECURRENCE=2 \
MEAN_BACKPROP_DEPTH=1 \
TRAIN_BATCH_TOKENS=16384 \
TRAIN_SEQ_LEN=512 \
ITERATIONS=1000000 \
MAX_WALLCLOCK_SECONDS=300 \
WARMUP_STEPS=500 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=0 \
COMPILE_MODEL=0 \
COMPILE_MUON_BACKEND=0 \
ROPE_DIMS=16 \
QK_NORM=1 \
USE_VALUE_EMBEDDINGS=0 \
BIGRAM_HASH_BUCKETS=4096 \
BIGRAM_HASH_DIM=128 \
BIGRAM_HASH_HEADS=2 \
BIGRAM_HASH_GATE=1 \
python train_gpt_parcae.py
```

## Hyperloop / Hyperconnection Attempt

Status as of 2026-04-25: deprioritize as a main path. The best Hyperloop result is still worse than both the comparable recurrent `2/2` baseline and the current `2/1` SOTA.

Implementation lived in the separate worktree `/workspace/parameter-golf-hyperconn` on branch `hyperconn-loop`.
The main checkout was left untouched except for this experiment note.

Implemented behavior:

- Added loop-level Hyperloop-style hyperconnections for the recurrent middle block.
- Duplicated the post-prelude residual stream into `HYPERCONN_STREAMS` parallel streams.
- Used input-dependent `H_pre`, `H_post`, and diagonal sigmoid `H_res` per loop.
- Added loop position embeddings.
- Applied hyperconnections only after each complete recurrent loop, not after every layer.
- Removed the existing Parcae adapter when Hyperloop is enabled, so this tested replacing that recurrent residual adapter with Hyperloop routing.
- Current implementation requires `MEAN_BACKPROP_DEPTH == MEAN_RECURRENCE`, which prevents a direct comparison to the current `2/1` SOTA.
- Added diagnostics via `HYPERCONN_DIAG_SEQS`: per-loop gate stats, per-stream norms, and inter-stream cosine similarity after final roundtrip validation.

Focused checks run before trusting numbers:

- `python -m py_compile train_gpt_parcae.py`
- `git diff --check`
- Disabled-path equivalence to the pre-Hyperloop script for state keys, initialized weights, and deterministic loss.
- Gate math checks for duplicate streams, read/write equations, expected `H_post` and `H_res` initial values, and input-dependent nonzero gates.
- Integrated recurrence check against a hand-computed deterministic middle block.
- Gradient and optimizer checks confirming hyperconnection parameters receive gradients and use their own Adam group through `HYPERCONN_LR`.
- Quantization/roundtrip checks confirming hyperconnection control tensors restore as fp32.
- Tiny CUDA smoke/stress tests after the LR split confirmed training no longer collapsed.

Run results:

| Run | Log | Config delta | Final BPB | Steps | Notes |
| --- | --- | --- | ---: | ---: | --- |
| Pre-fix Hyperloop | `/workspace/parameter-golf-hyperconn/runs/hyperconn_stream4_recur2_bptt2_solo_20260425/logs/hyperconn_stream4_recur2_bptt2_solo_20260425.txt` | Hyperconn params accidentally used `SCALAR_LR=0.04` | 4.10520466 | 1429 | Invalid. Gates saturated; logits collapsed to zero and train loss stuck at `6.9315`. |
| Hyperloop fixed LR | `/workspace/parameter-golf-hyperconn/logs/hyperconn_stream4_recur2_bptt2_lr001_rerun_20260425.txt` | `HYPERCONN_LR=0.001`, `HYPERCONN_RES_INIT=0.05`, streams 4 | 1.72230267 | 1424 | Healthy training but worse than recurrent baselines. |
| Hyperloop ablation A | `/workspace/parameter-golf-hyperconn/logs/hyperconn_ablate_A_lr3e4_res005_clean_20260425.txt` | `HYPERCONN_LR=0.0003`, `HYPERCONN_RES_INIT=0.05`, streams 4 | 1.71694044 | 1430 | Best Hyperloop result so far; still worse than the comparable recurrent `2/2` baseline. |
| Hyperloop ablation B | `/workspace/parameter-golf-hyperconn/logs/hyperconn_ablate_B_lr3e4_res05_clean_20260425.txt` | `HYPERCONN_LR=0.0003`, `HYPERCONN_RES_INIT=0.5`, streams 4 | not completed | stopped at 400 | Aborted because external main-checkout workers started sharing the GPU; do not use this partial run. |

Comparisons:

- Current real SOTA: `1.70106946` BPB with `MEAN_RECURRENCE=2`, `MEAN_BACKPROP_DEPTH=1`.
- Comparable recurrent `2/2` baseline with seq512, MLP3, gated 2-head BigramHash: `1.71261107` BPB.
- Best Hyperloop so far: `1.71694044` BPB.
- Best Hyperloop is `+0.01587098` BPB worse than current SOTA and `+0.00432937` BPB worse than the comparable recurrent `2/2` baseline.

Diagnostics:

- The fixed-LR Hyperloop run was active, not dead: gates learned strongly input-dependent routing.
- With `HYPERCONN_LR=0.001`, gates were extreme. Loop 0 wrote almost entirely to one stream, causing major norm imbalance, and loop 1 nearly reset that stream.
- Raw and int8 roundtrip diagnostics were nearly identical, so quantization was not the cause.
- Lowering `HYPERCONN_LR` to `0.0003` improved BPB from `1.72230267` to `1.71694044` and reduced `H_pre` saturation.
- However, ablation A still showed stream recollapse: final post-loop stream cosine mean was `0.990655`, so the model paid for multi-stream routing without retaining useful stream diversity.

Conclusion:

- Scrap Hyperloop as the main path for now.
- The negative result is not proof that hyperconnections cannot work, but this small 2-loop Parcae setup did not benefit.
- The only follow-up worth considering later is a narrow implementation ablation with shared loop gates or another mechanism that supports `MEAN_BACKPROP_DEPTH=1`, since the current SOTA depends heavily on the `2/1` training regime.

## Parcae 5-minute ablations

As of 2026-04-24, the best protocol-clean Parcae run is:

| Run | Log | Val BPB | Steps | Train time | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Partial RoPE 25% | `logs/parcae_min_5min_partialrope25_eager_20260424_082528.txt` | 1.79185059 | 1603 | 300111 ms | `ROPE_DIMS=16`, eager mode, `WARMUP_STEPS=500`, synchronized CUDA timing |
| MuonEq-R port | `logs/parcae_min_5min_muoneqr_sync_20260424_075025.txt` | 1.79851330 | 1593 | 300106 ms | Current-record MuonEq-R-style optimizer port, synchronized CUDA timing |
| LeakyReLU squared | `logs/parcae_min_5min_leakyrelu2_eager_20260424_080536.txt` | 1.83837160 | 1596 | 300024 ms | Replaced GELU MLP with LeakyReLU squared, eager mode |

The historical raw best number is:

| Run | Log | Val BPB | Steps | Notes |
| --- | --- | ---: | ---: | --- |
| Old local Parcae baseline | `logs/parcae_min_5min.txt` | 1.78675924 | 1806 | Not protocol-clean: used older timing behavior and `WARMUP_STEPS=0` |

Treat the old baseline as useful context, not as the comparable best. For current comparisons, use the synchronized timing protocol with a fixed warmup and matched runtime budget.

## Parcae attention/retrieval ablations

Protocol for these runs:

- 5-minute synchronized wall-clock cap.
- `WARMUP_STEPS=500`.
- `COMPILE_MODEL=0`, `COMPILE_MUON_BACKEND=0`.
- Base shape unless noted: `MODEL_DIM=256`, `NUM_HEADS=4`, `NUM_KV_HEADS=1`, `N_LAYERS_IN_PRELUDE=1`, `N_LAYERS_IN_RECURRENT_BLOCK=2`, `N_LAYERS_IN_CODA=1`, `RECURRENT_DIM=256`, `RECURRENT_NUM_HEADS=4`, `MEAN_RECURRENCE=2`, `MEAN_BACKPROP_DEPTH=2`, `TRAIN_BATCH_TOKENS=16384`, `TRAIN_SEQ_LEN=256`, `ROPE_DIMS=16`.
- FlashAttention 4 imported but failed at runtime on this GPU; all runs used the SDPA fallback after the wrapper disabled FA4.

| Run | Log | Val BPB | Steps | Params | Total int8+zlib size | Retrieval diagnostic |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline Partial RoPE 25% | `logs/parcae_min_5min_partialrope25_eager_20260424_082528.txt` | 1.79185059 | 1603 | 3,148,704 | 3,535,927 bytes | `logs/diag_parcae_partialrope25_baseline_20260424.json` |
| QK norm | `logs/parcae_min_5min_qknorm_20260425.txt` | 1.78353586 | 1580 | 3,148,704 | 3,529,763 bytes | `logs/diag_parcae_qknorm_20260425.json` |
| Mixed RoPE, recurrent 32 | `logs/parcae_min_5min_mixedrope_rec32_20260425.txt` | 1.80385179 | 1574 | 3,148,704 | 3,536,146 bytes | `logs/diag_parcae_mixedrope_rec32_20260425.json` |
| Core block 1 full RoPE + QK norm | `logs/parcae_min_5min_core1fullrope_qknorm_20260425.txt` | 1.79310933 | 1547 | 3,148,704 | 3,525,804 bytes | `logs/diag_parcae_core1fullrope_qknorm_20260425.json` |
| Outer KV heads 2 | `logs/parcae_min_5min_kvheads2_20260425.txt` | 1.78149694 | 1629 | 3,279,808 | 3,676,450 bytes | `logs/diag_parcae_kvheads2_20260425.json` |
| No value embeddings | `logs/parcae_min_5min_no_value_embeds_20260425.txt` | 1.77349693 | 1725 | 2,821,024 | 3,218,618 bytes | `logs/diag_parcae_no_value_embeds_20260425.json` |
| No value embeddings + QK norm | `logs/parcae_min_5min_no_value_embeds_qknorm_20260425.txt` | 1.78756470 | 1694 | 2,821,024 | 3,205,347 bytes | not run |
| No value embeddings + outer KV heads 2 | `logs/parcae_min_5min_no_value_embeds_kvheads2_20260425.txt` | 1.76844642 | 1731 | 2,886,592 | 3,324,882 bytes | `logs/diag_parcae_no_value_embeds_kvheads2_20260425.json` |
| No value embeddings + outer KV heads 2 + QK norm | `logs/parcae_min_5min_no_value_embeds_kvheads2_qknorm_20260425.txt` | 1.76437800 | 1748 | 2,886,592 | 3,320,308 bytes | `logs/diag_parcae_no_value_embeds_kvheads2_qknorm_20260425.json` |
| No value embeddings + outer KV heads 2 + QK norm, direct Python | `logs/parcae_min_5min_best_no_ddp_no_value_kv2_qknorm_rope16_20260425.txt` | 1.76239973 | 1785 | 2,886,592 | 3,328,383 bytes | `logs/diag_parcae_best_no_ddp_no_value_kv2_qknorm_rope16_20260425.json` |
| No value embeddings + outer KV heads 2 + QK norm, direct Python restore | `logs/parcae_min_5min_best_restore2_no_ddp_no_value_kv2_qknorm_rope16_20260425.txt` | 1.76515237 | 1737 | 2,886,592 | 3,322,637 bytes | not run |
| Same architecture, no recurrent loop | `logs/parcae_min_5min_no_loop_same_arch_no_value_kv2_qknorm_rope16_20260425_full.txt` | 1.74502258 | 2487 | 2,886,592 | 3,383,037 bytes | not run |
| Recurrence 3, backprop depth 1 | `logs/parcae_min_5min_recur3_bptt1_no_value_kv2_qknorm_rope16_20260425.txt` | 1.77740456 | 1922 | 2,886,592 | 3,347,343 bytes | not run |
| One-layer recurrent core, recurrence 4, backprop depth 2 | `logs/parcae_min_5min_core1_recur4_bptt2_no_value_kv2_qknorm_rope16_20260425_rerun.txt` | 1.78728707 | 2158 | 2,099,520 | 2,487,224 bytes | not run |
| Recurrent 2/2 + seq512 + MLP3 | `logs/parcae_min_5min_recur2_seq512_mlp3_no_value_kv2_qknorm_rope16_20260425.txt` | 1.72050566 | 1702 | 3,148,736 | 3,540,682 bytes | not run |
| Recurrent 2/2 + seq512 + MLP3 + gated 2-head BigramHash 4096x128 | `logs/parcae_min_5min_recur2_seq512_mlp3_bigram4096x128h2_gate_no_value_kv2_qknorm_rope16_20260425.txt` | 1.71261107 | 1618 | 4,295,873 | 4,678,173 bytes | not run |
| Recurrent 2/1 + seq512 + MLP3 + gated 2-head BigramHash 4096x128, solo | `runs/exp_recur2_bptt1_solo/logs/exp_recur2_bptt1_solo.txt` | 1.70106946 | 2042 | 4,295,873 | 4,820,852 bytes | not run |
| Recurrent 2/1 + seq512 + MLP3 + gated 2-head BigramHash 4096x128, 4-way parallel | `runs/exp_recur2_bptt1/logs/exp_recur2_bptt1.txt` | 1.7477 | 1373 | 4,295,873 | 4,825,376 bytes | throughput-confounded parallel run |
| Recurrent 2/2 + seq512 + MLP3 + gated 2-head BigramHash 8192x128, 4-way parallel | `runs/exp_bigram8192/logs/exp_bigram8192.txt` | 1.7595 | 1362 | not recorded | 5,836,479 bytes | throughput-confounded parallel run |
| Recurrent 2/2 + seq512 + MLP3 + gated 2-head BigramHash 4096x64, 4-way parallel | `runs/exp_bigram64/logs/exp_bigram64.txt` | 1.7676 | 1364 | not recorded | 4,308,914 bytes | throughput-confounded parallel run |
| Recurrent 2/2 + seq512 + MLP4 + gated 2-head BigramHash 4096x128, 4-way parallel | `runs/exp_mlp4/logs/exp_mlp4.txt` | 1.7635 | 1362 | not recorded | 5,064,612 bytes | throughput-confounded parallel run |
| No-loop + seq512 + MLP3 + gated 2-head BigramHash 4096x128 | `logs/parcae_min_5min_noloop_seq512_mlp3_bigram4096x128h2_gate_no_value_kv2_qknorm_rope16_20260425.txt` | 1.73720295 | 2195 | 4,295,873 | 4,527,969 bytes | not run |
| No value embeddings + outer KV heads 2 + QK norm + RoPE 8 | `logs/parcae_min_5min_no_value_embeds_kvheads2_qknorm_rope8_20260425.txt` | 1.79321577 | 1697 | 2,886,592 | 3,314,402 bytes | `logs/diag_parcae_no_value_embeds_kvheads2_qknorm_rope8_20260425.json` |
| No value embeddings + outer KV heads 2 + QK norm + RoPE 32 | `logs/parcae_min_5min_no_value_embeds_kvheads2_qknorm_rope32_20260425.txt` | 1.76441375 | 1714 | 2,886,592 | 3,322,038 bytes | `logs/diag_parcae_no_value_embeds_kvheads2_qknorm_rope32_20260425.json` |
| No value embeddings + outer KV heads 2 + QK norm + RoPE 32, direct Python | `logs/parcae_min_5min_no_ddp_no_value_kv2_qknorm_rope32_20260425.txt` | 1.76400278 | 1723 | 2,886,592 | 3,324,306 bytes | `logs/diag_parcae_no_ddp_no_value_kv2_qknorm_rope32_20260425.json` |

Best no-loop clean result: direct `python train_gpt_parcae.py` with `USE_VALUE_EMBEDDINGS=0 NUM_KV_HEADS=2 ROPE_DIMS=16 QK_NORM=1 MEAN_RECURRENCE=1 MEAN_BACKPROP_DEPTH=1`, with exact final `val_bpb=1.74502258`.

Best recurrent clean result and current recurrent working baseline: direct `.venv/bin/python train_gpt_parcae.py` with `MODEL_DIM=256 RECURRENT_DIM=256 NUM_HEADS=4 RECURRENT_NUM_HEADS=4 N_LAYERS_IN_RECURRENT_BLOCK=2 BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=128 BIGRAM_HASH_HEADS=2 BIGRAM_HASH_GATE=1 USE_VALUE_EMBEDDINGS=0 NUM_KV_HEADS=2 ROPE_DIMS=16 QK_NORM=1 MLP_MULT=3 TRAIN_SEQ_LEN=512 MEAN_RECURRENCE=2 MEAN_BACKPROP_DEPTH=1`, with exact final `val_bpb=1.70106946`.

Observed diagnostic pattern:

- Removing value embeddings improved both speed and quality in this local 5-minute regime.
- Increasing outer KV heads from 1 to 2 helped both with and without value embeddings. The stacked no-value + KV2 + QK-norm variant was best, and running it directly with `python train_gpt_parcae.py` was faster than one-rank `torch.distributed.run`.
- QK norm helped with value embeddings on, hurt for no-value/KV1, but helped again for no-value/KV2.
- Larger recurrent RoPE did not help. `RECURRENT_ROPE_DIMS=32` was the worst run, and making only `core_block.1` full-RoPE with QK norm also lost to QK norm alone. In the best no-value/KV2/QK-norm setup, `ROPE_DIMS=8` was much worse and `ROPE_DIMS=32` was close but still worse than `ROPE_DIMS=16`, including under direct Python.
- The best run still mostly improves local/recent-context behavior rather than becoming an obvious exact-copy retrieval model. On the 64-sequence diagnostic slice, seen-last-32 loss improved from 2.4584 in the baseline to 2.3411 in the best direct-Python run, while leading-space loss improved from 4.3271 to 4.2782.
- Turning off the recurrent loop while keeping the same module shape and parameter count was the biggest single gain so far: `MEAN_RECURRENCE=1 MEAN_BACKPROP_DEPTH=1` improved exact final BPB from 1.76239973 to 1.74502258 and increased measured steps from 1785 to 2487 in the same 300-second training budget. This suggests the looped core was not paying for its extra compute at this scale; fewer effective passes trained more examples and won decisively.
- `MEAN_RECURRENCE=3 MEAN_BACKPROP_DEPTH=1` did not recover the recurrent advantage. It improved early quality per step relative to no-loop at step 1000, but finished worse than both the recurrent `2/2` baseline and no-loop because it was still slow at 156.11 ms/step and reached only 1922 measured steps.
- A one-layer recurrent core with `MEAN_RECURRENCE=4 MEAN_BACKPROP_DEPTH=2` was smaller and faster than the 2-layer recurrent baseline, but quality fell to 1.78728707 BPB. The smaller core reached 2158 steps at 139.02 ms/step, so the failure was not only speed; one recurrent block layer removed too much capacity.
- Increasing context to `TRAIN_SEQ_LEN=512` and outer/coda MLP expansion to `MLP_MULT=3` was the strongest recurrent improvement so far. It raised params from 2.89M to 3.15M and slowed steps from 168.16 ms to 176.30 ms, but exact roundtrip BPB improved from 1.76239973 to 1.72050566. The model used more capacity and longer context effectively despite fewer steps.
- Adding gated 2-head BigramHash on top of seq512/MLP3 improved exact BPB further from 1.72050566 to 1.71261107. It slowed steps from 176.30 ms to 185.48 ms and raised params to 4.30M, but the explicit bigram signal paid for the extra compute and artifact size.
- Reducing backprop depth from 2 to 1 while keeping `MEAN_RECURRENCE=2` produced the new best clean recurrent run: exact final BPB improved from 1.71261107 to 1.70106946 and measured steps increased from 1618 to 2042. The result suggests the second recurrent forward pass is useful, but backpropagating through both passes is not worth the compute in the 5-minute regime.
- Four-way parallel screening was useful only as a directional filter. The parallel runs reached only about 1362-1373 steps at roughly 218-220 ms/step, versus the solo `MEAN_BACKPROP_DEPTH=1` rerun reaching 2042 steps at 146.92 ms/step. Treat the parallel BPB numbers as throughput-confounded until rerun solo.
- BigramHash implementation has since been simplified to the fixed hash only: explicit bucket-0 sentinel, shifted real bigrams, independent per-head hash constants, and small-random projection init. The recorded 1.71261107 run used the earlier hash mapping, while the 1.70106946 backprop-depth-1 run used the simplified mapping.
- Re-running no-loop with the full current stack (`TRAIN_SEQ_LEN=512 MLP_MULT=3 BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_HEADS=2 BIGRAM_HASH_GATE=1`) did not beat recurrence. It reached 2195 steps at 136.74 ms/step, but final exact BPB was 1.73720295 versus the recurrent stack's 1.71261107. In the current higher-capacity + BigramHash setup, recurrence is clearly useful despite slower steps.
Attempted `NUM_KV_HEADS=4` direct Python (`logs/parcae_min_5min_no_ddp_no_value_kv4_qknorm_rope16_20260425.txt`), but the process exited before final validation after step 500. Early train loss and step time were worse than KV2, so this did not look promising enough to rerun immediately.
