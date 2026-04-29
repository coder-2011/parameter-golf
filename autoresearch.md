# Autoresearch: Parcae BPB Optimization

This file is the working reference for agents running `./autoresearch.sh`.
Read it every few runs. The shell script is allowed to change during research,
and stale assumptions waste GPU time.

## Current Goal

Optimize validation bits-per-byte for `train_gpt_parcae.py` on the local
single-GPU autoresearch harness.

The local harness is a proxy, not the final leaderboard environment. It targets
fast, comparable 300 second experiments on 1 GPU and records exact roundtrip
BPB from the compressed artifact path. A lower `val_bpb` is better.

Current focus:

- SP1024 baseline data.
- 300 second training wall-clock cap.
- QAT6 plus int6 rANS export for small artifacts.
- Parallel residual in the recurrent core only.
- Muon momentum near `0.85`.
- LeakyReLU(0.5)^2 MLP with `MLP_MULT=4.0`.

## Run Command

Use a plain invocation unless the orchestration wrapper explicitly supports
environment overrides:

```bash
./autoresearch.sh
```

The runner writes the full training log to:

```bash
logs/${RUN_ID}.txt
```

and prints structured lines:

```text
METRIC val_bpb=...
METRIC steps=...
METRIC step_avg_ms=...
METRIC submission_bytes=...
METRIC train_loss=...
METRIC peak_memory_mb=...
METRIC train_time_ms=...
```

The wrapper previously rejected `run_experiment env ... bash autoresearch.sh`.
When that happens, edit `autoresearch.sh` defaults directly, then run plain
`bash autoresearch.sh` or `./autoresearch.sh`.

## Metric Contract

Primary metric:

- `val_bpb`: exact roundtrip validation BPB from the final compressed artifact.

The parser currently accepts these final log forms:

- `final_int8_zlib_roundtrip_exact val_loss:... val_bpb:...`
- `final_int6_zlib_roundtrip_exact val_loss:... val_bpb:...`
- `final_int6_rans_zlib_roundtrip_exact val_loss:... val_bpb:...`
- GPTQ variants with `final_gptq_int*_...`

Secondary metrics:

- `steps`: completed optimizer steps inside the training wall-clock cap.
- `step_avg_ms`: average step time from the final training progress line.
- `submission_bytes`: total code plus compressed model bytes.
- `train_loss`: final logged training loss.
- `peak_memory_mb`: peak allocated GPU memory.
- `train_time_ms`: wall-clock training time consumed before eval/export.

Do not compare runs if the tokenizer, data split, byte-count override,
wall-clock cap, quantization/export path, or validation method changed unless
the difference is explicitly the experiment.

## Current Runner Defaults

These are the effective defaults in `autoresearch.sh` as of this note.
Some variables are exported once and then forced later; the later force wins.

### Data And Tokenizer

| Variable | Current default | Meaning |
| --- | --- | --- |
| `DATA_PATH` | `./data/datasets/fineweb10B_sp1024` | Training and validation shard directory. |
| `TOKENIZER_PATH` | `./data/tokenizers/fineweb_1024_bpe.model` | SentencePiece tokenizer for SP1024. |
| `TOKENIZER_META_PATH` | empty | Optional metadata sidecar. |
| `TOKENIZER_META_VALIDATE` | `0` | Metadata validation disabled by default. |
| `VAL_BYTE_COUNT_OVERRIDE` | `0` | No byte-count override for SP1024. |
| `VOCAB_SIZE` | `1024` | Must match tokenizer. |

SP1024 is the cleanest default because BPB accounting needs no override.
Scylla/tokenmonster and any future CaseOps tokenizer require extra care around
byte reconstruction and byte counts.

### Shape

| Variable | Current default | Meaning |
| --- | --- | --- |
| `MODEL_DIM` | `384` | Outer embedding, prelude, and coda width. |
| `RECURRENT_DIM` | `384` | Recurrent-state width. |
| `RECURRENT_INTERMEDIATE_DIM` | `0` | Uses `MLP_MULT * RECURRENT_DIM` when zero. |
| `NUM_HEADS` | `4` | Outer attention query heads. |
| `NUM_KV_HEADS` | `2` | Outer KV heads for GQA. |
| `RECURRENT_NUM_HEADS` | `4` | Core attention heads. |
| `N_LAYERS_IN_PRELUDE` | `1` | Blocks before recurrence. |
| `N_LAYERS_IN_RECURRENT_BLOCK` | `3` | Physical blocks inside the recurrent core. |
| `N_LAYERS_IN_CODA` | `2` | Blocks after recurrence. |
| `MEAN_RECURRENCE` | `2` | Forward recurrence depth. |
| `MEAN_BACKPROP_DEPTH` | `1` | Backprop depth through recurrence. |

Physical block indices for the default shape:

- prelude: block `0`
- recurrent core: blocks `1..3`
- coda: blocks `4..5`

`PARALLEL_RESIDUAL_START` uses these physical indices, not recurrence-expanded
depth.

### Training Length

| Variable | Current default | Meaning |
| --- | --- | --- |
| `TRAIN_SEQ_LEN` | `1024` | Training context length. |
| `TRAIN_BATCH_TOKENS` | `131072` | Tokens per optimizer step. |
| `GRAD_ACCUM_STEPS` | script default `max(1, 8 // WORLD_SIZE)` | Gradient accumulation factor. The runner does not override it. |
| `ITERATIONS` | `1000000` | High ceiling; wall-clock stops first. |
| `MAX_WALLCLOCK_SECONDS` | `300` | Training wall-clock cap. |
| `WARMDOWN_ITERS` | `1200` | Warmdown schedule length. |
| `WARMUP_STEPS` | `500` | Warmup steps before timed training. |
| `VAL_BATCH_SIZE` | `524288` | Validation batch tokens. |
| `TRAIN_LOG_EVERY` | `100` | Training log cadence. |
| `VAL_LOSS_EVERY` | `0` | No periodic validation during training. |

Warmup consumes data and compile time, but the training state is restored after
warmup. Be careful when reasoning about "epochs": the data loader advances
during warmup even though optimizer/model state is reset.

### Runner Hygiene And Fixed Controls

The runner unsets several values before exporting its config:

- `NUM_LAYERS`: Parcae ignores it and requires prelude/core/coda layer counts.
- `QK_GAIN_INIT`: only applies to an unused baseline path.
- `CLIP_QKV`: incompatible with the current packed Tri Dao RoPE constraints.
- `RANK`, `LOCAL_RANK`, `WORLD_SIZE`: keeps the harness single-process.
- `CONTROL_TENSOR_NAME_PATTERNS`, `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS`: avoids
  stale quantization/control pattern overrides.

Additional fixed controls:

| Variable | Current default | Meaning |
| --- | --- | --- |
| `SEED` | `1337` | Random seed. Keep fixed for comparable runs. |
| `MONITORING` | `0` | Runtime monitoring disabled. |
| `LOCKSTEP_N` | `0` | Lockstep recurrence feature disabled. |
| `LOCKSTEP_K` | `0` | Lockstep recurrence feature disabled. |
| `EMB_SCALE` | `1.0` | Embedding scale. |
| `LOGIT_SCALE` | `1.0` | Logit scale. |
| `SAVE_RAW_MODEL` | `0` | Do not save raw float artifact by default. |

### Current Architecture Defaults

| Variable | Current default | Meaning |
| --- | --- | --- |
| `MLP_MULT` | `4.0` | MLP hidden multiplier. Parsed as float in current code. |
| `MLP_CLASS_NAME` | `LeakyReluSquaredMLP` | Outer/prelude/coda MLP class. |
| `RECURRENT_MLP_CLASS_NAME` | `LeakyReluSquaredMLP` | Recurrent core MLP class. |
| `ROPE_DIMS` | `32` | Partial RoPE dimension used when scope-specific dims are zero. |
| `OUTER_ROPE_DIMS` | `0` | Falls back to `ROPE_DIMS`. |
| `RECURRENT_ROPE_DIMS` | `0` | Falls back to `ROPE_DIMS`. |
| `RECURRENT_LAYER_ROPE_DIMS` | empty | Optional comma-separated per-core-block RoPE dims. |
| `QK_NORM` | `1` | RMSNorm on Q and K before attention. |
| `QK_BIAS` | `0` | Q/K bias disabled. |
| `USE_VALUE_EMBEDDINGS` | `0` | Per-layer value embeddings disabled. |
| `PRELUDE_NORM` | `0` | Extra prelude norm disabled. |
| `STATE_INIT` | `like-init` | Recurrent state initialization mode. |
| `INJECTION_TYPE` | `diagonal` | Input-to-state adapter type. |
| `INJECTION_SWIGLU_SCALE` | `0.0` | SwiGLU reinjection disabled. |
| `TIE_EMBEDDINGS` | `1` | Tied input/output embeddings. |
| `LOGIT_SOFTCAP` | `30.0` | Logit softcap before CE/export eval. |

MLP classes currently available:

- `BaseMLP`: dense up projection, GELU, dense down projection.
- `LeakyReluSquaredMLP`: dense up projection, `LeakyReLU(0.5)^2`, dense down
  projection, zero-initialized down projection.
- `GatedMLP`: fused dense projection split into GELU gate and value.
- `LigerStyleGatedMLP` / `LigerGEGLUMLP`: separate gate/up/down projections
  with Liger GELU multiply when available.

The current runner intentionally tests `LeakyReluSquaredMLP`. Older Parcae
5-minute ablations showed LeakyReLU squared was worse in a different protocol,
so treat this as an active experiment, not a proven permanent default.

### Attention And QKV Projection

| Variable | Runner default | Meaning |
| --- | --- | --- |
| `ATTN_QKV_MODE` | script default `packed` | QKV projection layout. Runner does not override it. |
| `RRHP_REDUCTION_FACTOR` | script default `4` | Reduction factor if `ATTN_QKV_MODE=rrhp`. |
| `PWA_NUM_BASES` | script default `6` | Number of shared PWA bases if PWA mode is enabled. |
| `PWA_PERMUTE_SIZE` | script default `4` | Permutation tuple size for PWA indexing. |
| `PWA_INIT_SCALE` | script default `1/sqrt(2)` | PWA basis init scale. |
| `PWA_LR` | script default `MATRIX_LR` | Optional PWA-specific learning rate. |
| `ATTN_PRECONV_KERNEL` | script default `0` | Depthwise pre-attention conv disabled by default. |
| `ATTENTION_WINDOW` | script default `-1` | Full attention unless overridden. |
| `TRIDAO_PACKED_ROPE` | `0` | Custom packed RoPE path disabled in autoresearch. |
| `LIGER_ROPE` | `0` | Liger RoPE disabled. |

QKV modes in current code:

- `packed`: standard packed QKV projection.
- `rrhp`: reduced-rank/head projection variant.
- `pwa`: PWA-compressed QKV projection.
- `pwa_qk_dense_v`: PWA-compressed Q/K with dense V.

Important PWA note: `pwa_qk_dense_v` is smaller but previous SP1892 testing
showed a large quality hit. It saved about 3.64M parameters and 2.57MB total
submission size versus packed, but lost about `+0.144` BPB. Do not assume the
saved parameters can be spent back into an equivalent score without evidence.

Important RoPE note: if `TRIDAO_PACKED_ROPE=1` is revisited, Q/K must be made
dense before the custom packed RoPE kernel. Passing a non-dense packed QK view
from a QKV layout with V-head gaps previously caused CUDA illegal memory access
under compiled backward. The safe path is to concatenate dense Q and K before
RoPE, then concatenate V afterward.

### Bigram Hash

| Variable | Current default | Meaning |
| --- | --- | --- |
| `BIGRAM_HASH_BUCKETS` | `8192` | Number of hash buckets. |
| `BIGRAM_HASH_DIM` | `128` | Hash embedding dimension. |
| `BIGRAM_HASH_HEADS` | `2` | Number of hash heads. |
| `BIGRAM_HASH_GATE` | `1` | Learned gate enabled. |
| `BIGRAM_HASH_SCALE_INIT` | `0.05` | Initial scale. |
| `BIGRAM_HASH_INIT_STD` | `0.02` | Hash embedding init std. |

BigramHash has been a strong feature in earlier local sweeps. Increasing
buckets from 4096 to 8192 was a small but positive change in the SP1024 stack.

### Parallel Residual

The runner currently forces:

```bash
RESIDUAL_MODE=parallel
PARALLEL_RESIDUAL_SCOPE=core
PARALLEL_RESIDUAL_START=-1
PARALLEL_RESIDUAL_LN_SCALE=1
```

Meaning:

- Only recurrent core blocks use parallel residual.
- Prelude and coda stay sequential.
- `START=-1` means all blocks in the selected scope.
- With the default physical indices, `core` means blocks `1..3`.

`PARALLEL_RESIDUAL_START=N` works only with `PARALLEL_RESIDUAL_IMPL=immediate`.
The delayed path currently requires `PARALLEL_RESIDUAL_START=-1`,
`PARALLEL_RESIDUAL_RECORD_CONTROLS=0`, and no gradient checkpointing.

Known result:

- Core-only parallel residual helped slightly.
- `PARALLEL_RESIDUAL_SCOPE=all` was worse.

### Attention Residuals And XSA

| Variable | Current default | Meaning |
| --- | --- | --- |
| `ATTN_RES_MODE` | `none` | Attention-residual feature disabled. |
| `ATTN_RES_SCOPE` | `all` | Scope if enabled. |
| `ATTN_RES_BLOCK_SIZE` | `2` | Block size if block mode is enabled. |
| `XSA_LAST_N` | `0` | XSA disabled. |

Constraints:

- Attention residuals require `RESIDUAL_MODE=sequential`.
- Attention residuals are not wired for gradient checkpointing.
- XSA is available and controlled by `XSA_LAST_N`, but tested values have been
  marginal/negative and slower in the current SP1024 family.

### Optimizer

| Variable | Current default | Meaning |
| --- | --- | --- |
| `EMBED_LR` | `0.6` | Embedding LR. |
| `HEAD_LR` | `0.008` | Head LR. |
| `TIED_EMBED_LR` | `0.05` | Tied embedding LR. |
| `MATRIX_LR` | `0.04` | Matrix LR. |
| `SCALAR_LR` | `0.04` | Scalar/control LR. |
| `MUON_MOMENTUM` | forced to `0.85` | Muon momentum. |
| `MUON_BACKEND_STEPS` | `5` | Newton-Schulz steps. |
| `MUON_MOMENTUM_WARMUP_START` | `0.85` | Momentum warmup start. |
| `MUON_MOMENTUM_WARMUP_STEPS` | `500` | Momentum warmup steps. |
| `MUON_ROW_NORMALIZE` | `1` | Row normalization enabled. |
| `MUON_WD` | `0.095` | Muon weight decay unless overridden before force points. |
| `BETA1` | `0.9` | Adam beta1. |
| `BETA2` | `0.95` | Adam beta2. |
| `ADAM_EPS` | `1e-8` | Adam epsilon. |
| `GRAD_CLIP_NORM` | `0.3` | Gradient clipping. |

Known result:

- `MUON_MOMENTUM=0.85` beat `0.90` and `0.95`.
- `MUON_MOMENTUM=0.80` was worse.
- `MUON_WD=0.04` was worse than `0.095`.
- `COMPILE_MUON_BACKEND=1` was not a win in earlier sweeps and complicated
  debugging because async CUDA errors can surface in the optimizer.

### Compile And Kernels

| Variable | Current default | Meaning |
| --- | --- | --- |
| `COMPILE_MODEL` | `1` | Compile the model. Strong throughput win on Blackwell. |
| `COMPILE_MUON_BACKEND` | `0` | Muon backend compile disabled in autoresearch. |
| `LIGER_CE` | `1` | Use Liger CE where possible. |
| `TRIDAO_PACKED_ROPE` | `0` | Disabled in autoresearch. |

Liger CE may graph-break or fall back around compiled paths. The current CE
wrapper uses plain PyTorch cross entropy while compiling and Liger outside that
compiled region when possible.

### Averaging

| Variable | Current default | Meaning |
| --- | --- | --- |
| `EMA_ENABLED` | `0` | EMA disabled. |
| `EMA` | `0` | Legacy EMA alias disabled. |
| `EMA_DECAY` | `0.997` | Decay if EMA is enabled. |
| `EMA_UPDATE_EVERY` | `1` | EMA cadence. |
| `SWA_ENABLED` | `1` | SWA enabled. |
| `SWA_START_FRAC` | `0.2` | Start SWA after 20 percent of training budget. |
| `SWA_EVERY` | `50` | SWA update cadence. |

Known result:

- SWA was a small positive.
- EMA with very high decay, especially `0.999`, badly underfit in short local
  runs because the EMA barely moved away from initialization.

### Quantization And Export

The runner currently forces the final quantization block to:

```bash
GPTQ_ENABLED=0
QAT_BITS=6
QAT_START_STEP=500
QUANT_BITS=6
RANS_INT6=1
```

This overrides earlier defaultable assignments in the script. Effective export
is int6 rANS, not GPTQ int8.

| Variable | Current effective value | Meaning |
| --- | --- | --- |
| `QAT_BITS` | `6` | Fake-quant training bits. |
| `QAT_START_STEP` | `500` | Step at which QAT begins. |
| `QAT_GROUP_SIZE` | script default `32` | TorchAO QAT group size. |
| `QAT_ACTIVATION_BITS` | script default `0` | Activation fake quantization disabled; only `0` or `8` is supported. |
| `QAT_LINEAR` | `1` | Apply QAT to linear modules. |
| `QAT_TIED_OUTPUT` | `1` | Include tied output path if applicable. |
| `QUANT_BITS` | `6` | Export bit width. |
| `QUANT_KEEP_FLOAT_PATTERNS` | empty | Optional comma-separated name patterns to keep in float during export. |
| `RANS_INT6` | `1` | rANS entropy coding for int6 symbols. |
| `GPTQ_ENABLED` | `0` | GPTQ disabled by force. |
| `GPTQ_MATRIX_CLIP_SIGMAS` | `12.85` | SDClip-style row clipping if GPTQ is enabled. |
| `GPTQ_EMBED_CLIP_SIGMAS` | `20.0` | Embedding row clipping if GPTQ is enabled. |

Important distinctions:

- QAT alone does not shrink the artifact unless `QUANT_BITS` is also lowered.
- GPTQ has SDClip-style per-row sigma clipping through the `GPTQ_*_CLIP_SIGMAS`
  knobs.
- Plain int6/RANS export uses the normal int export clipping path, not GPTQ's
  Hessian/blockwise quantization.
- `RANS_INT6=1` requires `QUANT_BITS=6`.

Known result:

- GPTQ int8 produced strong SP1024 results around the low/mid 1.42s.
- int6 rANS reduced artifact size substantially but was slightly worse at the
  same 384d shape.
- Scaling to 448d with int6 rANS was worse in the tested run because the slower
  step rate cost too many updates.

### MoE And Expert Features

| Variable | Current default | Meaning |
| --- | --- | --- |
| `POE_NUM_EXPERTS` | `1` | Prediction-of-experts head count; `1` means effectively off. |
| `POE_HEAD_LR` | `0.008` | LR for PoE heads. |
| `CODA_MOE_NUM_EXPERTS` | `0` | Coda MoE disabled. |
| `CODA_MOE_TOP_K` | `1` | Top-k if coda MoE enabled. |
| `CODA_MOE_MLP_MULT` | `0` | Expert MLP width override. |
| `DEEPSEEK_MOE_NUM_BASE_EXPERTS` | `0` | DeepSeek MoE disabled. |
| `DEEPSEEK_MOE_EXPERT_SEGMENTS` | `4` | Expert segmentation if enabled. |
| `DEEPSEEK_MOE_SHARED_EXPERTS` | `1` | Shared experts if enabled. |
| `DEEPSEEK_MOE_ACTIVE_EXPERTS` | `0` | Active routed experts. |
| `DEEPSEEK_MOE_BALANCE_ALPHA` | `0.0` | Balance loss weight. |
| `DEEPSEEK_MOE_NORM_TOPK_PROB` | `1` | Normalize top-k probabilities. |

PoE was strong in some prior Scylla work but has not been proven as a default
in the current SP1024 int6 rANS runner. DeepSeek/coda MoE paths are available
but should be treated as high-risk architecture experiments.

### Gradient Checkpointing

| Variable | Current default | Meaning |
| --- | --- | --- |
| `GRADIENT_CHECKPOINTING` | `0` | Disabled. |
| `ACTIVATION_CHECKPOINT_IMPL` | `none` | No activation checkpointing. |

Delayed parallel residual and attention residuals are not currently wired for
gradient checkpointing. Do not combine them casually.

## Eval-Only Mixers

These run after the model roundtrip path. They can produce additional logged
BPB variants, but the autoresearch primary parser still extracts the standard
final roundtrip metric. They can be very slow.

### Sliding Window

| Variable | Current default | Meaning |
| --- | --- | --- |
| `SLIDING_WINDOW_ENABLED` | `0` | Disabled. Required by PPM and LZP. |
| `SLIDING_COMPILE_LOGITS` | `0` | Optional compiled logits path. |
| `EVAL_STRIDE` | `64` | Sliding eval stride. |

Sliding evaluation can produce better eval-only scores, but previous runs took
far longer than the normal wrapper budget. Use a larger outer timeout if testing
it, and record it as eval-only unless the primary parser is changed.

### PPM

| Variable | Current default | Meaning |
| --- | --- | --- |
| `PPM_ENABLED` | `0` | Disabled. Requires sliding window. |
| `PPM_ORDER` | `5` | Byte-level PPM order. |
| `PPM_SUBSET_TOKENS` | `5000000` | Prefix token budget for PPM scoring. |
| `PPM_LAMBDA_HI` | `0.9` | High interpolation weight. |
| `PPM_LAMBDA_LO` | `0.05` | Low interpolation weight. |
| `PPM_CONF_THRESHOLD` | `0.9` | Confidence threshold for weight selection. |
| `PPM_ESCAPE_METHOD` | `d` | Escape method, must be `c` or `d`. |
| `PPM_USE_META_MIX` | `0` | Meta-mixer disabled by default. |
| `PPM_TOKEN_ORDER` | `3` | Token-order context for meta-mix. |
| `PPM_META_ALPHA` | `0.995` | Meta update decay. |
| `PPM_META_ETA` | `2.0` | Meta learning rate scale. |
| `PPM_META_WARMUP_BYTES` | `4096` | Meta warmup bytes. |

PPM needs tokenizer byte reconstruction support. For TokenMonster/Scylla this
requires the tokenmonster package.

### LZP

| Variable | Current default | Meaning |
| --- | --- | --- |
| `LZP_ENABLED` | `0` | Disabled. Requires sliding window. |
| `LZP_SUBSET_TOKENS` | `5000000` | Prefix token budget. |
| `LZP_ORDERS` | `4,5,6,8` | Context orders. |
| `LZP_TABLE_BITS` | `20` | Hash table size exponent. |
| `LZP_ALPHA_MIN` | `0.0` | Minimum mix weight. |
| `LZP_ALPHA_MAX` | `0.20` | Maximum mix weight. |
| `LZP_MIN_STREAK` | `1` | Minimum hit streak. |
| `LZP_MAX_STREAK` | `8` | Maximum hit streak. |
| `LZP_HIT_PROB` | `0.98` | Hit probability assigned by LZP. |

Known result: sliding plus LZP gave a much better eval-only BPB in one run, but
eval took about 2000 seconds and did not improve the primary metric.

### Hashed N-Gram Eval

| Variable | Current default | Meaning |
| --- | --- | --- |
| `NGRAM_EVAL_ORDER` | `0` | Disabled. Must be `0` or at least `2`. |
| `NGRAM_EVAL_MIN_ORDER` | `2` | Minimum order. |
| `NGRAM_EVAL_ALPHA` | `0.30` | Base mix weight. |
| `NGRAM_EVAL_ADAPTIVE` | `1` | Entropy-adaptive mix weight enabled. |
| `NGRAM_EVAL_ALPHA_MIN` | `0.05` | Adaptive minimum. |
| `NGRAM_EVAL_ALPHA_MAX` | `0.60` | Adaptive maximum. |
| `NGRAM_EVAL_ENTROPY_CENTER` | `4.0` | Entropy center. |
| `NGRAM_EVAL_ENTROPY_SCALE` | `2.0` | Entropy scale. |
| `NGRAM_EVAL_MIN_COUNT` | `2` | Minimum context count. |
| `NGRAM_EVAL_BUCKETS` | `4194304` | Count table buckets, power of two. |
| `NGRAM_EVAL_MAX_SECONDS` | `0.0` | No internal time cap. |
| `NGRAM_ENTROPY_SHIFT` | `0` | Entropy shift disabled. |
| `NGRAM_ORDER_MULTS` | empty | Optional comma-separated order multipliers. |
| `NGRAM_CUBRIC_CADENCE` | `0` | Must stay zero; nonzero is rejected. |
| `NGRAM_CHUNK_TOKENS` | `1048576` | Chunk size. |
| `NGRAM_BATCH_SEQS` | `128` | Batch sequences. |
| `NGRAM_MIX_MODE` | `expert` | Current code only accepts `expert`. |
| `NGRAM_EXPERT_TOPK` | `8` | Top-k experts. |
| `NGRAM_EXPERT_BOOST_SCALE` | `0.25` | Expert boost scale. |
| `NGRAM_EXPERT_MAX_BOOST` | `12.0` | Expert boost cap. |

Do not set `NGRAM_MIX_MODE=linear`; current code rejects it. Earlier notes about
linear mode are stale for this branch.

### TTT

| Variable | Current default | Meaning |
| --- | --- | --- |
| `TTT_ENABLED` | `0` | Disabled. |
| `TTT_LR` | `0.005` | SGD learning rate for test-time training. |
| `TTT_MOMENTUM` | `0.9` | SGD momentum. |
| `TTT_EPOCHS` | `3` | Per-chunk epochs. |
| `TTT_CHUNK_TOKENS` | `32768` | Validation chunk size. |
| `TTT_BATCH_SEQS` | `32` | Batch sequences. |
| `TTT_GRAD_CLIP` | `1.0` | TTT gradient clipping. |

TTT scores a validation chunk before training on it, then updates on that chunk
for future chunks. It can improve eval-only BPB but is too slow unless heavily
optimized.

## Available Tokenizers

| Tokenizer | Vocab | Data path | Tokenizer path | Byte override | Notes |
| --- | ---: | --- | --- | --- | --- |
| SP1024 | 1024 | `data/datasets/fineweb10B_sp1024/` | `data/tokenizers/fineweb_1024_bpe.model` | none | Current default and cleanest accounting. |
| SP1892 | 1892 | `data_sp1892/datasets/fineweb10B_sp1892/` | `data_sp1892/tokenizers/fineweb_1892_bpe.model` | none | Larger vocab; used in recent full-shape tests. |
| Scylla | 998 | `data_scylla/fineweb_scylla/` | TokenMonster path in Scylla setup | `151080363` in old notes | Needs byte-count/token byte care. |
| SP4096 | 4096 | `data_sp4096/...` | local SP4096 model exists | verify before use | Available locally but not the default harness. |
| SP8192 BPE | 8192 | record artifacts | record tokenizer artifacts | verify before use | Record artifacts exist; no active CaseOps tokenizer found. |

No active SP8192 CaseOps tokenizer with reversible case-control operators is
wired into `train_gpt_parcae.py` at this time.

## Recent Results To Preserve

These are reference points from `autoresearch.jsonl` and recent local runs.
Use matching protocol before comparing.

| Result | Metric | Notes |
| --- | ---: | --- |
| SP1024 GPTQ int8, parallel residual core, Muon 0.85 | `1.43469617` | Fresh HEAD baseline before int6 sweep; 1419 steps, 12.44MB. |
| SP1024 int6 rANS, 384d | `1.43625301` | Slightly worse BPB than GPTQ int8 but much smaller artifact, about 9.0MB. |
| SP1024 int6 rANS, 448d | `1.44806049` | Worse; slower steps erased parameter benefit. |
| Parallel residual core + GPTQ int8 | `1.42201464` | Helped versus previous GPTQ baseline in earlier segment. |
| Parallel residual all scopes | `1.42613223` | Worse than core-only. |
| Muon momentum 0.85 | `1.42051443` | Best in that earlier GPTQ segment. |
| Muon momentum 0.80 | `1.42388340` | Worse; 0.85 was local optimum. |
| XSA last 3 + GPTQ | `1.42577685` | Slightly worse and slower. |
| LZP sliding eval-only | primary `1.42563446`, sliding+LZP `1.37507` | Eval-only score was better, but eval took far too long. |
| SP1892 packed full-shape int6 rANS | `1.55520458` | 19.95M params, 14.25MB total submission. |
| SP1892 packed no-PWA prior | `1.41159149` | 23.59M params, 16.82MB total submission. |
| SP1892 PWA QK dense V | `1.55520458` | Saved parameters/bytes but quality was much worse. |

The current default now includes LeakyReLU squared MLP and QAT6/int6 rANS.
Treat its first clean run as a new baseline before making conclusions.

## Experiment Rules For Agents

1. Read this file every few runs.
2. Change one primary idea at a time unless testing an explicitly named combo.
3. Keep `MAX_WALLCLOCK_SECONDS`, tokenizer, quantization/export path, and seed
   fixed for comparable deltas.
4. If a run crashes or times out, record whether training finished and whether
   the failure was export/eval/parser/compile.
5. Prefer 300 second runs for architecture and optimizer features.
6. Use longer wrapper timeouts only for known slow eval-only features.
7. Do not report eval-only sliding/PPM/LZP/TTT numbers as the primary BPB unless
   the parser and objective were intentionally changed.
8. Watch artifact size. int6 rANS is useful mainly because it frees bytes for
   capacity, but capacity only helps if step count does not collapse.
9. Preserve tokenizer/evaluation accounting. Do not use byte-count overrides
   unless the tokenizer requires them and the value is verified.
10. Keep changes minimal and revert failed architecture changes unless they are
    useful scaffolding for the next planned test.

## Promising Next Tests

- Establish a clean current default run after the LeakyReLU squared MLP change.
- If LeakyReLU squared is worse, switch `MLP_CLASS_NAME` and
  `RECURRENT_MLP_CLASS_NAME` back to `BaseMLP` while keeping QAT6/int6 rANS.
- Try int6 rANS capacity increases smaller than 448d, such as 400d or 416d, if
  they keep step count close to 384d.
- Try `MLP_MULT=3.5` or `MLP_MULT=4.5` now that float multipliers parse.
- Try `PARALLEL_RESIDUAL_START` with immediate residual if testing layer-x
  onset; do not use delayed for starts.
- Try `POE_NUM_EXPERTS=3` only as a controlled architecture test.
- Re-test GPTQ int8 only if the objective changes back from int6 artifact size
  to raw BPB.

## Known Bad Or Risky Tests

- `PARALLEL_RESIDUAL_SCOPE=all`: worse than core-only in prior run.
- `MUON_MOMENTUM=0.80`: worse than `0.85`.
- `MUON_WD=0.04`: worse than current weight decay.
- `MODEL_DIM=448` with int6 rANS: slower and worse in one tested run.
- `NGRAM_MIX_MODE=linear`: rejected by current code.
- `LZP_ENABLED=1` or `PPM_ENABLED=1` without `SLIDING_WINDOW_ENABLED=1`: fail.
- `COMPILE_MUON_BACKEND=1`: not a proven win and makes CUDA fault attribution
  harder.
- SP8192 CaseOps assumptions: no active wired tokenizer exists in this repo.
