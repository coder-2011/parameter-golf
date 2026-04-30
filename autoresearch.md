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

- SP8192 CaseOps/lossless-caps data from the active experiment block.
- 300 second training wall-clock cap.
- QAT6 plus int6 rANS export for small artifacts.
- Parallel residual in the recurrent core only.
- Residual forget gate default-on.
- Sparse attention gate enabled; alternative attention gates are exposed but
  mutually exclusive with it.
- Muon momentum near `0.85`.
- LeakyReLU(0.5)^2 MLP with `MLP_MULT=4.0`.

The script now exposes all recent feature knobs directly in the experiment
block or the default section: residual forget, U-Net skips, delayed parallel
residual options, attention output/gated/sparse gates, fused softcapped CE,
fused QKV postprocess, document packing, GPTQ shuffled calibration, LQER, PWA
QKV controls, and Muon Newton-Schulz coefficient variants.

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
| `DATA_PATH` | `./caseops_sp8192/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | Training and validation shard directory from the active experiment block. |
| `TOKENIZER_PATH` | `./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` | SentencePiece tokenizer for the active CaseOps/SP8192 run. |
| `TOKENIZER_META_PATH` | empty | Optional metadata sidecar. |
| `TOKENIZER_META_VALIDATE` | `0` | Metadata validation disabled by default. |
| `VAL_BYTE_COUNT_OVERRIDE` | `0` | No byte-count override is applied by the runner. Verify before changing CaseOps accounting. |
| `VOCAB_SIZE` | `8192` | Must match tokenizer. |

The shell script still has fallback SP1024 defaults below the experiment block,
but the checked-in active experiment block overrides them to CaseOps/SP8192.
CaseOps tokenizers require extra care around byte reconstruction and byte
counts, so do not compare against SP1024 unless that is the explicit test.

### Shape

| Variable | Current default | Meaning |
| --- | --- | --- |
| `MODEL_DIM` | `384` | Outer embedding, prelude, and coda width. |
| `RECURRENT_DIM` | `384` | Recurrent-state width. |
| `RECURRENT_INTERMEDIATE_DIM` | `1536` | Recurrent MLP hidden width; active block sets this explicitly to `4 * RECURRENT_DIM`. |
| `NUM_HEADS` | `8` | Outer attention query heads. |
| `NUM_KV_HEADS` | `4` | Outer KV heads for GQA. |
| `RECURRENT_NUM_HEADS` | `8` | Core attention heads. |
| `N_LAYERS_IN_PRELUDE` | `1` | Blocks before recurrence. |
| `N_LAYERS_IN_RECURRENT_BLOCK` | `3` | Physical blocks inside the recurrent core. |
| `N_LAYERS_IN_CODA` | `3` | Blocks after recurrence. |
| `MEAN_RECURRENCE` | `2` | Forward recurrence depth. |
| `MEAN_BACKPROP_DEPTH` | `1` | Backprop depth through recurrence. |

Physical block indices for the default shape:

- prelude: block `0`
- recurrent core: blocks `1..3`
- coda: blocks `4..6`

`PARALLEL_RESIDUAL_START` uses these physical indices, not recurrence-expanded
depth.

### Training Length

| Variable | Current default | Meaning |
| --- | --- | --- |
| `TRAIN_SEQ_LEN` | `1024` | Training context length. |
| `TRAIN_BATCH_TOKENS` | `131072` | Tokens per optimizer step. |
| `GRAD_ACCUM_STEPS` | `8` | Gradient accumulation factor in the active experiment block. |
| `ITERATIONS` | `1000000` | High ceiling; wall-clock stops first. |
| `MAX_WALLCLOCK_SECONDS` | `300` | Training wall-clock cap. |
| `WARMDOWN_ITERS` | `1200` | Warmdown schedule length. |
| `WARMUP_STEPS` | `500` | Warmup steps before timed training. |
| `VAL_BATCH_SIZE` | `524288` | Validation batch tokens. |
| `EVAL_SEQ_LEN` | `1000` | Evaluation context length in the active experiment block. |
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

### Recent Feature Knobs Exposed In `autoresearch.sh`

The runner now has explicit defaults for these recently added knobs, so agents
can change them in one place without relying on hidden `train_gpt_parcae.py`
defaults.

| Variable | Current value | Notes |
| --- | --- | --- |
| `RESIDUAL_FORGET_GATE` | `1` | RG-LRU-style residual decay is default-on. |
| `RESIDUAL_FORGET_GATE_BIAS` | `-4.0` | Starts gate near identity. |
| `RESIDUAL_FORGET_MIN_RAD` / `MAX_RAD` | `0.99` / `0.999` | Valid range requires `0 < min <= max < 1`. |
| `UNET_SKIP_ENABLED` | `0` | Prelude-to-coda skip feature disabled unless explicitly tested. |
| `UNET_SKIP_INIT` | `1.0` | Skip scale initialization when enabled. |
| `PARALLEL_RESIDUAL_IMPL` | `immediate` | Alternative is `delayed` with constraints below. |
| `PARALLEL_RESIDUAL_RECORD_CONTROLS` | `1` | Enables per-branch control tensors in immediate mode. |
| `PARALLEL_RESIDUAL_TIED_NORM` | `0` | Delayed-mode-only option. |
| `PARALLEL_RESIDUAL_IN_FP32` | `0` | Delayed-mode-only option. |
| `ATTN_OUT_GATE_ENABLED` | `0` | Mutually exclusive with `GATED_ATTN` and `SPARSE_ATTN_GATE`. |
| `ATTN_OUT_GATE_SRC` | `proj` | Must be `proj` or `q`. |
| `GATED_ATTN_ENABLED` | `0` | Mutually exclusive with the other attention gates. |
| `GATED_ATTN_INIT_STD` | `0.01` | Non-negative init std. |
| `SPARSE_ATTN_GATE` | `1` | Active attention gate in the checked-in experiment block. |
| `FUSED_SOFTCAPPED_CE` | `0` | Requires Triton; off until benchmarked against Liger CE here. |
| `FUSED_QKV_POSTPROCESS` | `1` | Active in the experiment block; verify throughput on the active path. |
| `DOCUMENT_PACKING` | `0` | Requires `flash_attn_interface.flash_attn_varlen_func`. |
| `DOCUMENT_PACKING_CU_BUCKET_SIZE` | `64` | Positive bucket size required. |
| `GPTQ_CALIBRATION_MODE` | `stream` | Alternative `shuffled` is available for GPTQ experiments. |
| `LQER_ENABLED` | `0` | Low-rank quantization-error repair disabled by default. |
| `LQER_RANK` / `TOP_K` | `4` / `3` | Must be non-negative. |
| `LQER_FACTOR_BITS` | `4` | Must be in `[2, 8]`. |
| `LQER_ASYM_ENABLED` / `GROUP` | `1` / `64` | Group must be positive and divide factor tensor size. |
| `MUON_NS_COEFFS` | `classic` | `polar_express` is available for an optimizer ablation. |
| `RRHP_REDUCTION_FACTOR` | `4` | Used when `ATTN_QKV_MODE=rrhp`. |
| `PWA_NUM_BASES` / `PERMUTE_SIZE` | `6` / `4` | Used by PWA QKV modes. |
| `PWA_INIT_SCALE` | `0.7071067811865475` | PWA basis init scale. |
| `PWA_LR` | `MATRIX_LR` | Optional PWA-specific LR. |

### Feature Glossary For Autoresearch Agents

Use this section to understand what each experimental family is changing before
editing the shell script. For comparable runs, change one family at a time.

| Feature family | What it is | Primary knobs | Main risk |
| --- | --- | --- | --- |
| Residual forget gate | Per-block learned residual decay. Each residual stream is multiplied by `exp(-8 * sigmoid(gate(normed_x)) * softplus(a_param))` before adding the attention or MLP branch. It is inspired by RG-LRU decay but used on transformer residuals, not a recurrent hidden scan. | `RESIDUAL_FORGET_GATE`, `RESIDUAL_FORGET_GATE_BIAS`, `RESIDUAL_FORGET_MIN_RAD`, `RESIDUAL_FORGET_MAX_RAD` | Too much decay can erase useful state and hurt early training. Disable with `RESIDUAL_FORGET_GATE=0` for a clean ablation. |
| U-Net skips | Adds learned skip connections from prelude activations into matching coda blocks. This can help shallow features bypass the recurrent core. | `UNET_SKIP_ENABLED`, `UNET_SKIP_INIT` | Changes block execution path and parameter set; test separately from residual changes. |
| Parallel residual | Computes attention and MLP branches from the same normalized input and combines them, instead of strictly feeding attention output into MLP. Core-only mode changes only recurrent core blocks. | `RESIDUAL_MODE`, `PARALLEL_RESIDUAL_SCOPE`, `PARALLEL_RESIDUAL_START`, `PARALLEL_RESIDUAL_IMPL`, `PARALLEL_RESIDUAL_RECORD_CONTROLS` | Scope and implementation change semantics; `all` was worse before. |
| Delayed parallel residual | Streaming implementation of parallel residual that carries one branch forward to reduce immediate branch coupling. | `PARALLEL_RESIDUAL_IMPL=delayed`, `PARALLEL_RESIDUAL_TIED_NORM`, `PARALLEL_RESIDUAL_IN_FP32` | Only valid with `START=-1`, `RECORD_CONTROLS=0`, and no gradient checkpointing. |
| Attention output gate | Learned gate applied after attention output projection, initialized to be transparent. `proj` gates from projected attention output; `q` gates from query-side activations. | `ATTN_OUT_GATE_ENABLED`, `ATTN_OUT_GATE_SRC` | Mutually exclusive with sparse/gated attention. Extra gate can slow training or over-control attention. |
| Gated attention | Learned multiplicative gate on attention output, initialized from `GATED_ATTN_INIT_STD`. | `GATED_ATTN_ENABLED`, `GATED_ATTN_INIT_STD` | Mutually exclusive with other attention gates; zero/std choices change initial scale. |
| Sparse attention gate | A learned/local gate that suppresses attention output using a configured window. It is the currently active attention-gate experiment. | `SPARSE_ATTN_GATE`, `SPARSE_ATTN_GATE_WINDOW`, `SPARSE_ATTN_GATE_INIT_STD`, `SPARSE_ATTN_GATE_SCALE` | Can reduce useful long-range attention; compare against `SPARSE_ATTN_GATE=0` first. |
| Attention residuals | Separate residual connections around attention groups or blocks. They are different from parallel residual and currently require sequential residual mode. | `ATTN_RES_MODE`, `ATTN_RES_SCOPE`, `ATTN_RES_BLOCK_SIZE` | Invalid with `RESIDUAL_MODE=parallel`; not wired for checkpointing. |
| XSA | Extra attention behavior applied to the last `N` effective layers. | `XSA_LAST_N` | Prior runs were slower/marginal, so keep it isolated. |
| Fused QKV postprocess | Triton kernel for Q/K RMSNorm/RoPE postprocessing after packed QKV projection. Intended to reduce Python/Tensor overhead in attention setup. | `FUSED_QKV_POSTPROCESS`, `LIGER_ROPE`, `ROPE_DIMS` | Prior local note found it slower than the Liger path in one active shape. Verify on the exact shape. |
| Fused softcapped CE | Custom Triton cross entropy that includes the logit softcap transform in the CE path. | `FUSED_SOFTCAPPED_CE`, `LOGIT_SOFTCAP` | Requires Triton and CUDA support; compare against Liger fused CE. |
| Liger CE/RoPE | External optimized kernels for CE and RoPE where available. | `LIGER_CE`, `LIGER_FUSED_CE`, `LIGER_ROPE` | Can graph-break around compile; benchmark actual end-to-end step rate. |
| Document packing | Packs multiple documents into varlen attention batches using cumulative sequence lengths. This reduces padding but changes loader behavior. | `DOCUMENT_PACKING`, `DOCUMENT_PACKING_CU_BUCKET_SIZE` | Requires `flash_attn_interface.flash_attn_varlen_func`; compare only against same data/tokenizer. |
| QAT | Fake quantization during training so weights adapt to low-bit export. | `QAT_BITS`, `QAT_START_STEP`, `QAT_GROUP_SIZE`, `QAT_ACTIVATION_BITS`, `QAT_LINEAR`, `QAT_EMBEDDINGS`, `QAT_TIED_OUTPUT` | Does not shrink artifacts by itself; export `QUANT_BITS` must also match. |
| int6 rANS export | Entropy-coded six-bit artifact path. Keeps artifact small enough to spend bytes elsewhere. | `QUANT_BITS=6`, `RANS_INT6=1`, `GROUPED_ARTIFACT` | Requires `QUANT_BITS=6`; quality can drop if model capacity/step count are not balanced. |
| GPTQ export | Post-training Hessian/blockwise quantization path. | `GPTQ_ENABLED`, `GPTQ_CALIBRATION_BATCHES`, `GPTQ_CALIBRATION_MODE`, `GPTQ_BLOCKSIZE`, `GPTQ_DAMPENING`, `GPTQ_ACT_ORDER` | Changes export/eval path. Do not compare GPTQ and non-GPTQ as architecture-only deltas. |
| LQER | Low-rank quantization-error repair applied around quantized tensors. | `LQER_ENABLED`, `LQER_RANK`, `LQER_TOP_K`, `LQER_FACTOR_BITS`, `LQER_ASYM_ENABLED`, `LQER_ASYM_GROUP` | Adds repair factors/bytes and has divisibility constraints. |
| Mixed quant bits | Uses different bit widths for embeddings, attention, MLP, low-bit patterns, and control tensors. | `MIXED_QUANT_BITS`, `QUANT_EMBED_BITS`, `QUANT_ATTN_BITS`, `QUANT_MLP_BITS`, `QUANT_LOW_BITS`, `QUANT_CONTROL_BITS`, `QUANT_LOW_BIT_PATTERNS` | Artifact size and quality move together; record exact bit map. |
| PWA/RRHP QKV | Parameter-saving attention projection variants. RRHP reduces rank/head projection; PWA uses shared permutation-weighted bases. | `ATTN_QKV_MODE`, `RRHP_REDUCTION_FACTOR`, `PWA_NUM_BASES`, `PWA_PERMUTE_SIZE`, `PWA_INIT_SCALE`, `PWA_LR` | Prior PWA QK dense V saved bytes but had a large BPB regression. |
| Muon NS coefficients | Changes Newton-Schulz orthogonalization coefficients inside Muon updates. | `MUON_NS_COEFFS`, `MUON_BACKEND_STEPS`, `COMPILE_MUON_BACKEND` | Optimizer-only ablation; async CUDA failures are harder with compiled Muon backend. |
| Bigram hash | Token bigram feature embedding added to the model input path. | `BIGRAM_HASH_BUCKETS`, `BIGRAM_HASH_DIM`, `BIGRAM_HASH_HEADS`, `BIGRAM_HASH_GATE`, `BIGRAM_HASH_SCALE_INIT`, `BIGRAM_HASH_INIT_STD` | Larger tables cost bytes and may overfit tokenizer-specific statistics. |
| Eval-only mixers | Post-roundtrip validation-time scoring aids. They can improve reported auxiliary BPB but are slow and not the primary parser target. | `SLIDING_WINDOW_ENABLED`, `PPM_ENABLED`, `LZP_ENABLED`, `NGRAM_EVAL_ORDER`, `TTT_ENABLED` | Do not report as primary unless the objective/parser is intentionally changed. |

### Current Architecture Defaults

| Variable | Current default | Meaning |
| --- | --- | --- |
| `MLP_MULT` | `4` | MLP hidden multiplier used when intermediate dim is not explicitly set. |
| `MLP_CLASS_NAME` | `FusedLeakyReLUSqMLP` | Outer/prelude/coda MLP class. |
| `RECURRENT_MLP_CLASS_NAME` | `FusedLeakyReLUSqMLP` | Recurrent core MLP class. |
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
- `LeakyReluSquaredMLP` / `FusedLeakyReLUSqMLP`: dense up projection, `LeakyReLU(0.5)^2`, dense down
  projection, zero-initialized down projection.
- `GatedMLP`: fused dense projection split into GELU gate and value.
- `LigerStyleGatedMLP` / `LigerGEGLUMLP`: separate gate/up/down projections
  with Liger GELU multiply when available.

The current runner intentionally tests the fused LeakyReLU-squared MLP. Older Parcae
5-minute ablations showed LeakyReLU squared was worse in a different protocol,
so treat this as an active experiment, not a proven permanent default.

### Attention And QKV Projection

| Variable | Runner default | Meaning |
| --- | --- | --- |
| `ATTN_QKV_MODE` | `packed` | QKV projection layout. |
| `RRHP_REDUCTION_FACTOR` | `4` | Reduction factor if `ATTN_QKV_MODE=rrhp`. |
| `PWA_NUM_BASES` | `6` | Number of shared PWA bases if PWA mode is enabled. |
| `PWA_PERMUTE_SIZE` | `4` | Permutation tuple size for PWA indexing. |
| `PWA_INIT_SCALE` | `0.7071067811865475` | PWA basis init scale. |
| `PWA_LR` | `MATRIX_LR` | Optional PWA-specific learning rate. |
| `ATTN_PRECONV_KERNEL` | script default `0` | Depthwise pre-attention conv disabled by default. |
| `ATTENTION_WINDOW` | script default `-1` | Full attention unless overridden. |
| `TRIDAO_PACKED_ROPE` | `0` | Custom packed RoPE path disabled in autoresearch. |
| `LIGER_ROPE` | `1` | Liger RoPE enabled in the active experiment block. |

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
| `BIGRAM_HASH_HEADS` | `4` | Number of hash heads. |
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
PARALLEL_RESIDUAL_IMPL=immediate
PARALLEL_RESIDUAL_RECORD_CONTROLS=1
```

Meaning:

- Only recurrent core blocks use parallel residual.
- Prelude and coda stay sequential.
- `START=-1` means all blocks in the selected scope.
- With the default physical indices, `core` means blocks `1..3`.

`PARALLEL_RESIDUAL_START=N` works only with `PARALLEL_RESIDUAL_IMPL=immediate`.
The delayed path currently requires `PARALLEL_RESIDUAL_START=-1`,
`PARALLEL_RESIDUAL_RECORD_CONTROLS=0`, and no gradient checkpointing.
`PARALLEL_RESIDUAL_TIED_NORM=1` and `PARALLEL_RESIDUAL_IN_FP32=1` are delayed
mode options only.

Residual forget gate is independent of immediate/delayed residual mode and is
currently default-on. To measure its impact cleanly, run one matched baseline
with only `RESIDUAL_FORGET_GATE=0` changed.

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
| `ATTN_OUT_GATE_ENABLED` | `0` | Output gate disabled. |
| `ATTN_OUT_GATE_SRC` | `proj` | Gate source when output gate is enabled. |
| `GATED_ATTN_ENABLED` | `0` | Multiplicative attention gate disabled. |
| `GATED_ATTN_INIT_STD` | `0.01` | Init std if gated attention is enabled. |
| `SPARSE_ATTN_GATE` | `1` | Sparse attention gate enabled in the active experiment block. |
| `SPARSE_ATTN_GATE_WINDOW` | `12` | Window size for sparse gate. |

Constraints:

- Attention residuals require `RESIDUAL_MODE=sequential`.
- Attention residuals are not wired for gradient checkpointing.
- `ATTN_OUT_GATE_ENABLED`, `GATED_ATTN_ENABLED`, and `SPARSE_ATTN_GATE` are
  mutually exclusive; exactly one or zero should be enabled.
- XSA is available and controlled by `XSA_LAST_N`, but tested values have been
  marginal/negative and slower in the current SP1024 family.

### Optimizer

| Variable | Current default | Meaning |
| --- | --- | --- |
| `EMBED_LR` | `0.12` | Embedding LR. |
| `HEAD_LR` | `0.008` | Head LR. |
| `TIED_EMBED_LR` | `0.03` | Tied embedding LR. |
| `MATRIX_LR` | `0.04` | Matrix LR. |
| `SCALAR_LR` | `0.04` | Scalar/control LR. |
| `MUON_MOMENTUM` | forced to `0.85` | Muon momentum. |
| `MUON_BACKEND_STEPS` | `5` | Newton-Schulz steps. |
| `MUON_NS_COEFFS` | `classic` | Coefficient family for Newton-Schulz; try `polar_express` only as a controlled ablation. |
| `MUON_MOMENTUM_WARMUP_START` | `0.85` | Momentum warmup start. |
| `MUON_MOMENTUM_WARMUP_STEPS` | `500` | Momentum warmup steps. |
| `MUON_ROW_NORMALIZE` | `1` | Row normalization enabled. |
| `MUON_WD` | `0.105` | Muon weight decay. |
| `BETA1` | `0.9` | Adam beta1. |
| `BETA2` | `0.99` | Adam beta2. |
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
| `LIGER_FUSED_CE` | `1` | Use fused Liger linear CE when the path supports it. |
| `FUSED_SOFTCAPPED_CE` | `0` | Custom Triton softcapped CE disabled by default. |
| `FUSED_QKV_POSTPROCESS` | `1` | Active in the experiment block. |
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
| `SWA_START_STEP` | `800` | Start SWA at this optimizer step. |
| `SWA_EVERY` | `50` | SWA update cadence. |
| `SWA_DYNAMIC` | `1` | Dynamic SWA cadence/weighting enabled. |
| `SWA_DYNAMIC_MIN_EVERY` | `1` | Minimum dynamic SWA cadence. |
| `SWA_DYNAMIC_POWER` | `1.0` | Dynamic weighting exponent. |
| `SWA_DYNAMIC_WEIGHT_MAX` | `2.0` | Maximum dynamic SWA weight. |

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
| `QAT_EMBEDDINGS` | `1` | Apply QAT preparation to embedding modules where supported. |
| `QAT_TIED_OUTPUT` | `1` | Include tied output path if applicable. |
| `QUANT_BITS` | `6` | Export bit width. |
| `MIXED_QUANT_BITS` | `0` | Single global quant bit width; mixed per-module bits disabled. |
| `QUANT_EMBED_BITS` | `7` | Embedding bit width if mixed quant is enabled. |
| `QUANT_ATTN_BITS` | `6` | Attention tensor bit width if mixed quant is enabled. |
| `QUANT_MLP_BITS` | `6` | MLP tensor bit width if mixed quant is enabled. |
| `QUANT_LOW_BITS` | `4` | Low-bit override width for matching patterns if mixed quant is enabled. |
| `QUANT_CONTROL_BITS` | `0` | Control tensors stay float when `0`; `2..8` integer-quantizes them. |
| `QUANT_LOW_BIT_PATTERNS` | `project_out,poe_heads` | Name substrings receiving `QUANT_LOW_BITS` under mixed quant. |
| `QUANT_KEEP_FLOAT_PATTERNS` | empty | Optional comma-separated name patterns to keep in float during export. |
| `RANS_INT6` | `1` | rANS entropy coding for int6 symbols. |
| `GPTQ_ENABLED` | `0` | GPTQ disabled by force. |
| `GPTQ_CALIBRATION_BATCHES` | `32` | Number of batches used to collect GPTQ activation/Hessian statistics. |
| `GPTQ_CALIBRATION_MODE` | `stream` | Calibration data order; `shuffled` is available for a randomized calibration sample. |
| `GPTQ_RESERVE_SECONDS` | `12` | Time held back from wall-clock budget for GPTQ/export/eval. |
| `GPTQ_BLOCKSIZE` | `128` | GPTQ block size. |
| `GPTQ_DAMPENING` | `0.01` | Hessian dampening factor. |
| `GPTQ_MIN_NUMEL` | `65536` | Minimum tensor size for GPTQ handling. |
| `GPTQ_ACT_ORDER` | `1` | Activation-order GPTQ heuristic enabled. |
| `GPTQ_QUANTIZE_EMBEDDINGS` | `1` | Include embeddings in GPTQ when GPTQ is enabled. |
| `GPTQ_MATRIX_CLIP_SIGMAS` | `12.85` | SDClip-style row clipping if GPTQ is enabled. |
| `GPTQ_EMBED_CLIP_SIGMAS` | `20.0` | Embedding row clipping if GPTQ is enabled. |
| `LQER_ENABLED` | `0` | Low-rank quantization-error repair disabled. |
| `LQER_RANK` | `4` | Rank of each repair factor when LQER is enabled. |
| `LQER_TOP_K` | `3` | Number of largest/most important tensors to repair. |
| `LQER_FACTOR_BITS` | `4` | Bit width for LQER repair factors. |
| `LQER_ASYM_ENABLED` | `1` | Use asymmetric factor quantization. |
| `LQER_ASYM_GROUP` | `64` | Asymmetric quantization group size for repair factors. |

Important distinctions:

- QAT alone does not shrink the artifact unless `QUANT_BITS` is also lowered.
- GPTQ has SDClip-style per-row sigma clipping through the `GPTQ_*_CLIP_SIGMAS`
  knobs.
- Plain int6/RANS export uses the normal int export clipping path, not GPTQ's
  Hessian/blockwise quantization.
- `RANS_INT6=1` requires `QUANT_BITS=6`.
- `GPTQ_CALIBRATION_MODE` must be `stream` or `shuffled`.
- `LQER_FACTOR_BITS` must be in `[2, 8]`; `LQER_RANK` and `LQER_TOP_K` must be
  non-negative.
- `LQER_ASYM_GROUP` must be positive and divide the relevant repair-factor
  tensor size.

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
in this runner family. DeepSeek/coda MoE paths are available
but should be treated as high-risk architecture experiments.

### Gradient Checkpointing

| Variable | Current default | Meaning |
| --- | --- | --- |
| `GRADIENT_CHECKPOINTING` | `0` | Disabled. |
| `ACTIVATION_CHECKPOINT_IMPL` | `none` | No activation checkpointing. |

Delayed parallel residual and attention residuals are not currently wired for
gradient checkpointing. Do not combine them casually.

Document packing is also exposed through `DOCUMENT_PACKING=1` and
`DOCUMENT_PACKING_CU_BUCKET_SIZE`. It changes batch construction from fixed
length token windows to varlen packed documents and requires varlen
FlashAttention support, so treat it as a data-loader experiment rather than
mixing it into unrelated architecture tests.

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
| `TTT_ENABLED` | `0` | Disabled in the active experiment block. |
| `TTT_LR` | `0.005` | SGD learning rate for test-time training. |
| `TTT_MOMENTUM` | `0.9` | SGD momentum. |
| `TTT_EPOCHS` | `3` | Per-chunk epochs. |
| `TTT_CHUNK_TOKENS` | `32768` | Validation chunk size. |
| `TTT_BATCH_SEQS` | `32` | Batch sequences. |
| `TTT_GRAD_CLIP` | `1.0` | TTT gradient clipping. |
| `TTT_MASK` | `all` | Default mask; older no-Q/V experiments used `no_qv`. |

TTT scores a validation chunk before training on it, then updates on that chunk
for future chunks. It can improve eval-only BPB but is too slow unless heavily
optimized.

## Available Tokenizers

| Tokenizer | Vocab | Data path | Tokenizer path | Byte override | Notes |
| --- | ---: | --- | --- | --- | --- |
| SP1024 | 1024 | `data/datasets/fineweb10B_sp1024/` | `data/tokenizers/fineweb_1024_bpe.model` | none | Fallback defaults and clean accounting baseline. |
| SP1892 | 1892 | `data_sp1892/datasets/fineweb10B_sp1892/` | `data_sp1892/tokenizers/fineweb_1892_bpe.model` | none | Larger vocab; used in recent full-shape tests. |
| Scylla | 998 | `data_scylla/fineweb_scylla/` | TokenMonster path in Scylla setup | `151080363` in old notes | Needs byte-count/token byte care. |
| SP4096 | 4096 | `data_sp4096/...` | local SP4096 model exists | verify before use | Available locally but not the default harness. |
| SP8192 CaseOps | 8192 | `caseops_sp8192/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/` | `data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` | verify before changing accounting | Active experiment block. |

The active runner now uses the SP8192 CaseOps row. Older notes saying no active
CaseOps tokenizer was wired are stale for this branch.

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
- Run a matched residual-forget ablation with only `RESIDUAL_FORGET_GATE=0`.
- Try `RESIDUAL_FORGET_GATE_BIAS=-5.0` or `-6.0` if the default gate hurts
  early optimization; try `-3.0` only if the default is positive.
- Compare `SPARSE_ATTN_GATE=0` against the current sparse-gate run before
  testing `ATTN_OUT_GATE_ENABLED=1` or `GATED_ATTN_ENABLED=1`.
- Test `MUON_NS_COEFFS=polar_express` as a single optimizer ablation.
- Test `GPTQ_CALIBRATION_MODE=shuffled` only when `GPTQ_ENABLED=1`.
- Test `DOCUMENT_PACKING=1` only after confirming varlen FlashAttention imports
  in the active environment.
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
- Combining attention gates: `ATTN_OUT_GATE_ENABLED`, `GATED_ATTN_ENABLED`, and
  `SPARSE_ATTN_GATE` are mutually exclusive.
- Delayed parallel residual with gradient checkpointing or
  `PARALLEL_RESIDUAL_RECORD_CONTROLS=1`: rejected by current validation.
