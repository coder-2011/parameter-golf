# Codex Learning Notes

## 2026-04-26 RWKV-LM-V7 Blackwell CUDA compiler

RWKV-LM-V7 native extension builds require an `nvcc` that knows Blackwell `compute_120`. The initial environment had PyTorch `2.11.0+cu128` and an RTX PRO 4500 Blackwell GPU, but `/usr/local/cuda` pointed at CUDA 12.4 whose `nvcc --list-gpu-arch` stopped at `compute_90`. That caused RWKV/DeepSpeed native compilation failures unless forced through `TORCH_CUDA_ARCH_LIST=9.0+PTX`.

Installed `cuda-nvcc-12-8` from the NVIDIA apt repo, which moved `/usr/local/cuda` to `/usr/local/cuda-12.8`. After that, `nvcc --list-gpu-arch` includes `compute_120`; a fresh RWKV `wind_backstepping` extension build compiled `sm_120` directly and the FineWeb smoke script ran without the PTX workaround. The pip `nvidia-cuda-nvcc-cu12` package was not sufficient by itself here because it installed `ptxas`/NVVM files but no `nvcc` executable.

DeepSpeed FusedAdam additionally needs CUDA library dev headers such as `cusparse.h`, not just `nvcc`. Installing `cuda-libraries-dev-12-8` supplied cuSPARSE/cuBLAS/cuSOLVER/etc. headers and libs under `/usr/local/cuda-12.8`; a forced fresh `FusedAdamBuilder().load()` then compiled successfully for `sm_120`.

RWKV-LM-V7 `--compile 1` plus `deepspeed_stage_2` originally failed during backward with `CompiledFunctionBackward returned an invalid gradient at index 4 - got [512] but expected [1, 1, 512]`. The root cause was semantic channel-vector mix parameters being registered with fake broadcast shape `[1, 1, C]`; DeepSpeed/AOTAutograd returned the natural `[C]` gradient. Fix both attention time-mix parameters and FFN `x_k` to true vectors `[C]`, while keeping checkpoint load reshape compatibility for old `[1,1,C]` checkpoints. After that, the default 8-layer/512-dim FineWeb run with `COMPILE=1 STRATEGY=deepspeed_stage_2 M_BSZ=16` completed 3,052 steps and saved `out/fineweb-10min-L8-D512-x070/rwkv-final.pth`; it stopped on the 50M-token cap before the 600s wall-clock cap.

## 2026-04-25 Scylla / Parcae / PoE session

Scylla support is now wired through `train_gpt_parcae.py` and `scripts/run_parcae_scylla_current_best.sh`.

Key implementation facts:

| item | note |
|---|---|
| tokenizer kind | Scylla uses TokenMonster via `candidate.vocab`, not SentencePiece |
| metadata | `candidate.meta.npz` has per-token bytes but not a sufficient BPB denominator by itself |
| BPB denominator | use `VAL_BYTE_COUNT_OVERRIDE=151080363` for the current seq512 Scylla setup |
| current best | `POE_NUM_EXPERTS=3`, `POE_HEAD_LR=0.002` |
| current best exact BPB | `1.69388607` |

Important result summary:

| variant | BPB |
|---|---:|
| PoE3 lr `0.002` | `1.69388607` |
| dense Scylla baseline corrected | about `1.69637493` |
| XSA4 corrected | about `1.70250566` |
| coda MoE4 top-1 | `1.71156502` |

Implementation caveats:

| caveat | consequence |
|---|---|
| concurrent training runs share `final_model*` filenames | validation lines are usable, but clean official artifact claims should use solo reruns |
| Scylla metadata byte count over-counts decoded bytes | always use the override for comparable BPB |
| PoE LR/count response is sharp | avoid assuming more experts or nearby LR values improve |
| XSA implementation checks passed | XSA likely hurts due removing useful recurrent value-aligned state, not due a basic implementation bug |

Next best clean check: rerun `POE_NUM_EXPERTS=3 POE_HEAD_LR=0.002` alone with Scylla/current-best config and current byte override.

## 2026-04-26 SwiGLU recurrent input injection

`train_gpt_parcae.py` has an opt-in `INJECTION_TYPE=swiglu-add` path. It adds `INJECTION_SWIGLU_SCALE`-scaled `silu(gate) * value` from the original input embeddings into the recurrent state before the core blocks. `INJECTION_SWIGLU_SCALE` defaults to `0.0`, so the new injection contribution is off unless a run explicitly enables a nonzero scale. This is intended as a direct test of nonlinear token/BigramHash reinjection quality without changing the transformer block residuals.

Matched 300s Scylla comparison on the current local `data_scylla/fineweb_scylla` files, with TTT off and `POE_NUM_EXPERTS=3 POE_HEAD_LR=0.002`. These two logs accidentally omitted `VAL_BYTE_COUNT_OVERRIDE`, so the logged BPB used TokenMonster metadata-summed bytes. Corrected BPB below uses the intended `151080363` denominator from each exact loss.

| run | injection | corrected BPB | logged BPB | exact loss | steps | note |
|---|---|---:|---:|---:|---:|---|
| `logs/exp_scylla_poe3_diag_scale_compare_300_rerun.txt` | `diagonal` | `1.73644542` | `1.63007385` | `2.91578821` | 1991 | baseline on current local Scylla files |
| `logs/exp_scylla_poe3_swiglu_add_scale0p1_compare_300.txt` | `swiglu-add`, scale `0.1` | `1.79727366` | `1.68717587` | `3.01792921` | 2017 | worse despite slightly faster step time |

Conclusion: full additive SwiGLU injection at scale `0.1` is too disruptive in this form. If revisited, try a much smaller scale (`0.01` or learned zero-init with warmup) or a modulation of the existing diagonal input term rather than an additional raw residual injection.

## 2026-04-26 Parcae score-first TTT implementation

`train_gpt_parcae.py` has an opt-in `TTT_ENABLED=1` quantized roundtrip eval path that scores validation chunks first with sliding windows, then trains on the scored chunk via SGD. Defaults: `TTT_CHUNK_TOKENS=32768`, `TTT_EPOCHS=3`, `TTT_GRAD_CLIP=1.0`, `EVAL_STRIDE=64`.

Key implementation facts:

| item | note |
|---|---|
| placement | runs only after quantized roundtrip load; it does not alter saved artifacts |
| legality invariant | each chunk is scored before that chunk is used for SGD |
| sliding scoring | windows score only the first window from `0`, then the `EVAL_STRIDE` tail after a context prefix |
| distributed path | ranks split windows/sequences and manually all-reduce active gradients before clipping |
| metric path | standard validation and TTT share byte accounting and `VAL_BYTE_COUNT_OVERRIDE` handling |
| final chunk | skipped for training because no later validation tokens can benefit |
| logits compile | intentionally uncompiled for robustness after synthetic smoke showed compile startup overhead |

Focused checks run:

- `python -m py_compile train_gpt_parcae.py`
- Synthetic CUDA one-epoch TTT smoke passed.
- Synthetic `TTT_EPOCHS=0` equivalence check matched a direct bf16 sliding-window reference exactly (`diff=0` for loss and BPB).

Refactor caveat: cleanup extracted `_validation_result`, `_rank_bounds`, `_token_byte_sum`, `_ttt_chunk_windows`, and `_window_batch`. This reduced repeated logic and made metric handling safer, but it did not shrink line count. If minimizing code size becomes the priority, `_window_batch` is the least essential helper.

Performance update: `runs/exp_scylla_poe3_diag_ttt_compare_300.console.log` completed a local score-first TTT pass with `VAL_BYTE_COUNT_OVERRIDE`, improving roundtrip BPB from `1.73147091` to `1.68571400`, but the TTT eval pass took `682647ms`. Treat TTT as promising but too slow for a competition-ready default until optimized.

## 2026-04-26 diagnostic and architecture triage

Current belief from checkpoint diagnostics: no token-collapse degeneracy was found, but recurrent state scale is pathological. Trained recurrent norms reached roughly `85-622` while final hidden norms were normalized to roughly `1.1-1.5`; recurrent relative residual stayed around `0.39-0.42`, so the loop is active rather than a no-op. Output matrices are full rank with `r99` around `229-235`, so the stronger bottleneck is hidden/latent utilization and recurrent scale management rather than output-rank collapse.

Architecture triage:

| idea | note |
|---|---|
| MoE | legacy coda top-1 MoE lost; DeepSeek-style coda MoE is now available default-off for router-diagnostic tests, but should not be treated as proven without a matched run |
| MTP | auxiliary only under current evaluator; BPB path scores shifted next-token CE |
| TensorSLM/LBLLM | artifact-budget tools, not direct BPB improvements unless offline reconstruction/distillation preserves BPB |
| Gemma 4 PLE | promising as small zero-init PLE-lite/token-conditioning, safer than raw additive reinjection |
| Kimi/Gated Delta | possible later recurrent-core test after live latent diagnostics |

## 2026-04-26 PLE-lite implementation

`train_gpt_parcae.py` now has default-off PLE-lite token conditioning. Use `PLE_SCOPE=prelude|core|coda|all`, `PLE_DIM>0`, `PLE_SCALE_INIT=0.0`, and `PLE_INIT_STD=0.02`. Each selected physical layer gets a small token lookup, a projection to the stream width, and a scalar gate injected before that block. PLE modules are constructed after existing model initialization, so zero-scale enabled PLE preserved common params and gave `max_logit_diff=0.0` in a tiny deterministic check. First recommended run is coda-only with `PLE_DIM=32` on the Scylla PoE3/bigram8192 baseline.

## 2026-04-26 Validation scorer alignment

`train_gpt_parcae.py::eval_val` was changed back to the same main scorer body as canonical `train_gpt.py::eval_val`: same sequence partitioning, contiguous next-token windows, bf16 autocast model loss, token-byte LUT calculation, distributed all-reduce order, and BPB formula. The only intentional extra line is `VAL_BYTE_COUNT_OVERRIDE`, applied after byte-count all-reduce for Scylla denominator control. An AST body comparison passed with that override line ignored.

## 2026-04-26 Parallel Residual Switch

`train_gpt_parcae.py` now has default-off parallel residual support. Use `RESIDUAL_MODE=parallel PARALLEL_RESIDUAL_SCOPE=core` to make recurrent core blocks compute `x + attn(norm_1(x)) + mlp(norm_2(x))` while leaving prelude/coda sequential. `PARALLEL_RESIDUAL_SCOPE=all` also applies it to prelude/coda. Defaults are `RESIDUAL_MODE=sequential` and `PARALLEL_RESIDUAL_SCOPE=none`, preserving current behavior.

Follow-up correction: this was rewritten to match `curr_record_sub.py` more faithfully when enabled. Parallel mode now creates record-style per-channel `attn_scale`, `mlp_scale`, and `resid_mix` controls in the selected scope, uses `x_in = resid_mix[0] * x + resid_mix[1] * x0`, applies optional `1/sqrt(physical_layer_idx + 1)` normalized-input scaling, and supports `PARALLEL_RESIDUAL_START` as a physical block-index gate. In Parcae, `x0` is stream-local: outer input embeddings for prelude/coda and the initialized recurrent state for the recurrent core. Default sequential mode still does not create these extra tensors.

## 2026-04-26 DeepSeekMoE coda implementation

`train_gpt_parcae.py` has a default-off DeepSeek-style coda MoE path integrated from the `deepseek-moe-coda` worktree. It replaces coda MLPs only when `DEEPSEEK_MOE_NUM_BASE_EXPERTS > 0`; otherwise the dense coda remains unchanged unless the older `CODA_MOE_NUM_EXPERTS` path is enabled.

Implementation scope:

| item | note |
|---|---|
| placement | coda MLP replacement only |
| controls | `DEEPSEEK_MOE_NUM_BASE_EXPERTS`, `DEEPSEEK_MOE_EXPERT_SEGMENTS`, `DEEPSEEK_MOE_SHARED_EXPERTS`, `DEEPSEEK_MOE_ACTIVE_EXPERTS` |
| paper mapping | total fine experts = base experts x segments; shared experts are always on; routed top-k = active - shared |
| default derived active count | if `DEEPSEEK_MOE_ACTIVE_EXPERTS=0`, active count is `shared + max(1, segments - shared)` |
| disabled behavior | `DEEPSEEK_MOE_NUM_BASE_EXPERTS=0` preserves dense coda unless legacy `CODA_MOE_NUM_EXPERTS>0` is set |

## 2026-04-26 LAuReL-LR removal

`train_gpt_parcae.py` no longer carries the default-off LAuReL low-rank residual path. The env knobs, module class, block injections, attention-residual mutual-exclusion checks, and logging were removed after local tests showed the feature was not promising and added code surface.

## 2026-04-26 PLE diagnostics and stabilization

PLE coda-after is mechanically live, but unnormalized PLE over-injects. A 300s Scylla PoE3/bigram8192 run with `PLE_SCOPE=coda PLE_DIM=32 PLE_SCALE_INIT=0.02 PLE_PLACEMENT=after PLE_PROJ_OPTIMIZER=adam PLE_PROJ_LR=0.01` reached exact BPB `1.81262890`; diagnostics showed injected RMS at roughly `30-45%` of stream RMS. Adding `PLE_SCALE_LR=0.004` slowed early growth but still reached exact BPB `1.80265695` with late injected ratio around `20-30%`.

Implementation correction: PLE now logs raw/injected/x RMS, injected ratio, scale abs, and PLE grad norms; supports `PLE_PLACEMENT=before|after`; can route PLE projection to Adam; routes PLE scales through `PLE_SCALE_LR`; and defaults enabled PLE output to RMS-normalized (`PLE_NORM=1`) to stop raw projection norm growth. A short normalized smoke kept raw RMS near `1.0` and injected ratio around `5-8%` through step 118. The full normalized run was interrupted before training, so no BPB is available yet.

First 300s Scylla PoE3/bigram8192 tests with `PLE_SCOPE=none` and `TTT_ENABLED=0` were negative versus the current-local baseline around `1.73508599`: coda rank 8, scale 0.01 reached exact int8 roundtrip BPB `1.74979565`; core rank 8, scale 0.005 reached `1.76566406` and was slower (`1537` steps vs coda `1820` under the same cap). Treat LAuReL-LR as implemented but not yet promising in these placements.

## 2026-04-26 No-loop recurrent-core comparison

Keeping roughly the same effective block count but removing recurrent reuse helped the 300s Scylla PoE3/bigram8192 local baseline: `N_LAYERS_IN_PRELUDE=1 N_LAYERS_IN_RECURRENT_BLOCK=4 N_LAYERS_IN_CODA=1 MEAN_RECURRENCE=1 MEAN_BACKPROP_DEPTH=1`, with PLE/LAuReL/TTT off, reached exact int8 roundtrip BPB `1.72512145` at `1601` steps. It was slower than the recurrent baseline (`~187.5ms/step` vs about `165ms/step`) and larger (`8,026,003` byte int8+zlib model), but beat the matched recent recurrent 300s rerun around `1.73508599`. This suggests recurrence reuse is not clearly paying for the local 300s budget, even if the best longer/solo recurrent result remains around `1.69`.

Larger width/depth recurrence was strongly negative: `MODEL_DIM=512 RECURRENT_DIM=512 N_LAYERS_IN_PRELUDE=2 N_LAYERS_IN_RECURRENT_BLOCK=3 N_LAYERS_IN_CODA=3 MEAN_RECURRENCE=2 MEAN_BACKPROP_DEPTH=1`, with PLE/LAuReL/TTT off, reached exact int8 roundtrip BPB `1.80886096` at `1114` steps. It was too slow (`~269.5ms/step`) and over budget (`23,913,791` byte int8+zlib model, `24,072,355` bytes with code). Do not chase this 512-wide recurrent shape as-is.

## 2026-04-26 Parcae audit fixes

`train_gpt_parcae.py` was cleaned up after a source-level audit found correctness footguns. Defaults now keep the prior `STATE_INIT=like-init` training behavior, with random state modes evaluating as zero state so validation is deterministic. Monitoring and value embeddings default off. Warmup now restores Python, NumPy, CPU torch, and CUDA RNG states.

Correctness fixes landed: distributed train loader carries one overlap token across global spans so rank-boundary and batch-boundary targets are not dropped; `QK_BIAS=1` now has separate query/KV-head bias tensors for GQA; value embeddings are small-initialized if enabled; GPTQ sigma clipping is capped by row amax; QAT covers token, bigram, PLE, and value embedding lookups; low-bit `QUANT_BITS<8` now packs payloads instead of storing int8-shaped tensors; TTT distributed gradient sync is token-weighted for uneven local batches.

Silent no-ops are now guarded: explicit `NUM_LAYERS` and explicit `QK_GAIN_INIT` fail fast in `train_gpt_parcae.py`. Timed runs skip step-0 validation and raw `final_model.pt` save unless `SAVE_RAW_MODEL=1`.

`RECURRENT_INTERMEDIATE_DIM` now defaults to `MLP_MULT * RECURRENT_DIM`; `scripts/run_parcae_scylla_current_best.sh` pins `RECURRENT_INTERMEDIATE_DIM=1024` and `STATE_INIT=like-init` to preserve the old current-best training shape/init while retaining deterministic validation.

## 2026-04-26 PLE removal

`train_gpt_parcae.py` no longer carries the default-off PLE-lite path. The PLE env knobs, per-layer token embedding module, forward injections, PLE monitoring metrics, dedicated optimizer groups, and PLE train-log suffix were removed to keep the active experiment surface smaller.

## 2026-04-26 RWKV-v7 Muon port

`RWKV-LM-V7` now has a default-off `--optimizer muon` path. The port keeps AdamW as the default and uses manual optimization only for Muon mode. Muon is applied to hidden 2D matrix parameters, while `emb.weight`, `head.weight`, vectors/scalars, norms, and `blocks.*.att.r_k` stay on AdamW. Follow-up needed before serious Muon runs: the training callback still schedules only `trainer.optimizers[0]`, so multi-optimizer LR and momentum scheduling needs an audit.

## 2026-04-26 RWKV SP1892 BPB path

`RWKV-LM-V7/eval_fineweb_bpb.py` mirrors the Parcae SentencePiece byte accounting: byte pieces count as one byte, normal pieces count their UTF-8 payload bytes after stripping leading `▁`, and a leading `▁` adds one space byte only when the previous token is not a boundary/control token. The reported BPB is `loss_sum / (ln(2) * scored_bytes)`. Local SP1892 artifacts are not present yet; the checked-in data currently has only `data/datasets/fineweb10B_sp1024` and `data/tokenizers/fineweb_1024_bpe.model`.

Follow-up: SP1892 local data was installed under `data_sp1892/` using `data/download_hf_docs_and_tokenize.py --max-train-shards 1` with HF cache redirected to `/workspace/.cache/huggingface`. The export contains one 100M-token train shard plus the full first validation shard, and the docs JSONL is symlinked to the HF cache to avoid duplicating the 48GB source file. RWKV V7 time-only runs need `src/trainer.py` to initialize `lr = args.lr_init` before the token-based exit branch; otherwise `MY_EXIT_TOKENS=0` fails before the first optimizer step.

## 2026-04-26 RWKV RoPE experiment

`RWKV-LM-V7` has a default-off RoPE path controlled by `--rope_mode none|rk` and `--rope_theta`. The enabled mode rotates the RWKV time-mix recurrent query/key analogues (`r` and `k`) per head after their linear projections and before `fused_k_rwkv7`, so both `kk` and the updated key entering the RWKV7 kernel carry the positional rotation. The inverse-frequency buffer is non-persistent, so RoPE adds no checkpoint tensors. Use `ROPE_MODE=rk` in the FineWeb run scripts to enable it; output dirs get a `-roperk` suffix by default.

## 2026-04-27 RWKV plain int6 artifact path

`RWKV-LM-V7` now has a plain post-training integer quantization path in `src/quant.py`. It quantizes all floating state-dict tensors with symmetric signed intN, using per-row scales for 2D tensors and per-tensor scales otherwise, packs sub-8-bit payloads bitwise, stores a zlib-compressed `.ptz`, and can dequantize back into a strict RWKV state dict. Training writes `rwkv-final.int{bits}.ptz` when `--quant_bits > 0`; `eval_fineweb_bpb.py` can load `.ptz` files directly. Scale tensors are stored as fp32 after overflow tests showed fp16 scales can become `inf` for very large finite tensors. A real smoke quantized `out/fineweb-sp1892-roperkd16-10min-L8-D512-x070/rwkv-final.pth` to int6 and evaluated one validation span successfully: fp checkpoint `val_bpb=1.48937716`, int6 roundtrip `val_bpb=1.49020700` on 1024 tokens. The all-tensor int6 artifact was `21,525,093` bytes, so this proves mechanics but is not yet artifact-budget competitive; next work should be selective int6 and/or lower-overhead serialization.

- 2026-04-27: RWKV Muon now delegates the matrix optimizer step to `torch.optim.Muon`; local code only keeps RWKV-specific parameter grouping and a compatibility constructor.
- 2026-04-27: RWKV partial RoPE is controlled by `--rope_dims`; for the current head size 64, `rope_dims=16` is 25%. A 300s SP1892 FineWeb run scored `val_bpb=1.53697412`, slightly better than full RoPE `1.53729809` but still worse than no-RoPE `1.51916871`.
- 2026-04-27: RWKV Muon mode still uses AdamW for embeddings, head, vectors/scalars, norms, and gain-like tensors, so `--adam_eps` remains active for those fallback groups. A 300s SP1892 FineWeb Muon trial scored `val_bpb=1.69464814`, worse than the AdamW baseline; logged Muon step/token counters appear inflated by Lightning multi-optimizer manual optimization.
- 2026-04-27: RWKV RMSNorm (`--norm_type rmsnorm`) ran cleanly for 300s on SP1892 FineWeb and scored `val_bpb=1.52069558`, slightly worse than the LayerNorm baseline `1.51916871`.
- 2026-04-27: RWKV deep/narrow scaling was negative locally. `L29-D512` was too slow for the 300s budget (~54-55 Ktok/s). The scaled-down `L16-D384` run reached step 1928 / 31.6M tokens in 300s and scored `val_bpb=1.59235983`, worse than the 8x512 LayerNorm baseline `1.51916871`.
- 2026-04-27: RWKV has a default-off PyTorch SDPA hybrid attention path. Layer 0 is forced to remain RWKV so `v_first` is initialized; selected later layers use causal SDPA with optional q/k RoPE and zero-initialized output projection. Tiny train/eval smokes passed with `ATTN_EVERY=1 ATTN_OFFSET=1` on `L2-D128`.
- 2026-04-27: The first real RWKV SDPA hybrid run was positive. `L8-D512` with one attention anchor at layer 4 (`ATTN_EVERY=4 ATTN_OFFSET=4`) ran 3375 steps / 55.3M tokens in 300s and scored `val_bpb=1.50430779`, improving over the no-RoPE LayerNorm baseline `1.51916871` by about `0.01486` BPB.
- 2026-04-27: RWKV SDPA attention RoPE now supports partial rotation via the existing `rope_dims`; `0` means full attention head dim, otherwise only the first `rope_dims` per-head dimensions rotate. If RWKV RoPE and attention RoPE are both enabled, they share the same `rope_dims` setting.
- 2026-04-27: Follow-up partial-attention-RoPE audit found and fixed a demo script gap: `demo-training-run-fineweb.sh` did not pass `ROPE_DIMS`, so demo smokes would silently use full attention RoPE. Tiny CUDA train/eval with `ATTN_EVERY=1 ATTN_OFFSET=1 ATTN_ROPE=1 ROPE_DIMS=16` now passes and eval prints `rope_dims:16 attn_rope:1`.

- 2026-04-27: RWKV learned warm-start experiment added as default-off `--learned_shift_state`. It trains per-layer time-mix and channel-mix token-shift initial caches via `rwkvfla.token_shift(..., cache=...)`, preserving the WKV CUDA kernel ABI. This is not a learned WKV matrix state; it is the lower-risk warm prior that counts as checkpoint parameters when enabled.

- 2026-04-27: RWKV attention layers now support `--attn_mode full|moba`. `full` preserves the existing causal SDPA path; `moba` uses a default-off pure-PyTorch MoBA-style top-k chunk sparse attention fallback with local causal chunk attention plus top-k previous chunks selected by mean-pooled chunk keys. Shell wrappers expose `ATTN_MODE`, `MOBA_CHUNK_SIZE`, and `MOBA_TOPK`. The fallback disables `--compile` because routing uses Python control flow.

- 2026-04-27: `train_gpt_parcae.py` now has a default-off quantized roundtrip sliding/PPM evaluator path, ported against `current_record_sub2.py`. Use `SLIDING_WINDOW_ENABLED=1` for quantized sliding-window BPB and `PPM_ENABLED=1` for byte-level online PPM mixing on the scored sliding prefix (`PPM_SUBSET_TOKENS`, `PPM_ORDER`, `PPM_USE_META_MIX`, `PPM_TOKEN_ORDER`). The PPM path reconstructs tokenizer bytes separately from the BPB byte-count LUT, gathers per-position sliding NLLs across ranks, trims if the prefix is not fully covered, and logs `final_int*_zlib_roundtrip_sliding_ppm_exact`; it does not alter the saved artifact.

- 2026-04-27: `train_gpt_parcae.py` also has default-off validation-time hashed n-gram mixing from `curre_cord_sub_3.py`. Set `NGRAM_EVAL_ORDER>=2` to run it after quantized roundtrip load; it uses score-first chunks (`NGRAM_CHUNK_TOKENS`), rank-split scoring with shared deterministic count-table updates, entropy-adaptive alpha, optional `NGRAM_ORDER_MULTS`, and optional `NGRAM_CUBRIC_CADENCE`. Focused tests covered alpha-zero neural equivalence, zero order-mult equivalence, pure n-gram improvement on repeated streams, Gloo distributed parity, and a tiny CUDA end-to-end roundtrip.

- 2026-04-27: The Parcae n-gram eval path defaults to `NGRAM_MIX_MODE=linear`. Same-checkpoint ablation showed linear n-gram helped strongly, while sparse residual/product expert at boost scale 1.0 hurt. Expert mode remains available via `NGRAM_MIX_MODE=expert`, but its default `NGRAM_EXPERT_BOOST_SCALE` is conservative (`0.25`) and should be tuned against a same-checkpoint baseline before use.

- 2026-04-28: `train_gpt_parcae.py` now has a default-off LZP validation-time expert integrated into the sliding context-mix path. Use `SLIDING_WINDOW_ENABLED=1 LZP_ENABLED=1`; optional PPM+LZP ensembling works with `PPM_ENABLED=1`. The implementation is multi-order/context-confirmed LZP (`LZP_ORDERS`, `LZP_TABLE_BITS`) with conservative alpha ramping from observed causal match streaks, so it does not alter training or saved artifacts. Synthetic repeated-stream tests and a tiny `eval_val_sliding` smoke passed; same-checkpoint real validation is still needed to know whether it improves FineWeb BPB beyond PPM/ngram.

- 2026-04-28: LZP confirmation in `train_gpt_parcae.py` is exact, not probabilistic: default orders up to 8 bytes store a packed `uint64` context key beside the direct-mapped prediction position, and longer experimental orders fall back to byte-for-byte context comparison. A forced hash-slot collision test verifies that different contexts cannot produce false LZP predictions.

- 2026-04-28: `train_gpt_parcae.py` has default-off Attention Residuals controlled by `ATTN_RES_MODE=none|full|block`, `ATTN_RES_SCOPE=prelude|core|coda|all`, and `ATTN_RES_BLOCK_SIZE`. The implementation applies paper-style depth softmax separately inside same-width Parcae stacks, uses one zero-init pseudo-query before each attention/MLP sublayer, RMS-normalizes depth keys, and keeps GQA unchanged. It is intentionally mutually exclusive with parallel residuals, LAuReL, and gradient checkpointing until those combined semantics are defined.

- 2026-04-28: Parcae delayed QAT no longer computes `self.step >= QAT_START_STEP` inside `forward_model`. The training loop calls `GPT.set_training_step(step)`, which latches QAT mode and updates fake-quantizer `enabled` flags outside compiled forward. This avoids `torch.compile` guarding on the per-step Python integer when `COMPILE_MODEL=1`; delayed QAT should now require at most a mode-change graph, not one graph per step. Warmup also primes a QAT-on step when `QAT_START_STEP` is beyond the normal warmup range, so the mode-change graph is not first compiled during measured training.

- 2026-04-28: `train_gpt_parcae.py` XSA support was restored using the record-style efficient GQA projection (`Y` reshaped by KV-head group, normalized self `V`, subtract `(Y dot Vn) Vn`). Routing is by effective depth: prelude uses static layer ids, core uses `prelude + recurrent_step * core_layers + core_layer`, and coda is offset after the sampled recurrent depth. `XSA_LAST_N=0` remains default-off.

- 2026-04-28: `train_gpt_parcae.py` has default-off weight averaging. `EMA_ENABLED=1 EMA_DECAY=0.997` keeps an fp32 EMA updated after each optimizer step and applies it before raw save, quantization, and final validation. `SWA_ENABLED=1 SWA_START_FRAC=0.2 SWA_EVERY=50` averages CPU fp32 checkpoints during warmdown when EMA is off. This follows the later record scripts' priority: EMA wins over SWA if both are enabled.

- 2026-04-28: Parcae EMA/SWA was optimized after profiling showed the initial implementation was too slow because it walked `state_dict()` every step and cast each tensor manually. `ParameterAverager` now snapshots parameter names once, keeps fp32 shadows, updates EMA with `torch._foreach_mul_`/`torch._foreach_add_`, and preserves current buffers at final load. `EMA_UPDATE_EVERY` defaults to `1` for exact per-step EMA; higher values use an effective `decay ** skipped_steps` sparse update for speed.

- 2026-04-28: Parcae recurrent core routing skips `step.item()`/`total_steps.item()` in the default core path where `ATTN_RES_MODE=none` and `XSA_LAST_N=0`. Those scalar layer-id decisions are only needed for XSA/attention-residual routing, so default `COMPILE_MODEL=1` QAT runs avoid Dynamo `Tensor.item()` scalar-capture warnings without changing recurrent checkpoint argument plumbing.

- 2026-04-28: `autoresearch.sh` cleanup must not use broad `pkill -f train_gpt_parcae.py`; overlapping or interrupted runs can kill unrelated fresh experiments. Track the launched training PID and only terminate that process tree from the trap.

- 2026-04-28: Parcae single-process synthetic efficiency microbench (`TRAIN_SEQ_LEN=128`, batch 4, no compile, forward+backward only) showed baseline around `22.5ms`. Relative overheads: no BigramHash `21.3ms`, no QK norm `21.7ms`, QAT8 `31.3ms`, XSA2 `24.4ms`, parallel-core residual `24.3ms`, LAuReL core rank8 `25.8ms`, attention residual block `28.0ms`, coda MoE4 `32.5ms`, DeepSeek coda MoE `33.9ms`, value embeddings `23.8ms`, SwiGLU injection `25.4ms`, per-iteration gradient checkpointing `28.0ms`, EMA update `23.1ms`, CPU SWA add `39.0ms`. Tiny eval-path bench on 4096 validation tokens: standard eval `107ms`, sliding `82ms`, sliding+PPM `155ms`, sliding+LZP `182ms`, ngram4 `112ms`, TTT epoch1 `264ms`. Treat these as relative hot-path signals, not quality results.

- 2026-04-28: Parcae RoPE helper microbench on RTX PRO 4500 Blackwell showed the default full-head RoPE path benefits from skipping empty pass-through concatenations in `apply_rotary_emb_complex_like`. The exact-output patch gave about `1.17-1.30x` forward speedup and `1.13-1.15x` forward+backward speedup on common full-RoPE shapes. Split-half RoPE and bf16-math RoPE were faster forward-only but are not exact semantic matches; keep them as architecture/numerics experiments, not safe micro-optimizations.

- 2026-04-28: Parcae now has a local Liger-style Triton cross-entropy path behind `LIGER_CE` (default off pending benchmark evidence). It replaces the training `F.cross_entropy` over already materialized softcapped logits, computes per-row online softmax loss and logits gradients in one Triton forward kernel, and falls back to PyTorch CE on CPU/no-Triton. Fused linear CE was intentionally not wired yet because the current head path includes tied-output QAT, optional PoE heads, and logit softcap before CE.

- 2026-04-29: Parcae SWA has an opt-in dynamic modulation mode. `SWA_DYNAMIC=1` keeps the existing warmdown start condition but changes snapshot cadence from `SWA_EVERY` toward `SWA_DYNAMIC_MIN_EVERY` as LR scale approaches zero, and can upweight later snapshots up to `SWA_DYNAMIC_WEIGHT_MAX`. Static SWA remains unchanged when `SWA_DYNAMIC=0`; EMA still takes priority.

- 2026-04-28: For Parcae partial RoPE, Tri Dao's FlashAttention-style rotary kernel should be benchmarked through the packed contiguous QKV path, not only as separate post-split Q/K calls. The upstream `apply_rotary_emb_qkv_` fast path takes `(B, T, Hq + 2*Hk, D)` with `num_heads_q`, rotates Q+K in-place in one kernel, leaves V untouched, and matches Parcae's adjacent-pair/interleaved convention when `interleaved=True`. On the local RTX PRO 4500 Blackwell autoresearch shape `B=4,T=1024,Hq=4,Hkv=2,D=96,rope_dims=32`, the packed path was roughly tied forward-only but lower memory, and was consistently faster than current PyTorch for forward+backward in `rope-bench.py`.

- 2026-04-28: `train_gpt_parcae.py` now exposes the packed Tri Dao RoPE path behind `TRIDAO_PACKED_ROPE=1`. It is mutually exclusive with `LIGER_ROPE` and currently requires `QK_BIAS=0` plus unset `CLIP_QKV`, because the existing slow path applies bias/clip before RoPE and the packed fast path rotates immediately after the packed QKV projection. Focused CUDA checks passed for Q/K output, QKV gradients, untouched partial-RoPE tails, untouched V, and a small full attention module output/input-gradient comparison.

- 2026-04-28: Parcae QAT now uses TorchAO only when `QAT_BITS>0`: it prepares `CastedLinear` modules as `FakeQuantizedLinear` with int8 per-token activation fake quant and int4/int8 grouped weight fake quant, prepares embeddings with weight fake quant, gates fake quant by `QAT_START_STEP`, and converts fake-quant modules back to ordinary modules before the existing `.ptz` artifact quantizer. The old custom STE weight-fake-quant fallback was removed. TorchAO 0.17 imports under the current `torch==2.9.1+cu130` pin but warns that its C++ extensions require `torch>=2.11`; pure Python QAT smoke passed, but kernel-accelerated TorchAO behavior needs a torch-stack compatibility decision before assuming speedups.

- 2026-04-28: Parcae parallel residuals now have an opt-in delayed execution path inspired by flash-attn's `ParallelBlock`: `PARALLEL_RESIDUAL_IMPL=delayed` carries attention output, MLP output, and residual separately, then adds the prior block's branches at the start of the next block. It requires `RESIDUAL_MODE=parallel`, `PARALLEL_RESIDUAL_START=-1`, and `PARALLEL_RESIDUAL_RECORD_CONTROLS=0`; tied norms (`PARALLEL_RESIDUAL_TIED_NORM=1`) intentionally leave `norm_2` unused, and `PARALLEL_RESIDUAL_IN_FP32=1` keeps only the carried residual stream in fp32 while feeding norms at the branch dtype.

- 2026-04-28: Parcae sliding eval now permits `EVAL_STRIDE > TRAIN_SEQ_LEN` as sparse validation sampling. `_ttt_chunk_windows` clamps overlap context to zero in that case, so sparse strides score full disjoint windows instead of creating a negative context size. A regression test covers `train_seq_len=8, eval_stride=16`.

- 2026-04-28: Parcae RMSNorm experiment surface was removed. `dropout_add_norm_standalone.py` is gone, the Triton RMSNorm flag/logging was removed from `train_gpt_parcae.py`, and `ParcaeRMSNorm` now stays on plain `F.rms_norm`.

- 2026-04-28: Parcae removed the default-off SwiGLU recurrent injection and coda MoE branches. `INJECTION_TYPE` now supports only `diagonal`, `linear`, and `add`; coda blocks always keep the configured dense MLP class rather than replacing with legacy TopKMoE or DeepSeekMoE.

- 2026-04-28: Parcae RoPE now registers separate non-persistent `outer/recurrent_freqs_cos` and `outer/recurrent_freqs_sin` buffers instead of one stacked cos/sin buffer. This keeps the active prelude/core/coda RoPE tables cached while avoiding per-attention extraction from the stacked last dimension. Exact helper equivalence passed (`maxdiff=0.0`); microbench showed roughly `1.02-1.09x` forward speedup and `1.02-1.03x` forward+backward speedup over the stacked-buffer fast path.

- 2026-04-28: `train_gpt_parcae.py` has an opt-in Triton RMSNorm path via `TRITON_RMSNORM=1`; the old PyTorch-expression `ParcaeRMSNorm` remains the default because a QAT8 model-level benchmark regressed full training step time (`old=29.0370ms`, Triton=`30.2933ms`, `0.96x`) even though the isolated kernel is much faster. The supplied external kernel's backward formula was wrong; the implemented opt-in version uses `dot=sum(dy*weight*x)` and fp32 partial `dw` accumulation. Correctness against `F.rms_norm` passed on bf16/fp16 4096x64, 4096x512, and bf16 16384x512 (`dw` max error around `1e-5`; forward/dx differences within low-precision rounding).

- 2026-04-28: The opt-in Triton RMSNorm is now integrated through `torch.library.triton_op`/`wrap_triton` with registered autograd instead of a direct `torch.autograd.Function`. Correctness smoke passed and QAT8 tiny GPT train step passed, but the dispatcher path still regressed the representative QAT8 fwd+bwd benchmark (`old=29.9685ms`, Triton dispatcher=`31.3430ms`, `0.956x`). Profiler evidence: CUDA work is lower, but each norm still pays generated-autograd/custom-op CPU overhead plus two backward Triton launches. Keep the PyTorch-expression RMSNorm as default unless a fused surrounding block or lower-launch backward is implemented.

- 2026-04-28: `train_gpt_parcae.py` now wires record-style SmearGate into the embedding path. The exact supplied gate blends each position with the previous position via a learned per-channel sigmoid gate, after token embedding plus optional BigramHash and before embedding scale/prelude/core/coda. Focused checks passed: `.venv/bin/python -m py_compile train_gpt_parcae.py`, `python -m pytest tests/test_parcae_parallel_residual.py -q`.

- 2026-04-28: The experimental custom Triton RMSNorm path in `train_gpt_parcae.py` was removed. `ParcaeRMSNorm` now uses plain PyTorch `F.rms_norm`, and `TRITON_RMSNORM` is effectively disabled. Focused checks passed: `.venv/bin/python -m py_compile train_gpt_parcae.py`, `python -m pytest tests/test_parcae_parallel_residual.py -q`.

- 2026-04-28: Parcae attention now always uses a packed QKV projection. `ParcaeCausalSelfAttention` replaces separate `c_q/c_k/c_v` linears with one `c_qkv` linear and splits the output, while strict load compatibility converts old separate weights to packed. CUDA bf16 attention fwd+bwd microbench on shape `B=4,T=128,D=512,H=8,KV=4` showed exact output parity and about `1.21x` speedup (`1.609ms` unpacked vs `1.328ms` packed).

- 2026-04-28: Parcae sliding eval now fuses LZP/context mix and expert n-gram scoring into the same sliding neural forward pass when `SLIDING_WINDOW_ENABLED=1`. The old standalone n-gram evaluator remains for non-sliding use, but final eval skips it when sliding already produced fused n-gram metrics. Synthetic checks verified fused n-gram equals the old standalone n-gram exactly, LZP/context metrics are unchanged by n-gram chunking, and `NGRAM_EVAL_MAX_SECONDS` cuts off only n-gram work while preserving sliding/LZP coverage.

- 2026-04-28: Parcae sliding context-mix PPM uses explicit PPM-D-style half-escape mass (`PPM_ESCAPE_METHOD=d`) with exclusion during backoff. `CONTEXT_MIX_MAX_SECONDS` marks partial scoring with coverage/scored-byte metrics, and token-level PPM precompute also honors the cutoff so meta-mix cannot silently run past the eval budget.

- 2026-04-29: Liger MHC is on upstream `linkedin/Liger-Kernel` main as `liger_kernel.transformers.mhc.LigerMHC` plus functional APIs, but the local autoresearch venv (`liger-kernel==0.7.0`) does not include MHC files and lacks `transformers`. For local MHC benchmarking without changing the venv, clone upstream to `/tmp/Liger-Kernel` and run with `LIGER_KERNEL_SRC=/tmp/Liger-Kernel/src .venv/bin/python ...`.

- 2026-04-29: `train_gpt.py` now has default-off Hedgehog attention via `ATTENTION_TYPE=hedgehog`. It uses per-head trainable Q/K feature maps with the paper's stable `softmax(x) || softmax(-x)` activation (`HEDGEHOG_FEATURE_ACTIVATION=softmax`) and leaves V unmapped. GQA keys/values are repeated before the linear attention path. Optional paper-style attention mimicry is available with `HEDGEHOG_MIMICRY_WEIGHT>0`; it is quadratic and only added during training, so validation remains next-token CE/BPB. Hedgehog feature-map weights/biases are routed to Adam rather than Muon because their per-head stacked shapes are not ordinary 2D hidden matrices. The initial full-head feature map was too slow and memory-heavy; the practical defaults are now `HEDGEHOG_FEATURE_DIM=4` and `HEDGEHOG_CHUNK_SIZE=TRAIN_SEQ_LEN`. Direct CUDA attention fwd+bwd on shape `B=4,T=1024,D=384,H=4,KV=2` measured softmax about `2.30ms`, Hedgehog feature 2 about `6.55ms`, feature 4 about `5.28ms`, feature 8 about `6.31ms`. A truly softmax-speed Hedgehog path still needs a fused linear-attention kernel; PyTorch ops alone are roughly 2-3x slower at usable low feature dims.

- 2026-04-29: Frozen-QK was removed from Parcae after smoke benchmarks showed it was slower, not faster. The packed hook/restore version was about 10% slower, and the split-buffer/custom-autograd attempt also benchmarked about 9% slower on the tiny real-model smoke path. Do not re-add this unless there is a different implementation with a measured speed win.

- 2026-04-29: Parcae PWA audit/fix: `PWAQKVProjection` should use full-width learned bases, not `RRHPWeight`, otherwise PWA secretly also performs RRHP-style input-column compression. Added `ATTN_QKV_MODE=pwa_qk_dense_v` so Q/K can be compressed while V remains independent and dense. Pre-attention conv params must not use the generic scalar Adam LR; the first `ATTN_PRECONV_KERNEL=3` run used scale `0.05` plus scalar LR `0.02` and went NaN at step 2. Conv params now have `ATTN_PRECONV_LR` default `0.001`, default scale `0.005`, and optimizer groups fail fast on duplicate parameter ids. The fixed 90s compiled smoke stayed finite but scored exact BPB `3.33867651`, so this conv design is not yet promising.

- 2026-04-29: PWA shared bases should not be optimized like ordinary full matrices. They are now excluded from the Muon+WD group and use a separate no-WD Adam group controlled by `PWA_LR`. PWA base init is scaled by `PWA_INIT_SCALE` (default `1/sqrt(2)`) to reduce shared-Q/K correlation at initialization. A focused `pwa_qk_dense_v` name smoke confirmed the filter catches `prelude/core/coda.*.attn.qkv_proj.qk_proj.bases`.

- 2026-04-29: The first "fixed" pre-attention conv was still suspect: at step 1800 it was `train_loss=6.6073` versus `3.0756` for the no-conv SOTA at the same step. The implementation has been corrected to start exactly at identity by initializing the causal depthwise kernel's current-token tap to 1, all other taps to 0, and applying `x + scale * (conv(x) - x)`. This makes local mixing a learned delta instead of adding a random conv branch into attention inputs.

- 2026-04-29: The preconv identity run's CUDA illegal memory access was not caused by the conv kernel or Muon. `CUDA_LAUNCH_BLOCKING=1` moved the fault to compiled backward before the optimizer, at an Inductor-generated clone of a non-dense packed-QK view feeding the custom Tri Dao RoPE wrapper. In packed `[Q,K,V]` layout, `qkv[:, :, :Hq+Hk]` has V-head gaps between sequence positions, so the wrapper now builds a dense `torch.cat((q, k), dim=2)` buffer for RoPE and concatenates V back. Checks passed: `python -m py_compile train_gpt_parcae.py`, small CUDA packed-RoPE forward/backward parity (`max_diff=0`, `grad_diff=0`), and full-shape compiled forward/backward with `ATTN_PRECONV_KERNEL=3 TRIDAO_PACKED_ROPE=1 CUDA_LAUNCH_BLOCKING=1`.

- 2026-04-29: Parcae Liger CE graph-break warning under `COMPILE_MODEL=1` comes from upstream `liger_kernel.ops.cross_entropy.cross_entropy_forward`, which always executes `target_mask.sum().item()` even when our training labels have no ignore tokens. `train_gpt_parcae.py::liger_cross_entropy` now detects Dynamo tracing and uses the equivalent softcapped PyTorch `F.cross_entropy` path during compilation, while preserving Liger CE for non-compiled CUDA calls.
