# Codex Learning Notes

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

## 2026-04-26 LAuReL-LR implementation

`train_gpt_parcae.py` now has default-off LAuReL low-rank residual augmentation. Use `LAUREL_SCOPE=prelude|core|coda|all`, `LAUREL_RANK>0`, `LAUREL_SCALE_INIT`, and `LAUREL_NORM=0|1`. Each selected `TransformerPreNormBlock` adds `scale * RMSNorm(right(left(block_input)))` to the block output, using the record-style mixed residual input when parallel residual mode is active. Disabled default creates zero LAuReL modules.

Focused checks passed: py_compile under system Python and `.venv`, enabled CUDA forward/backward smoke, default-off module count, strict state-dict reload, and int8 quant/dequant strict reload.

## 2026-04-26 PLE diagnostics and stabilization

PLE coda-after is mechanically live, but unnormalized PLE over-injects. A 300s Scylla PoE3/bigram8192 run with `PLE_SCOPE=coda PLE_DIM=32 PLE_SCALE_INIT=0.02 PLE_PLACEMENT=after PLE_PROJ_OPTIMIZER=adam PLE_PROJ_LR=0.01` reached exact BPB `1.81262890`; diagnostics showed injected RMS at roughly `30-45%` of stream RMS. Adding `PLE_SCALE_LR=0.004` slowed early growth but still reached exact BPB `1.80265695` with late injected ratio around `20-30%`.

Implementation correction: PLE now logs raw/injected/x RMS, injected ratio, scale abs, and PLE grad norms; supports `PLE_PLACEMENT=before|after`; can route PLE projection to Adam; routes PLE scales through `PLE_SCALE_LR`; and defaults enabled PLE output to RMS-normalized (`PLE_NORM=1`) to stop raw projection norm growth. A short normalized smoke kept raw RMS near `1.0` and injected ratio around `5-8%` through step 118. The full normalized run was interrupted before training, so no BPB is available yet.

First 300s Scylla PoE3/bigram8192 tests with `PLE_SCOPE=none` and `TTT_ENABLED=0` were negative versus the current-local baseline around `1.73508599`: coda rank 8, scale 0.01 reached exact int8 roundtrip BPB `1.74979565`; core rank 8, scale 0.005 reached `1.76566406` and was slower (`1537` steps vs coda `1820` under the same cap). Treat LAuReL-LR as implemented but not yet promising in these placements.

## 2026-04-26 No-loop recurrent-core comparison

Keeping roughly the same effective block count but removing recurrent reuse helped the 300s Scylla PoE3/bigram8192 local baseline: `N_LAYERS_IN_PRELUDE=1 N_LAYERS_IN_RECURRENT_BLOCK=4 N_LAYERS_IN_CODA=1 MEAN_RECURRENCE=1 MEAN_BACKPROP_DEPTH=1`, with PLE/LAuReL/TTT off, reached exact int8 roundtrip BPB `1.72512145` at `1601` steps. It was slower than the recurrent baseline (`~187.5ms/step` vs about `165ms/step`) and larger (`8,026,003` byte int8+zlib model), but beat the matched recent recurrent 300s rerun around `1.73508599`. This suggests recurrence reuse is not clearly paying for the local 300s budget, even if the best longer/solo recurrent result remains around `1.69`.

Larger width/depth recurrence was strongly negative: `MODEL_DIM=512 RECURRENT_DIM=512 N_LAYERS_IN_PRELUDE=2 N_LAYERS_IN_RECURRENT_BLOCK=3 N_LAYERS_IN_CODA=3 MEAN_RECURRENCE=2 MEAN_BACKPROP_DEPTH=1`, with PLE/LAuReL/TTT off, reached exact int8 roundtrip BPB `1.80886096` at `1114` steps. It was too slow (`~269.5ms/step`) and over budget (`23,913,791` byte int8+zlib model, `24,072,355` bytes with code). Do not chase this 512-wide recurrent shape as-is.

## 2026-04-26 Parcae audit fixes

`train_gpt_parcae.py` was cleaned up after a source-level audit found correctness footguns. Defaults now use deterministic zero recurrent state, monitoring off, and value embeddings off. Legacy random state modes still work during training but evaluate as zero state so validation is deterministic. Warmup now restores Python, NumPy, CPU torch, and CUDA RNG states.

Correctness fixes landed: distributed train loader uses one shared overlap token instead of dropping rank-boundary targets; `QK_BIAS=1` now has separate query/KV-head bias tensors for GQA; value embeddings are small-initialized if enabled; GPTQ sigma clipping is capped by row amax; QAT covers token, bigram, PLE, and value embedding lookups; low-bit `QUANT_BITS<8` now packs payloads instead of storing int8-shaped tensors; TTT distributed gradient sync is token-weighted for uneven local batches.

Silent no-ops are now guarded: `XSA_LAST_N>0`, explicit `NUM_LAYERS`, and explicit `QK_GAIN_INIT` fail fast in `train_gpt_parcae.py`. Timed runs skip step-0 validation and raw `final_model.pt` save unless `SAVE_RAW_MODEL=1`.

`RECURRENT_INTERMEDIATE_DIM` now defaults to `MLP_MULT * RECURRENT_DIM`; `scripts/run_parcae_scylla_current_best.sh` pins `RECURRENT_INTERMEDIATE_DIM=1024` to preserve the old 4x current-best shape.
