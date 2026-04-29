# Experiment Notes

## 2026-04-28/29 SP1892 QAT, record-copy, and 9L Parcae session

This section consolidates the SP1892 work from the QID/QAT discussion through the record-copy comparisons and value-embedding check. The high-level outcome is that the best completed local run in this session is still the simple 9-effective-layer Parcae recipe at exact int6 BPB `1.47431031`; QAT/GPTQ/XSA-heavy paths were either slower, worse, or hit Blackwell/Triton compile issues.

Update: the best completed local run in this family is now
`runs/sp1892_9l512_mlp3_bh4096_swa_dyn_fullattn_b32k_int6rans_20260429`.
It uses SP1892, 9 effective 512d layers, `MLP_MULT=3`, BigramHash `4096x128`,
full attention, `TRAIN_BATCH_TOKENS=32768`, dynamic SWA, and int6 RANS+zlib export.
It reached step `6384` in the 600s train cap, pre-export `val_bpb:1.4059`, and exact
roundtrip `val_bpb:1.41159149`. Total submission size was `16,822,763` bytes, so it
improves quality substantially over the earlier `1.47431031` SP1892 baseline but is
still slightly above a strict 16 MiB budget.

Follow-up PWA QKV trial:

- Run: `runs/sp1892_9l512_mlp3_bh4096_swa_dyn_fullattn_b32k_pwa_int6rans_20260429`
- Change from the current best: set `ATTN_QKV_MODE=pwa`; all other major settings kept
  matched (`SP1892`, `TRAIN_BATCH_TOKENS=32768`, full attention, dynamic SWA, int6
  RANS+zlib).
- Header confirmed `qkv_projection:pwa`, `pwa_num_bases:6`, `pwa_permute_size:4`, and
  params dropped from `23,590,657` to `18,616,833`.
- Early result was clearly worse with no useful speed gain: at step `1600`, PWA train
  loss was `3.5953` at `92.85ms/step`; the packed-QKV best run was `3.0635` at roughly
  the same step and `93.17ms/step`.
- Stopped early at step `1600`; not worth completing under this exact configuration.

Fixed-PWA same-budget follow-up:

- Code fix: `PWAQKVProjection` no longer wraps its basis rows in `RRHPWeight`. It now
  stores full-width learned bases (`num_bases x dim`) and indexes those rows directly,
  matching the SLlama PWA pseudocode structure more closely. This keeps PWA independent
  from `ATTN_QKV_MODE=rrhp`.
- Parameter smoke: at width `512`, packed QKV has `1024 x 512 = 524,288` parameters per
  attention layer, while fixed PWA with `PWA_NUM_BASES=6` has `6 x 512 = 3,072`; PWA still
  saves `521,216` QKV parameters per layer.
- Run: `runs/sp1892_9l576_mlp3_bh4096_swa_dyn_fullattn_b32k_pwa8_fixed_int6rans_lr18_20260429`
- Config changes from current best: fixed PWA, width scaled to `MODEL_DIM=RECURRENT_DIM=576`
  to spend the saved QKV budget, `PWA_NUM_BASES=8`, `PWA_PERMUTE_SIZE=4`, and reduced
  LR knobs (`MATRIX_LR=0.018`, `SCALAR_LR=0.018`, `TIED_EMBED_LR=0.028`).
- Header confirmed near-matched size: `23,308,161` params versus packed best
  `23,590,657` params.
- Result: `4924` steps, `121.87ms/step`, pre-export `val_bpb:1.6477`, final exact
  int6 RANS+zlib roundtrip `val_bpb:1.65480940`, total submission size `16,352,119`
  bytes.
- Interpretation: fixed PWA is now actually parameter-saving and can fit the artifact
  budget after scaling, but quality is far worse than the packed-QKV current best
  (`1.41159149`). The loss gap is too large to explain as a minor LR issue.

PWA/conv audit after the poor PWA and conv runs:

- Code audit found the current PWA class had regressed to wrapping its basis rows in
  `RRHPWeight`, so PWA was also applying RRHP-style input-column compression. That has
  been corrected back to full-width learned bases.
- Added `ATTN_QKV_MODE=pwa_qk_dense_v` for the next controlled PWA test. It applies PWA
  to Q/K rows only and leaves V as an independent dense projection, because the earlier
  PWA path compressed/tied V and likely damaged content flow.
- Added a fail-fast optimizer duplicate check across all optimizer groups.
- Follow-up optimizer/init fix: PWA shared bases are now excluded from the ordinary
  Muon + weight-decay matrix group and routed to a separate Adam group controlled by
  `PWA_LR` with no weight decay. Their init is scaled by `PWA_INIT_SCALE`, defaulting
  to `1/sqrt(2)`, to reduce the correlated Q/K dot-product variance from reused bases.
  A focused name smoke with `ATTN_QKV_MODE=pwa_qk_dense_v` confirmed the optimizer
  filter catches `*.attn.qkv_proj.qk_proj.bases`.
- The first pre-attention conv full run
  `runs/sp1892_9l512_mlp3_bh4096_swa_dyn_fullattn_b32k_preconv3_int6rans_20260429`
  used `ATTN_PRECONV_KERNEL=3`, `ATTN_PRECONV_SCALE_INIT=0.05`, and let the 3-D conv
  weights fall into the scalar Adam group at `SCALAR_LR=0.02`. It hit `train_loss:nan`
  at step `2` and stayed NaN through the last logged step `2500`; no final score.
- Fix: pre-attention conv params now use their own Adam group via `ATTN_PRECONV_LR`
  (default `0.001`) and the default conv residual scale was lowered to `0.005`.
- Smoke after the fix:
  `runs/sp1892_9l512_mlp3_bh4096_swa_dyn_fullattn_b32k_preconv3_safe_smoke_20260429`.
  It stayed finite, reached step `804` in a 90s train cap, `step_avg:111.98ms`, and
  final exact int6 RANS+zlib BPB `3.33867651`. This proves the NaN was fixed, but the
  early quality is much worse than the no-conv SOTA trajectory, so a full conv rerun is
  not justified under this exact conv design.
- Full fixed-conv retry:
  `runs/sp1892_9l512_mlp3_bh4096_swa_dyn_fullattn_b32k_preconv3_fixed_int6rans_20260429`.
  It stayed finite but was still clearly broken relative to the no-conv baseline:
  at step `1800`, train loss was `6.6073` and `111.16ms/step`, while the no-conv SOTA
  run was `3.0756` and `93.23ms/step` at the same step. The run was interrupted before
  final export.
- Implementation correction after that comparison: pre-attention conv now starts as an
  exact identity delta. The depthwise kernel is initialized to current-token identity
  and the forward path is `x + scale * (conv(x) - x)`, instead of adding a random conv
  branch directly. This should make the conv path match the no-conv baseline at init
  and only learn local offsets.

### Quick tokenizer sweep: SP1024 vs SP1892 vs SP4096

Goal: get a low-impact directional comparison of tokenizer choices without tying up the GPU. This is not an official challenge-quality comparison: each run used a tiny matched train/val slice rather than full validation, and the first `131072` validation tokens cover different byte spans for different tokenizers. The comparison is still useful as a quick same-code, same-shape signal.

Shared config:

- Model: Parcae `MODEL_DIM=96`, `RECURRENT_DIM=96`, 1 prelude / 1 core / 1 coda, `MEAN_RECURRENCE=1`, `MLP_MULT=2`, no BigramHash, no EMA/SWA, no QAT/GPTQ.
- Training: `TRAIN_BATCH_TOKENS=1024`, `TRAIN_SEQ_LEN=128`, `MAX_WALLCLOCK_SECONDS=45`, `COMPILE_MODEL=0`, `LIGER_CE=1`, `VAL_BATCH_SIZE=4096`.
- Data: local tiny sweep shards with `1,048,576` train tokens and `131,072` val target tokens per tokenizer.
- GPU impact: peak allocated memory stayed below `215 MiB`; sampled SP1892/SP4096 process SM use was around `8-18%` after killing a stale unrelated `train_gpt.py` process.

| Tokenizer | Run | Params | Steps | Step avg | Val loss | Exact BPB | Peak alloc | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SP1024 | `logs/tokenizer_sweep_lowgpu_sp1024_20260429_022549.txt` | 314,368 | 2194 | 20.51ms | 4.0510 | 2.42985855 | 70 MiB | Smallest vocab, fastest steps, worst BPB in this slice |
| SP1892 | `logs/tokenizer_sweep_lowgpu_sp1892_20260429_022549.txt` | 397,696 | 2083 | 21.61ms | 4.5631 | 2.35906366 | 112 MiB | Middle vocab, middle result |
| SP4096 | `logs/tokenizer_sweep_lowgpu_sp4096_20260429_022549.txt` | 609,280 | 1843 | 24.42ms | 5.1549 | 2.24441973 | 215 MiB | Largest vocab, slowest and largest, best BPB in this slice |

Interpretation:

- In this low-budget controlled sweep, BPB improved monotonically with vocab size: SP4096 beat SP1892, which beat SP1024.
- The cost also rose monotonically: parameters, memory, and step time all increased with vocab size because tied embeddings and output projection grow with vocab.
- SP4096 looks worth testing in a real run, but full SP4096 validation needs care: default `VAL_BATCH_SIZE=524288` OOM'd on a tiny smoke because the current Liger CE wrapper applies softcap in fp32 before CE. Use a smaller validation batch or optimize eval softcap before serious SP4096 sweeps.

### Implementation and setup decisions

- QAT was reworked toward the TorchAO-style path: `QAT_BITS>0` prepares `CastedLinear` modules as TorchAO fake-quantized linears and gates fake quant outside compiled forward via `GPT.set_training_step(step)`. The old custom STE fallback was removed earlier in this session series.
- The training script now honors `GRAD_ACCUM_STEPS` from the environment instead of always using `8 // world_size`. This mattered a lot on the single RTX PRO 4500 because record scripts assumed 8 GPUs and otherwise did eight local microsteps.
- The active hardware target for these runs is one NVIDIA RTX PRO 4500 Blackwell, not the 8xH100 setup used by the leaderboard records.
- User correction applied for the main branch of experiments: use SP1892, not Scylla; QAT/quant target should be int6 where used, not int4 or int8.
- `COMPILE_MODEL=1` was kept for serious runs. Disabling compile is not considered an acceptable final workaround here.

### Early QAT/XSA/GPTQ runs

| Run | Main config | Result | Interpretation |
| --- | --- | --- | --- |
| `runs/manual_parallel_xsa_muon_qat4_gptq_int6_sp1892_20260428_212909` | 256d Parcae, XSA4, QAT4 activations, GPTQ int6, EMA, `grad_accum_steps=8` | Failed during compile after warmup: Triton `libdevice.10.bc` parse error | Environment/Inductor path failure before any useful score |
| `runs/manual_parallel_xsa_muon_qat4_gptq_int6_sp1892_20260428_215045` | same family, compile got past warmup | `1092` steps, pre-roundtrip BPB `1.7386`, final GPTQ int6 BPB `1.84045398`, `~263.8ms/step` by end | Too slow and GPTQ roundtrip hurt badly |
| `runs/fix_sp1892_parallel_xsa_muon_qat6_gptq6_keepfragile_20260428_223813` | 256d, XSA4, QAT6, GPTQ6, EMA, fragile tensors kept float | `2613` steps, pre-roundtrip BPB `1.6573`, final GPTQ int6 BPB `1.66229903`, `~110.2ms/step` | Better than QAT4 path but still far from the later no-QAT 512d baseline |
| `runs/sp1892_no_xsa_512_rec3x3_qat6_gptq6_20260428_224917` | 512d, recurrence 3x with backprop 1, XSA off, QAT6/GPTQ6, EMA | `2363` steps, pre-roundtrip BPB `1.6115`, final GPTQ int6 BPB `1.62056918`, artifact over budget at `17,690,502` bytes | Turning off XSA and going 512d helped, but recurrence/QAT/GPTQ remained too slow and too large |

Two follow-up run dirs were created while considering TTT/no-recurrence variants:

- `runs/sp1892_no_xsa_512_rec3x3_qat6_gptq6_ttt_bh64_20260428_230628`
- `runs/sp1892_no_xsa_512_core3_norecur_qat6_gptq6_ttt_bh64_20260428_230806`

Those were abandoned in favor of simplifying the shape and copying the leaderboard feature set more selectively. The main lesson from this block was that QAT+GPTQ+XSA was not the immediate path to a good single-GPU score under the 300s local cap.

### Leaderboard record comparison

The relevant non-LZMA records inspected were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`

Their recipe is not just scale. It combines 8xH100 throughput with an 11-layer 512d U-Net-style transformer, GQA, 3x MLP, SmearGate, BigramHash, last-4-layer XSA, partial RoPE, LN scaling, shared value embeddings, EMA/SWA variants, Muon/AdamW split, GPTQ-lite/int6 export, zstd/zlib compression, and sliding-window evaluation. The 03-21 README also notes that its late-QAT path likely had no effect because `torch.compile` constant-folded the QAT flag.

Scale is still a dominant difference: the record trained roughly `5.5B` tokens in 10 minutes on 8 H100s. The local 300s Parcae baseline below trained about `6079 * 16384 ~= 99.6M` tokens, around 55x fewer than the record run.

### Record-code attempts on one GPU

| Run | Change | Result | Interpretation |
| --- | --- | --- | --- |
| `runs/sp1892_9l_512_mlp3_smear_bh_swa_int6_zlib_20260428_232327` | Copied the 03-20 record script too literally: `TRAIN_BATCH_TOKENS=786432`, `TRAIN_SEQ_LEN=2048`, hardcoded `grad_accum_steps=8` on one GPU | `142` steps, pre-roundtrip BPB `2.9318`, about `2125ms/step`, int6 artifact `16,683,515` bytes and total `16,735,758` bytes | Not viable on one RTX PRO 4500; it was an 8xH100 recipe |
| `runs/sp1892_single_gpu_9l512_mlp3_bh_swa_int6_196k_ga2_20260428_233527` | Run-local record copy with `TRAIN_BATCH_TOKENS=196608`, `GRAD_ACCUM_STEPS=2` | Still around `545ms/step`; stopped before full result | Better than the literal copy, but still far too slow for local iteration |

This led to the decision to copy the feature set selectively rather than copying the record’s batch regime.

### Main 9-effective-layer Parcae branch

The stable local recipe moved to SP1892, `MODEL_DIM=512`, effective shape `4 prelude + 1 core + 4 coda`, no recurrence reuse (`MEAN_RECURRENCE=1`, `MEAN_BACKPROP_DEPTH=1`), `MLP_MULT=3`, delayed parallel residuals, partial RoPE 16, QK norm, SmearGate, BigramHash, int6 export, and no QAT/GPTQ.

| Run | Change | Steps | Step avg | Pre-roundtrip BPB | Final int6 BPB | Size | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `runs/sp1892_parcae_9l512_mlp3_smear_bh4096_swa_int6_single_20260428_234444` | 9-effective-layer Parcae, BigramHash `4096x128`, one head, ungated, SWA, no EMA | 6079 | 49.36ms | 1.4698 | 1.47431031 | total `17,802,177` bytes | Best completed run in this session, but over 16MB budget |
| `runs/sp1892_parcae_9l512_mlp3_bh4096h2gate_ema_int6_single_20260429_000536` | Same shape, BigramHash heads 2 + gate, EMA instead of SWA | stopped around step 5400 | about 50.6ms | no final | no final | n/a | Training looked only marginally different, so it was stopped |
| `runs/sp1892_parcae_9l512_mlp3_smear_bh4096_swa_int6_ve_single_20260429_ve` | Same as best baseline plus `USE_VALUE_EMBEDDINGS=1`, kept `LIGER_CE=1` | failed before training | n/a | n/a | n/a | n/a | `torch.compile` failed inside the Liger cross-entropy Triton kernel on Blackwell with fp32/fp64 loop-carried type assertion |
| `runs/sp1892_parcae_9l512_mlp3_smear_bh4096_swa_int6_ve_noliger_single_20260429` | Same VE test, `COMPILE_MODEL=1`, `LIGER_CE=0` | 5107 | 58.74ms | 1.4789 | 1.48408480 | total `19,900,288` bytes | VE added about 0.97M params, slowed training, and scored worse |

Interpretation: for this 300s single-GPU regime, value embeddings are not worth it in the current wiring. The result is not perfectly controlled because `LIGER_CE` had to be disabled to avoid the Blackwell/Triton compile failure, but the VE run was slower, larger, and worse on validation.

### SP1024 autoresearch replication and batch-size checks

This block tried to reproduce `logs/autoresearch_20260428_033810_261426.txt`, because that run had a much better SP1024 result than the later SP1892 attempts.

Original reference from `logs/autoresearch_20260428_033810_261426.txt`:

- Shape: SP1024, `MODEL_DIM=384`, `RECURRENT_DIM=384`, 1 prelude / 3 recurrent core / 2 coda, `MEAN_RECURRENCE=2`, `MEAN_BACKPROP_DEPTH=1`, `MLP_MULT=4`, GQA `4q/2kv`, BigramHash `8192x128` heads 2 gated, PoE3, SWA, int8 export.
- Batch/training: `TRAIN_BATCH_TOKENS=131072`, old implicit local `grad_accum_steps=8`, `MAX_WALLCLOCK_SECONDS=600`.
- Result: `step:2853`, `step_avg:210.35ms`, `val_loss:2.3339`, `val_bpb:1.3823`; final int8 roundtrip exact `val_loss:2.33427464`, `val_bpb:1.38248893`.

| Run | Change | Result | Interpretation |
| --- | --- | --- | --- |
| `runs/rep_autoresearch_20260428_033810_opt_20260429` | Copied the old shape but changed too much for speed: `TRAIN_BATCH_TOKENS=32768`, `GRAD_ACCUM_STEPS=1`, delayed parallel residuals, packed RoPE, compiled Muon, `LIGER_CE=0` | `10241` steps, `step_avg:58.59ms`, pre-roundtrip BPB `1.5066`, final int8 BPB `1.50863911` | Fast but not comparable; smaller effective batch and residual semantic changes likely broke the training recipe |
| `runs/rep_autoresearch_20260428_033810_batch131k_ga2_20260429` | Restored original effective batch semantics more closely: `TRAIN_BATCH_TOKENS=131072`, `GRAD_ACCUM_STEPS=2`, sequential residuals, packed RoPE, compiled Muon, `LIGER_CE=0` | Reached `step:1000`, `step_avg:228.18ms`, train loss `2.6248`; no final eval in log | Loss lined up closely with the old reference through step 900; throughput was slightly slower than the old `210ms/step`, but close enough to show the code path was not fundamentally broken |
| `runs/rep_autoresearch_20260428_033810_batch262k_ga2_ligerce_20260429` | Doubled effective batch to `TRAIN_BATCH_TOKENS=262144`, kept `GRAD_ACCUM_STEPS=2`, enabled `LIGER_CE=1`, otherwise preserved sequential residuals and old shape | Trained to `step:1200`, `step_avg:467.48ms`, train loss `2.4235`, then crashed entering validation | Bigger batch fit in memory (`~14.4GB`) and was stable at about `561k tokens/s`, but Liger CE with `torch.compile` is still not reliable; validation failed on a Dynamo guard mismatch involving `self.softcap` |

Observed train-loss comparison for the 131k reproduction versus the old reference:

| Step | Old reference train loss | 131k GA2 reproduction train loss |
| ---: | ---: | ---: |
| 100 | 3.7101 | 3.7209 |
| 200 | 3.2151 | 2.9805 |
| 300 | 2.9158 | 2.9032 |
| 400 | 2.8715 | 2.8866 |
| 500 | 2.7975 | 2.8048 |
| 600 | 2.7601 | 2.7712 |
| 700 | 2.5317 | 2.5511 |
| 800 | 2.6270 | 2.6507 |
| 900 | 2.6340 | 2.6502 |

Token-matched read on the 262k Liger CE run was initially encouraging, but not score-valid:

- At current step 1000, the 262k run had seen about as many tokens as the old 131k/reference step 2000.
- Old reference step 2000 train loss: `2.5653`.
- 262k Liger CE step 1000 train loss: `2.3923`.
- 262k Liger CE step 1100 train loss: `2.3232`.
- 262k Liger CE step 1200 train loss rose to `2.4235`, so the last train-loss point was worse than the previous one.

Crash details for `LIGER_CE=1`:

- During warmup/training it emitted Triton warnings from the Liger softcap path, including `invalid operands of type pointer<fp32> and triton.language.float32`.
- The run continued through training, but final validation crashed inside compiled Liger CE with `AssertionError: Guard failed on the same frame it was created` and `tensor '___from_numpy(self.softcap)' dispatch key set mismatch`.
- Fix applied after the crash: Parcae now applies logit softcap in PyTorch before calling Liger CE and always passes `softcap=None` into `LigerCrossEntropyLoss`. This avoids Liger's fused softcap Triton path while still using Liger for CE. Focused CUDA `torch.compile` smoke with softcap passed backward with finite gradients.

Current interpretation of the SP1024 replication:

- Restoring original batch semantics and sequential residuals fixed the worst regression from the fast 32k run.
- Doubling effective batch to 262k may be promising, but the Liger CE crash invalidated the score and the last train-loss line ticked up.
- Next clean test should rerun the 262k batch with the patched `LIGER_CE=1` path and verify that final validation/export completes.

### Current beliefs after this session

- The biggest remaining gap to the 1.12 leaderboard records is still token budget and architecture mismatch, not a single missing post-training quantization switch.
- QAT/GPTQ should not be the default next experiment until the base recipe is much stronger and artifact size is under control.
- XSA has repeatedly looked suspect locally; leave it off unless testing a narrow, controlled variant.
- The most useful record features to keep copying are shape-level features that do not explode step time: 11-ish effective layers, 512d, 3x MLP, SmearGate, BigramHash, partial RoPE, LN/residual scaling, and careful EMA/SWA/export.
- The current best local branch still needs artifact-size work: even the best no-VE int6 run is about `17.8MB` total, over the 16MB target.

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

- A full local Scylla TTT result exists below, but it is not yet an official-quality result because the TTT eval pass took longer than 10 minutes locally.
- Quantized-weight SGD TTT is experimental and may hurt; the previous record-submission code warned that SGD TTT on quantized weights was unfavorable in that setup.
- Manual full-parameter gradient all-reduce is correctness-oriented but may be expensive.
- The latest cleanup reduced duplicated expressions but added helper surface; `_validation_result` and `_token_byte_sum` are worth keeping for metric safety, while `_window_batch` is debatable if minimizing line count becomes the priority.

Full local Scylla TTT check:

| Run | Standard exact BPB | TTT exact BPB | TTT eval time | Notes |
| --- | ---: | ---: | ---: | --- |
| `runs/exp_scylla_poe3_diag_ttt_compare_300.console.log` | `1.73147091` | `1.68571400` | `682647ms` | Uses `VAL_BYTE_COUNT_OVERRIDE`; score-first TTT improved BPB but exceeded 10 minutes on the local eval path |

Interpretation: TTT is promising for BPB, but the current implementation is too slow to treat as a competition-ready default. The next TTT work should target stride/chunk/epoch reductions or a cheaper adapted-parameter subset while preserving the score-before-train legality invariant.

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

- A completed scale `0.1` result did not help.
- The default was changed from an earlier local `0.1` to `0.0` to keep the new branch inactive by default.

Completed check:

| Run | Injection | Corrected BPB | Logged BPB | Exact loss | Steps | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `logs/exp_scylla_poe3_diag_scale_compare_300_rerun.txt` | `diagonal` | `1.73644542` | `1.63007385` | `2.91578821` | 1991 | Baseline; omitted byte override, corrected from exact loss |
| `logs/exp_scylla_poe3_swiglu_add_scale0p1_compare_300.txt` | `swiglu-add`, scale `0.1` | `1.79727366` | `1.68717587` | `3.01792921` | 2017 | Worse despite slightly faster step time |

Interpretation: raw additive SwiGLU token reinjection at scale `0.1` is too disruptive. If revisited, use a much smaller scale, a learned zero-init gate, or modulation of the existing diagonal path instead of an additional raw residual injection.

### PLE-lite token conditioning

Implemented default-off Per-Layer Embedding style token conditioning in `train_gpt_parcae.py`.

Added knobs:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `PLE_SCOPE` | `none` | Enables PLE modules in `prelude`, `core`, `coda`, or `all` |
| `PLE_DIM` | `0` | Low-rank token lookup width; `0` disables PLE |
| `PLE_SCALE_INIT` | `0.0` | Per-layer scalar init, zero by default for exact no-op behavior |
| `PLE_INIT_STD` | `0.02` | Init std for PLE token lookup rows |

Implementation behavior:

- Each selected physical layer gets a small `vocab_size x PLE_DIM` lookup, a `PLE_DIM -> stream_dim` projection, and a scalar gate.
- PLE is injected before the selected block: `x = x + scale * proj(ple_embed[token])`.
- Coda/prelude PLE projects to `MODEL_DIM`; core PLE projects to `RECURRENT_DIM`.
- PLE modules are created after existing model weight initialization, so enabling zero-scale PLE does not perturb existing initialized weights.
- PLE lookup tables are optimized in the token Adam group; PLE projections use the matrix/Muon group; PLE scalar gates use scalar Adam.

Focused checks run:

- `.venv/bin/python -m py_compile train_gpt_parcae.py`
- `git diff --check`
- Default-off construction remains valid.
- Invalid PLE configs fail early: enabled scope with `PLE_DIM=0`, `PLE_SCOPE=none` with positive `PLE_DIM`, and unknown scope.
- Enabled coda PLE with `PLE_DIM=8 PLE_SCALE_INIT=0` preserved common initialized parameters and produced `max_logit_diff=0.0` versus disabled PLE on a deterministic tiny CPU forward.
- Scope wiring check produced expected layer counts: `prelude -> 1/0/0`, `core -> 0/2/0`, `coda -> 0/0/1`, `all -> 1/2/1` in a tiny config.
- Nonzero-scale coda PLE changed logits on a deterministic tiny CPU forward, proving the injection path is live.
- Tiny CPU backward with zero-scale coda PLE produced finite loss and nonzero PLE scale gradient.
- Tiny CPU backward with nonzero-scale coda PLE produced finite gradients for PLE lookup, projection, and scale params.
- QAT registration includes the PLE projection `CastedLinear`.
- Activation-checkpointed backward with all-scope PLE produced finite loss.
- Strict state-dict roundtrip with core PLE preserved logits exactly.
- Int8 quantize/dequantize state-dict roundtrip with all-scope PLE loaded strictly and produced finite logits; tiny expected max logit diff was `6.603635847568512e-05`.

Recommended first experiment:

```bash
PLE_SCOPE=coda \
PLE_DIM=32 \
PLE_SCALE_INIT=0.0 \
POE_NUM_EXPERTS=3 \
POE_HEAD_LR=0.002 \
BIGRAM_HASH_BUCKETS=8192 \
TTT_ENABLED=0 \
RUN_ID=exp_scylla_poe3_ple_coda32_300 \
bash scripts/run_parcae_scylla_current_best.sh
```

Decision rule: keep investigating only if BPB improves or if latent diagnostics show better final-hidden utilization without worsening recurrent-state scale.

### Validation scorer alignment

The standard `eval_val` path in `train_gpt_parcae.py` was realigned to the canonical `train_gpt.py` scorer body.

Implementation behavior now matches the canonical scorer for:

- sequence partitioning across ranks
- contiguous next-token `x`/`y` construction
- bf16 autocast model loss
- token-byte LUT calculation
- distributed all-reduce order
- BPB formula

The only intentional extra line is the Scylla denominator hook:

```python
if args.val_byte_count_override > 0:
    val_byte_count.fill_(args.val_byte_count_override)
```

That override is applied after byte-count all-reduce. An AST comparison passed with this override line ignored.

### Parallel residual switch

Implemented default-off parallel residual support in `train_gpt_parcae.py`.

Added knobs:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RESIDUAL_MODE` | `sequential` | Keeps the existing sequential residual block unless set to `parallel` |
| `PARALLEL_RESIDUAL_SCOPE` | `none` | Chooses where parallel residuals apply: `none`, `core`, or `all` |
| `PARALLEL_RESIDUAL_START` | `-1` | Physical block index at which blocks become parallel; `-1` means all blocks in scope |
| `PARALLEL_RESIDUAL_LN_SCALE` | `1` | Uses the `curr_record_sub.py` layer-dependent normalized-input scale for record-style residual blocks |

Implementation behavior:

- Default behavior remains the existing sequential block:

```python
x = x + attn(norm_1(x))
x = x + mlp(norm_2(x))
```

- When enabled, the block now follows the `curr_record_sub.py` residual mechanics inside the selected scope:

```python
x_in = resid_mix[0] * x + resid_mix[1] * x0
attn_out = attn(norm_1(x_in) * ln_scale_factor)
if parallel:
    mlp_out = mlp(norm_2(x_in) * ln_scale_factor)
    x = x_in + attn_scale * attn_out + mlp_scale * mlp_out
else:
    x = x_in + attn_scale * attn_out
    x = x + mlp_scale * mlp(norm_2(x) * ln_scale_factor)
```

- `attn_scale`, `mlp_scale`, and `resid_mix` are per-channel FP32 control tensors initialized like `curr_record_sub.py`.
- `RESIDUAL_MODE=parallel PARALLEL_RESIDUAL_SCOPE=core` applies record-style residual controls only to recurrent core blocks; `all` applies them to prelude, core, and coda.
- In Parcae, `x0` is the matching stream's initial state: original outer input embeddings for prelude/coda and the initialized recurrent state for core recurrence.
- `PARALLEL_RESIDUAL_START` uses physical block indices, not recurrence-expanded depth. With the default Parcae shape, prelude is index `0`, core is `1..4`, and coda is `5`.
- `RESIDUAL_MODE=parallel PARALLEL_RESIDUAL_SCOPE=none` is rejected to avoid accidentally believing the feature is enabled when it is not.
- Default `RESIDUAL_MODE=sequential` does not create the extra record-style control tensors, preserving the default architecture.

Older result note:

- The first completed parallel-core run used the simpler GPT-J formula `x + attn(norm(x)) + mlp(norm(x))`, without record-style branch scales, residual mixing, or layer scaling.
- That older result should not be treated as a faithful `curr_record_sub.py` parallel-residual test.

Checks run:

- `.venv/bin/python -m py_compile train_gpt_parcae.py`
- `git diff --check`
- Formula-level unit check confirmed sequential and parallel block equations.
- Fresh-process construction checks confirmed the original simple switch:
  - default: prelude/core/coda all false
  - `core`: prelude false, core true, coda false
  - `all`: prelude/core/coda all true
  - invalid flag combinations fail early
- After the record-style rewrite:
  - default construction has no `attn_scale`, `mlp_scale`, or `resid_mix` tensors.
  - `RESIDUAL_MODE=parallel PARALLEL_RESIDUAL_SCOPE=core PARALLEL_RESIDUAL_START=-1` creates record-style control tensors for core blocks and makes all core blocks parallel.
  - `RESIDUAL_MODE=parallel PARALLEL_RESIDUAL_SCOPE=all PARALLEL_RESIDUAL_START=3` leaves physical blocks before index `3` sequential while still giving selected-scope blocks record-style controls.
  - tiny CUDA forward/backward smoke passed with core record-style parallel residual enabled.

Recommended first experiment:

```bash
RESIDUAL_MODE=parallel \
PARALLEL_RESIDUAL_SCOPE=core \
POE_NUM_EXPERTS=3 \
POE_HEAD_LR=0.002 \
BIGRAM_HASH_BUCKETS=8192 \
TTT_ENABLED=0 \
RUN_ID=exp_scylla_poe3_bigram8192_parallel_core_300 \
bash scripts/run_parcae_scylla_current_best.sh
```

Result:

| Run | Scope | Final exact BPB | Exact loss | Steps | Step avg | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `logs/exp_scylla_poe3_bigram8192_parallel_core_300.txt` | recurrent core only | `1.75860255` | `2.95299381` | 1885 | `159.20ms` | Worse than matched sequential NS5 row norm run |

Matched comparison:

| Run | Residual mode | Final exact BPB | Notes |
| --- | --- | ---: | --- |
| `logs/exp_scylla_poe3_bigram8192_muon_ns5_row1_300_rerun.txt` | sequential | `1.73508599` | Same Scylla/PoE3/bigram8192/current-local 300s setup |
| `logs/exp_scylla_poe3_bigram8192_parallel_core_300.txt` | parallel core | `1.75860255` | `+0.02351656` BPB worse |

Conclusion: core-only parallel residuals hurt in this current-local Scylla 300s setup. The hypothesis that independent attention/MLP deltas would improve recurrent update quality did not hold under this config.

### Muon backend sanity check and QR removal

The temporary Muon backend comparison checked whether exact QR / Gram-Schmidt-style orthogonalization looked better than the existing Newton-Schulz Muon path.

Before interpreting results, Muon correctness was checked against canonical `train_gpt.py`:

| Check | Result |
| --- | --- |
| Parcae Muon with NS5, no row norm, no matrix WD vs canonical `train_gpt.py` Muon | bit-for-bit match |
| Two-step momentum behavior vs canonical `train_gpt.py` Muon | bit-for-bit match |
| Max absolute parameter diff | `0.0` |

Current-local Scylla 300s sweep setup:

| Setting | Value |
| --- | --- |
| `POE_NUM_EXPERTS` | `3` |
| `POE_HEAD_LR` | `0.002` |
| `BIGRAM_HASH_BUCKETS` | `8192` |
| `TTT_ENABLED` | `0` |
| Residuals | sequential; this was not a parallel-residual test |

Results:

| Run | Backend / row norm | Final exact BPB | Step | Notes |
| --- | --- | ---: | ---: | --- |
| `logs/exp_scylla_poe3_bigram8192_muon_ns5_row1_300_rerun.txt` | NS5, row norm on | `1.73508599` | 1863 | Best of this sweep so far |
| `logs/exp_scylla_poe3_bigram8192_muon_ns5_row0_300_rerun.txt` | NS5, row norm off | `1.76854122` | 1875 | Clearly worse than row norm on |
| `logs/exp_scylla_poe3_bigram8192_muon_qr_row0_300_rerun.txt` | QR, row norm off | no final roundtrip; mid/final pre-roundtrip `1.8640` | 1869 | Bad enough to abandon |

Speed read:

- QR was not meaningfully slower than NS5 in this setup: roughly `160.5 ms/step` versus `160.0-161.1 ms/step`.
- QR was much worse on quality, so the issue was not speed.

Conclusion:

- Keep NS5 Muon.
- Keep row-normalized MuonEq-R style path enabled for current-best style runs.
- Remove the temporary QR/SVD backend code and `MUON_BACKEND` knob. `train_gpt_parcae.py` is back to NS5-only, with `MUON_BACKEND_STEPS` still controlling Newton-Schulz iterations.
- Do not compare this current-local 300s sweep directly to the older `1.69172418` headline run; data token count and runtime are not fully apples-to-apples. Within current-local 300s comparisons, NS5 row norm on is around the existing baseline range and row norm off / QR are worse.

### 2026-04-26 diagnostic and architecture triage

This section records analysis and paper-triage notes. These are not new run results unless a log is named explicitly.

BPB scoring contract:

- The current repo evaluator constructs contiguous shifted targets with `x = local[:-1]` and `y = local[1:]`, computes mean token cross-entropy through `model(x, y)`, then reports `val_bpb = val_loss / ln(2) * tokens_per_byte`.
- BPB as a compression concept can be computed from any valid joint probability assignment, but the current code path scores one next-token target per position.
- Multi-token prediction heads would therefore be training auxiliaries unless the evaluation/compression path is rewritten to expose a valid block likelihood. Naive overlapping future-token heads cannot be counted as extra BPB credit because they double-count targets.

Latent diagnostics from existing final checkpoints:

| Run | State | Last-token corr | Recurrent corr | Last eff-rank frac | Recurrent eff-rank frac | Last norm | Recurrent norm | Relative residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sp_no_bigram` | init | 0.6718 | 0.6496 | 0.1761 | 0.1837 | 16.00 | 5.52 | 0.4065 |
| `sp_no_bigram` | trained | 0.1266 | 0.4045 | 0.2966 | 0.3818 | 1.46 | 621.88 | 0.3939 |
| `sp_bigram4096` | trained | 0.1500 | 0.4316 | 0.2644 | 0.4082 | 1.39 | 527.60 | 0.4215 |
| `sp_noloop_bigram4096` | trained | 0.1753 | 0.3325 | 0.3062 | 0.3863 | 1.15 | 85.64 | 1.0000 |
| `scylla_current` | trained | 0.1675 | 0.3842 | 0.2719 | 0.3480 | 1.37 | 140.63 | 0.4158 |

Interpretation:

- There is no evidence of token-collapse degeneracy: token correlations are far from `1.0` and effective rank improves after training.
- There is strong evidence of recurrent scale pathology: recurrent-state norms grow from single digits at init to roughly `85-622`, while final hidden norms are normalized down to roughly `1.1-1.5`.
- The recurrent update is not a no-op: relative residual is roughly `0.39-0.42` in recurrent runs.
- Highest-priority instrumentation gap: these latent monitor metrics exist in code but are not emitted in normal logs.

Output-rank check:

| Run / matrix | Numerical rank | Effective rank frac | Top singular energy | `r99` |
| --- | ---: | ---: | ---: | ---: |
| `sp_no_bigram tok_emb.weight` | 256 | 0.384 | 0.209 | 235 |
| `sp_bigram4096 tok_emb.weight` | 256 | 0.349 | 0.225 | 231 |
| `sp_noloop_bigram4096 tok_emb.weight` | 256 | 0.376 | 0.192 | 229 |
| `scylla_current tok_emb.weight` | 256 | 0.521 | 0.073 | 229 |
| `scylla_current poe_heads.0.weight` | 256 | 0.537 | 0.071 | 229 |
| `scylla_current combined_output_weight` | 256 | 0.520 | 0.078 | 229 |

Interpretation: output matrices are full numerical rank and use most dimensions by energy. The stronger bottleneck is hidden/latent utilization and recurrent scale management, not a degenerate low-rank output head. The mathematical output rank is still capped by model width `256`.

MoE status:

- The tested coda-only top-1 MoE is ruled out for this setup: `logs/exp_scylla_coda_moe4_top1_full_2000.txt` finished at `1.71156502`, worse than PoE3 and dense Scylla comparisons.
- This does not rule out all DeepSeek/Gemma-style MoE. The tested implementation was only a small coda `TopKMoE`, not a shared-expert plus routed-expert architecture with router diagnostics.
- Do not spend the next iteration on MoE unless expert usage entropy, per-expert counts, router temperature/load-balance behavior, active params, and step time are logged.

Paper and architecture applicability:

| Idea | Applicability | Read |
| --- | --- | --- |
| Kimi / Gated Delta attention | Later recurrent-core experiment | Could improve state tracking, but current `seq_len=512` and 300s budget make long-context benefits uncertain. Try only after recurrent scale diagnostics are logged. |
| LBLLM W(1+1)A4 | Artifact compression experiment | Useful idea for staged low-bit weight compression. First try weight-only / W(1+1) offline reconstruction or distillation; activation A4 is a later risk. |
| TensorSLM | Secondary embedding-table compression | Most relevant to `bigram_hash.embed.weight` and `tok_emb.weight`. For `BIGRAM_HASH_BUCKETS=4096`, `tok_emb + bigram_hash.embed` is about `0.78M` params, roughly `16%` of the `4.8M` PoE3 config before int8/zlib. For 8192 buckets, it is about `1.30M` params, roughly `20%` of the `6.37M` run. Needs an offline TT/SVD reconstruction sweep because it may worsen BPB and must beat existing int8+zlib artifact storage. |
| Gemma 4 PLE | Promising as PLE-lite | The transferable idea is controlled per-layer token conditioning. Use zero-init small-rank per-layer token signals, likely coda/core first. Do not add full Gemma-style per-layer embedding tables blindly. |
| Gemma 4 MoE / long context | Low priority | Full Gemma MoE is much more sophisticated than our coda MoE, but local evidence says MoE is not the next lever. Sliding/global attention and p-RoPE target 128K-256K contexts, not the current seq512 path. |
| Multi-token prediction | Possible auxiliary only | Could shape richer representations, but it is not directly scored by the current BPB evaluator. Keep main next-token CE dominant and log main CE separately from aux CE. |

Current belief update:

1. Fix recurrent-state scale first.
2. Emit live latent-rank/norm/correlation diagnostics before running more architecture ideas.
3. Try controlled token reinjection / PLE-lite before MoE.
4. Treat compression papers as artifact-budget tools, not direct BPB improvements, unless an offline reconstructed-weight evaluation shows negligible BPB loss.

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

### PPM vs non-PPM 300s control run

To keep it comparable to prior entries, both runs below use one shared wall-clock budget and identical core settings,
except `PPM_ENABLED` and `SLIDING_WINDOW_ENABLED`.

| Run | `PPM_ENABLED` | Final int8+zlib BPB | Val BPB (corrected) | Final exact loss | Steps | Train time (ms) | int8+zlib bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `logs/exp_compare_parcae_fp10B_base_20260427.txt` | `0` | `1.7077` | `1.7077` | `2.88341485` | `2016` | `300043` | `4,899,938` |
| `logs/exp_compare_parcae_fp10B_ppm_20260427.txt` | `1` | `1.7073` | `1.7073` | `2.88273369` | `2007` | `300022` | `4,899,513` |

Observations:

- PPM enabled run ended on wall-clock at 300s with slightly lower final corrected BPB than non-PPM in this short-budget setup.
- Non-sliding final eval is the same baseline metric used in earlier entries (`final_int8_zlib_roundtrip`).
- The PPM run also reported sliding-window final metrics and ppm mix quality (`mix_bpb:1.52126125`, `ppm_only_bpb:2.36868718`, `nn_only_bpb:1.67512907`) on a 5M-token subset.

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
