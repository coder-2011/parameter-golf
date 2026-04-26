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

Performance caveat: no full Scylla TTT run has proven this helps. Quantized-weight SGD TTT remains experimental and may hurt.
