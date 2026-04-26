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

## 2026-04-25 Parcae score-first TTT implementation

`train_gpt_parcae.py` has an opt-in `TTT_ENABLED=1` quantized roundtrip eval path that scores validation chunks first with sliding windows, then trains on the scored chunk via SGD. Defaults: `TTT_CHUNK_TOKENS=32768`, `TTT_EPOCHS=3`, `TTT_GRAD_CLIP=1.0`, `EVAL_STRIDE=64`.

Implementation caveat: the final chunk is not trained because all of its loss has already been scored. The logits path is intentionally uncompiled for robustness; add a separate compile knob only if profiling shows it is worth the startup/runtime risk.
