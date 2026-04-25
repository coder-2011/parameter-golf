# Experiment Notes

## Current Best Model

This is our best Parcae model so far.

| Field | Value |
| --- | --- |
| Exact final BPB | **1.76239973** |
| Exact final loss | **2.97573812** |
| Run log | `logs/parcae_min_5min_best_no_ddp_no_value_kv2_qknorm_rope16_20260425.txt` |
| Retrieval diagnostic | `logs/diag_parcae_best_no_ddp_no_value_kv2_qknorm_rope16_20260425.json` |
| Exported artifacts | Current `final_model.pt` and `final_model.int8.ptz` |
| Launch mode | Direct `python train_gpt_parcae.py` with no one-rank DDP wrapper |
| Key config | `USE_VALUE_EMBEDDINGS=0 NUM_KV_HEADS=2 ROPE_DIMS=16 QK_NORM=1` |
| Steps / measured train time | 1785 steps / 300166 ms |
| Params | 2,886,592 |
| Total int8+zlib submission size | 3,328,383 bytes |

Command shape used for the current best:

```bash
RUN_ID=parcae_min_5min_best_no_ddp_no_value_kv2_qknorm_rope16_20260425 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MODEL_DIM=256 \
NUM_HEADS=4 \
NUM_KV_HEADS=2 \
N_LAYERS_IN_PRELUDE=1 \
N_LAYERS_IN_RECURRENT_BLOCK=2 \
N_LAYERS_IN_CODA=1 \
RECURRENT_DIM=256 \
RECURRENT_NUM_HEADS=4 \
MEAN_RECURRENCE=2 \
MEAN_BACKPROP_DEPTH=2 \
TRAIN_BATCH_TOKENS=16384 \
TRAIN_SEQ_LEN=256 \
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
python train_gpt_parcae.py
```

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
| No value embeddings + outer KV heads 2 + QK norm + RoPE 8 | `logs/parcae_min_5min_no_value_embeds_kvheads2_qknorm_rope8_20260425.txt` | 1.79321577 | 1697 | 2,886,592 | 3,314,402 bytes | `logs/diag_parcae_no_value_embeds_kvheads2_qknorm_rope8_20260425.json` |
| No value embeddings + outer KV heads 2 + QK norm + RoPE 32 | `logs/parcae_min_5min_no_value_embeds_kvheads2_qknorm_rope32_20260425.txt` | 1.76441375 | 1714 | 2,886,592 | 3,322,038 bytes | `logs/diag_parcae_no_value_embeds_kvheads2_qknorm_rope32_20260425.json` |
| No value embeddings + outer KV heads 2 + QK norm + RoPE 32, direct Python | `logs/parcae_min_5min_no_ddp_no_value_kv2_qknorm_rope32_20260425.txt` | 1.76400278 | 1723 | 2,886,592 | 3,324,306 bytes | `logs/diag_parcae_no_ddp_no_value_kv2_qknorm_rope32_20260425.json` |

Best current clean result: direct `python train_gpt_parcae.py` with `USE_VALUE_EMBEDDINGS=0 NUM_KV_HEADS=2 ROPE_DIMS=16 QK_NORM=1`, with exact final `val_bpb=1.76239973`.

Observed diagnostic pattern:

- Removing value embeddings improved both speed and quality in this local 5-minute regime.
- Increasing outer KV heads from 1 to 2 helped both with and without value embeddings. The stacked no-value + KV2 + QK-norm variant was best, and running it directly with `python train_gpt_parcae.py` was faster than one-rank `torch.distributed.run`.
- QK norm helped with value embeddings on, hurt for no-value/KV1, but helped again for no-value/KV2.
- Larger recurrent RoPE did not help. `RECURRENT_ROPE_DIMS=32` was the worst run, and making only `core_block.1` full-RoPE with QK norm also lost to QK norm alone. In the best no-value/KV2/QK-norm setup, `ROPE_DIMS=8` was much worse and `ROPE_DIMS=32` was close but still worse than `ROPE_DIMS=16`, including under direct Python.
- The best run still mostly improves local/recent-context behavior rather than becoming an obvious exact-copy retrieval model. On the 64-sequence diagnostic slice, seen-last-32 loss improved from 2.4584 in the baseline to 2.3411 in the best direct-Python run, while leading-space loss improved from 4.3271 to 4.2782.
Attempted `NUM_KV_HEADS=4` direct Python (`logs/parcae_min_5min_no_ddp_no_value_kv4_qknorm_rope16_20260425.txt`), but the process exited before final validation after step 500. Early train loss and step time were worse than KV2, so this did not look promising enough to rerun immediately.

