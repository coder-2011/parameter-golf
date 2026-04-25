# Experiment Notes

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
