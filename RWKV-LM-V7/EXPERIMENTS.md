# RWKV Experiment Notes

This file is the local ledger for RWKV experiments inside `RWKV-LM-V7`.
Keep entries evidence-first: record the data path, tokenizer, config, checkpoint,
validation command, result, and comparability caveats.

## 2026-04-26 SP1892 FineWeb Setup

Local matched FineWeb SP1892 data was installed in the parent repo:

| Item | Path / value |
| --- | --- |
| Tokenizer | `../data_sp1892/tokenizers/fineweb_1892_bpe.model` |
| Dataset dir | `../data_sp1892/datasets/fineweb10B_sp1892` |
| Train shard | `fineweb_train_000000.bin`, 100,000,000 tokens |
| Validation shard | `fineweb_val_000000.bin`, 53,271,496 tokens |
| Tokenizer vocab | 1892 |

The local export is intentionally capped for iteration: one 100M-token train shard
plus the full first validation shard. It is not the full 10B-token training export.

## 2026-04-26 Baseline RWKV-7 SP1892, 5 Minutes

| Field | Value |
| --- | --- |
| Run dir | `out/fineweb-sp1892-10min-L8-D512-x070` |
| Checkpoint | `out/fineweb-sp1892-10min-L8-D512-x070/rwkv-final.pth` |
| Data | `../data_sp1892/datasets/fineweb10B_sp1892` |
| Data type | `fineweb_u16` |
| Tokenizer | SP1892 |
| Vocab size | 1892 |
| Model | RWKV x070, `n_layer=8`, `n_embd=512`, `dim_att=512`, `dim_ffn=2048` |
| Parameters | about 29.5M |
| Context | 1024 |
| Micro batch | 16 |
| Precision | bf16 |
| Strategy | DeepSpeed stage 2 |
| Compile | enabled |
| LR | `4e-4` to `4e-5`, warmup 50 steps |
| Weight decay | 0.001 |
| Time cap | 300 seconds |
| Exit token cap | disabled for this run, `my_exit_tokens=0` |

Observed training behavior:

- The run completed cleanly on the timed stop and saved `rwkv-final.pth`.
- Terminal output showed roughly 3,285 steps before stop.
- Throughput was roughly 187-192K tokens/s during the stable part of the run.
- GPU memory was roughly 8.2GB on a 32.6GB GPU.
- The last displayed smoothed training loss was around 3.31.

Validation command:

```bash
/usr/bin/python eval_fineweb_bpb.py \
  --load_model out/fineweb-sp1892-10min-L8-D512-x070/rwkv-final.pth \
  --data_file ../data_sp1892/datasets/fineweb10B_sp1892 \
  --tokenizer_path ../data_sp1892/tokenizers/fineweb_1892_bpe.model \
  --vocab_size 1892 \
  --ctx_len 1024 \
  --stride 1024 \
  --micro_bsz 16 \
  --n_layer 8 \
  --n_embd 512
```

Validation result:

| Metric | Value |
| --- | ---: |
| `val_loss` | 2.98638441 |
| `val_bpb` | 1.51916871 |
| Scored tokens | 53,270,528 |
| Scored bytes | 151,078,006 |
| Eval stride | 1024, non-overlapping |
| RoPE | disabled, `rope_mode=none` |

Caveats:

- This run happened before `loss_log.csv` was added, so there is no saved
  per-step loss curve.
- The local SP1892 train data has only one 100M-token train shard; do not compare
  this as if it used the full training export.
- W&B was disabled because the run script passed `--wandb ""`.
- This is not a sliding-window validation result. It is full validation with
  non-overlapping 1024-token scoring windows.

## 2026-04-26 Logging and Evaluation Changes

Added local training loss logging:

- `src/trainer.py` writes `loss_log.csv` in each run directory.
- Rows include `run_timestamp,step,gtokens,loss,avg_loss,lr,weight_decay,kt_s,wall_time`.
- `train.py` exposes `--loss_log_interval`, default `50`, for periodic console loss lines.

Expanded RWKV training telemetry to be closer to the Parcae training logs:

- `src/trainer.py` now also writes `metrics_log.csv` and `metrics_log.jsonl`
  in each run directory.
- Normal-mode step records include loss, average loss, perplexity, LR, weight
  decay, throughput, step time, elapsed time, token counters, and batch/context
  shape.
- `--metrics_log_interval` controls structured local/W&B metric frequency in
  normal mode; default is `50`.
- `--extreme_logging 1` enables every-step structured logging plus heavier
  diagnostics: distributed loss summaries when available and CUDA peak
  allocated/reserved memory.
- Console train lines now use explicit `key:value` fields for easier log grep:
  `step`, `train_loss`, `avg_loss`, `ppl`, `lr`, `kt_s`, `step_time`,
  `elapsed`, `gtokens`, and memory.
- W&B remains optional. When `--wandb` is non-empty, the same metrics are logged
  under flat namespaces such as `train/loss`, `optim/lr`, `throughput/kt_s`, and
  `tokens/gtokens`. Extreme mode also adds fields such as
  `system/gpu_mem_allocated_mb` and `dist/loss_rank_mean`.
- `train.py` now accepts `--wandb_run_name` and `--wandb_mode`; the FineWeb run
  scripts pass these from `WANDB_RUN_NAME` and `WANDB_MODE`, pass
  `WANDB_PROJECT` through to `--wandb`, and expose `METRICS_LOG_INTERVAL` /
  `EXTREME_LOGGING`.

Checks run:

- `python -m py_compile train.py src/trainer.py`
- `git -C RWKV-LM-V7 diff --check -- src/trainer.py train.py run-fineweb-10min.sh demo-training-run-fineweb.sh`
- W&B package import check showed `wandb` is not installed in the current local
  environment, so no real W&B network/offline run was possible without changing
  dependencies.
- Fake-W&B helper check verified `_wandb_log` emits the expected flat metric
  keys.
- Fake-W&B callback check verified `wandb.init(project=..., mode=...,
  name=...)` is reached and local `metrics_log.csv` / `metrics_log.jsonl` are
  created.

Added SP1892 BPB evaluation:

- `eval_fineweb_bpb.py` scores RWKV checkpoints against FineWeb uint16 validation shards.
- BPB is computed as `loss_sum / (ln(2) * scored_bytes)`.
- SentencePiece byte accounting follows the Parcae path: byte pieces count as one
  byte, normal pieces count UTF-8 payload bytes after stripping leading `▁`, and
  leading `▁` contributes a space byte only after non-boundary tokens.

## 2026-04-26 RoPE Experiment Path

Implemented a default-off RoPE path for RWKV time-mix:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--rope_mode` | `none` | Use `rk` to rotate RWKV time-mix `r` and `k` per head |
| `--rope_theta` | `10000.0` | RoPE frequency base |

Implementation details:

- RoPE is applied after the `receptance` and `key` linear projections.
- RoPE is applied before `fused_k_rwkv7`, so both the normalized key direction
  and the updated key entering the RWKV7 kernel carry positional rotation.
- The inverse-frequency buffer is non-persistent, so enabling RoPE adds no
  checkpoint tensors.
- The FineWeb run scripts accept `ROPE_MODE=rk`; output dirs get `-roperk` by
  default when enabled.

Example:

```bash
ROPE_MODE=rk ./run-fineweb-10min.sh
```

Checks run:

- `python -m py_compile src/model.py train.py eval_fineweb_bpb.py`
- `python -m unittest tests.test_eval_fineweb_bpb tests.test_rope`
- Tiny CUDA train smoke with `--rope_mode rk` completed a forward/backward step.
- One-window eval load worked for the existing no-RoPE checkpoint with
  `rope_mode=none`.

No full RoPE BPB run has been completed yet.

## 2026-04-26 Optional Embedding / Head Weight Tying

Implemented default-off RWKV embedding/head tying:

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--tie_embeddings` | `0` | When `1`, `head.weight` reuses `emb.weight` |

Implementation details:

- Tying is opt-in, so existing untied runs and checkpoints keep their behavior.
- In tied mode, `RWKV.__init__` assigns `self.head.weight = self.emb.weight`.
- PyTorch then exposes only one unique named parameter for optimizer grouping,
  while the state dict still accepts both `emb.weight` and `head.weight` keys.
- Loading an old untied state dict into a tied model works strictly; because
  both checkpoint keys map into the same parameter, the later `head.weight`
  value becomes the shared value.
- `run-fineweb-10min.sh` and `demo-training-run-fineweb.sh` expose
  `TIE_EMBEDDINGS=1`.
- `eval_fineweb_bpb.py` exposes `--tie_embeddings` so tied checkpoints can be
  evaluated with matching model construction.

Checks run:

- `python -m py_compile train.py src/model.py eval_fineweb_bpb.py tests/test_weight_tying.py`
- `python -m unittest tests.test_weight_tying`
- `python -m unittest tests.test_norms tests.test_rope tests.test_weight_tying`
