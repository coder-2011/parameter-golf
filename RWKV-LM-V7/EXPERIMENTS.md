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

## 2026-04-27 Muon Optimizer Trial, 5 Minutes

| Field | Value |
| --- | --- |
| Run dir | `out/fineweb-sp1892-muon-5min-L8-D512-x070-r2` |
| Checkpoint | `out/fineweb-sp1892-muon-5min-L8-D512-x070-r2/rwkv-final.pth` |
| Base config | Same SP1892 FineWeb, RWKV x070 8x512, ctx 1024, bf16 as baseline |
| Optimizer | `--optimizer muon` |
| Matrix optimizer | `torch.optim.Muon`, `muon_momentum=0.95`, `muon_wd=0.001` |
| Non-matrix optimizer | AdamW fallback groups using `adam_eps=1e-18` |
| Strategy | `auto`, single GPU; DeepSpeed is disabled for Muon |
| Time cap | 300 seconds |
| Exit token cap | disabled for this run, `my_exit_tokens=0` |

Training command:

```bash
python3 train.py \
  --data_file ../data_sp1892/datasets/fineweb10B_sp1892 \
  --data_type fineweb_u16 \
  --vocab_size 1892 \
  --n_layer 8 \
  --n_embd 512 \
  --ctx_len 1024 \
  --micro_bsz 16 \
  --optimizer muon \
  --strategy auto \
  --my_exit_seconds 300 \
  --my_exit_tokens 0 \
  --proj_dir out/fineweb-sp1892-muon-5min-L8-D512-x070-r2
```

Validation command:

```bash
python3 eval_fineweb_bpb.py \
  --load_model out/fineweb-sp1892-muon-5min-L8-D512-x070-r2/rwkv-final.pth \
  --rope_mode none
```

Validation result:

| Metric | Value |
| --- | ---: |
| `val_loss` | 3.33134217 |
| `val_bpb` | 1.69464814 |
| Scored tokens | 53,270,528 |
| Scored bytes | 151,078,006 |
| Eval stride | 1024, non-overlapping |

Caveats:

- Muon was worse than the no-RoPE AdamW baseline in this 5-minute trial
  (`1.69464814` vs `1.51916871` BPB).
- The run uses Lightning manual optimization with multiple optimizers. Logged
  `global_step`/token counters appear inflated because each optimizer step can
  advance Lightning's global step. The wall-clock stop and checkpoint are valid,
  but Muon token/step logs should not be compared directly until this accounting
  is fixed.
- `--adam_eps 1e-18` still matters in Muon mode because embeddings, output
  head, vectors/scalars, norms, and gain-like tensors stay on AdamW; only hidden
  2D matrix parameters use Muon.

## 2026-04-27 RMSNorm Trial, 5 Minutes

| Field | Value |
| --- | --- |
| Run dir | `out/fineweb-sp1892-rmsnorm-10min-L8-D512-x070` |
| Checkpoint | `out/fineweb-sp1892-rmsnorm-10min-L8-D512-x070/rwkv-final.pth` |
| Base config | Same SP1892 FineWeb, RWKV x070 8x512, ctx 1024, bf16 as baseline |
| Norm | `--norm_type rmsnorm` |
| RoPE | disabled, `rope_mode=none` |
| Optimizer | AdamW |
| Strategy | DeepSpeed stage 2 |
| Time cap | 300 seconds |
| Exit token cap | disabled for this run, `my_exit_tokens=0` |

Training command:

```bash
NORM_TYPE=rmsnorm ROPE_MODE=none MY_EXIT_SECONDS=300 MY_EXIT_TOKENS=0 \
  ./run-fineweb-10min.sh
```

Validation command:

```bash
python3 eval_fineweb_bpb.py \
  --load_model out/fineweb-sp1892-rmsnorm-10min-L8-D512-x070/rwkv-final.pth \
  --norm_type rmsnorm \
  --rope_mode none
```

Validation result:

| Metric | Value |
| --- | ---: |
| Final timed step | 3208 |
| Train avg loss | 3.31999591 |
| `val_loss` | 2.98938593 |
| `val_bpb` | 1.52069558 |
| Scored tokens | 53,270,528 |
| Scored bytes | 151,078,006 |
| Eval stride | 1024, non-overlapping |

Caveats:

- RMSNorm was slightly worse than the LayerNorm baseline in this comparable
  5-minute trial (`1.52069558` vs `1.51916871` BPB).
- This is close enough that random initialization noise may matter, but it is
  not an immediate improvement on the current best short-run result.

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

Completed 5-minute RoPE comparison runs:

| Variant | Run dir | RoPE dims / head | Final timed step | Train avg loss | Validation BPB |
| --- | --- | ---: | ---: | ---: | ---: |
| Full RoPE | `out/fineweb-sp1892-roperk-10min-L8-D512-x070` | 64 / 64, 100% | 3105 | about 3.336 | 1.53729809 |
| Partial RoPE | `out/fineweb-sp1892-roperkd16-10min-L8-D512-x070` | 16 / 64, 25% | 2907 | 3.34981295 | 1.53697412 |

Full RoPE validation command:

```bash
python3 eval_fineweb_bpb.py \
  --load_model out/fineweb-sp1892-roperk-10min-L8-D512-x070/rwkv-final.pth \
  --rope_mode rk
```

Partial RoPE validation command:

```bash
python3 eval_fineweb_bpb.py \
  --load_model out/fineweb-sp1892-roperkd16-10min-L8-D512-x070/rwkv-final.pth \
  --rope_mode rk \
  --rope_dims 16
```

Partial RoPE result:

| Metric | Value |
| --- | ---: |
| `val_loss` | 3.02138632 |
| `val_bpb` | 1.53697412 |
| Scored tokens | 53,270,528 |
| Scored bytes | 151,078,006 |
| Eval stride | 1024, non-overlapping |
| RoPE | `rope_mode=rk`, `rope_dims=16`, `rope_theta=10000` |

Caveats:

- Partial RoPE at 25% was slightly better than full RoPE in this short run, but
  both were worse than the no-RoPE baseline (`val_bpb=1.51916871`).
- This comparison is noisy because these are independent random initializations
  and 5-minute training runs.
- The partial run reached fewer timed-stop steps than the previous full-RoPE run
  (`2907` vs `3105`), likely due to transient throughput variation near the end.

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
- Deep CPU-safe behavior script covering missing-flag default-off construction,
  explicit untied construction, tied construction, Muon grouping, strict
  untied-to-tied checkpoint load, strict tied-to-untied checkpoint load, and
  disk state-dict roundtrip.
- Fresh-process CUDA forward/backward check with `RWKV_MY_TESTING=x070`,
  `RWKV_HEAD_SIZE=64`, tied embeddings enabled, and finite shared embedding/head
  gradient.
- Default-off checks: `train.py --help`, `eval_fineweb_bpb.py --help`,
  `TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-0}"` in both FineWeb run scripts, and
  `bash -n run-fineweb-10min.sh demo-training-run-fineweb.sh`.

## 2026-04-27 Fixed-Step LR Cooldown

Added a default-off fixed-step LR cooldown path for runs that do not use the
existing token-count LR schedule.

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--cooldown_steps` | `0` | Number of final epoch steps over which LR decays from `lr_init` to `lr_final` |

Implementation details:

- `scheduled_lr(args, step)` centralizes LR computation in `src/trainer.py`.
- Existing warmup behavior is preserved.
- Existing token-based cosine schedule remains higher priority when
  `--my_exit_tokens != 0`.
- When `--my_exit_tokens 0` and `--cooldown_steps > 0`, LR stays at `lr_init`
  until `epoch_steps - cooldown_steps`, then cosine-decays to `lr_final`.
- `run-fineweb-10min.sh` and `demo-training-run-fineweb.sh` expose
  `COOLDOWN_STEPS`, defaulting to `0`.

Checks run:

- `python -m py_compile train.py src/trainer.py tests/test_lr_schedule.py`
- `python -m unittest tests.test_lr_schedule`
- `python -m unittest tests.test_norms tests.test_rope tests.test_weight_tying tests.test_lr_schedule`
- `bash -n run-fineweb-10min.sh demo-training-run-fineweb.sh`
- `python train.py --help | rg -n -- "--cooldown_steps"`

## 2026-04-27 Deep/Narrow Scaled-Down RWKV

Tested a scaled-down deep/narrow shape after `L29-D512` proved too slow for the
5-minute local budget.

| Field | Value |
| --- | ---: |
| Shape | `L16-D384`, `dim_att=384`, `dim_ffn=1536`, `head_size=64` |
| Params | 33,028,608 |
| Norm / RoPE | LayerNorm, no RoPE |
| Runtime cap | 300s |
| Final step | 1,928 |
| Tokens | 31,588,352 |
| Avg train loss | 3.54086975 |
| Throughput near end | ~120 Ktok/s |
| Checkpoint | `out/fineweb-sp1892-deepnarrow-L16-D384-5min-x070/rwkv-final.pth` |
| Checkpoint size | 64M |
| Eval val_loss | 3.13026364 |
| Eval val_bpb | 1.59235983 |

Notes:

- The interrupted `L29-D512` attempt was about 102M params and ran around
  54-55 Ktok/s, so it was not a useful 5-minute local-budget shape.
- `L16-D384` was much faster, but still scored worse than the 8x512 LayerNorm
  baseline (`val_bpb=1.51916871`).
- This result suggests simply making RWKV deeper and narrower is not currently
  helping under the short local training budget.

## 2026-04-27 SDPA Hybrid Attention Path

Implemented a default-off hybrid architecture path where selected nonzero RWKV
layers replace `RWKV_Tmix_x070` with a PyTorch SDPA causal self-attention anchor.

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--attn_every` | `0` | Disabled when `0`; otherwise select every Nth nonzero layer |
| `--attn_offset` | `0` | First selected layer; `0` maps to `attn_every` in train/eval entrypoints |
| `--attn_heads` | `0` | Derived from `attn_dim / head_size` when `0` |
| `--attn_dim` | `0` | Uses `n_embd` when `0` |
| `--attn_dropout` | `0.0` | SDPA dropout during training |
| `--attn_rope` | `1` | Apply RoPE to attention q/k |

Implementation details:

- Layer 0 always remains RWKV so `v_first` is initialized for later recurrent
  layers.
- Attention layers preserve the existing block contract by returning
  `(x_attn, v_first)`.
- Attention output projection is zero-initialized, matching the residual-safe
  style used elsewhere in the RWKV blocks.
- `run-fineweb-10min.sh`, `demo-training-run-fineweb.sh`, and
  `eval_fineweb_bpb.py` all expose matching flags.

Checks run:

- `python3 -m py_compile train.py src/model.py eval_fineweb_bpb.py tests/test_hybrid_attention.py`
- `bash -n run-fineweb-10min.sh demo-training-run-fineweb.sh`
- `python3 -m unittest tests.test_hybrid_attention tests.test_rope tests.test_norms tests.test_weight_tying`
- Tiny train smoke:
  `N_LAYER=2 N_EMBD=128 CTX_LEN=128 M_BSZ=2 ATTN_EVERY=1 ATTN_OFFSET=1 MY_EXIT_TOKENS=8192 PROJ_DIR=out/smoke-hybrid-attn ./demo-training-run-fineweb.sh`
- Tiny eval smoke:
  `python3 eval_fineweb_bpb.py --load_model out/smoke-hybrid-attn/rwkv-final.pth --n_layer 2 --n_embd 128 --dim_att 128 --dim_ffn 512 --ctx_len 128 --micro_bsz 2 --attn_every 1 --attn_offset 1 --attn_rope 1 --max_spans 2`

Next comparable run should start with one attention anchor in the 8x512 baseline
shape, for example `ATTN_EVERY=4 ATTN_OFFSET=4`, then evaluate with matching
`--attn_every 4 --attn_offset 4`.

## 2026-04-27 Hybrid Attention Anchor Run

Ran the first comparable SDPA hybrid trial: 8x512 baseline shape with one
attention anchor at layer 4.

| Field | Value |
| --- | ---: |
| Shape | `L8-D512`, `dim_att=512`, `dim_ffn=2048`, `head_size=64` |
| Hybrid config | `attn_every=4`, `attn_offset=4`, `attn_rope=1` |
| Attention layers | layer 4 only |
| Params | 29,240,320 |
| Runtime cap | 300s |
| Final step | 3,375 |
| Tokens | 55,296,000 |
| Avg train loss | 3.28638426 |
| Throughput near end | ~199 Ktok/s |
| Checkpoint | `out/fineweb-sp1892-attne4o4-5min-L8-D512-x070/rwkv-final.pth` |
| Checkpoint size | 56M |
| Eval val_loss | 2.95717080 |
| Eval val_bpb | 1.50430779 |

Comparison:

- Previous 8x512 LayerNorm/no-RoPE baseline: `val_bpb=1.51916871`.
- Hybrid attention anchor improvement: `-0.01486092` BPB.
- This is the first local RWKV architectural change in this series that clearly
  beat the no-RoPE baseline under the same 300s-style budget.

Eval command:

```bash
python3 eval_fineweb_bpb.py \
  --load_model out/fineweb-sp1892-attne4o4-5min-L8-D512-x070/rwkv-final.pth \
  --n_layer 8 --n_embd 512 --dim_att 512 --dim_ffn 2048 \
  --attn_every 4 --attn_offset 4 --attn_rope 1 \
  --rope_mode none --norm_type layernorm
```

## 2026-04-27 Partial RoPE for SDPA Attention

Extended the SDPA hybrid attention path so attention RoPE can also be partial.
It uses the existing `rope_dims` setting:

- `rope_dims=0`: rotate the full attention head dimension.
- `rope_dims=N`: rotate only the first `N` dimensions of each attention head.

Notes:

- This applies when `attn_rope=1`.
- `rope_dims` must be positive, even, and no larger than the attention head
  dimension after default resolution.
- The same flag still controls RWKV time-mix RoPE when `rope_mode != none`, so
  a run combining RWKV RoPE and attention RoPE uses one shared partial dimension.

Checks run:

- `python3 -m py_compile src/model.py tests/test_hybrid_attention.py train.py eval_fineweb_bpb.py`
- `python3 -m unittest tests.test_hybrid_attention tests.test_rope`

Follow-up thorough check:

- Fixed `demo-training-run-fineweb.sh` so it passes `ROPE_DIMS`; before this,
  demo partial-attention-RoPE smokes would have silently used full attention
  RoPE.
- Updated run-directory suffixes to include attention partial-RoPE dims when
  attention RoPE is enabled.
- Updated eval output to print `rope_dims` and `attn_rope`.
- Re-ran static checks, 23 focused tests, a CUDA train smoke with
  `ATTN_EVERY=1 ATTN_OFFSET=1 ATTN_ROPE=1 ROPE_DIMS=16`, and a matching eval
  smoke that loaded the checkpoint and reported `rope_dims:16 attn_rope:1`.
