#!/usr/bin/env bash
set -euo pipefail

export RUN_ID=rep_autoresearch_20260428_033810_batch262k_ga2_ligerce_20260429
export DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export MAX_WALLCLOCK_SECONDS=600

export MODEL_DIM=384
export RECURRENT_DIM=384
export RECURRENT_INTERMEDIATE_DIM=0
export NUM_HEADS=4
export NUM_KV_HEADS=2
export RECURRENT_NUM_HEADS=4
export N_LAYERS_IN_PRELUDE=1
export N_LAYERS_IN_RECURRENT_BLOCK=3
export N_LAYERS_IN_CODA=2
export MLP_MULT=4
export MEAN_RECURRENCE=2
export MEAN_BACKPROP_DEPTH=1
export STATE_INIT=like-init

export TRAIN_SEQ_LEN=1024
export TRAIN_BATCH_TOKENS=262144
export GRAD_ACCUM_STEPS=2
export ITERATIONS=1000000
export WARMUP_STEPS=500
export TRAIN_LOG_EVERY=100
export VAL_LOSS_EVERY=0
export VAL_BATCH_SIZE=524288

export COMPILE_MODEL=1
export COMPILE_MUON_BACKEND=1
export LIGER_CE=1
export TRIDAO_PACKED_ROPE=1
export LIGER_ROPE=0
export RESIDUAL_MODE=sequential
export PARALLEL_RESIDUAL_SCOPE=none
export PARALLEL_RESIDUAL_IMPL=immediate
export PARALLEL_RESIDUAL_RECORD_CONTROLS=1
export ROPE_DIMS=32
export QK_NORM=1
export XSA_LAST_N=0
export USE_VALUE_EMBEDDINGS=0

export BIGRAM_HASH_BUCKETS=8192
export BIGRAM_HASH_DIM=128
export BIGRAM_HASH_HEADS=2
export BIGRAM_HASH_GATE=1
export BIGRAM_HASH_SCALE_INIT=0.05
export BIGRAM_HASH_INIT_STD=0.02

export POE_NUM_EXPERTS=3
export POE_HEAD_LR=0.005
export QAT_BITS=0
export GPTQ_ENABLED=0
export QUANT_BITS=8
export SAVE_RAW_MODEL=0

export EMA_ENABLED=0
export SWA_ENABLED=1
export SWA_START_FRAC=0.2
export SWA_EVERY=50

export TIED_EMBED_LR=0.05
export TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.04
export SCALAR_LR=0.04
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=500
export MUON_BACKEND_STEPS=5
export MUON_ROW_NORMALIZE=1
export MUON_WD=0.095
export GRAD_CLIP_NORM=0.3

export SLIDING_WINDOW_ENABLED=0
export EVAL_STRIDE=64
export TTT_ENABLED=0
export PPM_ENABLED=0
export LZP_ENABLED=0
export NGRAM_EVAL_ORDER=0
export SEED=1337

cd /workspace/parameter-golf/runs/rep_autoresearch_20260428_033810_batch262k_ga2_ligerce_20260429
exec /workspace/parameter-golf/.venv/bin/python /workspace/parameter-golf/train_gpt_parcae.py
