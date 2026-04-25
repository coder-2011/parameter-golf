#!/bin/bash
set -euo pipefail
cd /workspace/parameter-golf/runs/exp_mlp4
export DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4
export N_LAYERS_IN_PRELUDE=1 N_LAYERS_IN_RECURRENT_BLOCK=2 N_LAYERS_IN_CODA=1
export RECURRENT_DIM=256 RECURRENT_NUM_HEADS=4 MEAN_RECURRENCE=2 MEAN_BACKPROP_DEPTH=2
export TRAIN_BATCH_TOKENS=16384 TRAIN_SEQ_LEN=512 ITERATIONS=1000000 MAX_WALLCLOCK_SECONDS=300
export WARMUP_STEPS=500 TRAIN_LOG_EVERY=100 VAL_LOSS_EVERY=0 COMPILE_MODEL=0 COMPILE_MUON_BACKEND=0
export ROPE_DIMS=16 QK_NORM=1 USE_VALUE_EMBEDDINGS=0
export BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=128 BIGRAM_HASH_HEADS=2 BIGRAM_HASH_GATE=1
export RUN_ID=exp_mlp4
exec /workspace/parameter-golf/.venv/bin/python /workspace/parameter-golf/train_gpt_parcae.py
