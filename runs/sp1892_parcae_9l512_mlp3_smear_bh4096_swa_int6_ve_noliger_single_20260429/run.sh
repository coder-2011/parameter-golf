#!/usr/bin/env bash
set -euo pipefail

export RUN_ID=sp1892_parcae_9l512_mlp3_smear_bh4096_swa_int6_ve_noliger_single_20260429
export DATA_PATH=/workspace/parameter-golf/data_sp1892/datasets/fineweb10B_sp1892
export TOKENIZER_PATH=/workspace/parameter-golf/data_sp1892/tokenizers/fineweb_1892_bpe.model
export VOCAB_SIZE=1892
export MAX_WALLCLOCK_SECONDS=300

export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export N_LAYERS_IN_PRELUDE=4
export N_LAYERS_IN_RECURRENT_BLOCK=1
export N_LAYERS_IN_CODA=4
export RECURRENT_DIM=512
export RECURRENT_INTERMEDIATE_DIM=1536
export RECURRENT_NUM_HEADS=8
export MEAN_RECURRENCE=1
export MEAN_BACKPROP_DEPTH=1
export STATE_INIT=like-init

export TRAIN_BATCH_TOKENS=16384
export TRAIN_SEQ_LEN=512
export GRAD_ACCUM_STEPS=1
export ITERATIONS=1000000
export WARMUP_STEPS=200
export TRAIN_LOG_EVERY=100
export VAL_LOSS_EVERY=0

export COMPILE_MODEL=1
export COMPILE_MUON_BACKEND=1
export LIGER_CE=0
export RESIDUAL_MODE=parallel
export PARALLEL_RESIDUAL_SCOPE=all
export PARALLEL_RESIDUAL_IMPL=delayed
export PARALLEL_RESIDUAL_RECORD_CONTROLS=0
export PARALLEL_RESIDUAL_TIED_NORM=1
export PARALLEL_RESIDUAL_IN_FP32=1
export XSA_LAST_N=0
export ROPE_DIMS=16
export QK_NORM=1
export USE_VALUE_EMBEDDINGS=1

export BIGRAM_HASH_BUCKETS=4096
export BIGRAM_HASH_DIM=128
export BIGRAM_HASH_HEADS=1
export BIGRAM_HASH_GATE=0
export BIGRAM_HASH_SCALE_INIT=0.05

export QAT_BITS=0
export GPTQ_ENABLED=0
export QUANT_BITS=6
export QUANT_KEEP_FLOAT_PATTERNS=tok_emb.weight,bigram_hash.embed.weight

export EMA_ENABLED=0
export SWA_ENABLED=1
export SWA_START_FRAC=0.5
export SWA_EVERY=50

export MUON_WD=0.04
export MUON_ROW_NORMALIZE=1
export MATRIX_LR=0.02
export SCALAR_LR=0.02
export TIED_EMBED_LR=0.03

cd "/workspace/parameter-golf/runs/sp1892_parcae_9l512_mlp3_smear_bh4096_swa_int6_ve_noliger_single_20260429"
exec /workspace/parameter-golf/.venv/bin/python /workspace/parameter-golf/train_gpt_parcae.py
