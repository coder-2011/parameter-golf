#!/usr/bin/env bash
set -euo pipefail

export RUN_ID=sp1892_single_gpu_9l512_mlp3_bh_swa_int6_196k_ga2_20260428_233527
export DATA_PATH=/workspace/parameter-golf/data_sp1892/datasets/fineweb10B_sp1892
export TOKENIZER_PATH=/workspace/parameter-golf/data_sp1892/tokenizers/fineweb_1892_bpe.model
export VOCAB_SIZE=1892

export MAX_WALLCLOCK_SECONDS=300
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=100

export NUM_LAYERS=9
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export TIE_EMBEDDINGS=1

export TRAIN_BATCH_TOKENS=196608
export TRAIN_SEQ_LEN=2048
export GRAD_ACCUM_STEPS=2
export VAL_BATCH_SIZE=262144

export WARMUP_STEPS=20
export WARMDOWN_ITERS=3000
export MATRIX_LR=0.02
export SCALAR_LR=0.02
export TIED_EMBED_LR=0.03
export MUON_WD=0.04
export ADAM_WD=0.01
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export GRAD_CLIP_NORM=0.3

export BIGRAM_VOCAB_SIZE=4096
export BIGRAM_DIM=128
export SWA_ENABLED=1
export SWA_START_FRAC=0.5
export SWA_EVERY=50
export EVAL_STRIDE=64

cd "/workspace/parameter-golf/runs/sp1892_single_gpu_9l512_mlp3_bh_swa_int6_196k_ga2_20260428_233527"
exec /workspace/parameter-golf/.venv/bin/python train_gpt_single_gpu.py
