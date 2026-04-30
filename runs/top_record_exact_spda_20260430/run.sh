#!/usr/bin/env bash
set -euo pipefail

export RUN_ID=${RUN_ID:-top_record_exact_spda_20260430}
export DATA_PATH=${DATA_PATH:-/workspace/parameter-golf/data/datasets/fineweb10B_sp1024}
export TOKENIZER_PATH=${TOKENIZER_PATH:-/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model}
export VOCAB_SIZE=${VOCAB_SIZE:-1024}

# Keep the top-record recipe, but scale runtime shape for this single RTX PRO
# 4500. The record script itself is unchanged except for optional SDPA fallback
# when FlashAttention-3 is unavailable.
export SEED=${SEED:-314}
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}
export ITERATIONS=${ITERATIONS:-20000}
export WARMUP_STEPS=${WARMUP_STEPS:-20}
export WARMDOWN_ITERS=${WARMDOWN_ITERS:-4000}
export TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-20}
export VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-0}

export NUM_LAYERS=${NUM_LAYERS:-11}
export MODEL_DIM=${MODEL_DIM:-512}
export NUM_HEADS=${NUM_HEADS:-8}
export NUM_KV_HEADS=${NUM_KV_HEADS:-4}
export MLP_MULT=${MLP_MULT:-3.0}
export BIGRAM_VOCAB_SIZE=${BIGRAM_VOCAB_SIZE:-3072}
export BIGRAM_DIM=${BIGRAM_DIM:-112}
export XSA_LAST_N=${XSA_LAST_N:-11}
export ROPE_DIMS=${ROPE_DIMS:-16}
export LN_SCALE=${LN_SCALE:-1}
export VE_ENABLED=${VE_ENABLED:-1}
export VE_DIM=${VE_DIM:-128}
export VE_LAYERS=${VE_LAYERS:-9,10}

export TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-1024}
export EVAL_SEQ_LEN=${EVAL_SEQ_LEN:-1024}
export TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-32768}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-65536}
export EVAL_STRIDE=${EVAL_STRIDE:-64}
export COMPILE_MODEL=${COMPILE_MODEL:-1}
export COMPILE_FULLGRAPH=${COMPILE_FULLGRAPH:-1}

export MATRIX_LR=${MATRIX_LR:-0.025}
export SCALAR_LR=${SCALAR_LR:-0.025}
export TIED_EMBED_LR=${TIED_EMBED_LR:-0.035}
export MUON_MOMENTUM=${MUON_MOMENTUM:-0.99}
export MUON_MOMENTUM_WARMUP_START=${MUON_MOMENTUM_WARMUP_START:-0.92}
export MUON_MOMENTUM_WARMUP_STEPS=${MUON_MOMENTUM_WARMUP_STEPS:-1500}
export MUON_WD=${MUON_WD:-0.04}
export ADAM_WD=${ADAM_WD:-0.04}
export GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-0.3}
export SWA_ENABLED=${SWA_ENABLED:-1}
export SWA_EVERY=${SWA_EVERY:-50}
export QAT_ENABLED=${QAT_ENABLED:-0}
export TARGET_MB=${TARGET_MB:-15.9}

cd /workspace/parameter-golf/runs/top_record_exact_spda_20260430
exec /workspace/parameter-golf/.venv/bin/python \
  /workspace/parameter-golf/records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py
