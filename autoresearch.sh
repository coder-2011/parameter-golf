#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Unique run ID per invocation
RUN_ID="autoresearch_$(date +%Y%m%d_%H%M%S)_$$"
LOGFILE="logs/${RUN_ID}.txt"

echo "=== Autoresearch run: ${RUN_ID} ===" >&2

# Fast syntax check before GPU burn
python -m py_compile train_gpt_parcae.py 2>&1 | tail -5

# Baseline config (SP1024, 300s wall-clock, 1x GPU)
export RUN_ID
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export MODEL_DIM=384
export RECURRENT_DIM=384
export NUM_HEADS=4
export NUM_KV_HEADS=2
export RECURRENT_NUM_HEADS=4
export N_LAYERS_IN_PRELUDE=1
export N_LAYERS_IN_RECURRENT_BLOCK=3
export N_LAYERS_IN_CODA=2
export MLP_MULT=4
export TRAIN_SEQ_LEN=1024
export TRAIN_BATCH_TOKENS=32768
export ITERATIONS=1000000
export MAX_WALLCLOCK_SECONDS=300
export WARMUP_STEPS=500
export TRAIN_LOG_EVERY=100
export VAL_LOSS_EVERY=0
export MEAN_RECURRENCE=2
export MEAN_BACKPROP_DEPTH=1
export ROPE_DIMS=32
export QK_NORM=1
export USE_VALUE_EMBEDDINGS=0
export BIGRAM_HASH_BUCKETS=8192
export BIGRAM_HASH_DIM=128
export BIGRAM_HASH_HEADS=2
export BIGRAM_HASH_GATE=1
export COMPILE_MODEL=1
export COMPILE_MUON_BACKEND=0
export POE_NUM_EXPERTS=1
export GRAD_CLIP_NORM=0.3
export SEED=1337

# Run training
"${PYTHON:-/workspace/parameter-golf/.venv/bin/python}" train_gpt_parcae.py

# Parse exact roundtrip BPB and other metrics from log
if [[ ! -f "${LOGFILE}" ]]; then
    echo "ERROR: log file not found: ${LOGFILE}" >&2
    exit 1
fi

VAL_BPB=$(grep 'final_int8_zlib_roundtrip_exact val_loss:' "${LOGFILE}" | tail -n1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/' || echo "")
VAL_LOSS=$(grep 'final_int8_zlib_roundtrip_exact val_loss:' "${LOGFILE}" | tail -n1 | sed -E 's/.*val_loss:([0-9.]+).*/\1/' || echo "")
TRAIN_TIME=$(grep 'stopping_early: wallclock_cap' "${LOGFILE}" | tail -n1 | sed -E 's/.*train_time:([0-9.]+)ms.*/\1/' || echo "")
STEPS=$(grep 'stopping_early: wallclock_cap' "${LOGFILE}" | tail -n1 | sed -E 's/.*step:([0-9]+)\/.*/\1/' || echo "")
STEP_AVG=$(grep 'stopping_early: wallclock_cap' "${LOGFILE}" | tail -n1 | sed -E 's/.*step_avg:([0-9.]+)ms.*/\1/' || echo "")
SUBMISSION_BYTES=$(grep 'Total submission size int8+zlib:' "${LOGFILE}" | tail -n1 | sed -E 's/.*Total submission size int8\+zlib: ([0-9]+) bytes.*/\1/' || echo "")
PEAK_MEM=$(grep 'peak memory allocated:' "${LOGFILE}" | tail -n1 | sed -E 's/.*allocated: ([0-9]+) MiB.*/\1/' || echo "")
TRAIN_LOSS=$(grep 'step:[0-9]*/1000000 train_loss:' "${LOGFILE}" | tail -n1 | sed -E 's/.*train_loss:([0-9.]+).*/\1/' || echo "")

# Validate extraction
if [[ -z "${VAL_BPB}" ]]; then
    echo "ERROR: could not extract val_bpb from ${LOGFILE}" >&2
    exit 1
fi

# Output structured metrics
echo "METRIC val_bpb=${VAL_BPB}"
echo "METRIC steps=${STEPS:-0}"
echo "METRIC step_avg_ms=${STEP_AVG:-0}"
echo "METRIC submission_bytes=${SUBMISSION_BYTES:-0}"
echo "METRIC train_loss=${TRAIN_LOSS:-0}"
echo "METRIC peak_memory_mb=${PEAK_MEM:-0}"
echo "METRIC train_time_ms=${TRAIN_TIME:-0}"

# Also output raw key lines for human inspection
echo "=== SUMMARY ==="
grep 'stopping_early: wallclock_cap' "${LOGFILE}" | tail -n1 || true
grep 'Total submission size int8+zlib:' "${LOGFILE}" | tail -n1 || true
grep 'final_int8_zlib_roundtrip_exact' "${LOGFILE}" | tail -n1 || true
