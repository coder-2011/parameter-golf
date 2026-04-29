#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

TRAIN_PID=""

# Kill only this script's training process tree on exit.
cleanup() {
    if [[ -n "${TRAIN_PID}" ]]; then
        pkill -TERM -P "${TRAIN_PID}" 2>/dev/null || true
        kill -TERM "${TRAIN_PID}" 2>/dev/null || true
        sleep 1
        pkill -KILL -P "${TRAIN_PID}" 2>/dev/null || true
        kill -KILL "${TRAIN_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Unique run ID per invocation
RUN_ID="autoresearch_$(date +%Y%m%d_%H%M%S)_$$"
LOGFILE="logs/${RUN_ID}.txt"
PYTHON_BIN="${PYTHON:-/workspace/parameter-golf/.venv/bin/python}"
mkdir -p logs
unset NUM_LAYERS QK_GAIN_INIT CLIP_QKV
unset RANK LOCAL_RANK WORLD_SIZE
unset CONTROL_TENSOR_NAME_PATTERNS INT8_KEEP_FLOAT_FP32_NAME_PATTERNS

echo "=== Autoresearch run: ${RUN_ID} ===" >&2

# Fast syntax check before GPU burn
"${PYTHON_BIN}" -m py_compile train_gpt_parcae.py 2>&1 | tail -5

# Baseline config (SP1024, 300s wall-clock, 1x GPU). Tokenizer/data fields are
# overrideable so tokenizer ablations can reuse this exact harness.
export RUN_ID
export DATA_PATH=${DATA_PATH:-"./data/datasets/fineweb10B_sp1024"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"./data/tokenizers/fineweb_1024_bpe.model"}
export TOKENIZER_META_PATH=${TOKENIZER_META_PATH:-""}
export TOKENIZER_META_VALIDATE=${TOKENIZER_META_VALIDATE:-0}
export VAL_BYTE_COUNT_OVERRIDE=${VAL_BYTE_COUNT_OVERRIDE:-0}
export VOCAB_SIZE=${VOCAB_SIZE:-1024}
export MODEL_DIM=${MODEL_DIM:-384}
export RECURRENT_DIM=${RECURRENT_DIM:-384}
export RECURRENT_INTERMEDIATE_DIM=0
export NUM_HEADS=4
export NUM_KV_HEADS=2
export RECURRENT_NUM_HEADS=4
export N_LAYERS_IN_PRELUDE=${N_LAYERS_IN_PRELUDE:-1}
export N_LAYERS_IN_RECURRENT_BLOCK=${N_LAYERS_IN_RECURRENT_BLOCK:-3}
export N_LAYERS_IN_CODA=${N_LAYERS_IN_CODA:-2}
export MLP_MULT=${MLP_MULT:-4}
export TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-1024}
export TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-131072}
export ITERATIONS=1000000
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-300}
export WARMDOWN_ITERS=1200
export WARMUP_STEPS=500
export VAL_BATCH_SIZE=524288
export TRAIN_LOG_EVERY=100
export VAL_LOSS_EVERY=0
export MEAN_RECURRENCE=${MEAN_RECURRENCE:-2}
export MEAN_BACKPROP_DEPTH=${MEAN_BACKPROP_DEPTH:-1}
export RECURRENT_ITERATION_METHOD=per-batch
export SAMPLING_SCHEME=fixed
export CURRICULUM_TARGET=forward
export ROPE_DIMS=${ROPE_DIMS:-32}
export ROPE_BASE=10000.0
export OUTER_ROPE_DIMS=0
export RECURRENT_ROPE_DIMS=0
export RECURRENT_LAYER_ROPE_DIMS=""
export QK_NORM=${QK_NORM:-1}
export QK_BIAS=0
export USE_VALUE_EMBEDDINGS=0
export PRELUDE_NORM=0
export STATE_INIT=like-init
export INJECTION_TYPE=diagonal
export INJECTION_SWIGLU_SCALE=0.0
export BIGRAM_HASH_BUCKETS=${BIGRAM_HASH_BUCKETS:-8192}
export BIGRAM_HASH_DIM=${BIGRAM_HASH_DIM:-128}
export BIGRAM_HASH_HEADS=${BIGRAM_HASH_HEADS:-2}
export BIGRAM_HASH_GATE=${BIGRAM_HASH_GATE:-1}
export BIGRAM_HASH_SCALE_INIT=0.05
export BIGRAM_HASH_INIT_STD=0.02
export COMPILE_MODEL=1
export COMPILE_MUON_BACKEND=0
export LIGER_CE=${LIGER_CE:-1}
export LIGER_ROPE=${LIGER_ROPE:-0}
export TRIDAO_PACKED_ROPE=${TRIDAO_PACKED_ROPE:-0}
export MONITORING=0
export LOCKSTEP_N=0
export LOCKSTEP_K=0
export MLP_CLASS_NAME=BaseMLP
export RECURRENT_MLP_CLASS_NAME=BaseMLP
export TIE_EMBEDDINGS=1
export EMB_SCALE=1.0
export LOGIT_SCALE=1.0
export LOGIT_SOFTCAP=30.0
export POE_NUM_EXPERTS=${POE_NUM_EXPERTS:-1}
export POE_HEAD_LR=${POE_HEAD_LR:-0.008}
export EMBED_LR=0.6
export HEAD_LR=0.008
export TIED_EMBED_LR=0.05
export TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.04
export SCALAR_LR=0.04
export MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
export MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=500
export MUON_ROW_NORMALIZE=1
export MUON_WD=${MUON_WD:-0.095}
export BETA1=0.9
export BETA2=0.95
export ADAM_EPS=1e-8
export GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-0.3}
export EMA_ENABLED=${EMA_ENABLED:-0}
export EMA=${EMA:-0}
export EMA_DECAY=${EMA_DECAY:-0.997}
export EMA_UPDATE_EVERY=${EMA_UPDATE_EVERY:-1}
export SWA_ENABLED=${SWA_ENABLED:-1}
export SWA_START_FRAC=${SWA_START_FRAC:-0.2}
export SWA_EVERY=${SWA_EVERY:-50}
export QAT_BITS=${QAT_BITS:-0}
export QAT_START_STEP=${QAT_START_STEP:-500}
export QAT_LINEAR=${QAT_LINEAR:-1}
export QAT_TIED_OUTPUT=${QAT_TIED_OUTPUT:-1}
if [[ -z "${QUANT_BITS:-}" ]]; then
    if [[ -n "${QAT_BITS:-}" && "${QAT_BITS}" -gt 0 ]]; then
        export QUANT_BITS="${QAT_BITS}"
    else
        export QUANT_BITS=8
    fi
fi
export GPTQ_ENABLED=${GPTQ_ENABLED:-0}
export GPTQ_CALIBRATION_BATCHES=${GPTQ_CALIBRATION_BATCHES:-32}
export GPTQ_RESERVE_SECONDS=${GPTQ_RESERVE_SECONDS:-12}
export GPTQ_BLOCKSIZE=${GPTQ_BLOCKSIZE:-128}
export GPTQ_DAMPENING=${GPTQ_DAMPENING:-0.01}
export GPTQ_MIN_NUMEL=${GPTQ_MIN_NUMEL:-65536}
export GPTQ_ACT_ORDER=${GPTQ_ACT_ORDER:-1}
export GPTQ_QUANTIZE_EMBEDDINGS=${GPTQ_QUANTIZE_EMBEDDINGS:-1}
export GPTQ_MATRIX_CLIP_SIGMAS=${GPTQ_MATRIX_CLIP_SIGMAS:-12.85}
export GPTQ_EMBED_CLIP_SIGMAS=${GPTQ_EMBED_CLIP_SIGMAS:-20.0}
export SAVE_RAW_MODEL=${SAVE_RAW_MODEL:-0}
export RANS_INT6=${RANS_INT6:-0}
export XSA_LAST_N=${XSA_LAST_N:-0}
export ATTN_RES_MODE=${ATTN_RES_MODE:-none}
export ATTN_RES_SCOPE=${ATTN_RES_SCOPE:-all}
export ATTN_RES_BLOCK_SIZE=${ATTN_RES_BLOCK_SIZE:-2}
export RESIDUAL_MODE=parallel
export PARALLEL_RESIDUAL_SCOPE=core
export MUON_MOMENTUM=0.85
export GPTQ_ENABLED=0
export QUANT_BITS=6
export RANS_INT6=1
export PARALLEL_RESIDUAL_START=${PARALLEL_RESIDUAL_START:--1}
export PARALLEL_RESIDUAL_LN_SCALE=${PARALLEL_RESIDUAL_LN_SCALE:-1}
export CODA_MOE_NUM_EXPERTS=${CODA_MOE_NUM_EXPERTS:-0}
export CODA_MOE_TOP_K=${CODA_MOE_TOP_K:-1}
export CODA_MOE_MLP_MULT=${CODA_MOE_MLP_MULT:-0}
export DEEPSEEK_MOE_NUM_BASE_EXPERTS=${DEEPSEEK_MOE_NUM_BASE_EXPERTS:-0}
export DEEPSEEK_MOE_EXPERT_SEGMENTS=${DEEPSEEK_MOE_EXPERT_SEGMENTS:-4}
export DEEPSEEK_MOE_SHARED_EXPERTS=${DEEPSEEK_MOE_SHARED_EXPERTS:-1}
export DEEPSEEK_MOE_ACTIVE_EXPERTS=${DEEPSEEK_MOE_ACTIVE_EXPERTS:-0}
export DEEPSEEK_MOE_MLP_MULT=${DEEPSEEK_MOE_MLP_MULT:-0}
export DEEPSEEK_MOE_BALANCE_ALPHA=${DEEPSEEK_MOE_BALANCE_ALPHA:-0.0}
export DEEPSEEK_MOE_NORM_TOPK_PROB=${DEEPSEEK_MOE_NORM_TOPK_PROB:-1}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}
export ACTIVATION_CHECKPOINT_IMPL=${ACTIVATION_CHECKPOINT_IMPL:-none}
export SLIDING_WINDOW_ENABLED=${SLIDING_WINDOW_ENABLED:-0}
export SLIDING_COMPILE_LOGITS=${SLIDING_COMPILE_LOGITS:-0}
export EVAL_STRIDE=${EVAL_STRIDE:-64}
export PPM_ENABLED=${PPM_ENABLED:-0}
export PPM_ORDER=${PPM_ORDER:-5}
export PPM_SUBSET_TOKENS=${PPM_SUBSET_TOKENS:-5000000}
export PPM_LAMBDA_HI=${PPM_LAMBDA_HI:-0.9}
export PPM_LAMBDA_LO=${PPM_LAMBDA_LO:-0.05}
export PPM_CONF_THRESHOLD=${PPM_CONF_THRESHOLD:-0.9}
export PPM_ESCAPE_METHOD=${PPM_ESCAPE_METHOD:-d}
export PPM_USE_META_MIX=${PPM_USE_META_MIX:-0}
export PPM_TOKEN_ORDER=${PPM_TOKEN_ORDER:-3}
export PPM_META_ALPHA=${PPM_META_ALPHA:-0.995}
export PPM_META_ETA=${PPM_META_ETA:-2.0}
export PPM_META_WARMUP_BYTES=${PPM_META_WARMUP_BYTES:-4096}
export CONTEXT_MIX_MAX_SECONDS=${CONTEXT_MIX_MAX_SECONDS:-0}
export LZP_ENABLED=${LZP_ENABLED:-0}
export LZP_SUBSET_TOKENS=${LZP_SUBSET_TOKENS:-5000000}
export LZP_ORDERS=${LZP_ORDERS:-4,5,6,8}
export LZP_TABLE_BITS=${LZP_TABLE_BITS:-20}
export LZP_ALPHA_MIN=${LZP_ALPHA_MIN:-0.0}
export LZP_ALPHA_MAX=${LZP_ALPHA_MAX:-0.20}
export LZP_MIN_STREAK=${LZP_MIN_STREAK:-1}
export LZP_MAX_STREAK=${LZP_MAX_STREAK:-8}
export LZP_HIT_PROB=${LZP_HIT_PROB:-0.98}
export NGRAM_EVAL_ORDER=${NGRAM_EVAL_ORDER:-0}
export NGRAM_EVAL_MIN_ORDER=${NGRAM_EVAL_MIN_ORDER:-2}
export NGRAM_EVAL_ALPHA=${NGRAM_EVAL_ALPHA:-0.30}
export NGRAM_EVAL_ADAPTIVE=${NGRAM_EVAL_ADAPTIVE:-1}
export NGRAM_EVAL_ALPHA_MIN=${NGRAM_EVAL_ALPHA_MIN:-0.05}
export NGRAM_EVAL_ALPHA_MAX=${NGRAM_EVAL_ALPHA_MAX:-0.60}
export NGRAM_EVAL_ENTROPY_CENTER=${NGRAM_EVAL_ENTROPY_CENTER:-4.0}
export NGRAM_EVAL_ENTROPY_SCALE=${NGRAM_EVAL_ENTROPY_SCALE:-2.0}
export NGRAM_EVAL_MIN_COUNT=${NGRAM_EVAL_MIN_COUNT:-2}
export NGRAM_EVAL_BUCKETS=${NGRAM_EVAL_BUCKETS:-4194304}
export NGRAM_EVAL_MAX_SECONDS=${NGRAM_EVAL_MAX_SECONDS:-0.0}
export NGRAM_ENTROPY_SHIFT=${NGRAM_ENTROPY_SHIFT:-0}
export NGRAM_ORDER_MULTS=${NGRAM_ORDER_MULTS:-}
export NGRAM_CUBRIC_CADENCE=${NGRAM_CUBRIC_CADENCE:-0}
export CUBRIC_CADENCE=${CUBRIC_CADENCE:-0}
export NGRAM_CHUNK_TOKENS=${NGRAM_CHUNK_TOKENS:-1048576}
export NGRAM_BATCH_SEQS=${NGRAM_BATCH_SEQS:-128}
export NGRAM_MIX_MODE=${NGRAM_MIX_MODE:-expert}
export NGRAM_EXPERT_TOPK=${NGRAM_EXPERT_TOPK:-8}
export NGRAM_EXPERT_BOOST_SCALE=${NGRAM_EXPERT_BOOST_SCALE:-0.25}
export NGRAM_EXPERT_MAX_BOOST=${NGRAM_EXPERT_MAX_BOOST:-12.0}
export TTT_ENABLED=${TTT_ENABLED:-0}
export TTT_LR=${TTT_LR:-0.005}
export TTT_MOMENTUM=${TTT_MOMENTUM:-0.9}
export TTT_EPOCHS=${TTT_EPOCHS:-3}
export TTT_CHUNK_TOKENS=${TTT_CHUNK_TOKENS:-32768}
export TTT_BATCH_SEQS=${TTT_BATCH_SEQS:-32}
export TTT_GRAD_CLIP=${TTT_GRAD_CLIP:-1.0}
export SEED=${SEED:-1337}

# Run training
"${PYTHON_BIN}" train_gpt_parcae.py &
TRAIN_PID=$!
set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
TRAIN_PID=""
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
    exit "${TRAIN_STATUS}"
fi

# Parse exact roundtrip BPB and other metrics from log
if [[ ! -f "${LOGFILE}" ]]; then
    echo "ERROR: log file not found: ${LOGFILE}" >&2
    exit 1
fi

ROUNDTRIP_LINE=$(grep -E 'final_(gptq_)?int[0-9]+(_rans)?_zlib_roundtrip_exact val_loss:' "${LOGFILE}" | tail -n1 || true)
FINAL_STEP_LINE=$(grep -E 'step:[0-9]+/[0-9]+ val_loss:' "${LOGFILE}" | tail -n1 || true)
STOP_LINE=$(grep 'stopping_early: wallclock_cap' "${LOGFILE}" | tail -n1 || true)
SUBMISSION_LINE=$(grep -E 'Total submission size (gptq_)?int[0-9]+(_rans)?\+zlib:' "${LOGFILE}" | tail -n1 || true)

VAL_BPB=$(sed -E 's/.*val_bpb:([0-9.]+).*/\1/' <<<"${ROUNDTRIP_LINE}")
VAL_LOSS=$(sed -E 's/.*val_loss:([0-9.]+).*/\1/' <<<"${ROUNDTRIP_LINE}")
TRAIN_TIME=$(sed -E 's/.*train_time:([0-9.]+)ms.*/\1/' <<<"${STOP_LINE:-${FINAL_STEP_LINE}}")
STEPS=$(sed -E 's/.*step:([0-9]+)\/.*/\1/' <<<"${STOP_LINE:-${FINAL_STEP_LINE}}")
STEP_AVG=$(sed -E 's/.*step_avg:([0-9.]+)ms.*/\1/' <<<"${FINAL_STEP_LINE}")
SUBMISSION_BYTES=$(sed -E 's/.*Total submission size (gptq_)?int[0-9]+\+zlib: ([0-9]+) bytes.*/\2/' <<<"${SUBMISSION_LINE}")
PEAK_MEM=$(grep 'peak memory allocated:' "${LOGFILE}" | tail -n1 | sed -E 's/.*allocated: ([0-9]+) MiB.*/\1/' || echo "")
TRAIN_LOSS=$(grep -E 'step:[0-9]+/[0-9]+ train_loss:' "${LOGFILE}" | tail -n1 | sed -E 's/.*train_loss:([0-9.]+).*/\1/' || echo "")

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
[[ -n "${STOP_LINE}" ]] && echo "${STOP_LINE}"
[[ -n "${SUBMISSION_LINE}" ]] && echo "${SUBMISSION_LINE}"
[[ -n "${ROUNDTRIP_LINE}" ]] && echo "${ROUNDTRIP_LINE}"
