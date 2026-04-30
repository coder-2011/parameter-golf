#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-/workspace/parameter-golf/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN:-$(dirname "${PYTHON_BIN}")/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SMOKE="${SMOKE:-1}"
RUN_TAG="${RUN_TAG:-caseops_sp8192_8h100}"

if [[ ! -d "${DATA_PATH:-./caseops_sp8192/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}" ]]; then
    echo "ERROR: CaseOps dataset not found. Set DATA_PATH or prepare ./caseops_sp8192 first." >&2
    exit 1
fi

mkdir -p logs
"${PYTHON_BIN}" -m py_compile train_gpt_parcae.py

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
if [[ "${GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]]; then
    echo "ERROR: requested NPROC_PER_NODE=${NPROC_PER_NODE}, but nvidia-smi sees ${GPU_COUNT} GPUs" >&2
    exit 1
fi

export RUN_ID="${RUN_ID:-${RUN_TAG}_$(date -u +%Y%m%d_%H%M%S)}"
export DATA_PATH="${DATA_PATH:-./caseops_sp8192/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"

# CaseOps Parcae stack. Keep this close to autoresearch.sh, but do not unset
# torchrun's RANK/WORLD_SIZE environment.
export MODEL_DIM="${MODEL_DIM:-384}"
export RECURRENT_DIM="${RECURRENT_DIM:-384}"
export RECURRENT_INTERMEDIATE_DIM="${RECURRENT_INTERMEDIATE_DIM:-1536}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export RECURRENT_NUM_HEADS="${RECURRENT_NUM_HEADS:-8}"
export N_LAYERS_IN_PRELUDE="${N_LAYERS_IN_PRELUDE:-1}"
export N_LAYERS_IN_RECURRENT_BLOCK="${N_LAYERS_IN_RECURRENT_BLOCK:-3}"
export N_LAYERS_IN_CODA="${N_LAYERS_IN_CODA:-3}"
export MEAN_RECURRENCE="${MEAN_RECURRENCE:-2}"
export MEAN_BACKPROP_DEPTH="${MEAN_BACKPROP_DEPTH:-1}"
export RECURRENT_ITERATION_METHOD="${RECURRENT_ITERATION_METHOD:-per-batch}"
export SAMPLING_SCHEME="${SAMPLING_SCHEME:-fixed}"
export STATE_INIT="${STATE_INIT:-like-init}"

export MLP_CLASS_NAME="${MLP_CLASS_NAME:-FusedLeakyReLUSqMLP}"
export RECURRENT_MLP_CLASS_NAME="${RECURRENT_MLP_CLASS_NAME:-FusedLeakyReLUSqMLP}"
export MLP_LEAKY_RELU_SLOPE="${MLP_LEAKY_RELU_SLOPE:-0.5}"
export MLP_MULT="${MLP_MULT:-4}"

export RESIDUAL_MODE="${RESIDUAL_MODE:-parallel}"
export PARALLEL_RESIDUAL_SCOPE="${PARALLEL_RESIDUAL_SCOPE:-core}"
export PARALLEL_RESIDUAL_START="${PARALLEL_RESIDUAL_START:--1}"
export PARALLEL_RESIDUAL_IMPL="${PARALLEL_RESIDUAL_IMPL:-immediate}"
export ATTN_RES_MODE="${ATTN_RES_MODE:-none}"

export ROPE_DIMS="${ROPE_DIMS:-32}"
export ROPE_BASE="${ROPE_BASE:-10000}"
export QK_NORM="${QK_NORM:-1}"
export LIGER_CE="${LIGER_CE:-1}"
export LIGER_FUSED_CE="${LIGER_FUSED_CE:-1}"
export LIGER_ROPE="${LIGER_ROPE:-1}"
export FUSED_QKV_POSTPROCESS="${FUSED_QKV_POSTPROCESS:-1}"
export TRIDAO_PACKED_ROPE="${TRIDAO_PACKED_ROPE:-0}"
export COMPILE_MODEL="${COMPILE_MODEL:-1}"
export COMPILE_MUON_BACKEND="${COMPILE_MUON_BACKEND:-0}"

export ATTN_QKV_MODE="${ATTN_QKV_MODE:-packed}"
export ATTN_PRECONV_KERNEL="${ATTN_PRECONV_KERNEL:-0}"
export SPARSE_ATTN_GATE="${SPARSE_ATTN_GATE:-1}"
export SPARSE_ATTN_GATE_WINDOW="${SPARSE_ATTN_GATE_WINDOW:-12}"
export SPARSE_ATTN_GATE_INIT_STD="${SPARSE_ATTN_GATE_INIT_STD:-0.0}"
export SPARSE_ATTN_GATE_SCALE="${SPARSE_ATTN_GATE_SCALE:-0.5}"

export BIGRAM_HASH_BUCKETS="${BIGRAM_HASH_BUCKETS:-8192}"
export BIGRAM_HASH_DIM="${BIGRAM_HASH_DIM:-128}"
export BIGRAM_HASH_HEADS="${BIGRAM_HASH_HEADS:-4}"
export BIGRAM_HASH_GATE="${BIGRAM_HASH_GATE:-1}"
export BIGRAM_HASH_SCALE_INIT="${BIGRAM_HASH_SCALE_INIT:-0.05}"

export POE_NUM_EXPERTS="${POE_NUM_EXPERTS:-1}"
export CODA_MOE_NUM_EXPERTS="${CODA_MOE_NUM_EXPERTS:-0}"
export DEEPSEEK_MOE_NUM_BASE_EXPERTS="${DEEPSEEK_MOE_NUM_BASE_EXPERTS:-0}"
export DEEPSEEK_MOE_ACTIVE_EXPERTS="${DEEPSEEK_MOE_ACTIVE_EXPERTS:-0}"
export XSA_LAST_N="${XSA_LAST_N:-0}"
export TTT_ENABLED="${TTT_ENABLED:-0}"

export MATRIX_LR="${MATRIX_LR:-0.04}"
export SCALAR_LR="${SCALAR_LR:-0.04}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.03}"
export EMBED_LR="${EMBED_LR:-0.12}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.85}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.85}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-500}"
export MUON_ROW_NORMALIZE="${MUON_ROW_NORMALIZE:-1}"
export MUON_WD="${MUON_WD:-0.105}"
export BETA1="${BETA1:-0.9}"
export BETA2="${BETA2:-0.99}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.3}"

export EMA_ENABLED="${EMA_ENABLED:-0}"
export SWA_ENABLED="${SWA_ENABLED:-1}"
export SWA_START_STEP="${SWA_START_STEP:-800}"
export SWA_EVERY="${SWA_EVERY:-50}"
export SWA_DYNAMIC="${SWA_DYNAMIC:-1}"
export SWA_DYNAMIC_MIN_EVERY="${SWA_DYNAMIC_MIN_EVERY:-1}"
export SWA_DYNAMIC_POWER="${SWA_DYNAMIC_POWER:-1.0}"
export SWA_DYNAMIC_WEIGHT_MAX="${SWA_DYNAMIC_WEIGHT_MAX:-2.0}"

export QUANT_BITS="${QUANT_BITS:-6}"
export RANS_INT6="${RANS_INT6:-1}"
export GROUPED_ARTIFACT="${GROUPED_ARTIFACT:-1}"
export MIXED_QUANT_BITS="${MIXED_QUANT_BITS:-0}"
export GPTQ_ENABLED="${GPTQ_ENABLED:-0}"
export SAVE_RAW_MODEL="${SAVE_RAW_MODEL:-0}"

export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-1000}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1048576}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
export ITERATIONS="${ITERATIONS:-1000000}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-1200}"

if [[ "${SMOKE}" == "1" ]]; then
    export RUN_ID="${RUN_ID}_smoke"
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-120}"
    export WARMUP_STEPS="${WARMUP_STEPS:-40}"
    export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
    export SAVE_RAW_MODEL=0
else
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
    export WARMUP_STEPS="${WARMUP_STEPS:-500}"
fi

# Global batch tokens. On 8 GPUs this gives each rank 131k tokens with no
# gradient accumulation by default, matching the single-GPU local per-step work.
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-1048576}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

LOGFILE="logs/${RUN_ID}.console.txt"
echo "=== ${RUN_ID} ===" | tee "${LOGFILE}"
echo "nproc=${NPROC_PER_NODE} smoke=${SMOKE} data=${DATA_PATH}" | tee -a "${LOGFILE}"
echo "batch_tokens=${TRAIN_BATCH_TOKENS} grad_accum=${GRAD_ACCUM_STEPS} max_seconds=${MAX_WALLCLOCK_SECONDS} swa_start_step=${SWA_START_STEP}" | tee -a "${LOGFILE}"

"${TORCHRUN_BIN}" \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    train_gpt_parcae.py 2>&1 | tee -a "${LOGFILE}"
