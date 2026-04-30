#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

die() {
    echo "ERROR: $*" >&2
    exit 1
}

set_default() {
    local name="$1"
    local value="$2"
    if [[ -z "${!name:-}" ]]; then
        export "${name}=${value}"
    fi
}

bytes_available() {
    df -PB1 "$1" | awk 'NR == 2 {print $4}'
}

gib() {
    awk -v bytes="$1" 'BEGIN {printf "%.1f", bytes / 1024 / 1024 / 1024}'
}

PYTHON_BIN="${PYTHON:-/workspace/parameter-golf/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN:-$(dirname "${PYTHON_BIN}")/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [[ -z "${PROFILE:-}" ]]; then
    if [[ "${SMOKE:-1}" == "0" ]]; then
        PROFILE="balanced"
    else
        PROFILE="smoke"
    fi
fi
RUN_TAG="${RUN_TAG:-caseops_sp8192_8h100_sxm}"
REQUIRE_H100_SXM="${REQUIRE_H100_SXM:-1}"
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"
DRY_RUN="${DRY_RUN:-0}"
DATA_PATH_DEFAULT="/workspace/caseops_sp8192/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"

[[ -x "${PYTHON_BIN}" ]] || die "python not executable: ${PYTHON_BIN}"
[[ -x "${TORCHRUN_BIN}" ]] || die "torchrun not executable: ${TORCHRUN_BIN}"
[[ -d "${DATA_PATH:-${DATA_PATH_DEFAULT}}" ]] || die "CaseOps dataset not found. Set DATA_PATH or prepare ${DATA_PATH_DEFAULT}."

mkdir -p logs .torchinductor_cache
"${PYTHON_BIN}" -m py_compile train_gpt_parcae.py

GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
[[ -n "${GPU_INFO}" ]] || die "nvidia-smi did not report any GPUs"
GPU_COUNT=$(printf '%s\n' "${GPU_INFO}" | sed '/^$/d' | wc -l | tr -d ' ')
[[ "${GPU_COUNT}" -ge "${NPROC_PER_NODE}" ]] || die "requested NPROC_PER_NODE=${NPROC_PER_NODE}, but nvidia-smi sees ${GPU_COUNT} GPUs"

if [[ "${REQUIRE_H100_SXM}" == "1" ]]; then
    [[ "${NPROC_PER_NODE}" == "8" ]] || die "8xH100 SXM profile expects NPROC_PER_NODE=8; set REQUIRE_H100_SXM=0 to override"
    bad_gpu_count=$(printf '%s\n' "${GPU_INFO}" | awk -F, 'tolower($2) !~ /h100/ || $3 + 0 < 79000 {bad++} END {print bad + 0}')
    [[ "${bad_gpu_count}" == "0" ]] || {
        printf '%s\n' "${GPU_INFO}" >&2
        die "expected all visible GPUs to be H100-class with ~80GB memory; set REQUIRE_H100_SXM=0 to override"
    }
fi

FREE_BYTES=$(bytes_available /workspace)
MIN_FREE_BYTES=$((MIN_FREE_GIB * 1024 * 1024 * 1024))
[[ "${FREE_BYTES}" -ge "${MIN_FREE_BYTES}" ]] || die "/workspace has $(gib "${FREE_BYTES}") GiB free; need at least ${MIN_FREE_GIB} GiB"

export RUN_ID="${RUN_ID:-${RUN_TAG}_${PROFILE}_$(date -u +%Y%m%d_%H%M%S)}"
export DATA_PATH="${DATA_PATH:-${DATA_PATH_DEFAULT}}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"

# Single-node 8xH100 SXM defaults. The local 1x RTX PRO 4500 run used only
# ~4 GiB reserved at 16k tokens/microbatch, so these profiles spend H100 memory
# on larger per-rank microbatches while keeping optimizer-step count reasonable.
case "${PROFILE}" in
    smoke)
        set_default MAX_WALLCLOCK_SECONDS 180
        set_default WARMUP_STEPS 40
        set_default TRAIN_LOG_EVERY 10
        set_default TRAIN_BATCH_TOKENS 1048576
        set_default VAL_BATCH_SIZE 4096000
        set_default SAVE_RAW_MODEL 0
        ;;
    balanced)
        set_default MAX_WALLCLOCK_SECONDS 600
        set_default WARMUP_STEPS 400
        set_default TRAIN_LOG_EVERY 50
        set_default TRAIN_BATCH_TOKENS 1048576
        set_default VAL_BATCH_SIZE 4096000
        ;;
    throughput)
        set_default MAX_WALLCLOCK_SECONDS 600
        set_default WARMUP_STEPS 300
        set_default TRAIN_LOG_EVERY 25
        set_default TRAIN_BATCH_TOKENS 2097152
        set_default VAL_BATCH_SIZE 8192000
        ;;
    maxmem)
        set_default MAX_WALLCLOCK_SECONDS 240
        set_default WARMUP_STEPS 80
        set_default TRAIN_LOG_EVERY 10
        set_default TRAIN_BATCH_TOKENS 4194304
        set_default VAL_BATCH_SIZE 12288000
        ;;
    *)
        die "unknown PROFILE=${PROFILE}; use smoke, balanced, throughput, or maxmem"
        ;;
esac

# Model/artifact stack.
set_default MODEL_DIM 384
set_default RECURRENT_DIM 384
set_default RECURRENT_INTERMEDIATE_DIM 1536
set_default NUM_HEADS 8
set_default NUM_KV_HEADS 4
set_default RECURRENT_NUM_HEADS 8
set_default N_LAYERS_IN_PRELUDE 1
set_default N_LAYERS_IN_RECURRENT_BLOCK 3
set_default N_LAYERS_IN_CODA 3
set_default MEAN_RECURRENCE 2
set_default MEAN_BACKPROP_DEPTH 1
set_default RECURRENT_ITERATION_METHOD per-batch
set_default SAMPLING_SCHEME fixed
set_default STATE_INIT like-init

set_default MLP_CLASS_NAME FusedLeakyReLUSqMLP
set_default RECURRENT_MLP_CLASS_NAME FusedLeakyReLUSqMLP
set_default MLP_LEAKY_RELU_SLOPE 0.5
set_default MLP_MULT 4

set_default RESIDUAL_MODE parallel
set_default PARALLEL_RESIDUAL_SCOPE core
set_default PARALLEL_RESIDUAL_START -1
set_default PARALLEL_RESIDUAL_IMPL immediate
set_default ATTN_RES_MODE none

set_default ROPE_DIMS 32
set_default ROPE_BASE 10000
set_default QK_NORM 1
set_default LIGER_CE 1
set_default LIGER_FUSED_CE 1
set_default LIGER_ROPE 1
set_default FUSED_QKV_POSTPROCESS 1
set_default TRIDAO_PACKED_ROPE 0
set_default COMPILE_MODEL 1
set_default COMPILE_MUON_BACKEND 0

set_default ATTN_QKV_MODE packed
set_default ATTN_PRECONV_KERNEL 0
set_default SPARSE_ATTN_GATE 1
set_default SPARSE_ATTN_GATE_WINDOW 12
set_default SPARSE_ATTN_GATE_INIT_STD 0.0
set_default SPARSE_ATTN_GATE_SCALE 0.5

set_default BIGRAM_HASH_BUCKETS 8192
set_default BIGRAM_HASH_DIM 128
set_default BIGRAM_HASH_HEADS 4
set_default BIGRAM_HASH_GATE 1
set_default BIGRAM_HASH_SCALE_INIT 0.05

set_default POE_NUM_EXPERTS 1
set_default CODA_MOE_NUM_EXPERTS 0
set_default DEEPSEEK_MOE_NUM_BASE_EXPERTS 0
set_default DEEPSEEK_MOE_ACTIVE_EXPERTS 0
set_default XSA_LAST_N 0
set_default TTT_ENABLED 0

# Optimizer. Full model compile stays enabled; compiled Muon is opt-in because
# the recent illegal-memory-access failure came from that backend.
set_default MATRIX_LR 0.04
set_default SCALAR_LR 0.04
set_default TIED_EMBED_LR 0.03
set_default EMBED_LR 0.12
set_default TIE_EMBEDDINGS 1
set_default MUON_BACKEND_STEPS 5
set_default MUON_MOMENTUM 0.85
set_default MUON_MOMENTUM_WARMUP_START 0.85
set_default MUON_MOMENTUM_WARMUP_STEPS 500
set_default MUON_ROW_NORMALIZE 1
set_default MUON_WD 0.105
set_default BETA1 0.9
set_default BETA2 0.99
set_default GRAD_CLIP_NORM 0.3

set_default EMA_ENABLED 0
set_default SWA_ENABLED 1
set_default SWA_START_STEP 800
set_default SWA_EVERY 50
set_default SWA_DYNAMIC 1
set_default SWA_DYNAMIC_MIN_EVERY 1
set_default SWA_DYNAMIC_POWER 1.0
set_default SWA_DYNAMIC_WEIGHT_MAX 2.0

set_default QUANT_BITS 6
set_default RANS_INT6 1
set_default GROUPED_ARTIFACT 1
set_default MIXED_QUANT_BITS 0
set_default GPTQ_ENABLED 0
set_default SAVE_RAW_MODEL 0

set_default TRAIN_SEQ_LEN 1024
set_default EVAL_SEQ_LEN 1000
set_default VAL_LOSS_EVERY 0
set_default ITERATIONS 1000000
set_default WARMDOWN_ITERS 1200
set_default GRAD_ACCUM_STEPS 1

# H100 SXM/NVSwitch runtime defaults. NCCL's own docs warn against over-forcing
# debug knobs, so this enables NVLink SHARP/P2P and leaves algorithm choice auto.
set_default CUDA_DEVICE_ORDER PCI_BUS_ID
set_default TORCHINDUCTOR_CACHE_DIR "$(pwd)/.torchinductor_cache"
set_default PYTORCH_CUDA_ALLOC_CONF expandable_segments:True
set_default TORCH_NCCL_ASYNC_ERROR_HANDLING 1
set_default NCCL_DEBUG WARN
set_default NCCL_P2P_DISABLE 0
set_default NCCL_P2P_LEVEL NVL
set_default NCCL_NVLS_ENABLE 1
set_default NCCL_IB_MERGE_NICS 1
set_default NCCL_SET_THREAD_NAME 1

LOCAL_TRAIN_TOKENS=$((TRAIN_BATCH_TOKENS / (NPROC_PER_NODE * GRAD_ACCUM_STEPS)))
[[ $((TRAIN_BATCH_TOKENS % (NPROC_PER_NODE * GRAD_ACCUM_STEPS))) -eq 0 ]] || die "TRAIN_BATCH_TOKENS must be divisible by NPROC_PER_NODE * GRAD_ACCUM_STEPS"
[[ $((LOCAL_TRAIN_TOKENS % TRAIN_SEQ_LEN)) -eq 0 ]] || die "local train tokens ${LOCAL_TRAIN_TOKENS} must be divisible by TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN}"

LOGFILE="logs/${RUN_ID}.console.txt"
{
    echo "=== ${RUN_ID} ==="
    echo "profile=${PROFILE} nproc=${NPROC_PER_NODE} require_h100_sxm=${REQUIRE_H100_SXM}"
    echo "data=${DATA_PATH}"
    echo "free_workspace_gib=$(gib "${FREE_BYTES}") torchinductor_cache=${TORCHINDUCTOR_CACHE_DIR}"
    echo "train_batch_tokens=${TRAIN_BATCH_TOKENS} local_train_tokens=${LOCAL_TRAIN_TOKENS} grad_accum=${GRAD_ACCUM_STEPS} train_seq_len=${TRAIN_SEQ_LEN}"
    echo "val_batch_size=${VAL_BATCH_SIZE} eval_seq_len=${EVAL_SEQ_LEN} max_seconds=${MAX_WALLCLOCK_SECONDS} warmup_steps=${WARMUP_STEPS}"
    echo "compile_model=${COMPILE_MODEL} compile_muon_backend=${COMPILE_MUON_BACKEND} nccl_p2p_level=${NCCL_P2P_LEVEL} nccl_nvls=${NCCL_NVLS_ENABLE}"
    echo "--- GPUs ---"
    printf '%s\n' "${GPU_INFO}"
    echo "--- nvidia-smi topo -m ---"
    nvidia-smi topo -m 2>/dev/null || true
    echo "--- nvidia-smi nvlink -s ---"
    nvidia-smi nvlink -s 2>/dev/null || true
} | tee "${LOGFILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry_run: exiting before torchrun" | tee -a "${LOGFILE}"
    exit 0
fi

TORCHRUN_ARGS=(
    --rdzv-backend=c10d
    --rdzv-endpoint=localhost:0
    --nnodes=1
    --nproc-per-node="${NPROC_PER_NODE}"
)

if [[ "${TORCHRUN_NUMA_BINDING:-1}" == "1" ]] && "${TORCHRUN_BIN}" --help 2>&1 | grep -q -- "--numa-binding"; then
    TORCHRUN_ARGS=(--numa-binding=node "${TORCHRUN_ARGS[@]}")
fi

"${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" train_gpt_parcae.py 2>&1 | tee -a "${LOGFILE}"
