#!/bin/bash
################################################################################
#
# Train RWKV-7 on local Parameter Golf FineWeb SP1892 uint16 shards.
# This reads ../data_sp1892/datasets/fineweb10B_sp1892/fineweb_train_*.bin directly.
#
################################################################################

MODEL_TYPE="${MODEL_TYPE:-x070}"
N_LAYER="${N_LAYER:-2}"
N_EMBD="${N_EMBD:-128}"
CTX_LEN="${CTX_LEN:-512}"
VOCAB_SIZE="${VOCAB_SIZE:-1892}"
TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-0}"
ROPE_MODE="${ROPE_MODE:-none}"
ROPE_THETA="${ROPE_THETA:-10000}"
ROPE_SUFFIX=""
if [ "$ROPE_MODE" != "none" ]; then
 ROPE_SUFFIX="-rope${ROPE_MODE}"
fi
PROJ_DIR="${PROJ_DIR:-out/fineweb-sp${VOCAB_SIZE}${ROPE_SUFFIX}-L${N_LAYER}-D${N_EMBD}-${MODEL_TYPE}}"

M_BSZ="${M_BSZ:-4}"
LR_INIT="${LR_INIT:-6e-4}"
LR_FINAL="${LR_FINAL:-6e-5}"
GRAD_CP="${GRAD_CP:-0}"
EPOCH_SAVE="${EPOCH_SAVE:-1}"

N_NODE="${N_NODE:-1}"
GPU_PER_NODE="${GPU_PER_NODE:-1}"
DS_BUCKET_MB="${DS_BUCKET_MB:-2}"

DATA_FILE="${DATA_FILE:-../data_sp1892/datasets/fineweb10B_sp1892}"
DATA_TYPE="${DATA_TYPE:-fineweb_u16}"
MY_EXIT_TOKENS="${MY_EXIT_TOKENS:-1048576}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
LOSS_LOG_INTERVAL="${LOSS_LOG_INTERVAL:-50}"
METRICS_LOG_INTERVAL="${METRICS_LOG_INTERVAL:-50}"
EXTREME_LOGGING="${EXTREME_LOGGING:-0}"

python train.py \
 --wandb "$WANDB_PROJECT" \
 --wandb_mode "$WANDB_MODE" \
 --wandb_run_name "$WANDB_RUN_NAME" \
 --accelerator gpu \
 --adam_eps 1e-18 \
 --beta1 0.9 \
 --beta2 0.99 \
 --compile 0 \
 --ctx_len $CTX_LEN \
 --data_file $DATA_FILE \
 --data_type $DATA_TYPE \
 --devices $GPU_PER_NODE \
 --ds_bucket_mb $DS_BUCKET_MB \
 --enable_progress_bar True \
 --epoch_begin 0 \
 --epoch_count 1 \
 --epoch_save $EPOCH_SAVE \
 --epoch_steps 128 \
 --grad_cp $GRAD_CP \
 --head_size 64 \
 --load_model "" \
 --lr_final $LR_FINAL \
 --lr_init $LR_INIT \
 --loss_log_interval $LOSS_LOG_INTERVAL \
 --metrics_log_interval $METRICS_LOG_INTERVAL \
 --extreme_logging $EXTREME_LOGGING \
 --magic_prime 0 \
 --micro_bsz $M_BSZ \
 --my_exit_tokens $MY_EXIT_TOKENS \
 --my_testing $MODEL_TYPE \
 --n_embd $N_EMBD \
 --n_layer $N_LAYER \
 --num_nodes $N_NODE \
 --precision bf16 \
 --proj_dir $PROJ_DIR \
 --rope_mode $ROPE_MODE \
 --rope_theta $ROPE_THETA \
 --strategy auto \
 --train_stage 0 \
 --tie_embeddings $TIE_EMBEDDINGS \
 --vocab_size $VOCAB_SIZE \
 --warmup_steps 10 \
 --weight_decay 0.001
