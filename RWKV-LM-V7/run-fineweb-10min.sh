#!/bin/bash
################################################################################
#
# Timed RWKV-7 FineWeb run for Parameter Golf local uint16 shards.
#
################################################################################

MODEL_TYPE="${MODEL_TYPE:-x070}"
N_LAYER="${N_LAYER:-8}"
N_EMBD="${N_EMBD:-512}"
CTX_LEN="${CTX_LEN:-1024}"
M_BSZ="${M_BSZ:-16}"
VOCAB_SIZE="${VOCAB_SIZE:-1892}"
TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-0}"
QUANT_BITS="${QUANT_BITS:-0}"
ROPE_MODE="${ROPE_MODE:-none}"
ROPE_THETA="${ROPE_THETA:-10000}"
ROPE_DIMS="${ROPE_DIMS:-0}"
NORM_TYPE="${NORM_TYPE:-layernorm}"
ROPE_SUFFIX=""
if [ "$ROPE_MODE" != "none" ]; then
 ROPE_SUFFIX="-rope${ROPE_MODE}"
 if [ "$ROPE_DIMS" != "0" ]; then
  ROPE_SUFFIX="${ROPE_SUFFIX}d${ROPE_DIMS}"
 fi
fi
NORM_SUFFIX=""
if [ "$NORM_TYPE" != "layernorm" ]; then
 NORM_SUFFIX="-${NORM_TYPE}"
fi
PROJ_DIR="${PROJ_DIR:-out/fineweb-sp${VOCAB_SIZE}${ROPE_SUFFIX}${NORM_SUFFIX}-10min-L${N_LAYER}-D${N_EMBD}-${MODEL_TYPE}}"

LR_INIT="${LR_INIT:-4e-4}"
LR_FINAL="${LR_FINAL:-4e-5}"
GRAD_CP="${GRAD_CP:-0}"
EPOCH_SAVE="${EPOCH_SAVE:-1}"

N_NODE="${N_NODE:-1}"
GPU_PER_NODE="${GPU_PER_NODE:-1}"
DS_BUCKET_MB="${DS_BUCKET_MB:-64}"

DATA_FILE="${DATA_FILE:-../data_sp1892/datasets/fineweb10B_sp1892}"
DATA_TYPE="${DATA_TYPE:-fineweb_u16}"
MY_EXIT_TOKENS="${MY_EXIT_TOKENS:-50000000}"
MY_EXIT_SECONDS="${MY_EXIT_SECONDS:-600}"
EPOCH_STEPS="${EPOCH_STEPS:-100000}"
STRATEGY="${STRATEGY:-deepspeed_stage_2}"
COMPILE="${COMPILE:-1}"
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
 --compile $COMPILE \
 --ctx_len $CTX_LEN \
 --data_file $DATA_FILE \
 --data_type $DATA_TYPE \
 --devices $GPU_PER_NODE \
 --ds_bucket_mb $DS_BUCKET_MB \
 --enable_progress_bar True \
 --epoch_begin 0 \
 --epoch_count 1 \
 --epoch_save $EPOCH_SAVE \
 --epoch_steps $EPOCH_STEPS \
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
 --my_exit_seconds $MY_EXIT_SECONDS \
 --my_exit_tokens $MY_EXIT_TOKENS \
 --my_testing $MODEL_TYPE \
 --n_embd $N_EMBD \
 --n_layer $N_LAYER \
 --norm_type $NORM_TYPE \
 --num_nodes $N_NODE \
 --precision bf16 \
 --proj_dir $PROJ_DIR \
 --quant_bits $QUANT_BITS \
 --rope_mode $ROPE_MODE \
 --rope_theta $ROPE_THETA \
 --rope_dims $ROPE_DIMS \
 --strategy $STRATEGY \
 --train_stage 0 \
 --tie_embeddings $TIE_EMBEDDINGS \
 --vocab_size $VOCAB_SIZE \
 --warmup_steps 50 \
 --weight_decay 0.001
