#!/bin/bash
set -euo pipefail

# Autoresearch run script for RWKV-7 FineWeb SP1892 5-min BPB minimization.
# This script trains for ~300s, saves final checkpoint, quantizes to INT6,
# and evaluates BPB.  It reads env vars for model/search configuration.

cd "$(dirname "$0")"

# ---- defaults ----
N_LAYER="${N_LAYER:-8}"
N_EMBD="${N_EMBD:-512}"
DIM_ATT="${DIM_ATT:-0}"
DIM_FFN="${DIM_FFN:-0}"
CTX_LEN="${CTX_LEN:-1024}"
M_BSZ="${M_BSZ:-16}"
VOCAB_SIZE="${VOCAB_SIZE:-1892}"
HEAD_SIZE="${HEAD_SIZE:-64}"

ROPE_MODE="${ROPE_MODE:-none}"
ROPE_DIMS="${ROPE_DIMS:-0}"
NORM_TYPE="${NORM_TYPE:-layernorm}"
ATTN_EVERY="${ATTN_EVERY:-0}"
ATTN_OFFSET="${ATTN_OFFSET:-0}"
ATTN_MODE="${ATTN_MODE:-full}"
ATTN_HEADS="${ATTN_HEADS:-0}"
ATTN_DIM="${ATTN_DIM:-0}"
ATTN_DROPOUT="${ATTN_DROPOUT:-0.0}"
ATTN_ROPE="${ATTN_ROPE:-1}"
MOBA_CHUNK_SIZE="${MOBA_CHUNK_SIZE:-256}"
MOBA_TOPK="${MOBA_TOPK:-4}"
TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-0}"
LEARNED_SHIFT_STATE="${LEARNED_SHIFT_STATE:-0}"

LR_INIT="${LR_INIT:-4e-4}"
LR_FINAL="${LR_FINAL:-4e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
MY_EXIT_SECONDS="${MY_EXIT_SECONDS:-300}"
MY_EXIT_TOKENS="${MY_EXIT_TOKENS:-0}"

STRATEGY="${STRATEGY:-deepspeed_stage_2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EPOCH_STEPS="${EPOCH_STEPS:-100000}"

DATA_FILE="${DATA_FILE:-../data_sp1892/datasets/fineweb10B_sp1892}"
DATA_TYPE="${DATA_TYPE:-fineweb_u16}"
TOKENIZER_PATH="${TOKENIZER_PATH:-../data_sp1892/tokenizers/fineweb_1892_bpe.model}"

# ---- proj dir ----
PROJ_DIR="${PROJ_DIR:-out/autoresearch-run}"
rm -rf "$PROJ_DIR"
mkdir -p "$PROJ_DIR"

# ---- train ----
python train.py \
  --wandb "" \
  --accelerator gpu \
  --adam_eps 1e-18 \
  --beta1 0.9 --beta2 0.99 \
  --compile 1 \
  --ctx_len "$CTX_LEN" \
  --data_file "$DATA_FILE" \
  --data_type "$DATA_TYPE" \
  --devices 1 \
  --epoch_begin 0 \
  --epoch_count 1 \
  --epoch_save 1 \
  --epoch_steps "$EPOCH_STEPS" \
  --grad_cp 0 \
  --head_size "$HEAD_SIZE" \
  --load_model "" \
  --lr_final "$LR_FINAL" \
  --lr_init "$LR_INIT" \
  --loss_log_interval 50 \
  --metrics_log_interval 50 \
  --micro_bsz "$M_BSZ" \
  --my_exit_seconds "$MY_EXIT_SECONDS" \
  --my_exit_tokens "$MY_EXIT_TOKENS" \
  --my_testing x070 \
  --n_embd "$N_EMBD" \
  --n_layer "$N_LAYER" \
  --norm_type "$NORM_TYPE" \
  --precision bf16 \
  --proj_dir "$PROJ_DIR" \
  --strategy "$STRATEGY" \
  --train_stage 0 \
  --vocab_size "$VOCAB_SIZE" \
  --warmup_steps "$WARMUP_STEPS" \
  --weight_decay "$WEIGHT_DECAY" \
  --grad_clip "$GRAD_CLIP" \
  --rope_mode "$ROPE_MODE" \
  --rope_dims "$ROPE_DIMS" \
  --learned_shift_state "$LEARNED_SHIFT_STATE" \
  --attn_every "$ATTN_EVERY" \
  --attn_offset "$ATTN_OFFSET" \
  --attn_mode "$ATTN_MODE" \
  --attn_heads "$ATTN_HEADS" \
  --attn_dim "$ATTN_DIM" \
  --attn_dropout "$ATTN_DROPOUT" \
  --attn_rope "$ATTN_ROPE" \
  --moba_chunk_size "$MOBA_CHUNK_SIZE" \
  --moba_topk "$MOBA_TOPK" \
  --tie_embeddings "$TIE_EMBEDDINGS" \
  2>&1 | tee "$PROJ_DIR/train_log.txt"

# ---- quantize ----
python -c "
import os, sys
sys.path.insert(0, '.')
from src.quant import save_quantized_state_dict
import torch
pth = '$PROJ_DIR/rwkv-final.pth'
ptz = '$PROJ_DIR/rwkv-final.int6.ptz'
if not os.path.exists(pth):
    print('ERROR: no rwkv-final.pth', file=sys.stderr)
    sys.exit(1)
st = torch.load(pth, map_location='cpu')
for k in list(st.keys()):
    if k.startswith('_forward_module.'):
        st[k.replace('_forward_module.', '')] = st.pop(k)
raw, compressed = save_quantized_state_dict(st, ptz, bits=6)
print(f'INT6 raw={raw} compressed={compressed}')
" 2>&1 | tee -a "$PROJ_DIR/train_log.txt"

INT6_SIZE="${INT6_SIZE:-$(stat --format=%s "$PROJ_DIR/rwkv-final.int6.ptz" 2>/dev/null || echo 0)}"

# ---- eval ----
python eval_fineweb_bpb.py \
  --load_model "$PROJ_DIR/rwkv-final.pth" \
  --data_file "$DATA_FILE" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --vocab_size "$VOCAB_SIZE" \
  --ctx_len "$CTX_LEN" \
  --stride 1024 \
  --micro_bsz 16 \
  --n_layer "$N_LAYER" \
  --n_embd "$N_EMBD" \
  --dim_att "$DIM_ATT" \
  --dim_ffn "$DIM_FFN" \
  --head_size "$HEAD_SIZE" \
  --rope_mode "$ROPE_MODE" \
  --rope_dims "$ROPE_DIMS" \
  --norm_type "$NORM_TYPE" \
  --attn_every "$ATTN_EVERY" \
  --attn_offset "$ATTN_OFFSET" \
  --attn_mode "$ATTN_MODE" \
  --attn_heads "$ATTN_HEADS" \
  --attn_dim "$ATTN_DIM" \
  --attn_rope "$ATTN_ROPE" \
  --moba_chunk_size "$MOBA_CHUNK_SIZE" \
  --moba_topk "$MOBA_TOPK" \
  --tie_embeddings "$TIE_EMBEDDINGS" \
  2>&1 | tee "$PROJ_DIR/eval_log.txt"

# ---- parse metrics ----
VAL_BPB=$(grep -oP 'val_bpb:\K[0-9.]+' "$PROJ_DIR/eval_log.txt" | head -1 || echo "")
VAL_LOSS=$(grep -oP 'val_loss:\K[0-9.]+' "$PROJ_DIR/eval_log.txt" | head -1 || echo "")
if [ -z "$VAL_BPB" ]; then
    VAL_BPB="999"
fi
if [ -z "$VAL_LOSS" ]; then
    VAL_LOSS="999"
fi

# parse train step count from last line of loss_log.csv if present
TRAIN_STEPS=0
if [ -f "$PROJ_DIR/loss_log.csv" ]; then
    TRAIN_STEPS=$(tail -1 "$PROJ_DIR/loss_log.csv" | cut -d',' -f2 | tr -d ' ' || echo 0)
fi

# parse train avg loss from last line
TRAIN_LOSS=""
if [ -f "$PROJ_DIR/loss_log.csv" ]; then
    TRAIN_LOSS=$(tail -1 "$PROJ_DIR/loss_log.csv" | cut -d',' -f5 | tr -d ' ' || echo "")
fi

INT6_SIZE=$(stat --format=%s "$PROJ_DIR/rwkv-final.int6.ptz" 2>/dev/null || echo 0)
INT6_MB=$(python3 -c "print(f'{int('$INT6_SIZE')/1024/1024:.4f}')" 2>/dev/null || echo "0")

echo "METRIC val_bpb=$VAL_BPB"
echo "METRIC val_loss=$VAL_LOSS"
echo "METRIC train_steps=$TRAIN_STEPS"
echo "METRIC train_avg_loss=${TRAIN_LOSS:-0}"
echo "METRIC int6_size_mb=$INT6_MB"
