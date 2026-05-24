#!/bin/bash
# =====================================================================
# ETTh2 channel-group wrapper.
#
# Mirrors legacy decoder-only scripts, but runs through
# src/training/channel_group_pipeline.sh.
#
# Usage from repo root:
#   bash scripts/new_etth2.sh
#
# ETTh2 has 7 variables. With MAX_CHANNELS_PER_MODEL=7 this is equivalent to
# one group; set MAX_CHANNELS_PER_MODEL=3 or 4 to test grouped training.
# =====================================================================

set -uo pipefail
export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Dataset / grouping
export DSET=electricity
export TOTAL_CHANNELS=321
export MAX_CHANNELS_PER_MODEL=321
export BASE_MODEL_ID="${BASE_MODEL_ID:-1}"
export FORCE_RETRAIN_ALL=0
export FORCE_RETRAIN_PRETRAIN=0
export RESUME_LATEST_RUN=1
export USE_CORR_CHANNEL_GROUPS=1
export CHANNEL_GROUPS_DIR="${CHANNEL_GROUPS_DIR:-scripts/channel_groups}"
export RUN_HISTORY_PREFIX="${RUN_HISTORY_PREFIX:-${DSET}_freq_m${MAX_CHANNELS_PER_MODEL}_base${BASE_MODEL_ID}}"
export RETAIN_RUNS="${RETAIN_RUNS:-5}"


# Match etth2.sh VQVAE/codebook params
export PATCH_SIZE="${PATCH_SIZE:-4}"
export COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-2}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
export CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
export NUM_HIDDENS=128
export NUM_RESIDUAL_LAYERS=2
export NUM_RESIDUAL_HIDDENS=128
export VQVAE_BACKBONE="${VQVAE_BACKBONE:-mlp}"          # mlp=旧结构, tcn=Conv1d/TCN Encoder+Decoder
export VQVAE_TCN_KERNEL_SIZE="${VQVAE_TCN_KERNEL_SIZE:-3}"
export PER_CHANNEL_CODEBOOK=0
export N_RQ_LAYERS=2
export RQ_LAYER_WEIGHTS="1.0 0.5"

# Codebook training params
export CB_CONTEXT_POINTS=512
export CB_BATCH_SIZE=64
export CB_EPOCHS=50
export CB_LR=3e-4
export SPARSE_WEIGHT=0.3
export SPARSE_AMPLITUDE=0.05
export LAMBDA_ORD=0.01
export ORTH_WEIGHT=0.01
export ORTH_START_EPOCH="${ORTH_START_EPOCH:-0}"
export ORTH_WARMUP_EPOCHS="${ORTH_WARMUP_EPOCHS:-2}"
export SOFT_NEIGHBOR_K=20
export SOFT_NEIGHBOR_ALPHA=0.5
export SOFT_NEIGHBOR_TAU=1

# Match etth2.sh NTP pretrain params
export PRETRAIN_CONTEXT_POINTS=256
export PROGRESSIVE_STEP_SIZE=6
export PRETRAIN_PRED_LEN=6
export N_LAYERS=3
export N_HEADS=4
export D_FF=512
export DROPOUT=0.1
export PRETRAIN_EPOCHS=100
export PRETRAIN_BATCH_SIZE=32
export PRETRAIN_LR=1e-3
export TEMPORAL_BACKBONE=timefilter_lite
export TIMEFILTER_TOPK=128
export TIMEFILTER_TEMPERATURE=1.0

# Match etth2.sh finetune params
export FINETUNE_CONTEXT_POINTS=96
export FINETUNE_EPOCHS=50
export FINETUNE_BATCH_SIZE=16
export FINETUNE_LR=1e-3
export TARGET_POINTS_LIST="${TARGET_POINTS_LIST:-336}"
export USE_GUMBEL_SOFTMAX=1
export GUMBEL_TEMPERATURE=0.8
export GUMBEL_HARD=0
export TRAIN_LOSS=huber
export HUBER_DELTA=1.5
export FORECAST_STEP_SIZE_LIST="${FORECAST_STEP_SIZE_LIST:-20}"
export FORECAST_PRED_LEN_LIST="${FORECAST_PRED_LEN_LIST:-24}"
export UNFREEZE_DECODER=0
export DECODER_LR_RATIO=1
export DECODER_WD_RATIO=1

export FEATURES=M
export SCALER=standard
export NUM_WORKERS="${NUM_WORKERS:-0}"
export REVIN=1
export WEIGHT_DECAY=1e-4
export STREAM_LOGS="${STREAM_LOGS:-1}"

echo "================================================="
echo "Electricity channel-group run"
echo "================================================="
echo "Repo root              : ${REPO_ROOT}"
echo "Max channels per model : ${MAX_CHANNELS_PER_MODEL}"
echo "Target points          : ${TARGET_POINTS_LIST}"
echo "Forecast step list     : ${FORECAST_STEP_SIZE_LIST}"
echo "Forecast pred list     : ${FORECAST_PRED_LEN_LIST}"
echo "Base model id          : ${BASE_MODEL_ID}"
echo "Finetune loss          : ${TRAIN_LOSS} (huber_delta=${HUBER_DELTA})"
echo "Run history prefix     : ${RUN_HISTORY_PREFIX}"
echo "Resume latest run      : ${RESUME_LATEST_RUN:-0}"
echo "Retain runs            : ${RETAIN_RUNS}"
echo "Force rerun all        : ${FORCE_RETRAIN_ALL:-${FORCE_RETRAIN:-0}}"
echo "Force rerun pretrain   : ${FORCE_RETRAIN_PRETRAIN:-0}"
echo "Freq channel grouping  : ${USE_CORR_CHANNEL_GROUPS}"
echo "================================================="

cd "${REPO_ROOT}"

if [ "${USE_CORR_CHANNEL_GROUPS}" = "1" ]; then
    mkdir -p "${CHANNEL_GROUPS_DIR}"
    export CHANNEL_GROUPS_FILE="${CHANNEL_GROUPS_FILE:-${CHANNEL_GROUPS_DIR}/${DSET}_freq${MAX_CHANNELS_PER_MODEL}.json}"
    echo "Generating frequency-feature channel groups: ${CHANNEL_GROUPS_FILE}"
    python scripts/make_channel_groups.py \
        --dset "${DSET}" \
        --max_channels "${MAX_CHANNELS_PER_MODEL}" \
        --output "${CHANNEL_GROUPS_FILE}"
fi

bash src/training/channel_group_pipeline.sh
