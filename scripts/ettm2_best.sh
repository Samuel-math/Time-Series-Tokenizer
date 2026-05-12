#!/bin/bash
# =====================================================================
# ETTm2 best channel-group run.
#
# Execution order:
#   1) Train codebook once.
#   2) Pretrain one shared NTP model for horizons 96/192/336.
#   3) Finetune horizons 96/192/336 from that shared pretrain.
#   4) Pretrain a separate NTP model for horizon 720.
#   5) Finetune horizon 720 from the long-context pretrain.
#
# Usage from repo root:
#   bash scripts/ettm2_best.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Dataset / grouping
export DSET=ettm2
export TOTAL_CHANNELS=7
export MAX_CHANNELS_PER_MODEL=7
export BASE_MODEL_ID="${BASE_MODEL_ID:-1}"
export FORCE_RETRAIN_ALL="${FORCE_RETRAIN_ALL:-0}"
export FORCE_RETRAIN_PRETRAIN="${FORCE_RETRAIN_PRETRAIN:-0}"
export RESUME_LATEST_RUN="${RESUME_LATEST_RUN:-0}"
export USE_CORR_CHANNEL_GROUPS=1
export CHANNEL_GROUPS_DIR="${CHANNEL_GROUPS_DIR:-scripts/channel_groups}"
export RUN_HISTORY_PREFIX="${RUN_HISTORY_PREFIX:-${DSET}_freq_m${MAX_CHANNELS_PER_MODEL}_base${BASE_MODEL_ID}}"
export GROUP_RUN_NAME="${GROUP_RUN_NAME:-${RUN_HISTORY_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
export RETAIN_RUNS="${RETAIN_RUNS:-5}"

# Shared VQVAE/codebook params
export PATCH_SIZE="${PATCH_SIZE:-8}"
export COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-4}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
export CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
export NUM_HIDDENS=128
export NUM_RESIDUAL_LAYERS=2
export NUM_RESIDUAL_HIDDENS=128
export VQVAE_BACKBONE="${VQVAE_BACKBONE:-mlp}"    # mlp=旧结构, chunk_mlp=分块Linear, tcn=Conv1d/TCN
export VQVAE_TCN_KERNEL_SIZE="${VQVAE_TCN_KERNEL_SIZE:-5}"
export VQVAE_CHUNK_SIZE="${VQVAE_CHUNK_SIZE:-2}"
export PER_CHANNEL_CODEBOOK=0
export N_RQ_LAYERS=2
export RQ_LAYER_WEIGHTS="1.0 1.0"

# Shared codebook training params
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

# Shared model/training defaults
export PROGRESSIVE_STEP_SIZE=6
export PRETRAIN_PRED_LEN=6
export N_LAYERS=3
export N_HEADS=4
export D_FF=336
export DROPOUT=0.1
export PRETRAIN_EPOCHS=100
export PRETRAIN_BATCH_SIZE=64
export PRETRAIN_LR=3e-4

export FINETUNE_CONTEXT_POINTS=96
export FINETUNE_EPOCHS=50
export FINETUNE_BATCH_SIZE=128
export USE_GUMBEL_SOFTMAX=1
export GUMBEL_HARD=0
export TRAIN_LOSS=huber
export UNFREEZE_DECODER=0
export DECODER_LR_RATIO=1
export DECODER_WD_RATIO=1

export FEATURES=M
export SCALER=standard
export NUM_WORKERS="${NUM_WORKERS:-0}"
export REVIN=1
export WEIGHT_DECAY=1e-4
export STREAM_LOGS="${STREAM_LOGS:-1}"

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

echo "================================================="
echo "ETTm2 best channel-group run"
echo "================================================="
echo "Repo root              : ${REPO_ROOT}"
echo "Run name               : ${GROUP_RUN_NAME}"
echo "Base model id          : ${BASE_MODEL_ID}"
echo "Max channels per model : ${MAX_CHANNELS_PER_MODEL}"
echo "Run history prefix     : ${RUN_HISTORY_PREFIX}"
echo "Retain runs            : ${RETAIN_RUNS}"
echo "Force rerun all        : ${FORCE_RETRAIN_ALL}"
echo "Force rerun pretrain   : ${FORCE_RETRAIN_PRETRAIN}"
echo "Freq channel grouping  : ${USE_CORR_CHANNEL_GROUPS}"
echo "================================================="

run_pipeline_stage() {
    local stage_name="$1"
    echo
    echo "================================================="
    echo "Stage: ${stage_name}"
    echo "Targets                : ${TARGET_POINTS_LIST}"
    echo "Pretrain context       : ${PRETRAIN_CONTEXT_POINTS}"
    echo "Finetune lr list       : ${FINETUNE_LR_LIST:-${FINETUNE_LR:-}}"
    echo "Gumbel temperature list: ${GUMBEL_TEMPERATURE_LIST:-${GUMBEL_TEMPERATURE:-}}"
    echo "Huber delta list       : ${HUBER_DELTA_LIST:-${HUBER_DELTA:-}}"
    echo "Forecast step list     : ${FORECAST_STEP_SIZE_LIST}"
    echo "Forecast pred list     : ${FORECAST_PRED_LEN_LIST}"
    echo "Log dir                : ${LOG_DIR}"
    echo "================================================="
    bash scripts/decoder_only_NTP/channel_group_pipeline.sh
}

# Phase A:
#   codebook once -> shared pretrain(context=336) -> finetune 96/192/336.
export PRETRAIN_CONTEXT_POINTS=336
export TARGET_POINTS_LIST="96 192 336"
export FINETUNE_LR_LIST="2e-5 2e-5 1e-5"
export GUMBEL_TEMPERATURE_LIST="1.1 0.9 0.9"
export HUBER_DELTA_LIST="0.9 0.9 0.8"
export FORECAST_STEP_SIZE_LIST="12 20 5"
export FORECAST_PRED_LEN_LIST="12 24 10"
export LOG_DIR="${REPO_ROOT}/logs/${GROUP_RUN_NAME}/short_96_192_336"
run_pipeline_stage "short horizons 96/192/336"

# Phase B:
#   reuse the same codebook -> separate pretrain(context=672) -> finetune 720.
# Ensure the second pipeline call never retrains codebook even when the caller
# forced the first stage to rerun from scratch.
export FORCE_RETRAIN_ALL=0
export PRETRAIN_CONTEXT_POINTS=672
export TARGET_POINTS_LIST="720"
export FINETUNE_LR_LIST="2e-5"
export GUMBEL_TEMPERATURE_LIST="0.9"
export HUBER_DELTA_LIST="1.4"
export FORECAST_STEP_SIZE_LIST="4"
export FORECAST_PRED_LEN_LIST="8"
export LOG_DIR="${REPO_ROOT}/logs/${GROUP_RUN_NAME}/long_720"
run_pipeline_stage "long horizon 720"

echo
echo "================================================="
echo "ETTm2 best run complete"
echo "Run name : ${GROUP_RUN_NAME}"
echo "Logs     : ${REPO_ROOT}/logs/${GROUP_RUN_NAME}"
echo "================================================="
