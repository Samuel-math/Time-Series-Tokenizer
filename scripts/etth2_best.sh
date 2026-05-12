#!/bin/bash
# =====================================================================
# ETTh2 best-known horizon runner.
#
# Runs all four ETTh2 horizons in one pipeline pass, with the per-horizon
# finetune settings from:
#   scripts/etth2_pred96.sh
#   scripts/etth2_pred192.sh
#   scripts/etth2_pred336.sh
#   scripts/etth2_pred720.sh
#
# Usage from repo root:
#   bash scripts/etth2_best.sh
#
# Keep the assignments below fixed to match the four source scripts exactly.
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Dataset / grouping
export DSET=etth2
export TOTAL_CHANNELS=7
export MAX_CHANNELS_PER_MODEL=7
export BASE_MODEL_ID="${BASE_MODEL_ID:-1}"
export FORCE_RETRAIN_ALL=0
export FORCE_RETRAIN_PRETRAIN=0
export RESUME_LATEST_RUN=1
export USE_CORR_CHANNEL_GROUPS=1
export CHANNEL_GROUPS_DIR="${CHANNEL_GROUPS_DIR:-scripts/channel_groups}"
export RUN_HISTORY_PREFIX="${RUN_HISTORY_PREFIX:-${DSET}_freq_m${MAX_CHANNELS_PER_MODEL}_base${BASE_MODEL_ID}}"
export RETAIN_RUNS="${RETAIN_RUNS:-5}"

# VQVAE / codebook params
export PATCH_SIZE="${PATCH_SIZE:-8}"
export COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-4}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
export CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
export NUM_HIDDENS=128
export NUM_RESIDUAL_LAYERS=2
export NUM_RESIDUAL_HIDDENS=128
export VQVAE_BACKBONE=mlp
export PER_CHANNEL_CODEBOOK=0
export N_RQ_LAYERS=2
export RQ_LAYER_WEIGHTS="1.0 1.0"

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
export ORTH_WARMUP_EPOCHS="${ORTH_WARMUP_EPOCHS:-5}"

# NTP pretrain params
export PRETRAIN_CONTEXT_POINTS=296
export PROGRESSIVE_STEP_SIZE=3
export PRETRAIN_PRED_LEN=6
export N_LAYERS=3
export N_HEADS=4
export D_FF=256
export DROPOUT=0.1
export PRETRAIN_EPOCHS=100
export PRETRAIN_BATCH_SIZE=64
export PRETRAIN_LR=3e-4

# Shared finetune params
export FINETUNE_CONTEXT_POINTS=96
export FINETUNE_EPOCHS=50
export FINETUNE_BATCH_SIZE=32
export FINETUNE_LR=2e-4
export FINETUNE_LR_LIST="2e-4 2e-4 2e-4 1e-4"
export TARGET_POINTS_LIST="96 192 336 720"
export FORECAST_STEP_SIZE_LIST="3 6 7 4"
export FORECAST_PRED_LEN_LIST="9 12 14 6"
export USE_GUMBEL_SOFTMAX=1
# *_LIST 才是按 horizon 真正生效的参数：
#   index 对应 TARGET_POINTS_LIST 的位置 (96, 192, 336, 720)
# 单值变量只在对应 *_LIST 未提供时作为 fallback。
export GUMBEL_TEMPERATURE_LIST="0.8 0.8 0.8 0.6"
export GUMBEL_TEMPERATURE=0.8
export GUMBEL_HARD=0
export TRAIN_LOSS=huber
export HUBER_DELTA_LIST="2.0 2.0 2.0 1.8"
export HUBER_DELTA=2.0

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
echo "ETTh2 best run"
echo "================================================="
echo "Repo root              : ${REPO_ROOT}"
echo "Target points          : ${TARGET_POINTS_LIST}"
echo "Forecast step list     : ${FORECAST_STEP_SIZE_LIST}"
echo "Forecast pred list     : ${FORECAST_PRED_LEN_LIST}"
echo "Finetune lr list       : ${FINETUNE_LR_LIST}"
echo "Gumbel temperature list: ${GUMBEL_TEMPERATURE_LIST}"
echo "Huber delta list       : ${HUBER_DELTA_LIST}"
echo "Base model id          : ${BASE_MODEL_ID}"
echo "Run history prefix     : ${RUN_HISTORY_PREFIX}"
echo "Resume latest run      : ${RESUME_LATEST_RUN}"
echo "Force rerun all        : ${FORCE_RETRAIN_ALL}"
echo "Force rerun pretrain   : ${FORCE_RETRAIN_PRETRAIN}"
echo "================================================="

read -ra _TP_ARR <<<"${TARGET_POINTS_LIST}"
read -ra _LR_ARR <<<"${FINETUNE_LR_LIST}"
read -ra _STEP_ARR <<<"${FORECAST_STEP_SIZE_LIST}"
read -ra _PRED_ARR <<<"${FORECAST_PRED_LEN_LIST}"
read -ra _TAU_ARR <<<"${GUMBEL_TEMPERATURE_LIST}"
read -ra _DELTA_ARR <<<"${HUBER_DELTA_LIST}"
echo "Per-horizon resolved params:"
printf "  %-10s %-8s %-8s %-12s %-6s %-12s\n" "target" "step" "pred_len" "lr" "tau" "huber_delta"
for i in "${!_TP_ARR[@]}"; do
    printf "  %-10s %-8s %-8s %-12s %-6s %-12s\n" \
        "${_TP_ARR[$i]}" "${_STEP_ARR[$i]:-?}" "${_PRED_ARR[$i]:-?}" \
        "${_LR_ARR[$i]:-?}" "${_TAU_ARR[$i]:-?}" "${_DELTA_ARR[$i]:-?}"
done
echo "================================================="

bash scripts/decoder_only_NTP/channel_group_pipeline.sh
