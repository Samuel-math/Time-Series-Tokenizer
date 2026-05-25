#!/bin/bash
# =====================================================================
# Traffic merged channel-group runner.
#
# Merges:
#   scripts/traffic_96.sh
#   scripts/traffic_192.sh
#   scripts/traffic_336.sh
#   scripts/traffic_720.sh
#
# Usage from repo root:
#   bash scripts/traffic.sh
#   HORIZONS="96 192" bash scripts/traffic.sh
#   HORIZONS="96,192,336,720" bash scripts/traffic.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Space- or comma-separated horizons to run. Default: all four source scripts.
HORIZONS="${HORIZONS:-96 192 336 720}"
HORIZONS="${HORIZONS//,/ }"

validate_horizons() {
    local h
    for h in ${HORIZONS}; do
        case "${h}" in
            96|192|336|720) ;;
            *)
                echo "ERROR: unsupported Traffic horizon '${h}'. Allowed: 96 192 336 720" >&2
                exit 1
                ;;
        esac
    done
}

contains_horizon() {
    local target="$1"
    local h
    for h in ${HORIZONS}; do
        if [ "${h}" = "${target}" ]; then
            return 0
        fi
    done
    return 1
}

setup_common() {
    # Dataset / grouping
    export DSET=traffic
    export TOTAL_CHANNELS=862
    export MAX_CHANNELS_PER_MODEL="${MAX_CHANNELS_PER_MODEL:-862}"
    export BASE_MODEL_ID="${BASE_MODEL_ID:-1}"
    export FORCE_RETRAIN_ALL=0
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
    export VQVAE_BACKBONE="${VQVAE_BACKBONE:-mlp}"
    export VQVAE_TCN_KERNEL_SIZE="${VQVAE_TCN_KERNEL_SIZE:-5}"
    export VQVAE_CHUNK_SIZE="${VQVAE_CHUNK_SIZE:-2}"
    export PER_CHANNEL_CODEBOOK=0
    export N_RQ_LAYERS=2
    export RQ_LAYER_WEIGHTS="1.0 0.5"

    # Shared training params
    export CB_CONTEXT_POINTS=336
    export CB_BATCH_SIZE=64
    export CB_EPOCHS=50
    export CB_LR=3e-4
    export SPARSE_WEIGHT=0.3
    export SPARSE_AMPLITUDE=0.05
    export LAMBDA_ORD=0.01
    export ORTH_WEIGHT=0.01
    export ORTH_START_EPOCH="${ORTH_START_EPOCH:-0}"
    export ORTH_WARMUP_EPOCHS="${ORTH_WARMUP_EPOCHS:-2}"

    export PRETRAIN_CONTEXT_POINTS=128
    export PROGRESSIVE_STEP_SIZE=6
    export PRETRAIN_PRED_LEN=6
    export N_LAYERS=3
    export N_HEADS=4
    export D_FF=512
    export DROPOUT=0.1
    export PRETRAIN_EPOCHS=100
    export PRETRAIN_BATCH_SIZE=16
    export PRETRAIN_LR=1e-3
    export SOFT_NEIGHBOR_K=20
    export SOFT_NEIGHBOR_ALPHA=0.3
    export SOFT_NEIGHBOR_TAU=0.5
    export TEMPORAL_BACKBONE=timefilter_lite
    export TIMEFILTER_TOPK=8
    export TIMEFILTER_TEMPERATURE=1.0

    export FINETUNE_CONTEXT_POINTS=96
    export FINETUNE_EPOCHS=30
    export USE_GUMBEL_SOFTMAX=1
    export GUMBEL_TEMPERATURE=0.8
    export GUMBEL_HARD=0
    export TRAIN_LOSS=huber
    export HUBER_DELTA=3.0
    export UNFREEZE_DECODER=0
    export DECODER_LR_RATIO=1
    export DECODER_WD_RATIO=1

    export FEATURES=M
    export SCALER=standard
    export NUM_WORKERS="${NUM_WORKERS:-0}"
    export REVIN=1
    export WEIGHT_DECAY=1e-4
    export STREAM_LOGS="${STREAM_LOGS:-1}"
}

generate_channel_groups() {
    if [ "${USE_CORR_CHANNEL_GROUPS}" = "1" ]; then
        mkdir -p "${CHANNEL_GROUPS_DIR}"
        export CHANNEL_GROUPS_FILE="${CHANNEL_GROUPS_FILE:-${CHANNEL_GROUPS_DIR}/${DSET}_freq${MAX_CHANNELS_PER_MODEL}.json}"
        echo "Generating frequency-feature channel groups: ${CHANNEL_GROUPS_FILE}"
        python scripts/make_channel_groups.py \
            --dset "${DSET}" \
            --max_channels "${MAX_CHANNELS_PER_MODEL}" \
            --output "${CHANNEL_GROUPS_FILE}"
    fi
}

print_config() {
    echo "================================================="
    echo "Traffic merged channel-group run: ${1}"
    echo "================================================="
    echo "Repo root              : ${REPO_ROOT}"
    echo "Max channels per model : ${MAX_CHANNELS_PER_MODEL}"
    echo "Target points          : ${TARGET_POINTS_LIST}"
    echo "Forecast step list     : ${FORECAST_STEP_SIZE_LIST}"
    echo "Forecast pred list     : ${FORECAST_PRED_LEN_LIST}"
    echo "Finetune batch size    : ${FINETUNE_BATCH_SIZE}"
    echo "Finetune lr            : ${FINETUNE_LR}"
    echo "Finetune loss          : ${TRAIN_LOSS} (huber_delta=${HUBER_DELTA})"
    echo "Base model id          : ${BASE_MODEL_ID}"
    echo "Run history prefix     : ${RUN_HISTORY_PREFIX}"
    echo "Resume latest run      : ${RESUME_LATEST_RUN:-0}"
    echo "Retain runs            : ${RETAIN_RUNS}"
    echo "Force rerun all        : ${FORCE_RETRAIN_ALL:-${FORCE_RETRAIN:-0}}"
    echo "Force rerun pretrain   : ${FORCE_RETRAIN_PRETRAIN:-0}"
    echo "Freq channel grouping  : ${USE_CORR_CHANNEL_GROUPS}"
    echo "================================================="
}

run_profile() {
    local horizon="$1"
    contains_horizon "${horizon}" || return 0

    setup_common
    export TARGET_POINTS_LIST="${horizon}"

    case "${horizon}" in
        96)
            export FORCE_RETRAIN_PRETRAIN=1
            export FINETUNE_BATCH_SIZE=16
            export FINETUNE_LR=2e-3
            export FORECAST_STEP_SIZE_LIST=10
            export FORECAST_PRED_LEN_LIST=12
            ;;
        192)
            export FORCE_RETRAIN_PRETRAIN=0
            export FINETUNE_BATCH_SIZE=16
            export FINETUNE_LR=2e-3
            export FORECAST_STEP_SIZE_LIST=12
            export FORECAST_PRED_LEN_LIST=14
            ;;
        336)
            export FORCE_RETRAIN_PRETRAIN=0
            export FINETUNE_BATCH_SIZE=10
            export FINETUNE_LR=6e-4
            export FORECAST_STEP_SIZE_LIST=24
            export FORECAST_PRED_LEN_LIST=28
            ;;
        720)
            export FORCE_RETRAIN_PRETRAIN=0
            export FINETUNE_BATCH_SIZE=10
            export FINETUNE_LR=1e-3
            export FORECAST_STEP_SIZE_LIST=24
            export FORECAST_PRED_LEN_LIST=28
            ;;
    esac

    print_config "pred${horizon}"
    bash scripts/decoder_only_NTP/channel_group_pipeline.sh
}

cd "${REPO_ROOT}"
validate_horizons
setup_common
generate_channel_groups
run_profile 96
run_profile 192
run_profile 336
run_profile 720

echo "================================================="
echo "Traffic merged run finished: ${HORIZONS}"
echo "================================================="
