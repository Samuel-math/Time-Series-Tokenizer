#!/bin/bash
# =====================================================================
# ETTm2 merged channel-group runner.
#
# Merges:
#   scripts/ettm2_96.sh
#   scripts/ettm2_192.sh
#   scripts/ettm2_336.sh
#   scripts/ettm2_720.sh
#
# Usage from repo root:
#   bash scripts/ettm2.sh
#   HORIZONS="96 192" bash scripts/ettm2.sh
#   HORIZONS="720" CUDA_VISIBLE_DEVICES_720=0 bash scripts/ettm2.sh
# =====================================================================

set -euo pipefail

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
                echo "ERROR: unsupported ETTm2 horizon '${h}'. Allowed: 96 192 336 720" >&2
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

join_selected() {
    local out=""
    local h
    for h in "$@"; do
        if contains_horizon "${h}"; then
            out="${out}${out:+ }${h}"
        fi
    done
    echo "${out}"
}

setup_common() {
    # Dataset / grouping
    export DSET=ettm2
    export TOTAL_CHANNELS=7
    export MAX_CHANNELS_PER_MODEL="${MAX_CHANNELS_PER_MODEL:-7}"
    export BASE_MODEL_ID="${BASE_MODEL_ID:-1}"
    export RESUME_LATEST_RUN="${RESUME_LATEST_RUN:-1}"
    export USE_CORR_CHANNEL_GROUPS="${USE_CORR_CHANNEL_GROUPS:-1}"
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
    export ORTH_WARMUP_EPOCHS="${ORTH_WARMUP_EPOCHS:-2}"

    # Shared finetune params
    export FINETUNE_CONTEXT_POINTS=96
    export FINETUNE_EPOCHS=50
    export FINETUNE_BATCH_SIZE=128
    export FINETUNE_LR=2e-5
    export USE_GUMBEL_SOFTMAX=1
    export GUMBEL_TEMPERATURE=0.9
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
    echo "ETTm2 merged channel-group run: ${1}"
    echo "================================================="
    echo "Repo root              : ${REPO_ROOT}"
    echo "CUDA_VISIBLE_DEVICES   : ${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "Max channels per model : ${MAX_CHANNELS_PER_MODEL}"
    echo "Target points          : ${TARGET_POINTS_LIST}"
    echo "Forecast step list     : ${FORECAST_STEP_SIZE_LIST}"
    echo "Forecast pred list     : ${FORECAST_PRED_LEN_LIST}"
    echo "Huber delta list       : ${HUBER_DELTA_LIST:-${HUBER_DELTA}}"
    echo "Pretrain context       : ${PRETRAIN_CONTEXT_POINTS}"
    echo "Temporal backbone      : ${TEMPORAL_BACKBONE}"
    echo "D_FF / Dropout         : ${D_FF} / ${DROPOUT}"
    echo "Decoder lowpass        : ${DECODER_LOWPASS}"
    echo "Base model id          : ${BASE_MODEL_ID}"
    echo "Run history prefix     : ${RUN_HISTORY_PREFIX}"
    echo "Resume latest run      : ${RESUME_LATEST_RUN}"
    echo "Retain runs            : ${RETAIN_RUNS}"
    echo "Force rerun all        : ${FORCE_RETRAIN_ALL}"
    echo "Force rerun pretrain   : ${FORCE_RETRAIN_PRETRAIN}"
    echo "Freq channel grouping  : ${USE_CORR_CHANNEL_GROUPS}"
    echo "================================================="
}

run_96_192_336_profile() {
    local selected
    selected="$(join_selected 96 192 336)"
    [ -n "${selected}" ] || return 0

    setup_common
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_96_336:-${CUDA_VISIBLE_DEVICES:-1}}"
    export FORCE_RETRAIN_ALL="${FORCE_RETRAIN_ALL_96_336:-0}"
    if contains_horizon 96; then
        export FORCE_RETRAIN_PRETRAIN="${FORCE_RETRAIN_PRETRAIN_96_336:-1}"
    else
        export FORCE_RETRAIN_PRETRAIN="${FORCE_RETRAIN_PRETRAIN_96_336:-0}"
    fi

    export DECODER_LOWPASS=1
    export PRETRAIN_CONTEXT_POINTS=336
    export PROGRESSIVE_STEP_SIZE=6
    export PRETRAIN_PRED_LEN=6
    export N_LAYERS=3
    export N_HEADS=4
    export D_FF=128
    export DROPOUT=0.15
    export PRETRAIN_EPOCHS=100
    export PRETRAIN_BATCH_SIZE=64
    export PRETRAIN_LR=3e-4
    export TEMPORAL_BACKBONE=timefilter_lite
    export TIMEFILTER_TOPK=4
    export TIMEFILTER_TEMPERATURE="${TIMEFILTER_TEMPERATURE:-1.0}"

    export TARGET_POINTS_LIST="${selected}"
    export FORECAST_STEP_SIZE_LIST="$(for h in ${selected}; do printf '%s ' 10; done | sed 's/[[:space:]]*$//')"
    local pred_lens=""
    local deltas=""
    local h
    for h in ${selected}; do
        case "${h}" in
            96|192)
                pred_lens="${pred_lens}${pred_lens:+ }12"
                deltas="${deltas}${deltas:+ }0.4"
                ;;
            336)
                pred_lens="${pred_lens}${pred_lens:+ }10"
                deltas="${deltas}${deltas:+ }0.27"
                ;;
        esac
    done
    export FORECAST_PRED_LEN_LIST="${pred_lens}"
    export HUBER_DELTA_LIST="${deltas}"
    export HUBER_DELTA="${deltas%% *}"

    print_config "96/192/336 profile"
    bash scripts/decoder_only_NTP/channel_group_pipeline.sh
}

run_720_profile() {
    contains_horizon 720 || return 0

    setup_common
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_720:-${CUDA_VISIBLE_DEVICES:-5}}"
    export FORCE_RETRAIN_ALL="${FORCE_RETRAIN_ALL_720:-1}"
    export FORCE_RETRAIN_PRETRAIN="${FORCE_RETRAIN_PRETRAIN_720:-0}"

    export DECODER_LOWPASS=0
    export PRETRAIN_CONTEXT_POINTS=672
    export PROGRESSIVE_STEP_SIZE=6
    export PRETRAIN_PRED_LEN=6
    export N_LAYERS=3
    export N_HEADS=4
    export D_FF=336
    export DROPOUT=0.1
    export PRETRAIN_EPOCHS=100
    export PRETRAIN_BATCH_SIZE=64
    export PRETRAIN_LR=3e-4
    export TEMPORAL_BACKBONE=causal_transformer
    export TIMEFILTER_TOPK="${TIMEFILTER_TOPK:-8}"
    export TIMEFILTER_TEMPERATURE="${TIMEFILTER_TEMPERATURE:-1.0}"

    export TARGET_POINTS_LIST=720
    export FORECAST_STEP_SIZE_LIST=4
    export FORECAST_PRED_LEN_LIST=8
    export HUBER_DELTA_LIST=1.5
    export HUBER_DELTA=1.5

    print_config "720 profile"
    bash scripts/decoder_only_NTP/channel_group_pipeline.sh
}

cd "${REPO_ROOT}"
validate_horizons
setup_common
generate_channel_groups
run_96_192_336_profile
run_720_profile

echo "================================================="
echo "ETTm2 merged run finished: ${HORIZONS}"
echo "================================================="
