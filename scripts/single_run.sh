#!/bin/bash
# =====================================================================
# Single dataset/input_len/output_len channel-group runner.
#
# Usage from repo root:
#   bash scripts/single_run.sh --dset etth2 --input_len 96 --output_len 336
#   DSET=weather INPUT_LEN=96 OUTPUT_LEN=192 bash scripts/single_run.sh
#
# GPU is intentionally not specified here. Set CUDA_VISIBLE_DEVICES outside
# this script if needed.
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/single_run.sh --dset <dataset> --input_len <len> --output_len <len> [options]

Datasets:
  ettm1 ettm2 etth1 etth2 electricity/ecl traffic weather illness exchange

Required arguments, or equivalent env vars:
  --dset, --dataset          Dataset name. Env: DSET
  --input_len                Finetune input/context length. Env: INPUT_LEN
  --output_len               Forecast horizon/target length. Env: OUTPUT_LEN

Optional:
  --forecast_step_size N      Override finetune autoregressive step size (patch count)
  --forecast_pred_len N       Override finetune per-forward pred_len (patch count)
  --max_channels_per_model N  Override channel group size
  --force                    Set FORCE_RETRAIN_ALL=1
  --no_resume                Set RESUME_LATEST_RUN=0
  --no_channel_groups        Set USE_CORR_CHANNEL_GROUPS=0
  -h, --help                 Show this help

Examples:
  bash scripts/single_run.sh --dset etth2 --input_len 96 --output_len 720
  CUDA_VISIBLE_DEVICES=0 bash scripts/single_run.sh --dset traffic --input_len 96 --output_len 192
  DSET=weather INPUT_LEN=192 OUTPUT_LEN=336 FORECAST_STEP_SIZE=8 FORECAST_PRED_LEN=10 bash scripts/single_run.sh
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dset|--dataset)
            DSET="$2"; shift 2 ;;
        --input_len|--input-len|--seq_len|--seq-len)
            INPUT_LEN="$2"; shift 2 ;;
        --output_len|--output-len|--target_points|--target-points)
            OUTPUT_LEN="$2"; shift 2 ;;
        --forecast_step_size|--forecast-step-size|--ar_step_size|--ar-step-size)
            FORECAST_STEP_SIZE="$2"; shift 2 ;;
        --forecast_pred_len|--forecast-pred-len)
            FORECAST_PRED_LEN="$2"; shift 2 ;;
        --max_channels_per_model|--max-channels-per-model)
            MAX_CHANNELS_PER_MODEL="$2"; shift 2 ;;
        --force)
            FORCE_RETRAIN_ALL=1; shift ;;
        --no_resume|--no-resume)
            RESUME_LATEST_RUN=0; shift ;;
        --no_channel_groups|--no-channel-groups)
            USE_CORR_CHANNEL_GROUPS=0; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "ERROR: unknown argument '$1'" >&2
            usage >&2
            exit 1 ;;
    esac
done

DSET="${DSET:-}"
INPUT_LEN="${INPUT_LEN:-}"
OUTPUT_LEN="${OUTPUT_LEN:-}"

if [ -z "${DSET}" ] || [ -z "${INPUT_LEN}" ] || [ -z "${OUTPUT_LEN}" ]; then
    echo "ERROR: --dset, --input_len and --output_len are required." >&2
    usage >&2
    exit 1
fi

DSET="$(printf '%s' "${DSET}" | tr '[:upper:]' '[:lower:]')"
case "${DSET}" in
    ecl) DSET=electricity ;;
esac

case "${INPUT_LEN}" in
    ''|*[!0-9]*) echo "ERROR: input_len must be a positive integer: ${INPUT_LEN}" >&2; exit 1 ;;
esac
case "${OUTPUT_LEN}" in
    ''|*[!0-9]*) echo "ERROR: output_len must be a positive integer: ${OUTPUT_LEN}" >&2; exit 1 ;;
esac
if [ "${INPUT_LEN}" -le 0 ] || [ "${OUTPUT_LEN}" -le 0 ]; then
    echo "ERROR: input_len/output_len must be positive." >&2
    exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "ERROR: cannot find python3 or python in PATH" >&2
        exit 127
    fi
fi
export PYTHON_BIN

# Remember explicit env/CLI overrides before applying presets.
[ -n "${FORECAST_STEP_SIZE+x}" ] && USER_FORECAST_STEP_SIZE_SET=1 || USER_FORECAST_STEP_SIZE_SET=0
[ -n "${FORECAST_PRED_LEN+x}" ] && USER_FORECAST_PRED_LEN_SET=1 || USER_FORECAST_PRED_LEN_SET=0
[ -n "${HUBER_DELTA+x}" ] && USER_HUBER_DELTA_SET=1 || USER_HUBER_DELTA_SET=0
[ -n "${PRETRAIN_CONTEXT_POINTS+x}" ] && USER_PRETRAIN_CONTEXT_POINTS_SET=1 || USER_PRETRAIN_CONTEXT_POINTS_SET=0
[ -n "${TEMPORAL_BACKBONE+x}" ] && USER_TEMPORAL_BACKBONE_SET=1 || USER_TEMPORAL_BACKBONE_SET=0
[ -n "${D_FF+x}" ] && USER_D_FF_SET=1 || USER_D_FF_SET=0
[ -n "${DROPOUT+x}" ] && USER_DROPOUT_SET=1 || USER_DROPOUT_SET=0
[ -n "${DECODER_LOWPASS+x}" ] && USER_DECODER_LOWPASS_SET=1 || USER_DECODER_LOWPASS_SET=0
[ -n "${FINETUNE_LR+x}" ] && USER_FINETUNE_LR_SET=1 || USER_FINETUNE_LR_SET=0
[ -n "${GUMBEL_TEMPERATURE+x}" ] && USER_GUMBEL_TEMPERATURE_SET=1 || USER_GUMBEL_TEMPERATURE_SET=0
[ -n "${FINETUNE_BATCH_SIZE+x}" ] && USER_FINETUNE_BATCH_SIZE_SET=1 || USER_FINETUNE_BATCH_SIZE_SET=0
[ -n "${FINETUNE_EPOCHS+x}" ] && USER_FINETUNE_EPOCHS_SET=1 || USER_FINETUNE_EPOCHS_SET=0
[ -n "${TIMEFILTER_TOPK+x}" ] && USER_TIMEFILTER_TOPK_SET=1 || USER_TIMEFILTER_TOPK_SET=0

set_if_user_unset() {
    local var="$1"
    local value="$2"
    local flag
    eval "flag=\${USER_${var}_SET:-0}"
    if [ "${flag}" != "1" ]; then
        export "${var}=${value}"
    fi
}

# Dataset-level presets. Most values can be overridden through env vars.
case "${DSET}" in
    ettm1)
        export TOTAL_CHANNELS="${TOTAL_CHANNELS:-7}"
        export MAX_CHANNELS_PER_MODEL="${MAX_CHANNELS_PER_MODEL:-7}"
        export PATCH_SIZE="${PATCH_SIZE:-16}"
        export COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-8}"
        export EMBEDDING_DIM="${EMBEDDING_DIM:-32}"
        export CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
        export NUM_HIDDENS="${NUM_HIDDENS:-128}"
        export NUM_RESIDUAL_HIDDENS="${NUM_RESIDUAL_HIDDENS:-128}"
        export CB_CONTEXT_POINTS="${CB_CONTEXT_POINTS:-512}"
        export PRETRAIN_CONTEXT_POINTS="${PRETRAIN_CONTEXT_POINTS:-512}"
        export PROGRESSIVE_STEP_SIZE="${PROGRESSIVE_STEP_SIZE:-3}"
        export PRETRAIN_PRED_LEN="${PRETRAIN_PRED_LEN:-6}"
        export N_HEADS="${N_HEADS:-8}"
        export D_FF="${D_FF:-128}"
        export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-64}"
        export FINETUNE_LR="${FINETUNE_LR:-2e-4}"
        export HUBER_DELTA="${HUBER_DELTA:-0.9}"
        ;;
    ettm2)
        export TOTAL_CHANNELS="${TOTAL_CHANNELS:-7}"
        export MAX_CHANNELS_PER_MODEL="${MAX_CHANNELS_PER_MODEL:-7}"
        export PATCH_SIZE="${PATCH_SIZE:-8}"
        export COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-4}"
        export EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
        export CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
        export NUM_HIDDENS="${NUM_HIDDENS:-128}"
        export NUM_RESIDUAL_HIDDENS="${NUM_RESIDUAL_HIDDENS:-128}"
        export CB_CONTEXT_POINTS="${CB_CONTEXT_POINTS:-512}"
        export PRETRAIN_CONTEXT_POINTS="${PRETRAIN_CONTEXT_POINTS:-336}"
        export PROGRESSIVE_STEP_SIZE="${PROGRESSIVE_STEP_SIZE:-6}"
        export PRETRAIN_PRED_LEN="${PRETRAIN_PRED_LEN:-6}"
        export N_HEADS="${N_HEADS:-4}"
        export D_FF="${D_FF:-128}"
        export DROPOUT="${DROPOUT:-0.15}"
        export TEMPORAL_BACKBONE="${TEMPORAL_BACKBONE:-timefilter_lite}"
        export TIMEFILTER_TOPK="${TIMEFILTER_TOPK:-4}"
        export DECODER_LOWPASS="${DECODER_LOWPASS:-1}"
        export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-128}"
        export FINETUNE_LR="${FINETUNE_LR:-2e-5}"
        ;;
    etth1)
        export TOTAL_CHANNELS="${TOTAL_CHANNELS:-7}"
        export MAX_CHANNELS_PER_MODEL="${MAX_CHANNELS_PER_MODEL:-7}"
        export PATCH_SIZE="${PATCH_SIZE:-4}"
        export COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-4}"
        export EMBEDDING_DIM="${EMBEDDING_DIM:-32}"
        export CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
        export NUM_HIDDENS="${NUM_HIDDENS:-64}"
        export NUM_RESIDUAL_HIDDENS="${NUM_RESIDUAL_HIDDENS:-64}"
        export CB_CONTEXT_POINTS="${CB_CONTEXT_POINTS:-512}"
        export PRETRAIN_CONTEXT_POINTS="${PRETRAIN_CONTEXT_POINTS:-296}"
        export PROGRESSIVE_STEP_SIZE="${PROGRESSIVE_STEP_SIZE:-2}"
        export PRETRAIN_PRED_LEN="${PRETRAIN_PRED_LEN:-6}"
        export N_HEADS="${N_HEADS:-2}"
        export D_FF="${D_FF:-256}"
        export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-32}"
        export FINETUNE_LR="${FINETUNE_LR:-2e-4}"
        export HUBER_DELTA="${HUBER_DELTA:-1.0}"
        ;;
    etth2)
        export TOTAL_CHANNELS="${TOTAL_CHANNELS:-7}"
        export MAX_CHANNELS_PER_MODEL="${MAX_CHANNELS_PER_MODEL:-7}"
        export PATCH_SIZE="${PATCH_SIZE:-8}"
        export COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-4}"
        export EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
        export CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
        export NUM_HIDDENS="${NUM_HIDDENS:-128}"
        export NUM_RESIDUAL_HIDDENS="${NUM_RESIDUAL_HIDDENS:-128}"
        export CB_CONTEXT_POINTS="${CB_CONTEXT_POINTS:-512}"
        export PRETRAIN_CONTEXT_POINTS="${PRETRAIN_CONTEXT_POINTS:-296}"
        export PROGRESSIVE_STEP_SIZE="${PROGRESSIVE_STEP_SIZE:-3}"
        export PRETRAIN_PRED_LEN="${PRETRAIN_PRED_LEN:-6}"
        export N_HEADS="${N_HEADS:-4}"
        export D_FF="${D_FF:-256}"
        export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-32}"
        export FINETUNE_LR="${FINETUNE_LR:-2e-4}"
        ;;
    electricity)
        export TOTAL_CHANNELS="${TOTAL_CHANNELS:-321}"
        export MAX_CHANNELS_PER_MODEL="${MAX_CHANNELS_PER_MODEL:-321}"
        export PATCH_SIZE="${PATCH_SIZE:-4}"
        export COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-2}"
        export EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
        export CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
        export NUM_HIDDENS="${NUM_HIDDENS:-128}"
        export NUM_RESIDUAL_HIDDENS="${NUM_RESIDUAL_HIDDENS:-128}"
        export CB_CONTEXT_POINTS="${CB_CONTEXT_POINTS:-512}"
        export PRETRAIN_CONTEXT_POINTS="${PRETRAIN_CONTEXT_POINTS:-256}"
        export PROGRESSIVE_STEP_SIZE="${PROGRESSIVE_STEP_SIZE:-6}"
        export PRETRAIN_PRED_LEN="${PRETRAIN_PRED_LEN:-6}"
        export N_HEADS="${N_HEADS:-4}"
        export D_FF="${D_FF:-512}"
        export TEMPORAL_BACKBONE="${TEMPORAL_BACKBONE:-timefilter_lite}"
        export TIMEFILTER_TOPK="${TIMEFILTER_TOPK:-128}"
        export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-32}"
        export FINETUNE_LR="${FINETUNE_LR:-1e-3}"
        ;;
    traffic)
        export TOTAL_CHANNELS="${TOTAL_CHANNELS:-862}"
        export MAX_CHANNELS_PER_MODEL="${MAX_CHANNELS_PER_MODEL:-862}"
        export PATCH_SIZE="${PATCH_SIZE:-8}"
        export COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-4}"
        export EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
        export CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
        export NUM_HIDDENS="${NUM_HIDDENS:-128}"
        export NUM_RESIDUAL_HIDDENS="${NUM_RESIDUAL_HIDDENS:-128}"
        export CB_CONTEXT_POINTS="${CB_CONTEXT_POINTS:-336}"
        export PRETRAIN_CONTEXT_POINTS="${PRETRAIN_CONTEXT_POINTS:-128}"
        export PROGRESSIVE_STEP_SIZE="${PROGRESSIVE_STEP_SIZE:-6}"
        export PRETRAIN_PRED_LEN="${PRETRAIN_PRED_LEN:-6}"
        export N_HEADS="${N_HEADS:-4}"
        export D_FF="${D_FF:-512}"
        export TEMPORAL_BACKBONE="${TEMPORAL_BACKBONE:-timefilter_lite}"
        export TIMEFILTER_TOPK="${TIMEFILTER_TOPK:-8}"
        export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-16}"
        export FINETUNE_LR="${FINETUNE_LR:-2e-3}"
        export HUBER_DELTA="${HUBER_DELTA:-3.0}"
        ;;
    weather)
        export TOTAL_CHANNELS="${TOTAL_CHANNELS:-21}"
        export MAX_CHANNELS_PER_MODEL="${MAX_CHANNELS_PER_MODEL:-21}"
        export PATCH_SIZE="${PATCH_SIZE:-8}"
        export COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-4}"
        export EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
        export CODEBOOK_SIZE="${CODEBOOK_SIZE:-512}"
        export NUM_HIDDENS="${NUM_HIDDENS:-128}"
        export NUM_RESIDUAL_HIDDENS="${NUM_RESIDUAL_HIDDENS:-128}"
        export CB_CONTEXT_POINTS="${CB_CONTEXT_POINTS:-672}"
        export PRETRAIN_CONTEXT_POINTS="${PRETRAIN_CONTEXT_POINTS:-672}"
        export PROGRESSIVE_STEP_SIZE="${PROGRESSIVE_STEP_SIZE:-6}"
        export PRETRAIN_PRED_LEN="${PRETRAIN_PRED_LEN:-6}"
        export N_HEADS="${N_HEADS:-4}"
        export D_FF="${D_FF:-256}"
        export TEMPORAL_BACKBONE="${TEMPORAL_BACKBONE:-timefilter_lite}"
        export TIMEFILTER_TOPK="${TIMEFILTER_TOPK:-8}"
        export DECODER_LOWPASS="${DECODER_LOWPASS:-1}"
        export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-128}"
        export FINETUNE_LR="${FINETUNE_LR:-3e-4}"
        ;;
    illness)
        export TOTAL_CHANNELS="${TOTAL_CHANNELS:-7}"
        export MAX_CHANNELS_PER_MODEL="${MAX_CHANNELS_PER_MODEL:-7}"
        ;;
    exchange)
        export TOTAL_CHANNELS="${TOTAL_CHANNELS:-8}"
        export MAX_CHANNELS_PER_MODEL="${MAX_CHANNELS_PER_MODEL:-8}"
        ;;
    *)
        echo "ERROR: unsupported dataset '${DSET}'" >&2
        usage >&2
        exit 1 ;;
esac

# Generic defaults for unset variables.
export BASE_MODEL_ID="${BASE_MODEL_ID:-1}"
export FORCE_RETRAIN_ALL="${FORCE_RETRAIN_ALL:-0}"
export FORCE_RETRAIN_PRETRAIN="${FORCE_RETRAIN_PRETRAIN:-0}"
export RESUME_LATEST_RUN="${RESUME_LATEST_RUN:-1}"
export USE_CORR_CHANNEL_GROUPS="${USE_CORR_CHANNEL_GROUPS:-1}"
export CHANNEL_GROUPS_DIR="${CHANNEL_GROUPS_DIR:-scripts/channel_groups}"
export RUN_HISTORY_PREFIX="${RUN_HISTORY_PREFIX:-${DSET}_single_cw${INPUT_LEN}_tw${OUTPUT_LEN}_m${MAX_CHANNELS_PER_MODEL}_base${BASE_MODEL_ID}}"
export RETAIN_RUNS="${RETAIN_RUNS:-5}"

export NUM_RESIDUAL_LAYERS="${NUM_RESIDUAL_LAYERS:-2}"
export VQVAE_BACKBONE="${VQVAE_BACKBONE:-mlp}"
export VQVAE_TCN_KERNEL_SIZE="${VQVAE_TCN_KERNEL_SIZE:-5}"
export VQVAE_CHUNK_SIZE="${VQVAE_CHUNK_SIZE:-2}"
export DECODER_LOWPASS="${DECODER_LOWPASS:-0}"
export PER_CHANNEL_CODEBOOK="${PER_CHANNEL_CODEBOOK:-0}"
export N_RQ_LAYERS="${N_RQ_LAYERS:-2}"
export RQ_LAYER_WEIGHTS="${RQ_LAYER_WEIGHTS:-1.0 1.0}"

export CB_BATCH_SIZE="${CB_BATCH_SIZE:-64}"
export CB_EPOCHS="${CB_EPOCHS:-50}"
export CB_LR="${CB_LR:-3e-4}"
export SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.3}"
export SPARSE_AMPLITUDE="${SPARSE_AMPLITUDE:-0.05}"
export LAMBDA_ORD="${LAMBDA_ORD:-0.01}"
export ORTH_WEIGHT="${ORTH_WEIGHT:-0.01}"
export ORTH_START_EPOCH="${ORTH_START_EPOCH:-0}"
export ORTH_WARMUP_EPOCHS="${ORTH_WARMUP_EPOCHS:-5}"

export N_LAYERS="${N_LAYERS:-3}"
export DROPOUT="${DROPOUT:-0.1}"
export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
export PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-64}"
export PRETRAIN_LR="${PRETRAIN_LR:-3e-4}"
export TEMPORAL_BACKBONE="${TEMPORAL_BACKBONE:-causal_transformer}"
export TIMEFILTER_TOPK="${TIMEFILTER_TOPK:-8}"
export TIMEFILTER_TEMPERATURE="${TIMEFILTER_TEMPERATURE:-1.0}"
export SOFT_NEIGHBOR_K="${SOFT_NEIGHBOR_K:-20}"
export SOFT_NEIGHBOR_ALPHA="${SOFT_NEIGHBOR_ALPHA:-0.3}"
export SOFT_NEIGHBOR_TAU="${SOFT_NEIGHBOR_TAU:-0.5}"

export FINETUNE_CONTEXT_POINTS="${INPUT_LEN}"
export TARGET_POINTS_LIST="${OUTPUT_LEN}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"
export USE_GUMBEL_SOFTMAX="${USE_GUMBEL_SOFTMAX:-1}"
export GUMBEL_TEMPERATURE="${GUMBEL_TEMPERATURE:-0.8}"
export GUMBEL_HARD="${GUMBEL_HARD:-0}"
export TRAIN_LOSS="${TRAIN_LOSS:-huber}"
export UNFREEZE_DECODER="${UNFREEZE_DECODER:-0}"
export DECODER_LR_RATIO="${DECODER_LR_RATIO:-1}"
export DECODER_WD_RATIO="${DECODER_WD_RATIO:-1}"

# Best-known horizon presets where available. Env/CLI overrides win.
case "${DSET}:${OUTPUT_LEN}" in
    ettm1:96|ettm1:192)
        set_if_user_unset FORECAST_STEP_SIZE 3; set_if_user_unset FORECAST_PRED_LEN 6 ;;
    ettm1:336|ettm1:720)
        set_if_user_unset FORECAST_STEP_SIZE 6; set_if_user_unset FORECAST_PRED_LEN 12 ;;
    ettm2:96|ettm2:192)
        set_if_user_unset FORECAST_STEP_SIZE 10; set_if_user_unset FORECAST_PRED_LEN 12; set_if_user_unset HUBER_DELTA 0.4 ;;
    ettm2:336)
        set_if_user_unset FORECAST_STEP_SIZE 10; set_if_user_unset FORECAST_PRED_LEN 10; set_if_user_unset HUBER_DELTA 0.27 ;;
    ettm2:720)
        set_if_user_unset FORECAST_STEP_SIZE 4; set_if_user_unset FORECAST_PRED_LEN 8; set_if_user_unset HUBER_DELTA 1.5
        set_if_user_unset PRETRAIN_CONTEXT_POINTS 672; set_if_user_unset TEMPORAL_BACKBONE causal_transformer
        set_if_user_unset D_FF 336; set_if_user_unset DROPOUT 0.1; set_if_user_unset DECODER_LOWPASS 0 ;;
    etth1:96)
        set_if_user_unset FORECAST_STEP_SIZE 3; set_if_user_unset FORECAST_PRED_LEN 6 ;;
    etth1:192|etth1:336)
        set_if_user_unset FORECAST_STEP_SIZE 6; set_if_user_unset FORECAST_PRED_LEN 12 ;;
    etth1:720)
        set_if_user_unset FORECAST_STEP_SIZE 9; set_if_user_unset FORECAST_PRED_LEN 18 ;;
    etth2:96)
        set_if_user_unset FORECAST_STEP_SIZE 3; set_if_user_unset FORECAST_PRED_LEN 9; set_if_user_unset HUBER_DELTA 2.0 ;;
    etth2:192)
        set_if_user_unset FORECAST_STEP_SIZE 6; set_if_user_unset FORECAST_PRED_LEN 12; set_if_user_unset HUBER_DELTA 2.0 ;;
    etth2:336)
        set_if_user_unset FORECAST_STEP_SIZE 7; set_if_user_unset FORECAST_PRED_LEN 14; set_if_user_unset HUBER_DELTA 2.0 ;;
    etth2:720)
        set_if_user_unset FORECAST_STEP_SIZE 4; set_if_user_unset FORECAST_PRED_LEN 6; set_if_user_unset HUBER_DELTA 1.8
        set_if_user_unset FINETUNE_LR 1e-4; set_if_user_unset GUMBEL_TEMPERATURE 0.6 ;;
    electricity:96|electricity:192)
        set_if_user_unset FORECAST_STEP_SIZE 8; set_if_user_unset FORECAST_PRED_LEN 10 ;;
    electricity:336)
        set_if_user_unset FORECAST_STEP_SIZE 20; set_if_user_unset FORECAST_PRED_LEN 24; set_if_user_unset FINETUNE_BATCH_SIZE 16 ;;
    electricity:720)
        set_if_user_unset FORECAST_STEP_SIZE 24; set_if_user_unset FORECAST_PRED_LEN 28
        set_if_user_unset FINETUNE_BATCH_SIZE 8; set_if_user_unset FINETUNE_EPOCHS 30 ;;
    traffic:96)
        set_if_user_unset FORECAST_STEP_SIZE 10; set_if_user_unset FORECAST_PRED_LEN 12 ;;
    traffic:192)
        set_if_user_unset FORECAST_STEP_SIZE 12; set_if_user_unset FORECAST_PRED_LEN 14 ;;
    traffic:336|traffic:720)
        set_if_user_unset FORECAST_STEP_SIZE 24; set_if_user_unset FORECAST_PRED_LEN 28; set_if_user_unset FINETUNE_BATCH_SIZE 10 ;;
    weather:96)
        set_if_user_unset FORECAST_STEP_SIZE 6; set_if_user_unset FORECAST_PRED_LEN 6; set_if_user_unset HUBER_DELTA 2.0 ;;
    weather:192)
        set_if_user_unset FORECAST_STEP_SIZE 8; set_if_user_unset FORECAST_PRED_LEN 10; set_if_user_unset HUBER_DELTA 2.19 ;;
    weather:336)
        set_if_user_unset FORECAST_STEP_SIZE 8; set_if_user_unset FORECAST_PRED_LEN 10; set_if_user_unset HUBER_DELTA 2.2; set_if_user_unset DROPOUT 0.15 ;;
    weather:720)
        set_if_user_unset FORECAST_STEP_SIZE 12; set_if_user_unset FORECAST_PRED_LEN 14; set_if_user_unset HUBER_DELTA 1.5
        set_if_user_unset D_FF 512; set_if_user_unset DROPOUT 0.05; set_if_user_unset TIMEFILTER_TOPK 4 ;;
esac

# Fallback for arbitrary horizons.
if [ -z "${FORECAST_STEP_SIZE:-}" ]; then
    if [ "${OUTPUT_LEN}" -le 96 ]; then
        export FORECAST_STEP_SIZE=3
    elif [ "${OUTPUT_LEN}" -le 192 ]; then
        export FORECAST_STEP_SIZE=6
    elif [ "${OUTPUT_LEN}" -le 336 ]; then
        export FORECAST_STEP_SIZE=8
    else
        export FORECAST_STEP_SIZE=12
    fi
fi
if [ -z "${FORECAST_PRED_LEN:-}" ]; then
    if [ "${OUTPUT_LEN}" -le 96 ]; then
        export FORECAST_PRED_LEN=6
    elif [ "${OUTPUT_LEN}" -le 192 ]; then
        export FORECAST_PRED_LEN=10
    elif [ "${OUTPUT_LEN}" -le 336 ]; then
        export FORECAST_PRED_LEN=12
    else
        export FORECAST_PRED_LEN=18
    fi
fi
export FORECAST_STEP_SIZE_LIST="${FORECAST_STEP_SIZE}"
export FORECAST_PRED_LEN_LIST="${FORECAST_PRED_LEN}"

export FEATURES="${FEATURES:-M}"
export SCALER="${SCALER:-standard}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export REVIN="${REVIN:-1}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
export STREAM_LOGS="${STREAM_LOGS:-1}"
export HUBER_DELTA="${HUBER_DELTA:-1.0}"

cd "${REPO_ROOT}"

if [ "${USE_CORR_CHANNEL_GROUPS}" = "1" ]; then
    mkdir -p "${CHANNEL_GROUPS_DIR}"
    export CHANNEL_GROUPS_FILE="${CHANNEL_GROUPS_FILE:-${CHANNEL_GROUPS_DIR}/${DSET}_freq${MAX_CHANNELS_PER_MODEL}.json}"
    echo "Generating frequency-feature channel groups: ${CHANNEL_GROUPS_FILE}"
    "${PYTHON_BIN}" scripts/make_channel_groups.py \
        --dset "${DSET}" \
        --max_channels "${MAX_CHANNELS_PER_MODEL}" \
        --output "${CHANNEL_GROUPS_FILE}"
fi

echo "================================================="
echo "Single channel-group run"
echo "================================================="
echo "Repo root              : ${REPO_ROOT}"
echo "CUDA_VISIBLE_DEVICES   : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Dataset                : ${DSET}"
echo "Input len              : ${FINETUNE_CONTEXT_POINTS}"
echo "Output len             : ${TARGET_POINTS_LIST}"
echo "Forecast step/pred     : ${FORECAST_STEP_SIZE_LIST} / ${FORECAST_PRED_LEN_LIST}"
echo "Total/max channels     : ${TOTAL_CHANNELS} / ${MAX_CHANNELS_PER_MODEL}"
echo "Patch/comp/embed       : ${PATCH_SIZE} / ${COMPRESSION_FACTOR} / ${EMBEDDING_DIM}"
echo "Temporal backbone      : ${TEMPORAL_BACKBONE}"
echo "D_FF / Dropout         : ${D_FF} / ${DROPOUT}"
echo "Finetune epochs/bs/lr  : ${FINETUNE_EPOCHS} / ${FINETUNE_BATCH_SIZE} / ${FINETUNE_LR}"
echo "Finetune loss          : ${TRAIN_LOSS} (huber_delta=${HUBER_DELTA})"
echo "Run history prefix     : ${RUN_HISTORY_PREFIX}"
echo "Resume latest run      : ${RESUME_LATEST_RUN}"
echo "Force rerun all        : ${FORCE_RETRAIN_ALL}"
echo "Force rerun pretrain   : ${FORCE_RETRAIN_PRETRAIN}"
echo "Freq channel grouping  : ${USE_CORR_CHANNEL_GROUPS}"
echo "================================================="

bash scripts/decoder_only_NTP/channel_group_pipeline.sh
