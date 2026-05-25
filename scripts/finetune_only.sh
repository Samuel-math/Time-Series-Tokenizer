#!/bin/bash
# =====================================================================
# Standalone PatchVQVAE Transformer finetune runner.
#
# Usage from repo root:
#   bash scripts/finetune_only.sh --dset etth2 --input_len 96 --output_len 336 \
#     --pretrained_model /path/to/pretrain.pth
#
# GPU is intentionally not specified here. Set CUDA_VISIBLE_DEVICES outside
# this script if needed.
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTRA_PY_ARGS=()

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/finetune_only.sh --dset <dataset> --input_len <len> --output_len <len> --pretrained_model <pth> [options] [-- extra python args]

Required arguments, or equivalent env vars:
  --dset, --dataset          Dataset name. Env: DSET
  --input_len                Finetune input/context length. Env: INPUT_LEN or FINETUNE_CONTEXT_POINTS
  --output_len               Forecast horizon/target length. Env: OUTPUT_LEN or TARGET_POINTS
  --pretrained_model         Pretrain checkpoint. Env: PRETRAINED_MODEL or PRETRAIN_CKPT

Common options:
  --ar_step_size N           Autoregressive step size in patches. Env: AR_STEP_SIZE or FORECAST_STEP_SIZE
  --pred_len N               Per-forward prediction length in patches. Env: FORECAST_PRED_LEN
  --save_path PATH           Save root. Final ckpt is under PATH/<dset>/
  --model_id N               Model id. Env: MODEL_ID
  --channel_start N          Start channel, inclusive
  --channel_end N            End channel, exclusive
  --channel_indices LIST     Comma-separated channel indices, e.g. 0,2,5
  --channel_group_id N       Group id suffix when using channel_indices
  -h, --help                 Show this help

Useful env overrides:
  FINETUNE_BATCH_SIZE FINETUNE_EPOCHS FINETUNE_LR WEIGHT_DECAY TRAIN_LOSS HUBER_DELTA
  USE_GUMBEL_SOFTMAX GUMBEL_TEMPERATURE GUMBEL_HARD UNFREEZE_DECODER DECODER_LR_RATIO

Examples:
  bash scripts/finetune_only.sh --dset etth2 --input_len 96 --output_len 720 --pretrained_model /abs/pretrain.pth
  FORECAST_STEP_SIZE=4 FORECAST_PRED_LEN=6 bash scripts/finetune_only.sh --dset etth2 --input_len 96 --output_len 720 --pretrained_model /abs/pretrain.pth
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --) shift; EXTRA_PY_ARGS=("$@"); break ;;
        --dset|--dataset) DSET="$2"; shift 2 ;;
        --input_len|--input-len|--context_points|--context-points) FINETUNE_CONTEXT_POINTS="$2"; shift 2 ;;
        --output_len|--output-len|--target_points|--target-points) TARGET_POINTS="$2"; shift 2 ;;
        --pretrained_model|--pretrained-model|--pretrain_ckpt|--pretrain-ckpt) PRETRAINED_MODEL="$2"; shift 2 ;;
        --ar_step_size|--ar-step-size|--forecast_step_size|--forecast-step-size) AR_STEP_SIZE="$2"; shift 2 ;;
        --pred_len|--pred-len|--forecast_pred_len|--forecast-pred-len) FORECAST_PRED_LEN="$2"; shift 2 ;;
        --save_path|--save-path) SAVE_PATH="$2"; shift 2 ;;
        --model_id|--model-id) MODEL_ID="$2"; shift 2 ;;
        --channel_start|--channel-start) CHANNEL_START="$2"; shift 2 ;;
        --channel_end|--channel-end) CHANNEL_END="$2"; shift 2 ;;
        --channel_indices|--channel-indices) CHANNEL_INDICES="$2"; shift 2 ;;
        --channel_group_id|--channel-group-id) CHANNEL_GROUP_ID="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 1 ;;
    esac
done

DSET="${DSET:-}"
FINETUNE_CONTEXT_POINTS="${FINETUNE_CONTEXT_POINTS:-${INPUT_LEN:-}}"
TARGET_POINTS="${TARGET_POINTS:-${OUTPUT_LEN:-}}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${PRETRAIN_CKPT:-}}"
AR_STEP_SIZE="${AR_STEP_SIZE:-${FORECAST_STEP_SIZE:-}}"
if [ -z "${DSET}" ] || [ -z "${FINETUNE_CONTEXT_POINTS}" ] || [ -z "${TARGET_POINTS}" ] || [ -z "${PRETRAINED_MODEL}" ]; then
    echo "ERROR: --dset, --input_len, --output_len and --pretrained_model are required." >&2
    usage >&2
    exit 1
fi
if [ ! -f "${PRETRAINED_MODEL}" ]; then
    echo "ERROR: pretrained model not found: ${PRETRAINED_MODEL}" >&2
    exit 1
fi

DSET="$(printf '%s' "${DSET}" | tr '[:upper:]' '[:lower:]')"
case "${DSET}" in
    ecl) DSET=electricity ;;
esac
case "${DSET}" in
    ettm1|ettm2|etth1|etth2|electricity|traffic|weather|illness|exchange) ;;
    *) echo "ERROR: unsupported dataset '${DSET}'" >&2; exit 1 ;;
esac
case "${FINETUNE_CONTEXT_POINTS}" in
    ''|*[!0-9]*) echo "ERROR: input_len must be a positive integer: ${FINETUNE_CONTEXT_POINTS}" >&2; exit 1 ;;
esac
case "${TARGET_POINTS}" in
    ''|*[!0-9]*) echo "ERROR: output_len must be a positive integer: ${TARGET_POINTS}" >&2; exit 1 ;;
esac

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

RUN_NAME="${RUN_NAME:-${DSET}_finetune_cw${FINETUNE_CONTEXT_POINTS}_tw${TARGET_POINTS}_$(date +%Y%m%d_%H%M%S)}"
SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/decoder_only_NTP/saved_models/patch_vqvae_finetune/${RUN_NAME}}"

FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-64}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"
FINETUNE_LR="${FINETUNE_LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
REVIN="${REVIN:-1}"
AMP="${AMP:-1}"
SEED="${SEED:-42}"
FEATURES="${FEATURES:-M}"
SCALER="${SCALER:-standard}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MODEL_ID="${MODEL_ID:-1}"
TRAIN_LOSS="${TRAIN_LOSS:-huber}"
HUBER_DELTA="${HUBER_DELTA:-1.0}"
USE_GUMBEL_SOFTMAX="${USE_GUMBEL_SOFTMAX:-1}"
GUMBEL_TEMPERATURE="${GUMBEL_TEMPERATURE:-0.8}"
GUMBEL_HARD="${GUMBEL_HARD:-0}"
UNFREEZE_DECODER="${UNFREEZE_DECODER:-0}"
DECODER_LR_RATIO="${DECODER_LR_RATIO:-1}"
DECODER_WD_RATIO="${DECODER_WD_RATIO:-1}"

CMD=("${PYTHON_BIN}" -u patch_vqvae_finetune.py
    --dset "${DSET}"
    --context_points "${FINETUNE_CONTEXT_POINTS}"
    --target_points "${TARGET_POINTS}"
    --batch_size "${FINETUNE_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --scaler "${SCALER}"
    --features "${FEATURES}"
    --pretrained_model "${PRETRAINED_MODEL}"
    --n_epochs "${FINETUNE_EPOCHS}"
    --lr "${FINETUNE_LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --revin "${REVIN}"
    --amp "${AMP}"
    --seed "${SEED}"
    --train_loss "${TRAIN_LOSS}"
    --huber_delta "${HUBER_DELTA}"
    --unfreeze_decoder "${UNFREEZE_DECODER}"
    --decoder_lr_ratio "${DECODER_LR_RATIO}"
    --decoder_wd_ratio "${DECODER_WD_RATIO}"
    --use_gumbel_softmax "${USE_GUMBEL_SOFTMAX}"
    --gumbel_temperature "${GUMBEL_TEMPERATURE}"
    --gumbel_hard "${GUMBEL_HARD}"
    --save_path "${SAVE_PATH}"
    --model_id "${MODEL_ID}"
)

[ -n "${AR_STEP_SIZE:-}" ] && CMD+=(--ar_step_size "${AR_STEP_SIZE}")
[ -n "${FORECAST_PRED_LEN:-}" ] && CMD+=(--pred_len "${FORECAST_PRED_LEN}")
[ -n "${CHANNEL_START:-}" ] && CMD+=(--channel_start "${CHANNEL_START}")
[ -n "${CHANNEL_END:-}" ] && CMD+=(--channel_end "${CHANNEL_END}")
[ -n "${CHANNEL_INDICES:-}" ] && CMD+=(--channel_indices "${CHANNEL_INDICES}")
[ -n "${CHANNEL_GROUP_ID:-}" ] && CMD+=(--channel_group_id "${CHANNEL_GROUP_ID}")
CMD+=("${EXTRA_PY_ARGS[@]}")

cd "${REPO_ROOT}/decoder_only_NTP"
echo "================================================="
echo "Standalone PatchVQVAE finetune"
echo "================================================="
echo "Dataset        : ${DSET}"
echo "Input / output : ${FINETUNE_CONTEXT_POINTS} / ${TARGET_POINTS}"
echo "AR step/pred   : ${AR_STEP_SIZE:-<inherit>} / ${FORECAST_PRED_LEN:-<inherit>}"
echo "Pretrain ckpt  : ${PRETRAINED_MODEL}"
echo "Save path      : ${SAVE_PATH}/${DSET}"
echo "Model id       : ${MODEL_ID}"
echo "GPU            : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "================================================="
"${CMD[@]}"
