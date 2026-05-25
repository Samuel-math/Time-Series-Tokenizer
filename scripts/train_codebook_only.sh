#!/bin/bash
# =====================================================================
# Standalone codebook/VQVAE training runner.
#
# Usage from repo root:
#   bash scripts/train_codebook_only.sh --dset etth2 --context_points 512
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_codebook_only.sh --dset weather --context_points 672
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
  bash scripts/train_codebook_only.sh --dset <dataset> --context_points <len> [options] [-- extra python args]

Required arguments, or equivalent env vars:
  --dset, --dataset          Dataset name. Env: DSET
  --context_points, --input_len
                             Codebook/VQVAE input length. Env: CONTEXT_POINTS or INPUT_LEN

Common options:
  --save_path PATH           Save root. Final ckpt is under PATH/<dset>/
  --model_id N               Model id. Env: MODEL_ID
  --channel_start N          Start channel, inclusive
  --channel_end N            End channel, exclusive
  --channel_indices LIST     Comma-separated channel indices, e.g. 0,2,5
  --channel_group_id N       Group id suffix when using channel_indices
  -h, --help                 Show this help

Useful env overrides:
  PATCH_SIZE COMPRESSION_FACTOR EMBEDDING_DIM CODEBOOK_SIZE NUM_HIDDENS
  NUM_RESIDUAL_LAYERS NUM_RESIDUAL_HIDDENS VQVAE_BACKBONE N_RQ_LAYERS
  CB_BATCH_SIZE CB_EPOCHS CB_LR WEIGHT_DECAY SPARSE_WEIGHT LAMBDA_ORD

Examples:
  bash scripts/train_codebook_only.sh --dset etth2 --context_points 512
  PATCH_SIZE=8 COMPRESSION_FACTOR=4 EMBEDDING_DIM=64 bash scripts/train_codebook_only.sh --dset ettm2 --input_len 512
  bash scripts/train_codebook_only.sh --dset traffic --context_points 336 --channel_start 0 --channel_end 128
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --) shift; EXTRA_PY_ARGS=("$@"); break ;;
        --dset|--dataset) DSET="$2"; shift 2 ;;
        --context_points|--context-points|--input_len|--input-len) CONTEXT_POINTS="$2"; shift 2 ;;
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
CONTEXT_POINTS="${CONTEXT_POINTS:-${INPUT_LEN:-}}"
if [ -z "${DSET}" ] || [ -z "${CONTEXT_POINTS}" ]; then
    echo "ERROR: --dset and --context_points are required." >&2
    usage >&2
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
case "${CONTEXT_POINTS}" in
    ''|*[!0-9]*) echo "ERROR: context_points must be a positive integer: ${CONTEXT_POINTS}" >&2; exit 1 ;;
esac
if [ "${CONTEXT_POINTS}" -le 0 ]; then
    echo "ERROR: context_points must be positive." >&2
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

RUN_NAME="${RUN_NAME:-${DSET}_codebook_cw${CONTEXT_POINTS}_$(date +%Y%m%d_%H%M%S)}"
SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/vqvae-only/saved_models/vqvae_only/${RUN_NAME}}"

PATCH_SIZE="${PATCH_SIZE:-8}"
COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
NUM_HIDDENS="${NUM_HIDDENS:-128}"
NUM_RESIDUAL_LAYERS="${NUM_RESIDUAL_LAYERS:-2}"
NUM_RESIDUAL_HIDDENS="${NUM_RESIDUAL_HIDDENS:-128}"
VQVAE_BACKBONE="${VQVAE_BACKBONE:-mlp}"
VQVAE_TCN_KERNEL_SIZE="${VQVAE_TCN_KERNEL_SIZE:-5}"
VQVAE_CHUNK_SIZE="${VQVAE_CHUNK_SIZE:-2}"
DECODER_LOWPASS="${DECODER_LOWPASS:-0}"
DECODER_LOWPASS_KERNEL="${DECODER_LOWPASS_KERNEL:-binomial3}"
CODEBOOK_EMA="${CODEBOOK_EMA:-1}"
EMA_DECAY="${EMA_DECAY:-0.95}"
EMA_EPS="${EMA_EPS:-1e-5}"
PER_CHANNEL_CODEBOOK="${PER_CHANNEL_CODEBOOK:-0}"
N_RQ_LAYERS="${N_RQ_LAYERS:-2}"
CB_BATCH_SIZE="${CB_BATCH_SIZE:-64}"
CB_EPOCHS="${CB_EPOCHS:-50}"
CB_LR="${CB_LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
REVIN="${REVIN:-1}"
AMP="${AMP:-1}"
SEED="${SEED:-42}"
FEATURES="${FEATURES:-M}"
SCALER="${SCALER:-standard}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MODEL_ID="${MODEL_ID:-1}"
SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.3}"
SPARSE_AMPLITUDE="${SPARSE_AMPLITUDE:-0.05}"
LAMBDA_ORD="${LAMBDA_ORD:-0.01}"
ORTH_WEIGHT="${ORTH_WEIGHT:-0.01}"
ORTH_START_EPOCH="${ORTH_START_EPOCH:-0}"
ORTH_WARMUP_EPOCHS="${ORTH_WARMUP_EPOCHS:-5}"

CMD=("${PYTHON_BIN}" -u codebook_pretrain.py
    --dset "${DSET}"
    --context_points "${CONTEXT_POINTS}"
    --target_points 0
    --batch_size "${CB_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --scaler "${SCALER}"
    --features "${FEATURES}"
    --patch_size "${PATCH_SIZE}"
    --embedding_dim "${EMBEDDING_DIM}"
    --compression_factor "${COMPRESSION_FACTOR}"
    --codebook_size "${CODEBOOK_SIZE}"
    --num_hiddens "${NUM_HIDDENS}"
    --num_residual_layers "${NUM_RESIDUAL_LAYERS}"
    --num_residual_hiddens "${NUM_RESIDUAL_HIDDENS}"
    --vqvae_backbone "${VQVAE_BACKBONE}"
    --vqvae_tcn_kernel_size "${VQVAE_TCN_KERNEL_SIZE}"
    --vqvae_chunk_size "${VQVAE_CHUNK_SIZE}"
    --decoder_lowpass "${DECODER_LOWPASS}"
    --decoder_lowpass_kernel "${DECODER_LOWPASS_KERNEL}"
    --codebook_ema "${CODEBOOK_EMA}"
    --ema_decay "${EMA_DECAY}"
    --ema_eps "${EMA_EPS}"
    --per_channel_codebook "${PER_CHANNEL_CODEBOOK}"
    --n_rq_layers "${N_RQ_LAYERS}"
    --n_epochs "${CB_EPOCHS}"
    --lr "${CB_LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --revin "${REVIN}"
    --amp "${AMP}"
    --seed "${SEED}"
    --sparse_weight "${SPARSE_WEIGHT}"
    --sparse_amplitude "${SPARSE_AMPLITUDE}"
    --lambda_ord "${LAMBDA_ORD}"
    --orth_weight "${ORTH_WEIGHT}"
    --orth_start_epoch "${ORTH_START_EPOCH}"
    --orth_warmup_epochs "${ORTH_WARMUP_EPOCHS}"
    --save_path "${SAVE_PATH}"
    --model_id "${MODEL_ID}"
)

[ -n "${CHANNEL_START:-}" ] && CMD+=(--channel_start "${CHANNEL_START}")
[ -n "${CHANNEL_END:-}" ] && CMD+=(--channel_end "${CHANNEL_END}")
[ -n "${CHANNEL_INDICES:-}" ] && CMD+=(--channel_indices "${CHANNEL_INDICES}")
[ -n "${CHANNEL_GROUP_ID:-}" ] && CMD+=(--channel_group_id "${CHANNEL_GROUP_ID}")
CMD+=("${EXTRA_PY_ARGS[@]}")

cd "${REPO_ROOT}/vqvae-only"
echo "================================================="
echo "Standalone codebook training"
echo "================================================="
echo "Dataset        : ${DSET}"
echo "Context points : ${CONTEXT_POINTS}"
echo "Save path      : ${SAVE_PATH}/${DSET}"
echo "Model id       : ${MODEL_ID}"
echo "GPU            : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "================================================="
"${CMD[@]}"
