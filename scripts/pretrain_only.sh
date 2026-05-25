#!/bin/bash
# =====================================================================
# Standalone PatchVQVAE Transformer pretrain runner.
#
# Usage from repo root:
#   bash scripts/pretrain_only.sh --dset etth2 --context_points 296 \
#     --progressive_step_size 3 --vqvae_checkpoint /path/to/codebook.pth
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
  bash scripts/pretrain_only.sh --dset <dataset> --context_points <len> --vqvae_checkpoint <pth> [options] [-- extra python args]

Required arguments, or equivalent env vars:
  --dset, --dataset          Dataset name. Env: DSET
  --context_points, --input_len
                             Pretrain input length. Env: PRETRAIN_CONTEXT_POINTS or CONTEXT_POINTS or INPUT_LEN
  --vqvae_checkpoint, --codebook_ckpt
                             Codebook checkpoint from train_codebook_only.sh. Env: VQVAE_CHECKPOINT or CODEBOOK_CKPT

Common options:
  --progressive_step_size N   Progressive pretrain step size in patches. Env: PROGRESSIVE_STEP_SIZE
  --pred_len N                Pretrain prediction length in patches. Env: PRETRAIN_PRED_LEN
  --save_path PATH            Save root. Final ckpt is under PATH/<dset>/
  --model_id N                Model id. Env: MODEL_ID
  --channel_start N           Start channel, inclusive
  --channel_end N             End channel, exclusive
  --channel_indices LIST      Comma-separated channel indices, e.g. 0,2,5
  --channel_group_id N        Group id suffix when using channel_indices
  -h, --help                  Show this help

Useful env overrides:
  PATCH_SIZE COMPRESSION_FACTOR EMBEDDING_DIM CODEBOOK_SIZE N_LAYERS N_HEADS D_FF
  TEMPORAL_BACKBONE TIMEFILTER_TOPK PRETRAIN_BATCH_SIZE PRETRAIN_EPOCHS PRETRAIN_LR
  PER_CHANNEL_CODEBOOK N_RQ_LAYERS RQ_LAYER_WEIGHTS SOFT_NEIGHBOR_K

Examples:
  bash scripts/pretrain_only.sh --dset etth2 --context_points 296 --progressive_step_size 3 --vqvae_checkpoint /abs/codebook.pth
  TEMPORAL_BACKBONE=timefilter_lite TIMEFILTER_TOPK=8 bash scripts/pretrain_only.sh --dset weather --input_len 672 --vqvae_checkpoint /abs/codebook.pth
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --) shift; EXTRA_PY_ARGS=("$@"); break ;;
        --dset|--dataset) DSET="$2"; shift 2 ;;
        --context_points|--context-points|--input_len|--input-len) PRETRAIN_CONTEXT_POINTS="$2"; shift 2 ;;
        --vqvae_checkpoint|--vqvae-checkpoint|--codebook_ckpt|--codebook-ckpt) VQVAE_CHECKPOINT="$2"; shift 2 ;;
        --progressive_step_size|--progressive-step-size|--step_size|--step-size) PROGRESSIVE_STEP_SIZE="$2"; shift 2 ;;
        --pred_len|--pred-len|--pretrain_pred_len|--pretrain-pred-len) PRETRAIN_PRED_LEN="$2"; shift 2 ;;
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
PRETRAIN_CONTEXT_POINTS="${PRETRAIN_CONTEXT_POINTS:-${CONTEXT_POINTS:-${INPUT_LEN:-}}}"
VQVAE_CHECKPOINT="${VQVAE_CHECKPOINT:-${CODEBOOK_CKPT:-}}"
PROGRESSIVE_STEP_SIZE="${PROGRESSIVE_STEP_SIZE:-6}"
if [ -z "${DSET}" ] || [ -z "${PRETRAIN_CONTEXT_POINTS}" ] || [ -z "${VQVAE_CHECKPOINT}" ]; then
    echo "ERROR: --dset, --context_points and --vqvae_checkpoint are required." >&2
    usage >&2
    exit 1
fi
if [ ! -f "${VQVAE_CHECKPOINT}" ]; then
    echo "ERROR: vqvae checkpoint not found: ${VQVAE_CHECKPOINT}" >&2
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
case "${PRETRAIN_CONTEXT_POINTS}" in
    ''|*[!0-9]*) echo "ERROR: context_points must be a positive integer: ${PRETRAIN_CONTEXT_POINTS}" >&2; exit 1 ;;
esac
case "${PROGRESSIVE_STEP_SIZE}" in
    ''|*[!0-9]*) echo "ERROR: progressive_step_size must be a positive integer: ${PROGRESSIVE_STEP_SIZE}" >&2; exit 1 ;;
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

RUN_NAME="${RUN_NAME:-${DSET}_pretrain_cw${PRETRAIN_CONTEXT_POINTS}_step${PROGRESSIVE_STEP_SIZE}_$(date +%Y%m%d_%H%M%S)}"
SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/decoder_only_NTP/saved_models/patch_vqvae/${RUN_NAME}}"

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
PER_CHANNEL_CODEBOOK="${PER_CHANNEL_CODEBOOK:-0}"
N_RQ_LAYERS="${N_RQ_LAYERS:-2}"
N_LAYERS="${N_LAYERS:-3}"
N_HEADS="${N_HEADS:-4}"
D_FF="${D_FF:-256}"
DROPOUT="${DROPOUT:-0.1}"
TEMPORAL_BACKBONE="${TEMPORAL_BACKBONE:-causal_transformer}"
TIMEFILTER_TOPK="${TIMEFILTER_TOPK:-8}"
TIMEFILTER_TEMPERATURE="${TIMEFILTER_TEMPERATURE:-1.0}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-64}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_LR="${PRETRAIN_LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
REVIN="${REVIN:-1}"
SEED="${SEED:-42}"
FEATURES="${FEATURES:-M}"
SCALER="${SCALER:-standard}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MODEL_ID="${MODEL_ID:-1}"
FREEZE_VQVAE="${FREEZE_VQVAE:-1}"
LOAD_VQ_WEIGHTS="${LOAD_VQ_WEIGHTS:-1}"
DISABLE_EMA_UPDATE="${DISABLE_EMA_UPDATE:-1}"
VQ_WEIGHT="${VQ_WEIGHT:-0.0}"
RECON_WEIGHT="${RECON_WEIGHT:-0.0}"
SOFT_NEIGHBOR_K="${SOFT_NEIGHBOR_K:-0}"
SOFT_NEIGHBOR_ALPHA="${SOFT_NEIGHBOR_ALPHA:-0.25}"
SOFT_NEIGHBOR_TAU="${SOFT_NEIGHBOR_TAU:-0.3}"
USE_RAW_INPUT="${USE_RAW_INPUT:-0}"

CMD=("${PYTHON_BIN}" -u patch_vqvae_pretrain.py
    --dset "${DSET}"
    --context_points "${PRETRAIN_CONTEXT_POINTS}"
    --progressive_step_size "${PROGRESSIVE_STEP_SIZE}"
    --batch_size "${PRETRAIN_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --scaler "${SCALER}"
    --features "${FEATURES}"
    --patch_size "${PATCH_SIZE}"
    --embedding_dim "${EMBEDDING_DIM}"
    --compression_factor "${COMPRESSION_FACTOR}"
    --codebook_size "${CODEBOOK_SIZE}"
    --n_layers "${N_LAYERS}"
    --n_heads "${N_HEADS}"
    --d_ff "${D_FF}"
    --dropout "${DROPOUT}"
    --temporal_backbone "${TEMPORAL_BACKBONE}"
    --timefilter_topk "${TIMEFILTER_TOPK}"
    --timefilter_temperature "${TIMEFILTER_TEMPERATURE}"
    --num_hiddens "${NUM_HIDDENS}"
    --num_residual_layers "${NUM_RESIDUAL_LAYERS}"
    --num_residual_hiddens "${NUM_RESIDUAL_HIDDENS}"
    --vqvae_backbone "${VQVAE_BACKBONE}"
    --vqvae_tcn_kernel_size "${VQVAE_TCN_KERNEL_SIZE}"
    --vqvae_chunk_size "${VQVAE_CHUNK_SIZE}"
    --decoder_lowpass "${DECODER_LOWPASS}"
    --decoder_lowpass_kernel "${DECODER_LOWPASS_KERNEL}"
    --vqvae_checkpoint "${VQVAE_CHECKPOINT}"
    --freeze_vqvae "${FREEZE_VQVAE}"
    --load_vq_weights "${LOAD_VQ_WEIGHTS}"
    --disable_ema_update "${DISABLE_EMA_UPDATE}"
    --per_channel_codebook "${PER_CHANNEL_CODEBOOK}"
    --n_rq_layers "${N_RQ_LAYERS}"
    --soft_neighbor_k "${SOFT_NEIGHBOR_K}"
    --soft_neighbor_alpha "${SOFT_NEIGHBOR_ALPHA}"
    --soft_neighbor_tau "${SOFT_NEIGHBOR_TAU}"
    --use_raw_input "${USE_RAW_INPUT}"
    --n_epochs "${PRETRAIN_EPOCHS}"
    --lr "${PRETRAIN_LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --seed "${SEED}"
    --revin "${REVIN}"
    --vq_weight "${VQ_WEIGHT}"
    --recon_weight "${RECON_WEIGHT}"
    --save_path "${SAVE_PATH}"
    --model_id "${MODEL_ID}"
)

[ -n "${PRETRAIN_PRED_LEN:-}" ] && CMD+=(--pred_len "${PRETRAIN_PRED_LEN}")
[ -n "${RQ_LAYER_WEIGHTS:-}" ] && read -r -a RQ_LAYER_WEIGHTS_ARR <<< "${RQ_LAYER_WEIGHTS}" && CMD+=(--rq_layer_weights "${RQ_LAYER_WEIGHTS_ARR[@]}")
[ -n "${CHANNEL_START:-}" ] && CMD+=(--channel_start "${CHANNEL_START}")
[ -n "${CHANNEL_END:-}" ] && CMD+=(--channel_end "${CHANNEL_END}")
[ -n "${CHANNEL_INDICES:-}" ] && CMD+=(--channel_indices "${CHANNEL_INDICES}")
[ -n "${CHANNEL_GROUP_ID:-}" ] && CMD+=(--channel_group_id "${CHANNEL_GROUP_ID}")
CMD+=("${EXTRA_PY_ARGS[@]}")

cd "${REPO_ROOT}/decoder_only_NTP"
echo "================================================="
echo "Standalone PatchVQVAE pretrain"
echo "================================================="
echo "Dataset        : ${DSET}"
echo "Context points : ${PRETRAIN_CONTEXT_POINTS}"
echo "Step / pred_len: ${PROGRESSIVE_STEP_SIZE} / ${PRETRAIN_PRED_LEN:-<default=step>}"
echo "Codebook ckpt  : ${VQVAE_CHECKPOINT}"
echo "Save path      : ${SAVE_PATH}/${DSET}"
echo "Model id       : ${MODEL_ID}"
echo "GPU            : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "================================================="
"${CMD[@]}"
