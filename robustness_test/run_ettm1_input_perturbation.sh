#!/bin/bash
# Input perturbation robustness test on ETTm1.
# Fill the four finetuned checkpoint paths below before running.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---- Fill with either one checkpoint or horizon-specific checkpoints ----
CKPT="${CKPT:-}"
CKPT_96="${CKPT_96:-}"
CKPT_192="${CKPT_192:-}"
CKPT_336="${CKPT_336:-}"
CKPT_720="${CKPT_720:-}"
TARGETS="${TARGETS:-}"
if [ -z "${TARGETS}" ]; then
    if [ -n "${CKPT}" ]; then
        TARGETS="96"
    else
        TARGETS="96,192,336,720"
    fi
fi

if [ -z "${CKPT}" ] && { [ -z "${CKPT_96}" ] || [ -z "${CKPT_192}" ] || [ -z "${CKPT_336}" ] || [ -z "${CKPT_720}" ]; }; then
    echo "ERROR: set CKPT=/path/to/model.pth for one target, or set CKPT_96/192/336/720." >&2
    echo "Examples:" >&2
    echo "  CKPT=/path/to/tw96.pth TARGETS=96 bash robustness_test/run_ettm1_input_perturbation.sh" >&2
    echo "  CKPT_96=/path/to/tw96.pth CKPT_192=/path/to/tw192.pth CKPT_336=/path/to/tw336.pth CKPT_720=/path/to/tw720.pth bash robustness_test/run_ettm1_input_perturbation.sh" >&2
    exit 1
fi

"${PYTHON_BIN}" robustness_test/input_perturbation_robustness.py \
    --dset ettm1 \
    --context_points "${CONTEXT_POINTS:-96}" \
    --targets "${TARGETS}" \
    --checkpoint "${CKPT}" \
    --ckpt96 "${CKPT_96}" \
    --ckpt192 "${CKPT_192}" \
    --ckpt336 "${CKPT_336}" \
    --ckpt720 "${CKPT_720}" \
    --ar_steps "${AR_STEPS:-3,6,6,9}" \
    --pred_lens "${PRED_LENS:-6,12,12,18}" \
    --sigmas "${SIGMAS:-0.00,0.05,0.10,0.15,0.20,0.30}" \
    --seeds "${SEEDS:-42,43,44}" \
    --batch_size "${BATCH_SIZE:-32}" \
    --num_workers "${NUM_WORKERS:-0}" \
    --features "${FEATURES:-M}" \
    --scaler "${SCALER:-standard}" \
    --revin "${REVIN:-1}" \
    --model_ctor "${MODEL_CTOR:-src.models.patch_vqvae_transformer:PatchVQVAETransformer}" \
    --model_init_mode "${MODEL_INIT_MODE:-config}" \
    --model_config_key "${MODEL_CONFIG_KEY:-}" \
    --state_dict_key "${STATE_DICT_KEY:-}" \
    --strict_load "${STRICT_LOAD:-1}" \
    --strip_state_dict_prefix "${STRIP_STATE_DICT_PREFIX:-auto}" \
    --model_forward_method "${MODEL_FORWARD_METHOD:-auto}" \
    --forward_takes_target_len "${FORWARD_TAKES_TARGET_LEN:-auto}" \
    --prediction_key "${PREDICTION_KEY:-}" \
    --inject_n_channels "${INJECT_N_CHANNELS:-1}" \
    --inject_gumbel_args "${INJECT_GUMBEL_ARGS:-1}" \
    --use_gumbel_softmax "${USE_GUMBEL_SOFTMAX:-1}" \
    --gumbel_temperature "${GUMBEL_TEMPERATURE:-0.8}" \
    --gumbel_hard "${GUMBEL_HARD:-0}" \
    --noise_position "${NOISE_POSITION:-after_revin}" \
    --noise_scale "${NOISE_SCALE:-unit}" \
    --output_dir "${OUTPUT_DIR:-robustness_test/results/ettm1_input_perturbation}"
