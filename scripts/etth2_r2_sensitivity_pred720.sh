#!/bin/bash
# =====================================================================
# ETTh2 pred=720 R2 sensitivity sweep (finetune-only variable).
#
# Sweep:
#   M (AR_STEP_SIZE) in {2, 3, 4, 5}
#   PRED_LEN in {M, 2M, 4M, 8M}
#
# Notes:
# - Base hyper-parameters aligned with scripts/etth2_best.sh
#   (patch=8, embed=64, num_hiddens=128, RQ=2, etc.)
# - target_points fixed to 720; finetune lr / gumbel / huber take the
#   pred=720 column from etth2_best.sh (lr=1e-4, tau=0.6, huber=1.8).
# - Codebook/pretrain artifacts shared across all sweep runs to avoid
#   repeated expensive training; only finetune config varies per combo.
#
# Usage:
#   bash scripts/etth2_r2_sensitivity_pred720.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -------------------------------
# Base config (from etth2_best.sh)
# -------------------------------
export DSET=etth2
export TOTAL_CHANNELS=7
export MAX_CHANNELS_PER_MODEL=7
export BASE_MODEL_ID="${BASE_MODEL_ID:-1}"
export FORCE_RETRAIN_ALL=0
export FORCE_RETRAIN_PRETRAIN=0
export RESUME_LATEST_RUN=1
export USE_CORR_CHANNEL_GROUPS=1
export CHANNEL_GROUPS_DIR="${CHANNEL_GROUPS_DIR:-scripts/channel_groups}"
export RETAIN_RUNS="${RETAIN_RUNS:-5}"

# VQVAE / codebook params (match etth2_best.sh)
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

# NTP pretrain params (match etth2_best.sh)
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

# Finetune (pred=720 column from etth2_best.sh)
export FINETUNE_CONTEXT_POINTS=96
export FINETUNE_EPOCHS=50
export FINETUNE_BATCH_SIZE=32
export FINETUNE_LR=1e-4
export TARGET_POINTS_LIST="720"
export FINETUNE_LR_LIST="1e-4"
export USE_GUMBEL_SOFTMAX=1
export GUMBEL_TEMPERATURE=0.6
export GUMBEL_TEMPERATURE_LIST="0.6"
export GUMBEL_HARD=0
export TRAIN_LOSS=huber
export HUBER_DELTA=1.8
export HUBER_DELTA_LIST="1.8"

export FEATURES=M
export SCALER=standard
export NUM_WORKERS="${NUM_WORKERS:-0}"
export REVIN=1
export WEIGHT_DECAY=1e-4
export STREAM_LOGS="${STREAM_LOGS:-1}"

# -------------------------------
# Sweep definition
# -------------------------------
M_LIST="${M_LIST:-2 3 4 5}"
RATIO_LIST="${RATIO_LIST:-1 2 4 8}"

# -------------------------------
# Shared artifact roots
# -------------------------------
SWEEP_TAG="${SWEEP_TAG:-etth2_pred720_r2_sweep_$(date +%Y%m%d_%H%M%S)}"
RUN_HISTORY_PREFIX="${RUN_HISTORY_PREFIX:-${SWEEP_TAG}}"
SHARED_GROUP_RUN_NAME="${SHARED_GROUP_RUN_NAME:-${SWEEP_TAG}_shared_pretrain}"

export CB_SAVE_PATH="${CB_SAVE_PATH:-${REPO_ROOT}/saved_models/vqvae_only/${SHARED_GROUP_RUN_NAME}}"
export PRETRAIN_SAVE_PATH="${PRETRAIN_SAVE_PATH:-${REPO_ROOT}/saved_models/patch_vqvae/${SHARED_GROUP_RUN_NAME}}"

MASTER_LOG_DIR="${MASTER_LOG_DIR:-${REPO_ROOT}/logs/${SWEEP_TAG}}"
mkdir -p "${MASTER_LOG_DIR}"
SUMMARY_TSV="${MASTER_LOG_DIR}/r2_sweep_summary.tsv"
SUMMARY_MD="${MASTER_LOG_DIR}/r2_sweep_summary.md"

echo -e "M\trequested_ratio\tpred_len\tactual_ratio\tweighted_mse\tweighted_mae\tlog_dir" > "${SUMMARY_TSV}"

echo "================================================="
echo "ETTh2 pred=720 R2 sensitivity sweep"
echo "================================================="
echo "Sweep tag              : ${SWEEP_TAG}"
echo "M list                 : ${M_LIST}"
echo "ratio list             : ${RATIO_LIST}"
echo "Shared CB save_path    : ${CB_SAVE_PATH}"
echo "Shared PRE save_path   : ${PRETRAIN_SAVE_PATH}"
echo "Master log dir         : ${MASTER_LOG_DIR}"
echo "================================================="

if [ "${USE_CORR_CHANNEL_GROUPS}" = "1" ]; then
    mkdir -p "${CHANNEL_GROUPS_DIR}"
    export CHANNEL_GROUPS_FILE="${CHANNEL_GROUPS_FILE:-${CHANNEL_GROUPS_DIR}/${DSET}_freq${MAX_CHANNELS_PER_MODEL}.json}"
    echo "Generating frequency-feature channel groups: ${CHANNEL_GROUPS_FILE}"
    python scripts/make_channel_groups.py \
        --dset "${DSET}" \
        --max_channels "${MAX_CHANNELS_PER_MODEL}" \
        --output "${CHANNEL_GROUPS_FILE}"
fi

round_half_up() {
    local m="$1"
    local ratio="$2"
    python - <<PY
import math
m = float("${m}")
r = float("${ratio}")
print(int(math.floor(m * r + 0.5)))
PY
}

for M in ${M_LIST}; do
    for RATIO in ${RATIO_LIST}; do
        PRED_LEN="$(round_half_up "${M}" "${RATIO}")"
        if [ "${PRED_LEN}" -lt 1 ]; then
            PRED_LEN=1
        fi
        ACTUAL_RATIO="$(python - <<PY
m = int("${M}")
n = int("${PRED_LEN}")
print(f"{(n / m):.6f}")
PY
)"

        export FORECAST_STEP_SIZE_LIST="${M}"
        export FORECAST_PRED_LEN_LIST="${PRED_LEN}"

        COMBO_TAG="m${M}_r${RATIO//./p}_n${PRED_LEN}"
        COMBO_GROUP_RUN_NAME="${SWEEP_TAG}_${COMBO_TAG}"
        export FINETUNE_SAVE_PATH="${REPO_ROOT}/saved_models/patch_vqvae_finetune/${COMBO_GROUP_RUN_NAME}"
        export LOG_DIR="${MASTER_LOG_DIR}/${COMBO_TAG}"
        mkdir -p "${LOG_DIR}"

        echo
        echo "-------------------------------------------------"
        echo "Run combo: ${COMBO_TAG}"
        echo "  M (step)       : ${M}"
        echo "  requested ratio: ${RATIO}"
        echo "  pred_len       : ${PRED_LEN}"
        echo "  actual ratio   : ${ACTUAL_RATIO}"
        echo "  log dir        : ${LOG_DIR}"
        echo "-------------------------------------------------"

        bash src/training/channel_group_pipeline.sh
        RC=$?
        if [ "${RC}" -ne 0 ]; then
            echo "ERROR: combo failed (${COMBO_TAG}), rc=${RC}" | tee -a "${MASTER_LOG_DIR}/failed.log"
            continue
        fi

        OVERALL_TSV="${LOG_DIR}/summary_overall.tsv"
        if [ ! -f "${OVERALL_TSV}" ]; then
            echo "WARN: missing ${OVERALL_TSV} for ${COMBO_TAG}" | tee -a "${MASTER_LOG_DIR}/failed.log"
            continue
        fi

        ROW="$(OVERALL_TSV="${OVERALL_TSV}" python - <<'PY'
import csv, os
path = os.environ["OVERALL_TSV"]
with open(path, newline='') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
if not rows:
    print("NA\tNA")
else:
    row = rows[0]
    print(f"{row.get('weighted_mse', 'NA')}\t{row.get('weighted_mae', 'NA')}")
PY
)"
        MSE="$(echo "${ROW}" | awk -F '\t' '{print $1}')"
        MAE="$(echo "${ROW}" | awk -F '\t' '{print $2}')"
        echo -e "${M}\t${RATIO}\t${PRED_LEN}\t${ACTUAL_RATIO}\t${MSE}\t${MAE}\t${LOG_DIR}" >> "${SUMMARY_TSV}"
    done
done

SUMMARY_TSV="${SUMMARY_TSV}" SUMMARY_MD="${SUMMARY_MD}" python - <<'PY'
import csv
import os
from pathlib import Path

tsv_path = Path(os.environ["SUMMARY_TSV"])
md_path = Path(os.environ["SUMMARY_MD"])
rows = []
with tsv_path.open(newline='') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        rows.append(r)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("inf")

rows_sorted = sorted(rows, key=lambda r: safe_float(r.get("weighted_mse", "inf")))

lines = [
    "# ETTh2 pred=720 R2 Sensitivity Summary",
    "",
    "| Rank | M | Requested R2 | Pred Len | Actual R2 | Weighted MSE | Weighted MAE |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for i, r in enumerate(rows_sorted, start=1):
    lines.append(
        f"| {i} | {r['M']} | {r['requested_ratio']} | {r['pred_len']} | "
        f"{r['actual_ratio']} | {r['weighted_mse']} | {r['weighted_mae']} |"
    )

md_path.write_text("\n".join(lines) + "\n")
print(f"Saved markdown summary: {md_path}")
PY

echo
echo "Sweep finished."
echo "TSV summary : ${SUMMARY_TSV}"
echo "MD summary  : ${SUMMARY_MD}"
