#!/bin/bash
# =====================================================================
# Channel-group pipeline for high-dimensional datasets.
#
# Split variables by their original order into contiguous groups, then run:
#   codebook pretrain -> NTP pretrain -> finetune
# for each group independently.
#
# Usage:
#   cd <repo_root>
#   DSET=traffic TOTAL_CHANNELS=862 MAX_CHANNELS_PER_MODEL=64 \
#     bash src/training/channel_group_pipeline.sh
#
# If TOTAL_CHANNELS is omitted, the script tries to infer it from datautils.py.
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAINING_DIR="${REPO_ROOT}/src/training"
SAVED_MODELS_DIR="${REPO_ROOT}/saved_models"

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

DSET="${DSET:-traffic}"
TOTAL_CHANNELS="${TOTAL_CHANNELS:-}"
MAX_CHANNELS_PER_MODEL="${MAX_CHANNELS_PER_MODEL:-64}"
CHANNEL_GROUPS_FILE="${CHANNEL_GROUPS_FILE:-}"
BASE_MODEL_ID="${BASE_MODEL_ID:-800}"

FEATURES="${FEATURES:-M}"
SCALER="${SCALER:-standard}"
NUM_WORKERS="${NUM_WORKERS:-0}"

# VQVAE / codebook
PATCH_SIZE="${PATCH_SIZE:-8}"
EMBEDDING_DIM="${EMBEDDING_DIM:-32}"
COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-8}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
NUM_HIDDENS="${NUM_HIDDENS:-64}"
NUM_RESIDUAL_LAYERS="${NUM_RESIDUAL_LAYERS:-2}"
NUM_RESIDUAL_HIDDENS="${NUM_RESIDUAL_HIDDENS:-64}"
VQVAE_BACKBONE="${VQVAE_BACKBONE:-mlp}"
VQVAE_TCN_KERNEL_SIZE="${VQVAE_TCN_KERNEL_SIZE:-5}"
VQVAE_CHUNK_SIZE="${VQVAE_CHUNK_SIZE:-2}"
DECODER_LOWPASS="${DECODER_LOWPASS:-0}"
N_RQ_LAYERS="${N_RQ_LAYERS:-2}"
PER_CHANNEL_CODEBOOK="${PER_CHANNEL_CODEBOOK:-0}"
CODE_DIM=$((EMBEDDING_DIM * PATCH_SIZE / COMPRESSION_FACTOR))

CB_CONTEXT_POINTS="${CB_CONTEXT_POINTS:-512}"
CB_BATCH_SIZE="${CB_BATCH_SIZE:-64}"
CB_EPOCHS="${CB_EPOCHS:-50}"
CB_LR="${CB_LR:-3e-4}"
SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.2}"
SPARSE_AMPLITUDE="${SPARSE_AMPLITUDE:-0.5}"
LAMBDA_ORD="${LAMBDA_ORD:-0.01}"
ORTH_WEIGHT="${ORTH_WEIGHT:-0.01}"
ORTH_START_EPOCH="${ORTH_START_EPOCH:-0}"
ORTH_WARMUP_EPOCHS="${ORTH_WARMUP_EPOCHS:-5}"

# NTP pretrain
PRETRAIN_CONTEXT_POINTS="${PRETRAIN_CONTEXT_POINTS:-512}"
PROGRESSIVE_STEP_SIZE="${PROGRESSIVE_STEP_SIZE:-2}"
PRETRAIN_PRED_LEN="${PRETRAIN_PRED_LEN:-6}"
N_LAYERS="${N_LAYERS:-3}"
N_HEADS="${N_HEADS:-4}"
D_FF="${D_FF:-256}"
DROPOUT="${DROPOUT:-0.1}"
TEMPORAL_BACKBONE="${TEMPORAL_BACKBONE:-causal_transformer}"
CHANNEL_MIXER_HEADS="${CHANNEL_MIXER_HEADS:-}"
TIMEFILTER_TOPK="${TIMEFILTER_TOPK:-8}"
TIMEFILTER_TEMPERATURE="${TIMEFILTER_TEMPERATURE:-1.0}"
TIMEFILTER_ATTN_HEADS="${TIMEFILTER_ATTN_HEADS:-}"
CHANNEL_SUMMARY_WINDOW="${CHANNEL_SUMMARY_WINDOW:-4}"
CHANNEL_SUMMARY_GATE_INIT="${CHANNEL_SUMMARY_GATE_INIT:--4.0}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-64}"
PRETRAIN_LR="${PRETRAIN_LR:-3e-4}"
RQ_LAYER_WEIGHTS="${RQ_LAYER_WEIGHTS:-1.0 1.0}"
SOFT_NEIGHBOR_K="${SOFT_NEIGHBOR_K:-0}"
SOFT_NEIGHBOR_ALPHA="${SOFT_NEIGHBOR_ALPHA:-0.25}"
SOFT_NEIGHBOR_TAU="${SOFT_NEIGHBOR_TAU:-0.3}"
USE_RAW_INPUT="${USE_RAW_INPUT:-0}"

# Finetune
FINETUNE_CONTEXT_POINTS="${FINETUNE_CONTEXT_POINTS:-192}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-32}"
FINETUNE_LR="${FINETUNE_LR:-2e-4}"
FINETUNE_LR_LIST=(${FINETUNE_LR_LIST:-})
TARGET_POINTS_LIST=(${TARGET_POINTS_LIST:-96 192 336 720})
FORECAST_STEP_SIZE="${FORECAST_STEP_SIZE:-2}"
FORECAST_PRED_LEN="${FORECAST_PRED_LEN:-6}"
FORECAST_STEP_SIZE_LIST=(${FORECAST_STEP_SIZE_LIST:-})
FORECAST_PRED_LEN_LIST=(${FORECAST_PRED_LEN_LIST:-})
USE_GUMBEL_SOFTMAX="${USE_GUMBEL_SOFTMAX:-1}"
GUMBEL_TEMPERATURE="${GUMBEL_TEMPERATURE:-0.6}"
GUMBEL_TEMPERATURE_LIST=(${GUMBEL_TEMPERATURE_LIST:-})
GUMBEL_HARD="${GUMBEL_HARD:-0}"
TRAIN_LOSS="${TRAIN_LOSS:-mse}"
HUBER_DELTA="${HUBER_DELTA:-1.0}"
HUBER_DELTA_LIST=(${HUBER_DELTA_LIST:-})
UNFREEZE_DECODER="${UNFREEZE_DECODER:-0}"
DECODER_LR_RATIO="${DECODER_LR_RATIO:-0.05}"
DECODER_WD_RATIO="${DECODER_WD_RATIO:-5.0}"

REVIN="${REVIN:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
STREAM_LOGS="${STREAM_LOGS:-0}"  # 1: 训练过程打印到终端，仅临时捕获用于摘要；0: 完整写入 log 文件
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"  # deprecated alias: 1 等价于 FORCE_RETRAIN_ALL=1
FORCE_RETRAIN_ALL="${FORCE_RETRAIN_ALL:-${FORCE_RETRAIN}}"  # 1: codebook/pretrain/finetune 全部重跑
FORCE_RETRAIN_PRETRAIN="${FORCE_RETRAIN_PRETRAIN:-0}"       # 1: 保留 codebook，只重跑 pretrain + finetune
RETAIN_RUNS="${RETAIN_RUNS:-5}"                             # 只保留同前缀最近 N 次运行目录
RESUME_LATEST_RUN="${RESUME_LATEST_RUN:-0}"                 # 1: 非强制重跑时自动复用同前缀最新 run

# Put all channel-split checkpoints from the same run under one folder.
RUN_HISTORY_PREFIX="${RUN_HISTORY_PREFIX:-${DSET}_chgrp_m${MAX_CHANNELS_PER_MODEL}_base${BASE_MODEL_ID}}"
if [ -z "${GROUP_RUN_NAME:-}" ] && [ "${RESUME_LATEST_RUN}" = "1" ] && [ "${FORCE_RETRAIN_ALL}" != "1" ]; then
    for parent in \
        "${SAVED_MODELS_DIR}/vqvae_only" \
        "${SAVED_MODELS_DIR}/patch_vqvae" \
        "${SAVED_MODELS_DIR}/patch_vqvae_finetune" \
        "${REPO_ROOT}/logs"; do
        latest_dir=$(ls -dt "${parent}/${RUN_HISTORY_PREFIX}"* 2>/dev/null | head -1)
        if [ -n "${latest_dir}" ] && [ -d "${latest_dir}" ]; then
            GROUP_RUN_NAME="$(basename "${latest_dir}")"
            echo "Resume latest run: ${GROUP_RUN_NAME} (from ${parent})"
            break
        fi
    done
fi
GROUP_RUN_NAME="${GROUP_RUN_NAME:-${RUN_HISTORY_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
CB_SAVE_PATH="${CB_SAVE_PATH:-${SAVED_MODELS_DIR}/vqvae_only/${GROUP_RUN_NAME}}"
PRETRAIN_SAVE_PATH="${PRETRAIN_SAVE_PATH:-${SAVED_MODELS_DIR}/patch_vqvae/${GROUP_RUN_NAME}}"
FINETUNE_SAVE_PATH="${FINETUNE_SAVE_PATH:-${SAVED_MODELS_DIR}/patch_vqvae_finetune/${GROUP_RUN_NAME}}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/${GROUP_RUN_NAME}}"
mkdir -p "${LOG_DIR}"

# 早一点确定 SOFT_NEIGHBOR_SUFFIX，下面的 PROGRESS_LOG 等命名都会用到。
# 注意：alpha/tau 用 Python float repr 标准化，确保与 Python 侧
# (src/training/patch_vqvae_pretrain_common.py) 生成的 ckpt 文件名一致
# （否则 1.0 / 0.50 等写法会两侧不匹配，导致找不到 pretrain ckpt）。
SOFT_NEIGHBOR_SUFFIX=""
if [ "${SOFT_NEIGHBOR_K}" -gt 0 ]; then
    _SN_ALPHA=$("${PYTHON_BIN}" -c "print(float(${SOFT_NEIGHBOR_ALPHA}))")
    _SN_TAU=$("${PYTHON_BIN}" -c "print(float(${SOFT_NEIGHBOR_TAU}))")
    SOFT_NEIGHBOR_SUFFIX="_snk${SOFT_NEIGHBOR_K}a${_SN_ALPHA}t${_SN_TAU}"
    SOFT_NEIGHBOR_SUFFIX="${SOFT_NEIGHBOR_SUFFIX//./p}"
fi

PROGRESS_LOG="${LOG_DIR}/_progress${SOFT_NEIGHBOR_SUFFIX}.log"

prune_matching_dirs() {
    local parent="$1"
    local prefix="$2"
    local keep="$3"
    [ -d "${parent}" ] || return 0
    [ -n "${prefix}" ] || return 0
    [ "${keep}" -gt 0 ] 2>/dev/null || return 0

    local count=0
    local dir
    ls -dt "${parent}/${prefix}"* 2>/dev/null | while IFS= read -r dir; do
        [ -d "${dir}" ] || continue
        count=$((count + 1))
        if [ "${count}" -gt "${keep}" ]; then
            echo "Prune old run dir: ${dir}"
            rm -rf "${dir}"
        fi
    done
}

remove_model_artifacts() {
    local ckpt="$1"
    local base="${ckpt%.pth}"
    local f
    rm -f "${base}.pth" "${base}_history.csv" "${base}_config.json" "${base}_results.csv"
    for f in "${base}"_final_epoch*.pth; do
        [ -e "${f}" ] && rm -f "${f}"
    done
}

prune_matching_dirs "$(dirname "${CB_SAVE_PATH}")" "${RUN_HISTORY_PREFIX}" "${RETAIN_RUNS}"
prune_matching_dirs "$(dirname "${PRETRAIN_SAVE_PATH}")" "${RUN_HISTORY_PREFIX}" "${RETAIN_RUNS}"
prune_matching_dirs "$(dirname "${FINETUNE_SAVE_PATH}")" "${RUN_HISTORY_PREFIX}" "${RETAIN_RUNS}"
prune_matching_dirs "$(dirname "${LOG_DIR}")" "${RUN_HISTORY_PREFIX}" "${RETAIN_RUNS}"

echo "===== Channel group pipeline started at $(date) =====" | tee -a "${PROGRESS_LOG}"
echo "Rerun mode: FORCE_RETRAIN_ALL=${FORCE_RETRAIN_ALL} | FORCE_RETRAIN_PRETRAIN=${FORCE_RETRAIN_PRETRAIN} | RESUME_LATEST_RUN=${RESUME_LATEST_RUN}" | tee -a "${PROGRESS_LOG}"
echo "Finetune decoder: unfreeze=${UNFREEZE_DECODER} | lr_ratio=${DECODER_LR_RATIO} | wd_ratio=${DECODER_WD_RATIO}" | tee -a "${PROGRESS_LOG}"
echo "Retention: keep latest ${RETAIN_RUNS} runs matching prefix '${RUN_HISTORY_PREFIX}'" | tee -a "${PROGRESS_LOG}"

if [ -z "${TOTAL_CHANNELS}" ]; then
    TOTAL_CHANNELS=$(cd "${REPO_ROOT}" && "${PYTHON_BIN}" - <<PY
from types import SimpleNamespace
from datautils import get_dls
args = SimpleNamespace(
    dset="${DSET}", context_points=${CB_CONTEXT_POINTS}, target_points=96,
    batch_size=1, num_workers=0, scaler="${SCALER}", features="${FEATURES}",
    use_time_features=False,
)
dls = get_dls(args)
print(dls.vars)
PY
)
fi

echo "Dataset=${DSET} total_channels=${TOTAL_CHANNELS} max_per_model=${MAX_CHANNELS_PER_MODEL}" | tee -a "${PROGRESS_LOG}"
if [ -n "${CHANNEL_GROUPS_FILE}" ]; then
    echo "Channel groups file: ${CHANNEL_GROUPS_FILE}" | tee -a "${PROGRESS_LOG}"
fi
echo "Checkpoint group folder: ${GROUP_RUN_NAME}" | tee -a "${PROGRESS_LOG}"
echo "  Codebook save_path : ${CB_SAVE_PATH}" | tee -a "${PROGRESS_LOG}"
echo "  Pretrain save_path : ${PRETRAIN_SAVE_PATH}" | tee -a "${PROGRESS_LOG}"
echo "  Finetune save_path : ${FINETUNE_SAVE_PATH}" | tee -a "${PROGRESS_LOG}"

PERCH_SUFFIX=""
[ "${PER_CHANNEL_CODEBOOK}" -eq 1 ] && PERCH_SUFFIX="_perch"
RVQ_SUFFIX=""
[ "${N_RQ_LAYERS}" -gt 1 ] && RVQ_SUFFIX="_rvq${N_RQ_LAYERS}"
BACKBONE_SUFFIX=""
if [ "${VQVAE_BACKBONE}" = "tcn" ]; then
    BACKBONE_SUFFIX="_tcnk${VQVAE_TCN_KERNEL_SIZE}"
elif [ "${VQVAE_BACKBONE}" = "linear" ]; then
    BACKBONE_SUFFIX="_linear"
elif [ "${VQVAE_BACKBONE}" = "conv_linear" ]; then
    BACKBONE_SUFFIX="_convlineark${VQVAE_TCN_KERNEL_SIZE}"
elif [ "${VQVAE_BACKBONE}" != "mlp" ]; then
    BACKBONE_SUFFIX="_${VQVAE_BACKBONE}c${VQVAE_CHUNK_SIZE}"
fi
if [ "${DECODER_LOWPASS}" = "1" ]; then
    BACKBONE_SUFFIX="${BACKBONE_SUFFIX}_dlp"
fi
TEMPORAL_SUFFIX=""
if [ "${TEMPORAL_BACKBONE}" = "timefilter_lite" ]; then
    TEMPORAL_SUFFIX="_timefilterlitek${TIMEFILTER_TOPK}"
elif [ "${TEMPORAL_BACKBONE}" = "encoder_transformer" ] || [ "${TEMPORAL_BACKBONE}" = "transformer_encoder" ] || [ "${TEMPORAL_BACKBONE}" = "noncausal_transformer" ]; then
    TEMPORAL_SUFFIX="_encoder"
elif [ "${TEMPORAL_BACKBONE}" = "encoder_timefilter_lite" ] || [ "${TEMPORAL_BACKBONE}" = "encoder_timefilter" ]; then
    TEMPORAL_SUFFIX="_encodertfk${TIMEFILTER_TOPK}"
elif [ "${TEMPORAL_BACKBONE}" = "encoder_cluster_timefilter_lite" ] || [ "${TEMPORAL_BACKBONE}" = "encoder_cluster_timefilter" ]; then
    TEMPORAL_SUFFIX="_encoderclustertfk${TIMEFILTER_TOPK}"
elif [ "${TEMPORAL_BACKBONE}" = "cluster_timefilter_lite" ]; then
    TEMPORAL_SUFFIX="_clustertfk${TIMEFILTER_TOPK}"
elif [ "${TEMPORAL_BACKBONE}" = "timefilter_attn" ]; then
    _TF_ATTN_HEADS="${TIMEFILTER_ATTN_HEADS:-${N_HEADS}}"
    TEMPORAL_SUFFIX="_timefilterattnh${_TF_ATTN_HEADS}k${TIMEFILTER_TOPK}"
elif [ "${TEMPORAL_BACKBONE}" = "channel_summary_adapter" ]; then
    TEMPORAL_SUFFIX="_chsummaryw${CHANNEL_SUMMARY_WINDOW}"
elif [ "${TEMPORAL_BACKBONE}" != "causal_transformer" ] && [ "${TEMPORAL_BACKBONE}" != "transformer" ] && [ "${TEMPORAL_BACKBONE}" != "patchtst" ]; then
    TEMPORAL_SUFFIX="_${TEMPORAL_BACKBONE}"
fi
NMPP_SUFFIX=""
[ "${USE_RAW_INPUT}" -eq 1 ] && NMPP_SUFFIX="_nmpp"
# SOFT_NEIGHBOR_SUFFIX 已在前面早一点定义（PROGRESS_LOG 之前）

run_and_capture() {
    local log_file="$1"
    shift
    if [ "${STREAM_LOGS}" = "1" ]; then
        "$@" 2>&1 | tee "${log_file}.tmp"
        RC=${PIPESTATUS[0]}
        PARSE_LOG="${log_file}.tmp"
    else
        "$@" > "${log_file}" 2>&1
        RC=$?
        PARSE_LOG="${log_file}"
    fi
}

cleanup_parse_log() {
    [ "${STREAM_LOGS}" = "1" ] && rm -f "${PARSE_LOG}"
}

resolve_pretrain_ckpt_after_run() {
    # Single source of truth fallback: Python prints the actual saved path as "模型: ...".
    # If shell-side naming ever lags behind Python naming, adopt the real path instead
    # of reporting a false PRE FAILED with rc=0.
    if [ -f "${PRETRAIN_CKPT}" ]; then
        return 0
    fi

    local actual=""
    if [ -n "${PARSE_LOG:-}" ] && [ -f "${PARSE_LOG}" ]; then
        actual=$(grep -E "模型:[[:space:]]*.*\.pth" "${PARSE_LOG}" | tail -1 | sed -E 's/^.*模型:[[:space:]]*//')
        if [ -n "${actual}" ] && [ -f "${actual}" ]; then
            echo "[$(date +%H:%M:%S)] PRE checkpoint path resolved from log: ${actual}" | tee -a "${PROGRESS_LOG}"
            PRETRAIN_CKPT="${actual}"
            PRETRAIN_NAME="$(basename "${actual}" .pth)"
            return 0
        fi
    fi

    local found=""
    found=$(ls -t "${PRETRAIN_SAVE_PATH}/${DSET}/patch_vqvae_ps${PATCH_SIZE}_cb${CODEBOOK_SIZE}_cd${CODE_DIM}_l${N_LAYERS}_in${PRETRAIN_CONTEXT_POINTS}_step${PROGRESSIVE_STEP_SIZE}"*"_model${MODEL_ID}"*"${CH_SUFFIX}.pth" 2>/dev/null | head -1)
    if [ -n "${found}" ] && [ -f "${found}" ]; then
        echo "[$(date +%H:%M:%S)] PRE checkpoint path resolved by glob: ${found}" | tee -a "${PROGRESS_LOG}"
        PRETRAIN_CKPT="${found}"
        PRETRAIN_NAME="$(basename "${found}" .pth)"
        return 0
    fi

    return 1
}

set_group_vars() {
    GROUP_ID="$1"
    if [ -n "${CHANNEL_GROUPS_FILE}" ]; then
        CHANNEL_INDICES=$(CHANNEL_GROUPS_FILE="${CHANNEL_GROUPS_FILE}" GROUP_ID="${GROUP_ID}" "${PYTHON_BIN}" - <<'PY'
import json, os
with open(os.environ["CHANNEL_GROUPS_FILE"]) as f:
    groups = json.load(f)["groups"]
print(",".join(str(x) for x in groups[int(os.environ["GROUP_ID"])]))
PY
)
        START=$(echo "${CHANNEL_INDICES}" | awk -F, '{print $1}')
        END=$(CHANNEL_INDICES="${CHANNEL_INDICES}" "${PYTHON_BIN}" - <<'PY'
import os
idx = [int(x) for x in os.environ["CHANNEL_INDICES"].split(",") if x]
print(max(idx) + 1)
PY
)
        CH_SUFFIX="_grp${GROUP_ID}"
        GROUP_TAG="g${GROUP_ID}_grp"
        CHANNEL_ARGS="--channel_indices '${CHANNEL_INDICES}' --channel_group_id '${GROUP_ID}'"
        GROUP_WEIGHT=$(echo "${CHANNEL_INDICES}" | awk -F, '{print NF}')
    else
        START=$((GROUP_ID * MAX_CHANNELS_PER_MODEL))
        END=$((START + MAX_CHANNELS_PER_MODEL))
        [ "${END}" -gt "${TOTAL_CHANNELS}" ] && END="${TOTAL_CHANNELS}"
        CH_SUFFIX="_ch${START}-${END}"
        GROUP_TAG="g${GROUP_ID}_ch${START}-${END}"
        CHANNEL_ARGS="--channel_start '${START}' --channel_end '${END}'"
        GROUP_WEIGHT=$((END-START))
    fi
    MODEL_ID=$((BASE_MODEL_ID + GROUP_ID))
    CB_CKPT="${CB_SAVE_PATH}/${DSET}/codebook_ps${PATCH_SIZE}_cb${CODEBOOK_SIZE}_cd${CODE_DIM}${PERCH_SUFFIX}${RVQ_SUFFIX}${BACKBONE_SUFFIX}_model${MODEL_ID}${CH_SUFFIX}.pth"
    PRETRAIN_NAME="patch_vqvae_ps${PATCH_SIZE}_cb${CODEBOOK_SIZE}_cd${CODE_DIM}_l${N_LAYERS}_in${PRETRAIN_CONTEXT_POINTS}_step${PROGRESSIVE_STEP_SIZE}_model${MODEL_ID}${PERCH_SUFFIX}${RVQ_SUFFIX}${BACKBONE_SUFFIX}${TEMPORAL_SUFFIX}${NMPP_SUFFIX}${SOFT_NEIGHBOR_SUFFIX}${CH_SUFFIX}"
    PRETRAIN_CKPT="${PRETRAIN_SAVE_PATH}/${DSET}/${PRETRAIN_NAME}.pth"
}

if [ -n "${CHANNEL_GROUPS_FILE}" ]; then
    NUM_GROUPS=$(CHANNEL_GROUPS_FILE="${CHANNEL_GROUPS_FILE}" "${PYTHON_BIN}" - <<'PY'
import json, os
with open(os.environ["CHANNEL_GROUPS_FILE"]) as f:
    print(len(json.load(f)["groups"]))
PY
)
else
    NUM_GROUPS=$(((TOTAL_CHANNELS + MAX_CHANNELS_PER_MODEL - 1) / MAX_CHANNELS_PER_MODEL))
fi
if [ -z "${NUM_GROUPS}" ] || [ "${NUM_GROUPS}" -le 0 ]; then
    echo "ERROR: failed to determine NUM_GROUPS (DSET=${DSET}, TOTAL_CHANNELS=${TOTAL_CHANNELS}, CHANNEL_GROUPS_FILE=${CHANNEL_GROUPS_FILE})" | tee -a "${PROGRESS_LOG}"
    exit 1
fi
echo "Execution order: all codebooks -> all pretrains -> all finetunes" | tee -a "${PROGRESS_LOG}"
echo "Groups: ${NUM_GROUPS}" | tee -a "${PROGRESS_LOG}"

select_horizon_param() {
    local label="$1"
    local default_value="$2"
    local idx="$3"
    shift 3
    local values=("$@")

    if [ "${#values[@]}" -eq 0 ]; then
        echo "${default_value}"
    elif [ "${#values[@]}" -eq 1 ]; then
        echo "${values[0]}"
    elif [ "${#values[@]}" -eq "${#TARGET_POINTS_LIST[@]}" ]; then
        echo "${values[$idx]}"
    else
        echo "ERROR: ${label} list length ${#values[@]} must be 1 or match TARGET_POINTS_LIST length ${#TARGET_POINTS_LIST[@]}" >&2
        return 1
    fi
}

echo "===== Phase 1/3: Train all codebooks =====" | tee -a "${PROGRESS_LOG}"
for ((GROUP_ID=0; GROUP_ID<NUM_GROUPS; GROUP_ID++)); do
    set_group_vars "${GROUP_ID}"
    CB_LOG="${LOG_DIR}/cb_${GROUP_TAG}.log"
    if [ "${FORCE_RETRAIN_ALL}" != "1" ] && [ -f "${CB_CKPT}" ]; then
        echo "[$(date +%H:%M:%S)] CB SKIP ${GROUP_TAG} (checkpoint exists; FORCE_RETRAIN_ALL=1 to rerun)" | tee -a "${PROGRESS_LOG}"
        continue
    fi
    if [ "${FORCE_RETRAIN_ALL}" = "1" ]; then
        remove_model_artifacts "${CB_CKPT}"
    fi
    echo "[$(date +%H:%M:%S)] CB START ${GROUP_TAG}" | tee -a "${PROGRESS_LOG}"
    run_and_capture "${CB_LOG}" bash -lc "cd '${REPO_ROOT}' && '${PYTHON_BIN}' -u '${TRAINING_DIR}/codebook_pretrain.py' \
        --dset '${DSET}' --context_points '${CB_CONTEXT_POINTS}' \
        --batch_size '${CB_BATCH_SIZE}' --num_workers '${NUM_WORKERS}' \
        --scaler '${SCALER}' --features '${FEATURES}' \
        --patch_size '${PATCH_SIZE}' --embedding_dim '${EMBEDDING_DIM}' \
        --compression_factor '${COMPRESSION_FACTOR}' --codebook_size '${CODEBOOK_SIZE}' \
        --num_hiddens '${NUM_HIDDENS}' --num_residual_layers '${NUM_RESIDUAL_LAYERS}' \
        --num_residual_hiddens '${NUM_RESIDUAL_HIDDENS}' \
        --vqvae_backbone '${VQVAE_BACKBONE}' --vqvae_tcn_kernel_size '${VQVAE_TCN_KERNEL_SIZE}' \
        --vqvae_chunk_size '${VQVAE_CHUNK_SIZE}' --decoder_lowpass '${DECODER_LOWPASS}' \
        --codebook_ema 1 --ema_decay 0.95 \
        --n_epochs '${CB_EPOCHS}' --lr '${CB_LR}' --weight_decay '${WEIGHT_DECAY}' \
        --revin '${REVIN}' --per_channel_codebook '${PER_CHANNEL_CODEBOOK}' \
        --n_rq_layers '${N_RQ_LAYERS}' \
        --sparse_weight '${SPARSE_WEIGHT}' --sparse_amplitude '${SPARSE_AMPLITUDE}' \
        --lambda_ord '${LAMBDA_ORD}' \
        --orth_weight '${ORTH_WEIGHT}' --orth_start_epoch '${ORTH_START_EPOCH}' \
        --orth_warmup_epochs '${ORTH_WARMUP_EPOCHS}' \
        ${CHANNEL_ARGS} \
        --save_path '${CB_SAVE_PATH}' \
        --model_id '${MODEL_ID}'"
    if [ ${RC} -ne 0 ] || [ ! -f "${CB_CKPT}" ]; then
        echo "[$(date +%H:%M:%S)] CB FAILED ${GROUP_TAG} rc=${RC}" | tee -a "${PROGRESS_LOG}"
        tail -20 "${PARSE_LOG}" | sed 's/^/    /' | tee -a "${PROGRESS_LOG}"
        cleanup_parse_log
        exit 1
    fi
    CB_TAIL=$(grep -iE "码本预训练完成|最佳验证损失|best model saved|best val|val_loss|valid loss" "${PARSE_LOG}" | tail -4 | tr '\n' '|')
    echo "[$(date +%H:%M:%S)] CB DONE ${GROUP_TAG} ${CB_TAIL}" | tee -a "${PROGRESS_LOG}"
    cleanup_parse_log
done

echo "===== Phase 2/3: Pretrain all transformers =====" | tee -a "${PROGRESS_LOG}"
for ((GROUP_ID=0; GROUP_ID<NUM_GROUPS; GROUP_ID++)); do
    set_group_vars "${GROUP_ID}"
    PRE_LOG="${LOG_DIR}/pre_${GROUP_TAG}${SOFT_NEIGHBOR_SUFFIX}.log"
    if [ ! -f "${CB_CKPT}" ]; then
        echo "[$(date +%H:%M:%S)] PRE SKIP ${GROUP_TAG} (missing codebook)" | tee -a "${PROGRESS_LOG}"
        continue
    fi
    if [ "${FORCE_RETRAIN_ALL}" != "1" ] && [ "${FORCE_RETRAIN_PRETRAIN}" != "1" ] && [ -f "${PRETRAIN_CKPT}" ]; then
        echo "[$(date +%H:%M:%S)] PRE SKIP ${GROUP_TAG} (checkpoint exists; FORCE_RETRAIN_PRETRAIN=1 or FORCE_RETRAIN_ALL=1 to rerun)" | tee -a "${PROGRESS_LOG}"
        continue
    fi
    if [ "${FORCE_RETRAIN_ALL}" = "1" ] || [ "${FORCE_RETRAIN_PRETRAIN}" = "1" ]; then
        remove_model_artifacts "${PRETRAIN_CKPT}"
    fi
    echo "[$(date +%H:%M:%S)] PRE START ${GROUP_TAG}" | tee -a "${PROGRESS_LOG}"
    run_and_capture "${PRE_LOG}" bash -lc "cd '${REPO_ROOT}' && '${PYTHON_BIN}' -u '${TRAINING_DIR}/patch_vqvae_pretrain.py' \
        --dset '${DSET}' --context_points '${PRETRAIN_CONTEXT_POINTS}' \
        --progressive_step_size '${PROGRESSIVE_STEP_SIZE}' --pred_len '${PRETRAIN_PRED_LEN}' \
        --batch_size '${PRETRAIN_BATCH_SIZE}' --num_workers '${NUM_WORKERS}' \
        --patch_size '${PATCH_SIZE}' --embedding_dim '${EMBEDDING_DIM}' \
        --compression_factor '${COMPRESSION_FACTOR}' --codebook_size '${CODEBOOK_SIZE}' \
        --n_layers '${N_LAYERS}' --n_heads '${N_HEADS}' --d_ff '${D_FF}' --dropout '${DROPOUT}' \
        --temporal_backbone '${TEMPORAL_BACKBONE}' \
        ${CHANNEL_MIXER_HEADS:+--channel_mixer_heads '${CHANNEL_MIXER_HEADS}'} \
        --timefilter_topk '${TIMEFILTER_TOPK}' --timefilter_temperature '${TIMEFILTER_TEMPERATURE}' \
        ${TIMEFILTER_ATTN_HEADS:+--timefilter_attn_heads '${TIMEFILTER_ATTN_HEADS}'} \
        --channel_summary_window '${CHANNEL_SUMMARY_WINDOW}' \
        --channel_summary_gate_init '${CHANNEL_SUMMARY_GATE_INIT}' \
        --num_hiddens '${NUM_HIDDENS}' --num_residual_layers '${NUM_RESIDUAL_LAYERS}' \
        --num_residual_hiddens '${NUM_RESIDUAL_HIDDENS}' \
        --vqvae_backbone '${VQVAE_BACKBONE}' --vqvae_tcn_kernel_size '${VQVAE_TCN_KERNEL_SIZE}' \
        --vqvae_chunk_size '${VQVAE_CHUNK_SIZE}' --decoder_lowpass '${DECODER_LOWPASS}' \
        --vqvae_checkpoint '${CB_CKPT}' --freeze_vqvae 1 --load_vq_weights 1 \
        --per_channel_codebook '${PER_CHANNEL_CODEBOOK}' --n_rq_layers '${N_RQ_LAYERS}' \
        --use_raw_input '${USE_RAW_INPUT}' \
        --rq_layer_weights ${RQ_LAYER_WEIGHTS} \
        --soft_neighbor_k '${SOFT_NEIGHBOR_K}' --soft_neighbor_alpha '${SOFT_NEIGHBOR_ALPHA}' \
        --soft_neighbor_tau '${SOFT_NEIGHBOR_TAU}' \
        --n_epochs '${PRETRAIN_EPOCHS}' --lr '${PRETRAIN_LR}' --weight_decay '${WEIGHT_DECAY}' \
        --revin '${REVIN}' --vq_weight 0.0 --recon_weight 0.0 \
        ${CHANNEL_ARGS} \
        --save_path '${PRETRAIN_SAVE_PATH}' \
        --model_id '${MODEL_ID}'"
    if [ ${RC} -eq 0 ] && [ ! -f "${PRETRAIN_CKPT}" ]; then
        resolve_pretrain_ckpt_after_run
    fi
    if [ ${RC} -ne 0 ] || [ ! -f "${PRETRAIN_CKPT}" ]; then
        echo "[$(date +%H:%M:%S)] PRE FAILED ${GROUP_TAG} rc=${RC}" | tee -a "${PROGRESS_LOG}"
        echo "    Expected checkpoint: ${PRETRAIN_CKPT}" | tee -a "${PROGRESS_LOG}"
        tail -20 "${PARSE_LOG}" | sed 's/^/    /' | tee -a "${PROGRESS_LOG}"
        cleanup_parse_log
        exit 1
    fi
    PRE_TAIL=$(grep -iE "预训练完成|最佳验证损失|best model saved|val_loss|valid loss|模型:" "${PARSE_LOG}" | tail -4 | tr '\n' '|')
    echo "[$(date +%H:%M:%S)] PRE DONE ${GROUP_TAG} ${PRE_TAIL}" | tee -a "${PROGRESS_LOG}"
    cleanup_parse_log
done

echo "===== Phase 3/3: Finetune all groups by horizon =====" | tee -a "${PROGRESS_LOG}"
SUMMARY_TSV="${LOG_DIR}/summary${SOFT_NEIGHBOR_SUFFIX}.tsv"
echo -e "target_points\tgroup_id\tgroup_tag\tstart\tend\tweight\tmse\tmae" > "${SUMMARY_TSV}"
for TP_IDX in "${!TARGET_POINTS_LIST[@]}"; do
    TARGET_POINTS="${TARGET_POINTS_LIST[$TP_IDX]}"
    CURRENT_FORECAST_STEP_SIZE=$(select_horizon_param "FORECAST_STEP_SIZE_LIST" "${FORECAST_STEP_SIZE}" "${TP_IDX}" "${FORECAST_STEP_SIZE_LIST[@]}") || exit 1
    CURRENT_FORECAST_PRED_LEN=$(select_horizon_param "FORECAST_PRED_LEN_LIST" "${FORECAST_PRED_LEN}" "${TP_IDX}" "${FORECAST_PRED_LEN_LIST[@]}") || exit 1
    CURRENT_FINETUNE_LR=$(select_horizon_param "FINETUNE_LR_LIST" "${FINETUNE_LR}" "${TP_IDX}" "${FINETUNE_LR_LIST[@]}") || exit 1
    CURRENT_GUMBEL_TEMPERATURE=$(select_horizon_param "GUMBEL_TEMPERATURE_LIST" "${GUMBEL_TEMPERATURE}" "${TP_IDX}" "${GUMBEL_TEMPERATURE_LIST[@]}") || exit 1
    CURRENT_HUBER_DELTA=$(select_horizon_param "HUBER_DELTA_LIST" "${HUBER_DELTA}" "${TP_IDX}" "${HUBER_DELTA_LIST[@]}") || exit 1
    echo "===== Finetune horizon ${TARGET_POINTS} (ar_step=${CURRENT_FORECAST_STEP_SIZE}, pred_len=${CURRENT_FORECAST_PRED_LEN}, lr=${CURRENT_FINETUNE_LR}, tau=${CURRENT_GUMBEL_TEMPERATURE}, huber_delta=${CURRENT_HUBER_DELTA}) =====" | tee -a "${PROGRESS_LOG}"
    for ((GROUP_ID=0; GROUP_ID<NUM_GROUPS; GROUP_ID++)); do
        set_group_vars "${GROUP_ID}"
        FT_LOG="${LOG_DIR}/ft_${GROUP_TAG}_tp${TARGET_POINTS}${SOFT_NEIGHBOR_SUFFIX}.log"
        FT_NAME="patch_vqvae_finetune_cw${FINETUNE_CONTEXT_POINTS}_tw${TARGET_POINTS}_model${MODEL_ID}${TEMPORAL_SUFFIX}${SOFT_NEIGHBOR_SUFFIX}${CH_SUFFIX}"
        FT_CKPT="${FINETUNE_SAVE_PATH}/${DSET}/${FT_NAME}.pth"
        if [ ! -f "${PRETRAIN_CKPT}" ]; then
            echo "[$(date +%H:%M:%S)] FT SKIP ${GROUP_TAG} tp=${TARGET_POINTS} (missing pretrain)" | tee -a "${PROGRESS_LOG}"
            continue
        fi
        if [ "${FORCE_RETRAIN_ALL}" = "1" ] || [ "${FORCE_RETRAIN_PRETRAIN}" = "1" ]; then
            remove_model_artifacts "${FT_CKPT}"
        fi
        echo "[$(date +%H:%M:%S)] FT START ${GROUP_TAG} tp=${TARGET_POINTS}" | tee -a "${PROGRESS_LOG}"
        run_and_capture "${FT_LOG}" bash -lc "cd '${REPO_ROOT}' && '${PYTHON_BIN}' -u '${TRAINING_DIR}/patch_vqvae_finetune.py' \
            --dset '${DSET}' --context_points '${FINETUNE_CONTEXT_POINTS}' \
            --target_points '${TARGET_POINTS}' --batch_size '${FINETUNE_BATCH_SIZE}' \
            --num_workers '${NUM_WORKERS}' --pretrained_model '${PRETRAIN_CKPT}' \
            --n_epochs '${FINETUNE_EPOCHS}' --lr '${CURRENT_FINETUNE_LR}' \
            --weight_decay '${WEIGHT_DECAY}' --revin '${REVIN}' \
            --use_gumbel_softmax '${USE_GUMBEL_SOFTMAX}' \
            --gumbel_temperature '${CURRENT_GUMBEL_TEMPERATURE}' --gumbel_hard '${GUMBEL_HARD}' \
            --train_loss '${TRAIN_LOSS}' --huber_delta '${CURRENT_HUBER_DELTA}' \
            --unfreeze_decoder '${UNFREEZE_DECODER}' \
            --decoder_lr_ratio '${DECODER_LR_RATIO}' \
            --decoder_wd_ratio '${DECODER_WD_RATIO}' \
            --ar_step_size '${CURRENT_FORECAST_STEP_SIZE}' --pred_len '${CURRENT_FORECAST_PRED_LEN}' \
            ${CHANNEL_ARGS} \
            --save_path '${FINETUNE_SAVE_PATH}' \
            --model_id '${MODEL_ID}'"
        TAIL=$(grep -iE "测试 MSE|Test MSE|mse" "${PARSE_LOG}" | tail -2 | tr '\n' '|')
        echo "[$(date +%H:%M:%S)] FT DONE rc=${RC} ${GROUP_TAG} tp=${TARGET_POINTS} weight=${GROUP_WEIGHT} ${TAIL}" | tee -a "${PROGRESS_LOG}"
        [ ${RC} -ne 0 ] && tail -20 "${PARSE_LOG}" | sed 's/^/    /' | tee -a "${PROGRESS_LOG}"
        if [ ${RC} -eq 0 ]; then
            METRICS_LINE=$(grep -iE "测试 MSE|Test MSE" "${PARSE_LOG}" | tail -1)
            METRICS=$(METRICS_LINE="${METRICS_LINE}" "${PYTHON_BIN}" - <<'PY'
import os, re
line = os.environ.get("METRICS_LINE", "")
nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", line)
if len(nums) >= 2:
    print(nums[-2], nums[-1])
PY
)
            if [ -n "${METRICS}" ]; then
                MSE=$(echo "${METRICS}" | awk '{print $1}')
                MAE=$(echo "${METRICS}" | awk '{print $2}')
                echo -e "${TARGET_POINTS}\t${GROUP_ID}\t${GROUP_TAG}\t${START}\t${END}\t${GROUP_WEIGHT}\t${MSE}\t${MAE}" >> "${SUMMARY_TSV}"
            fi
        fi
        cleanup_parse_log
    done
    SUMMARY_LINE=$(TARGET_POINTS="${TARGET_POINTS}" SUMMARY_TSV="${SUMMARY_TSV}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

target = os.environ["TARGET_POINTS"]
path = Path(os.environ["SUMMARY_TSV"])
rows = []
for i, line in enumerate(path.read_text().splitlines()):
    if i == 0 or not line.strip():
        continue
    tp, gid, tag, start, end, weight, mse, mae = line.split("\t")
    if tp == target:
        rows.append((int(weight), float(mse), float(mae)))

if not rows:
    print(f"tp={target} summary unavailable (no successful groups)")
else:
    total_w = sum(w for w, _, _ in rows)
    wmse = sum(w * mse for w, mse, _ in rows) / total_w
    wmae = sum(w * mae for w, _, mae in rows) / total_w
    print(f"tp={target} groups={len(rows)} channels={total_w} weighted_mse={wmse:.6f} weighted_mae={wmae:.6f}")
PY
)
    echo "===== Weighted Summary: ${SUMMARY_LINE} =====" | tee -a "${PROGRESS_LOG}"
done

OVERALL_SUMMARY_TSV="${LOG_DIR}/summary_overall${SOFT_NEIGHBOR_SUFFIX}.tsv"
OVERALL_SUMMARY=$(SUMMARY_TSV="${SUMMARY_TSV}" OVERALL_SUMMARY_TSV="${OVERALL_SUMMARY_TSV}" "${PYTHON_BIN}" - <<'PY'
import os
from collections import defaultdict
from pathlib import Path

summary_path = Path(os.environ["SUMMARY_TSV"])
overall_path = Path(os.environ["OVERALL_SUMMARY_TSV"])
rows_by_tp = defaultdict(list)

if summary_path.exists():
    for i, line in enumerate(summary_path.read_text().splitlines()):
        if i == 0 or not line.strip():
            continue
        tp, gid, tag, start, end, weight, mse, mae = line.split("\t")
        rows_by_tp[tp].append((int(weight), float(mse), float(mae)))

out_lines = ["target_points\tgroups\tchannels\tweighted_mse\tweighted_mae"]
tp_summaries = []
for tp in sorted(rows_by_tp, key=lambda x: int(x)):
    rows = rows_by_tp[tp]
    total_w = sum(w for w, _, _ in rows)
    if total_w <= 0:
        continue
    wmse = sum(w * mse for w, mse, _ in rows) / total_w
    wmae = sum(w * mae for w, _, mae in rows) / total_w
    tp_summaries.append((tp, len(rows), total_w, wmse, wmae))
    out_lines.append(f"{tp}\t{len(rows)}\t{total_w}\t{wmse:.6f}\t{wmae:.6f}")

if tp_summaries:
    mean_mse = sum(x[3] for x in tp_summaries) / len(tp_summaries)
    mean_mae = sum(x[4] for x in tp_summaries) / len(tp_summaries)
    out_lines.append(f"mean\t{len(tp_summaries)}\t-\t{mean_mse:.6f}\t{mean_mae:.6f}")
    overall_path.write_text("\n".join(out_lines) + "\n")
    print(
        f"horizons={len(tp_summaries)} mean_weighted_mse={mean_mse:.6f} "
        f"mean_weighted_mae={mean_mae:.6f}"
    )
else:
    overall_path.write_text("\n".join(out_lines) + "\n")
    print("overall summary unavailable (no successful horizons)")
PY
)
echo "===== Overall Horizon Mean: ${OVERALL_SUMMARY} =====" | tee -a "${PROGRESS_LOG}"

echo "===== Channel group pipeline finished at $(date) =====" | tee -a "${PROGRESS_LOG}"
echo "Logs: ${LOG_DIR}"
echo "Summary TSV: ${SUMMARY_TSV}"
echo "Overall Summary TSV: ${OVERALL_SUMMARY_TSV}"

if [[ "${OVERALL_SUMMARY}" == overall\ summary\ unavailable* ]]; then
    exit 1
fi
