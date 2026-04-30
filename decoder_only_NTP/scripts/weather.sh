#!/bin/bash

# =====================================================
# NTP 预训练 + 微调 一键流程脚本
#
# 使用已训练好的码本模型（Encoder + VQ + Decoder），
# 全程冻结 VQVAE 参数，先渐进式预训练 Transformer，
# 再针对多个预测长度进行微调。
#
# 使用方法:
#   bash patch_vqvae_pretrain_then_finetune.sh <dataset> <context_points> <progressive_step_size>
# 例如:
#   bash patch_vqvae_pretrain_then_finetune.sh ettm1 1152 6
# =====================================================

# =====================================================
# 输入参数检验
# =====================================================

DSET=weather
PRETRAIN_CONTEXT_POINTS=1024
PROGRESSIVE_STEP_SIZE=3
MODEL_ID=1

# =====================================================
# Patch / VQVAE 参数（自动从 checkpoint 读取，这里作为备用）
# =====================================================
PATCH_SIZE=16
COMPRESSION_FACTOR=8
EMBEDDING_DIM=32
CODEBOOK_SIZE=256
NUM_HIDDENS=64
NUM_RESIDUAL_LAYERS=2
NUM_RESIDUAL_HIDDENS=64

# =====================================================
# Per-channel 码本（需与 vqvae-only 训练时保持一致）
# 0 = 所有通道共享同一码本
# 1 = 每通道独立码本（需由 codebook_pretrain.py --per_channel_codebook 1 训练）
# =====================================================
PER_CHANNEL_CODEBOOK=0

# =====================================================
# RVQ 层数
# 1 = 普通 VQ（与原有行为兼容）
# 2 = 2层残差 VQ（需与 vqvae-only 训练时保持一致）
# =====================================================
N_RQ_LAYERS=2
RQ_LAYER_WEIGHTS="1.0 1.0"

# =====================================================
# NMPP 模式（Next Masked Patch Prediction with Raw Input）
# 0 = 标准 NTP：Transformer 输入为 VQ 编码 embedding
# 1 = NMPP：Transformer 输入为原始 patch 的线性投影，
#            VQVAE 仅作为 teacher 提供目标 token ID
#            （需要同时设置 FREEZE_VQVAE=1 且指定有效的 CODEBOOK_CHECKPOINT）
# =====================================================
USE_RAW_INPUT=0

# =====================================================
# 码本模型路径
# 留空时脚本会自动在 ../vqvae-only/saved_models/vqvae_only 下查找；
# per-channel 模式自动优先匹配 *_perch*.pth
# =====================================================
CODE_DIM_CB=$((EMBEDDING_DIM * PATCH_SIZE / COMPRESSION_FACTOR))
# 构造码本路径时须与 vqvae-only/codebook_pretrain.py 的命名保持一致：
#   codebook_ps{P}_cb{C}_cd{D}{_perch}{_rvqN}_model{ID}.pth
CB_RVQ_SUFFIX=""
[ "${N_RQ_LAYERS:-1}" -gt 1 ] && CB_RVQ_SUFFIX="_rvq${N_RQ_LAYERS}"
if [ "${PER_CHANNEL_CODEBOOK}" -eq 1 ]; then
    CODEBOOK_CHECKPOINT="../vqvae-only/saved_models/vqvae_only/${DSET}/codebook_ps${PATCH_SIZE}_cb${CODEBOOK_SIZE}_cd${CODE_DIM_CB}_perch${CB_RVQ_SUFFIX}_model${MODEL_ID}.pth"
else
    CODEBOOK_CHECKPOINT="../vqvae-only/saved_models/vqvae_only/${DSET}/codebook_ps${PATCH_SIZE}_cb${CODEBOOK_SIZE}_cd${CODE_DIM_CB}${CB_RVQ_SUFFIX}_model${MODEL_ID}.pth"
fi
# 也可手动覆盖，例如：
# CODEBOOK_CHECKPOINT="/absolute/path/to/your_codebook.pth"

# =====================================================
# Transformer 参数
# =====================================================
N_LAYERS=3
N_HEADS=8
D_FF=256
DROPOUT=0.2
CODEBOOK_EMA=1
EMA_DECAY=0.99
EMA_EPS=1e-5
TRANSFORMER_HIDDEN_DIM=""   # 留空表示使用 code_dim（不额外升维）

# =====================================================
# 预训练参数
# =====================================================
PRETRAIN_EPOCHS=100
PRETRAIN_BATCH_SIZE=64
PRETRAIN_LR=3e-4
VQ_WEIGHT=0.0       # 码本已冻结，设为 0
RECON_WEIGHT=0.0    # 码本已冻结，设为 0
DISABLE_EMA_UPDATE=1

# =====================================================
# 微调参数
# =====================================================
FINETUNE_CONTEXT_POINTS=512
FINETUNE_EPOCHS=50
FINETUNE_BATCH_SIZE=64
FINETUNE_LR=1e-4
TARGET_POINTS_LIST=(96 192 336 720)
# TARGET_POINTS_LIST=(96 192 336 720)

# Gumbel-Softmax（微调阶段的码本查找）
USE_GUMBEL_SOFTMAX=1
GUMBEL_TEMPERATURE=0.6
GUMBEL_HARD=0
EVAL_TOP_K=4

# 训练 loss（验证/测试始终用 MSE 报告，保持 benchmark 可比）
#   mse       : 默认，与评估一致
#   huber     : 对 outlier 鲁棒；delta 控制 L2→L1 切换阈值
#   smooth_l1 : 同上，PyTorch 早期定义
# RevIN 归一空间下 HUBER_DELTA 建议 0.3~1.0
TRAIN_LOSS=mse
HUBER_DELTA=3.5

# =====================================================
# Overlapping chunk prediction（pretrain / forecast 解耦）
# =====================================================
# Pretrain: M_p = PROGRESSIVE_STEP_SIZE，N_p = PRETRAIN_PRED_LEN
#   PRETRAIN_PRED_LEN 留空 = 等于 PROGRESSIVE_STEP_SIZE（不启用预训练重叠）
# Forecast: M_f = FORECAST_STEP_SIZE，N_f = FORECAST_PRED_LEN
#   FORECAST_STEP_SIZE 留空 = 继承 checkpoint 的 progressive_step_size
#   FORECAST_PRED_LEN  留空 = 继承 checkpoint 的 pred_len；若 checkpoint 未记录则等于 M_f
#   长序列稳定性建议：N_f = 2~3 * M_f，让重叠 chunk 在概率级融合后再提交 token
# 示例：PROGRESSIVE_STEP_SIZE=6, PRETRAIN_PRED_LEN=6, FORECAST_STEP_SIZE=6, FORECAST_PRED_LEN=12
PRETRAIN_PRED_LEN=9
FORECAST_STEP_SIZE=3
FORECAST_PRED_LEN=9

# =====================================================
# 预训练早停参数
# =====================================================
EARLY_STOP_PATIENCE=5
EARLY_STOP_WARMUP=5
EARLY_STOP_MIN_DELTA=1e-4
EARLY_STOP_SMOOTH_K=1      # 用最近 K 个 epoch 的 val_loss 均值判据；1 = 不平滑

# =====================================================
# 跳过预训练（直接用已有模型做微调）
# 0 = 正常流程（先预训练再微调）
# 1 = 跳过预训练，直接加载下面指定的模型路径
# =====================================================
SKIP_PRETRAIN=0
EXISTING_PRETRAINED_MODEL=""   # SKIP_PRETRAIN=1 时填入绝对路径，例如：
# EXISTING_PRETRAINED_MODEL="/root/autodl-tmp/.../patch_vqvae_ps16_cb64_cd64_l3_in1152_step6_model1_perch.pth"

# =====================================================
# 其他参数
# =====================================================
REVIN=1
WEIGHT_DECAY=1e-4
FREEZE_VQVAE=1

# =====================================================
# 自动查找码本模型（如果指定路径不存在）
# =====================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 转换为绝对路径
if [[ ! "${CODEBOOK_CHECKPOINT}" = /* ]]; then
    CODEBOOK_CHECKPOINT="${SCRIPT_DIR}/${CODEBOOK_CHECKPOINT}"
fi
CODEBOOK_CHECKPOINT=$(readlink -f "${CODEBOOK_CHECKPOINT}" 2>/dev/null || realpath "${CODEBOOK_CHECKPOINT}" 2>/dev/null || echo "${CODEBOOK_CHECKPOINT}")

if [ ! -f "${CODEBOOK_CHECKPOINT}" ]; then
    echo "指定路径不存在，正在自动查找码本模型..."
    VQ_ONLY_DIR="${SCRIPT_DIR}/../vqvae-only/saved_models/vqvae_only"

    # per-channel 模式优先查找 _perch 后缀的模型
    if [ "${PER_CHANNEL_CODEBOOK}" -eq 1 ]; then
        CODEBOOK_CHECKPOINT=$(find "${VQ_ONLY_DIR}" -name "codebook_*_perch*.pth" -type f 2>/dev/null | head -1)
    fi
    # 若仍未找到，查找任意码本模型
    if [ -z "${CODEBOOK_CHECKPOINT}" ] || [ ! -f "${CODEBOOK_CHECKPOINT}" ]; then
        CODEBOOK_CHECKPOINT=$(find "${VQ_ONLY_DIR}" -name "codebook_*.pth" -type f 2>/dev/null | head -1)
    fi

    if [ -z "${CODEBOOK_CHECKPOINT}" ] || [ ! -f "${CODEBOOK_CHECKPOINT}" ]; then
        echo "错误: 未找到码本模型！"
        echo "请先运行 vqvae-only/codebook_pretrain.sh 训练码本，或手动设置 CODEBOOK_CHECKPOINT"
        exit 1
    fi
fi

echo "找到码本模型: ${CODEBOOK_CHECKPOINT}"

# =====================================================
# 计算 code_dim 与模型名称
# =====================================================
CODE_DIM=${CODE_DIM_CB}
PERCH_SUFFIX=""
[ "${PER_CHANNEL_CODEBOOK}" -eq 1 ] && PERCH_SUFFIX="_perch"
NMPP_SUFFIX=""
[ "${USE_RAW_INPUT}" -eq 1 ] && NMPP_SUFFIX="_nmpp"
RVQ_SUFFIX=""
[ "${N_RQ_LAYERS:-1}" -gt 1 ] && RVQ_SUFFIX="_rvq${N_RQ_LAYERS}"

# 注意：后缀顺序须与 patch_vqvae_pretrain_common.py 的命名保持一致
# Python 保存格式：..._step{N}_model{ID}{_perch}{_rvqN}{_nmpp}.pth
MODEL_NAME="patch_vqvae_ps${PATCH_SIZE}_cb${CODEBOOK_SIZE}_cd${CODE_DIM}_l${N_LAYERS}_in${PRETRAIN_CONTEXT_POINTS}_step${PROGRESSIVE_STEP_SIZE}_model${MODEL_ID}${PERCH_SUFFIX}${RVQ_SUFFIX}${NMPP_SUFFIX}"

echo "================================================="
echo "NTP 预训练 → 微调 流程"
echo "================================================="
echo "数据集         : ${DSET}"
echo "码本模型       : ${CODEBOOK_CHECKPOINT}"
echo "模型名称       : ${MODEL_NAME}"
echo "Pretrain M_p   : ${PROGRESSIVE_STEP_SIZE} patches"
echo "Pretrain N_p   : ${PRETRAIN_PRED_LEN:-<等于 M_p (无重叠)>}"
echo "Forecast M_f   : ${FORECAST_STEP_SIZE:-<继承 checkpoint M_p>} patches"
echo "Forecast N_f   : ${FORECAST_PRED_LEN:-<继承 checkpoint N_p / 默认等于 M_f>}"
echo "Transformer 维度(code_dim): ${CODE_DIM}"
echo "Per-channel VQ : ${PER_CHANNEL_CODEBOOK}"
echo "NMPP 模式      : ${USE_RAW_INPUT}"
echo "冻结 VQVAE     : ${FREEZE_VQVAE}"
echo "================================================="

# =====================================================
# 阶段 1: 渐进式预训练
# =====================================================
echo ""
echo "================================================="
echo "阶段 1: 渐进式预训练（NTP）"
echo "================================================="

PRETRAIN_ARGS=(
    --dset "${DSET}"
    --context_points "${PRETRAIN_CONTEXT_POINTS}"
    --progressive_step_size "${PROGRESSIVE_STEP_SIZE}"
    --batch_size "${PRETRAIN_BATCH_SIZE}"
    --patch_size "${PATCH_SIZE}"
    --embedding_dim "${EMBEDDING_DIM}"
    --compression_factor "${COMPRESSION_FACTOR}"
    --codebook_size "${CODEBOOK_SIZE}"
    --n_layers "${N_LAYERS}"
    --n_heads "${N_HEADS}"
    --d_ff "${D_FF}"
    --dropout "${DROPOUT}"
    --num_hiddens "${NUM_HIDDENS}"
    --num_residual_layers "${NUM_RESIDUAL_LAYERS}"
    --num_residual_hiddens "${NUM_RESIDUAL_HIDDENS}"
    --codebook_ema "${CODEBOOK_EMA}"
    --ema_decay "${EMA_DECAY}"
    --ema_eps "${EMA_EPS}"
    --disable_ema_update "${DISABLE_EMA_UPDATE}"
    --vqvae_checkpoint "${CODEBOOK_CHECKPOINT}"
    --freeze_vqvae "${FREEZE_VQVAE}"
    --load_vq_weights 1
    --per_channel_codebook "${PER_CHANNEL_CODEBOOK}"
    --use_raw_input "${USE_RAW_INPUT}"
    --n_rq_layers "${N_RQ_LAYERS:-1}"
    --n_epochs "${PRETRAIN_EPOCHS}"
    --lr "${PRETRAIN_LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --revin "${REVIN}"
    --vq_weight "${VQ_WEIGHT}"
    --recon_weight "${RECON_WEIGHT}"
    --model_id "${MODEL_ID}"
)
[ -n "${TRANSFORMER_HIDDEN_DIM}" ] && PRETRAIN_ARGS+=(--transformer_hidden_dim "${TRANSFORMER_HIDDEN_DIM}")
# nargs='+' 参数需展开为多个独立值
[ -n "${RQ_LAYER_WEIGHTS}" ] && PRETRAIN_ARGS+=(--rq_layer_weights ${RQ_LAYER_WEIGHTS})

# Pretrain overlapping chunk 参数（留空时不传，Python 端默认等于 step_size）
[ -n "${PRETRAIN_PRED_LEN}" ] && PRETRAIN_ARGS+=(--pred_len "${PRETRAIN_PRED_LEN}")

# 早停参数
PRETRAIN_ARGS+=(--early_stop_patience  "${EARLY_STOP_PATIENCE}")
PRETRAIN_ARGS+=(--early_stop_warmup    "${EARLY_STOP_WARMUP}")
PRETRAIN_ARGS+=(--early_stop_min_delta "${EARLY_STOP_MIN_DELTA}")
PRETRAIN_ARGS+=(--early_stop_smooth_k  "${EARLY_STOP_SMOOTH_K}")

python patch_vqvae_pretrain.py "${PRETRAIN_ARGS[@]}"

# =====================================================
# 阶段 2: 微调（多预测长度）
# =====================================================
PRETRAINED_MODEL="${SCRIPT_DIR}/saved_models/patch_vqvae/${DSET}/${MODEL_NAME}.pth"
PRETRAINED_MODEL=$(readlink -f "${PRETRAINED_MODEL}" 2>/dev/null || realpath "${PRETRAINED_MODEL}" 2>/dev/null || echo "${PRETRAINED_MODEL}")

if [ ! -f "${PRETRAINED_MODEL}" ]; then
    echo "错误: 预训练模型不存在: ${PRETRAINED_MODEL}"
    echo "期望模型名: ${MODEL_NAME}.pth"
    if [ -d "$(dirname "${PRETRAINED_MODEL}")" ]; then
        echo "目录内文件:"
        ls -la "$(dirname "${PRETRAINED_MODEL}")" | head -20
    fi
    exit 1
fi

echo ""
echo "================================================="
echo "阶段 2: 微调（${TARGET_POINTS_LIST[*]} 步预测）"
echo "================================================="
echo "预训练模型: ${PRETRAINED_MODEL}"

for TARGET_POINTS in "${TARGET_POINTS_LIST[@]}"; do
    echo ""
    echo "-------------------------------------------------"
    echo "微调: Target Points = ${TARGET_POINTS}"
    echo "Forecast M_f: ${FORECAST_STEP_SIZE:-<继承 checkpoint M_p>} | N_f: ${FORECAST_PRED_LEN:-<继承 checkpoint N_p / 默认等于 M_f>}"
    echo "-------------------------------------------------"

    python patch_vqvae_finetune.py \
        --dset "${DSET}" \
        --context_points "${FINETUNE_CONTEXT_POINTS}" \
        --target_points "${TARGET_POINTS}" \
        --batch_size "${FINETUNE_BATCH_SIZE}" \
        --pretrained_model "${PRETRAINED_MODEL}" \
        --n_epochs "${FINETUNE_EPOCHS}" \
        --lr "${FINETUNE_LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --revin "${REVIN}" \
        --use_gumbel_softmax "${USE_GUMBEL_SOFTMAX}" \
        --gumbel_temperature "${GUMBEL_TEMPERATURE}" \
        --gumbel_hard "${GUMBEL_HARD}" \
        --eval_top_k "${EVAL_TOP_K}" \
        --train_loss "${TRAIN_LOSS}" \
        --huber_delta "${HUBER_DELTA}" \
        ${FORECAST_STEP_SIZE:+--ar_step_size "${FORECAST_STEP_SIZE}"} \
        ${FORECAST_PRED_LEN:+--pred_len "${FORECAST_PRED_LEN}"} \
        --model_id "${MODEL_ID}"
done

echo ""
echo "================================================="
echo "全部完成！"
echo "================================================="
echo "预训练模型 : ${PRETRAINED_MODEL}"
echo "微调结果   : saved_models/patch_vqvae_finetune/${DSET}/"
