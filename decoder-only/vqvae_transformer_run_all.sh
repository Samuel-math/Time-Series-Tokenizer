#!/bin/bash

# =====================================================
# 基于预训练码本模型的Transformer训练脚本
# 使用已训练好的码本模型（Encoder + VQ + Decoder），全程冻结VQVAE参数
# =====================================================
# 
# 架构说明:
# 1. 加载预训练的码本模型（encoder、decoder、VQ，冻结所有参数）
# 2. 在码本基础上训练Transformer进行NTP预训练
# 3. 微调Transformer进行时间序列预测
# =====================================================

# =====================================================
# 配置参数
# =====================================================

DSET="ettm1"
MODEL_ID=1

# ----- 码本模型路径（相对于decoder-only目录）-----
# 如果为空，脚本会自动查找，或手动指定完整路径
CODEBOOK_CHECKPOINT="../vqvae-only/saved_models/vqvae_only/${DSET}/codebook_ps16_cb256_cd128_model1.pth"
# CODEBOOK_CHECKPOINT=""  # 如果为空，脚本会自动查找

# ----- Patch 参数（会从码本checkpoint中自动读取，这里作为备用）-----
PATCH_SIZE=16
COMPRESSION_FACTOR=4

# ----- VQVAE 参数（会从码本checkpoint中自动读取，这里作为备用）-----
EMBEDDING_DIM=32
# CODEBOOK_SIZE: 仅在不使用残差量化时使用
# 如果使用残差量化，请使用RESIDUAL_VQ_CODEBOOK_SIZES参数
CODEBOOK_SIZE=256
NUM_HIDDENS=64
NUM_RESIDUAL_LAYERS=2
NUM_RESIDUAL_HIDDENS=32

# ----- Transformer 参数 -----
# code_dim = embedding_dim * (patch_size / compression_factor)
# n_heads 需要整除 code_dim
N_LAYERS=4
N_HEADS=4
D_FF=256
DROPOUT=0.1
CODEBOOK_EMA=1
EMA_DECAY=0.99
EMA_EPS=1e-5
# transformer_hidden_dim: Transformer的hidden_dim，默认使用code_dim
# 如果设置，Transformer内部将使用此维度，输入输出通过投影层与code_dim转换
TRANSFORMER_HIDDEN_DIM=""  # 留空表示使用默认值（code_dim），可根据需要设置

# ----- 预训练参数 -----
PRETRAIN_CONTEXT_POINTS=512
PRETRAIN_TARGET_POINTS=96  # 预训练时target序列的长度
PRETRAIN_EPOCHS=100
PRETRAIN_BATCH_SIZE=128
PRETRAIN_LR=3e-4
VQ_WEIGHT=0.0  # 码本已冻结，设为0
RECON_WEIGHT=0.0  # 码本已冻结，设为0

# ----- 微调参数 -----
FINETUNE_CONTEXT_POINTS=512
FINETUNE_EPOCHS=50
FINETUNE_BATCH_SIZE=64
FINETUNE_LR=1e-4
TARGET_POINTS_LIST=(96 192 336 720)

# ----- 残差量化参数（减少量化误差）-----
USE_RESIDUAL_VQ=1  # 是否使用残差量化（1启用，0禁用）
RESIDUAL_VQ_LAYERS=2  # 残差量化层数（建议2-3层）
RESIDUAL_VQ_COMBINE_METHOD="sum"  # 合并方式：sum（相加）或concat（拼接）
RESIDUAL_VQ_CODEBOOK_SIZES="256,128"  # 每层codebook大小，用逗号分隔，如 "256,128"。如果为空，所有层使用统一的CODEBOOK_SIZE
VQ_INIT_METHOD="uniform"  # 码本初始化方法: uniform/normal/xavier/kaiming

# ----- 其他参数 -----
REVIN=1
WEIGHT_DECAY=1e-4
FREEZE_VQVAE=1  # 是否冻结VQVAE组件（1冻结，0不冻结）

# =====================================================
# 自动查找码本模型（如果未指定）
# =====================================================

if [ -z "${CODEBOOK_CHECKPOINT}" ]; then
    echo "正在查找码本模型..."
    # 查找vqvae-only目录下的码本模型
    CODEBOOK_CHECKPOINT=$(find ../vqvae-only/saved_models/vqvae_only -name "codebook_*.pth" -type f 2>/dev/null | head -1)
    
    if [ -z "${CODEBOOK_CHECKPOINT}" ]; then
        echo "错误: 未找到码本模型！"
        echo "请手动设置 CODEBOOK_CHECKPOINT 变量，或确保 ../vqvae-only/saved_models/vqvae_only 下有 codebook_*.pth 文件"
        exit 1
    fi
fi

# 处理相对路径（相对于decoder-only目录）
if [[ ! "${CODEBOOK_CHECKPOINT}" = /* ]]; then
    # 获取脚本所在目录（decoder-only）
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # 转换为绝对路径
    CODEBOOK_CHECKPOINT="${SCRIPT_DIR}/${CODEBOOK_CHECKPOINT}"
fi

# 规范化路径（处理 .. 和 .）
CODEBOOK_CHECKPOINT=$(readlink -f "${CODEBOOK_CHECKPOINT}" 2>/dev/null || realpath "${CODEBOOK_CHECKPOINT}" 2>/dev/null || echo "${CODEBOOK_CHECKPOINT}")

if [ ! -f "${CODEBOOK_CHECKPOINT}" ]; then
    echo "错误: 码本模型文件不存在: ${CODEBOOK_CHECKPOINT}"
    exit 1
fi

echo "找到码本模型: ${CODEBOOK_CHECKPOINT}"

# =====================================================
# 计算 code_dim 用于模型命名（会从checkpoint中读取实际值）
# =====================================================
CODE_DIM=$((EMBEDDING_DIM * PATCH_SIZE / COMPRESSION_FACTOR))
# 模型命名：如果使用残差量化且指定了每层大小，使用第一层大小；否则使用CODEBOOK_SIZE
if [ "${USE_RESIDUAL_VQ}" -eq 1 ] && [ -n "${RESIDUAL_VQ_CODEBOOK_SIZES}" ]; then
    # 提取第一层codebook大小
    FIRST_CB_SIZE=$(echo "${RESIDUAL_VQ_CODEBOOK_SIZES}" | cut -d',' -f1)
    MODEL_NAME="patch_vqvae_ps${PATCH_SIZE}_cb${FIRST_CB_SIZE}_cd${CODE_DIM}_l${N_LAYERS}_model${MODEL_ID}"
else
    MODEL_NAME="patch_vqvae_ps${PATCH_SIZE}_cb${CODEBOOK_SIZE}_cd${CODE_DIM}_l${N_LAYERS}_model${MODEL_ID}"
fi

echo "================================================="
echo "基于预训练码本模型的Transformer训练"
echo "================================================="
echo "数据集: ${DSET}"
echo "码本模型: ${CODEBOOK_CHECKPOINT}"
echo "模型名称: ${MODEL_NAME}"
echo "Transformer 输入维度 (code_dim): ${CODE_DIM} (实际值从checkpoint读取)"
if [ "${FREEZE_VQVAE}" -eq 1 ]; then
    echo "冻结VQVAE: 是（Encoder + Decoder + VQ）"
else
    echo "冻结VQVAE: 否（所有参数可训练）"
fi
echo "================================================="

# =====================================================
# 阶段 1: 预训练（NTP，基于码本模型）
# =====================================================
echo ""
echo "================================================="
echo "阶段 1: NTP预训练（冻结码本：Encoder + Decoder + VQ）"
echo "================================================="
echo "Context Points: ${PRETRAIN_CONTEXT_POINTS}"
echo "Epochs: ${PRETRAIN_EPOCHS}"
echo "Batch Size: ${PRETRAIN_BATCH_SIZE}"
echo "================================================="

# 构建命令参数
PRETRAIN_ARGS=(
    --dset ${DSET}
    --context_points ${PRETRAIN_CONTEXT_POINTS}
    --target_points ${PRETRAIN_TARGET_POINTS}
    --batch_size ${PRETRAIN_BATCH_SIZE}
    --patch_size ${PATCH_SIZE}
    --embedding_dim ${EMBEDDING_DIM}
    --compression_factor ${COMPRESSION_FACTOR}
    --n_layers ${N_LAYERS}
    --n_heads ${N_HEADS}
    --d_ff ${D_FF}
    --dropout ${DROPOUT}
)

# 只在未使用残差量化或残差量化未指定每层大小时传递codebook_size
if [ "${USE_RESIDUAL_VQ}" -eq 0 ] || [ -z "${RESIDUAL_VQ_CODEBOOK_SIZES}" ]; then
    PRETRAIN_ARGS+=(--codebook_size ${CODEBOOK_SIZE})
fi

# 只在 TRANSFORMER_HIDDEN_DIM 不为空时添加该参数
if [ -n "${TRANSFORMER_HIDDEN_DIM}" ]; then
    PRETRAIN_ARGS+=(--transformer_hidden_dim ${TRANSFORMER_HIDDEN_DIM})
fi

PRETRAIN_ARGS+=(
    --num_hiddens ${NUM_HIDDENS} \
    --num_residual_layers ${NUM_RESIDUAL_LAYERS} \
    --num_residual_hiddens ${NUM_RESIDUAL_HIDDENS} \
    --codebook_ema ${CODEBOOK_EMA} \
    --ema_decay ${EMA_DECAY} \
    --ema_eps ${EMA_EPS} \
    --use_residual_vq ${USE_RESIDUAL_VQ} \
    --residual_vq_layers ${RESIDUAL_VQ_LAYERS} \
    --residual_vq_combine_method ${RESIDUAL_VQ_COMBINE_METHOD} \
    --residual_vq_codebook_sizes "${RESIDUAL_VQ_CODEBOOK_SIZES}" \
    --vq_init_method ${VQ_INIT_METHOD} \
    --vqvae_checkpoint "${CODEBOOK_CHECKPOINT}" \
    --freeze_vqvae ${FREEZE_VQVAE} \
    --load_vq_weights 1 \
    --n_epochs ${PRETRAIN_EPOCHS} \
    --lr ${PRETRAIN_LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --revin ${REVIN} \
    --vq_weight ${VQ_WEIGHT} \
    --recon_weight ${RECON_WEIGHT} \
    --model_id ${MODEL_ID}
)

python patch_vqvae_pretrain.py "${PRETRAIN_ARGS[@]}"

if [ $? -ne 0 ]; then
    echo "预训练失败，退出"
    exit 1
fi

# =====================================================
# 阶段 2: 微调 (多个预测长度)
# =====================================================
PRETRAINED_MODEL="saved_models/patch_vqvae/${DSET}/${MODEL_NAME}.pth"

echo ""
echo "================================================="
echo "阶段 2: 微调（冻结码本：Encoder + Decoder + VQ）"
echo "================================================="
echo "预训练模型: ${PRETRAINED_MODEL}"
echo "Context Points: ${FINETUNE_CONTEXT_POINTS}"
echo "Target Points: ${TARGET_POINTS_LIST[@]}"
echo "================================================="

for TARGET_POINTS in ${TARGET_POINTS_LIST[@]}; do
    echo ""
    echo "-------------------------------------------------"
    echo "微调: Target Points = ${TARGET_POINTS}"
    echo "-------------------------------------------------"
    
    python patch_vqvae_finetune.py \
        --dset ${DSET} \
        --context_points ${FINETUNE_CONTEXT_POINTS} \
        --target_points ${TARGET_POINTS} \
        --batch_size ${FINETUNE_BATCH_SIZE} \
        --pretrained_model ${PRETRAINED_MODEL} \
        --n_epochs ${FINETUNE_EPOCHS} \
        --lr ${FINETUNE_LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --revin ${REVIN} \
        --model_id ${MODEL_ID}
done

echo ""
echo "================================================="
echo "全部完成！"
echo "================================================="
echo "预训练模型: ${PRETRAINED_MODEL}"
echo "微调模型保存在: saved_models/patch_vqvae_finetune/${DSET}/"
