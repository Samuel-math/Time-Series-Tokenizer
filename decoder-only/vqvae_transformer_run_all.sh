#!/bin/bash

# =====================================================
# 基于预训练VQVAE的Transformer训练脚本
# 使用已训练好的VQVAE模型，全程冻结VQVAE参数
# =====================================================
# 
# 架构说明:
# 1. 加载预训练的VQVAE模型（冻结所有参数）
# 2. 在VQVAE基础上训练Transformer进行NTP预训练
# 3. 微调Transformer进行时间序列预测
# =====================================================

# =====================================================
# 配置参数
# =====================================================

DSET="ettm1"
MODEL_ID=1

# ----- VQVAE模型路径 -----
# 自动查找saved_models/vqvae下的best_model.pth，或手动指定
VQVAE_CHECKPOINT=""  # 如果为空，脚本会自动查找
# VQVAE_CHECKPOINT="saved_models/vqvae/ettm1/.../best_model.pth"  # 手动指定路径

# ----- Patch 参数 -----
PATCH_SIZE=16
COMPRESSION_FACTOR=4

# ----- VQVAE 参数（需要与预训练VQVAE匹配）-----
EMBEDDING_DIM=64
CODEBOOK_SIZE=14
NUM_HIDDENS=64
NUM_RESIDUAL_LAYERS=2
NUM_RESIDUAL_HIDDENS=32

# ----- Transformer 参数 -----
# code_dim = embedding_dim * (patch_size / compression_factor) = 64 * 4 = 256
# n_heads 需要整除 code_dim
N_LAYERS=3
N_HEADS=4
D_FF=256
DROPOUT=0.3

# ----- 预训练参数 -----
PRETRAIN_CONTEXT_POINTS=512
PRETRAIN_EPOCHS=100
PRETRAIN_BATCH_SIZE=128
PRETRAIN_LR=3e-4
VQ_WEIGHT=0.0  # VQVAE已冻结，设为0
RECON_WEIGHT=0.1

# ----- 微调参数 -----
FINETUNE_CONTEXT_POINTS=512
FINETUNE_EPOCHS=50
FINETUNE_BATCH_SIZE=64
FINETUNE_LR=1e-4
TARGET_POINTS_LIST=(96 192 336 720)

# ----- Patch Attention 参数 -----
USE_PATCH_ATTENTION=1  # 启用patch内时序建模(1启用)
PATCH_ATTENTION_TYPE="tcn"  # 时序建模类型: 'tcn' 或 'attention'
TCN_NUM_LAYERS=2  # TCN层数（仅TCN模式使用）
TCN_KERNEL_SIZE=3  # TCN卷积核大小（仅TCN模式使用）

# ----- 其他参数 -----
REVIN=1
WEIGHT_DECAY=1e-4
FREEZE_ENCODER_VQ=1  # 冻结encoder和VQ层
LOAD_VQ_WEIGHTS=1  # 加载VQ权重

# =====================================================
# 自动查找VQVAE模型（如果未指定）
# =====================================================

if [ -z "${VQVAE_CHECKPOINT}" ]; then
    echo "正在查找VQVAE模型..."
    # 查找saved_models/vqvae下的best_model.pth
    VQVAE_CHECKPOINT=$(find saved_models/vqvae -name "best_model.pth" -type f 2>/dev/null | head -1)
    
    if [ -z "${VQVAE_CHECKPOINT}" ]; then
        # 尝试查找checkpoints目录下的
        VQVAE_CHECKPOINT=$(find saved_models/vqvae -path "*/checkpoints/best_model.pth" -type f 2>/dev/null | head -1)
    fi
    
    if [ -z "${VQVAE_CHECKPOINT}" ]; then
        echo "错误: 未找到VQVAE模型！"
        echo "请手动设置 VQVAE_CHECKPOINT 变量，或确保 saved_models/vqvae 下有 best_model.pth 文件"
        exit 1
    fi
fi

if [ ! -f "${VQVAE_CHECKPOINT}" ]; then
    echo "错误: VQVAE模型文件不存在: ${VQVAE_CHECKPOINT}"
    exit 1
fi

echo "找到VQVAE模型: ${VQVAE_CHECKPOINT}"

# =====================================================
# 计算 code_dim 用于模型命名
# =====================================================
CODE_DIM=$((EMBEDDING_DIM * PATCH_SIZE / COMPRESSION_FACTOR))
MODEL_NAME="vqvae_transformer_ps${PATCH_SIZE}_cb${CODEBOOK_SIZE}_cd${CODE_DIM}_l${N_LAYERS}_model${MODEL_ID}"

echo "================================================="
echo "基于预训练VQVAE的Transformer训练"
echo "================================================="
echo "数据集: ${DSET}"
echo "VQVAE模型: ${VQVAE_CHECKPOINT}"
echo "模型名称: ${MODEL_NAME}"
echo "Transformer 输入维度 (code_dim): ${CODE_DIM}"
echo "冻结VQVAE: 是"
echo "Patch Attention: ${USE_PATCH_ATTENTION}"
if [ "${USE_PATCH_ATTENTION}" -eq 1 ]; then
    echo "Patch Attention 类型: ${PATCH_ATTENTION_TYPE}"
    if [ "${PATCH_ATTENTION_TYPE}" = "tcn" ]; then
        echo "  TCN层数: ${TCN_NUM_LAYERS}"
        echo "  TCN卷积核大小: ${TCN_KERNEL_SIZE}"
    fi
fi
echo "================================================="

# =====================================================
# 阶段 1: 预训练（NTP）
# =====================================================
echo ""
echo "================================================="
echo "阶段 1: NTP预训练（冻结VQVAE）"
echo "================================================="
echo "Context Points: ${PRETRAIN_CONTEXT_POINTS}"
echo "Epochs: ${PRETRAIN_EPOCHS}"
echo "Batch Size: ${PRETRAIN_BATCH_SIZE}"
echo "================================================="

python patch_vqvae_pretrain.py \
    --dset ${DSET} \
    --context_points ${PRETRAIN_CONTEXT_POINTS} \
    --batch_size ${PRETRAIN_BATCH_SIZE} \
    --patch_size ${PATCH_SIZE} \
    --embedding_dim ${EMBEDDING_DIM} \
    --compression_factor ${COMPRESSION_FACTOR} \
    --codebook_size ${CODEBOOK_SIZE} \
    --n_layers ${N_LAYERS} \
    --n_heads ${N_HEADS} \
    --d_ff ${D_FF} \
    --dropout ${DROPOUT} \
    --num_hiddens ${NUM_HIDDENS} \
    --num_residual_layers ${NUM_RESIDUAL_LAYERS} \
    --num_residual_hiddens ${NUM_RESIDUAL_HIDDENS} \
    --use_patch_attention ${USE_PATCH_ATTENTION} \
    --patch_attention_type ${PATCH_ATTENTION_TYPE} \
    --tcn_num_layers ${TCN_NUM_LAYERS} \
    --tcn_kernel_size ${TCN_KERNEL_SIZE} \
    --vqvae_checkpoint ${VQVAE_CHECKPOINT} \
    --freeze_encoder_vq ${FREEZE_ENCODER_VQ} \
    --load_vq_weights ${LOAD_VQ_WEIGHTS} \
    --n_epochs ${PRETRAIN_EPOCHS} \
    --lr ${PRETRAIN_LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --revin ${REVIN} \
    --vq_weight ${VQ_WEIGHT} \
    --recon_weight ${RECON_WEIGHT} \
    --model_id ${MODEL_ID}

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
echo "阶段 2: 微调（冻结VQVAE）"
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
