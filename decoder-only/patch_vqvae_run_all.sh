#!/bin/bash

# =====================================================
# Patch VQVAE + Transformer 预训练 + 微调脚本
# =====================================================
# 
# 架构说明:
# 1. 输入: [B, T, C] 时间序列
# 2. Patch划分 + VQVAE Encoder -> [B, num_patches, C, code_dim]
# 3. VQ 量化后直接作为 Transformer 输入
# 4. Transformer (Decoder-only): 预测下一个码本索引
# 5. 预训练: NTP loss + Reconstruction loss
# 6. 微调: 预测未来patch -> 解码 -> MSE loss
#
# Transformer 输入维度 = embedding_dim * (patch_size / compression_factor)
# 例: 64 * (16 / 4) = 256
# =====================================================

# =====================================================
# 配置参数
# =====================================================

DSET="ettm1"
MODEL_ID=1

# ----- Patch 参数 -----
PATCH_SIZE=16
COMPRESSION_FACTOR=4

# ----- VQVAE 参数 -----
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
PRETRAIN_CONTEXT_POINTS=512  # 预训练使用更长的窗口
PRETRAIN_EPOCHS=100
PRETRAIN_BATCH_SIZE=128  # 窗口变大，batch_size 相应减小
PRETRAIN_LR=3e-4
VQ_WEIGHT=0.5
RECON_WEIGHT=0.1

# ----- 微调参数 -----
FINETUNE_CONTEXT_POINTS=512  # 微调使用标准窗口
FINETUNE_EPOCHS=50
FINETUNE_BATCH_SIZE=64
FINETUNE_LR=1e-4
TARGET_POINTS_LIST=(96 192 336 720)

# ----- 残差量化参数（减少量化误差）-----
USE_RESIDUAL_VQ=0  # 是否使用残差量化（1启用，0禁用）
RESIDUAL_VQ_LAYERS=2  # 残差量化层数（建议2-3层）
RESIDUAL_VQ_COMBINE_METHOD="sum"  # 合并方式：sum（相加）或concat（拼接）
RESIDUAL_VQ_CODEBOOK_SIZES=""  # 每层codebook大小，用逗号分隔，如 "256,128"。如果为空，所有层使用统一的CODEBOOK_SIZE
VQ_INIT_METHOD="uniform"  # 码本初始化方法: uniform/normal/xavier/kaiming

# ----- 其他参数 -----
REVIN=1
WEIGHT_DECAY=1e-4

# =====================================================
# 计算 code_dim 用于模型命名
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
echo "Patch VQVAE + Transformer 预训练 + 微调"
echo "================================================="
echo "数据集: ${DSET}"
echo "模型名称: ${MODEL_NAME}"
echo "Transformer 输入维度 (code_dim): ${CODE_DIM}"
if [ "${USE_RESIDUAL_VQ}" -eq 1 ]; then
    echo "残差量化: 启用 (${RESIDUAL_VQ_LAYERS}层)"
    echo "  合并方式: ${RESIDUAL_VQ_COMBINE_METHOD}"
    if [ -n "${RESIDUAL_VQ_CODEBOOK_SIZES}" ]; then
        echo "  每层codebook大小: ${RESIDUAL_VQ_CODEBOOK_SIZES}"
    else
        echo "  每层codebook大小: ${CODEBOOK_SIZE} (统一)"
    fi
else
    echo "残差量化: 禁用"
fi
echo "================================================="

# =====================================================
# 阶段 1: 预训练
# =====================================================
echo ""
echo "================================================="
echo "阶段 1: 预训练"
echo "================================================="
echo "Context Points: ${PRETRAIN_CONTEXT_POINTS}"
echo "Epochs: ${PRETRAIN_EPOCHS}"
echo "Batch Size: ${PRETRAIN_BATCH_SIZE}"
echo "================================================="

# 构建命令参数
PRETRAIN_ARGS=(
    --dset ${DSET}
    --context_points ${PRETRAIN_CONTEXT_POINTS}
    --batch_size ${PRETRAIN_BATCH_SIZE}
    --patch_size ${PATCH_SIZE}
    --embedding_dim ${EMBEDDING_DIM}
    --compression_factor ${COMPRESSION_FACTOR}
    --n_layers ${N_LAYERS}
    --n_heads ${N_HEADS}
    --d_ff ${D_FF}
    --dropout ${DROPOUT}
    --num_hiddens ${NUM_HIDDENS}
    --num_residual_layers ${NUM_RESIDUAL_LAYERS}
    --num_residual_hiddens ${NUM_RESIDUAL_HIDDENS}
    --use_residual_vq ${USE_RESIDUAL_VQ}
    --residual_vq_layers ${RESIDUAL_VQ_LAYERS}
    --residual_vq_combine_method ${RESIDUAL_VQ_COMBINE_METHOD}
    --residual_vq_codebook_sizes "${RESIDUAL_VQ_CODEBOOK_SIZES}"
    --vq_init_method ${VQ_INIT_METHOD}
    --n_epochs ${PRETRAIN_EPOCHS}
    --lr ${PRETRAIN_LR}
    --weight_decay ${WEIGHT_DECAY}
    --revin ${REVIN}
    --vq_weight ${VQ_WEIGHT}
    --recon_weight ${RECON_WEIGHT}
    --model_id ${MODEL_ID}
)

# 只在未使用残差量化或残差量化未指定每层大小时传递codebook_size
if [ "${USE_RESIDUAL_VQ}" -eq 0 ] || [ -z "${RESIDUAL_VQ_CODEBOOK_SIZES}" ]; then
    PRETRAIN_ARGS+=(--codebook_size ${CODEBOOK_SIZE})
fi

python patch_vqvae_pretrain.py "${PRETRAIN_ARGS[@]}"

# =====================================================
# 阶段 2: 微调 (多个预测长度)
# =====================================================
PRETRAINED_MODEL="saved_models/patch_vqvae/${DSET}/${MODEL_NAME}.pth"

echo ""
echo "================================================="
echo "阶段 2: 微调"
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
