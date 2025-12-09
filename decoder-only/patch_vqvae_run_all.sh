#!/bin/bash

# Patch VQVAE + Transformer 完整训练流程
# 
# 特性:
# - Overlapping Patches (stride 控制重叠)
# - Patch Encoder: 线性投影 [patch_size * C] -> [d_model]
# - VQ Codebook (对每个 patch 的 d_model 表示进行量化)
# - Decoder-only Transformer (NTP 预训练)

DSET="ettm1"
MODEL_ID=1

# ========== 窗口大小配置 ==========
PRETRAIN_CONTEXT_POINTS=1024    # 预训练使用较大窗口
FINETUNE_CONTEXT_POINTS=336     # 微调使用较小窗口

# ========== Patch 配置 ==========
PATCH_SIZE=32
STRIDE=16         # stride < patch_size 有重叠

# ========== 模型配置 ==========
D_MODEL=128           # 每个 patch 编码后的维度
CODEBOOK_SIZE=16
COMMITMENT_COST=0.25

# ========== Transformer 配置 ==========
N_LAYERS=3
N_HEADS=8
D_FF=256
DROPOUT=0.3

# ========== 训练配置 ==========
PRETRAIN_EPOCHS=100
FINETUNE_EPOCHS=50
BATCH_SIZE=64
LR=1e-4

# ========== 损失权重 ==========
VQ_WEIGHT=0.3
RECON_WEIGHT=0.1

# ========== 路径配置 ==========
PRETRAIN_SAVE_PATH="saved_models/patch_vqvae/"
FINETUNE_SAVE_PATH="saved_models/patch_vqvae_finetune/"

# 计算一些值
OVERLAP=$((PATCH_SIZE - STRIDE))
PRETRAIN_NUM_PATCHES=$(( (PRETRAIN_CONTEXT_POINTS - PATCH_SIZE) / STRIDE + 1 ))
FINETUNE_NUM_PATCHES=$(( (FINETUNE_CONTEXT_POINTS - PATCH_SIZE) / STRIDE + 1 ))

echo "=========================================="
echo "Patch VQVAE + Transformer"
echo "=========================================="
echo ""
echo "窗口配置:"
echo "  - 预训练窗口: $PRETRAIN_CONTEXT_POINTS (num_patches: $PRETRAIN_NUM_PATCHES)"
echo "  - 微调窗口: $FINETUNE_CONTEXT_POINTS (num_patches: $FINETUNE_NUM_PATCHES)"
echo ""
echo "Patch配置:"
echo "  - patch_size: $PATCH_SIZE"
echo "  - stride: $STRIDE"
echo "  - overlap: $OVERLAP"
echo ""
echo "模型配置:"
echo "  - d_model: $D_MODEL"
echo "  - codebook_size: $CODEBOOK_SIZE"
echo ""
echo "Transformer:"
echo "  - n_layers: $N_LAYERS"
echo "  - n_heads: $N_HEADS"
echo "=========================================="

echo ""
echo "=========================================="
echo "阶段1: 预训练 (NTP + Reconstruction)"
echo "=========================================="

python patch_vqvae_pretrain.py \
    --dset $DSET \
    --context_points $PRETRAIN_CONTEXT_POINTS \
    --batch_size $BATCH_SIZE \
    --patch_size $PATCH_SIZE \
    --stride $STRIDE \
    --d_model $D_MODEL \
    --codebook_size $CODEBOOK_SIZE \
    --commitment_cost $COMMITMENT_COST \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --d_ff $D_FF \
    --dropout $DROPOUT \
    --n_epochs $PRETRAIN_EPOCHS \
    --lr $LR \
    --weight_decay 1e-4 \
    --revin 1 \
    --vq_weight $VQ_WEIGHT \
    --recon_weight $RECON_WEIGHT \
    --save_path $PRETRAIN_SAVE_PATH \
    --model_id $MODEL_ID

# 预训练模型路径 (包含窗口大小)
PRETRAIN_MODEL="${PRETRAIN_SAVE_PATH}${DSET}/patch_vqvae_v2_cw${PRETRAIN_CONTEXT_POINTS}_ps${PATCH_SIZE}_st${STRIDE}_d${D_MODEL}_cb${CODEBOOK_SIZE}_model${MODEL_ID}.pth"

# 多个预测长度
TARGET_POINTS_LIST=(96 192 336 720)

echo ""
echo "=========================================="
echo "阶段2: 微调 (多个预测长度)"
echo "Target Points: ${TARGET_POINTS_LIST[@]}"
echo "=========================================="

for TARGET_POINTS in "${TARGET_POINTS_LIST[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "Target Points: $TARGET_POINTS"
    echo "------------------------------------------"
    
    python patch_vqvae_finetune.py \
        --dset $DSET \
        --context_points $FINETUNE_CONTEXT_POINTS \
        --target_points $TARGET_POINTS \
        --batch_size $BATCH_SIZE \
        --pretrained_model $PRETRAIN_MODEL \
        --freeze_encoder 0 \
        --freeze_transformer 0 \
        --n_epochs $FINETUNE_EPOCHS \
        --lr $LR \
        --weight_decay 1e-4 \
        --revin 1 \
        --save_path $FINETUNE_SAVE_PATH \
        --model_id $MODEL_ID
done

echo ""
echo "=========================================="
echo "完成！所有预测长度的结果已保存"
echo "=========================================="
echo "预训练模型: $PRETRAIN_MODEL"
echo "微调结果保存在: $FINETUNE_SAVE_PATH"
echo "预测长度: ${TARGET_POINTS_LIST[@]}"
