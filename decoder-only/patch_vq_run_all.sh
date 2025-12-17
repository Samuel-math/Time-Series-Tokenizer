#!/bin/bash

# =====================================================
# Patch VQ + Transformer 预训练 + 微调脚本
# 使用MLM训练的VQ_encoder和codebook
# =====================================================

# =====================================================
# 配置参数
# =====================================================

DSET="ettm1"
MODEL_ID=1

# ----- Patch 参数 -----
PATCH_SIZE=16

# ----- MLM codebook路径（必需）-----
MLM_CODEBOOK_PATH="../codebook/saved_models/codebook/${DSET}/mlm_ps16_dm128_l4_mask0.4_model1_codebook_cb256.pth"

# ----- Codebook 参数（从MLM codebook获取）-----
CODEBOOK_SIZE=256

# ----- Transformer 参数 -----
# code_dim 从MLM的d_model获取（通常是128）
N_LAYERS=4
N_HEADS=4
D_FF=256
DROPOUT=0.1

# ----- 预训练参数 -----
PRETRAIN_CONTEXT_POINTS=512
PRETRAIN_TARGET_POINTS=96
PRETRAIN_EPOCHS=100
PRETRAIN_BATCH_SIZE=128
PRETRAIN_LR=1e-4
VQ_WEIGHT=1.0
RECON_WEIGHT=0.1

# ----- 微调参数 -----
FINETUNE_CONTEXT_POINTS=512
FINETUNE_EPOCHS=50
FINETUNE_BATCH_SIZE=64
FINETUNE_LR=1e-4
TARGET_POINTS_LIST=(96 192 336 720)

# ----- Patch Attention 参数 -----
USE_PATCH_ATTENTION=0
PATCH_ATTENTION_TYPE="tcn"
TCN_NUM_LAYERS=2
TCN_KERNEL_SIZE=3

# ----- 其他参数 -----
REVIN=1
WEIGHT_DECAY=1e-4

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
echo "MLM Codebook: ${MLM_CODEBOOK_PATH}"
echo "================================================="

python patch_vq_pretrain.py \
    --dset ${DSET} \
    --context_points ${PRETRAIN_CONTEXT_POINTS} \
    --target_points ${PRETRAIN_TARGET_POINTS} \
    --batch_size ${PRETRAIN_BATCH_SIZE} \
    --patch_size ${PATCH_SIZE} \
    --codebook_size ${CODEBOOK_SIZE} \
    --n_layers ${N_LAYERS} \
    --n_heads ${N_HEADS} \
    --d_ff ${D_FF} \
    --dropout ${DROPOUT} \
    --mlm_codebook_path "${MLM_CODEBOOK_PATH}" \
    --load_mlm_vq_encoder 1 \
    --load_mlm_codebook 1 \
    --freeze_mlm_components 1 \
    --use_patch_attention ${USE_PATCH_ATTENTION} \
    --patch_attention_type ${PATCH_ATTENTION_TYPE} \
    --tcn_num_layers ${TCN_NUM_LAYERS} \
    --tcn_kernel_size ${TCN_KERNEL_SIZE} \
    --n_epochs ${PRETRAIN_EPOCHS} \
    --lr ${PRETRAIN_LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --revin ${REVIN} \
    --vq_weight ${VQ_WEIGHT} \
    --recon_weight ${RECON_WEIGHT} \
    --model_id ${MODEL_ID}

# =====================================================
# 阶段 2: 微调 (多个预测长度)
# =====================================================
PRETRAINED_MODEL="saved_models/patch_vq/${DSET}/patch_vq_ps${PATCH_SIZE}_cb${CODEBOOK_SIZE}_cd128_l${N_LAYERS}_model${MODEL_ID}.pth"

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
    
    python patch_vq_finetune.py \
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
