#!/bin/bash

# 两阶段预训练 + 微调 完整流程
# 
# 阶段1: Masked Reconstruction (类似 PatchTST)
# 中间步骤: 构建码本 (聚类)
# 阶段2: Next Token Prediction (NTP) - 使用更长的上下文
# 微调: 时间序列预测 (MSE) - 多个预测长度

DSET="ettm1"
CONTEXT_POINTS=512           # 阶段1和构建码本使用
CONTEXT_POINTS_STAGE2=1024   # 阶段2使用更长的上下文
PATCH_SIZE=16
D_MODEL=128
N_HEADS=8
N_LAYERS=4
CODEBOOK_SIZE=16
MASK_RATIO=0.4
MODEL_ID=1

SAVE_PATH="saved_models/two_stage/"
FINETUNE_SAVE_PATH="saved_models/two_stage_finetune/"

echo "=========================================="
echo "阶段1: Masked Reconstruction"
echo "Context Points: $CONTEXT_POINTS"
echo "=========================================="

python two_stage_pretrain_stage1.py \
    --dset $DSET \
    --context_points $CONTEXT_POINTS \
    --batch_size 128 \
    --patch_size $PATCH_SIZE \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_layers $N_LAYERS \
    --d_ff 256 \
    --dropout 0.1 \
    --mask_ratio $MASK_RATIO \
    --codebook_size $CODEBOOK_SIZE \
    --n_epochs 50 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --revin 1 \
    --save_path $SAVE_PATH \
    --model_id $MODEL_ID

# 阶段1模型路径
STAGE1_MODEL="${SAVE_PATH}${DSET}/stage1_ps${PATCH_SIZE}_dm${D_MODEL}_l${N_LAYERS}_mask${MASK_RATIO}_model${MODEL_ID}.pth"

echo ""
echo "=========================================="
echo "中间步骤: 构建码本"
echo "Context Points: $CONTEXT_POINTS"
echo "=========================================="

python two_stage_build_codebook.py \
    --dset $DSET \
    --context_points $CONTEXT_POINTS \
    --batch_size 128 \
    --stage1_model $STAGE1_MODEL \
    --codebook_size $CODEBOOK_SIZE \
    --cluster_method minibatch_kmeans \
    --save_path $SAVE_PATH \
    --revin 1

# 带码本的模型路径
MODEL_WITH_CB="${SAVE_PATH}${DSET}/stage1_ps${PATCH_SIZE}_dm${D_MODEL}_l${N_LAYERS}_mask${MASK_RATIO}_model${MODEL_ID}_cb${CODEBOOK_SIZE}.pth"

echo ""
echo "=========================================="
echo "阶段2: Next Token Prediction"
echo "Context Points: $CONTEXT_POINTS_STAGE2 (更长)"
echo "=========================================="

python two_stage_pretrain_stage2.py \
    --dset $DSET \
    --context_points $CONTEXT_POINTS_STAGE2 \
    --batch_size 64 \
    --model_with_codebook $MODEL_WITH_CB \
    --n_epochs 30 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --revin 1 \
    --save_path $SAVE_PATH \
    --model_id $MODEL_ID

# 阶段2模型路径
STAGE2_MODEL="${SAVE_PATH}${DSET}/stage1_ps${PATCH_SIZE}_dm${D_MODEL}_l${N_LAYERS}_mask${MASK_RATIO}_model${MODEL_ID}_cb${CODEBOOK_SIZE}_ntp_model${MODEL_ID}.pth"

# 多个预测长度
TARGET_POINTS_LIST=(96 192 336 512)

echo ""
echo "=========================================="
echo "微调: 多个预测长度"
echo "Target Points: ${TARGET_POINTS_LIST[@]}"
echo "=========================================="

for TARGET_POINTS in "${TARGET_POINTS_LIST[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "Target Points: $TARGET_POINTS"
    echo "------------------------------------------"
    
    echo ""
    echo ">>> Linear Probe (TARGET=$TARGET_POINTS)"
    python two_stage_finetune.py \
        --dset $DSET \
        --context_points $CONTEXT_POINTS \
        --target_points $TARGET_POINTS \
        --batch_size 64 \
        --pretrained_model $STAGE2_MODEL \
        --finetune_mode linear_probe \
        --n_epochs 10 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --revin 1 \
        --save_path $FINETUNE_SAVE_PATH \
        --model_id $MODEL_ID
    
    echo ""
    echo ">>> Full Finetune (TARGET=$TARGET_POINTS)"
    python two_stage_finetune.py \
        --dset $DSET \
        --context_points $CONTEXT_POINTS \
        --target_points $TARGET_POINTS \
        --batch_size 64 \
        --pretrained_model $STAGE2_MODEL \
        --finetune_mode full \
        --n_epochs 20 \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --revin 1 \
        --save_path $FINETUNE_SAVE_PATH \
        --model_id $MODEL_ID
done

echo ""
echo "=========================================="
echo "完成！所有预测长度的结果已保存"
echo "=========================================="
echo "结果保存在: $FINETUNE_SAVE_PATH"
echo "预测长度: ${TARGET_POINTS_LIST[@]}"
