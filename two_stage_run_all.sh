#!/bin/bash

# 两阶段预训练 + 微调 完整流程
# 
# 阶段1: Masked Reconstruction (类似 PatchTST)
# 中间步骤: 构建码本 (聚类)
# 阶段2: Next Token Prediction (NTP)
# 微调: 时间序列预测 (MSE)

DSET="ettm1"
CONTEXT_POINTS=512
TARGET_POINTS=96
PATCH_SIZE=16
D_MODEL=128
N_HEADS=8
N_LAYERS=4
CODEBOOK_SIZE=256
MASK_RATIO=0.4
MODEL_ID=1

SAVE_PATH="saved_models/two_stage/"

echo "=========================================="
echo "阶段1: Masked Reconstruction"
echo "=========================================="

python two_stage_pretrain_stage1.py \
    --dset $DSET \
    --context_points $CONTEXT_POINTS \
    --target_points $TARGET_POINTS \
    --batch_size 128 \
    --patch_size $PATCH_SIZE \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_layers $N_LAYERS \
    --d_ff 256 \
    --dropout 0.1 \
    --mask_ratio $MASK_RATIO \
    --codebook_size $CODEBOOK_SIZE \
    --n_epochs 100 \
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
echo "=========================================="

python two_stage_build_codebook.py \
    --dset $DSET \
    --context_points $CONTEXT_POINTS \
    --target_points $TARGET_POINTS \
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
echo "=========================================="

python two_stage_pretrain_stage2.py \
    --dset $DSET \
    --context_points $CONTEXT_POINTS \
    --target_points $TARGET_POINTS \
    --batch_size 128 \
    --model_with_codebook $MODEL_WITH_CB \
    --n_epochs 100 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --revin 1 \
    --save_path $SAVE_PATH \
    --model_id $MODEL_ID

# 阶段2模型路径
STAGE2_MODEL="${SAVE_PATH}${DSET}/stage1_ps${PATCH_SIZE}_dm${D_MODEL}_l${N_LAYERS}_mask${MASK_RATIO}_model${MODEL_ID}_cb${CODEBOOK_SIZE}_ntp_model${MODEL_ID}.pth"

echo ""
echo "=========================================="
echo "微调1: Linear Probe (只训练 pred_head)"
echo "=========================================="

python two_stage_finetune.py \
    --dset $DSET \
    --context_points $CONTEXT_POINTS \
    --target_points $TARGET_POINTS \
    --batch_size 64 \
    --pretrained_model $STAGE2_MODEL \
    --finetune_mode linear_probe \
    --n_epochs 50 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --revin 1 \
    --save_path "saved_models/two_stage_finetune/" \
    --model_id $MODEL_ID

echo ""
echo "=========================================="
echo "微调2: Full Finetune (全量微调)"
echo "=========================================="

python two_stage_finetune.py \
    --dset $DSET \
    --context_points $CONTEXT_POINTS \
    --target_points $TARGET_POINTS \
    --batch_size 64 \
    --pretrained_model $STAGE2_MODEL \
    --finetune_mode full \
    --n_epochs 50 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --revin 1 \
    --save_path "saved_models/two_stage_finetune/" \
    --model_id $MODEL_ID

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="
