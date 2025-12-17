#!/bin/bash

# Patch VQ Transformer 微调脚本

# 预训练模型路径 (需要先运行 patch_vq_pretrain.sh)
PRETRAINED_MODEL="saved_models/patch_vq/ettm1/patch_vq_ps16_cb256_cd128_l4_model1.pth"

python patch_vq_finetune.py \
    --dset ettm1 \
    --context_points 512 \
    --target_points 96 \
    --batch_size 64 \
    --pretrained_model $PRETRAINED_MODEL \
    --n_epochs 50 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --revin 1 \
    --model_id 1
