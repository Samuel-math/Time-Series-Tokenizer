#!/bin/bash

# Patch VQVAE Transformer 微调脚本
# 默认冻结 encoder 和 VQ 层，只训练 Transformer 和 Decoder

# 预训练模型路径 (根据实际情况修改)
PRETRAINED_MODEL="saved_models/patch_vqvae/ettm1/patch_vqvae_ps16_cb14_cd256_l3_model1.pth"

python patch_vqvae_finetune.py \
    --dset ettm1 \
    --context_points 512 \
    --target_points 96 \
    --batch_size 128 \
    --pretrained_model $PRETRAINED_MODEL \
    --n_epochs 30 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --revin 1 \
    --model_id 1
