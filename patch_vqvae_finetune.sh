#!/bin/bash

# Patch VQVAE Transformer 微调脚本

# 预训练模型路径 (根据实际情况修改)
PRETRAINED_MODEL="saved_models/patch_vqvae/ettm1/patch_vqvae_ps16_cb512_d256_l6_model1.pth"

python patch_vqvae_finetune.py \
    --dset ettm1 \
    --context_points 512 \
    --target_points 96 \
    --batch_size 64 \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_encoder 0 \
    --freeze_transformer 0 \
    --n_epochs 50 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --revin 1 \
    --model_id 1
