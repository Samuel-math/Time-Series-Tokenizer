#!/bin/bash

# Patch VQVAE Transformer 微调脚本

# 预训练模型路径 (需要先运行 patch_vqvae_pretrain.sh)
PRETRAINED_MODEL="saved_models/patch_vqvae/ettm1/patch_vqvae_ps16_cb14_cd256_l3_model1.pth"

# Gumbel-Softmax参数（微调阶段的码本查找）
USE_GUMBEL_SOFTMAX=1           # 是否使用Gumbel-Softmax（1启用，0使用普通Softmax）
GUMBEL_TEMPERATURE=1.0         # Gumbel-Softmax温度（越小越接近argmax，建议0.5-2.0）
GUMBEL_HARD=0                  # 是否使用Straight-Through（前向硬采样，反向软梯度）

python patch_vqvae_finetune.py \
    --dset ettm1 \
    --context_points 512 \
    --target_points 96 \
    --batch_size 64 \
    --pretrained_model $PRETRAINED_MODEL \
    --n_epochs 50 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --revin 1 \
    --use_gumbel_softmax $USE_GUMBEL_SOFTMAX \
    --gumbel_temperature $GUMBEL_TEMPERATURE \
    --gumbel_hard $GUMBEL_HARD \
    --model_id 1
