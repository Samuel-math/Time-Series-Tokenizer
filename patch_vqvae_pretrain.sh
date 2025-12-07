#!/bin/bash

# Patch VQVAE Transformer 预训练脚本
# Transformer输入维度 = embedding_dim * (patch_size / compression_factor)
# 例: 32 * (16 / 4) = 128

python patch_vqvae_pretrain.py \
    --dset ettm1 \
    --context_points 512 \
    --batch_size 128 \
    --patch_size 16 \
    --embedding_dim 32 \
    --compression_factor 4 \
    --codebook_size 256 \
    --n_layers 4 \
    --n_heads 4 \
    --d_ff 256 \
    --dropout 0.1 \
    --num_hiddens 64 \
    --num_residual_layers 2 \
    --num_residual_hiddens 32 \
    --n_epochs 50 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --revin 1 \
    --vq_weight 0.5 \
    --recon_weight 0.1 \
    --model_id 1
