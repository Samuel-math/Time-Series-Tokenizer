#!/bin/bash

# Patch VQVAE Transformer 预训练脚本
# Transformer输入维度 = embedding_dim * (patch_size / compression_factor)
# 例: 64 * (16 / 4) = 256

python patch_vqvae_pretrain.py \
    --dset ettm1 \
    --context_points 512 \
    --batch_size 128 \
    --patch_size 16 \
    --embedding_dim 64 \
    --compression_factor 4 \
    --codebook_size 14 \
    --n_layers 3 \
    --n_heads 4 \
    --d_ff 256 \
    --dropout 0.3 \
    --num_hiddens 64 \
    --num_residual_layers 2 \
    --num_residual_hiddens 32 \
    --n_epochs 100 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --revin 1 \
    --vq_weight 0.5 \
    --recon_weight 0.1 \
    --model_id 1
