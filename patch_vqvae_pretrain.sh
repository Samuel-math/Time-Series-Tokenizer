#!/bin/bash

# Patch VQVAE Transformer 预训练脚本
# 复用 vqvae.py 中的 Encoder 和 Decoder

python patch_vqvae_pretrain.py \
    --dset ettm1 \
    --context_points 512 \
    --batch_size 64 \
    --patch_size 16 \
    --embedding_dim 64 \
    --compression_factor 4 \
    --codebook_size 512 \
    --d_model 256 \
    --n_layers 6 \
    --n_heads 8 \
    --d_ff 1024 \
    --dropout 0.1 \
    --num_hiddens 128 \
    --num_residual_layers 2 \
    --num_residual_hiddens 64 \
    --n_epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --revin 1 \
    --vq_weight 1.0 \
    --recon_weight 0.1 \
    --model_id 1
    # --vqvae_checkpoint "path/to/vqvae_model.pth"  # 可选：加载预训练VQVAE
