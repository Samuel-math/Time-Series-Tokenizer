#!/bin/bash

# Patch VQVAE Transformer 预训练脚本
# Transformer输入维度 = embedding_dim * (patch_size / compression_factor)
# 例: 64 * (16 / 4) = 256

# 残差量化参数（减少量化误差）
USE_RESIDUAL_VQ=0  # 是否使用残差量化（1启用，0禁用）
RESIDUAL_VQ_LAYERS=2  # 残差量化层数（建议2-3层）
RESIDUAL_VQ_COMBINE_METHOD="sum"  # 合并方式：sum（相加）或concat（拼接）
VQ_INIT_METHOD="uniform"  # 码本初始化方法: uniform/normal/xavier/kaiming

python patch_vqvae_pretrain.py \
    --dset ettm1 \
    --context_points 1024 \
    --batch_size 64 \
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
    --use_residual_vq ${USE_RESIDUAL_VQ} \
    --residual_vq_layers ${RESIDUAL_VQ_LAYERS} \
    --residual_vq_combine_method ${RESIDUAL_VQ_COMBINE_METHOD} \
    --vq_init_method ${VQ_INIT_METHOD} \
    --n_epochs 100 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --revin 1 \
    --vq_weight 0.5 \
    --recon_weight 0.1 \
    --model_id 1
