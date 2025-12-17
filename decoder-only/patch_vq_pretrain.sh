#!/bin/bash

# Patch VQ Transformer 预训练脚本
# 使用MLM训练的VQ_encoder和codebook

# MLM codebook路径（必需）
MLM_CODEBOOK_PATH="../codebook/saved_models/codebook/ettm1/mlm_ps16_dm128_l4_mask0.4_model1_codebook_cb256.pth"

python patch_vq_pretrain.py \
    --dset ettm1 \
    --context_points 512 \
    --target_points 96 \
    --batch_size 64 \
    --patch_size 16 \
    --codebook_size 256 \
    --n_layers 4 \
    --n_heads 4 \
    --d_ff 256 \
    --dropout 0.1 \
    --mlm_codebook_path "${MLM_CODEBOOK_PATH}" \
    --load_mlm_vq_encoder 1 \
    --load_mlm_codebook 1 \
    --freeze_mlm_components 1 \
    --n_epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --revin 1 \
    --vq_weight 1.0 \
    --recon_weight 0.1 \
    --model_id 1
