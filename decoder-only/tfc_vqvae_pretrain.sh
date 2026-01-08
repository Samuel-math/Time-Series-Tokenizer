#!/bin/bash
# TF-C VQVAE 预训练脚本
# 使用时频一致性对比学习增强码本的语义表达能力
# 包含熵正则化损失提高码本利用率

# 设置默认参数
DATASET=${1:-"ettm1"}
PATCH_SIZE=${2:-16}
CODEBOOK_SIZE=${3:-256}
TEMPERATURE=${4:-0.07}
GAMMA=${5:-0.5}
ENTROPY_WEIGHT=${6:-0.1}

echo "=============================================="
echo "TF-C VQVAE 预训练"
echo "=============================================="
echo "数据集: $DATASET"
echo "Patch大小: $PATCH_SIZE"
echo "码本大小: $CODEBOOK_SIZE"
echo "对比损失温度系数: $TEMPERATURE"
echo "对比损失权重 (γ): $GAMMA"
echo "熵正则化权重: $ENTROPY_WEIGHT"
echo "=============================================="

python tfc_vqvae_pretrain.py \
    --dset $DATASET \
    --context_points 512 \
    --target_points 96 \
    --batch_size 64 \
    --patch_size $PATCH_SIZE \
    --embedding_dim 32 \
    --compression_factor 4 \
    --codebook_size $CODEBOOK_SIZE \
    --commitment_cost 0.25 \
    --codebook_ema 1 \
    --ema_decay 0.99 \
    --num_hiddens 64 \
    --num_residual_layers 2 \
    --num_residual_hiddens 32 \
    --n_layers 4 \
    --n_heads 4 \
    --d_ff 256 \
    --dropout 0.1 \
    --freq_encoder_type mlp \
    --freq_encoder_hidden 256 \
    --proj_hidden_dim 256 \
    --proj_output_dim 128 \
    --temperature $TEMPERATURE \
    --alpha 1.0 \
    --beta 1.0 \
    --gamma $GAMMA \
    --entropy_weight $ENTROPY_WEIGHT \
    --entropy_temperature 1.0 \
    --n_epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --revin 1 \
    --scheduler cosine \
    --early_stop_patience 15 \
    --save_path saved_models/tfc_vqvae/ \
    --model_id 1

echo "训练完成！"

