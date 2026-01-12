#!/bin/bash

# Patch-based VQVAE + Transformer 模型评估脚本示例
# 用法: bash patch_vqvae_evaluate.sh <dataset> <model_path> <target_points>

DSET=${1:-"etth1"}
MODEL_PATH=${2:-"saved_models/patch_vqvae/${DSET}/patch_vqvae_ps16_cb256_cd64_l4_in512_tg96_ca0_model1.pth"}
TARGET_POINTS=${3:-96}

echo "=========================================="
echo "评估模型"
echo "=========================================="
echo "数据集: ${DSET}"
echo "模型路径: ${MODEL_PATH}"
echo "预测长度: ${TARGET_POINTS}"
echo "=========================================="

python patch_vqvae_evaluate.py \
    --dset ${DSET} \
    --context_points 512 \
    --target_points ${TARGET_POINTS} \
    --batch_size 64 \
    --num_workers 0 \
    --features M \
    --model_path ${MODEL_PATH} \
    --revin 1 \
    --amp 1

echo "=========================================="
echo "评估完成"
echo "=========================================="
