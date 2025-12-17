#!/bin/bash

# =====================================================
# MLM码本训练脚本
# 包含两个阶段：
# 1. MLM预训练：训练VQ_encoder（PatchEmbedding）
# 2. 构建码本：通过聚类生成codebook
# =====================================================

# =====================================================
# 配置参数
# =====================================================

DSET="ettm1"
MODEL_ID=1

# ----- Patch 参数 -----
PATCH_SIZE=16

# ----- MLM模型参数 -----
D_MODEL=128
N_HEADS=8
N_LAYERS=4
D_FF=256
DROPOUT=0.1
MASK_RATIO=0.4

# ----- Codebook参数 -----
CODEBOOK_SIZE=256
CLUSTER_METHOD="minibatch_kmeans"  # 'kmeans' 或 'minibatch_kmeans'
MAX_SAMPLES=100000  # 用于聚类的最大样本数

# ----- 训练参数 -----
CONTEXT_POINTS=512
BATCH_SIZE=128
N_EPOCHS=100
LR=1e-3
WEIGHT_DECAY=1e-4
REVIN=1
AMP=1

# ----- 数据采样参数（用于加速大数据集训练）-----
TRAIN_SAMPLE_RATIO=1.0  # 训练集采样比例 (0.0-1.0)
VALID_SAMPLE_RATIO=1.0  # 验证集采样比例 (0.0-1.0)

# ----- 保存路径 -----
SAVE_PATH="saved_models/codebook"

# ----- 其他参数 -----
SEED=42

# =====================================================
# 阶段 1: MLM预训练
# =====================================================
echo ""
echo "================================================="
echo "阶段 1: MLM预训练"
echo "================================================="
echo "数据集: ${DSET}"
echo "Patch Size: ${PATCH_SIZE}"
echo "Model Dim: ${D_MODEL}"
echo "Layers: ${N_LAYERS}"
echo "Mask Ratio: ${MASK_RATIO}"
echo "Epochs: ${N_EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "================================================="

python mlm_pretrain.py \
    --dset ${DSET} \
    --context_points ${CONTEXT_POINTS} \
    --batch_size ${BATCH_SIZE} \
    --patch_size ${PATCH_SIZE} \
    --d_model ${D_MODEL} \
    --n_heads ${N_HEADS} \
    --n_layers ${N_LAYERS} \
    --d_ff ${D_FF} \
    --dropout ${DROPOUT} \
    --mask_ratio ${MASK_RATIO} \
    --codebook_size ${CODEBOOK_SIZE} \
    --n_epochs ${N_EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --revin ${REVIN} \
    --amp ${AMP} \
    --train_sample_ratio ${TRAIN_SAMPLE_RATIO} \
    --valid_sample_ratio ${VALID_SAMPLE_RATIO} \
    --save_path ${SAVE_PATH} \
    --model_id ${MODEL_ID} \
    --seed ${SEED}

# 检查MLM预训练是否成功
MLM_MODEL="${SAVE_PATH}/${DSET}/mlm_ps${PATCH_SIZE}_dm${D_MODEL}_l${N_LAYERS}_mask${MASK_RATIO}_model${MODEL_ID}.pth"
if [ ! -f "${MLM_MODEL}" ]; then
    echo "错误: MLM预训练模型未找到: ${MLM_MODEL}"
    exit 1
fi

echo ""
echo "✓ MLM预训练完成"
echo "模型保存至: ${MLM_MODEL}"

# =====================================================
# 阶段 2: 构建码本
# =====================================================
echo ""
echo "================================================="
echo "阶段 2: 构建码本"
echo "================================================="
echo "MLM模型: ${MLM_MODEL}"
echo "Codebook Size: ${CODEBOOK_SIZE}"
echo "聚类方法: ${CLUSTER_METHOD}"
echo "最大样本数: ${MAX_SAMPLES}"
echo "================================================="

python mlm_build_codebook.py \
    --dset ${DSET} \
    --context_points ${CONTEXT_POINTS} \
    --batch_size ${BATCH_SIZE} \
    --mlm_model "${MLM_MODEL}" \
    --codebook_size ${CODEBOOK_SIZE} \
    --cluster_method ${CLUSTER_METHOD} \
    --max_samples ${MAX_SAMPLES} \
    --train_sample_ratio ${TRAIN_SAMPLE_RATIO} \
    --save_path ${SAVE_PATH} \
    --revin ${REVIN}

# 检查码本是否成功生成
CODEBOOK_NAME="mlm_ps${PATCH_SIZE}_dm${D_MODEL}_l${N_LAYERS}_mask${MASK_RATIO}_model${MODEL_ID}_codebook_cb${CODEBOOK_SIZE}.pth"
CODEBOOK_PATH="${SAVE_PATH}/${DSET}/${CODEBOOK_NAME}"
if [ ! -f "${CODEBOOK_PATH}" ]; then
    echo "错误: 码本文件未找到: ${CODEBOOK_PATH}"
    exit 1
fi

echo ""
echo "================================================="
echo "全部完成！"
echo "================================================="
echo "MLM模型: ${MLM_MODEL}"
echo "码本文件: ${CODEBOOK_PATH}"
echo ""
echo "下一步: 使用码本文件进行预训练"
echo "  python ../decoder-only/patch_vq_pretrain.py \\"
echo "      --mlm_codebook_path ${CODEBOOK_PATH} \\"
echo "      ..."
echo "================================================="
