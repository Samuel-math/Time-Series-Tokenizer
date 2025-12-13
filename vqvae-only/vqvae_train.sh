#!/bin/bash

# VQVAE 训练脚本

# 数据集参数
DSET="ettm1"
CONTEXT_POINTS=512
BATCH_SIZE=64
NUM_WORKERS=0
SCALER="standard"
FEATURES="M"

# VQVAE 模型参数
PATCH_SIZE=16
EMBEDDING_DIM=32
NUM_EMBEDDINGS=256
COMPRESSION_FACTOR=4
BLOCK_HIDDEN_SIZE=64
NUM_RESIDUAL_LAYERS=2
RES_HIDDEN_SIZE=32
COMMITMENT_COST=0.25

# 训练参数
N_EPOCHS=100
LR=1e-4
WEIGHT_DECAY=1e-4
REVIN=1
AMP=1

# 保存参数
SAVE_PATH="saved_models/vqvae_only/"
MODEL_ID=1

# 运行训练
python vqvae_train.py \
    --dset ${DSET} \
    --context_points ${CONTEXT_POINTS} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --scaler ${SCALER} \
    --features ${FEATURES} \
    --patch_size ${PATCH_SIZE} \
    --embedding_dim ${EMBEDDING_DIM} \
    --num_embeddings ${NUM_EMBEDDINGS} \
    --compression_factor ${COMPRESSION_FACTOR} \
    --block_hidden_size ${BLOCK_HIDDEN_SIZE} \
    --num_residual_layers ${NUM_RESIDUAL_LAYERS} \
    --res_hidden_size ${RES_HIDDEN_SIZE} \
    --commitment_cost ${COMMITMENT_COST} \
    --n_epochs ${N_EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --revin ${REVIN} \
    --amp ${AMP} \
    --save_path ${SAVE_PATH} \
    --model_id ${MODEL_ID}
