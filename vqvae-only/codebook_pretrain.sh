#!/bin/bash

# 码本预训练脚本
# 用于在decoder-only预训练之前先训练好encoder、codebook和decoder

# 数据集参数
DSET="ettm1"
CONTEXT_POINTS=512
BATCH_SIZE=64
NUM_WORKERS=0
SCALER="standard"
FEATURES="M"

# 模型参数（与PatchVQVAETransformer一致）
PATCH_SIZE=16
EMBEDDING_DIM=32
CODEBOOK_SIZE=256
COMPRESSION_FACTOR=4
NUM_HIDDENS=64
NUM_RESIDUAL_LAYERS=2
NUM_RESIDUAL_HIDDENS=32
COMMITMENT_COST=0.25
CODEBOOK_EMA=1  # 不使用EMA
EMA_DECAY=0.99
EMA_EPS=1e-5

# 训练参数
N_EPOCHS=50
LR=1e-4
WEIGHT_DECAY=1e-4
REVIN=1
AMP=1
VQ_WEIGHT=1.0
RECON_WEIGHT=1.0

# 保存参数
SAVE_PATH="saved_models/vqvae_only/"
MODEL_ID=1

python vqvae-only/codebook_pretrain.py \
    --dset $DSET \
    --context_points $CONTEXT_POINTS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --scaler $SCALER \
    --features $FEATURES \
    --patch_size $PATCH_SIZE \
    --embedding_dim $EMBEDDING_DIM \
    --codebook_size $CODEBOOK_SIZE \
    --compression_factor $COMPRESSION_FACTOR \
    --num_hiddens $NUM_HIDDENS \
    --num_residual_layers $NUM_RESIDUAL_LAYERS \
    --num_residual_hiddens $NUM_RESIDUAL_HIDDENS \
    --commitment_cost $COMMITMENT_COST \
    --codebook_ema $CODEBOOK_EMA \
    --ema_decay $EMA_DECAY \
    --ema_eps $EMA_EPS \
    --n_epochs $N_EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --revin $REVIN \
    --amp $AMP \
    --vq_weight $VQ_WEIGHT \
    --recon_weight $RECON_WEIGHT \
    --save_path $SAVE_PATH \
    --model_id $MODEL_ID
