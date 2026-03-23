#!/bin/bash

# 码本预训练脚本（带序列间对比学习损失）
# 用于在decoder-only预训练之前先训练好encoder、codebook和decoder
# 
# 新增功能：
# - Batch 内序列间对比学习 Loss（基于MSE距离）
# - 如果原始序列之间的MSE距离小于阈值，标记为相似
# - 对量化后的序列进行对比学习，让相似序列对的量化距离也小，不相似序列对的量化距离也大
# - 相似度的衡量始终使用MSE范式

# ============ 数据集参数 ============
DSET="ettm1"
CONTEXT_POINTS=512
BATCH_SIZE=64
NUM_WORKERS=0
SCALER="standard"
FEATURES="M"

# ============ 模型参数 ============
PATCH_SIZE=16
EMBEDDING_DIM=32
CODEBOOK_SIZE=256
COMPRESSION_FACTOR=4
NUM_HIDDENS=64
NUM_RESIDUAL_LAYERS=2
NUM_RESIDUAL_HIDDENS=32
COMMITMENT_COST=0.25
CODEBOOK_EMA=1  # 使用EMA
EMA_DECAY=0.99
EMA_EPS=1e-5

# ============ 码本初始化参数 ============
VQ_INIT_METHOD="random"
CODEBOOK_REPORT_INTERVAL=5
SEED=42

# ============ 训练参数 ============
N_EPOCHS=100
LR=1e-4
WEIGHT_DECAY=1e-4
REVIN=1
AMP=1
VQ_WEIGHT=1.0
RECON_WEIGHT=1.0

# ============ 对比学习损失参数 ============
INTER_WEIGHT=0.1                  # 对比学习损失权重
SIMILARITY_THRESHOLD=0.5         # MSE距离阈值（低于此阈值的样本对视为相似）
INTER_LOSS_TYPE="mse"             # 损失类型: "mse"（对齐距离矩阵）或 "contrastive"（对比学习）
INTER_TEMPERATURE=0.1            # 对比学习温度系数（仅用于contrastive模式）

# ============ 对比学习损失延迟参数 ============
INTER_DELAY_EPOCHS=5             # 前N个epoch只使用intra_loss，之后直接加入inter_loss

# ============ 数据采样参数 ============
TRAIN_SAMPLE_RATIO=1.0
VALID_SAMPLE_RATIO=1.0

# ============ 保存参数 ============
SAVE_PATH="saved_models/vqvae_only_inter/"
MODEL_ID=1

echo "=============================================="
echo "码本预训练（带序列间对比学习损失）"
echo "=============================================="
echo "数据集: $DSET"
echo "Patch大小: $PATCH_SIZE"
echo "码本大小: $CODEBOOK_SIZE"
echo "对比学习损失权重: $INTER_WEIGHT"
echo "延迟epochs: $INTER_DELAY_EPOCHS (前${INTER_DELAY_EPOCHS}个epoch只使用intra_loss，之后直接加入inter_loss)"
echo "对比学习损失类型: $INTER_LOSS_TYPE"
echo "相似度阈值（MSE距离）: $SIMILARITY_THRESHOLD"
echo "=============================================="

python codebook_pretrain_freq.py \
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
    --vq_init_method $VQ_INIT_METHOD \
    --codebook_report_interval $CODEBOOK_REPORT_INTERVAL \
    --seed $SEED \
    --n_epochs $N_EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --revin $REVIN \
    --amp $AMP \
    --vq_weight $VQ_WEIGHT \
    --recon_weight $RECON_WEIGHT \
    --inter_weight $INTER_WEIGHT \
    --similarity_threshold $SIMILARITY_THRESHOLD \
    --inter_loss_type $INTER_LOSS_TYPE \
    --inter_temperature $INTER_TEMPERATURE \
    --inter_delay_epochs $INTER_DELAY_EPOCHS \
    --train_sample_ratio $TRAIN_SAMPLE_RATIO \
    --valid_sample_ratio $VALID_SAMPLE_RATIO \
    --save_path $SAVE_PATH \
    --model_id $MODEL_ID

echo "训练完成！"
