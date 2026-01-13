#!/bin/bash

# 码本预训练脚本（带序列间频域一致性损失）
# 用于在decoder-only预训练之前先训练好encoder、codebook和decoder
# 
# 新增功能：
# - Batch 内序列间频域一致性 Loss
# - 确保频率相似的原始序列，其量化编码在周期性上也相似

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

# ============ 频域一致性损失参数 ============
FREQ_WEIGHT=0.1                  # 频域一致性损失最终权重
FREQ_SIMILARITY_THRESHOLD=0.8    # 相似度阈值（高于此阈值的样本对视为正样本）
FREQ_LOSS_TYPE="mse"             # 损失类型: "mse" 或 "infonce"
FREQ_TEMPERATURE=0.1             # InfoNCE温度系数

# ============ 频域损失延迟和Warmup参数 ============
FREQ_DELAY_EPOCHS=20             # 前N个epoch完全禁用freq_loss（权重=0）
FREQ_WARMUP_EPOCHS=10            # 延迟后，Warmup的epoch数
FREQ_WEIGHT_START=0.01           # Warmup起始权重（从小到大逐渐增加到FREQ_WEIGHT）

# ============ 数据采样参数 ============
TRAIN_SAMPLE_RATIO=1.0
VALID_SAMPLE_RATIO=1.0

# ============ 保存参数 ============
SAVE_PATH="saved_models/vqvae_only_freq/"
MODEL_ID=1

echo "=============================================="
echo "码本预训练（带序列间频域一致性损失）"
echo "=============================================="
echo "数据集: $DSET"
echo "Patch大小: $PATCH_SIZE"
echo "码本大小: $CODEBOOK_SIZE"
echo "频域损失权重: $FREQ_WEIGHT (delay: ${FREQ_DELAY_EPOCHS} epochs, warmup: $FREQ_WEIGHT_START -> $FREQ_WEIGHT, ${FREQ_WARMUP_EPOCHS} epochs)"
echo "频域损失类型: $FREQ_LOSS_TYPE"
echo "相似度阈值: $FREQ_SIMILARITY_THRESHOLD"
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
    --freq_weight $FREQ_WEIGHT \
    --freq_similarity_threshold $FREQ_SIMILARITY_THRESHOLD \
    --freq_loss_type $FREQ_LOSS_TYPE \
    --freq_temperature $FREQ_TEMPERATURE \
    --freq_delay_epochs $FREQ_DELAY_EPOCHS \
    --freq_warmup_epochs $FREQ_WARMUP_EPOCHS \
    --freq_weight_start $FREQ_WEIGHT_START \
    --train_sample_ratio $TRAIN_SAMPLE_RATIO \
    --valid_sample_ratio $VALID_SAMPLE_RATIO \
    --save_path $SAVE_PATH \
    --model_id $MODEL_ID

echo "训练完成！"

