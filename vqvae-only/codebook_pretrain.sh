#!/bin/bash

# 码本预训练脚本
# 用于在decoder-only预训练之前先训练好encoder、codebook和decoder

# 数据集参数
DSET="etth1"
CONTEXT_POINTS=512
BATCH_SIZE=64
NUM_WORKERS=0
SCALER="standard"
FEATURES="M"

# 模型参数（与PatchVQVAETransformer一致）
PATCH_SIZE=8
EMBEDDING_DIM=64
CODEBOOK_SIZE=256
COMPRESSION_FACTOR=8
NUM_HIDDENS=64
NUM_RESIDUAL_LAYERS=2
NUM_RESIDUAL_HIDDENS=32
COMMITMENT_COST=0.25
CODEBOOK_EMA=1  # 使用EMA
EMA_DECAY=0.99
EMA_EPS=1e-5

# 码本初始化参数
CODEBOOK_REPORT_INTERVAL=5  # 码本利用率报告间隔（每N个epoch报告一次）
SEED=42  # 随机数种子（用于训练可复现性，但不影响码本初始化）

# 训练参数
N_EPOCHS=100
LR=1e-4
WEIGHT_DECAY=1e-4
REVIN=1
AMP=1
VQ_WEIGHT=1.0
RECON_WEIGHT=1.0

# 数据采样参数（用于加速大数据集训练）
TRAIN_SAMPLE_RATIO=1.0  # 训练集采样比例 (0.0-1.0)，例如0.1表示只使用10%的训练数据
VALID_SAMPLE_RATIO=1.0  # 验证集采样比例 (0.0-1.0)，例如0.1表示只使用10%的验证数据

# 保存参数
SAVE_PATH="saved_models/vqvae_only/"
MODEL_ID=1

# Per-channel 码本（每通道独立 VQ）
PER_CHANNEL_CODEBOOK=0  # 0=共享码本, 1=per-channel独立码本

# RVQ 层数
# 1 = 普通 VQ（默认，与旧行为兼容）
# 2 = 2层残差 VQ（RQVAE）
N_RQ_LAYERS=1

# Robust VQVAE: 稀疏分量
# SPARSE_WEIGHT=0 表示不启用 Robust 分解（标准 VQVAE 行为）
# 建议初始值: 0.01；调大可让码本更专注干净主体结构，调小则减弱稀疏约束
SPARSE_WEIGHT=0.0
# SparseNet 输出 tanh 振幅上界（防止异常分量学走主体结构）
# 建议范围: 0.2 ~ 1.0（相对归一化后的 patch 值域）
SPARSE_AMPLITUDE=0.5

# ===== 趋势-残差双码本分解（TrendResidualCodebookModel）=====
# 1 启用：X_patch = X_trend + X_residual（低通滤波）
#          - X_trend    用 Codebook_T 重构
#          - X_residual 用 Codebook_R + SparseNet(s) 重构
#          - 两套独立的 Encoder/Decoder/VQ
# 0 禁用：使用标准 CodebookModel / PerChannelCodebookModel
USE_TREND_DECOMP=1
# 低通滤波（移动平均）核长度：越大趋势越平滑，建议 3~9；patch_size 较小可取 3
TREND_KERNEL_SIZE=5
# 低通滤波核是否可学习
#   1 = 可学习：权重经 softmax 归一化为非负且和为 1，起点等价于均值核（推荐）
#   0 = 固定移动平均（无参数）
TREND_LEARNABLE_FILTER=1
# 两个码本的大小（设为 0 则各自使用 CODEBOOK_SIZE）
CODEBOOK_SIZE_TREND=0
CODEBOOK_SIZE_RES=0
# 两个重构损失的相对权重
TREND_RECON_WEIGHT=1.0
RES_RECON_WEIGHT=1.0

python codebook_pretrain.py \
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
    --codebook_report_interval $CODEBOOK_REPORT_INTERVAL \
    --seed $SEED \
    --n_epochs $N_EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --revin $REVIN \
    --amp $AMP \
    --vq_weight $VQ_WEIGHT \
    --recon_weight $RECON_WEIGHT \
    --train_sample_ratio $TRAIN_SAMPLE_RATIO \
    --valid_sample_ratio $VALID_SAMPLE_RATIO \
    --save_path $SAVE_PATH \
    --model_id $MODEL_ID \
    --per_channel_codebook $PER_CHANNEL_CODEBOOK \
    --n_rq_layers $N_RQ_LAYERS \
    --sparse_weight $SPARSE_WEIGHT \
    --sparse_amplitude $SPARSE_AMPLITUDE \
    --use_trend_decomp $USE_TREND_DECOMP \
    --trend_kernel_size $TREND_KERNEL_SIZE \
    --trend_learnable_filter $TREND_LEARNABLE_FILTER \
    --codebook_size_trend $CODEBOOK_SIZE_TREND \
    --codebook_size_res $CODEBOOK_SIZE_RES \
    --trend_recon_weight $TREND_RECON_WEIGHT \
    --res_recon_weight $RES_RECON_WEIGHT
