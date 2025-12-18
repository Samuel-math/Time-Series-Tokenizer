#!/bin/bash

# =====================================================
# Patch VQVAE Transformer 批量训练脚本
# 批量运行多个 input_size 和 target_size 组合的预训练和微调
# =====================================================

# =====================================================
# 配置参数
# =====================================================

DSET="ettm1"
MODEL_ID=1

# ----- 码本模型路径（可选）-----
# 如果为空，脚本会自动查找，或手动指定完整路径
# VQVAE_CHECKPOINT=""  # 留空表示不使用预训练VQVAE
VQVAE_CHECKPOINT="../vqvae-only/saved_models/vqvae_only/${DSET}/codebook_ps16_cb256_cd128_model1.pth"

# ----- Patch 参数 -----
PATCH_SIZE=16
EMBEDDING_DIM=64
COMPRESSION_FACTOR=8
CODEBOOK_SIZE=256

# ----- VQVAE Encoder/Decoder 参数 -----
NUM_HIDDENS=64
NUM_RESIDUAL_LAYERS=2
NUM_RESIDUAL_HIDDENS=32
COMMITMENT_COST=0.25
CODEBOOK_EMA=1
EMA_DECAY=0.99
EMA_EPS=1e-5

# ----- Transformer 参数 -----
N_LAYERS=4
N_HEADS=4
D_FF=256
DROPOUT=0.3

# ----- Channel Attention 参数 -----
USE_CHANNEL_ATTENTION=1
CHANNEL_ATTENTION_DROPOUT=0.1

# ----- 批量训练配置 -----
# 定义多个 input_size (context_points) 和 target_size (target_points) 组合
# 格式: "input_size:target_size"
# 预训练时使用这些组合，微调时使用 TARGET_POINTS_LIST
INPUT_TARGET_PAIRS=(
    # 96 作为 input_size
    "96:96"
    "96:128"
    "96:256"
    "96:336"
    "96:512"
    "96:720"
    "96:1024"
    
    # 128 作为 input_size
    "128:96"
    "128:128"
    "128:256"
    "128:336"
    "128:512"
    "128:720"
    "128:1024"
    
    # 256 作为 input_size
    "256:96"
    "256:128"
    "256:256"
    "256:336"
    "256:512"
    "256:720"
    "256:1024"
    
    # 336 作为 input_size
    "336:96"
    "336:128"
    "336:256"
    "336:336"
    "336:512"
    "336:720"
    "336:1024"
    
    # 512 作为 input_size
    "512:96"
    "512:128"
    "512:256"
    "512:336"
    "512:512"
    "512:720"
    "512:1024"
    
    # 720 作为 input_size
    "720:96"
    "720:128"
    "720:256"
    "720:336"
    "720:512"
    "720:720"
    "720:1024"
    
    # 1024 作为 input_size
    "1024:96"
    "1024:128"
    "1024:256"
    "1024:336"
    "1024:512"
    "1024:720"
    "1024:1024"
)

# 微调时的 target_points 列表（对每个预训练模型都会运行这些微调）
TARGET_POINTS_LIST=(96 192 336 720)

# ----- 预训练参数 -----
PRETRAIN_EPOCHS=100
PRETRAIN_BATCH_SIZE=64
PRETRAIN_LR=3e-4
VQ_WEIGHT=0.5
RECON_WEIGHT=0.1

# ----- 微调参数 -----
FINETUNE_EPOCHS=50
FINETUNE_BATCH_SIZE=64
FINETUNE_LR=1e-4

# ----- 其他参数 -----
REVIN=1
WEIGHT_DECAY=1e-4
FREEZE_VQVAE=1
LOAD_VQ_WEIGHTS=1

# =====================================================
# 自动查找码本模型（如果未指定）
# =====================================================

if [ -z "${VQVAE_CHECKPOINT}" ]; then
    echo "未指定VQVAE checkpoint，将从头训练"
else
    # 处理相对路径
    if [[ ! "${VQVAE_CHECKPOINT}" = /* ]]; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        VQVAE_CHECKPOINT="${SCRIPT_DIR}/${VQVAE_CHECKPOINT}"
    fi
    
    # 规范化路径
    VQVAE_CHECKPOINT=$(readlink -f "${VQVAE_CHECKPOINT}" 2>/dev/null || realpath "${VQVAE_CHECKPOINT}" 2>/dev/null || echo "${VQVAE_CHECKPOINT}")
    
    if [ ! -f "${VQVAE_CHECKPOINT}" ]; then
        echo "警告: VQVAE模型文件不存在: ${VQVAE_CHECKPOINT}"
        echo "将从头训练"
        VQVAE_CHECKPOINT=""
    else
        echo "找到VQVAE模型: ${VQVAE_CHECKPOINT}"
    fi
fi

# =====================================================
# 计算 code_dim 用于模型命名
# =====================================================
CODE_DIM=$((EMBEDDING_DIM * PATCH_SIZE / COMPRESSION_FACTOR))

echo "================================================="
echo "Patch VQVAE Transformer 批量训练"
echo "================================================="
echo "数据集: ${DSET}"
echo "模型ID: ${MODEL_ID}"
echo "Code Dim: ${CODE_DIM}"
echo "输入-目标组合数: ${#INPUT_TARGET_PAIRS[@]}"
echo "微调目标长度数: ${#TARGET_POINTS_LIST[@]}"
echo "总任务数: $(( ${#INPUT_TARGET_PAIRS[@]} * (1 + ${#TARGET_POINTS_LIST[@]}) ))"
echo "================================================="

# =====================================================
# 批量训练循环
# =====================================================

TOTAL_TASKS=${#INPUT_TARGET_PAIRS[@]}
CURRENT_TASK=0

for PAIR in "${INPUT_TARGET_PAIRS[@]}"; do
    CURRENT_TASK=$((CURRENT_TASK + 1))
    
    # 解析 input_size 和 target_size
    IFS=':' read -r INPUT_SIZE TARGET_SIZE <<< "${PAIR}"
    
    echo ""
    echo "================================================="
    echo "任务 ${CURRENT_TASK}/${TOTAL_TASKS}: Input=${INPUT_SIZE}, Target=${TARGET_SIZE}"
    echo "================================================="
    
    # 构建模型名称（包含 input_size 和 target_size）
    MODEL_NAME="patch_vqvae_ps${PATCH_SIZE}_cb${CODEBOOK_SIZE}_cd${CODE_DIM}_l${N_LAYERS}_in${INPUT_SIZE}_tg${TARGET_SIZE}_model${MODEL_ID}"
    
    # =====================================================
    # 阶段 1: 预训练
    # =====================================================
    echo ""
    echo "-------------------------------------------------"
    echo "阶段 1: 预训练"
    echo "-------------------------------------------------"
    echo "Context Points: ${INPUT_SIZE}"
    echo "Target Points: ${TARGET_SIZE}"
    echo "Epochs: ${PRETRAIN_EPOCHS}"
    echo "Batch Size: ${PRETRAIN_BATCH_SIZE}"
    echo "-------------------------------------------------"
    
    # 构建预训练命令参数
    PRETRAIN_ARGS=(
        --dset ${DSET}
        --context_points ${INPUT_SIZE}
        --target_points ${TARGET_SIZE}
        --batch_size ${PRETRAIN_BATCH_SIZE}
        --patch_size ${PATCH_SIZE}
        --embedding_dim ${EMBEDDING_DIM}
        --compression_factor ${COMPRESSION_FACTOR}
        --codebook_size ${CODEBOOK_SIZE}
        --n_layers ${N_LAYERS}
        --n_heads ${N_HEADS}
        --d_ff ${D_FF}
        --dropout ${DROPOUT}
        --num_hiddens ${NUM_HIDDENS}
        --num_residual_layers ${NUM_RESIDUAL_LAYERS}
        --num_residual_hiddens ${NUM_RESIDUAL_HIDDENS}
        --commitment_cost ${COMMITMENT_COST}
        --codebook_ema ${CODEBOOK_EMA}
        --ema_decay ${EMA_DECAY}
        --ema_eps ${EMA_EPS}
        --use_channel_attention ${USE_CHANNEL_ATTENTION}
        --channel_attention_dropout ${CHANNEL_ATTENTION_DROPOUT}
        --n_epochs ${PRETRAIN_EPOCHS}
        --lr ${PRETRAIN_LR}
        --weight_decay ${WEIGHT_DECAY}
        --revin ${REVIN}
        --vq_weight ${VQ_WEIGHT}
        --recon_weight ${RECON_WEIGHT}
        --model_id ${MODEL_ID}
    )
    
    # 如果指定了VQVAE checkpoint，添加相关参数
    if [ -n "${VQVAE_CHECKPOINT}" ]; then
        PRETRAIN_ARGS+=(
            --vqvae_checkpoint "${VQVAE_CHECKPOINT}"
            --freeze_vqvae ${FREEZE_VQVAE}
            --load_vq_weights ${LOAD_VQ_WEIGHTS}
        )
    fi
    
    # 运行预训练
    python patch_vqvae_pretrain.py "${PRETRAIN_ARGS[@]}"
    
    if [ $? -ne 0 ]; then
        echo "错误: 预训练失败 (Input=${INPUT_SIZE}, Target=${TARGET_SIZE})"
        echo "跳过该组合的微调任务"
        continue
    fi
    
    # =====================================================
    # 阶段 2: 微调（多个预测长度）
    # =====================================================
    PRETRAINED_MODEL="saved_models/patch_vqvae/${DSET}/${MODEL_NAME}.pth"
    
    # 检查预训练模型是否存在
    if [ ! -f "${PRETRAINED_MODEL}" ]; then
        echo "警告: 预训练模型不存在: ${PRETRAINED_MODEL}"
        echo "跳过微调任务"
        continue
    fi
    
    echo ""
    echo "-------------------------------------------------"
    echo "阶段 2: 微调"
    echo "-------------------------------------------------"
    echo "预训练模型: ${PRETRAINED_MODEL}"
    echo "Context Points: ${INPUT_SIZE}"
    echo "Target Points: ${TARGET_POINTS_LIST[@]}"
    echo "-------------------------------------------------"
    
    for FINETUNE_TARGET in ${TARGET_POINTS_LIST[@]}; do
        echo ""
        echo "  └─ 微调: Target Points = ${FINETUNE_TARGET}"
        
        python patch_vqvae_finetune.py \
            --dset ${DSET} \
            --context_points ${INPUT_SIZE} \
            --target_points ${FINETUNE_TARGET} \
            --batch_size ${FINETUNE_BATCH_SIZE} \
            --pretrained_model "${PRETRAINED_MODEL}" \
            --n_epochs ${FINETUNE_EPOCHS} \
            --lr ${FINETUNE_LR} \
            --weight_decay ${WEIGHT_DECAY} \
            --revin ${REVIN} \
            --model_id ${MODEL_ID}
        
        if [ $? -ne 0 ]; then
            echo "    警告: 微调失败 (Target=${FINETUNE_TARGET})"
        fi
    done
    
    echo ""
    echo "✓ 完成: Input=${INPUT_SIZE}, Target=${TARGET_SIZE}"
done

echo ""
echo "================================================="
echo "全部批量训练完成！"
echo "================================================="
echo "预训练模型保存在: saved_models/patch_vqvae/${DSET}/"
echo "微调模型保存在: saved_models/patch_vqvae_finetune/${DSET}/"
echo ""
echo "训练统计:"
echo "  总组合数: ${TOTAL_TASKS}"
echo "  每个组合的微调任务数: ${#TARGET_POINTS_LIST[@]}"
echo "  总微调任务数: $(( ${TOTAL_TASKS} * ${#TARGET_POINTS_LIST[@]} ))"
echo "================================================="
