#!/bin/bash

# =====================================================
# 批量超参数搜索脚本
# 遍历多个超参数组合进行训练
# =====================================================

# 不使用 set -e，允许单个组合失败时继续执行

# =====================================================
# 基础配置参数
# =====================================================

DSET='etth1'
MODEL_ID=1

# ----- 码本模型路径（相对于decoder-only目录）-----
CODEBOOK_CHECKPOINT="../vqvae-only/saved_models/vqvae_only/${DSET}/codebook_ps8_cb256_cd32_model1.pth"

# ----- Patch 参数（会从码本checkpoint中自动读取，这里作为备用）-----
PATCH_SIZE=8
COMPRESSION_FACTOR=8

# ----- VQVAE 参数（会从码本checkpoint中自动读取，这里作为备用）-----
EMBEDDING_DIM=32
CODEBOOK_SIZE=256
NUM_HIDDENS=64
NUM_RESIDUAL_LAYERS=2

# ----- Transformer 参数（固定值）-----
DROPOUT=0.2
CODEBOOK_EMA=1
EMA_DECAY=0.99
EMA_EPS=1e-5
TRANSFORMER_HIDDEN_DIM=""

# ----- 预训练参数（固定值）-----
PRETRAIN_EPOCHS=100
PRETRAIN_LR=3e-4
VQ_WEIGHT=0.0
RECON_WEIGHT=0.0

# ----- 微调参数 -----
FINETUNE_CONTEXT_POINTS=192
FINETUNE_EPOCHS=50
FINETUNE_BATCH_SIZE=32
FINETUNE_LR=3e-4
TARGET_POINTS_LIST=(96 192 336 720)

# ----- Gumbel-Softmax参数 -----
USE_GUMBEL_SOFTMAX=1
GUMBEL_TEMPERATURE=1.3
GUMBEL_HARD=0

# ----- 自回归预测参数 -----
AR_STEP_SIZE=""

# ----- 其他参数 -----
REVIN=1
WEIGHT_DECAY=1e-4
FREEZE_VQVAE=1

# =====================================================
# 超参数搜索空间
# =====================================================

PRETRAIN_BATCH_SIZES=(64 128 256)
PROGRESSIVE_STEP_SIZES=(2 3 4 5 6 7 8)
CONTEXT_MULTIPLIERS=(4 8 12 16)
N_LAYERS_LIST=(1 2)
N_HEADS_LIST=(2 4 8 16)
D_FF_LIST=(64 128 256 512)  # FFN维度遍历
NUM_RESIDUAL_HIDDENS_LIST=(64)  # 固定为64，不进行遍历

# =====================================================
# 日志文件设置
# =====================================================

LOG_DIR="grid_search_logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/grid_search_${TIMESTAMP}.log"
SUMMARY_FILE="${LOG_DIR}/grid_search_summary_${TIMESTAMP}.txt"

# 同时输出到终端和日志文件
# 注意：如果系统不支持 stdbuf，可以移除 stdbuf 部分
if command -v stdbuf >/dev/null 2>&1; then
    exec > >(stdbuf -oL -eL tee -a "${LOG_FILE}")
else
    exec > >(tee -a "${LOG_FILE}")
fi
exec 2>&1

# =====================================================
# 初始化码本模型路径
# =====================================================

if [ -z "${CODEBOOK_CHECKPOINT}" ]; then
    echo "正在查找码本模型..."
    CODEBOOK_CHECKPOINT=$(find ../vqvae-only/saved_models/vqvae_only -name "codebook_*.pth" -type f 2>/dev/null | head -1)
    
    if [ -z "${CODEBOOK_CHECKPOINT}" ]; then
        echo "错误: 未找到码本模型！"
        exit 1
    fi
fi

# 处理相对路径
if [[ ! "${CODEBOOK_CHECKPOINT}" = /* ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CODEBOOK_CHECKPOINT="${SCRIPT_DIR}/${CODEBOOK_CHECKPOINT}"
fi

CODEBOOK_CHECKPOINT=$(readlink -f "${CODEBOOK_CHECKPOINT}" 2>/dev/null || realpath "${CODEBOOK_CHECKPOINT}" 2>/dev/null || echo "${CODEBOOK_CHECKPOINT}")

if [ ! -f "${CODEBOOK_CHECKPOINT}" ]; then
    echo "错误: 码本模型文件不存在: ${CODEBOOK_CHECKPOINT}"
    exit 1
fi

echo "================================================="
echo "批量超参数搜索"
echo "================================================="
echo "数据集: ${DSET}"
echo "码本模型: ${CODEBOOK_CHECKPOINT}"
echo "日志文件: ${LOG_FILE}"
echo "摘要文件: ${SUMMARY_FILE}"
echo "================================================="

# =====================================================
# 计算总组合数
# =====================================================

TOTAL_COMBINATIONS=$((${#PRETRAIN_BATCH_SIZES[@]} * ${#PROGRESSIVE_STEP_SIZES[@]} * ${#CONTEXT_MULTIPLIERS[@]} * ${#N_LAYERS_LIST[@]} * ${#N_HEADS_LIST[@]} * ${#D_FF_LIST[@]} * ${#NUM_RESIDUAL_HIDDENS_LIST[@]}))
echo "总组合数: ${TOTAL_COMBINATIONS}"
echo ""

# =====================================================
# 训练函数
# =====================================================

run_training() {
    local PRETRAIN_BATCH_SIZE=$1
    local PROGRESSIVE_STEP_SIZE=$2
    local CONTEXT_MULTIPLIER=$3
    local N_LAYERS=$4
    local N_HEADS=$5
    local D_FF=$6
    local NUM_RESIDUAL_HIDDENS=$7
    local COMBINATION_NUM=$8
    
    # 计算 PRETRAIN_CONTEXT_POINTS
    local PRETRAIN_CONTEXT_POINTS=$((CONTEXT_MULTIPLIER * PROGRESSIVE_STEP_SIZE * PATCH_SIZE))
    
    # 计算 code_dim
    local CODE_DIM=$((EMBEDDING_DIM * PATCH_SIZE / COMPRESSION_FACTOR))
    
    # 检查 n_heads 是否能整除 code_dim
    if [ $((CODE_DIM % N_HEADS)) -ne 0 ]; then
        echo "[跳过] 组合 #${COMBINATION_NUM}: n_heads=${N_HEADS} 不能整除 code_dim=${CODE_DIM}"
        echo "BS=${PRETRAIN_BATCH_SIZE}, STEP=${PROGRESSIVE_STEP_SIZE}, CTX_MUL=${CONTEXT_MULTIPLIER}, L=${N_LAYERS}, H=${N_HEADS}, D_FF=${D_FF}, RH=${NUM_RESIDUAL_HIDDENS}"
        # 直接写入摘要文件（不使用重定向，避免缓冲问题）
        printf "[跳过] BS=%d, STEP=%d, CTX_MUL=%d, L=%d, H=%d, D_FF=%d, RH=%d\n" \
            "${PRETRAIN_BATCH_SIZE}" "${PROGRESSIVE_STEP_SIZE}" "${CONTEXT_MULTIPLIER}" \
            "${N_LAYERS}" "${N_HEADS}" "${D_FF}" "${NUM_RESIDUAL_HIDDENS}" >> "${SUMMARY_FILE}"
        return 2  # 返回2表示跳过
    fi
    
    # 生成模型名称（格式必须与 patch_vqvae_pretrain.py 中保存的格式一致）
    # 格式：patch_vqvae_ps{args.patch_size}_cb{args.codebook_size}_cd{code_dim}_l{args.n_layers}_in{args.context_points}_step{step_size}_model{args.model_id}
    local MODEL_NAME="patch_vqvae_ps${PATCH_SIZE}_cb${CODEBOOK_SIZE}_cd${CODE_DIM}_l${N_LAYERS}_in${PRETRAIN_CONTEXT_POINTS}_step${PROGRESSIVE_STEP_SIZE}_model${MODEL_ID}"
    
    echo ""
    echo "================================================="
    echo "[${COMBINATION_NUM}/${TOTAL_COMBINATIONS}] 开始训练"
    echo "================================================="
    echo "参数配置:"
    echo "  PRETRAIN_BATCH_SIZE: ${PRETRAIN_BATCH_SIZE}"
    echo "  PROGRESSIVE_STEP_SIZE: ${PROGRESSIVE_STEP_SIZE}"
    echo "  PRETRAIN_CONTEXT_POINTS: ${PRETRAIN_CONTEXT_POINTS} (${CONTEXT_MULTIPLIER} * ${PROGRESSIVE_STEP_SIZE} * ${PATCH_SIZE})"
    echo "  N_LAYERS: ${N_LAYERS}"
    echo "  N_HEADS: ${N_HEADS}"
    echo "  D_FF: ${D_FF}"
    echo "  NUM_RESIDUAL_HIDDENS: ${NUM_RESIDUAL_HIDDENS}"
    echo "  CODE_DIM: ${CODE_DIM}"
    echo "  模型名称: ${MODEL_NAME}"
    echo "================================================="
    
    # 记录到摘要文件（直接写入，避免缓冲）
    printf "[%d/%d] BS=%d, STEP=%d, CTX=%d (%dx), L=%d, H=%d, D_FF=%d, RH=%d, MODEL=%s\n" \
        "${COMBINATION_NUM}" "${TOTAL_COMBINATIONS}" "${PRETRAIN_BATCH_SIZE}" \
        "${PROGRESSIVE_STEP_SIZE}" "${PRETRAIN_CONTEXT_POINTS}" "${CONTEXT_MULTIPLIER}" \
        "${N_LAYERS}" "${N_HEADS}" "${D_FF}" "${NUM_RESIDUAL_HIDDENS}" "${MODEL_NAME}" >> "${SUMMARY_FILE}"
    
    # 构建预训练参数
    local PRETRAIN_ARGS=(
        --dset ${DSET}
        --context_points ${PRETRAIN_CONTEXT_POINTS}
        --progressive_step_size ${PROGRESSIVE_STEP_SIZE}
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
        --codebook_ema ${CODEBOOK_EMA}
        --ema_decay ${EMA_DECAY}
        --ema_eps ${EMA_EPS}
        --vqvae_checkpoint "${CODEBOOK_CHECKPOINT}"
        --freeze_vqvae ${FREEZE_VQVAE}
        --load_vq_weights 1
        --n_epochs ${PRETRAIN_EPOCHS}
        --lr ${PRETRAIN_LR}
        --weight_decay ${WEIGHT_DECAY}
        --revin ${REVIN}
        --vq_weight ${VQ_WEIGHT}
        --recon_weight ${RECON_WEIGHT}
        --model_id ${MODEL_ID}
    )
    
    if [ -n "${TRANSFORMER_HIDDEN_DIM}" ]; then
        PRETRAIN_ARGS+=(--transformer_hidden_dim ${TRANSFORMER_HIDDEN_DIM})
    fi
    
    # 运行预训练
    echo "开始预训练..."
    if ! python patch_vqvae_pretrain.py "${PRETRAIN_ARGS[@]}"; then
        echo "[失败] 组合 #${COMBINATION_NUM}: 预训练失败"
        printf "[失败] BS=%d, STEP=%d, CTX_MUL=%d, L=%d, H=%d, D_FF=%d, RH=%d\n" \
            "${PRETRAIN_BATCH_SIZE}" "${PROGRESSIVE_STEP_SIZE}" "${CONTEXT_MULTIPLIER}" \
            "${N_LAYERS}" "${N_HEADS}" "${D_FF}" "${NUM_RESIDUAL_HIDDENS}" >> "${SUMMARY_FILE}"
        return 1  # 返回1表示失败
    fi
    
    # 检查预训练模型是否存在
    local PRETRAINED_MODEL="saved_models/patch_vqvae/${DSET}/${MODEL_NAME}.pth"
    if [[ ! "${PRETRAINED_MODEL}" = /* ]]; then
        local SCRIPT_DIR_FINETUNE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        PRETRAINED_MODEL="${SCRIPT_DIR_FINETUNE}/${PRETRAINED_MODEL}"
    fi
    
    PRETRAINED_MODEL=$(readlink -f "${PRETRAINED_MODEL}" 2>/dev/null || realpath "${PRETRAINED_MODEL}" 2>/dev/null || echo "${PRETRAINED_MODEL}")
    
    if [ ! -f "${PRETRAINED_MODEL}" ]; then
        echo "[失败] 组合 #${COMBINATION_NUM}: 预训练模型文件不存在: ${PRETRAINED_MODEL}"
        echo "期望的模型名称: ${MODEL_NAME}.pth"
        echo "检查目录是否存在: $(dirname "${PRETRAINED_MODEL}")"
        if [ -d "$(dirname "${PRETRAINED_MODEL}")" ]; then
            echo "目录中的文件列表（前10个）:"
            ls -la "$(dirname "${PRETRAINED_MODEL}")" | head -12 || echo "无法列出目录内容"
        fi
        printf "[失败] BS=%d, STEP=%d, CTX_MUL=%d, L=%d, H=%d, D_FF=%d, RH=%d, MODEL=%s\n" \
            "${PRETRAIN_BATCH_SIZE}" "${PROGRESSIVE_STEP_SIZE}" "${CONTEXT_MULTIPLIER}" \
            "${N_LAYERS}" "${N_HEADS}" "${D_FF}" "${NUM_RESIDUAL_HIDDENS}" "${MODEL_NAME}" >> "${SUMMARY_FILE}"
        return 1  # 返回1表示失败
    fi
    
    echo "预训练完成，开始微调..."
    
    # 微调（多个预测长度）
    for TARGET_POINTS in ${TARGET_POINTS_LIST[@]}; do
        echo "  微调: Target Points = ${TARGET_POINTS}"
        
        if ! python patch_vqvae_finetune.py \
            --dset ${DSET} \
            --context_points ${FINETUNE_CONTEXT_POINTS} \
            --target_points ${TARGET_POINTS} \
            --batch_size ${FINETUNE_BATCH_SIZE} \
            --pretrained_model ${PRETRAINED_MODEL} \
            --n_epochs ${FINETUNE_EPOCHS} \
            --lr ${FINETUNE_LR} \
            --weight_decay ${WEIGHT_DECAY} \
            --revin ${REVIN} \
            --use_gumbel_softmax ${USE_GUMBEL_SOFTMAX} \
            --gumbel_temperature ${GUMBEL_TEMPERATURE} \
            --gumbel_hard ${GUMBEL_HARD} \
            ${AR_STEP_SIZE:+--ar_step_size ${AR_STEP_SIZE}} \
            --model_id ${MODEL_ID}; then
            echo "  [警告] 微调失败: Target Points = ${TARGET_POINTS}"
        fi
    done
    
    echo "[成功] 组合 #${COMBINATION_NUM} 完成"
    printf "[成功] BS=%d, STEP=%d, CTX_MUL=%d, L=%d, H=%d, D_FF=%d, RH=%d\n" \
        "${PRETRAIN_BATCH_SIZE}" "${PROGRESSIVE_STEP_SIZE}" "${CONTEXT_MULTIPLIER}" \
        "${N_LAYERS}" "${N_HEADS}" "${D_FF}" "${NUM_RESIDUAL_HIDDENS}" >> "${SUMMARY_FILE}"
    return 0
}

# =====================================================
# 主循环：遍历所有组合
# =====================================================

COMBINATION_NUM=0
SUCCESS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

START_TIME=$(date +%s)

for PRETRAIN_BATCH_SIZE in "${PRETRAIN_BATCH_SIZES[@]}"; do
    for PROGRESSIVE_STEP_SIZE in "${PROGRESSIVE_STEP_SIZES[@]}"; do
        for CONTEXT_MULTIPLIER in "${CONTEXT_MULTIPLIERS[@]}"; do
            for N_LAYERS in "${N_LAYERS_LIST[@]}"; do
                for N_HEADS in "${N_HEADS_LIST[@]}"; do
                    for D_FF in "${D_FF_LIST[@]}"; do
                        for NUM_RESIDUAL_HIDDENS in "${NUM_RESIDUAL_HIDDENS_LIST[@]}"; do
                            COMBINATION_NUM=$((COMBINATION_NUM + 1))
                            
                            # 显示进度
                            CURRENT_TIME=$(date +%s)
                            ELAPSED=$((CURRENT_TIME - START_TIME))
                            if [ ${COMBINATION_NUM} -gt 1 ]; then
                                AVG_TIME=$((ELAPSED / (COMBINATION_NUM - 1)))
                                REMAINING=$((AVG_TIME * (TOTAL_COMBINATIONS - COMBINATION_NUM)))
                                ETA_HOURS=$((REMAINING / 3600))
                                ETA_MINS=$(((REMAINING % 3600) / 60))
                                echo ""
                                echo "进度: ${COMBINATION_NUM}/${TOTAL_COMBINATIONS} | 已用时间: ${ELAPSED}s | 预计剩余: ${ETA_HOURS}h ${ETA_MINS}m"
                            fi
                            
                            # 运行训练（不因单个失败而停止）
                            run_training "${PRETRAIN_BATCH_SIZE}" "${PROGRESSIVE_STEP_SIZE}" "${CONTEXT_MULTIPLIER}" "${N_LAYERS}" "${N_HEADS}" "${D_FF}" "${NUM_RESIDUAL_HIDDENS}" "${COMBINATION_NUM}"
                            TRAIN_RESULT=$?
                            
                            if [ ${TRAIN_RESULT} -eq 0 ]; then
                                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                            elif [ ${TRAIN_RESULT} -eq 2 ]; then
                                SKIP_COUNT=$((SKIP_COUNT + 1))
                            else
                                FAIL_COUNT=$((FAIL_COUNT + 1))
                            fi
                        done
                    done
                done
            done
        done
    done
done

# =====================================================
# 总结
# =====================================================

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_TIME / 3600))
TOTAL_MINS=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "================================================="
echo "批量搜索完成！"
echo "================================================="
echo "总组合数: ${TOTAL_COMBINATIONS}"
echo "成功: ${SUCCESS_COUNT}"
echo "跳过: ${SKIP_COUNT}"
echo "失败: ${FAIL_COUNT}"
echo "总耗时: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
echo "日志文件: ${LOG_FILE}"
echo "摘要文件: ${SUMMARY_FILE}"
echo "================================================="

# 写入摘要文件（使用子shell避免影响主输出流）
{
    echo ""
    echo "================================================="
    echo "批量搜索总结"
    echo "================================================="
    echo "总组合数: ${TOTAL_COMBINATIONS}"
    echo "成功: ${SUCCESS_COUNT}"
    echo "跳过: ${SKIP_COUNT}"
    echo "失败: ${FAIL_COUNT}"
    echo "总耗时: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
    echo "================================================="
} >> "${SUMMARY_FILE}" 2>&1

