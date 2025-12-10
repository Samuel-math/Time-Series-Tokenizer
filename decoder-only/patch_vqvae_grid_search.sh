#!/bin/bash

# =====================================================
# Patch VQVAE + Transformer 网格搜索脚本
# =====================================================
# 
# 对不同的参数组合进行网格搜索
# 记录每个参数组合在不同target_len下的test_mse和test_mae
# 结果保存到 grid.txt
# =====================================================

# =====================================================
# 基础配置参数
# =====================================================

DSET="ettm1"
BASE_MODEL_ID=1

# ----- 固定参数 -----
NUM_HIDDENS=64
NUM_RESIDUAL_LAYERS=2
NUM_RESIDUAL_HIDDENS=32
PRETRAIN_EPOCHS=100
PRETRAIN_BATCH_SIZE=128
PRETRAIN_LR=3e-4
RECON_WEIGHT=0.1
FINETUNE_CONTEXT_POINTS=512
FINETUNE_EPOCHS=50
FINETUNE_BATCH_SIZE=64
FINETUNE_LR=1e-4
TARGET_POINTS_LIST=(96 192 336 720)
REVIN=1
WEIGHT_DECAY=1e-4
USE_PATCH_ATTENTION=1  # 启用patch内attention

# =====================================================
# 网格搜索参数范围
# =====================================================

# 定义要搜索的参数组合
PATCH_SIZES=(16 32)
EMBEDDING_DIMS=(32 64)
CODEBOOK_SIZES=(14 16 18 20)
N_LAYERS_LIST=(2 3 4)
N_HEADS_LIST=(4 8)
D_FF_LIST=(128 256)
DROPOUT_LIST=(0.1 0.3)
COMPRESSION_FACTORS=(2 4)
PRETRAIN_CONTEXT_POINTS_LIST=(256 512)
VQ_WEIGHTS=(0.25 0.5 0.7)

# =====================================================
# 结果文件
# =====================================================

RESULT_FILE="grid.txt"

# 初始化结果文件（如果不存在）
if [ ! -f "${RESULT_FILE}" ]; then
    echo "Grid Search Results" > ${RESULT_FILE}
    echo "===================" >> ${RESULT_FILE}
    echo "" >> ${RESULT_FILE}
    echo "Format: patch_size,embedding_dim,codebook_size,n_layers,n_heads,d_ff,dropout,compression_factor,pretrain_context_points,vq_weight,target_len,test_mse,test_mae" >> ${RESULT_FILE}
    echo "" >> ${RESULT_FILE}
fi

# 辅助函数：追加结果到文件（立即刷新）
append_result() {
    echo "$1" >> ${RESULT_FILE}
    # 立即刷新到磁盘
    sync
}

# =====================================================
# 网格搜索主循环
# =====================================================

MODEL_ID=${BASE_MODEL_ID}
TOTAL_COMBINATIONS=0

# 计算总组合数
for PATCH_SIZE in ${PATCH_SIZES[@]}; do
    for EMBEDDING_DIM in ${EMBEDDING_DIMS[@]}; do
        for CODEBOOK_SIZE in ${CODEBOOK_SIZES[@]}; do
            for N_LAYERS in ${N_LAYERS_LIST[@]}; do
                for N_HEADS in ${N_HEADS_LIST[@]}; do
                    for D_FF in ${D_FF_LIST[@]}; do
                        for DROPOUT in ${DROPOUT_LIST[@]}; do
                            for COMPRESSION_FACTOR in ${COMPRESSION_FACTORS[@]}; do
                                for PRETRAIN_CONTEXT_POINTS in ${PRETRAIN_CONTEXT_POINTS_LIST[@]}; do
                                    for VQ_WEIGHT in ${VQ_WEIGHTS[@]}; do
                                        CODE_DIM=$((EMBEDDING_DIM * PATCH_SIZE / COMPRESSION_FACTOR))
                                        # 检查n_heads是否能整除code_dim
                                        if [ $((CODE_DIM % N_HEADS)) -eq 0 ]; then
                                            TOTAL_COMBINATIONS=$((TOTAL_COMBINATIONS + 1))
                                        fi
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "总共有 ${TOTAL_COMBINATIONS} 个有效参数组合"
echo "开始网格搜索..."
echo ""

CURRENT_COMBINATION=0

for PATCH_SIZE in ${PATCH_SIZES[@]}; do
    for EMBEDDING_DIM in ${EMBEDDING_DIMS[@]}; do
        for CODEBOOK_SIZE in ${CODEBOOK_SIZES[@]}; do
            for N_LAYERS in ${N_LAYERS_LIST[@]}; do
                for N_HEADS in ${N_HEADS_LIST[@]}; do
                    for D_FF in ${D_FF_LIST[@]}; do
                        for DROPOUT in ${DROPOUT_LIST[@]}; do
                            for COMPRESSION_FACTOR in ${COMPRESSION_FACTORS[@]}; do
                                for PRETRAIN_CONTEXT_POINTS in ${PRETRAIN_CONTEXT_POINTS_LIST[@]}; do
                                    for VQ_WEIGHT in ${VQ_WEIGHTS[@]}; do
                                        CODE_DIM=$((EMBEDDING_DIM * PATCH_SIZE / COMPRESSION_FACTOR))
                                        
                                        # 检查n_heads是否能整除code_dim
                                        if [ $((CODE_DIM % N_HEADS)) -ne 0 ]; then
                                            continue
                                        fi
                                        
                                        CURRENT_COMBINATION=$((CURRENT_COMBINATION + 1))
                                        MODEL_NAME="patch_vqvae_ps${PATCH_SIZE}_cb${CODEBOOK_SIZE}_cd${CODE_DIM}_l${N_LAYERS}_model${MODEL_ID}"
                                        
                                        echo "================================================="
                                        echo "[${CURRENT_COMBINATION}/${TOTAL_COMBINATIONS}] 参数组合:"
                                        echo "  PATCH_SIZE=${PATCH_SIZE}"
                                        echo "  EMBEDDING_DIM=${EMBEDDING_DIM}"
                                        echo "  CODEBOOK_SIZE=${CODEBOOK_SIZE}"
                                        echo "  N_LAYERS=${N_LAYERS}"
                                        echo "  N_HEADS=${N_HEADS}"
                                        echo "  D_FF=${D_FF}"
                                        echo "  DROPOUT=${DROPOUT}"
                                        echo "  COMPRESSION_FACTOR=${COMPRESSION_FACTOR}"
                                        echo "  PRETRAIN_CONTEXT_POINTS=${PRETRAIN_CONTEXT_POINTS}"
                                        echo "  VQ_WEIGHT=${VQ_WEIGHT}"
                                        echo "  CODE_DIM=${CODE_DIM}"
                                        echo "================================================="
                            
                            # =====================================================
                            # 阶段 1: 预训练
                            # =====================================================
                            echo ""
                            echo "阶段 1: 预训练..."
                            
                            python patch_vqvae_pretrain.py \
                                --dset ${DSET} \
                                --context_points ${PRETRAIN_CONTEXT_POINTS} \
                                --batch_size ${PRETRAIN_BATCH_SIZE} \
                                --patch_size ${PATCH_SIZE} \
                                --embedding_dim ${EMBEDDING_DIM} \
                                --compression_factor ${COMPRESSION_FACTOR} \
                                --codebook_size ${CODEBOOK_SIZE} \
                                --n_layers ${N_LAYERS} \
                                --n_heads ${N_HEADS} \
                                --d_ff ${D_FF} \
                                --dropout ${DROPOUT} \
                                --num_hiddens ${NUM_HIDDENS} \
                                --num_residual_layers ${NUM_RESIDUAL_LAYERS} \
                                --num_residual_hiddens ${NUM_RESIDUAL_HIDDENS} \
                                --use_patch_attention ${USE_PATCH_ATTENTION} \
                                --tcn_num_layers 2 \
                                --tcn_kernel_size 3 \
                                --n_epochs ${PRETRAIN_EPOCHS} \
                                --lr ${PRETRAIN_LR} \
                                --weight_decay ${WEIGHT_DECAY} \
                                --revin ${REVIN} \
                                --vq_weight ${VQ_WEIGHT} \
                                --recon_weight ${RECON_WEIGHT} \
                                --model_id ${MODEL_ID}
                            
                                        if [ $? -ne 0 ]; then
                                            echo "预训练失败，跳过此参数组合"
                                            for TARGET_POINTS in ${TARGET_POINTS_LIST[@]}; do
                                                append_result "${PATCH_SIZE},${EMBEDDING_DIM},${CODEBOOK_SIZE},${N_LAYERS},${N_HEADS},${D_FF},${DROPOUT},${COMPRESSION_FACTOR},${PRETRAIN_CONTEXT_POINTS},${VQ_WEIGHT},${TARGET_POINTS},FAILED,FAILED"
                                            done
                                            continue
                                        fi
                            
                                        # =====================================================
                                        # 阶段 2: 微调 (多个预测长度)
                                        # =====================================================
                                        PRETRAINED_MODEL="saved_models/patch_vqvae/${DSET}/${MODEL_NAME}.pth"
                                        
                                        echo ""
                                        echo "阶段 2: 微调..."
                                        
                                        for TARGET_POINTS in ${TARGET_POINTS_LIST[@]}; do
                                            echo ""
                                            echo "微调: Target Points = ${TARGET_POINTS}"
                                            
                                            python patch_vqvae_finetune.py \
                                                --dset ${DSET} \
                                                --context_points ${FINETUNE_CONTEXT_POINTS} \
                                                --target_points ${TARGET_POINTS} \
                                                --batch_size ${FINETUNE_BATCH_SIZE} \
                                                --pretrained_model ${PRETRAINED_MODEL} \
                                                --n_epochs ${FINETUNE_EPOCHS} \
                                                --lr ${FINETUNE_LR} \
                                                --weight_decay ${WEIGHT_DECAY} \
                                                --revin ${REVIN} \
                                                --model_id ${MODEL_ID}
                                            
                                            if [ $? -eq 0 ]; then
                                                # 读取结果CSV文件
                                                MODEL_NAME_FINETUNE="patch_vqvae_finetune_cw${FINETUNE_CONTEXT_POINTS}_tw${TARGET_POINTS}_model${MODEL_ID}"
                                                RESULTS_CSV="saved_models/patch_vqvae_finetune/${DSET}/${MODEL_NAME_FINETUNE}_results.csv"
                                                
                                                if [ -f "${RESULTS_CSV}" ]; then
                                                    # 从CSV文件中提取MSE和MAE
                                                    # CSV格式: metric,value (第一行是MSE，第二行是MAE)
                                                    TEST_MSE=$(awk -F',' 'NR==2 && $1=="MSE" {print $2}' ${RESULTS_CSV})
                                                    TEST_MAE=$(awk -F',' 'NR==3 && $1=="MAE" {print $2}' ${RESULTS_CSV})
                                                    
                                                    # 如果上面的方法不行，尝试另一种方式
                                                    if [ -z "${TEST_MSE}" ]; then
                                                        TEST_MSE=$(awk -F',' 'NR==2 {print $2}' ${RESULTS_CSV})
                                                    fi
                                                    if [ -z "${TEST_MAE}" ]; then
                                                        TEST_MAE=$(awk -F',' 'NR==3 {print $2}' ${RESULTS_CSV})
                                                    fi
                                                    
                                                    # 写入结果文件（立即刷新）
                                                    append_result "${PATCH_SIZE},${EMBEDDING_DIM},${CODEBOOK_SIZE},${N_LAYERS},${N_HEADS},${D_FF},${DROPOUT},${COMPRESSION_FACTOR},${PRETRAIN_CONTEXT_POINTS},${VQ_WEIGHT},${TARGET_POINTS},${TEST_MSE},${TEST_MAE}"
                                                    
                                                    echo "结果已记录: MSE=${TEST_MSE}, MAE=${TEST_MAE}"
                                                else
                                                    echo "警告: 结果文件不存在 ${RESULTS_CSV}"
                                                    append_result "${PATCH_SIZE},${EMBEDDING_DIM},${CODEBOOK_SIZE},${N_LAYERS},${N_HEADS},${D_FF},${DROPOUT},${COMPRESSION_FACTOR},${PRETRAIN_CONTEXT_POINTS},${VQ_WEIGHT},${TARGET_POINTS},NOT_FOUND,NOT_FOUND"
                                                fi
                                            else
                                                echo "微调失败"
                                                append_result "${PATCH_SIZE},${EMBEDDING_DIM},${CODEBOOK_SIZE},${N_LAYERS},${N_HEADS},${D_FF},${DROPOUT},${COMPRESSION_FACTOR},${PRETRAIN_CONTEXT_POINTS},${VQ_WEIGHT},${TARGET_POINTS},FAILED,FAILED"
                                            fi
                                        done
                                        
                                        MODEL_ID=$((MODEL_ID + 1))
                                        echo ""
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo ""
echo "================================================="
echo "网格搜索完成！"
echo "结果已保存到: ${RESULT_FILE}"
echo "================================================="
