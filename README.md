# VQVAE + Transformer 预训练和微调指南

本指南介绍如何使用VQVAE + Transformer架构进行时间序列的预训练和微调。

## 架构概述

### 预训练阶段
1. **输入**: 时间序列 `[B, T, C]` (Batch, Time, Channels)
2. **VQVAE编码**: 对每个通道使用VQVAE encoder，得到 `[B, embedding_dim, T/compression_factor, C]`
3. **Transformer处理**: 使用channel independent的Transformer backbone（mask attention）
4. **输出**: `[B, codebook_size, T/compression_factor, C]` - 每个位置对应codebook的概率分布
5. **损失**: Next token prediction（交叉熵损失）

### 微调阶段
1. **输入**: 时间序列 `[B, context_len, C]`
2. **预测**: 使用预训练的Transformer + VQVAE Decoder
3. **输出**: `[B, target_len, C]` - 预测结果
4. **损失**: MSE损失
5. **冻结**: VQVAE的所有参数（encoder、codebook、decoder）均冻结

## 使用步骤

### 步骤1: 预训练VQVAE（如果还没有）

```bash
python vqvae_pretrain.py \
    --dset_pretrain ettm1 \
    --context_points 512 \
    --batch_size 64 \
    --n_epochs_pretrain 100 \
    --config_path model_config/ettm1_vqvae.json \
    --save_path saved_models/vqvae/
```

### 步骤2: 预训练VQVAE + Transformer

```bash
python vqvae_transformer_pretrain.py \
    --dset_pretrain ettm1 \
    --context_points 512 \
    --batch_size 64 \
    --n_epochs_pretrain 50 \
    --vqvae_config_path saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/configs/config_file.json \
    --vqvae_checkpoint saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/checkpoints/best_model.pth \
    --d_model 128 \
    --n_layers 3 \
    --n_heads 8 \
    --d_ff 256 \
    --dropout 0.1 \
    --learning_rate 1e-4 \
    --save_path saved_models/vqvae_transformer/
```

**参数说明**:
- `--dset_pretrain`: 数据集名称
- `--context_points`: 输入序列长度
- `--vqvae_config_path`: VQVAE配置文件路径
- `--vqvae_checkpoint`: 预训练的VQVAE模型路径（可选，如果不提供则随机初始化）
- `--d_model`: Transformer的模型维度
- `--n_layers`: Transformer层数
- `--n_heads`: 注意力头数
- `--d_ff`: Feed-forward网络维度
- `--dropout`: Dropout率

### 步骤3: 微调模型

```bash
python vqvae_transformer_finetune.py \
    --dset_finetune ettm1 \
    --context_points 512 \
    --target_points 96 \
    --batch_size 64 \
    --n_epochs_finetune 20 \
    --pretrained_model_path saved_models/vqvae_transformer/vqvae_transformer_d128_l3_h8_epochs50/checkpoints/best_model.pth \
    --vqvae_config_path saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/configs/config_file.json \
    --vqvae_checkpoint saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/checkpoints/best_model.pth \
    --learning_rate 1e-4 \
    --save_path saved_models/vqvae_transformer_finetune/
```

**参数说明**:
- `--dset_finetune`: 微调数据集名称
- `--context_points`: 输入序列长度
- `--target_points`: 预测长度
- `--pretrained_model_path`: 预训练的Transformer模型路径
- `--vqvae_checkpoint`: VQVAE模型路径（用于加载decoder）

## 模型架构细节

### VQVAETransformerPretrain

**主要组件**:
1. **VQVAE Encoder**: 对每个通道独立编码
2. **VQ Layer**: 向量量化层（codebook）
3. **Projection Layer**: 将embedding维度投影到Transformer维度
4. **Transformer Encoder**: 使用mask attention的Transformer层
5. **Codebook Head**: 预测codebook索引的概率分布

**关键特性**:
- Channel Independent: 每个通道独立处理
- Mask Attention: 使用causal mask实现next token prediction
- 可选的VQVAE权重加载: 可以从预训练的VQVAE加载encoder和codebook权重

### VQVAETransformerFinetune

**主要组件**:
1. **预训练的Transformer**: 冻结的预训练模型
2. **VQVAE Decoder**: 冻结的decoder

**工作流程**:
1. 使用预训练Transformer获取codebook logits
2. 选择最可能的codebook索引
3. 从codebook中获取对应的embeddings
4. 通过decoder解码得到预测结果

## 注意事项

1. **数据标准化**: 数据在进入模型前已经通过StandardScaler标准化（在datautils.py中处理）

2. **VQVAE权重**: 
   - 如果提供了VQVAE checkpoint，会自动加载encoder和codebook权重
   - 可以选择冻结VQVAE部分（在代码中取消注释相关行）

3. **微调时的冻结**:
   - 默认情况下，预训练模型和decoder的所有参数都被冻结
   - 如果需要微调某些层，可以修改`VQVAETransformerFinetune`类中的冻结逻辑

4. **内存使用**:
   - 由于channel independent的处理方式，内存使用与通道数成正比
   - 如果内存不足，可以减小batch_size或使用梯度累积

5. **训练技巧**:
   - 预训练时可以使用较大的学习率（1e-4）
   - 微调时建议使用较小的学习率（1e-5到1e-4）
   - 可以使用学习率调度器（如CosineAnnealingLR）

## 文件结构

```
Time-Series-Tokenizer/
├── src/
│   └── models/
│       └── vqvae_transformer.py      # 模型架构定义
├── vqvae_transformer_pretrain.py    # 预训练脚本
├── vqvae_transformer_finetune.py    # 微调脚本
└── README_VQVAE_TRANSFORMER.md      # 本文件
```

## 示例输出

### 预训练阶段
```
加载预训练的VQVAE权重: saved_models/vqvae/...
✓ VQVAE Encoder权重已加载
✓ VQ Codebook权重已加载
模型参数总数: 1,234,567
可训练参数: 1,234,567
开始训练，总epoch数: 50
Epoch 000, Batch 0000, Loss: 5.234567
Epoch 000 | train_loss = 4.123456 | val_loss = 4.234567
New best model saved at epoch 0 (val_loss=4.234567)
...
```

### 微调阶段
```
✓ 预训练Transformer模型已加载
✓ VQVAE Decoder已加载
模型参数总数: 1,234,567
可训练参数: 0
警告：所有参数都被冻结，无法训练。请取消冻结某些层。
```

（如果需要微调，需要取消冻结某些层）

## 故障排除

1. **CUDA out of memory**: 减小batch_size或使用梯度累积
2. **模型加载失败**: 检查checkpoint路径是否正确
3. **维度不匹配**: 确保VQVAE配置与预训练时一致
4. **训练不收敛**: 尝试调整学习率或使用学习率调度器

## 扩展功能

可以进一步扩展的功能：
1. 添加学习率调度器
2. 支持混合精度训练（FP16）
3. 添加更多的评估指标
4. 支持多GPU训练
5. 添加tensorboard日志记录

