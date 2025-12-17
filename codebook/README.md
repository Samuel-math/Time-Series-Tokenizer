# VQVAE 独立训练

本文件夹包含独立训练 VQVAE（Vector Quantized Variational AutoEncoder）模型的脚本。

## 文件说明

- `vqvae_train.py`: VQVAE 训练脚本
- `vqvae_train.sh`: 训练启动脚本
- `README.md`: 本文件

## 使用方法

### 1. 直接运行脚本

```bash
cd vqvae-only
bash vqvae_train.sh
```

### 2. 使用 Python 命令

```bash
python vqvae_train.py \
    --dset ettm1 \
    --context_points 512 \
    --batch_size 64 \
    --embedding_dim 32 \
    --num_embeddings 256 \
    --compression_factor 4 \
    --n_epochs 100 \
    --lr 1e-4 \
    --save_path saved_models/vqvae_only/
```

## 参数说明

### 数据集参数
- `--dset`: 数据集名称（默认: ettm1）
- `--context_points`: 输入序列长度（默认: 512）
- `--batch_size`: 批次大小（默认: 64）
- `--scaler`: 数据缩放方式（默认: standard）
- `--features`: 特征类型（默认: M，多变量）

### VQVAE 模型参数
- `--patch_size`: Patch大小（时间步数，默认: 16）
- `--embedding_dim`: Embedding维度（默认: 32）
- `--num_embeddings`: 码本大小（默认: 256）
- `--compression_factor`: 压缩因子，可选 4/8/12/16（默认: 4）
- `--block_hidden_size`: 隐藏层维度（默认: 64）
- `--num_residual_layers`: 残差层数（默认: 2）
- `--res_hidden_size`: 残差隐藏层维度（默认: 32）
- `--commitment_cost`: VQ commitment cost（默认: 0.25）

### 训练参数
- `--n_epochs`: 训练轮数（默认: 100）
- `--lr`: 学习率（默认: 1e-4）
- `--weight_decay`: 权重衰减（默认: 1e-4）
- `--revin`: 是否使用RevIN（默认: 1，启用）
- `--amp`: 是否启用混合精度（默认: 1，启用）

### 保存参数
- `--save_path`: 模型保存路径（默认: saved_models/vqvae_only/）
- `--model_id`: 模型ID（默认: 1）

## 注意事项

1. **Patch处理**: VQVAE对时间序列的patches进行编码和量化。序列会被划分为大小为`patch_size`的patches，每个patch独立进行编码-量化-解码过程。

2. **单通道限制**: 当前 VQVAE 实现只支持单通道输入，训练脚本默认使用第一个通道。如果需要为每个通道训练独立的模型，需要修改代码。

3. **Patch大小选择**: `patch_size`应该能被`context_points`整除，以确保序列能被完整划分为patches。例如，如果`context_points=512`，`patch_size=16`，则会有32个patches。

2. **Loss 计算**: 
   - Reconstruction loss: MSE loss（使用 sum reduction，按样本数加权）
   - VQ loss: Commitment loss + Codebook loss
   - Total loss: Recon loss + VQ loss

3. **模型保存**: 
   - 最佳模型保存在 `{save_path}/{dset}/{model_name}.pth`
   - 训练历史保存在 `{save_path}/{dset}/{model_name}_history.csv`
   - 配置文件保存在 `{save_path}/{dset}/{model_name}_config.json`

## 输出文件

训练完成后会生成：
- `{model_name}.pth`: 模型checkpoint（包含模型权重、配置、训练参数等）
- `{model_name}_history.csv`: 训练历史（loss曲线）
- `{model_name}_config.json`: 模型配置

## 示例输出

```
Epoch   1/100 | Train Loss: 0.123456 (VQ: 0.045678, Recon: 0.077890, Perp: 45.23) | Valid Loss: 0.112345 (VQ: 0.043456, Recon: 0.068901, Perp: 48.12)
  -> Best model saved (val_loss: 0.112345)
...
```
