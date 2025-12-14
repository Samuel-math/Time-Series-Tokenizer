# Codebook预训练模块

这个文件夹包含独立的码本（Codebook）预训练代码，用于训练 VQVAE 的核心组件：Encoder、Vector Quantizer (VQ) 和 Decoder。

## 文件结构

```
codebook-only/
├── codebook_pretrain.py    # 码本预训练脚本
├── saved_models/           # 保存的模型（自动创建）
│   └── codebook/
│       └── {dataset}/
│           └── codebook_*.pth
└── README.md
```

## 功能

- **独立训练**: 只训练 Encoder、VQ 和 Decoder，不包含 Transformer
- **EMA支持**: 支持使用 Exponential Moving Average 更新码本
- **早停机制**: 自动早停防止过拟合
- **模型保存**: 保存最佳模型、训练历史和配置

## 使用方法

### 直接运行

```bash
cd codebook-only
python codebook_pretrain.py \
    --dset ettm1 \
    --context_points 512 \
    --batch_size 64 \
    --patch_size 16 \
    --embedding_dim 32 \
    --codebook_size 256 \
    --compression_factor 4 \
    --n_epochs 50 \
    --lr 5e-5 \
    --codebook_ema 1 \
    --model_id 1
```

### 从 decoder-only 调用

在 `decoder-only/run_all_three_stages.sh` 中会自动调用此脚本：

```bash
cd decoder-only
bash run_all_three_stages.sh
```

## 输出

训练完成后，模型会保存在：
```
codebook-only/saved_models/codebook/{dataset}/codebook_ps{patch_size}_cb{codebook_size}_cd{code_dim}_model{model_id}.pth
```

同时会生成：
- `*_history.csv`: 训练历史记录
- `*_config.json`: 模型配置

## 参数说明

主要参数：
- `--dset`: 数据集名称（如 ettm1）
- `--patch_size`: Patch 大小
- `--embedding_dim`: Embedding 维度
- `--codebook_size`: 码本大小
- `--compression_factor`: 压缩因子（4, 8, 12, 16）
- `--codebook_ema`: 是否使用 EMA（1=是，0=否）
- `--lr`: 学习率（默认 5e-5）
- `--n_epochs`: 训练轮数

## 注意事项

1. 训练好的码本模型会被 `decoder-only` 中的 Transformer 预训练脚本使用
2. 确保 `codebook-only` 文件夹与项目根目录在同一层级
3. 模型保存路径是相对于 `codebook-only` 文件夹的
