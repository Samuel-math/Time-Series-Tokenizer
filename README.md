# Time Series Tokenizer

基于 Patch-VQVAE + Transformer 的时间序列离散化与预训练框架。

---

## 整体 Motivation

核心思想是：**先把时间序列"离散化"，再用语言模型的方式建模序列结构**。

| 类比 | NLP | 本模型 |
|------|-----|--------|
| 基本单元 | 词（word） | patch（定长时间窗口） |
| 词表 | 词汇表 | VQ 码本（codebook） |
| 序列建模 | 语言模型 | Causal Transformer |
| 预训练任务 | Next Word Prediction | Next Token Prediction（NTP） |

与直接用连续表示相比，离散 token 有如下优点：
- 训练信号更强（分类 CrossEntropy vs 回归 MSE）
- 自然支持 top-k/nucleus 采样
- 码本可解释，便于分析时间序列结构

额外引入两个设计动机：

- **Robust VQVAE**：把每个 patch 分解为"干净主体 + 稀疏异常分量 s"，让码本只编码结构性信息，异常值不污染 latent space
- **2 层残差量化（RVQ）**：第 0 层编码主要信息，第 1 层编码残差，提升 patch 表达精度，同时在 NTP 中保留两层的预测信号

---

## 项目结构

```
.
├── src/
│   ├── models/
│   │   ├── patch_vqvae_transformer.py  # 主模型（VQVAE + Transformer）
│   │   ├── codebook_model.py           # 轻量码本模型（VQ-only 训练用）
│   │   └── vqvae.py                    # Encoder / Decoder / SparseNet 基础组件
│   └── training/
│       └── patch_vqvae_pretrain_common.py  # NTP 预训练公共逻辑
├── vqvae-only/
│   ├── codebook_pretrain.py            # 阶段1：VQ 预训练入口
│   └── codebook_pretrain.sh
├── decoder_only_NTP/
│   ├── patch_vqvae_pretrain.py         # 阶段2：NTP 预训练入口
│   └── patch_vqvae_pretrain_then_finetune.sh
└── decoder_only_forcasting/
    └── patch_vqvae_pretrain.py         # 预测任务预训练入口
```

---

## VQ 模块结构

```
输入 patch  [B × P, 1, patch_size]
       │
       ▼
  ┌─────────────┐
  │  SparseNet  │  轻量 2层 Conv1d + tanh
  │             │  → 输出稀疏异常分量 s
  └─────────────┘
  x_clean = x_patch − s
       │
       ▼
  ┌─────────────┐
  │   Encoder   │  Conv1d 下采样（× compression_factor）
  │             │  + ResidualStack（残差卷积块）
  └─────────────┘
  z_e: [B×P, embedding_dim, compressed_len]
  z_flat = flatten → [B×P, code_dim]
  （code_dim = embedding_dim × compressed_len）
       │
       ▼
  ┌───────────────────────────────────┐
  │         ResidualVQ（2层）          │
  │                                   │
  │  Layer 0:  z_flat → 最近邻 → z_q0 │  ← EMA 更新码本
  │            residual = z_flat − z_q0│
  │  Layer 1:  residual → 最近邻 → z_q1│  ← EMA 更新码本
  │                                   │
  │  z_q = z_q0 + z_q1               │
  │  indices = [idx_0, idx_1]         │
  └───────────────────────────────────┘
       │
       ▼
  ┌─────────────┐
  │   Decoder   │  Conv1d 上采样 + ResidualStack
  └─────────────┘
  x_recon + s  →  最终重建结果
```

**码本更新机制**：
- 训练时使用 EMA（`ema_cluster_size` + `ema_w`）更新码本，无梯度回传
- 配合**死码复活**：当某个码字的 `ema_cluster_size < threshold` 时，用当前 batch 的随机样本替换该码字，防止码本坍塌

**Channel-Independent 设计**：
所有通道共享同一套 Encoder / VQ / Decoder 参数，通道维度通过 batch 展开并行处理（`[B × C × P, ...]`），通道间无信息交互。

---

## 两阶段训练流程

### 阶段 1：VQ 预训练（`vqvae-only/`）

**目标**：单独训练 Encoder + RVQ + Decoder，让码本收敛并覆盖数据分布。

**损失函数**：

```
L = recon_weight × MSE(x_recon, x_patch)
  + vq_weight   × Σ_layer commitment_loss_l
  + sparse_weight × ||s||₁
```

- `commitment_loss`：约束 encoder 输出对齐码本（stop-gradient trick）
- `sparse_weight × ||s||₁`：L1 正则鼓励异常分量稀疏

训练流程：
1. 启动时用 K-means 对 encoder 输出做数据驱动的码本初始化（`init_codebook_from_data`）
2. 训练过程中 EMA 持续更新码本
3. 按 validation loss 早停，保存最优检查点

---

### 阶段 2：NTP 预训练（`decoder_only_NTP/`）

**目标**：加载冻结的 VQ，训练 Transformer 做 Next Token Prediction。

#### 渐进式（Progressive）预训练策略

用不同长度的上下文窗口预测固定步长（`step_size`）的未来 patches，一次 forward 同时训练多个 stage：

```
Stage 1:  [z₁]                    → 预测 [z₂ … z_{s+1}]
Stage 2:  [z₁, …, z_{s+1}]       → 预测 [z_{s+2} … z_{2s+1}]
Stage k:  [z₁, …, z_{(k-1)s+1}]  → 预测 [z_{(k-1)s+2} … z_{ks+1}]
```

每个 stage 的前向过程：

```
上下文 tokens + zero placeholder → CausalTransformer（RoPE）
                                        │
                              取 placeholder 位置输出 h_target
                                   ┌────┴────┐
                              Head_0         Head_1
                              （预测 idx_0）  （预测 idx_1）
```

**损失函数**：

```
L = Σ_stage Σ_layer (w_l / Σw) × CrossEntropy(logits_l, idx_l)
  + vq_weight   × commitment_loss    （freeze_vqvae=1 时为 0）
  + recon_weight × MSE               （freeze_vqvae=1 时为 0）
```

- `w_l`：每层 RVQ 的 pred_loss 权重，可通过 `--rq_layer_weights` 配置（默认均等）
- 建议 `w_0=1.0, w_1=0.5`，因为第 1 层编码残差、信息量更少

#### NMPP 模式（`--use_raw_input 1`）

Transformer 接收原始 patch 的线性投影，而非 VQ 量化向量；VQ 仅作为 teacher 提供 token 标签，梯度不流向 VQ。

---

## 关键超参速查

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `patch_size` | 每个 patch 的时间步数 | 16 |
| `embedding_dim` | Encoder 输出维度 | 32 或 64 |
| `compression_factor` | 下采样倍数 | 8 |
| `codebook_size` | 每层码本大小 | 256 |
| `n_rq_layers` | RVQ 层数 | 2 |
| `progressive_step_size` | NTP 每阶段预测 patch 数 | 6 |
| `rq_layer_weights` | 两层 pred_loss 权重 | `1.0 0.5` |
| `use_raw_input` | NMPP 模式开关 | 0 / 1 |
| `sparse_weight` | SparseNet L1 正则权重 | 0.0（不用时关闭） |
| `channel_chunk_size` | encode 时每次并行通道数 | 0（全部） |
