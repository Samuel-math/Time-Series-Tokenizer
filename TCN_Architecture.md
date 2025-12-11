# TCN (Temporal Convolutional Network) 架构详解

## 1. TCN 概述

**TCN (Temporal Convolutional Network)** 是一种专门用于时序数据建模的卷积神经网络架构。它结合了**因果卷积 (Causal Convolution)** 和**膨胀卷积 (Dilated Convolution)**，能够有效捕捉时序依赖关系。

### 核心特点：
- ✅ **因果性保证**：只使用当前和过去的信息，不会泄露未来信息
- ✅ **长距离依赖**：通过膨胀卷积扩大感受野
- ✅ **并行计算**：相比RNN，卷积可以并行计算，训练更快
- ✅ **稳定梯度**：残差连接帮助梯度传播

---

## 2. 关键组件

### 2.1 因果卷积 (Causal Convolution)

**定义**：输出位置 `t` 只依赖于输入位置 `≤ t` 的值。

**实现方式**：通过左侧padding实现
```
Padding = (kernel_size - 1) * dilation
```

**示例**（kernel_size=3, dilation=1）：
```
输入:  [x₁, x₂, x₃, x₄, x₅]
Padding: [0, 0, x₁, x₂, x₃, x₄, x₅]
输出:  [y₁, y₂, y₃, y₄, y₅]
       ↑    ↑    ↑    ↑    ↑
       只依赖  只依赖  只依赖  只依赖  只依赖
       x₁    x₁,x₂ x₁,x₂,x₃ x₂,x₃,x₄ x₃,x₄,x₅
```

### 2.2 膨胀卷积 (Dilated Convolution)

**定义**：在卷积核元素之间插入空洞，扩大感受野而不增加参数。

**膨胀率 (Dilation)**：元素之间的间隔
- dilation=1: 正常卷积，感受野 = kernel_size
- dilation=2: 间隔1个元素，感受野 = 2×(kernel_size-1)+1
- dilation=4: 间隔3个元素，感受野 = 4×(kernel_size-1)+1

**示例**（kernel_size=3）：
```
dilation=1: [●, ●, ●]         感受野=3
dilation=2: [●, _, ●, _, ●]   感受野=5
dilation=4: [●, _, _, _, ●, _, _, _, ●] 感受野=9
```

### 2.3 残差连接 (Residual Connection)

**作用**：
- 帮助梯度反向传播
- 允许模型学习恒等映射
- 稳定训练过程

---

## 3. PatchTCN 架构图

### 3.1 整体架构

```
输入: [B*num_patches, patch_size, C]
      ↓
   [Reshape]
      ↓
输入: [B, C, patch_size]  ←──┐
      ↓                        │
   ┌──────────────────────┐   │
   │   TCN Layer 1        │   │
   │   dilation=1         │   │
   │   kernel=3           │   │
   │   padding=2          │   │
   │   Conv1d → BN → GELU │   │
   └──────────────────────┘   │
      ↓                        │
   [裁剪padding]               │
      ↓                        │
   [残差连接] ────────────────┘
      ↓
   ┌──────────────────────┐
   │   TCN Layer 2        │
   │   dilation=2         │
   │   kernel=3           │
   │   padding=4          │
   │   Conv1d → BN → GELU │
   └──────────────────────┘
      ↓
   [裁剪padding]
      ↓
   [残差连接]
      ↓
   ┌──────────────────────┐
   │   Output Projection  │ (可选)
   │   Conv1d(1x1)        │
   └──────────────────────┘
      ↓
   [Reshape]
      ↓
输出: [B, patch_size, C]
      ↓
   [整体残差连接]
      ↓
输出: [B*num_patches, patch_size, C]
```

### 3.2 单层TCN Block详细结构

```
输入: [B, C_in, T]
      ↓
┌─────────────────────────────────┐
│  Causal Conv1d                  │
│  - kernel_size: 3               │
│  - dilation: d                  │
│  - padding: (3-1)*d = 2*d       │
│  - bias: False                  │
└─────────────────────────────────┘
      ↓
输出: [B, C_out, T + padding]
      ↓
┌─────────────────────────────────┐
│  BatchNorm1d                    │
│  - 归一化特征                   │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│  GELU Activation               │
│  - 非线性激活                   │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│  Dropout                        │
│  - 防止过拟合                   │
└─────────────────────────────────┘
      ↓
输出: [B, C_out, T + padding]
      ↓
[裁剪到原始长度 T]
      ↓
输出: [B, C_out, T]
      ↓
┌─────────────────────────────────┐
│  残差连接 (如果维度匹配)        │
│  output = input + output        │
└─────────────────────────────────┘
      ↓
输出: [B, C_out, T]
```

### 3.3 多层TCN的膨胀率设计

```
Layer 1: dilation=1  (2⁰)  → 感受野 = 3
Layer 2: dilation=2  (2¹)  → 感受野 = 5
Layer 3: dilation=4  (2²)  → 感受野 = 9
Layer 4: dilation=8  (2³)  → 感受野 = 17
...

总感受野 = 1 + Σ(2^i × (kernel_size-1))
         = 1 + 2 + 4 + 8 + ... = 2^(num_layers+1) - 1
```

**示例**（num_layers=2, kernel_size=3）：
```
输入序列: [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈]

Layer 1 (dilation=1):
  y₁ ← [x₁]
  y₂ ← [x₁, x₂]
  y₃ ← [x₁, x₂, x₃]
  y₄ ← [x₂, x₃, x₄]
  ...

Layer 2 (dilation=2):
  z₁ ← [y₁]
  z₂ ← [y₁, y₃]      (跳过y₂)
  z₃ ← [y₁, y₃, y₅]  (跳过y₂, y₄)
  z₄ ← [y₃, y₅, y₇]  (跳过y₄, y₆)
  ...

最终感受野: z₃ 可以看到 [x₁, x₂, x₃, x₄, x₅] (5个时间步)
```

---

## 4. 在我们项目中的应用

### 4.1 在 PatchVQVAETransformer 中的位置

```
时间序列输入: [B, T, C]
      ↓
[Patch划分]
      ↓
Patches: [B, num_patches, patch_size, C]
      ↓
[Reshape]
      ↓
[B*num_patches, patch_size, C]
      ↓
┌─────────────────────────────────┐
│      PatchTCN                  │ ← 在这里！
│  处理每个patch内的时序信息     │
└─────────────────────────────────┘
      ↓
[B*num_patches, patch_size, C]
      ↓
[VQVAE Encoder]
      ↓
[Vector Quantization]
      ↓
[Transformer]
      ↓
[预测]
```

### 4.2 为什么使用TCN？

1. **短序列建模**：patch_size通常较小（如16），TCN比Transformer更适合短序列
2. **计算效率**：卷积比attention更快，特别是对于短序列
3. **因果性**：天然保证时序因果性，适合时间序列预测
4. **参数效率**：相比attention，参数更少

### 4.3 参数配置

```python
PatchTCN(
    patch_size=16,        # patch长度
    n_channels=7,         # 通道数（如ETTm1数据集）
    dropout=0.1,          # Dropout率
    num_layers=2,         # TCN层数
    kernel_size=3,        # 卷积核大小
    hidden_dim=None       # 隐藏层维度（默认等于n_channels）
)
```

**感受野计算**：
- num_layers=2, kernel_size=3
- 总感受野 = 1 + 2×(3-1) + 4×(3-1) = 1 + 4 + 8 = 13
- 对于patch_size=16，可以覆盖81%的patch

---

## 5. 数据流示例

### 5.1 输入输出维度

```
输入: [B*num_patches, patch_size, C]
     例如: [128, 16, 7]  (128个patches, 每个16个时间步, 7个通道)

      ↓ Reshape
      
输入: [B, C, patch_size]
     例如: [128, 7, 16]

      ↓ TCN Layer 1 (dilation=1)
      
中间: [128, hidden_dim, 18]  (padding后)
      ↓ 裁剪
中间: [128, hidden_dim, 16]

      ↓ TCN Layer 2 (dilation=2)
      
中间: [128, hidden_dim, 20]  (padding后)
      ↓ 裁剪
中间: [128, hidden_dim, 16]

      ↓ Output Projection (如果需要)
      
输出: [128, 7, 16]
      ↓ Reshape
      
输出: [128, 16, 7]
      ↓ 残差连接
      
最终: [128, 16, 7]
```

### 5.2 感受野可视化

假设 patch_size=16, num_layers=2, kernel_size=3：

```
时间步:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
        ────────────────────────────────────────────────
Layer 1: ●  ●  ●
         ●  ●  ●
         ●  ●  ●
         ... (dilation=1, 每个位置看3个时间步)

Layer 2: ●     ●     ●
         ●     ●     ●
         ●     ●     ●
         ... (dilation=2, 每个位置看5个时间步)

位置8的感受野:
  Layer 2位置8 ← Layer 1位置 [6, 8, 10]
    ↓           ↓           ↓
  Layer 1位置6 ← [5, 6, 7]
  Layer 1位置8 ← [7, 8, 9]
  Layer 1位置10 ← [9, 10, 11]

因此位置8可以看到: [5, 6, 7, 8, 9, 10, 11] (7个时间步)
```

---

## 6. 优势与局限

### 优势 ✅
1. **计算效率高**：卷积操作可以并行，比RNN快
2. **稳定训练**：残差连接帮助梯度传播
3. **长距离依赖**：通过多层膨胀卷积扩大感受野
4. **因果性保证**：天然适合时序预测任务
5. **参数较少**：相比Transformer，参数更少

### 局限 ⚠️
1. **感受野受限**：需要多层才能获得大感受野
2. **固定模式**：膨胀率固定为2的幂次，可能不够灵活
3. **短序列优势**：对于非常长的序列，Transformer可能更好

---

## 7. 代码实现要点

### 7.1 关键实现细节

```python
# 1. 因果padding计算
padding = (kernel_size - 1) * dilation

# 2. 膨胀卷积
nn.Conv1d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    dilation=dilation,      # 关键：膨胀率
    padding=padding,         # 关键：因果padding
    bias=False
)

# 3. 残差连接
if x.shape[1] == x_out.shape[1]:
    x = x + x_out  # 维度匹配时使用残差
else:
    x = x_out      # 维度不匹配时直接替换
```

### 7.2 为什么使用BatchNorm而不是LayerNorm？

- **BatchNorm1d**：在batch维度上归一化，适合卷积层
- **LayerNorm**：在特征维度上归一化，适合Transformer
- TCN使用卷积，所以用BatchNorm更合适

---

## 8. 总结

TCN通过**因果卷积 + 膨胀卷积 + 残差连接**的组合，实现了高效的时序建模。在我们的项目中，它被用于处理patch内的时序信息，为后续的VQVAE编码和Transformer预测提供更好的特征表示。

**关键设计选择**：
- ✅ 使用TCN而非Self-Attention处理patch内时序
- ✅ 多层膨胀卷积扩大感受野
- ✅ 残差连接稳定训练
- ✅ 因果性保证适合预测任务

