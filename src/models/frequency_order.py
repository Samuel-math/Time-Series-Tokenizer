"""
频率分工正则化（soft 主频 + 平滑排序损失）

核心架构设计：
    X_patch → Encoder → RQ → [z_q^(1), z_q^(2), ..., z_q^(L)]
                           → Decoder → [X_1, X_2, ..., X_L]

    X_l     = Decoder(z_q^(l))          每层码本单独解码出的时间域分量
    x_recon = X_1 + X_2 + ... + X_L    各层叠加 = VQ 重构结果（不含稀疏 s）

频率分工目标：g_1 < g_2 < ... < g_L
    浅层（l=1）重构低频主体，深层（l=L）补充高频细节。

Soft 主频计算（全程可导，无 argmax/hinge）：
    X̃_l = X_l - mean_t(X_l)              去均值（消除 DC）
    X̄_l = X̃_l / (‖X̃_l‖₂ + ε)          能量归一化
    P_l(k) = |FFT(X̄_l)_k|²               功率谱（丢弃 k=0 的 DC bin）
    π_l(k) = softmax(P_l(k) / τ_f)       soft peak pooling
    g_l    = Σ_k ω̄_k · π_l(k)            soft 主频分数，ω̄_k ∈ (0, 1]

排序损失（softplus 替代 hinge，全可导）：
    L_order = mean_{l=1}^{L-1} softplus(g_l - g_{l+1})
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F


def decode_per_layer(
    per_layer_z_q: List[torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    把每层码本向量堆成一个大 batch，**一次 Decoder 前向**同时得到：
      - x_components[l] = Decoder(z_q^(l))    每层单独解码出的时间域分量
      - x_recon          = Σ_l x_components[l]  各层叠加 = VQ 重构结果

    与 Decoder(z_q_sum) 的区别：
      这里是先分别解码再相加，而非先相加码本向量再解码。
      这样每层解码分量 X_l 具有独立的物理意义，可以直接做频率分析。

    Args:
        per_layer_z_q: List[L] of [B, num_patches, C, code_dim]
        decode_fn:     model.decode_from_codes，[B, P, C, D] → [B, T, C]
    Returns:
        x_components: List[L] of [B, T, C]
        x_recon:      [B, T, C]  = sum(x_components)，送入 recon_loss 使用
    """
    if not per_layer_z_q:
        raise ValueError("per_layer_z_q is empty")
    L = len(per_layer_z_q)
    B, P, C, D = per_layer_z_q[0].shape

    # 所有层堆成一个大 batch：[L*B, P, C, D]
    stacked = torch.stack(per_layer_z_q, dim=0)   # [L, B, P, C, D]
    batched = stacked.reshape(L * B, P, C, D)
    decoded = decode_fn(batched)                   # [L*B, T, C]
    T = decoded.shape[1]
    decoded = decoded.reshape(L, B, T, C)

    x_components = [decoded[l] for l in range(L)]
    x_recon = decoded.sum(dim=0)                   # [B, T, C]
    return x_components, x_recon


def _soft_dominant_frequency(
    x: torch.Tensor,
    tau_f: float,
    eps: float,
) -> torch.Tensor:
    """
    单层分量 X_l → 标量 g_l（对 B、C 维求均值后的 soft 主频分数）

    Args:
        x:  [B, T, C]
    Returns:
        g: 0-dim tensor, 值域 (0, 1]
    """
    # 1) 时间维去均值（消除 DC，之后 FFT 的 k=0 接近零，直接丢弃）
    x_tilde = x - x.mean(dim=1, keepdim=True)
    # 2) 能量归一化（每个 (B, C) 独立，避免能量差异主导频率分数）
    norm = x_tilde.norm(p=2, dim=1, keepdim=True) + eps
    x_bar = x_tilde / norm

    # 3) rfft 取正频，丢弃 DC bin（k=0）
    spec = torch.fft.rfft(x_bar, dim=1)                      # [B, K, C]
    P = (spec.real ** 2 + spec.imag ** 2)[:, 1:, :]          # [B, K-1, C]
    K = P.shape[1]
    if K == 0:
        return x.new_tensor(0.0)

    # 4) 归一化频率坐标 ω̄_k ∈ (0, 1]
    omega = torch.arange(1, K + 1, device=P.device, dtype=P.dtype) / K  # [K]

    # 5) soft peak pooling：softmax over 频率维
    pi = torch.softmax(P / tau_f, dim=1)                      # [B, K, C]

    # 6) soft 主频分数，对 batch / channel 均值得到标量
    g = (pi * omega.view(1, -1, 1)).sum(dim=1)                # [B, C]
    return g.mean()


def compute_frequency_order_loss(
    x_components: List[torch.Tensor],
    tau_f: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    计算频率排序损失，要求 g_1 < g_2 < ... < g_L。

    Args:
        x_components: List[L] of [B, T, C]，每层单独解码出的时间域分量
        tau_f:        softmax 温度（越小越接近 hard peak，建议 0.3 ~ 1.0）
        eps:          能量归一化 eps
    Returns:
        L_order:  scalar（≥ 0，越小说明频率分工越好）
        g_list:   List[L] of scalar，每层 soft 主频分数
        gap_list: List[L-1] of scalar，g_{l+1} - g_l（正值说明分工正确）
    """
    g_list = [
        _soft_dominant_frequency(x, tau_f=tau_f, eps=eps)
        for x in x_components
    ]

    L = len(g_list)
    if L <= 1:
        zero = x_components[0].new_tensor(0.0) if L == 1 else torch.tensor(0.0)
        return zero, g_list, []

    gap_list = [g_list[l + 1] - g_list[l] for l in range(L - 1)]
    penalties = torch.stack([
        F.softplus(g_list[l] - g_list[l + 1]) for l in range(L - 1)
    ])
    L_order = penalties.mean()
    return L_order, g_list, gap_list
