"""
频率分工正则化（soft 主频 + 平滑排序损失）

用于多层 VQ / RQ-VAE：约束从浅到深的码本层在频率上逐渐升高
（g_1 < g_2 < ... < g_L），以鼓励不同层分担不同频带。

设计要点：
- 不手动划分 high/low band；也不指定具体频带。
- 全程使用可导平滑操作（softmax、softplus、FFT），没有 argmax/max/hinge。
- 对任意层数 L 通用；不依赖各层 codebook size 是否相同。

主要算法（每层独立计算 soft 主频 g_l）：
    Δr^(l) = Decoder(z_q^(l))                   (单层码本独立解码出的时间域分量)
    r~_l  = Δr^(l) - mean_t(Δr^(l))
    r_bar = r~_l / (||r~_l||_2 + eps)
    P_l(k) = |FFT(r_bar)_k|^2                   (仅正频)
    π_l(k) = softmax(P_l(k) / τ_f)              (soft peak pooling)
    g_l    = Σ_k ω̄_k · π_l(k)                   (ω̄_k ∈ (0, 1])

排序损失：
    L_order = (1 / (L - 1)) · Σ_l softplus(g_l - g_{l+1})
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F


def compute_per_layer_deltas(
    per_layer_z_q: List[torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    return_recon: bool = False,
):
    """
    计算每层码本单独解码出的时间域分量 Δr^(l) = Decoder(z_q^(l))。

    实现上把 [Σ z_q, z_q^(1), ..., z_q^(L)] 堆到一个大 batch，
    **共用一次 Decoder 前向** 同时产出 x_recon 和所有 Δr^(l)，
    训练循环可以省掉单独的 reconstruction decode。

    Args:
        per_layer_z_q: List[L] of [B, num_patches, C, code_dim]，每层单独的量化向量
        decode_fn:     model.decode_from_codes，签名 [B, P, C, code_dim] → [B, T, C]
        return_recon:  若为 True，额外返回 x_recon = Decoder(Σ z_q^(l))（免费复用）
    Returns:
        delta_r_list: List[L] of [B, T, C]
        x_recon (optional): [B, T, C]
    """
    if not per_layer_z_q:
        raise ValueError("per_layer_z_q is empty")
    L = len(per_layer_z_q)
    B, P, C, D = per_layer_z_q[0].shape

    # 堆叠 [Σ z_q, z_q^(1), ..., z_q^(L)] 一次解码
    z_sum = torch.stack(per_layer_z_q, dim=0).sum(dim=0)       # [B, P, C, D]
    stacked = torch.stack([z_sum] + list(per_layer_z_q), dim=0)  # [L+1, B, P, C, D]
    batched = stacked.reshape((L + 1) * B, P, C, D)
    decoded = decode_fn(batched)                                # [(L+1)*B, T, C]
    T = decoded.shape[1]
    decoded = decoded.reshape(L + 1, B, T, C)
    x_recon = decoded[0]
    delta_r_list = [decoded[l + 1] for l in range(L)]

    if return_recon:
        return delta_r_list, x_recon
    return delta_r_list


def _soft_dominant_frequency(
    dr: torch.Tensor,
    tau_f: float,
    eps: float,
) -> torch.Tensor:
    """
    单层 Δr^(l) -> 标量 g_l（对 B、C 求均值）

    Args:
        dr:  [B, T, C]
    Returns:
        g: 0-dim tensor, 值域 (0, 1]
    """
    # 1) 时间维去均值
    dr_tilde = dr - dr.mean(dim=1, keepdim=True)
    # 2) 能量归一化（每个 (B, C) 独立；避免能量差异主导频率分数）
    norm = dr_tilde.norm(p=2, dim=1, keepdim=True) + eps
    dr_bar = dr_tilde / norm

    # 3) rfft 只取正频；去均值后 DC(k=0) 为 0，直接丢弃
    spec = torch.fft.rfft(dr_bar, dim=1)        # [B, K, C] complex
    P = (spec.real ** 2 + spec.imag ** 2)[:, 1:, :]  # 丢弃 DC，[B, K-1, C]
    K = P.shape[1]
    if K == 0:
        return dr.new_tensor(0.0)

    # 4) 归一化频率坐标 ω̄_k ∈ (0, 1]
    omega = torch.arange(1, K + 1, device=P.device, dtype=P.dtype) / K  # [K]

    # 5) soft peak pooling
    pi = torch.softmax(P / tau_f, dim=1)         # [B, K, C]

    # 6) soft 主频分数
    g = (pi * omega.view(1, -1, 1)).sum(dim=1)   # [B, C]
    return g.mean()


def compute_frequency_order_loss(
    delta_r_list: List[torch.Tensor],
    tau_f: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Args:
        delta_r_list: List[L] of [B, T, C]
        tau_f:   softmax 温度（越小越接近 hard peak）
        eps:     能量归一化 eps
    Returns:
        L_order: scalar（≥ 0）
        g_list:  List[L] of scalar，每层的 soft 主频分数
        gap_list: List[L-1] of scalar，相邻层的 g_{l+1} - g_l
    """
    g_list: List[torch.Tensor] = [
        _soft_dominant_frequency(dr, tau_f=tau_f, eps=eps)
        for dr in delta_r_list
    ]

    L = len(g_list)
    if L <= 1:
        zero = delta_r_list[0].new_tensor(0.0) if L == 1 else torch.tensor(0.0)
        return zero, g_list, []

    gap_list = [g_list[l + 1] - g_list[l] for l in range(L - 1)]
    penalties = torch.stack([F.softplus(g_list[l] - g_list[l + 1]) for l in range(L - 1)])
    L_order = penalties.mean()
    return L_order, g_list, gap_list
