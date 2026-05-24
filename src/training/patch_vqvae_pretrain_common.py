"""
Patch VQVAE + Transformer 渐进式预训练公共逻辑
（decoder_only_NTP / decoder_only_forcasting 共用）

入口脚本只需:
    sys.path.insert(0, repo_root)
    from src.training.patch_vqvae_pretrain_common import run_pretrain
    run_pretrain()
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datautils import get_dls
from src.models.patch_vqvae_transformer import (
    PatchVQVAETransformer, FlattenedVectorQuantizerEMA, get_model_config,
)
from src.models.layers.revin import RevIN


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(description='Patch VQVAE Transformer 渐进式预训练')

    # 数据
    p.add_argument('--dset', type=str, default='ettm1')
    p.add_argument('--context_points', type=int, default=512)
    p.add_argument('--progressive_step_size', type=int, required=True,
                   help='渐进式预训练步长（patches）')
    p.add_argument('--progressive_max_stages', type=int, default=None,
                   help='最大阶段数，None 表示全部')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--scaler', type=str, default='standard')
    p.add_argument('--features', type=str, default='M')
    p.add_argument('--channel_start', type=int, default=None,
                   help='只使用变量维度中的起始 channel（包含）；None 表示从 0 开始')
    p.add_argument('--channel_end', type=int, default=None,
                   help='只使用变量维度中的结束 channel（不包含）；None 表示到最后')
    p.add_argument('--channel_indices', type=str, default=None,
                   help='逗号分隔的任意 channel 索引列表；设置后优先于 channel_start/end')
    p.add_argument('--channel_group_id', type=int, default=None,
                   help='任意 channel_indices 分组时用于 checkpoint 后缀的 group id')

    # 模型结构
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--embedding_dim', type=int, default=32)
    p.add_argument('--compression_factor', type=int, default=4)
    p.add_argument('--codebook_size', type=int, default=256)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--d_ff', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--transformer_hidden_dim', type=int, default=None)
    p.add_argument('--temporal_backbone', type=str, default='causal_transformer',
                   choices=['causal_transformer', 'encoder_transformer', 'encoder_timefilter_lite',
                            'encoder_cluster_timefilter_lite', 'relative_transformer', 'itransformer_lite',
                            'timefilter_lite', 'cluster_timefilter_lite', 'timefilter_attn',
                            'channel_summary_adapter'],
                   help='NTP temporal backbone: causal_transformer=旧CI Transformer, '
                        'encoder_transformer=非因果Transformer Encoder, '
                        'encoder_timefilter_lite=非因果Encoder+timefilter_lite, '
                        'encoder_cluster_timefilter_lite=非因果Encoder+cluster_timefilter_lite, '
                        'relative_transformer=相对位置偏置的因果Transformer, '
                        'itransformer_lite=时间因果建模+通道attention, '
                        'timefilter_lite=时间因果建模+patch-specific通道图过滤, '
                        'cluster_timefilter_lite=在channel cluster内做timefilter_lite, '
                        'timefilter_attn=多头channel attention版timefilter_lite, '
                        'channel_summary_adapter=滞后跨通道均值条件分支')
    p.add_argument('--channel_mixer_heads', type=int, default=None,
                   help='itransformer_lite 的 channel attention heads；None=沿用 n_heads')
    p.add_argument('--timefilter_topk', type=int, default=8,
                   help='timefilter_lite 每个 patch/channel 保留的通道邻居数')
    p.add_argument('--timefilter_temperature', type=float, default=1.0,
                   help='timefilter_lite 通道 affinity softmax 温度')
    p.add_argument('--timefilter_attn_heads', type=int, default=None,
                   help='timefilter_attn 的 channel attention heads；None=沿用 n_heads')
    p.add_argument('--channel_summary_window', type=int, default=4,
                   help='channel_summary_adapter 使用的历史 patch 窗口 W')
    p.add_argument('--channel_summary_gate_init', type=float, default=-4.0,
                   help='channel_summary_adapter gate 初始化值；-4 约等于 0.018')
    p.add_argument('--commitment_cost', type=float, default=0.25)
    p.add_argument('--codebook_ema', type=int, default=1)
    p.add_argument('--disable_ema_update', type=int, default=1,
                   help='禁用 EMA 更新（1=禁用，用于稳定 recon_loss）')
    p.add_argument('--ema_decay', type=float, default=0.99)
    p.add_argument('--ema_eps', type=float, default=1e-5)
    p.add_argument('--num_hiddens', type=int, default=64)
    p.add_argument('--num_residual_layers', type=int, default=2)
    p.add_argument('--num_residual_hiddens', type=int, default=32)
    p.add_argument('--vqvae_backbone', type=str, default='mlp',
                   help='VQVAE Encoder/Decoder backbone: mlp=旧结构, linear=单层线性结构, conv_linear=一层卷积+线性投影, tcn=Conv1d/TCN结构, chunk_mlp=分块Linear结构')
    p.add_argument('--vqvae_tcn_kernel_size', type=int, default=5,
                   help='TCN backbone 的 Conv1d kernel size（需为奇数；仅 vqvae_backbone=tcn 时使用）')
    p.add_argument('--vqvae_chunk_size', type=int, default=2,
                   help='chunk_mlp backbone 的 patch 分块大小（需整除 patch_size）')
    p.add_argument('--decoder_lowpass', type=int, default=0,
                   help='1=在 VQVAE decoder 输出末尾应用低通滤波（核类型由 --decoder_lowpass_kernel 决定；默认关闭）')
    p.add_argument('--decoder_lowpass_kernel', type=str, default='binomial3',
                   choices=['binomial3', 'mean3', 'mean5', 'mean7', 'mean9',
                            'binomial5', 'binomial7', 'triangular5'],
                   help='--decoder_lowpass=1 时使用的低通核：binomial3=[1,2,1]/4 (默认，向后兼容)，'
                        'mean*=均匀均值核 (越大越平滑)，binomial5/7=更宽的二项式核，triangular5=[1,2,3,2,1]/9')

    # VQVAE checkpoint
    p.add_argument('--vqvae_checkpoint', type=str, default=None,
                   help='预训练 VQVAE 路径（可选）')
    p.add_argument('--freeze_vqvae', type=int, default=1,
                   help='加载后冻结 VQVAE（1=冻结）')
    p.add_argument('--load_vq_weights', type=int, default=1,
                   help='是否加载 VQ 层权重（1=加载）')

    # Per-channel 码本
    p.add_argument('--per_channel_codebook', type=int, default=0,
                   help='每通道独立码本（1=启用，需与 vqvae-only 训练一致）')

    # RVQ 层数
    p.add_argument('--n_rq_layers', type=int, default=1,
                   help='残差向量量化层数（1=普通VQ，2=2层RVQ）')
    p.add_argument('--rq_layer_weights', type=float, nargs='+', default=None,
                   help='各 RVQ 层 pred_loss 的权重，顺序对应第0层、第1层…'
                        '（默认 None = 均等权重）。示例: --rq_layer_weights 1.0 0.5')
    p.add_argument('--soft_neighbor_k', type=int, default=0,
                   help='NTP soft label 使用的 codebook 近邻数；0=关闭，等价普通 hard CE')
    p.add_argument('--soft_neighbor_alpha', type=float, default=0.25,
                   help='soft label 分给近邻 code 的总概率质量；仅 soft_neighbor_k>0 时生效')
    p.add_argument('--soft_neighbor_tau', type=float, default=0.3,
                   help='近邻 softmax 温度；仅 soft_neighbor_k>0 时生效')

    # NMPP 模式
    p.add_argument('--use_raw_input', type=int, default=0,
                   help='1: NMPP 模式，Transformer 接收原始 patch，VQVAE 仅作为 teacher')

    # Overlapping chunk prediction（pred_len > step_size 时启用）
    p.add_argument('--pred_len', type=int, default=None,
                   help='每个 stage 预测的 patch 数 N（默认 None = 等于 progressive_step_size）。'
                        'N > M 时产生 overlapping chunk，同一位置的多个预测在 logit 层面融合。')

    # 训练超参
    p.add_argument('--n_epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--revin', type=int, default=1)
    p.add_argument('--vq_weight', type=float, default=1.0)
    p.add_argument('--recon_weight', type=float, default=0.1)

    # 早停
    p.add_argument('--early_stop_patience', type=int, default=5,
                   help='val_loss 连续未显著下降多少 epoch 就早停（默认 5）')
    p.add_argument('--early_stop_warmup', type=int, default=5,
                   help='前多少 epoch 不触发早停（但仍会保存 best model，默认 5）')
    p.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                   help='视为"有效改善"的最小 val_loss 降幅（默认 1e-4）')
    p.add_argument('--early_stop_smooth_k', type=int, default=1,
                   help='用最近 K 个 epoch 的 val_loss 均值做早停判据（K=1 表示不平滑，默认 1）')

    # 保存
    p.add_argument('--save_path', type=str, default='saved_models/patch_vqvae/')
    p.add_argument('--model_id', type=int, default=1)
    p.add_argument('--run_id', type=int, default=None)

    return p


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _channel_suffix(args):
    indices = getattr(args, 'channel_indices', None)
    if indices:
        gid = getattr(args, 'channel_group_id', None)
        return f'_grp{gid if gid is not None else "custom"}'
    start = getattr(args, 'channel_start', None)
    end = getattr(args, 'channel_end', None)
    if start is None and end is None:
        return ''
    return f'_ch{0 if start is None else start}-{end if end is not None else "end"}'


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _soft_neighbor_cross_entropy(logits_l, tgt_l, model, layer_idx, neighbor_k, alpha, tau, neighbor_tables=None):
    """Cross entropy with target probability spread to codebook nearest neighbors.

    全向量化路径（共享码本 + per-channel 码本均一次性处理所有通道）。
    """
    B, P, C, K = logits_l.shape
    if model is None or neighbor_k <= 0 or alpha <= 0:
        return F.cross_entropy(logits_l.reshape(-1, K), tgt_l.reshape(-1))

    neighbor_k = min(int(neighbor_k), K - 1)
    if neighbor_k <= 0:
        return F.cross_entropy(logits_l.reshape(-1, K), tgt_l.reshape(-1))

    alpha = min(max(float(alpha), 0.0), 1.0)
    tau = max(float(tau), 1e-6)
    per_channel = bool(getattr(model, 'per_channel_codebook', False))

    if not per_channel:
        logits_flat = logits_l.reshape(-1, K)
        tgt_flat = tgt_l.reshape(-1)
        logp = F.log_softmax(logits_flat, dim=-1)
        if neighbor_tables is not None:
            neighbor_idx, neighbor_prob = neighbor_tables[(layer_idx, 0)]
        else:
            weight = _get_codebook_weight(model, layer_idx, 0)
            dist = torch.cdist(weight.float(), weight.float(), p=2)
            dist = dist.masked_fill(torch.eye(K, device=dist.device, dtype=torch.bool), float('inf'))
            neighbor_dist, neighbor_idx = torch.topk(dist, k=neighbor_k, dim=-1, largest=False)
            neighbor_prob = F.softmax(-neighbor_dist / tau, dim=-1)

        target_neighbors = neighbor_idx[tgt_flat]
        target_prob = neighbor_prob[tgt_flat].to(logp.dtype)
        true_logp = logp.gather(1, tgt_flat.unsqueeze(1)).squeeze(1)
        neighbor_logp = logp.gather(1, target_neighbors)
        return -((1.0 - alpha) * true_logp + alpha * (target_prob * neighbor_logp).sum(dim=1)).mean()

    # ---- per-channel：一次性处理所有通道，避免 Python 循环 ----
    # 优先用预计算的 stacked 表
    stacked = neighbor_tables.get((layer_idx, 'stacked')) if neighbor_tables is not None else None
    if stacked is None:
        if neighbor_tables is not None:
            idx_list = [neighbor_tables[(layer_idx, c)][0] for c in range(C)]
            prob_list = [neighbor_tables[(layer_idx, c)][1] for c in range(C)]
        else:
            idx_list, prob_list = [], []
            for c in range(C):
                weight = _get_codebook_weight(model, layer_idx, c)
                dist = torch.cdist(weight.float(), weight.float(), p=2)
                dist = dist.masked_fill(torch.eye(K, device=dist.device, dtype=torch.bool), float('inf'))
                neighbor_dist, neighbor_idx = torch.topk(dist, k=neighbor_k, dim=-1, largest=False)
                idx_list.append(neighbor_idx)
                prob_list.append(F.softmax(-neighbor_dist / tau, dim=-1))
        idx_stack = torch.stack(idx_list, dim=0)    # [C, K, k]
        prob_stack = torch.stack(prob_list, dim=0)  # [C, K, k]
    else:
        idx_stack, prob_stack = stacked

    # [C, B*P, K]
    logits_perm = logits_l.permute(2, 0, 1, 3).reshape(C, -1, K).contiguous()
    tgt_perm = tgt_l.permute(2, 0, 1).reshape(C, -1).contiguous()  # [C, B*P]
    logp = F.log_softmax(logits_perm, dim=-1)

    # gather neighbors per channel: [C, B*P, k]
    tgt_idx_expand = tgt_perm.unsqueeze(-1).expand(-1, -1, neighbor_k)
    target_neighbors = idx_stack.gather(1, tgt_idx_expand)
    target_prob = prob_stack.gather(1, tgt_idx_expand).to(logp.dtype)

    true_logp = logp.gather(2, tgt_perm.unsqueeze(-1)).squeeze(-1)        # [C, B*P]
    neighbor_logp = logp.gather(2, target_neighbors)                       # [C, B*P, k]
    loss = -((1.0 - alpha) * true_logp + alpha * (target_prob * neighbor_logp).sum(dim=-1))
    return loss.mean()


def _build_soft_neighbor_tables(model, n_layers, n_channels, codebook_size, neighbor_k, tau):
    """Precompute codebook nearest-neighbor tables once per epoch.

    per-channel 模式下会额外在 ``(l, 'stacked')`` 处保存沿通道堆叠后的张量，
    供 :func:`_soft_neighbor_cross_entropy` 一次性处理所有通道。
    """
    neighbor_k = min(int(neighbor_k), codebook_size - 1)
    if model is None or neighbor_k <= 0:
        return None

    tau = max(float(tau), 1e-6)
    per_channel = bool(getattr(model, 'per_channel_codebook', False))
    tables = {}
    with torch.no_grad():
        for l in range(n_layers):
            idx_list, prob_list = [], []
            for c in range(n_channels):
                weight = _get_codebook_weight(model, l, c)
                dist = torch.cdist(weight.float(), weight.float(), p=2)
                dist = dist.masked_fill(
                    torch.eye(codebook_size, device=dist.device, dtype=torch.bool),
                    float('inf'),
                )
                neighbor_dist, neighbor_idx = torch.topk(dist, k=neighbor_k, dim=-1, largest=False)
                neighbor_prob = F.softmax(-neighbor_dist / tau, dim=-1)
                tables[(l, c)] = (neighbor_idx, neighbor_prob)
                idx_list.append(neighbor_idx)
                prob_list.append(neighbor_prob)
            if per_channel and idx_list:
                tables[(l, 'stacked')] = (
                    torch.stack(idx_list, dim=0),
                    torch.stack(prob_list, dim=0),
                )
    return tables


def _build_semantic_rank_tables(model, n_layers, n_channels, codebook_size):
    """Precompute rank[pred] among each true code's nearest codebook entries."""
    tables = {}
    with torch.no_grad():
        for l in range(n_layers):
            for c in range(n_channels):
                weight = _get_codebook_weight(model, l, c)
                dist = torch.cdist(weight.float(), weight.float(), p=2)
                order = dist.argsort(dim=-1)
                ranks = torch.empty_like(order)
                rank_values = torch.arange(1, codebook_size + 1, device=dist.device).expand_as(order)
                ranks.scatter_(dim=1, index=order, src=rank_values)
                tables[(l, c)] = ranks
    return tables


def _progressive_loss(
    all_logits, all_target_indices, rq_layer_weights=None,
    model=None, soft_neighbor_k=0, soft_neighbor_alpha=0.25, soft_neighbor_tau=0.3,
    neighbor_tables=None,
):
    """
    all_logits:        List[stage] of List[rq_layer] of [B, step_size, C, codebook_size]
    all_target_indices: List[stage] of List[rq_layer] of [B, step_size, C]
    rq_layer_weights:  List[float] | None — 各 RVQ 层的损失权重（None 表示均等）
    """
    n_layers = len(all_logits[0])
    if rq_layer_weights is None:
        weights = [1.0] * n_layers
    else:
        if len(rq_layer_weights) != n_layers:
            raise ValueError(
                f"rq_layer_weights 长度 ({len(rq_layer_weights)}) 与 RVQ 层数 ({n_layers}) 不匹配"
            )
        weights = list(rq_layer_weights)

    total_loss = 0.0
    total_weight = 0.0
    for logits_layers, tgt_layers in zip(all_logits, all_target_indices):
        for l, (logits_l, tgt_l) in enumerate(zip(logits_layers, tgt_layers)):
            total_loss += weights[l] * _soft_neighbor_cross_entropy(
                logits_l, tgt_l, model, l,
                soft_neighbor_k, soft_neighbor_alpha, soft_neighbor_tau,
                neighbor_tables=neighbor_tables,
            )
            total_weight += weights[l]

    return total_loss / (total_weight * len(all_logits) / n_layers)


def _get_codebook_weight(model, layer_idx, channel_idx=None):
    """Return the codebook embedding weight for one RVQ layer."""
    if getattr(model, 'per_channel_codebook', False):
        return model.vqs[channel_idx].layers[layer_idx].embedding.weight
    return model.vq.layers[layer_idx].embedding.weight


def _progressive_accuracy(all_logits, all_target_indices):
    """Lightweight fast path: only top-1 accuracy (avg + per-layer)."""
    n_layers = len(all_logits[0])
    correct = [0] * n_layers
    total = [0] * n_layers

    with torch.no_grad():
        for logits_layers, tgt_layers in zip(all_logits, all_target_indices):
            for l, (logits_l, tgt_l) in enumerate(zip(logits_layers, tgt_layers)):
                pred_l = logits_l.argmax(dim=-1)
                correct[l] += (pred_l == tgt_l).sum().item()
                total[l] += tgt_l.numel()

    layer_acc = [
        (correct[l] / total[l]) if total[l] > 0 else 0.0
        for l in range(n_layers)
    ]
    avg_acc = sum(correct) / sum(total) if sum(total) > 0 else 0.0
    return avg_acc, layer_acc


def _progressive_token_diagnostics(
    all_logits, all_target_indices, model=None,
    include_semantic_rank=False, semantic_rank_tables=None,
):
    """统计 NMPP token 预测诊断：top-k、真实 code rank，以及可选的 codebook 近邻 rank。

    所有计数器在 GPU 上累加，函数末尾一次性同步回 CPU，避免每个 batch 多次 GPU↔CPU sync。
    """
    n_layers = len(all_logits[0])
    device = all_logits[0][0].device
    topk_levels = (1, 3, 5, 10)
    n_topk = len(topk_levels)

    correct = torch.zeros(n_layers, device=device, dtype=torch.long)
    total = torch.zeros(n_layers, device=device, dtype=torch.long)
    layer_topk_hits = torch.zeros(n_layers, n_topk, device=device, dtype=torch.long)
    layer_rank_sum = torch.zeros(n_layers, device=device, dtype=torch.float64)
    layer_rank_values = [[] for _ in range(n_layers)]

    layer_sem_sum = torch.zeros(n_layers, device=device, dtype=torch.float64)
    layer_sem_count = torch.zeros(n_layers, device=device, dtype=torch.long)

    per_channel = bool(getattr(model, 'per_channel_codebook', False)) if model is not None else False

    with torch.no_grad():
        for logits_layers, tgt_layers in zip(all_logits, all_target_indices):
            for l, (logits_l, tgt_l) in enumerate(zip(logits_layers, tgt_layers)):
                B, P, C, K = logits_l.shape
                pred_l = logits_l.argmax(dim=-1)
                correct[l] += (pred_l == tgt_l).sum()
                total[l] += tgt_l.numel()

                flat_logits = logits_l.reshape(-1, K)
                flat_tgt = tgt_l.reshape(-1)
                true_score = flat_logits.gather(1, flat_tgt.unsqueeze(1))
                ranks = (flat_logits > true_score).sum(dim=1) + 1  # [N], int64
                ranks_f = ranks.to(torch.float64)

                layer_rank_sum[l] += ranks_f.sum()
                layer_rank_values[l].append(ranks)
                for ki, k in enumerate(topk_levels):
                    layer_topk_hits[l, ki] += (ranks <= min(k, K)).sum()

                if include_semantic_rank and semantic_rank_tables is not None:
                    if per_channel:
                        # 每个通道用各自的 table；尽量减少 .item() 调用
                        for c in range(C):
                            pred_c = pred_l[:, :, c].reshape(-1)
                            tgt_c = tgt_l[:, :, c].reshape(-1)
                            sem = semantic_rank_tables[(l, c)][tgt_c, pred_c]
                            sem_f = sem.to(torch.float64)
                            layer_sem_sum[l] += sem_f.sum()
                            layer_sem_count[l] += sem.numel()
                    else:
                        # 共享码本：一次 gather 拿下所有通道
                        table = semantic_rank_tables[(l, 0)]
                        sem_all = table[flat_tgt, pred_l.reshape(-1)]
                        layer_sem_sum[l] += sem_all.to(torch.float64).sum()
                        layer_sem_count[l] += sem_all.numel()

    # ---- 一次性同步到 CPU ----
    correct_cpu = correct.cpu().tolist()
    total_cpu = total.cpu().tolist()
    layer_topk_hits_cpu = layer_topk_hits.cpu().tolist()
    layer_rank_sum_cpu = layer_rank_sum.cpu().tolist()
    layer_sem_sum_cpu = layer_sem_sum.cpu().tolist()
    layer_sem_count_cpu = layer_sem_count.cpu().tolist()

    total_tokens = sum(total_cpu)
    rank_sum_v = sum(layer_rank_sum_cpu)
    sem_sum_v = sum(layer_sem_sum_cpu)
    sem_count_v = sum(layer_sem_count_cpu)
    topk_hits_total = [sum(layer_topk_hits_cpu[l][ki] for l in range(n_layers)) for ki in range(n_topk)]

    layer_acc = [
        (correct_cpu[l] / total_cpu[l]) if total_cpu[l] > 0 else 0.0
        for l in range(n_layers)
    ]
    avg_acc = (sum(correct_cpu) / total_tokens) if total_tokens > 0 else 0.0

    # Median: 每层一次 cat + median，整体一次 cat + median；都在 GPU 上算
    layer_median_rank = []
    all_ranks = []
    for vals in layer_rank_values:
        if vals:
            cat = torch.cat(vals)
            all_ranks.append(cat)
            layer_median_rank.append(cat.float().median().item())
        else:
            layer_median_rank.append(0.0)
    median_rank = torch.cat(all_ranks).float().median().item() if all_ranks else 0.0

    return {
        'token_acc': avg_acc,
        'layer_acc': layer_acc,
        'topk_acc': {
            topk_levels[ki]: (topk_hits_total[ki] / total_tokens if total_tokens > 0 else 0.0)
            for ki in range(n_topk)
        },
        'mean_rank': rank_sum_v / total_tokens if total_tokens > 0 else 0.0,
        'median_rank': median_rank,
        'layer_topk_acc': [
            {
                topk_levels[ki]: (layer_topk_hits_cpu[l][ki] / total_cpu[l] if total_cpu[l] > 0 else 0.0)
                for ki in range(n_topk)
            }
            for l in range(n_layers)
        ],
        'layer_mean_rank': [
            layer_rank_sum_cpu[l] / total_cpu[l] if total_cpu[l] > 0 else 0.0
            for l in range(n_layers)
        ],
        'layer_median_rank': layer_median_rank,
        'semantic_neighbor_rank': (
            sem_sum_v / sem_count_v if sem_count_v > 0 else None
        ),
        'layer_semantic_neighbor_rank': [
            (
                layer_sem_sum_cpu[l] / layer_sem_count_cpu[l]
                if layer_sem_count_cpu[l] > 0 else None
            )
            for l in range(n_layers)
        ],
    }


# ---------------------------------------------------------------------------
# Train / validate epochs
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scheduler, revin, args, device, trainable_params):
    model.train()

    use_raw = bool(args.use_raw_input)
    compute_recon = args.recon_weight > 0 and not use_raw
    vq_w    = 0. if use_raw else args.vq_weight
    recon_w = 0. if use_raw else args.recon_weight
    rq_weights = getattr(args, 'rq_layer_weights', None)
    pred_len    = getattr(args, 'pred_len', None)            # N；None → 等于 step_size
    step_size   = args.progressive_step_size
    soft_neighbor_k = int(getattr(args, 'soft_neighbor_k', 0))
    soft_neighbor_alpha = float(getattr(args, 'soft_neighbor_alpha', 0.25))
    soft_neighbor_tau = float(getattr(args, 'soft_neighbor_tau', 0.3))
    compute_diag = soft_neighbor_k > 0
    n_channels = (getattr(model, '_n_channels', None) or 1) if getattr(model, 'per_channel_codebook', False) else 1
    soft_neighbor_tables = _build_soft_neighbor_tables(
        model, model.n_rq_layers, n_channels, model.codebook_size, soft_neighbor_k, soft_neighbor_tau,
    ) if soft_neighbor_k > 0 and soft_neighbor_alpha > 0 else None

    if compute_diag:
        totals = dict(
            loss=0., pred_loss=0., vq_loss=0., recon_loss=0.,
            token_acc=0., top3_acc=0., top5_acc=0., top10_acc=0.,
            mean_rank=0., median_rank=0.,
        )
        layer_acc_sum = None
        layer_topk_sum = None
        layer_mean_rank_sum = None
        layer_median_rank_sum = None
    else:
        totals = dict(loss=0., pred_loss=0., vq_loss=0., recon_loss=0., token_acc=0.)
        layer_acc_sum = None
    n = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        if revin:
            # 用 batch_x 的 stats 同时归一化 batch_x 和 batch_y，使拼接后的序列
            # 和推理/finetune 行为一致（只用 context 的 stats）。
            # 注意：直接调用两次 revin(_, 'norm') 会用各自的 stats 覆盖存储，导致
            # 两段在不同归一化空间下拼接，产生边界不连续，pretrain/inference 分布失配。
            batch_x = revin(batch_x, 'norm')         # 存 stats(batch_x)
            batch_y = revin._normalize(batch_y)      # 复用 batch_x 的 stats

        batch_full = torch.cat([batch_x, batch_y], dim=1)
        all_logits, all_tgt, vq_loss, recon_loss = model.forward_progressive_pretrain(
            batch_full,
            step_size=step_size,
            max_stages=args.progressive_max_stages,
            compute_recon_loss=compute_recon,
            use_raw_input=use_raw,
            pred_len=pred_len,
        )
        pred_loss = _progressive_loss(
            all_logits, all_tgt, rq_weights,
            model=model,
            soft_neighbor_k=soft_neighbor_k,
            soft_neighbor_alpha=soft_neighbor_alpha,
            soft_neighbor_tau=soft_neighbor_tau,
            neighbor_tables=soft_neighbor_tables,
        )

        loss = pred_loss + vq_w * vq_loss + recon_w * recon_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        totals['loss']       += loss.item()
        totals['pred_loss']  += pred_loss.item()
        totals['vq_loss']    += vq_loss.item()
        totals['recon_loss'] += recon_loss.item()

        if compute_diag:
            token_diag = _progressive_token_diagnostics(all_logits, all_tgt)
            layer_acc = token_diag['layer_acc']
            totals['token_acc']  += token_diag['token_acc']
            totals['top3_acc']   += token_diag['topk_acc'][3]
            totals['top5_acc']   += token_diag['topk_acc'][5]
            totals['top10_acc']  += token_diag['topk_acc'][10]
            totals['mean_rank']  += token_diag['mean_rank']
            totals['median_rank'] += token_diag['median_rank']
            if layer_acc_sum is None:
                layer_acc_sum = [0.0] * len(layer_acc)
                layer_topk_sum = [{3: 0.0, 5: 0.0, 10: 0.0} for _ in layer_acc]
                layer_mean_rank_sum = [0.0] * len(layer_acc)
                layer_median_rank_sum = [0.0] * len(layer_acc)
            for i, acc in enumerate(layer_acc):
                layer_acc_sum[i] += acc
                layer_topk_sum[i][3] += token_diag['layer_topk_acc'][i][3]
                layer_topk_sum[i][5] += token_diag['layer_topk_acc'][i][5]
                layer_topk_sum[i][10] += token_diag['layer_topk_acc'][i][10]
                layer_mean_rank_sum[i] += token_diag['layer_mean_rank'][i]
                layer_median_rank_sum[i] += token_diag['layer_median_rank'][i]
        else:
            token_acc, layer_acc = _progressive_accuracy(all_logits, all_tgt)
            totals['token_acc'] += token_acc
            if layer_acc_sum is None:
                layer_acc_sum = [0.0] * len(layer_acc)
            for i, acc in enumerate(layer_acc):
                layer_acc_sum[i] += acc
        n += 1

    scheduler.step()
    out = {k: v / n for k, v in totals.items()}
    out['layer_acc'] = [v / n for v in layer_acc_sum] if layer_acc_sum is not None else []
    if compute_diag:
        out['layer_topk_acc'] = [
            {k: v / n for k, v in layer_sum.items()}
            for layer_sum in layer_topk_sum
        ] if layer_topk_sum is not None else []
        out['layer_mean_rank'] = [v / n for v in layer_mean_rank_sum] if layer_mean_rank_sum is not None else []
        out['layer_median_rank'] = [v / n for v in layer_median_rank_sum] if layer_median_rank_sum is not None else []
    return out


def validate_epoch(model, dataloader, revin, args, device):
    model.eval()

    use_raw = bool(args.use_raw_input)
    compute_recon = args.recon_weight > 0 and not use_raw
    vq_w    = 0. if use_raw else args.vq_weight
    recon_w = 0. if use_raw else args.recon_weight
    rq_weights  = getattr(args, 'rq_layer_weights', None)
    pred_len    = getattr(args, 'pred_len', None)
    step_size   = args.progressive_step_size
    soft_neighbor_k = int(getattr(args, 'soft_neighbor_k', 0))
    soft_neighbor_alpha = float(getattr(args, 'soft_neighbor_alpha', 0.25))
    soft_neighbor_tau = float(getattr(args, 'soft_neighbor_tau', 0.3))
    compute_diag = soft_neighbor_k > 0
    n_channels = (getattr(model, '_n_channels', None) or 1) if getattr(model, 'per_channel_codebook', False) else 1
    soft_neighbor_tables = _build_soft_neighbor_tables(
        model, model.n_rq_layers, n_channels, model.codebook_size, soft_neighbor_k, soft_neighbor_tau,
    ) if soft_neighbor_k > 0 and soft_neighbor_alpha > 0 else None
    semantic_rank_tables = _build_semantic_rank_tables(
        model, model.n_rq_layers, n_channels, model.codebook_size
    ) if compute_diag else None

    if compute_diag:
        totals = dict(
            loss=0., pred_loss=0., vq_loss=0., recon_loss=0.,
            token_acc=0., top3_acc=0., top5_acc=0., top10_acc=0.,
            mean_rank=0., median_rank=0., semantic_neighbor_rank=0.,
        )
        layer_acc_sum = None
        layer_topk_sum = None
        layer_mean_rank_sum = None
        layer_median_rank_sum = None
        layer_semantic_rank_sum = None
        layer_semantic_batches = None
        semantic_batches = 0
    else:
        totals = dict(loss=0., pred_loss=0., vq_loss=0., recon_loss=0., token_acc=0.)
        layer_acc_sum = None
    n = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if revin:
                batch_x = revin(batch_x, 'norm')     # 存 stats(batch_x)
                batch_y = revin._normalize(batch_y)  # 复用 batch_x 的 stats

            batch_full = torch.cat([batch_x, batch_y], dim=1)
            all_logits, all_tgt, vq_loss, recon_loss = model.forward_progressive_pretrain(
                batch_full,
                step_size=step_size,
                max_stages=args.progressive_max_stages,
                compute_recon_loss=compute_recon,
                use_raw_input=use_raw,
                pred_len=pred_len,
            )
            pred_loss = _progressive_loss(
                all_logits, all_tgt, rq_weights,
                model=model,
                soft_neighbor_k=soft_neighbor_k,
                soft_neighbor_alpha=soft_neighbor_alpha,
                soft_neighbor_tau=soft_neighbor_tau,
                neighbor_tables=soft_neighbor_tables,
            )
            loss = pred_loss + vq_w * vq_loss + recon_w * recon_loss

            totals['loss']       += loss.item()
            totals['pred_loss']  += pred_loss.item()
            totals['vq_loss']    += vq_loss.item()
            totals['recon_loss'] += recon_loss.item()

            if compute_diag:
                token_diag = _progressive_token_diagnostics(
                    all_logits, all_tgt, model=model, include_semantic_rank=True,
                    semantic_rank_tables=semantic_rank_tables,
                )
                layer_acc = token_diag['layer_acc']
                totals['token_acc']  += token_diag['token_acc']
                totals['top3_acc']   += token_diag['topk_acc'][3]
                totals['top5_acc']   += token_diag['topk_acc'][5]
                totals['top10_acc']  += token_diag['topk_acc'][10]
                totals['mean_rank']  += token_diag['mean_rank']
                totals['median_rank'] += token_diag['median_rank']
                if token_diag['semantic_neighbor_rank'] is not None:
                    totals['semantic_neighbor_rank'] += token_diag['semantic_neighbor_rank']
                    semantic_batches += 1
                if layer_acc_sum is None:
                    layer_acc_sum = [0.0] * len(layer_acc)
                    layer_topk_sum = [{3: 0.0, 5: 0.0, 10: 0.0} for _ in layer_acc]
                    layer_mean_rank_sum = [0.0] * len(layer_acc)
                    layer_median_rank_sum = [0.0] * len(layer_acc)
                    layer_semantic_rank_sum = [0.0] * len(layer_acc)
                    layer_semantic_batches = [0] * len(layer_acc)
                for i, acc in enumerate(layer_acc):
                    layer_acc_sum[i] += acc
                    layer_topk_sum[i][3] += token_diag['layer_topk_acc'][i][3]
                    layer_topk_sum[i][5] += token_diag['layer_topk_acc'][i][5]
                    layer_topk_sum[i][10] += token_diag['layer_topk_acc'][i][10]
                    layer_mean_rank_sum[i] += token_diag['layer_mean_rank'][i]
                    layer_median_rank_sum[i] += token_diag['layer_median_rank'][i]
                    sem_rank_i = token_diag['layer_semantic_neighbor_rank'][i]
                    if sem_rank_i is not None:
                        layer_semantic_rank_sum[i] += sem_rank_i
                        layer_semantic_batches[i] += 1
            else:
                token_acc, layer_acc = _progressive_accuracy(all_logits, all_tgt)
                totals['token_acc'] += token_acc
                if layer_acc_sum is None:
                    layer_acc_sum = [0.0] * len(layer_acc)
                for i, acc in enumerate(layer_acc):
                    layer_acc_sum[i] += acc
            n += 1

    out = {k: v / n for k, v in totals.items()}
    out['layer_acc'] = [v / n for v in layer_acc_sum] if layer_acc_sum is not None else []
    if compute_diag:
        if semantic_batches > 0:
            out['semantic_neighbor_rank'] = totals['semantic_neighbor_rank'] / semantic_batches
        else:
            out['semantic_neighbor_rank'] = None
        out['layer_topk_acc'] = [
            {k: v / n for k, v in layer_sum.items()}
            for layer_sum in layer_topk_sum
        ] if layer_topk_sum is not None else []
        out['layer_mean_rank'] = [v / n for v in layer_mean_rank_sum] if layer_mean_rank_sum is not None else []
        out['layer_median_rank'] = [v / n for v in layer_median_rank_sum] if layer_median_rank_sum is not None else []
        out['layer_semantic_neighbor_rank'] = [
            (
                layer_semantic_rank_sum[i] / layer_semantic_batches[i]
                if layer_semantic_batches[i] > 0 else None
            )
            for i in range(len(layer_semantic_rank_sum))
        ] if layer_semantic_rank_sum is not None else []
    return out


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _disable_ema(model):
    """冻结所有 VQ 模块的 EMA 更新（兼容 shared / per-channel + 单层/RVQ 模式）"""
    def _disable_rvq(rvq_mod):
        for single_vq in rvq_mod.layers:
            if isinstance(single_vq, FlattenedVectorQuantizerEMA):
                single_vq._disable_ema_update = True

    if model.per_channel_codebook:
        for rvq_mod in model.vqs:
            _disable_rvq(rvq_mod)
        print('✓ 已禁用 EMA 更新（per-channel 模式）')
    elif hasattr(model, 'vq'):
        _disable_rvq(model.vq)
        print('✓ 已禁用 EMA 更新（shared 模式）')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pretrain():
    args = build_arg_parser().parse_args()
    print('Args:', args)
    set_global_seed(int(args.seed))
    print(f'Seed set to: {int(args.seed)}')

    # NMPP 校验
    if args.use_raw_input:
        if not args.vqvae_checkpoint:
            raise ValueError('NMPP (--use_raw_input=1) 需要指定 --vqvae_checkpoint')
        args.freeze_vqvae = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    save_dir = Path(args.save_path) / args.dset
    save_dir.mkdir(parents=True, exist_ok=True)

    # 模型命名
    code_dim  = args.embedding_dim * (args.patch_size // args.compression_factor)
    step_size = args.progressive_step_size
    nmpp_sfx  = '_nmpp'  if args.use_raw_input        else ''
    perch_sfx = '_perch' if args.per_channel_codebook else ''
    rvq_sfx   = f'_rvq{args.n_rq_layers}' if getattr(args, 'n_rq_layers', 1) > 1 else ''
    backbone = str(getattr(args, 'vqvae_backbone', 'mlp')).lower()
    if backbone == 'mlp':
        backbone_sfx = ''
    elif backbone == 'linear':
        backbone_sfx = '_linear'
    elif backbone == 'conv_linear':
        backbone_sfx = f'_convlineark{int(getattr(args, "vqvae_tcn_kernel_size", 5))}'
    elif backbone == 'tcn':
        backbone_sfx = f'_tcnk{int(getattr(args, "vqvae_tcn_kernel_size", 5))}'
    else:
        backbone_sfx = f'_{backbone}c{int(getattr(args, "vqvae_chunk_size", 2))}'
    if bool(getattr(args, 'decoder_lowpass', 0)):
        backbone_sfx = f'{backbone_sfx}_dlp'
        lp_kernel = str(getattr(args, 'decoder_lowpass_kernel', 'binomial3'))
        # default kernel keeps backward-compatible filenames; other kernels
        # add a short suffix so different settings save to distinct ckpts.
        if lp_kernel != 'binomial3':
            backbone_sfx = f'{backbone_sfx}_{lp_kernel}'
    temporal_backbone = str(getattr(args, 'temporal_backbone', 'causal_transformer')).lower()
    if temporal_backbone in ('causal_transformer', 'transformer', 'patchtst'):
        temporal_sfx = ''
    elif temporal_backbone in ('encoder_transformer', 'transformer_encoder', 'noncausal_transformer'):
        temporal_sfx = '_encoder'
    elif temporal_backbone in ('encoder_timefilter_lite', 'encoder_timefilter'):
        temporal_sfx = f'_encodertfk{int(getattr(args, "timefilter_topk", 8))}'
    elif temporal_backbone in ('encoder_cluster_timefilter_lite', 'encoder_cluster_timefilter'):
        temporal_sfx = f'_encoderclustertfk{int(getattr(args, "timefilter_topk", 8))}'
    elif temporal_backbone == 'relative_transformer':
        temporal_sfx = '_relpos'
    elif temporal_backbone == 'timefilter_lite':
        temporal_sfx = f'_timefilterlitek{int(getattr(args, "timefilter_topk", 8))}'
    elif temporal_backbone == 'cluster_timefilter_lite':
        temporal_sfx = f'_clustertfk{int(getattr(args, "timefilter_topk", 8))}'
    elif temporal_backbone == 'timefilter_attn':
        heads = getattr(args, 'timefilter_attn_heads', None)
        heads = int(heads if heads is not None else getattr(args, 'n_heads', 4))
        temporal_sfx = f'_timefilterattnh{heads}k{int(getattr(args, "timefilter_topk", 8))}'
    elif temporal_backbone == 'channel_summary_adapter':
        temporal_sfx = f'_chsummaryw{int(getattr(args, "channel_summary_window", 4))}'
    else:
        temporal_sfx = f'_{temporal_backbone}'
    soft_k = int(getattr(args, 'soft_neighbor_k', 0))
    if soft_k > 0:
        soft_alpha = float(getattr(args, 'soft_neighbor_alpha', 0.25))
        soft_tau = float(getattr(args, 'soft_neighbor_tau', 0.3))
        # 用 Python 默认 float repr，保留尾随的 .0，与 bash pipeline 中的命名规则一致
        soft_sfx = f'_snk{soft_k}a{soft_alpha}t{soft_tau}'.replace('.', 'p')
    else:
        soft_sfx = ''
    rid_sfx   = f'_run{args.run_id}' if args.run_id is not None else ''
    ch_sfx    = _channel_suffix(args)
    model_name = (
        f'patch_vqvae_ps{args.patch_size}_cb{args.codebook_size}_cd{code_dim}'
        f'_l{args.n_layers}_in{args.context_points}_step{step_size}'
        f'{rid_sfx}_model{args.model_id}{perch_sfx}{rvq_sfx}'
        f'{backbone_sfx}{temporal_sfx}{nmpp_sfx}{soft_sfx}{ch_sfx}'
    )
    # 同名 pretrain 文件存在时，先清理旧文件再写入新结果（保持文件名稳定）。
    existing_ckpt = save_dir / f'{model_name}.pth'
    if existing_ckpt.exists():
        old_artifacts = [
            existing_ckpt,
            save_dir / f'{model_name}_history.csv',
            save_dir / f'{model_name}_results.csv',
            save_dir / f'{model_name}_config.json',
        ]
        removed = []
        for path in old_artifacts:
            if path.exists():
                path.unlink()
                removed.append(path.name)
        if removed:
            print(f'检测到同名历史pretrain结果，已先删除: {", ".join(removed)}')

    # 数据
    args.dset_pretrain = args.dset
    dls = get_dls(args)
    print(f'Channels: {dls.vars} | Train batches: {len(dls.train)} | Val batches: {len(dls.valid)}')
    if getattr(dls, 'channel_start', None) is not None:
        print(f'Channel group: [{dls.channel_start}, {dls.channel_end}) / full channels = {dls.full_vars}')

    # 如果提供了 VQVAE checkpoint，先从其 config 覆盖 VQVAE 结构参数，
    # 防止 num_residual_hiddens 等参数与命令行默认值不一致导致 size mismatch
    if args.vqvae_checkpoint:
        try:
            ckpt_meta = torch.load(args.vqvae_checkpoint, map_location='cpu')
            ckpt_cfg  = ckpt_meta.get('config', {})
            vqvae_keys = [
                'patch_size', 'embedding_dim', 'compression_factor',
                'codebook_size', 'num_hiddens', 'num_residual_layers',
                'num_residual_hiddens', 'commitment_cost',
                'vqvae_backbone', 'vqvae_tcn_kernel_size', 'vqvae_chunk_size',
                'decoder_lowpass', 'decoder_lowpass_kernel',
                'codebook_ema', 'ema_decay', 'ema_eps',
            ]
            overridden = []
            for k in vqvae_keys:
                if k in ckpt_cfg:
                    old_v = getattr(args, k, None)
                    new_v = ckpt_cfg[k]
                    if old_v != new_v:
                        setattr(args, k, new_v)
                        overridden.append(f'{k}: {old_v} → {new_v}')
            if overridden:
                print('\n[VQVAE config 自动覆盖]')
                for s in overridden:
                    print(f'  {s}')
        except Exception as e:
            print(f'[警告] 读取 checkpoint config 失败，使用命令行参数: {e}')

    # 模型
    config = get_model_config(args)
    config['n_channels'] = dls.vars
    config['channel_start'] = getattr(args, 'channel_start', None)
    config['channel_end'] = getattr(args, 'channel_end', None)
    config['channel_indices'] = getattr(args, 'channel_indices', None)
    config['channel_group_id'] = getattr(args, 'channel_group_id', None)
    model = PatchVQVAETransformer(config).to(device)

    # 加载预训练 VQVAE
    if args.vqvae_checkpoint:
        print(f'\n加载预训练 VQVAE: {args.vqvae_checkpoint}')
        model.load_vqvae_weights(
            args.vqvae_checkpoint,
            device,
            load_vq=bool(args.load_vq_weights),
            freeze=bool(args.freeze_vqvae),
        )

    # 禁用 EMA 更新
    if args.disable_ema_update:
        _disable_ema(model)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n参数: 总计 {total_p:,} | 可训练 {train_p:,} | 冻结 {total_p - train_p:,}')

    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)

    # 早停配置（全部可通过 CLI 覆盖）
    patience  = int(getattr(args, 'early_stop_patience',    5))
    warmup    = int(getattr(args, 'early_stop_warmup',      5))
    min_delta = float(getattr(args, 'early_stop_min_delta', 1e-4))
    smooth_k  = max(1, int(getattr(args, 'early_stop_smooth_k', 1)))

    soft_neighbor_k = int(getattr(args, 'soft_neighbor_k', 0))
    compute_diag = soft_neighbor_k > 0

    best_val   = float('inf')
    no_improve = 0
    train_losses, valid_losses = [], []
    train_token_accs, valid_token_accs = [], []
    if compute_diag:
        train_top3_accs, valid_top3_accs = [], []
        train_top5_accs, valid_top5_accs = [], []
        train_top10_accs, valid_top10_accs = [], []
        train_mean_ranks, valid_mean_ranks = [], []
        train_median_ranks, valid_median_ranks = [], []
        valid_semantic_ranks = []

    print(
        f'\n开始预训练，共 {args.n_epochs} epoch '
        f'(early stop: patience={patience}, warmup={warmup}, '
        f'min_delta={min_delta}, smooth_k={smooth_k})'
    )
    if compute_diag:
        print(
            f"NTP loss: codebook-neighbor soft label "
            f"(k={int(args.soft_neighbor_k)}, alpha={float(args.soft_neighbor_alpha):g}, "
            f"tau={float(args.soft_neighbor_tau):g})"
        )
    else:
        print("NTP loss: hard cross entropy (soft_neighbor_k=0)")
    print('=' * 80)

    for epoch in range(args.n_epochs):
        tr = train_epoch(model, dls.train, optimizer, scheduler, revin,
                         args, device, trainable_params)
        va = validate_epoch(model, dls.valid, revin, args, device)

        train_losses.append(tr['loss'])
        valid_losses.append(va['loss'])
        train_token_accs.append(tr['token_acc'])
        valid_token_accs.append(va['token_acc'])
        if compute_diag:
            train_top3_accs.append(tr['top3_acc'])
            valid_top3_accs.append(va['top3_acc'])
            train_top5_accs.append(tr['top5_acc'])
            valid_top5_accs.append(va['top5_acc'])
            train_top10_accs.append(tr['top10_acc'])
            valid_top10_accs.append(va['top10_acc'])
            train_mean_ranks.append(tr['mean_rank'])
            valid_mean_ranks.append(va['mean_rank'])
            train_median_ranks.append(tr['median_rank'])
            valid_median_ranks.append(va['median_rank'])
            valid_semantic_ranks.append(va.get('semantic_neighbor_rank'))

        if smooth_k > 1 and len(valid_losses) >= smooth_k:
            va_signal = sum(valid_losses[-smooth_k:]) / smooth_k
        else:
            va_signal = va['loss']

        print(
            f"Epoch {epoch+1:3d}/{args.n_epochs} | "
            f"Train {tr['loss']:.4f} (Pred {tr['pred_loss']:.4f}  "
            f"VQ {tr['vq_loss']:.4f}  Recon {tr['recon_loss']:.4f})"
            f" | Val {va['loss']:.4f} (Pred {va['pred_loss']:.4f})"
        )
        tr_layers = ', '.join(f'L{i}:{a * 100:.1f}%' for i, a in enumerate(tr.get('layer_acc', [])))
        va_layers = ', '.join(f'L{i}:{a * 100:.1f}%' for i, a in enumerate(va.get('layer_acc', [])))
        print(
            f"  └─ NTP Acc: Train {tr['token_acc'] * 100:.2f}%"
            f" | Val {va['token_acc'] * 100:.2f}%"
        )
        if compute_diag:
            print(
                f"      Top-k Val: top3 {va['top3_acc'] * 100:.2f}%"
                f" | top5 {va['top5_acc'] * 100:.2f}%"
                f" | top10 {va['top10_acc'] * 100:.2f}%"
            )
            sem_rank = va.get('semantic_neighbor_rank')
            sem_text = f"{sem_rank:.2f}" if sem_rank is not None else "N/A"
            print(
                f"      Code Rank: Train mean/median {tr['mean_rank']:.2f}/{tr['median_rank']:.1f}"
                f" | Val mean/median {va['mean_rank']:.2f}/{va['median_rank']:.1f}"
                f" | SemanticNeighborRank {sem_text}"
            )
            va_layer_topk = va.get('layer_topk_acc', [])
            va_layer_mean_rank = va.get('layer_mean_rank', [])
            va_layer_median_rank = va.get('layer_median_rank', [])
            va_layer_sem_rank = va.get('layer_semantic_neighbor_rank', [])
            for i, topk_i in enumerate(va_layer_topk):
                sem_i = va_layer_sem_rank[i] if i < len(va_layer_sem_rank) else None
                sem_i_text = f"{sem_i:.2f}" if sem_i is not None else "N/A"
                mean_i = va_layer_mean_rank[i] if i < len(va_layer_mean_rank) else 0.0
                median_i = va_layer_median_rank[i] if i < len(va_layer_median_rank) else 0.0
                print(
                    f"      L{i} Top-k/Rank: top3 {topk_i[3] * 100:.2f}%"
                    f" | top5 {topk_i[5] * 100:.2f}%"
                    f" | top10 {topk_i[10] * 100:.2f}%"
                    f" | rank {mean_i:.2f}/{median_i:.1f}"
                    f" | sem {sem_i_text}"
                )
        if tr_layers and va_layers:
            print(f"      Train Layers: {tr_layers}")
            print(f"      Val Layers  : {va_layers}")

        if va_signal < best_val - min_delta:
            best_val   = va_signal
            no_improve = 0
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'args':   vars(args),
                    'epoch':  epoch,
                    'train_loss': tr['loss'],
                    'val_loss':   va['loss'],
                },
                save_dir / f'{model_name}.pth',
            )
            print(f"  -> Best model saved (val_signal: {va_signal:.4f})")
        else:
            no_improve += 1

        if epoch + 1 > warmup and no_improve >= patience:
            print(f'\n>>> 早停: val_loss 连续 {patience} epoch 未显著下降'
                  f'（min_delta={min_delta}, smooth_k={smooth_k}）')
            break

        # 定期打印码本利用率及（可选）overlap coverage 统计
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample = next(iter(dls.train))[0].to(device)
                if revin:
                    sample = revin(sample, 'norm')
                usage, _ = model.get_codebook_usage(sample)
                print(f'  -> Codebook usage: {usage * 100:.1f}%')

            # 打印 overlap coverage（有重叠时）
            eff_pred_len = getattr(args, 'pred_len', None) or args.progressive_step_size
            if eff_pred_len != args.progressive_step_size:
                M, N = args.progressive_step_size, eff_pred_len
                # 理论覆盖：位置 p 被 min(floor(p/M)+1, ceil(N/M)) 个 chunk 覆盖
                import math
                max_cover = math.ceil(N / M)
                print(f'  -> Overlap config: step_size={M}, pred_len={N} '
                      f'| max_coverage_per_pos={max_cover} '
                      f'| overlap_len={N - M} patches/stage')

    history = {
        'epoch':       range(1, len(train_losses) + 1),
        'train_loss':  train_losses,
        'valid_loss':  valid_losses,
        'train_token_acc': train_token_accs,
        'valid_token_acc': valid_token_accs,
    }
    if compute_diag:
        history.update({
            'train_top3_acc': train_top3_accs,
            'valid_top3_acc': valid_top3_accs,
            'train_top5_acc': train_top5_accs,
            'valid_top5_acc': valid_top5_accs,
            'train_top10_acc': train_top10_accs,
            'valid_top10_acc': valid_top10_accs,
            'train_mean_rank': train_mean_ranks,
            'valid_mean_rank': valid_mean_ranks,
            'train_median_rank': train_median_ranks,
            'valid_median_rank': valid_median_ranks,
            'valid_semantic_neighbor_rank': valid_semantic_ranks,
        })
    pd.DataFrame(history).to_csv(save_dir / f'{model_name}_history.csv', index=False)

    with open(save_dir / f'{model_name}_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    print('=' * 80)
    print(f'预训练完成。最佳验证损失: {best_val:.4f}')
    print(f'模型: {save_dir / model_name}.pth')
