"""
码本预训练脚本（带序列间对比学习损失）
独立训练 Encoder + Codebook (VQ) + Decoder

新增功能：
- Batch 内序列间对比学习 Loss（基于MSE距离）
- 如果原始序列之间的MSE距离小于阈值，标记为相似
- 对量化后的序列进行对比学习，让相似序列对的量化距离也小，不相似序列对的量化距离也大
- 相似度的衡量始终使用MSE范式
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda import amp
from torch.utils.data import Subset, DataLoader
import argparse
from pathlib import Path
import random

# 添加根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.codebook_model import CodebookModel
from src.models.layers.revin import RevIN
from src.basics import set_device
from datautils import get_dls


def parse_args():
    parser = argparse.ArgumentParser(description='码本预训练（带序列间对比学习损失）')
    
    # 数据集参数
    parser.add_argument('--dset', type=str, default='ettm1', help='数据集名称')
    parser.add_argument('--context_points', type=int, default=512, help='输入序列长度')
    parser.add_argument('--target_points', type=int, default=0, help='预测长度（码本预训练不使用）')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--scaler', type=str, default='standard', help='数据缩放方式')
    parser.add_argument('--features', type=str, default='M', help='特征类型')
    
    # 模型参数
    parser.add_argument('--patch_size', type=int, default=16, help='Patch大小')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Embedding维度')
    parser.add_argument('--codebook_size', type=int, default=256, help='码本大小')
    parser.add_argument('--compression_factor', type=int, default=4, choices=[4, 8, 12, 16], help='压缩因子')
    parser.add_argument('--num_hiddens', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='残差层数')
    parser.add_argument('--num_residual_hiddens', type=int, default=32, help='残差隐藏层维度')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='VQ commitment cost')
    parser.add_argument('--codebook_ema', type=int, default=0, help='是否使用EMA更新码本')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA衰减率')
    parser.add_argument('--ema_eps', type=float, default=1e-5, help='EMA epsilon')
    
    # 码本初始化参数
    parser.add_argument('--vq_init_method', type=str, default='random', 
                       choices=['random', 'normal', 'xavier', 'kaiming'],
                       help='码本初始化方法')
    parser.add_argument('--codebook_report_interval', type=int, default=5,
                       help='码本利用率报告间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN')
    parser.add_argument('--amp', type=int, default=1, help='是否启用混合精度')
    parser.add_argument('--vq_weight', type=float, default=1.0, help='VQ损失权重')
    parser.add_argument('--recon_weight', type=float, default=1.0, help='重构损失权重')
    
    # ============ 对比学习损失参数 ============
    parser.add_argument('--inter_weight', type=float, default=0.1, 
                       help='对比学习损失权重')
    parser.add_argument('--similarity_threshold', type=float, default=0.5,
                       help='原始序列MSE距离阈值，低于此阈值的样本对被视为相似（正样本）')
    parser.add_argument('--inter_loss_type', type=str, default='mse',
                       choices=['mse', 'contrastive'],
                       help='对比学习损失类型：mse（对齐距离矩阵）或contrastive（对比学习）')
    parser.add_argument('--inter_temperature', type=float, default=0.1,
                       help='对比学习损失的温度系数（仅用于contrastive模式）')
    
    # ============ 对比学习损失延迟参数 ============
    parser.add_argument('--inter_delay_epochs', type=int, default=5,
                       help='前N个epoch完全禁用对比学习损失（只使用intra_loss），之后直接加入inter_loss')
    
    # ============ 软索引参数（Gumbel Softmax） ============
    parser.add_argument('--use_soft_indices', type=int, default=1,
                       help='是否使用软索引（1=启用，0=禁用，使用硬索引）')
    parser.add_argument('--soft_index_method', type=str, default='gumbel',
                       choices=['gumbel', 'softmax'],
                       help='软索引方法：gumbel或softmax')
    parser.add_argument('--gumbel_temperature', type=float, default=1.0,
                       help='Gumbel Softmax温度系数')
    parser.add_argument('--gumbel_hard', type=int, default=0,
                       help='Gumbel Softmax是否使用Straight-Through（1=hard，0=soft）')
    parser.add_argument('--soft_index_temperature', type=float, default=1.0,
                       help='Softmax温度系数（仅用于softmax方法）')
    
    # 数据采样参数
    parser.add_argument('--train_sample_ratio', type=float, default=1.0, 
                       help='训练集采样比例')
    parser.add_argument('--valid_sample_ratio', type=float, default=1.0,
                       help='验证集采样比例')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/vqvae_only_inter/', help='模型保存路径')
    parser.add_argument('--model_id', type=int, default=1, help='模型ID')
    
    return parser.parse_args()


def get_model_config(args):
    """构建模型配置"""
    config = {
        'patch_size': args.patch_size,
        'embedding_dim': args.embedding_dim,
        'compression_factor': args.compression_factor,
        'codebook_size': args.codebook_size,
        'commitment_cost': args.commitment_cost,
        'codebook_ema': bool(args.codebook_ema),
        'ema_decay': args.ema_decay,
        'ema_eps': args.ema_eps,
        'vq_init_method': args.vq_init_method,
        'num_hiddens': args.num_hiddens,
        'num_residual_layers': args.num_residual_layers,
        'num_residual_hiddens': args.num_residual_hiddens,
        'use_patch_attention': False,
        # 软索引配置
        'use_soft_indices': bool(getattr(args, 'use_soft_indices', 1)),
        'soft_index_method': getattr(args, 'soft_index_method', 'gumbel'),
        'gumbel_temperature': getattr(args, 'gumbel_temperature', 1.0),
        'gumbel_hard': bool(getattr(args, 'gumbel_hard', 0)),
        'soft_index_temperature': getattr(args, 'soft_index_temperature', 1.0),
    }
    return config


def get_inter_weight_with_delay(args, current_epoch):
    """
    计算当前epoch的对比学习损失权重（简单延迟开关）
    
    逻辑：
    1. epoch < delay_epochs: 权重 = 0（完全禁用对比学习损失，只使用intra_loss）
    2. epoch >= delay_epochs: 权重 = inter_weight（直接加入inter_loss）
    
    Args:
        args: 参数
        current_epoch: 当前epoch（从0开始）
    
    Returns:
        inter_weight: 当前的对比学习损失权重（0或inter_weight）
    """
    delay_epochs = getattr(args, 'inter_delay_epochs', 5)
    
    # 前k步：只使用intra_loss
    if current_epoch < delay_epochs:
        return 0.0
    
    # k步后：直接加入inter_loss
    return args.inter_weight


def compute_freq_magnitude(x, dim=-1):
    """
    计算序列的FFT幅值
    
    Args:
        x: [..., L] 输入序列
        dim: FFT计算的维度
    
    Returns:
        magnitude: [..., L//2+1] FFT幅值
    """
    fft_result = torch.fft.rfft(x, dim=dim)
    magnitude = torch.abs(fft_result)
    return magnitude


def compute_mse_distance_matrix(sequences):
    """
    计算序列之间的成对MSE距离矩阵（向量化实现）
    
    Args:
        sequences: [B, L] 或 [B, L, C] 序列
    
    Returns:
        dist_matrix: [B, B] MSE距离矩阵
    """
    if sequences.dim() == 3:
        # [B, L, C] -> [B, L*C] 展平
        sequences = sequences.reshape(sequences.shape[0], -1)
    
    B, L = sequences.shape
    # 向量化计算MSE距离矩阵
    # dist[i, j] = mean((sequences[i] - sequences[j])^2)
    # 使用广播：sequences[i] - sequences[j] for all i, j
    # sequences: [B, L]
    # sequences.unsqueeze(0): [1, B, L]
    # sequences.unsqueeze(1): [B, 1, L]
    # diff: [B, B, L]
    diff = sequences.unsqueeze(0) - sequences.unsqueeze(1)  # [B, B, L]
    dist_matrix = (diff ** 2).mean(dim=-1)  # [B, B]
    
    return dist_matrix


def compute_contrastive_loss_mse(D_orig, D_quantized, threshold):
    """
    计算对比学习损失（MSE对齐版本）
    让量化序列的距离矩阵与原始序列的距离矩阵对齐
    
    Args:
        D_orig: [B, B] 原始序列的MSE距离矩阵
        D_quantized: [B, B] 量化序列的MSE距离矩阵
        threshold: float, 相似度阈值（距离小于此值视为相似）
    
    Returns:
        loss: scalar MSE损失
    """
    loss = F.mse_loss(D_quantized, D_orig)
    return loss


def compute_contrastive_loss_contrastive(D_orig, D_quantized, threshold, temperature=0.1):
    """
    计算对比学习损失（对比学习版本）
    相似序列对（距离 < threshold）的量化距离应该小
    不相似序列对（距离 >= threshold）的量化距离应该大
    
    Args:
        D_orig: [B, B] 原始序列的MSE距离矩阵
        D_quantized: [B, B] 量化序列的MSE距离矩阵
        threshold: float, 相似度阈值（距离小于此值视为相似）
        temperature: float, 温度系数
    
    Returns:
        loss: scalar 对比学习损失
    """
    B = D_orig.shape[0]
    device = D_orig.device
    
    # 创建相似性掩码：距离 < threshold 的为相似（正样本）
    positive_mask = (D_orig < threshold).float()  # [B, B]
    # 对角线永远是正样本（自己和自己）
    positive_mask.fill_diagonal_(1.0)
    
    # 正样本损失：相似序列对的量化距离应该小
    positive_loss = (positive_mask * D_quantized).sum() / (positive_mask.sum() + 1e-8)
    
    # 负样本损失：不相似序列对的量化距离应该大（使用exp(-distance/temperature)作为权重）
    negative_mask = 1.0 - positive_mask  # [B, B]
    # 对于负样本，我们希望距离大，所以使用 exp(-D_quantized/temperature) 作为权重
    # 距离越大，权重越小（即我们希望距离大的样本对贡献更小）
    negative_weights = torch.exp(-D_quantized / temperature) * negative_mask
    negative_loss = (negative_weights * D_quantized).sum() / (negative_weights.sum() + 1e-8)
    
    # 总损失：正样本距离小 + 负样本距离大
    loss = positive_loss - negative_loss
    
    return loss


def compute_inter_sequence_loss(x_orig, z_q, args, patch_size):
    """
    计算Batch内序列间对比学习损失（基于MSE距离）
    
    逻辑：
    1. 计算原始序列之间的MSE距离矩阵（先在patch内取mean）
    2. 如果距离 < threshold，标记为相似（正样本）
    3. 计算量化序列之间的MSE距离矩阵（对code_dim取mean）
    4. 使用对比学习，让相似序列对的量化距离也小，不相似序列对的量化距离也大
    
    Args:
        x_orig: [B, T, C] 原始输入序列
        z_q: [B, num_patches, C, code_dim] 量化后的向量
        args: 参数
        patch_size: int, patch大小
    
    Returns:
        loss: scalar 对比学习损失
        info: dict 包含中间信息
    """
    B, T, C = x_orig.shape
    _, num_patches, _, code_dim = z_q.shape
    
    # ============ 原始序列的MSE距离矩阵 ============
    # x_orig: [B, T, C]
    # 先按patch_size分成patches，然后在每个patch内取mean
    # [B, T, C] -> [B, num_patches, patch_size, C] -> [B, num_patches, C] (在patch_size维度取mean)
    x_patches = x_orig[:, :num_patches * patch_size, :].reshape(B, num_patches, patch_size, C)
    x_patch_mean = x_patches.mean(dim=2)  # [B, num_patches, C]
    
    # 重组为 (B*C)*num_patches 格式：[B, num_patches, C] -> [B*C, num_patches]
    x_flat = x_patch_mean.permute(0, 2, 1).reshape(B * C, num_patches)  # [B*C, num_patches]
    
    # 计算原始序列之间的MSE距离矩阵
    D_orig = compute_mse_distance_matrix(x_flat)  # [B*C, B*C]
    D_orig = D_orig.detach()  # 梯度隔离，只作为目标
    
    # ============ 量化序列的MSE距离矩阵 ============
    # z_q: [B, num_patches, C, code_dim]
    # 对 code_dim 维度取平均，得到 [B, num_patches, C]
    z_q_reduced = z_q.mean(dim=-1)  # [B, num_patches, C]
    
    # 重组为 (B*C)*num_patches 格式：[B, num_patches, C] -> [B*C, num_patches]
    z_q_flat = z_q_reduced.permute(0, 2, 1).reshape(B * C, num_patches)  # [B*C, num_patches]
    
    # 计算量化序列之间的MSE距离矩阵
    D_quantized = compute_mse_distance_matrix(z_q_flat)  # [B*C, B*C]
    
    # ============ 计算对比学习损失 ============
    if args.inter_loss_type == 'mse':
        # MSE对齐：直接对齐距离矩阵
        loss = compute_contrastive_loss_mse(D_orig, D_quantized, args.similarity_threshold)
    else:  # contrastive
        # 对比学习：相似序列对距离小，不相似序列对距离大
        loss = compute_contrastive_loss_contrastive(
            D_orig, D_quantized,
            threshold=args.similarity_threshold,
            temperature=args.inter_temperature
        )
    
    # 计算相似性统计
    positive_mask = (D_orig < args.similarity_threshold).float()
    positive_mask.fill_diagonal_(1.0)
    num_total_samples = B * C
    num_positive_pairs = positive_mask.sum().item() - num_total_samples  # 减去对角线
    
    info = {
        'D_orig_mean': D_orig.mean().item(),
        'D_quantized_mean': D_quantized.mean().item(),
        'D_orig_diag_mean': D_orig.diag().mean().item(),
        'D_quantized_diag_mean': D_quantized.diag().mean().item(),
        'num_positive_pairs': num_positive_pairs,
        'num_total_pairs': num_total_samples * (num_total_samples - 1),
        'positive_ratio': num_positive_pairs / (num_total_samples * (num_total_samples - 1)) if num_total_samples > 1 else 0.0,
    }
    
    return loss, info


def compute_codebook_usage_stats(indices, codebook_size):
    """计算码本利用率统计信息"""
    indices_flat = indices.reshape(-1).cpu()
    unique_indices = torch.unique(indices_flat)
    num_used = len(unique_indices)
    usage_rate = num_used / codebook_size
    
    counts = torch.bincount(indices_flat, minlength=codebook_size)
    num_unused = (counts == 0).sum().item()
    
    top5_counts, top5_indices = torch.topk(counts, k=min(5, codebook_size))
    top5_usage = [(idx.item(), count.item()) for idx, count in zip(top5_indices, top5_counts) if count > 0]
    
    return {
        'num_used': num_used,
        'num_unused': num_unused,
        'usage_rate': usage_rate,
        'top5_usage': top5_usage,
        'total_tokens': len(indices_flat),
    }


def train_epoch(model, dataloader, optimizer, revin, args, device, scaler, current_epoch=0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    total_inter_loss = 0
    total_perplexity = 0
    n_batches = 0
    
    all_indices_list = []
    
    # 获取当前epoch的对比学习损失权重（简单延迟开关）
    current_inter_weight = get_inter_weight_with_delay(args, current_epoch)
    
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)  # [B, T, C]
        
        if revin:
            batch_x = revin(batch_x, 'norm')
        
        # 编码和解码
        indices, vq_loss, z_q = model.encode_to_indices(batch_x, return_distances=False)
        x_recon = model.decode_from_codes(z_q)
        
        # 计算重构损失
        B, T, C = batch_x.shape
        num_patches = indices.shape[1]
        recon_len = num_patches * model.patch_size
        recon_loss = F.mse_loss(x_recon, batch_x[:, :recon_len, :])
        
        # 计算对比学习损失（使用量化向量z_q，基于MSE距离）
        inter_loss, inter_info = compute_inter_sequence_loss(
            batch_x[:, :recon_len, :],  # 使用RevIN后的数据
            z_q,  # 量化后的向量 [B, num_patches, C, code_dim]
            args,
            model.patch_size  # 传递patch_size
        )
        
        # 总损失（使用延迟后的inter_weight）
        loss = (args.recon_weight * recon_loss + 
                args.vq_weight * vq_loss + 
                current_inter_weight * inter_loss)
        
        # 反向传播
        optimizer.zero_grad()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
        
        # 计算perplexity
        unique_indices = torch.unique(indices.reshape(-1))
        perplexity = len(unique_indices) / args.codebook_size
        
        all_indices_list.append(indices.detach().cpu())
        
        total_loss += loss.item()
        total_vq_loss += vq_loss.item()
        total_recon_loss += recon_loss.item()
        total_inter_loss += inter_loss.item()
        total_perplexity += perplexity
        n_batches += 1
    
    all_indices_epoch = torch.cat(all_indices_list, dim=0)
    codebook_stats = compute_codebook_usage_stats(all_indices_epoch, args.codebook_size)
    
    return {
        'loss': total_loss / n_batches if n_batches > 0 else 0.0,
        'vq_loss': total_vq_loss / n_batches if n_batches > 0 else 0.0,
        'recon_loss': total_recon_loss / n_batches if n_batches > 0 else 0.0,
        'inter_loss': total_inter_loss / n_batches if n_batches > 0 else 0.0,
        'perplexity': total_perplexity / n_batches if n_batches > 0 else 0.0,
        'codebook_stats': codebook_stats,
        'current_inter_weight': current_inter_weight,  # 返回当前使用的权重
    }


def validate_epoch(model, dataloader, revin, args, device, current_epoch=0):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    total_inter_loss = 0
    total_perplexity = 0
    n_batches = 0
    
    all_indices_list = []
    
    # 获取当前epoch的对比学习损失权重（简单延迟开关）
    current_inter_weight = get_inter_weight_with_delay(args, current_epoch)
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            # 编码和解码
            indices, vq_loss, z_q = model.encode_to_indices(batch_x, return_distances=False)
            x_recon = model.decode_from_codes(z_q)
            
            B, T, C = batch_x.shape
            num_patches = indices.shape[1]
            recon_len = num_patches * model.patch_size
            recon_loss = F.mse_loss(x_recon, batch_x[:, :recon_len, :])
            
            # 计算对比学习损失（使用量化向量z_q，基于MSE距离）
            inter_loss, inter_info = compute_inter_sequence_loss(
                batch_x[:, :recon_len, :],
                z_q,  # 量化后的向量 [B, num_patches, C, code_dim]
                args,
                model.patch_size  # 传递patch_size
            )
            
            loss = (args.recon_weight * recon_loss + 
                    args.vq_weight * vq_loss + 
                    current_inter_weight * inter_loss)
            
            unique_indices = torch.unique(indices.reshape(-1))
            perplexity = len(unique_indices) / args.codebook_size
            
            all_indices_list.append(indices.cpu())
            
            total_loss += loss.item()
            total_vq_loss += vq_loss.item()
            total_recon_loss += recon_loss.item()
            total_inter_loss += inter_loss.item()
            total_perplexity += perplexity
            n_batches += 1
    
    all_indices_epoch = torch.cat(all_indices_list, dim=0)
    codebook_stats = compute_codebook_usage_stats(all_indices_epoch, args.codebook_size)
    
    return {
        'loss': total_loss / n_batches if n_batches > 0 else 0.0,
        'vq_loss': total_vq_loss / n_batches if n_batches > 0 else 0.0,
        'recon_loss': total_recon_loss / n_batches if n_batches > 0 else 0.0,
        'inter_loss': total_inter_loss / n_batches if n_batches > 0 else 0.0,
        'perplexity': total_perplexity / n_batches if n_batches > 0 else 0.0,
        'codebook_stats': codebook_stats,
        'current_inter_weight': current_inter_weight,
    }


def set_seed(seed):
    """设置随机数种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ 随机数种子已设置为: {seed}")


def main():
    args = parse_args()
    print('=' * 80)
    print('码本预训练（带序列间对比学习损失 - Inter-Loss）')
    print('=' * 80)
    print(f'Args: {args}')
    
    # PyTorch 兼容性修复
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
        print('✓ 已禁用 flash/memory-efficient attention')
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建保存目录
    save_dir = Path(args.save_path) / args.dset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型文件名
    code_dim = args.embedding_dim * (args.patch_size // args.compression_factor)
    inter_suffix = f"_inter{args.inter_weight}"
    model_name = f'codebook_ps{args.patch_size}_cb{args.codebook_size}_cd{code_dim}{inter_suffix}_model{args.model_id}'
    
    # 获取数据
    args.dset_pretrain = args.dset
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Train batches: {len(dls.train)}, Valid batches: {len(dls.valid)}')
    
    # 数据采样
    if args.train_sample_ratio < 1.0:
        train_dataset = dls.train.dataset
        train_size = len(train_dataset)
        sample_size = int(train_size * args.train_sample_ratio)
        indices = torch.randperm(train_size)[:sample_size].tolist()
        train_subset = Subset(train_dataset, indices)
        dls.train = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=getattr(dls.train, 'collate_fn', None)
        )
        print(f'训练集采样: {sample_size}/{train_size} ({args.train_sample_ratio*100:.1f}%)')
    
    if args.valid_sample_ratio < 1.0:
        valid_dataset = dls.valid.dataset
        valid_size = len(valid_dataset)
        sample_size = int(valid_size * args.valid_sample_ratio)
        indices = torch.randperm(valid_size)[:sample_size].tolist()
        valid_subset = Subset(valid_dataset, indices)
        dls.valid = DataLoader(
            valid_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=getattr(dls.valid, 'collate_fn', None)
        )
        print(f'验证集采样: {sample_size}/{valid_size} ({args.valid_sample_ratio*100:.1f}%)')
    
    # 创建模型
    config = get_model_config(args)
    model = CodebookModel(config, dls.vars).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'\n模型参数统计:')
    print(f'  总参数: {total_params:,}')
    print(f'  可训练参数: {trainable_params:,}')
    print(f'\n对比学习损失配置:')
    print(f'  损失类型: {args.inter_loss_type}')
    print(f'  损失权重: {args.inter_weight}')
    print(f'  相似度阈值（MSE距离）: {args.similarity_threshold}')
    print(f'  温度系数: {args.inter_temperature}')
    
    print(f'\n软索引配置（解决argmax梯度断裂）:')
    if hasattr(args, 'use_soft_indices') and args.use_soft_indices:
        print(f'  启用软索引')
        print(f'  方法: {args.soft_index_method}')
        if args.soft_index_method == 'gumbel':
            print(f'  Gumbel温度: {args.gumbel_temperature}')
            print(f'  Straight-Through: {bool(args.gumbel_hard)}')
        else:
            print(f'  Softmax温度: {args.soft_index_temperature}')
    else:
        print(f'  使用硬索引（无梯度流向encoder）')
    
    # 检查可训练参数
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params_list) == 0:
        raise ValueError(
            "错误: 没有可训练参数！\n"
            "解决方案：禁用EMA: --codebook_ema 0"
        )
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 优化器
    optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    
    # AMP
    scaler = amp.GradScaler(enabled=bool(args.amp))
    
    # 训练
    best_val_loss = float('inf')
    train_losses, valid_losses = [], []
    no_improve_count = 0
    early_stop_patience = 10
    model_saved = False
    delay_epochs = getattr(args, 'inter_delay_epochs', 5)
    
    print(f'\n开始训练，共 {args.n_epochs} 个 epoch (早停: {early_stop_patience} epochs)')
    print('=' * 80)
    
    for epoch in range(args.n_epochs):
        train_metrics = train_epoch(model, dls.train, optimizer, revin, args, device, scaler, current_epoch=epoch)
        scheduler.step()
        
        val_metrics = validate_epoch(model, dls.valid, revin, args, device, current_epoch=epoch)
        
        train_losses.append(train_metrics['loss'])
        valid_losses.append(val_metrics['loss'])
        
        # 获取当前对比学习损失权重（用于打印）
        current_inter_weight = train_metrics.get('current_inter_weight', args.inter_weight)
        
        # 打印进度（显示当前状态：delay/normal）
        delay_epochs = getattr(args, 'inter_delay_epochs', 5)
        if epoch < delay_epochs:
            warmup_info = f"[intra_only {epoch+1}/{delay_epochs}]"
        else:
            warmup_info = f"[inter_enabled]"
        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"Train: total={train_metrics['loss']:.4f} "
              f"(recon={train_metrics['recon_loss']:.4f}, "
              f"vq={train_metrics['vq_loss']:.4f}, "
              f"inter={train_metrics['inter_loss']:.4f}{warmup_info}) | "
              f"Valid: total={val_metrics['loss']:.4f} "
              f"(recon={val_metrics['recon_loss']:.4f}, "
              f"inter={val_metrics['inter_loss']:.4f})")
        
        # 定期报告码本利用率
        if (epoch + 1) % args.codebook_report_interval == 0:
            train_stats = train_metrics.get('codebook_stats', {})
            val_stats = val_metrics.get('codebook_stats', {})
            train_usage = train_stats.get('usage_rate', 0.0) * 100
            val_usage = val_stats.get('usage_rate', 0.0) * 100
            print(f"  └─ 码本利用率: Train {train_usage:.1f}% | Valid {val_usage:.1f}%")
        
        # 保存最佳模型（只有当inter_loss启用时才保存）
        delay_epochs = getattr(args, 'inter_delay_epochs', 5)
        if epoch >= delay_epochs:  # 只有inter_loss启用后才保存模型
            if epoch >= 5:  # 同时满足至少5个epoch的条件
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    no_improve_count = 0
                    model_saved = True
                    
                    checkpoint = {
                        'encoder_state_dict': model.encoder.state_dict(),
                        'decoder_state_dict': model.decoder.state_dict(),
                        'vq_state_dict': model.vq.state_dict(),
                        'config': config,
                        'args': vars(args),
                        'epoch': epoch,
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'train_recon_loss': train_metrics['recon_loss'],
                        'val_recon_loss': val_metrics['recon_loss'],
                        'train_inter_loss': train_metrics['inter_loss'],
                        'val_inter_loss': val_metrics['inter_loss'],
                    }
                    torch.save(checkpoint, save_dir / f'{model_name}.pth')
                    print(f"  -> Best model saved (val_loss: {val_metrics['loss']:.4f})")
                else:
                    no_improve_count += 1
                    if no_improve_count >= early_stop_patience:
                        print(f"\n>>> 早停: val_loss 连续 {early_stop_patience} 个 epoch 未下降")
                        break
        else:
            # inter_loss未启用时，不保存模型，但更新no_improve_count用于早停判断
            # 注意：在inter_loss启用前，不进行早停判断
            pass
    
    # 保存训练历史
    actual_epochs = len(train_losses)
    history_df = pd.DataFrame({
        'epoch': range(1, actual_epochs + 1),
        'train_loss': train_losses,
        'valid_loss': valid_losses,
    })
    history_df.to_csv(save_dir / f'{model_name}_history.csv', index=False)
    
    # 保存配置
    with open(save_dir / f'{model_name}_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print('=' * 80)
    print(f'训练完成！')
    if model_saved:
        print(f'最佳验证损失: {best_val_loss:.4f}')
        print(f'模型保存至: {save_dir / model_name}.pth')
    else:
        print(f'注意: 模型未保存（inter_loss在epoch {delay_epochs}后才启用，可能未达到保存条件）')
        print(f'最终验证损失: {valid_losses[-1] if valid_losses else "N/A":.4f}')


if __name__ == '__main__':
    set_device()
    main()

