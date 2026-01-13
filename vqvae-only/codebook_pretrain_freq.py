"""
码本预训练脚本（带序列间频域一致性损失）
独立训练 Encoder + Codebook (VQ) + Decoder

新增功能：
- Batch 内序列间频域一致性 Loss
- 确保频率相似的原始序列，其量化编码在周期性上也相似
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
    parser = argparse.ArgumentParser(description='码本预训练（带频域一致性损失）')
    
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
    
    # ============ 频域一致性损失参数 ============
    parser.add_argument('--freq_weight', type=float, default=0.1, 
                       help='频域一致性损失权重')
    parser.add_argument('--freq_similarity_threshold', type=float, default=0.8,
                       help='频域相似度阈值，高于此阈值的样本对被视为正样本')
    parser.add_argument('--freq_loss_type', type=str, default='mse',
                       choices=['mse', 'infonce'],
                       help='频域一致性损失类型：mse或infonce')
    parser.add_argument('--freq_temperature', type=float, default=0.1,
                       help='InfoNCE损失的温度系数')
    
    # ============ 频域损失warmup参数 ============
    parser.add_argument('--freq_delay_epochs', type=int, default=20,
                       help='前N个epoch完全禁用频域损失（权重=0）')
    parser.add_argument('--freq_warmup_epochs', type=int, default=10,
                       help='延迟后，频域损失权重warmup的epoch数')
    parser.add_argument('--freq_weight_start', type=float, default=0.01,
                       help='频域损失权重的起始值（warmup开始时的值）')
    
    # 数据采样参数
    parser.add_argument('--train_sample_ratio', type=float, default=1.0, 
                       help='训练集采样比例')
    parser.add_argument('--valid_sample_ratio', type=float, default=1.0,
                       help='验证集采样比例')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/vqvae_only_freq/', help='模型保存路径')
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
    }
    return config


def get_freq_weight_with_warmup(args, current_epoch):
    """
    计算当前epoch的频域损失权重（带延迟和warmup）
    
    逻辑：
    1. epoch < delay_epochs: 权重 = 0（完全禁用freq_loss）
    2. delay_epochs <= epoch < delay_epochs + warmup_epochs: 线性warmup
    3. epoch >= delay_epochs + warmup_epochs: 权重 = freq_weight（目标权重）
    
    Args:
        args: 参数
        current_epoch: 当前epoch（从0开始）
    
    Returns:
        freq_weight: 当前的频域损失权重
    """
    delay_epochs = getattr(args, 'freq_delay_epochs', 20)
    warmup_epochs = getattr(args, 'freq_warmup_epochs', 10)
    weight_start = getattr(args, 'freq_weight_start', 0.01)
    weight_end = args.freq_weight
    
    # 阶段1：延迟期（完全禁用）
    if current_epoch < delay_epochs:
        return 0.0
    
    # 阶段2：warmup期
    warmup_start_epoch = delay_epochs
    warmup_end_epoch = delay_epochs + warmup_epochs
    
    if current_epoch < warmup_end_epoch:
        # 线性warmup：从 weight_start 到 weight_end
        progress = (current_epoch - warmup_start_epoch) / warmup_epochs
        freq_weight = weight_start + (weight_end - weight_start) * progress
        return freq_weight
    
    # 阶段3：正常训练
    return weight_end


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


def compute_cosine_similarity_matrix(features, eps=1e-8):
    """
    计算特征向量的成对余弦相似度矩阵
    
    Args:
        features: [B, D] 特征向量
        eps: 数值稳定性常数
    
    Returns:
        sim_matrix: [B, B] 余弦相似度矩阵
    """
    # L2归一化
    features_norm = F.normalize(features, p=2, dim=-1, eps=eps)
    # 计算余弦相似度
    sim_matrix = torch.matmul(features_norm, features_norm.t())
    return sim_matrix


def compute_freq_consistency_loss_mse(S_orig, S_discrete):
    """
    计算频域一致性损失（MSE版本）
    
    Args:
        S_orig: [B, B] 原始序列的频域相似度矩阵
        S_discrete: [B, B] 量化序列的频域相似度矩阵
    
    Returns:
        loss: scalar MSE损失
    """
    loss = F.mse_loss(S_discrete, S_orig)
    return loss


def compute_freq_consistency_loss_infonce(S_orig, S_discrete, threshold=0.8, temperature=0.1):
    """
    计算频域一致性损失（InfoNCE版本）
    使用原始序列的相似度作为权重/标签
    
    Args:
        S_orig: [B, B] 原始序列的频域相似度矩阵
        S_discrete: [B, B] 量化序列的频域相似度矩阵
        threshold: 相似度阈值，高于此阈值的视为正样本
        temperature: InfoNCE温度系数
    
    Returns:
        loss: scalar InfoNCE损失
    """
    B = S_orig.shape[0]
    device = S_orig.device
    
    # 根据阈值生成软标签
    # 高于阈值的样本对权重更高
    positive_mask = (S_orig > threshold).float()
    
    # 对角线永远是正样本
    positive_mask.fill_diagonal_(1.0)
    
    # 计算加权对比损失
    # logits = S_discrete / temperature
    logits = S_discrete / temperature
    
    # 对每个样本，计算其与所有正样本的对比损失
    # 使用 S_orig 作为软权重
    weights = S_orig.clamp(min=0)  # 确保非负
    
    # log_softmax
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 加权对比损失：正样本权重高，负样本权重低
    # loss = -sum(weights * log_probs) / sum(weights)
    weighted_log_probs = weights * log_probs
    
    # 每行求和，然后平均
    loss = -weighted_log_probs.sum(dim=-1) / (weights.sum(dim=-1) + 1e-8)
    loss = loss.mean()
    
    return loss


def compute_inter_sequence_freq_loss(x_orig, z_q, args):
    """
    计算Batch内序列间频域一致性损失（改进版）
    
    改进点：
    1. 使用量化后的向量 z_q 进行DFT，而不是离散索引
    2. 原始序列相似度矩阵使用 .detach()，只作为目标不产生梯度
    3. 对 z_q 在 code_dim 维度取平均，投影到1维后再做FFT
    
    Args:
        x_orig: [B, T, C] 原始输入序列
        z_q: [B, num_patches, C, code_dim] 量化后的向量
        args: 参数
    
    Returns:
        loss: scalar 频域一致性损失
        info: dict 包含中间信息
    """
    B, T, C = x_orig.shape
    _, num_patches, _, code_dim = z_q.shape
    
    # ============ FFT支路：原始序列的频域相似度 ============
    # 使用 .detach() 确保 S_orig 只作为目标，不产生反向梯度
    x_flat = x_orig.permute(0, 2, 1).reshape(B * C, T)  # [B*C, T]
    freq_orig = compute_freq_magnitude(x_flat, dim=-1)  # [B*C, T//2+1]
    S_orig = compute_cosine_similarity_matrix(freq_orig).detach()  # [B*C, B*C] - 梯度隔离
    
    # ============ DFT支路：量化向量的频域相似度 ============
    # z_q: [B, num_patches, C, code_dim]
    # 对 code_dim 维度取平均，得到 [B, num_patches, C]
    z_q_reduced = z_q.mean(dim=-1)  # [B, num_patches, C]
    
    # 重排为 [B*C, num_patches] 以便计算FFT
    z_q_flat = z_q_reduced.permute(0, 2, 1).reshape(B * C, num_patches)  # [B*C, num_patches]
    
    # 对量化向量序列计算FFT幅值
    freq_quantized = compute_freq_magnitude(z_q_flat, dim=-1)  # [B*C, num_patches//2+1]
    
    # 计算余弦相似度矩阵
    S_quantized = compute_cosine_similarity_matrix(freq_quantized)  # [B*C, B*C]
    
    # ============ 计算损失 ============
    if args.freq_loss_type == 'mse':
        loss = compute_freq_consistency_loss_mse(S_orig, S_quantized)
    else:  # infonce
        loss = compute_freq_consistency_loss_infonce(
            S_orig, S_quantized, 
            threshold=args.freq_similarity_threshold,
            temperature=args.freq_temperature
        )
    
    info = {
        'S_orig_mean': S_orig.mean().item(),
        'S_quantized_mean': S_quantized.mean().item(),
        'S_orig_diag_mean': S_orig.diag().mean().item(),
        'S_quantized_diag_mean': S_quantized.diag().mean().item(),
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
    total_freq_loss = 0
    total_perplexity = 0
    n_batches = 0
    
    all_indices_list = []
    
    # 获取当前epoch的频域损失权重（带warmup）
    current_freq_weight = get_freq_weight_with_warmup(args, current_epoch)
    
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
        
        # 计算频域一致性损失（使用量化向量z_q，不是索引）
        freq_loss, freq_info = compute_inter_sequence_freq_loss(
            batch_x[:, :recon_len, :],  # 使用RevIN后的数据
            z_q,  # 量化后的向量 [B, num_patches, C, code_dim]
            args
        )
        
        # 总损失（使用warmup后的freq_weight）
        loss = (args.recon_weight * recon_loss + 
                args.vq_weight * vq_loss + 
                current_freq_weight * freq_loss)
        
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
        total_freq_loss += freq_loss.item()
        total_perplexity += perplexity
        n_batches += 1
    
    all_indices_epoch = torch.cat(all_indices_list, dim=0)
    codebook_stats = compute_codebook_usage_stats(all_indices_epoch, args.codebook_size)
    
    return {
        'loss': total_loss / n_batches if n_batches > 0 else 0.0,
        'vq_loss': total_vq_loss / n_batches if n_batches > 0 else 0.0,
        'recon_loss': total_recon_loss / n_batches if n_batches > 0 else 0.0,
        'freq_loss': total_freq_loss / n_batches if n_batches > 0 else 0.0,
        'perplexity': total_perplexity / n_batches if n_batches > 0 else 0.0,
        'codebook_stats': codebook_stats,
        'current_freq_weight': current_freq_weight,  # 返回当前使用的权重
    }


def validate_epoch(model, dataloader, revin, args, device, current_epoch=0):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    total_freq_loss = 0
    total_perplexity = 0
    n_batches = 0
    
    all_indices_list = []
    
    # 获取当前epoch的频域损失权重（带warmup）
    current_freq_weight = get_freq_weight_with_warmup(args, current_epoch)
    
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
            
            # 计算频域一致性损失（使用量化向量z_q）
            freq_loss, freq_info = compute_inter_sequence_freq_loss(
                batch_x[:, :recon_len, :],
                z_q,  # 量化后的向量 [B, num_patches, C, code_dim]
                args
            )
            
            loss = (args.recon_weight * recon_loss + 
                    args.vq_weight * vq_loss + 
                    current_freq_weight * freq_loss)
            
            unique_indices = torch.unique(indices.reshape(-1))
            perplexity = len(unique_indices) / args.codebook_size
            
            all_indices_list.append(indices.cpu())
            
            total_loss += loss.item()
            total_vq_loss += vq_loss.item()
            total_recon_loss += recon_loss.item()
            total_freq_loss += freq_loss.item()
            total_perplexity += perplexity
            n_batches += 1
    
    all_indices_epoch = torch.cat(all_indices_list, dim=0)
    codebook_stats = compute_codebook_usage_stats(all_indices_epoch, args.codebook_size)
    
    return {
        'loss': total_loss / n_batches if n_batches > 0 else 0.0,
        'vq_loss': total_vq_loss / n_batches if n_batches > 0 else 0.0,
        'recon_loss': total_recon_loss / n_batches if n_batches > 0 else 0.0,
        'freq_loss': total_freq_loss / n_batches if n_batches > 0 else 0.0,
        'perplexity': total_perplexity / n_batches if n_batches > 0 else 0.0,
        'codebook_stats': codebook_stats,
        'current_freq_weight': current_freq_weight,
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
    print('码本预训练（带序列间频域一致性损失）')
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
    freq_suffix = f"_freq{args.freq_weight}"
    model_name = f'codebook_ps{args.patch_size}_cb{args.codebook_size}_cd{code_dim}{freq_suffix}_model{args.model_id}'
    
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
    print(f'\n频域一致性损失配置:')
    print(f'  损失类型: {args.freq_loss_type}')
    print(f'  损失权重: {args.freq_weight}')
    print(f'  相似度阈值: {args.freq_similarity_threshold}')
    print(f'  温度系数: {args.freq_temperature}')
    
    print(f'\n软索引配置（解决argmax梯度断裂）:')
    if hasattr(args, 'use_soft_indices') and args.use_soft_indices:
        print(f'  ✓ 启用软索引')
        print(f'  方法: {args.soft_index_method}')
        if args.soft_index_method == 'gumbel':
            print(f'  Gumbel温度: {args.gumbel_temperature}')
            print(f'  Straight-Through: {bool(args.gumbel_hard)}')
        else:
            print(f'  Softmax温度: {args.soft_index_temperature}')
    else:
        print(f'  ✗ 使用硬索引（无梯度流向encoder）')
    
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
    
    print(f'\n开始训练，共 {args.n_epochs} 个 epoch (早停: {early_stop_patience} epochs)')
    print('=' * 80)
    
    for epoch in range(args.n_epochs):
        train_metrics = train_epoch(model, dls.train, optimizer, revin, args, device, scaler, current_epoch=epoch)
        scheduler.step()
        
        val_metrics = validate_epoch(model, dls.valid, revin, args, device, current_epoch=epoch)
        
        train_losses.append(train_metrics['loss'])
        valid_losses.append(val_metrics['loss'])
        
        # 获取当前频域损失权重（用于打印）
        current_freq_weight = train_metrics.get('current_freq_weight', args.freq_weight)
        
        # 打印进度（显示当前状态：delay/warmup/normal）
        delay_epochs = getattr(args, 'freq_delay_epochs', 20)
        warmup_epochs = getattr(args, 'freq_warmup_epochs', 10)
        if epoch < delay_epochs:
            warmup_info = f"[delay {epoch+1}/{delay_epochs}]"
        elif epoch < delay_epochs + warmup_epochs:
            warmup_info = f"[warmup {current_freq_weight:.4f}]"
        else:
            warmup_info = ""
        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"Train: total={train_metrics['loss']:.4f} "
              f"(recon={train_metrics['recon_loss']:.4f}, "
              f"vq={train_metrics['vq_loss']:.4f}, "
              f"freq={train_metrics['freq_loss']:.4f}{warmup_info}) | "
              f"Valid: total={val_metrics['loss']:.4f} "
              f"(recon={val_metrics['recon_loss']:.4f}, "
              f"freq={val_metrics['freq_loss']:.4f})")
        
        # 定期报告码本利用率
        if (epoch + 1) % args.codebook_report_interval == 0:
            train_stats = train_metrics.get('codebook_stats', {})
            val_stats = val_metrics.get('codebook_stats', {})
            train_usage = train_stats.get('usage_rate', 0.0) * 100
            val_usage = val_stats.get('usage_rate', 0.0) * 100
            print(f"  └─ 码本利用率: Train {train_usage:.1f}% | Valid {val_usage:.1f}%")
        
        # 保存最佳模型
        if epoch >= 5:
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
                    'train_freq_loss': train_metrics['freq_loss'],
                    'val_freq_loss': val_metrics['freq_loss'],
                }
                torch.save(checkpoint, save_dir / f'{model_name}.pth')
                print(f"  -> Best model saved (val_loss: {val_metrics['loss']:.4f})")
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    print(f"\n>>> 早停: val_loss 连续 {early_stop_patience} 个 epoch 未下降")
                    break
    
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
    print(f'最佳验证损失: {best_val_loss:.4f}')
    print(f'模型保存至: {save_dir / model_name}.pth')


if __name__ == '__main__':
    set_device()
    main()

