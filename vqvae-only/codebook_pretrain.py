"""
码本预训练脚本
独立训练 Encoder + Codebook (VQ) + Decoder
用于在decoder-only预训练之前先训练好码本
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
from src.models.codebook_model import CodebookModel, PerChannelCodebookModel
from src.models.layers.revin import RevIN
from src.basics import set_device
from datautils import get_dls


def parse_args():
    parser = argparse.ArgumentParser(description='码本预训练')
    
    # 数据集参数
    parser.add_argument('--dset', type=str, default='ettm1', help='数据集名称')
    parser.add_argument('--context_points', type=int, default=512, help='输入序列长度')
    parser.add_argument('--target_points', type=int, default=0, help='预测长度（码本预训练不使用，但datautils需要此参数）')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--scaler', type=str, default='standard', help='数据缩放方式')
    parser.add_argument('--features', type=str, default='M', help='特征类型')
    
    # 模型参数（与PatchVQVAETransformer一致）
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
                       help='码本初始化方法（random/normal/xavier/kaiming），random为完全随机初始化')
    parser.add_argument('--codebook_report_interval', type=int, default=5,
                       help='码本利用率报告间隔（每N个epoch报告一次）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机数种子（用于训练可复现性，但不影响码本初始化）')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN')
    parser.add_argument('--amp', type=int, default=1, help='是否启用混合精度')
    parser.add_argument('--vq_weight', type=float, default=1.0, help='VQ损失权重')
    parser.add_argument('--recon_weight', type=float, default=1.0, help='重构损失权重')
    
    # 数据采样参数（用于加速大数据集训练）
    parser.add_argument('--train_sample_ratio', type=float, default=1.0, 
                       help='训练集采样比例 (0.0-1.0)，例如0.1表示只使用10%%的训练数据')
    parser.add_argument('--valid_sample_ratio', type=float, default=1.0,
                       help='验证集采样比例 (0.0-1.0)，例如0.1表示只使用10%%的验证数据')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/vqvae_only/', help='模型保存路径')
    parser.add_argument('--model_id', type=int, default=1, help='模型ID')
    
    # Per-channel 码本
    parser.add_argument('--per_channel_codebook', type=int, default=0,
                        help='每个通道使用独立码本 (1启用, 0共享码本)')

    # RVQ 层数
    parser.add_argument('--n_rq_layers', type=int, default=1,
                        help='残差向量量化层数（1=普通VQ，2=2层RVQ，以此类推）')

    # Robust VQVAE: 稀疏分量
    parser.add_argument('--sparse_weight', type=float, default=0.0,
                        help='稀疏分量 L1 惩罚权重 λ（0=不启用 Robust 分解，建议初始值 0.01）')
    parser.add_argument('--sparse_amplitude', type=float, default=0.5,
                        help='SparseNet tanh 振幅上界（限制 s 的最大绝对值，建议 0.3~1.0）')

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
        'n_rq_layers': int(getattr(args, 'n_rq_layers', 1)),
        'sparse_weight': float(getattr(args, 'sparse_weight', 0.0)),
        'sparse_amplitude': float(getattr(args, 'sparse_amplitude', 0.5)),
    }
    return config


def compute_codebook_usage_stats(indices, codebook_size):
    """
    计算码本利用率统计信息（所有通道合并，适用于共享码本）

    Args:
        indices: [B, num_patches, C, n_rq_layers] 码本索引
        codebook_size: 码本大小
    Returns:
        dict
    """
    # 对所有层取平均利用率
    n_rq = indices.shape[-1]
    all_usages, all_counts = [], []
    for l in range(n_rq):
        idx_l = indices[..., l].reshape(-1).cpu()
        unique_l = torch.unique(idx_l)
        all_usages.append(len(unique_l) / codebook_size)
        all_counts.append(torch.bincount(idx_l, minlength=codebook_size))

    avg_usage = sum(all_usages) / n_rq
    # top5 based on layer 0
    counts0 = all_counts[0]
    num_unused = (counts0 == 0).sum().item()
    top5_counts, top5_indices = torch.topk(counts0, k=min(5, codebook_size))
    top5_usage = [(idx.item(), cnt.item()) for idx, cnt in zip(top5_indices, top5_counts) if cnt > 0]

    per_layer_usage = [f'L{l}:{u*100:.1f}%' for l, u in enumerate(all_usages)]

    return {
        'num_used': int(avg_usage * codebook_size),
        'num_unused': num_unused,
        'usage_rate': avg_usage,
        'top5_usage': top5_usage,
        'total_tokens': indices.numel(),
        'per_layer_usage': per_layer_usage,
    }


def compute_per_channel_usage_stats(indices, codebook_size):
    """
    计算 per-channel 码本利用率统计信息

    Args:
        indices: [B, num_patches, C, n_rq_layers] 码本索引
        codebook_size: 码本大小
    Returns:
        dict
    """
    B, num_patches, C, n_rq = indices.shape
    per_ch_usage = []
    per_ch_num_used = []
    per_ch_num_unused = []

    # 对所有层取平均
    for c in range(C):
        layer_usages = []
        for l in range(n_rq):
            idx_cl = indices[:, :, c, l].reshape(-1).cpu()
            unique_cl = torch.unique(idx_cl)
            layer_usages.append(len(unique_cl) / codebook_size)
        avg_u = sum(layer_usages) / n_rq
        per_ch_usage.append(avg_u)
        per_ch_num_used.append(int(avg_u * codebook_size))
        per_ch_num_unused.append(codebook_size - int(avg_u * codebook_size))

    avg_usage = sum(per_ch_usage) / C

    return {
        'usage_rate': avg_usage,
        'num_used': sum(per_ch_num_used) / C,
        'num_unused': sum(per_ch_num_unused) / C,
        'per_channel_usage': per_ch_usage,
        'per_channel_num_used': per_ch_num_used,
        'top5_usage': [],
        'total_tokens': indices.numel(),
    }


def train_epoch(model, dataloader, optimizer, revin, args, device, scaler):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    total_perplexity = 0
    total_sparse_norm = 0
    n_batches = 0
    
    # 用于累积码本使用统计
    all_indices_list = []
    per_channel = bool(args.per_channel_codebook)
    use_sparse = getattr(args, 'sparse_weight', 0.0) > 0
    
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)  # [B, T, C]
        
        if revin:
            batch_x = revin(batch_x, 'norm')
        
        # 编码和解码（只训练encoder、vq、decoder）
        if use_sparse:
            indices, vq_loss, z_q, s = model.encode_to_indices(batch_x, return_sparse=True)
        else:
            indices, vq_loss, z_q = model.encode_to_indices(batch_x)
            s = None
        x_recon = model.decode_from_codes(z_q)  # [B, num_patches * patch_size, C]
        if s is not None:
            recon_len = x_recon.shape[1]
            x_recon = x_recon + s[:, :recon_len, :]  # add sparse component back
        
        # 计算重构损失
        B, T, C = batch_x.shape
        num_patches = indices.shape[1]
        recon_len = num_patches * model.patch_size
        recon_loss = F.mse_loss(x_recon, batch_x[:, :recon_len, :])

        # 稀疏 L1 惩罚
        if s is not None:
            sparse_norm = s.abs().mean()
            loss = args.recon_weight * recon_loss + args.vq_weight * vq_loss + args.sparse_weight * sparse_norm
        else:
            sparse_norm = torch.tensor(0.0)
            # 总损失
            loss = args.recon_weight * recon_loss + args.vq_weight * vq_loss
        
        # 反向传播（只对可训练参数）
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
        
        # 计算 perplexity（per-channel 时取各通道平均；取第 0 层索引计算）
        if per_channel:
            ch_usages = [
                len(torch.unique(indices[:, :, c, 0])) / args.codebook_size
                for c in range(indices.shape[2])
            ]
            perplexity = sum(ch_usages) / len(ch_usages)
        else:
            unique_indices = torch.unique(indices[:, :, :, 0].reshape(-1))
            perplexity = len(unique_indices) / args.codebook_size
        
        # 累积索引用于统计
        all_indices_list.append(indices.detach().cpu())
        
        total_loss += loss.item()
        total_vq_loss += vq_loss.item()
        total_recon_loss += recon_loss.item()
        total_perplexity += perplexity
        total_sparse_norm += sparse_norm.item()
        n_batches += 1
    
    # 计算整个epoch的码本利用率统计
    all_indices_epoch = torch.cat(all_indices_list, dim=0)  # [total_B, num_patches, C]
    if per_channel:
        codebook_stats = compute_per_channel_usage_stats(all_indices_epoch, args.codebook_size)
    else:
        codebook_stats = compute_codebook_usage_stats(all_indices_epoch, args.codebook_size)
    
    return {
        'loss': total_loss / n_batches if n_batches > 0 else 0.0,
        'vq_loss': total_vq_loss / n_batches if n_batches > 0 else 0.0,
        'recon_loss': total_recon_loss / n_batches if n_batches > 0 else 0.0,
        'perplexity': total_perplexity / n_batches if n_batches > 0 else 0.0,
        'sparse_norm': total_sparse_norm / n_batches if n_batches > 0 else 0.0,
        'codebook_stats': codebook_stats,
    }


def validate_epoch(model, dataloader, revin, args, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    total_perplexity = 0
    total_sparse_norm = 0
    n_batches = 0
    
    # 用于累积码本使用统计
    all_indices_list = []
    per_channel = bool(args.per_channel_codebook)
    use_sparse = getattr(args, 'sparse_weight', 0.0) > 0
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)  # [B, T, C]
            
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            # 编码和解码
            if use_sparse:
                indices, vq_loss, z_q, s = model.encode_to_indices(batch_x, return_sparse=True)
            else:
                indices, vq_loss, z_q = model.encode_to_indices(batch_x)
                s = None
            x_recon = model.decode_from_codes(z_q)  # [B, num_patches * patch_size, C]
            if s is not None:
                recon_len = x_recon.shape[1]
                x_recon = x_recon + s[:, :recon_len, :]
            
            # 计算重构损失
            B, T, C = batch_x.shape
            num_patches = indices.shape[1]
            recon_len = num_patches * model.patch_size
            recon_loss = F.mse_loss(x_recon, batch_x[:, :recon_len, :])

            if s is not None:
                sparse_norm = s.abs().mean()
                loss = args.recon_weight * recon_loss + args.vq_weight * vq_loss + args.sparse_weight * sparse_norm
            else:
                sparse_norm = torch.tensor(0.0)
                # 总损失
                loss = args.recon_weight * recon_loss + args.vq_weight * vq_loss
            
            # 计算 perplexity
            if per_channel:
                ch_usages = [
                    len(torch.unique(indices[:, :, c, 0])) / args.codebook_size
                    for c in range(indices.shape[2])
                ]
                perplexity = sum(ch_usages) / len(ch_usages)
            else:
                unique_indices = torch.unique(indices[:, :, :, 0].reshape(-1))
                perplexity = len(unique_indices) / args.codebook_size
            
            # 累积索引用于统计
            all_indices_list.append(indices.cpu())
            
            total_loss += loss.item()
            total_vq_loss += vq_loss.item()
            total_recon_loss += recon_loss.item()
            total_perplexity += perplexity
            total_sparse_norm += sparse_norm.item()
            n_batches += 1
    
    # 计算整个epoch的码本利用率统计
    all_indices_epoch = torch.cat(all_indices_list, dim=0)  # [total_B, num_patches, C]
    if per_channel:
        codebook_stats = compute_per_channel_usage_stats(all_indices_epoch, args.codebook_size)
    else:
        codebook_stats = compute_codebook_usage_stats(all_indices_epoch, args.codebook_size)
    
    return {
        'loss': total_loss / n_batches if n_batches > 0 else 0.0,
        'vq_loss': total_vq_loss / n_batches if n_batches > 0 else 0.0,
        'recon_loss': total_recon_loss / n_batches if n_batches > 0 else 0.0,
        'perplexity': total_perplexity / n_batches if n_batches > 0 else 0.0,
        'sparse_norm': total_sparse_norm / n_batches if n_batches > 0 else 0.0,
        'codebook_stats': codebook_stats,
    }


def set_seed(seed):
    """
    设置随机数种子以确保训练可复现性
    注意：此种子不影响码本初始化（init_from_data），码本初始化保持随机性
    
    Args:
        seed: 随机数种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保CUDA操作的确定性（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置Python的hash随机化（用于字典等）
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ 随机数种子已设置为: {seed}（不影响码本初始化）")


def worker_init_fn(worker_id):
    """
    数据加载器worker的初始化函数
    
    Args:
        worker_id: worker的ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    args = parse_args()
    print('Args:', args)
    
    # PyTorch 2.7+ 兼容性修复：禁用 flash attention 和 memory-efficient attention
    # 当使用 mask 时，这些优化可能导致 CUDA 错误
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
        print('✓ 已禁用 flash/memory-efficient attention（PyTorch 2.7+ 兼容性修复）')
    
    # 设置随机数种子（用于训练可复现性，但不影响码本初始化）
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建保存目录
    save_dir = Path(args.save_path) / args.dset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型文件名
    code_dim = args.embedding_dim * (args.patch_size // args.compression_factor)
    per_ch_suffix = '_perch' if args.per_channel_codebook else ''
    rvq_suffix = f'_rvq{args.n_rq_layers}' if getattr(args, 'n_rq_layers', 1) > 1 else ''
    model_name = f'codebook_ps{args.patch_size}_cb{args.codebook_size}_cd{code_dim}{per_ch_suffix}{rvq_suffix}_model{args.model_id}'
    
    # 获取数据
    args.dset_pretrain = args.dset
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Train batches: {len(dls.train)}, Valid batches: {len(dls.valid)}')
    
    # 对训练集和验证集进行采样（如果指定了采样比例）
    if args.train_sample_ratio < 1.0 or args.valid_sample_ratio < 1.0:
        # 采样训练集
        if args.train_sample_ratio < 1.0:
            train_dataset = dls.train.dataset
            train_size = len(train_dataset)
            sample_size = int(train_size * args.train_sample_ratio)
            # 随机采样
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
        
        # 采样验证集
        if args.valid_sample_ratio < 1.0:
            valid_dataset = dls.valid.dataset
            valid_size = len(valid_dataset)
            sample_size = int(valid_size * args.valid_sample_ratio)
            # 随机采样
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
        
        print(f'采样后 - Train batches: {len(dls.train)}, Valid batches: {len(dls.valid)}')
    
    # 创建轻量级码本模型（只包含encoder、vq、decoder）
    config = get_model_config(args)
    if args.per_channel_codebook:
        model = PerChannelCodebookModel(config, dls.vars).to(device)
        print(f'\n模式: Per-Channel 码本（每通道独立 VQ，共 {dls.vars} 个码本）')
    else:
        model = CodebookModel(config, dls.vars).to(device)
        print(f'\n模式: 共享码本（所有通道使用同一 VQ）')
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # 检查各层的可训练参数
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    encoder_total = sum(p.numel() for p in model.encoder.parameters())
    decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    decoder_total = sum(p.numel() for p in model.decoder.parameters())
    
    if args.per_channel_codebook:
        vq_trainable = sum(p.numel() for vq in model.vqs for p in vq.parameters() if p.requires_grad)
        vq_total = sum(p.numel() for vq in model.vqs for p in vq.parameters())
    else:
        vq_trainable = sum(p.numel() for p in model.vq.parameters() if p.requires_grad)
        vq_total = sum(p.numel() for p in model.vq.parameters())
    
    print(f'\n码本模型参数统计:')
    print(f'  总参数: {total_params:,}')
    print(f'  可训练参数: {trainable_params:,}')
    print(f'  冻结参数: {frozen_params:,}')
    print(f'  Encoder: {encoder_total:,} (可训练: {encoder_trainable:,})')
    print(f'  Decoder: {decoder_total:,} (可训练: {decoder_trainable:,})')
    print(f'  VQ层: {vq_total:,} (可训练: {vq_trainable:,})')
    print(f'  码本初始化方法: {args.vq_init_method}')
    print(f'  使用EMA: {bool(args.codebook_ema)}')
    
    # 检查是否有可训练参数
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params_list) == 0:
        raise ValueError(
            "错误: 没有可训练参数！\n"
            f"  - codebook_ema: {args.codebook_ema}\n"
            f"  - VQ层参数数量: {vq_total:,}\n"
            f"  - VQ层可训练参数数量: {vq_trainable:,}\n"
            "\n解决方案：禁用EMA: --codebook_ema 0"
        )
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 优化器和调度器（优化所有可训练参数）
    optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    
    # AMP
    scaler = amp.GradScaler(enabled=bool(args.amp))
    
    # 训练
    best_val_loss = float('inf')  # 跟踪验证集上最小的val_loss
    train_losses, valid_losses = [], []
    train_recon_losses, valid_recon_losses = [], []  # 记录recon_loss历史
    no_improve_count = 0
    early_stop_patience = 10
    model_saved = False
    
    print(f'\n开始码本预训练，共 {args.n_epochs} 个 epoch (早停: {early_stop_patience} epochs)')
    print('=' * 80)
    
    for epoch in range(args.n_epochs):
        # 训练
        train_metrics = train_epoch(model, dls.train, optimizer, revin, args, device, scaler)
        scheduler.step()
        
        # 验证
        val_metrics = validate_epoch(model, dls.valid, revin, args, device)
        
        train_losses.append(train_metrics['loss'])
        valid_losses.append(val_metrics['loss'])
        train_recon_losses.append(train_metrics['recon_loss'])
        valid_recon_losses.append(val_metrics['recon_loss'])
        
        # 打印进度
        train_stats = train_metrics.get('codebook_stats', {})
        val_stats = val_metrics.get('codebook_stats', {})
        
        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} (Recon: {train_metrics['recon_loss']:.4f}, "
              f"VQ: {train_metrics['vq_loss']:.4f}, Perplexity: {train_metrics['perplexity']:.3f}) | "
              f"Valid Loss: {val_metrics['loss']:.4f} (Recon: {val_metrics['recon_loss']:.4f}, "
              f"VQ: {val_metrics['vq_loss']:.4f}, Perplexity: {val_metrics['perplexity']:.3f})")

        # Robust VQVAE: 打印稀疏分量统计
        if getattr(args, 'sparse_weight', 0.0) > 0:
            print(f"  └─ SparseNorm (L1): Train {train_metrics['sparse_norm']:.5f} | "
                  f"Valid {val_metrics['sparse_norm']:.5f}")
        
        # 定期报告码本利用率（每5个epoch或每10个epoch）
        report_interval = getattr(args, 'codebook_report_interval', 5)
        if (epoch + 1) % report_interval == 0 or epoch == 0:
            train_usage = train_stats.get('usage_rate', 0.0) * 100
            val_usage = val_stats.get('usage_rate', 0.0) * 100
            train_used = train_stats.get('num_used', 0)
            val_used = val_stats.get('num_used', 0)
            train_unused = train_stats.get('num_unused', 0)
            val_unused = val_stats.get('num_unused', 0)
            
            print(f"  └─ 码本利用率(avg): Train {train_usage:.1f}% ({train_used:.0f}/{args.codebook_size}) | "
                  f"Valid {val_usage:.1f}% ({val_used:.0f}/{args.codebook_size})")

            # 多层 RVQ 时打印每层独立利用率
            per_layer = train_stats.get('per_layer_usage', [])
            if len(per_layer) > 1:
                print(f"  └─ 各层利用率 (Train): {', '.join(per_layer)}")

            if args.per_channel_codebook:
                per_ch_train = train_stats.get('per_channel_usage', [])
                per_ch_val   = val_stats.get('per_channel_usage', [])
                if per_ch_train:
                    ch_strs = [f"ch{c}:{u*100:.0f}%" for c, u in enumerate(per_ch_train)]
                    print(f"  └─ 各通道利用率 (Train): {', '.join(ch_strs)}")
            else:
                # 显示最常用的码本元素（仅训练集）
                if train_stats.get('top5_usage'):
                    top5_str = ', '.join([f"#{idx}({cnt})" for idx, cnt in train_stats['top5_usage'][:5]])
                    print(f"  └─ 最常用码本元素 (Train): {top5_str}")
        
        # 基于val_loss保存最佳模型
        if epoch >= 5:  # 前5个epoch不保存
            current_val_loss = val_metrics['loss']
            
            # 如果val_loss下降，保存模型
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                no_improve_count = 0
                model_saved = True
                
                # 只保存encoder、decoder和vq的权重
                if args.per_channel_codebook:
                    checkpoint = {
                        'encoder_state_dict': model.encoder.state_dict(),
                        'decoder_state_dict': model.decoder.state_dict(),
                        # model_state_dict 包含 vqs.0.*, vqs.1.*, ...
                        # 可被 PatchVQVAETransformer.load_vqvae_weights() 直接加载
                        'model_state_dict': model.state_dict(),
                        'n_channels': dls.vars,
                        'config': config,
                        'args': vars(args),
                        'epoch': epoch,
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'train_recon_loss': train_metrics['recon_loss'],
                        'val_recon_loss': val_metrics['recon_loss'],
                    }
                else:
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
                    }
                torch.save(checkpoint, save_dir / f'{model_name}.pth')
                print(f"  -> Best model saved (val_loss: {val_metrics['loss']:.4f})")
            else:
                # val_loss不再下降，不再保存模型
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
        'train_recon_loss': train_recon_losses,
        'valid_recon_loss': valid_recon_losses,
    })
    history_df.to_csv(save_dir / f'{model_name}_history.csv', index=False)
    
    # 保存配置
    with open(save_dir / f'{model_name}_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print('=' * 80)
    print(f'码本预训练完成！')
    print(f'最佳验证损失: {best_val_loss:.4f}')
    print(f'模型保存至: {save_dir / model_name}.pth')


if __name__ == '__main__':
    set_device()
    main()
