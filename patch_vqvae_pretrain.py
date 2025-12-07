"""
Patch-based VQVAE + Transformer 预训练脚本
使用 Next Token Prediction (NTP) 损失
"""

import numpy as np
import pandas as pd
import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from pathlib import Path

from src.models.patch_vqvae_transformer import PatchVQVAETransformer
from src.models.layers.revin import RevIN
from src.basics import set_device
from datautils import get_dls


def parse_args():
    parser = argparse.ArgumentParser(description='Patch VQVAE Transformer 预训练')
    
    # 数据集参数
    parser.add_argument('--dset', type=str, default='ettm1', help='数据集名称')
    parser.add_argument('--context_points', type=int, default=512, help='输入序列长度')
    parser.add_argument('--target_points', type=int, default=96, help='预测长度（预训练时不使用）')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--scaler', type=str, default='standard', help='数据缩放方式')
    parser.add_argument('--features', type=str, default='M', help='特征类型')
    
    # 模型参数 (Transformer输入维度 = embedding_dim * patch_size/compression_factor)
    parser.add_argument('--patch_size', type=int, default=16, help='Patch大小')
    parser.add_argument('--embedding_dim', type=int, default=32, help='VQVAE embedding维度')
    parser.add_argument('--compression_factor', type=int, default=4, help='压缩因子')
    parser.add_argument('--codebook_size', type=int, default=256, help='码本大小')
    parser.add_argument('--n_layers', type=int, default=4, help='Transformer层数')
    parser.add_argument('--n_heads', type=int, default=4, help='注意力头数 (需整除 code_dim)')
    parser.add_argument('--d_ff', type=int, default=256, help='FFN维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='VQ commitment cost')
    
    # VQVAE Encoder/Decoder 参数 (轻量化)
    parser.add_argument('--num_hiddens', type=int, default=64, help='VQVAE隐藏层维度')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='残差层数')
    parser.add_argument('--num_residual_hiddens', type=int, default=32, help='残差隐藏层维度')
    parser.add_argument('--vqvae_checkpoint', type=str, default=None, help='预训练VQVAE模型路径(可选)')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN')
    parser.add_argument('--vq_weight', type=float, default=1.0, help='VQ损失权重')
    parser.add_argument('--recon_weight', type=float, default=0.1, help='重构损失权重')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/patch_vqvae/', help='模型保存路径')
    parser.add_argument('--model_id', type=int, default=1, help='模型ID')
    
    return parser.parse_args()


def get_model_config(args):
    """构建模型配置"""
    # code_dim = embedding_dim * (patch_size / compression_factor)
    code_dim = args.embedding_dim * (args.patch_size // args.compression_factor)
    print(f"Transformer 输入维度 (code_dim) = {code_dim}")
    
    return {
        'patch_size': args.patch_size,
        'embedding_dim': args.embedding_dim,
        'compression_factor': args.compression_factor,
        'codebook_size': args.codebook_size,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'commitment_cost': args.commitment_cost,
        # VQVAE Encoder/Decoder 配置
        'num_hiddens': args.num_hiddens,
        'num_residual_layers': args.num_residual_layers,
        'num_residual_hiddens': args.num_residual_hiddens,
    }


def train_epoch(model, dataloader, optimizer, scheduler, revin, args, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_ntp_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    n_batches = 0
    
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)  # [B, T, C]
        
        # RevIN归一化
        if revin:
            batch_x = revin(batch_x, 'norm')
        
        # 前向传播
        logits, targets, vq_loss, recon_loss = model.forward_pretrain(batch_x)
        # logits: [B, num_patches-1, C, codebook_size]
        # targets: [B, num_patches-1, C]
        
        # NTP损失 (CrossEntropy)
        B, num_patches_m1, C, codebook_size = logits.shape
        logits_flat = logits.reshape(-1, codebook_size)  # [B*(num_patches-1)*C, codebook_size]
        targets_flat = targets.reshape(-1)  # [B*(num_patches-1)*C]
        ntp_loss = F.cross_entropy(logits_flat, targets_flat)
        
        # 总损失
        loss = ntp_loss + args.vq_weight * vq_loss + args.recon_weight * recon_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_ntp_loss += ntp_loss.item()
        total_vq_loss += vq_loss.item()
        total_recon_loss += recon_loss.item()
        n_batches += 1
    
    scheduler.step()
    
    return {
        'loss': total_loss / n_batches,
        'ntp_loss': total_ntp_loss / n_batches,
        'vq_loss': total_vq_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
    }


def validate_epoch(model, dataloader, revin, args, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_ntp_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            logits, targets, vq_loss, recon_loss = model.forward_pretrain(batch_x)
            
            B, num_patches_m1, C, codebook_size = logits.shape
            logits_flat = logits.reshape(-1, codebook_size)
            targets_flat = targets.reshape(-1)
            ntp_loss = F.cross_entropy(logits_flat, targets_flat)
            
            loss = ntp_loss + args.vq_weight * vq_loss + args.recon_weight * recon_loss
            
            total_loss += loss.item()
            total_ntp_loss += ntp_loss.item()
            total_vq_loss += vq_loss.item()
            total_recon_loss += recon_loss.item()
            n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'ntp_loss': total_ntp_loss / n_batches,
        'vq_loss': total_vq_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
    }


def main():
    args = parse_args()
    print('Args:', args)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建保存目录
    save_dir = Path(args.save_path) / args.dset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型文件名 (code_dim = embedding_dim * patch_size / compression_factor)
    code_dim = args.embedding_dim * (args.patch_size // args.compression_factor)
    model_name = f'patch_vqvae_ps{args.patch_size}_cb{args.codebook_size}_cd{code_dim}_l{args.n_layers}_model{args.model_id}'
    
    # 获取数据
    args.dset_pretrain = args.dset
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Train batches: {len(dls.train)}, Valid batches: {len(dls.valid)}')
    
    # 创建模型
    config = get_model_config(args)
    model = PatchVQVAETransformer(config).to(device)
    
    # 可选：加载预训练的 VQVAE 权重
    if args.vqvae_checkpoint:
        model.load_vqvae_weights(args.vqvae_checkpoint, device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    
    # 训练
    best_val_loss = float('inf')
    train_losses, valid_losses = [], []
    
    print(f'\n开始预训练，共 {args.n_epochs} 个 epoch')
    print('=' * 80)
    
    for epoch in range(args.n_epochs):
        # 训练
        train_metrics = train_epoch(model, dls.train, optimizer, scheduler, revin, args, device)
        
        # 验证
        val_metrics = validate_epoch(model, dls.valid, revin, args, device)
        
        train_losses.append(train_metrics['loss'])
        valid_losses.append(val_metrics['loss'])
        
        # 打印进度
        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} (NTP: {train_metrics['ntp_loss']:.4f}, "
              f"VQ: {train_metrics['vq_loss']:.4f}, Recon: {train_metrics['recon_loss']:.4f}) | "
              f"Valid Loss: {val_metrics['loss']:.4f}")
        
        # 保存最佳模型 (前20个epoch不保存，根据train_loss和val_loss均值判断)
        avg_loss = (train_metrics['loss'] + val_metrics['loss']) / 2
        if epoch >= 20 and avg_loss < best_val_loss:
            best_val_loss = avg_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'args': vars(args),
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'avg_loss': avg_loss,
            }
            torch.save(checkpoint, save_dir / f'{model_name}.pth')
            print(f"  -> Best model saved (avg_loss: {avg_loss:.4f}, train: {train_metrics['loss']:.4f}, val: {val_metrics['loss']:.4f})")
        
        # 每10个epoch检查码本使用率
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_batch = next(iter(dls.train))[0].to(device)
                if revin:
                    sample_batch = revin(sample_batch, 'norm')
                usage, _ = model.get_codebook_usage(sample_batch)
                print(f"  -> Codebook usage: {usage*100:.1f}%")
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, args.n_epochs + 1),
        'train_loss': train_losses,
        'valid_loss': valid_losses,
    })
    history_df.to_csv(save_dir / f'{model_name}_history.csv', index=False)
    
    # 保存配置
    with open(save_dir / f'{model_name}_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print('=' * 80)
    print(f'预训练完成！')
    print(f'最佳平均损失 (train_loss+val_loss)/2: {best_val_loss:.4f}')
    print(f'模型保存至: {save_dir / model_name}.pth')


if __name__ == '__main__':
    set_device()
    main()
