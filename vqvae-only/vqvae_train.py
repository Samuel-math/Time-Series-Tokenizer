"""
VQVAE 训练脚本
独立训练 VQVAE 模型（Encoder + Vector Quantizer + Decoder）
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
import argparse
from pathlib import Path

# 添加根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.vqvae import vqvae
from src.models.layers.revin import RevIN
from src.basics import set_device
from datautils import get_dls


def parse_args():
    parser = argparse.ArgumentParser(description='VQVAE 训练')
    
    # 数据集参数
    parser.add_argument('--dset', type=str, default='ettm1', help='数据集名称')
    parser.add_argument('--context_points', type=int, default=512, help='输入序列长度')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--scaler', type=str, default='standard', help='数据缩放方式')
    parser.add_argument('--features', type=str, default='M', help='特征类型')
    
    # VQVAE 模型参数
    parser.add_argument('--patch_size', type=int, default=16, help='Patch大小（时间步数）')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Embedding维度')
    parser.add_argument('--num_embeddings', type=int, default=256, help='码本大小')
    parser.add_argument('--compression_factor', type=int, default=4, choices=[4, 8, 12, 16], help='压缩因子')
    parser.add_argument('--block_hidden_size', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='残差层数')
    parser.add_argument('--res_hidden_size', type=int, default=32, help='残差隐藏层维度')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='VQ commitment cost')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN')
    parser.add_argument('--amp', type=int, default=1, help='是否启用混合精度')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/vqvae_only/', help='模型保存路径')
    parser.add_argument('--model_id', type=int, default=1, help='模型ID')
    
    return parser.parse_args()


def get_vqvae_config(args, n_channels):
    """构建VQVAE配置"""
    config = {
        'patch_size': args.patch_size,
        'embedding_dim': args.embedding_dim,
        'num_embeddings': args.num_embeddings,
        'compression_factor': args.compression_factor,
        'block_hidden_size': args.block_hidden_size,
        'num_residual_layers': args.num_residual_layers,
        'res_hidden_size': args.res_hidden_size,
        'commitment_cost': args.commitment_cost,
        'n_channels': n_channels,
    }
    return config


def create_patches(x, patch_size):
    """
    将时间序列划分为patches
    
    Args:
        x: [B, T] 单通道时间序列
        patch_size: patch大小
    Returns:
        patches: [B, num_patches, patch_size]
    """
    B, T = x.shape
    num_patches = T // patch_size
    x = x[:, :num_patches * patch_size]  # 截断到patch_size的倍数
    patches = x.reshape(B, num_patches, patch_size)
    return patches


def patches_to_sequence(patches):
    """
    将patches重组为时间序列
    
    Args:
        patches: [B, num_patches, patch_size]
    Returns:
        x: [B, num_patches * patch_size]
    """
    B, num_patches, patch_size = patches.shape
    x = patches.reshape(B, num_patches * patch_size)
    return x


def train_epoch(model, dataloader, optimizer, revin, args, device, scaler):
    """训练一个epoch"""
    model.train()
    total_loss_sum = 0
    total_vq_loss_sum = 0
    total_recon_loss_sum = 0
    total_perplexity = 0
    total_samples = 0
    
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)  # [B, T, C]
        B, T, C = batch_x.shape
        batch_size = B
        
        # VQVAE当前只支持单通道输入，我们处理第一个通道
        # 如果需要处理所有通道，可以为每个通道训练独立的模型
        x = batch_x[:, :, 0]  # [B, T] - 使用第一个通道
        
        # RevIN归一化
        if revin:
            x = revin(x.unsqueeze(-1), 'norm')  # [B, T, 1]
            x = x.squeeze(-1)  # [B, T]
        
        # 将序列划分为patches
        patches = create_patches(x, args.patch_size)  # [B, num_patches, patch_size]
        B, num_patches, patch_size = patches.shape
        
        # 将所有patches展平为 [B*num_patches, patch_size] 进行批量处理
        patches_flat = patches.reshape(B * num_patches, patch_size)  # [B*num_patches, patch_size]
        
        with amp.autocast(enabled=scaler.is_enabled()):
            # 批量编码所有patches
            z = model.encoder(patches_flat.unsqueeze(1), model.compression_factor)  # [B*num_patches, 1, patch_size] -> [B*num_patches, embedding_dim, patch_size/compression_factor]
            vq_loss, quantized, perplexity, _, _, _ = model.vq(z)
            
            # 批量解码所有patches
            patches_recon = model.decoder(quantized, model.compression_factor)  # [B*num_patches, patch_size]
            
            # 计算重构误差（使用sum以便按样本数加权）
            recon_error = F.mse_loss(patches_recon, patches_flat, reduction='sum')
            vq_loss_sum = vq_loss * (B * num_patches)  # vq_loss是mean，转换为sum
            
            # 总损失
            loss = recon_error + vq_loss_sum
        
        # 反向传播（一次性处理所有patches）
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # 累加loss（用于报告）
        total_loss_sum += loss.item()
        total_vq_loss_sum += vq_loss_sum.item()
        total_recon_loss_sum += recon_error.item()
        total_perplexity += perplexity.item()
        total_samples += batch_size
    
    return {
        'loss': total_loss_sum / total_samples if total_samples > 0 else 0.0,
        'vq_loss': total_vq_loss_sum / total_samples if total_samples > 0 else 0.0,
        'recon_loss': total_recon_loss_sum / total_samples if total_samples > 0 else 0.0,
        'perplexity': total_perplexity / len(dataloader) if len(dataloader) > 0 else 0.0,
    }


def validate_epoch(model, dataloader, revin, args, device, use_amp):
    """验证一个epoch"""
    model.eval()
    total_loss_sum = 0
    total_vq_loss_sum = 0
    total_recon_loss_sum = 0
    total_perplexity = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)  # [B, T, C]
            B, T, C = batch_x.shape
            batch_size = B
            
            # 使用第一个通道
            x = batch_x[:, :, 0]  # [B, T]
            
            if revin:
                x = revin(x.unsqueeze(-1), 'norm')
                x = x.squeeze(-1)
            
            # 将序列划分为patches
            patches = create_patches(x, args.patch_size)  # [B, num_patches, patch_size]
            B, num_patches, patch_size = patches.shape
            
            # 将所有patches展平为 [B*num_patches, patch_size] 进行批量处理
            patches_flat = patches.reshape(B * num_patches, patch_size)  # [B*num_patches, patch_size]
            
            with amp.autocast(enabled=use_amp):
                # 批量编码所有patches
                z = model.encoder(patches_flat.unsqueeze(1), model.compression_factor)
                vq_loss, quantized, perplexity, _, _, _ = model.vq(z)
                
                # 批量解码所有patches
                patches_recon = model.decoder(quantized, model.compression_factor)
                
                # 计算损失
                recon_error = F.mse_loss(patches_recon, patches_flat, reduction='sum')
                vq_loss_sum = vq_loss * (B * num_patches)
                
                loss = recon_error + vq_loss_sum
            
            total_loss_sum += loss.item()
            total_vq_loss_sum += vq_loss_sum.item()
            total_recon_loss_sum += recon_error.item()
            total_perplexity += perplexity.item()
            total_samples += batch_size
    
    return {
        'loss': total_loss_sum / total_samples if total_samples > 0 else 0.0,
        'vq_loss': total_vq_loss_sum / total_samples if total_samples > 0 else 0.0,
        'recon_loss': total_recon_loss_sum / total_samples if total_samples > 0 else 0.0,
        'perplexity': total_perplexity / len(dataloader) if len(dataloader) > 0 else 0.0,
    }


def main():
    args = parse_args()
    print('Args:', args)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('medium')
    
    # 创建保存目录
    save_dir = Path(args.save_path) / args.dset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取数据
    args.dset_finetune = args.dset
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Train batches: {len(dls.train)}, Valid batches: {len(dls.valid)}')
    
    # 创建VQVAE配置
    config = get_vqvae_config(args, dls.vars)
    
    # 创建模型（VQVAE当前只支持单通道输入）
    # 注意：当前实现只训练第一个通道的VQVAE模型
    # 如果需要为每个通道训练独立的模型，可以扩展此代码
    model = vqvae(config).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\n模型参数统计:')
    print(f'  总参数: {total_params:,}')
    print(f'  配置: {config}')
    print(f'  注意: 当前只训练第一个通道（共{dls.vars}个通道）')
    
    # AMP
    use_amp = bool(args.amp) and device.type == 'cuda'
    scaler = amp.GradScaler(enabled=use_amp)
    print(f'AMP enabled: {use_amp}')
    
    # RevIN（为每个通道创建）
    revin = RevIN(1, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 模型文件名
    model_name = f'vqvae_cw{args.context_points}_ps{args.patch_size}_ed{args.embedding_dim}_cf{args.compression_factor}_cb{args.num_embeddings}_model{args.model_id}'
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    
    # 训练
    best_val_loss = float('inf')
    train_losses, valid_losses = [], []
    train_vq_losses, valid_vq_losses = [], []
    train_recon_losses, valid_recon_losses = [], []
    no_improve_count = 0
    early_stop_patience = 10
    
    print(f'\n开始训练，共 {args.n_epochs} 个 epoch (早停: {early_stop_patience} epochs)')
    print('=' * 80)
    
    for epoch in range(args.n_epochs):
        # 训练
        train_metrics = train_epoch(model, dls.train, optimizer, revin, args, device, scaler)
        
        # 验证
        val_metrics = validate_epoch(model, dls.valid, revin, args, device, use_amp)
        
        # 更新学习率
        scheduler.step()
        
        train_losses.append(train_metrics['loss'])
        valid_losses.append(val_metrics['loss'])
        train_vq_losses.append(train_metrics['vq_loss'])
        valid_vq_losses.append(val_metrics['vq_loss'])
        train_recon_losses.append(train_metrics['recon_loss'])
        valid_recon_losses.append(val_metrics['recon_loss'])
        
        # 打印进度
        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"Train Loss: {train_metrics['loss']:.6f} (VQ: {train_metrics['vq_loss']:.6f}, "
              f"Recon: {train_metrics['recon_loss']:.6f}, Perp: {train_metrics['perplexity']:.2f}) | "
              f"Valid Loss: {val_metrics['loss']:.6f} (VQ: {val_metrics['vq_loss']:.6f}, "
              f"Recon: {val_metrics['recon_loss']:.6f}, Perp: {val_metrics['perplexity']:.2f})")
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            no_improve_count = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'args': vars(args),
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
            }
            torch.save(checkpoint, save_dir / f'{model_name}.pth')
            print(f"  -> Best model saved (val_loss: {val_metrics['loss']:.6f})")
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
        'train_vq_loss': train_vq_losses,
        'valid_vq_loss': valid_vq_losses,
        'train_recon_loss': train_recon_losses,
        'valid_recon_loss': valid_recon_losses,
    })
    history_df.to_csv(save_dir / f'{model_name}_history.csv', index=False)
    
    # 保存配置
    with open(save_dir / f'{model_name}_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print('=' * 80)
    print(f'训练完成！')
    print(f'最佳验证损失: {best_val_loss:.6f}')
    print(f'模型保存至: {save_dir / model_name}.pth')


if __name__ == '__main__':
    set_device()
    main()
