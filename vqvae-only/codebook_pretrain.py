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
import argparse
from pathlib import Path

# 添加根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.codebook_model import CodebookModel
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
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN')
    parser.add_argument('--amp', type=int, default=1, help='是否启用混合精度')
    parser.add_argument('--vq_weight', type=float, default=1.0, help='VQ损失权重')
    parser.add_argument('--recon_weight', type=float, default=1.0, help='重构损失权重')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/vqvae_only/', help='模型保存路径')
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
        'num_hiddens': args.num_hiddens,
        'num_residual_layers': args.num_residual_layers,
        'num_residual_hiddens': args.num_residual_hiddens,
        'use_patch_attention': False,  # 码本预训练不使用patch attention
    }
    return config


def train_epoch(model, dataloader, optimizer, revin, args, device, scaler):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    total_perplexity = 0
    n_batches = 0
    
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)  # [B, T, C]
        
        if revin:
            batch_x = revin(batch_x, 'norm')
        
        # 编码和解码（只训练encoder、vq、decoder）
        indices, vq_loss, z_q = model.encode_to_indices(batch_x)
        x_recon = model.decode_from_codes(z_q)  # [B, num_patches * patch_size, C]
        
        # 计算重构损失
        B, T, C = batch_x.shape
        num_patches = indices.shape[1]
        recon_len = num_patches * model.patch_size
        recon_loss = F.mse_loss(x_recon, batch_x[:, :recon_len, :])
        
        # 总损失
        loss = args.recon_weight * recon_loss + args.vq_weight * vq_loss
        
        # 反向传播
        optimizer.zero_grad()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 计算perplexity（码本使用率）
        unique_indices = torch.unique(indices.reshape(-1))
        perplexity = len(unique_indices) / args.codebook_size
        
        total_loss += loss.item()
        total_vq_loss += vq_loss.item()
        total_recon_loss += recon_loss.item()
        total_perplexity += perplexity
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches if n_batches > 0 else 0.0,
        'vq_loss': total_vq_loss / n_batches if n_batches > 0 else 0.0,
        'recon_loss': total_recon_loss / n_batches if n_batches > 0 else 0.0,
        'perplexity': total_perplexity / n_batches if n_batches > 0 else 0.0,
    }


def validate_epoch(model, dataloader, revin, args, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    total_perplexity = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)  # [B, T, C]
            
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            # 编码和解码
            indices, vq_loss, z_q = model.encode_to_indices(batch_x)
            x_recon = model.decode_from_codes(z_q)  # [B, num_patches * patch_size, C]
            
            # 计算重构损失
            B, T, C = batch_x.shape
            num_patches = indices.shape[1]
            recon_len = num_patches * model.patch_size
            recon_loss = F.mse_loss(x_recon, batch_x[:, :recon_len, :])
            
            # 总损失
            loss = args.recon_weight * recon_loss + args.vq_weight * vq_loss
            
            # 计算perplexity
            unique_indices = torch.unique(indices.reshape(-1))
            perplexity = len(unique_indices) / args.codebook_size
            
            total_loss += loss.item()
            total_vq_loss += vq_loss.item()
            total_recon_loss += recon_loss.item()
            total_perplexity += perplexity
            n_batches += 1
    
    return {
        'loss': total_loss / n_batches if n_batches > 0 else 0.0,
        'vq_loss': total_vq_loss / n_batches if n_batches > 0 else 0.0,
        'recon_loss': total_recon_loss / n_batches if n_batches > 0 else 0.0,
        'perplexity': total_perplexity / n_batches if n_batches > 0 else 0.0,
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
    
    # 模型文件名
    code_dim = args.embedding_dim * (args.patch_size // args.compression_factor)
    model_name = f'codebook_ps{args.patch_size}_cb{args.codebook_size}_cd{code_dim}_model{args.model_id}'
    
    # 获取数据
    args.dset_pretrain = args.dset
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Train batches: {len(dls.train)}, Valid batches: {len(dls.valid)}')
    
    # 创建轻量级码本模型（只包含encoder、vq、decoder）
    config = get_model_config(args)
    model = CodebookModel(config, dls.vars).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\n码本模型参数统计:')
    print(f'  总参数: {total_params:,}')
    print(f'  所有参数均可训练')
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    
    # AMP
    scaler = amp.GradScaler(enabled=bool(args.amp))
    
    # 训练
    best_val_loss = float('inf')
    train_losses, valid_losses = [], []
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
        
        # 打印进度
        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} (Recon: {train_metrics['recon_loss']:.4f}, "
              f"VQ: {train_metrics['vq_loss']:.4f}, Perplexity: {train_metrics['perplexity']:.3f}) | "
              f"Valid Loss: {val_metrics['loss']:.4f} (Recon: {val_metrics['recon_loss']:.4f}, "
              f"VQ: {val_metrics['vq_loss']:.4f}, Perplexity: {val_metrics['perplexity']:.3f})")
        
        # 保存最佳模型
        if epoch >= 5:  # 前5个epoch不保存
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                no_improve_count = 0
                model_saved = True
                
                # 只保存encoder、decoder和vq的权重
                checkpoint = {
                    'encoder_state_dict': model.encoder.state_dict(),
                    'decoder_state_dict': model.decoder.state_dict(),
                    'vq_state_dict': model.vq.state_dict(),
                    'config': config,
                    'args': vars(args),
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                }
                torch.save(checkpoint, save_dir / f'{model_name}.pth')
                print(f"  -> Best model saved (val_loss: {val_metrics['loss']:.4f})")
            elif model_saved:
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
    print(f'码本预训练完成！')
    print(f'最佳验证损失: {best_val_loss:.4f}')
    print(f'模型保存至: {save_dir / model_name}.pth')


if __name__ == '__main__':
    set_device()
    main()
