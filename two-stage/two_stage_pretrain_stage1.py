"""
阶段1预训练: Masked Reconstruction (类似 PatchTST)
- 随机 mask 一部分 patches
- 重建 masked patches
- Loss: MSE (只计算 masked 部分)
"""

import numpy as np
import pandas as pd
import os
import sys
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from pathlib import Path
import json

# 添加根目录到 path，使用共享模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.two_stage_pretrain import TwoStagePretrainModel, compute_stage1_loss
from src.models.layers.revin import RevIN
from src.basics import set_device
from datautils import get_dls


def parse_args():
    parser = argparse.ArgumentParser(description='阶段1预训练: Masked Reconstruction')
    
    # 数据集参数
    parser.add_argument('--dset', type=str, default='ettm1', help='数据集名称')
    parser.add_argument('--context_points', type=int, default=512, help='输入序列长度')
    parser.add_argument('--target_points', type=int, default=96, help='预测长度 (这里不使用)')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--scaler', type=str, default='standard', help='数据缩放方式')
    parser.add_argument('--features', type=str, default='M', help='特征类型')
    
    # 模型参数
    parser.add_argument('--patch_size', type=int, default=16, help='Patch大小')
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=4, help='Transformer层数')
    parser.add_argument('--d_ff', type=int, default=256, help='FFN维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--mask_ratio', type=float, default=0.4, help='Mask比例')
    parser.add_argument('--codebook_size', type=int, default=256, help='码本大小 (阶段2使用)')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/two_stage/', help='模型保存路径')
    parser.add_argument('--model_id', type=int, default=1, help='模型ID')
    
    return parser.parse_args()


def get_model_config(args):
    return {
        'patch_size': args.patch_size,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'mask_ratio': args.mask_ratio,
        'codebook_size': args.codebook_size,
    }


def train_epoch(model, dataloader, optimizer, revin, args, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)  # [B, T, C]
        
        if revin:
            batch_x = revin(batch_x, 'norm')
        
        # 前向传播
        recon, mask, target = model.forward_stage1(batch_x, args.mask_ratio)
        
        # 计算 loss
        loss = compute_stage1_loss(recon, mask, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate_epoch(model, dataloader, revin, args, device):
    """验证一个 epoch"""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            recon, mask, target = model.forward_stage1(batch_x, args.mask_ratio)
            loss = compute_stage1_loss(recon, mask, target)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def main():
    args = parse_args()
    print('Args:', args)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建保存目录
    save_dir = Path(args.save_path) / args.dset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型配置
    config = get_model_config(args)
    print(f'Model config: {config}')
    
    # 模型文件名
    model_name = f'stage1_ps{args.patch_size}_dm{args.d_model}_l{args.n_layers}_mask{args.mask_ratio}_model{args.model_id}'
    
    # 获取数据
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Train batches: {len(dls.train)}, Valid batches: {len(dls.valid)}')
    
    # 创建模型
    model = TwoStagePretrainModel(config).to(device)
    
    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}, Trainable: {trainable_params:,}')
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    
    # 训练
    best_val_loss = float('inf')
    train_losses, valid_losses = [], []
    
    print(f'\n开始阶段1预训练 (Masked Reconstruction)，共 {args.n_epochs} 个 epoch')
    print('=' * 80)
    
    for epoch in range(args.n_epochs):
        # 训练
        train_loss = train_epoch(model, dls.train, optimizer, revin, args, device)
        
        # 验证
        val_loss = validate_epoch(model, dls.valid, revin, args, device)
        
        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        
        scheduler.step()
        
        # 打印进度
        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"Train Loss: {train_loss:.6f} | Valid Loss: {val_loss:.6f}", end="")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'args': vars(args),
                'epoch': epoch,
                'val_loss': best_val_loss,
            }
            torch.save(checkpoint, save_dir / f'{model_name}.pth')
            print(f" | *Best*")
        else:
            print()
    
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
    print(f'阶段1预训练完成！')
    print(f'最佳验证损失: {best_val_loss:.6f}')
    print(f'模型保存至: {save_dir / model_name}.pth')
    print(f'\n下一步: 运行 two_stage_build_codebook.py 构建码本')


if __name__ == '__main__':
    set_device()
    main()
