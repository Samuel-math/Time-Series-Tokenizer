"""
Patch-based VQVAE + Transformer 预训练脚本

特性:
- Overlapping Patches with stride
- EMA 更新码本 (更稳定)
- 自动检测 VQ/Recon loss 上升趋势并冻结相关参数
- Next Token Prediction (NTP) 损失
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
import argparse
from pathlib import Path

# 添加根目录到 path，使用共享模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.patch_vqvae_transformer import PatchVQVAETransformer, get_model_config
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
    
    # Patch 参数
    parser.add_argument('--patch_size', type=int, default=16, help='Patch大小')
    parser.add_argument('--stride', type=int, default=8, help='Patch滑动步长 (stride < patch_size 有重叠)')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--codebook_size', type=int, default=256, help='码本大小')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='VQ commitment cost')
    parser.add_argument('--use_ema', type=int, default=1, help='是否使用 EMA 更新码本')
    
    # Transformer 参数
    parser.add_argument('--n_layers', type=int, default=4, help='Transformer层数')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=512, help='FFN维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN')
    parser.add_argument('--vq_weight', type=float, default=1.0, help='VQ损失权重')
    parser.add_argument('--recon_weight', type=float, default=0.1, help='重构损失权重')
    
    # 自动冻结参数
    parser.add_argument('--auto_freeze', type=int, default=1, help='是否启用自动冻结')
    parser.add_argument('--freeze_patience', type=int, default=5, help='连续多少个 epoch loss 上升后冻结')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/patch_vqvae/', help='模型保存路径')
    parser.add_argument('--model_id', type=int, default=1, help='模型ID')
    parser.add_argument('--min_codebook_usage', type=float, default=0.75, help='保存模型的最低码本使用率')
    
    return parser.parse_args()


def detect_increasing_trend(loss_history, window=5):
    """
    检测 loss 是否呈稳定上升趋势
    
    Args:
        loss_history: loss 历史列表
        window: 检测窗口大小
    
    Returns:
        True 如果最近 window 个 epoch 都在上升
    """
    if len(loss_history) < window + 1:
        return False
    
    recent = loss_history[-(window + 1):]
    # 检查是否连续上升
    increasing_count = sum(1 for i in range(len(recent) - 1) if recent[i + 1] > recent[i])
    return increasing_count >= window


def freeze_encoder_vq_decoder(model):
    """冻结 Encoder, VQ, Decoder，只保留 Transformer 和 Output Head"""
    # 冻结 encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # 冻结 VQ
    for param in model.vq.parameters():
        param.requires_grad = False
    
    # 冻结 decoder
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    # 统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f">>> 冻结 Encoder/VQ/Decoder, 可训练参数: {trainable:,} / {total:,}")


def train_epoch(model, dataloader, optimizer, scheduler, revin, args, device, frozen=False):
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
        # logits: [B, num_patches-1, codebook_size]
        # targets: [B, num_patches-1]
        
        # NTP损失 (CrossEntropy)
        B, num_patches_m1, codebook_size = logits.shape
        logits_flat = logits.reshape(-1, codebook_size)  # [B*(num_patches-1), codebook_size]
        targets_flat = targets.reshape(-1)  # [B*(num_patches-1)]
        ntp_loss = F.cross_entropy(logits_flat, targets_flat)
        
        # 总损失
        if frozen:
            # 冻结后只优化 NTP
            loss = ntp_loss
        else:
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


def validate_epoch(model, dataloader, revin, args, device, frozen=False):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_ntp_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    total_correct = 0
    total_tokens = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            logits, targets, vq_loss, recon_loss = model.forward_pretrain(batch_x)
            
            B, num_patches_m1, codebook_size = logits.shape
            logits_flat = logits.reshape(-1, codebook_size)
            targets_flat = targets.reshape(-1)
            ntp_loss = F.cross_entropy(logits_flat, targets_flat)
            
            if frozen:
                loss = ntp_loss
            else:
                loss = ntp_loss + args.vq_weight * vq_loss + args.recon_weight * recon_loss
            
            # 计算准确率
            pred = logits_flat.argmax(dim=-1)
            correct = (pred == targets_flat).sum().item()
            total_correct += correct
            total_tokens += targets_flat.numel()
            
            total_loss += loss.item()
            total_ntp_loss += ntp_loss.item()
            total_vq_loss += vq_loss.item()
            total_recon_loss += recon_loss.item()
            n_batches += 1
    
    accuracy = total_correct / total_tokens
    return {
        'loss': total_loss / n_batches,
        'ntp_loss': total_ntp_loss / n_batches,
        'vq_loss': total_vq_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'accuracy': accuracy,
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
    
    # 计算 patch 信息
    num_patches = (args.context_points - args.patch_size) // args.stride + 1
    overlap = args.patch_size - args.stride
    
    print(f'\n配置信息:')
    print(f'  Patch: size={args.patch_size}, stride={args.stride}, overlap={overlap}, num_patches={num_patches}')
    print(f'  Model: d_model={args.d_model}, codebook_size={args.codebook_size}, use_ema={args.use_ema}')
    print(f'  自动冻结: auto_freeze={args.auto_freeze}, freeze_patience={args.freeze_patience}')
    
    # 模型文件名 (包含窗口大小)
    model_name = f'patch_vqvae_v2_cw{args.context_points}_ps{args.patch_size}_st{args.stride}_d{args.d_model}_cb{args.codebook_size}_model{args.model_id}'
    
    # 获取数据
    args.dset_pretrain = args.dset
    dls = get_dls(args)
    print(f'\nNumber of channels: {dls.vars}')
    print(f'Train batches: {len(dls.train)}, Valid batches: {len(dls.valid)}')
    
    # 创建模型
    args.n_channels = dls.vars  # 设置通道数
    config = get_model_config(args)
    print(f'\nModel config: {config}')
    
    model = PatchVQVAETransformer(config).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    
    # 训练状态
    best_val_loss = float('inf')
    train_losses, valid_losses = [], []
    vq_losses, recon_losses = [], []  # 用于检测上升趋势
    codebook_usages = []
    model_saved = False
    
    # 冻结状态
    encoder_frozen = False
    freeze_epoch = None
    
    print(f'\n开始预训练，共 {args.n_epochs} 个 epoch')
    print(f'注意: 只有当码本使用率 >= {args.min_codebook_usage*100:.0f}% 时才会保存模型')
    if args.auto_freeze:
        print(f'自动冻结: 当 VQ/Recon loss 连续 {args.freeze_patience} 个 epoch 上升时冻结 Encoder/VQ/Decoder')
    print('=' * 110)
    
    for epoch in range(args.n_epochs):
        # 训练
        train_metrics = train_epoch(model, dls.train, optimizer, scheduler, revin, args, device, frozen=encoder_frozen)
        
        # 验证
        val_metrics = validate_epoch(model, dls.valid, revin, args, device, frozen=encoder_frozen)
        
        # 检查码本使用率
        with torch.no_grad():
            sample_batch = next(iter(dls.train))[0].to(device)
            if revin:
                sample_batch = revin(sample_batch, 'norm')
            codebook_usage, _ = model.get_codebook_usage(sample_batch)
        
        # 记录历史
        train_losses.append(train_metrics['loss'])
        valid_losses.append(val_metrics['loss'])
        vq_losses.append(train_metrics['vq_loss'])
        recon_losses.append(train_metrics['recon_loss'])
        codebook_usages.append(codebook_usage)
        
        # 状态标记
        usage_status = "✓" if codebook_usage >= args.min_codebook_usage else "✗"
        phase = "[Frozen]" if encoder_frozen else "[Train]"
        
        # 打印进度
        print(f"{phase} Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"Loss: {train_metrics['loss']:.4f} (NTP: {train_metrics['ntp_loss']:.4f}, "
              f"VQ: {train_metrics['vq_loss']:.4f}, Recon: {train_metrics['recon_loss']:.4f}) | "
              f"Val: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']*100:.1f}% | "
              f"CB: {codebook_usage*100:.1f}% {usage_status}")
        
        # 自动冻结逻辑
        if args.auto_freeze and not encoder_frozen:
            vq_increasing = detect_increasing_trend(vq_losses, window=args.freeze_patience)
            recon_increasing = detect_increasing_trend(recon_losses, window=args.freeze_patience)
            
            if vq_increasing or recon_increasing:
                trend_info = []
                if vq_increasing:
                    trend_info.append("VQ")
                if recon_increasing:
                    trend_info.append("Recon")
                
                print(f"\n>>> 检测到 {' 和 '.join(trend_info)} loss 连续 {args.freeze_patience} 个 epoch 上升")
                freeze_encoder_vq_decoder(model)
                encoder_frozen = True
                freeze_epoch = epoch + 1
                
                # 重新创建优化器，只包含可训练参数
                trainable_params = filter(lambda p: p.requires_grad, model.parameters())
                optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
                remaining_epochs = args.n_epochs - epoch - 1
                if remaining_epochs > 0:
                    scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)
                print(f">>> 从 epoch {freeze_epoch} 开始只训练 Transformer\n")
        
        # 保存逻辑
        if codebook_usage >= args.min_codebook_usage:
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'args': vars(args),
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                    'n_channels': dls.vars,
                    'codebook_usage': codebook_usage,
                    'encoder_frozen': encoder_frozen,
                    'freeze_epoch': freeze_epoch,
                }
                torch.save(checkpoint, save_dir / f'{model_name}.pth')
                model_saved = True
                print(f"  -> Best model saved (val_loss: {best_val_loss:.4f}, CB usage: {codebook_usage*100:.1f}%)")
        else:
            if val_metrics['loss'] < best_val_loss:
                print(f"  -> Skip saving: codebook usage ({codebook_usage*100:.1f}%) < {args.min_codebook_usage*100:.0f}%")
    
    print('=' * 110)
    print(f'预训练完成！')
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, args.n_epochs + 1),
        'train_loss': train_losses,
        'valid_loss': valid_losses,
        'vq_loss': vq_losses,
        'recon_loss': recon_losses,
        'codebook_usage': codebook_usages,
    })
    history_df.to_csv(save_dir / f'{model_name}_history.csv', index=False)
    
    if model_saved:
        # 保存配置
        config['n_channels'] = dls.vars
        with open(save_dir / f'{model_name}_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f'最佳验证损失: {best_val_loss:.4f}')
        print(f'模型保存至: {save_dir / model_name}.pth')
        if freeze_epoch:
            print(f'Encoder/VQ/Decoder 冻结于 epoch {freeze_epoch}')
    else:
        print(f'警告: 码本使用率始终低于 {args.min_codebook_usage*100:.0f}%，未保存模型！')
        print(f'最大码本使用率: {max(codebook_usages)*100:.1f}%')
        print(f'建议: 增大 codebook_size 或调整训练参数')


if __name__ == '__main__':
    set_device()
    main()
