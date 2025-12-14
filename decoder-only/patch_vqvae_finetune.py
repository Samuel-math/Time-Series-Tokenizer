"""
Patch-based VQVAE + Transformer 微调脚本
使用 MSE 损失进行时间序列预测
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
from datetime import datetime
import time

# 添加根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.patch_vqvae_transformer import PatchVQVAETransformer
from src.models.layers.revin import RevIN
from src.basics import set_device
from datautils import get_dls


def parse_args():
    parser = argparse.ArgumentParser(description='Patch VQVAE Transformer 微调')
    
    # 数据集参数
    parser.add_argument('--dset', type=str, default='ettm1', help='数据集名称')
    parser.add_argument('--context_points', type=int, default=512, help='输入序列长度')
    parser.add_argument('--target_points', type=int, default=96, help='预测长度')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--scaler', type=str, default='standard', help='数据缩放方式')
    parser.add_argument('--features', type=str, default='M', help='特征类型')
    
    # 预训练模型参数
    parser.add_argument('--pretrained_model', type=str, required=True, help='预训练模型路径')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN')
    parser.add_argument('--amp', type=int, default=1, help='是否启用混合精度')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/patch_vqvae_finetune/', help='模型保存路径')
    parser.add_argument('--model_id', type=int, default=1, help='模型ID')
    
    return parser.parse_args()


def load_pretrained_model(checkpoint_path, device, n_channels=None):
    """加载预训练模型
    
    Args:
        checkpoint_path: checkpoint路径
        device: 设备
        n_channels: 通道数（如果提供且启用patch_attention，会在创建模型时立即初始化）
    """
    print(f'加载预训练模型: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    
    # 检查checkpoint中是否有patch_attention权重
    has_patch_attention = any('patch_attention' in k for k in state_dict.keys())
    
    # 如果checkpoint有patch_attention权重但config未开启，强制开启
    if has_patch_attention and not config.get('use_patch_attention', False):
        print('检测到checkpoint中有patch_attention权重，强制启用use_patch_attention')
        config['use_patch_attention'] = True
    
    # 如果启用patch_attention且提供了通道数，添加到config中以便立即初始化
    if config.get('use_patch_attention', False) and n_channels is not None:
        config['n_channels'] = n_channels
    
    # 创建模型（如果use_patch_attention=True且n_channels存在，会自动初始化patch_attention）
    model = PatchVQVAETransformer(config).to(device)
    
    # 直接加载所有权重（包括patch_attention），使用strict=False允许架构差异
    model.load_state_dict(state_dict, strict=False)
    
    print(f'预训练模型配置: {config}')
    print(f'预训练验证损失: {checkpoint.get("val_loss", "N/A")}')
    
    return model, config


def freeze_encoder_vq(model, freeze_patch_attention=True):
    """冻结encoder、decoder、VQ层、patch attention、Transformer和output_head（将patch映射成码本前的所有参数）"""
    # 使用模型的方法冻结VQVAE组件
    model.freeze_vqvae(components=['Encoder', 'Decoder', 'VQ'])
    
    # 冻结 patch attention（如果存在）
    if freeze_patch_attention and hasattr(model, 'patch_attention') and model.patch_attention is not None:
        for param in model.patch_attention.parameters():
            param.requires_grad = False
        print('✓ 已冻结 Patch Attention')
    
    # 冻结 Transformer 和 output_head（避免argmax导致的梯度断开问题）
    model.freeze_transformer_and_output_head()


def train_batch(model, batch_x, batch_y, optimizer, revin, args, device, scaler):
    """训练一个batch"""
    batch_x = batch_x.to(device)  # [B, context_points, C]
    batch_y = batch_y.to(device)  # [B, target_points, C]
    
    # RevIN归一化
    if revin:
        batch_x = revin(batch_x, 'norm')
    
    with amp.autocast(enabled=scaler.is_enabled()):
        # 前向传播: 预测码本索引 -> 解码
        pred, _ = model.forward_finetune(batch_x, args.target_points)
        
        # RevIN反归一化
        if revin:
            pred = revin(pred, 'denorm')
        
        # 用于反向传播和报告的loss（使用mean）
        loss = F.mse_loss(pred, batch_y, reduction='mean')
    
    # 反向传播（只对可训练参数）
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # 只对可训练参数进行梯度裁剪
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()


def validate_epoch(model, dataloader, revin, args, device, use_amp):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            with amp.autocast(enabled=use_amp):
                pred, _ = model.forward_finetune(batch_x, args.target_points)
            
            if revin:
                pred = revin(pred, 'denorm')
            
            # 使用mean
            mse_loss = F.mse_loss(pred, batch_y, reduction='mean')
            total_loss += mse_loss.item()
            n_batches += 1
    
    # 按batch数平均
    return total_loss / n_batches if n_batches > 0 else 0.0


def test_model(model, dataloader, revin, args, device, use_amp):
    """测试模型"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            with amp.autocast(enabled=use_amp):
                pred, _ = model.forward_finetune(batch_x, args.target_points)
            
            if revin:
                pred = revin(pred, 'denorm')
            
            all_preds.append(pred.cpu())
            all_targets.append(batch_y.cpu())
    
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    
    return mse, mae, preds, targets


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
    
    # 先获取数据以知道通道数（用于提前初始化patch_attention）
    args.dset_finetune = args.dset
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Train batches: {len(dls.train)}, Valid batches: {len(dls.valid)}, Test batches: {len(dls.test)}')
    
    # 加载预训练模型（传入通道数以便立即初始化patch_attention）
    model, config = load_pretrained_model(args.pretrained_model, device, n_channels=dls.vars)
    
    # 冻结 encoder、VQ 层和 patch attention（将patch映射成码本前的所有参数）
    patch_attention_loaded = hasattr(model, 'patch_attention') and model.patch_attention is not None
    freeze_encoder_vq(model, freeze_patch_attention=patch_attention_loaded)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'可训练参数: {trainable_params:,} / {total_params:,}')
    
    # AMP
    use_amp = bool(args.amp) and device.type == 'cuda'
    scaler = amp.GradScaler(enabled=use_amp)
    print(f'AMP enabled: {use_amp}')
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 模型文件名
    model_name = f'patch_vqvae_finetune_cw{args.context_points}_tw{args.target_points}_model{args.model_id}'
    
    # 优化器和调度器
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    
    # 训练
    best_val_loss = float('inf')
    train_losses, valid_losses = [], []
    no_improve_epochs = 0  # 连续无改善的epoch数
    early_stop_patience = 10  # 连续10个epoch无下降就停止
    best_epoch = -1  # 最佳模型所在的epoch
    
    start_time = time.time()
    
    print(f'\n开始微调，共 {args.n_epochs} 个 epoch')
    print(f'早停: 连续 {early_stop_patience} 个 epoch 无改善则停止')
    print('=' * 80)
    
    for epoch in range(args.n_epochs):
        model.train()
        epoch_train_losses = []
        
        # 训练一个epoch
        for batch_x, batch_y in dls.train:
            loss = train_batch(model, batch_x, batch_y, optimizer, revin, args, device, scaler)
            epoch_train_losses.append(loss)
        
        # Epoch结束，更新学习率
        scheduler.step()
        
        # 计算平均训练loss
        avg_train_loss = np.mean(epoch_train_losses)
        
        # 验证评估
        val_loss = validate_epoch(model, dls.valid, revin, args, device, use_amp)
        
        train_losses.append(avg_train_loss)
        valid_losses.append(val_loss)
        
        total_time = time.time() - start_time
        
        # 检查当前epoch是否改善了最佳验证损失
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0  # 重置计数器
            
            # 保存最佳模型
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'args': vars(args),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat(),
                'total_training_time_seconds': total_time,
            }
            torch.save(checkpoint, save_dir / f'{model_name}.pth')
            status = "*Best*"
        else:
            no_improve_epochs += 1
            status = ""
        
        # 打印进度
        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | Valid Loss: {val_loss:.6f} | "
              f"Time: {total_time/60:.1f}min {status}")
        
        if not is_best:
            print(f"  -> 无改善 (当前最佳: epoch {best_epoch+1}, val_loss: {best_val_loss:.6f}, "
                  f"连续 {no_improve_epochs} 个epoch无改善)")
        
        # 早停检查：连续10个epoch无改善
        if no_improve_epochs >= early_stop_patience:
            print(f"\n>>> 早停: 连续 {early_stop_patience} 个 epoch 无改善")
            # 保存当前模型（10个epoch无改善时的模型）
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'args': vars(args),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat(),
                'total_training_time_seconds': total_time,
                'early_stopped': True,
            }
            final_model_name = f'{model_name}_final_epoch{epoch+1}.pth'
            torch.save(checkpoint, save_dir / final_model_name)
            print(f"  -> 最终模型已保存: {final_model_name}")
            break
    
    # 测试
    print('\n' + '=' * 80)
    print('测试最佳模型...')
    
    # 加载最佳模型
    best_checkpoint = torch.load(save_dir / f'{model_name}.pth', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    mse, mae, preds, targets = test_model(model, dls.test, revin, args, device, use_amp)
    print(f'测试结果: MSE = {mse:.6f}, MAE = {mae:.6f}')
    
    # 保存结果
    results_df = pd.DataFrame({
        'metric': ['MSE', 'MAE'],
        'value': [mse, mae]
    })
    results_df.to_csv(save_dir / f'{model_name}_results.csv', index=False)
    
    # 保存训练历史 (基于epoch)
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'valid_loss': valid_losses,
    })
    history_df.to_csv(save_dir / f'{model_name}_history.csv', index=False)
    
    print('=' * 80)
    print(f'微调完成！')
    print(f'最佳验证损失: {best_val_loss:.6f}')
    print(f'测试 MSE: {mse:.6f}, MAE: {mae:.6f}')
    print(f'模型保存至: {save_dir / model_name}.pth')


if __name__ == '__main__':
    set_device()
    main()
