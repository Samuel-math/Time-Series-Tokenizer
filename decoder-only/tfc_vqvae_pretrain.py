"""
TF-C (Time-Frequency Consistency) VQVAE 预训练脚本

核心创新：
1. 双编码器：时域编码器 + 频域编码器
2. 共享VQ码本：时频语义对齐
3. 对比学习：InfoNCE损失增强码本语义表达

损失函数：
    L_total = α·L_recon + β·L_vq + γ·L_tfc

其中：
    - L_recon: MSE重构损失
    - L_vq: VQ commitment loss
    - L_tfc: InfoNCE时频对比损失
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
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import argparse
from pathlib import Path

# 添加根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.tfc_patch_vqvae import TFCPatchVQVAE, get_tfc_model_config
from src.models.layers.revin import RevIN
from src.basics import set_device
from datautils import get_dls


def parse_args():
    parser = argparse.ArgumentParser(description='TF-C VQVAE 预训练')
    
    # ============ 数据集参数 ============
    parser.add_argument('--dset', type=str, default='ettm1', help='数据集名称')
    parser.add_argument('--context_points', type=int, default=512, help='输入序列长度')
    parser.add_argument('--target_points', type=int, default=96, help='目标序列长度')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--scaler', type=str, default='standard', help='数据缩放方式')
    parser.add_argument('--features', type=str, default='M', help='特征类型')
    
    # ============ 基础VQVAE参数 ============
    parser.add_argument('--patch_size', type=int, default=16, help='Patch大小')
    parser.add_argument('--embedding_dim', type=int, default=32, help='VQVAE embedding维度')
    parser.add_argument('--compression_factor', type=int, default=4, help='压缩因子')
    parser.add_argument('--codebook_size', type=int, default=256, help='码本大小')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='VQ commitment cost')
    parser.add_argument('--codebook_ema', type=int, default=1, help='码本使用EMA更新(1启用)')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA衰减系数')
    parser.add_argument('--ema_eps', type=float, default=1e-5, help='EMA平滑项')
    parser.add_argument('--vq_init_method', type=str, default='random', help='VQ初始化方法')
    
    # ============ VQVAE Encoder/Decoder 参数 ============
    parser.add_argument('--num_hiddens', type=int, default=64, help='VQVAE隐藏层维度')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='残差层数')
    parser.add_argument('--num_residual_hiddens', type=int, default=32, help='残差隐藏层维度')
    
    # ============ Transformer参数 ============
    parser.add_argument('--n_layers', type=int, default=4, help='Transformer层数')
    parser.add_argument('--n_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=256, help='FFN维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--transformer_hidden_dim', type=int, default=None, help='Transformer内部维度')
    
    # ============ TF-C 特定参数 ============
    parser.add_argument('--freq_encoder_type', type=str, default='mlp', 
                       choices=['mlp', 'cnn'], help='频域编码器类型')
    parser.add_argument('--freq_encoder_hidden', type=int, default=256, help='频域编码器隐藏维度')
    parser.add_argument('--proj_hidden_dim', type=int, default=256, help='投影头隐藏维度')
    parser.add_argument('--proj_output_dim', type=int, default=128, help='对比空间维度')
    parser.add_argument('--temperature', type=float, default=0.07, help='InfoNCE温度系数')
    
    # ============ 损失权重 ============
    parser.add_argument('--alpha', type=float, default=1.0, help='重构损失权重')
    parser.add_argument('--beta', type=float, default=1.0, help='VQ损失权重')
    parser.add_argument('--gamma', type=float, default=0.5, help='TF-C对比损失权重')
    
    # ============ 训练参数 ============
    parser.add_argument('--n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'onecycle'], help='学习率调度器')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='预热轮数')
    parser.add_argument('--early_stop_patience', type=int, default=15, help='早停耐心值')
    
    # ============ 保存参数 ============
    parser.add_argument('--save_path', type=str, default='saved_models/tfc_vqvae/', help='模型保存路径')
    parser.add_argument('--model_id', type=int, default=1, help='模型ID')
    parser.add_argument('--run_id', type=int, default=None, help='运行ID')
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, scheduler, revin, args, device):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_tfc_loss = 0
    n_batches = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)  # [B, context_points, C]
        
        # RevIN归一化
        if revin:
            batch_x = revin(batch_x, 'norm')
        
        # TF-C 前向传播
        x_recon, recon_loss, vq_loss, tfc_loss, total_loss_batch, info_dict = model.forward_tfc_pretrain(batch_x)
        
        # 反向传播
        optimizer.zero_grad()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 累计损失
        total_loss += total_loss_batch.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_tfc_loss += tfc_loss.item()
        n_batches += 1
    
    # 更新学习率调度器
    if args.scheduler == 'cosine':
        scheduler.step()
    
    return {
        'total_loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'vq_loss': total_vq_loss / n_batches,
        'tfc_loss': total_tfc_loss / n_batches,
    }


def validate_epoch(model, dataloader, revin, args, device):
    """验证一个epoch"""
    model.eval()
    
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_tfc_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            # TF-C 前向传播
            x_recon, recon_loss, vq_loss, tfc_loss, total_loss_batch, info_dict = model.forward_tfc_pretrain(batch_x)
            
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_tfc_loss += tfc_loss.item()
            n_batches += 1
    
    return {
        'total_loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'vq_loss': total_vq_loss / n_batches,
        'tfc_loss': total_tfc_loss / n_batches,
    }


def main():
    args = parse_args()
    print('=' * 80)
    print('TF-C VQVAE 预训练')
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
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建保存目录
    save_dir = Path(args.save_path) / args.dset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型文件名
    code_dim = args.embedding_dim * (args.patch_size // args.compression_factor)
    if args.run_id is not None:
        model_name = f'tfc_vqvae_ps{args.patch_size}_cb{args.codebook_size}_cd{code_dim}_temp{args.temperature}_run{args.run_id}_model{args.model_id}'
    else:
        model_name = f'tfc_vqvae_ps{args.patch_size}_cb{args.codebook_size}_cd{code_dim}_temp{args.temperature}_model{args.model_id}'
    
    # 获取数据
    args.dset_pretrain = args.dset
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Train batches: {len(dls.train)}, Valid batches: {len(dls.valid)}')
    
    # 创建模型配置
    config = get_tfc_model_config(args)
    config['n_channels'] = dls.vars
    
    # 打印关键配置
    print('\n模型配置:')
    print(f'  Patch size: {args.patch_size}')
    print(f'  Codebook size: {args.codebook_size}')
    print(f'  Code dim: {code_dim}')
    print(f'  Freq encoder: {args.freq_encoder_type}')
    print(f'  Temperature: {args.temperature}')
    print(f'  Loss weights: α={args.alpha}, β={args.beta}, γ={args.gamma}')
    
    # 创建模型
    model = TFCPatchVQVAE(config).to(device)
    
    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'\n模型参数统计:')
    print(f'  总参数: {total_params:,}')
    print(f'  可训练参数: {trainable_params:,}')
    
    # 分模块统计
    time_encoder_params = sum(p.numel() for p in model.time_encoder.parameters())
    freq_encoder_params = sum(p.numel() for p in model.freq_encoder.parameters())
    vq_params = sum(p.numel() for p in model.vq.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    proj_params = sum(p.numel() for p in model.proj_time.parameters()) + \
                  sum(p.numel() for p in model.proj_freq.parameters())
    
    print(f'  时域编码器: {time_encoder_params:,}')
    print(f'  频域编码器: {freq_encoder_params:,}')
    print(f'  VQ码本: {vq_params:,}')
    print(f'  解码器: {decoder_params:,}')
    print(f'  投影头: {proj_params:,}')
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    else:
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=args.lr,
            epochs=args.n_epochs,
            steps_per_epoch=len(dls.train),
            pct_start=args.warmup_epochs / args.n_epochs
        )
    
    # 训练
    best_val_loss = float('inf')
    train_history = []
    valid_history = []
    no_improve_count = 0
    model_saved = False
    
    print(f'\n开始训练，共 {args.n_epochs} 个 epoch (早停: {args.early_stop_patience} epochs)')
    print('=' * 80)
    
    for epoch in range(args.n_epochs):
        # 训练
        train_metrics = train_epoch(model, dls.train, optimizer, scheduler, revin, args, device)
        
        # 验证
        val_metrics = validate_epoch(model, dls.valid, revin, args, device)
        
        train_history.append(train_metrics)
        valid_history.append(val_metrics)
        
        # 打印进度
        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"Train: total={train_metrics['total_loss']:.4f} "
              f"(recon={train_metrics['recon_loss']:.4f}, "
              f"vq={train_metrics['vq_loss']:.4f}, "
              f"tfc={train_metrics['tfc_loss']:.4f}) | "
              f"Valid: total={val_metrics['total_loss']:.4f}")
        
        # 保存最佳模型
        if epoch >= args.warmup_epochs:
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                no_improve_count = 0
                model_saved = True
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'args': vars(args),
                    'epoch': epoch,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                }
                torch.save(checkpoint, save_dir / f'{model_name}.pth')
                print(f"  -> Best model saved (val_loss: {val_metrics['total_loss']:.4f})")
            elif model_saved:
                no_improve_count += 1
                if no_improve_count >= args.early_stop_patience:
                    print(f"\n>>> 早停: val_loss 连续 {args.early_stop_patience} 个 epoch 未下降")
                    break
        
        # 每10个epoch检查码本使用率
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_batch = next(iter(dls.train))[0].to(device)
                if revin:
                    sample_batch = revin(sample_batch, 'norm')
                
                # 检查时域和频域的码本使用率
                usage_time, _ = model.get_codebook_usage(sample_batch, use_freq=False)
                usage_freq, _ = model.get_codebook_usage(sample_batch, use_freq=True)
                print(f"  -> Codebook usage: Time={usage_time*100:.1f}%, Freq={usage_freq*100:.1f}%")
    
    # 保存训练历史
    actual_epochs = len(train_history)
    history_df = pd.DataFrame({
        'epoch': range(1, actual_epochs + 1),
        'train_total_loss': [h['total_loss'] for h in train_history],
        'train_recon_loss': [h['recon_loss'] for h in train_history],
        'train_vq_loss': [h['vq_loss'] for h in train_history],
        'train_tfc_loss': [h['tfc_loss'] for h in train_history],
        'valid_total_loss': [h['total_loss'] for h in valid_history],
        'valid_recon_loss': [h['recon_loss'] for h in valid_history],
        'valid_vq_loss': [h['vq_loss'] for h in valid_history],
        'valid_tfc_loss': [h['tfc_loss'] for h in valid_history],
    })
    history_df.to_csv(save_dir / f'{model_name}_history.csv', index=False)
    
    # 保存配置
    with open(save_dir / f'{model_name}_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print('=' * 80)
    print(f'TF-C VQVAE 预训练完成！')
    print(f'最佳验证损失: {best_val_loss:.4f}')
    print(f'模型保存至: {save_dir / model_name}.pth')


if __name__ == '__main__':
    set_device()
    main()

