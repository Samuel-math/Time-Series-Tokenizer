"""
微调: 时间序列预测
- 加载阶段2预训练的模型
- 预测未来序列
- Loss: MSE
"""

import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from pathlib import Path

# 添加根目录到 path，使用共享模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.two_stage_pretrain import TwoStagePretrainModel
from src.models.layers.revin import RevIN
from src.basics import set_device
from datautils import get_dls


def parse_args():
    parser = argparse.ArgumentParser(description='两阶段预训练模型微调')
    
    # 数据集参数
    parser.add_argument('--dset', type=str, default='ettm1', help='数据集名称')
    parser.add_argument('--context_points', type=int, default=512, help='输入序列长度')
    parser.add_argument('--target_points', type=int, default=96, help='预测长度')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--scaler', type=str, default='standard', help='数据缩放方式')
    parser.add_argument('--features', type=str, default='M', help='特征类型')
    
    # 模型参数
    parser.add_argument('--pretrained_model', type=str, required=True, 
                        help='阶段2预训练模型路径')
    parser.add_argument('--finetune_mode', type=str, default='full', 
                        choices=['linear_probe', 'full'],
                        help='微调模式: linear_probe (只训练pred_head) 或 full (全量微调)')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/two_stage_finetune/', help='模型保存路径')
    parser.add_argument('--model_id', type=int, default=1, help='模型ID')
    
    return parser.parse_args()


def setup_finetune_mode(model, mode):
    """
    设置微调模式
    
    mode='linear_probe': 只训练 pred_head
    mode='full': 解冻 transformer, 训练 pred_head
    
    始终冻结: patch_embedding, pos_encoding, mlm_head, mask_token, codebook, ntp_head
    """
    # 冻结 patch_embedding
    for param in model.patch_embedding.parameters():
        param.requires_grad = False
    
    # 冻结位置编码
    for param in model.pos_encoding.parameters():
        param.requires_grad = False
    
    # 冻结 MLM head 和 mask token
    for param in model.mlm_head.parameters():
        param.requires_grad = False
    model.mask_token.requires_grad = False
    
    # 冻结 codebook (使用聚类中心，不训练)
    for param in model.codebook.parameters():
        param.requires_grad = False
    
    # 冻结 ntp_head
    for param in model.ntp_head.parameters():
        param.requires_grad = False
    
    if mode == 'linear_probe':
        # 额外冻结 transformer，只训练 pred_head
        for param in model.transformer.parameters():
            param.requires_grad = False
        print('冻结了: patch_embedding, pos_encoding, transformer, mlm_head, mask_token, codebook, ntp_head')
        print('模式: Linear Probe - 只训练 pred_head')
    
    elif mode == 'full':
        # transformer 和 pred_head 可训练
        print('冻结了: patch_embedding, pos_encoding, mlm_head, mask_token, codebook, ntp_head')
        print('模式: Full Finetune - 训练 transformer, pred_head')


def train_epoch(model, dataloader, optimizer, revin, args, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        if revin:
            batch_x = revin(batch_x, 'norm')
        
        # 前向传播
        pred = model.forward_finetune(batch_x, args.target_points)
        
        # RevIN 反归一化
        if revin:
            pred = revin(pred, 'denorm')
        
        # MSE 损失
        loss = F.mse_loss(pred, batch_y)
        
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
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            pred = model.forward_finetune(batch_x, args.target_points)
            
            if revin:
                pred = revin(pred, 'denorm')
            
            loss = F.mse_loss(pred, batch_y)
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def test_model(model, dataloader, revin, args, device):
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
            
            pred = model.forward_finetune(batch_x, args.target_points)
            
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
    
    # 加载预训练模型
    print(f'\n加载预训练模型: {args.pretrained_model}')
    checkpoint = torch.load(args.pretrained_model, map_location=device)
    config = checkpoint['config']
    
    model = TwoStagePretrainModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'模型加载成功，验证损失: {checkpoint.get("val_loss", "N/A")}')
    
    # 设置微调模式
    setup_finetune_mode(model, args.finetune_mode)
    
    # 打印可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}, Trainable: {trainable_params:,}')
    
    # 创建保存目录
    save_dir = Path(args.save_path) / args.dset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型文件名
    model_name = f'two_stage_{args.finetune_mode}_cw{args.context_points}_tw{args.target_points}_model{args.model_id}'
    
    # 获取数据
    args.dset_finetune = args.dset
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Train: {len(dls.train)}, Valid: {len(dls.valid)}, Test: {len(dls.test)}')
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 优化器和调度器
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    
    # 训练
    best_val_loss = float('inf')
    train_losses, valid_losses = [], []
    
    print(f'\n开始微调，共 {args.n_epochs} 个 epoch')
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
            print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
                  f"Train: {train_loss:.6f} | Valid: {val_loss:.6f} | *Best*")
        else:
            print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
                  f"Train: {train_loss:.6f} | Valid: {val_loss:.6f}")
    
    # 测试
    print('\n' + '=' * 80)
    print('测试最佳模型...')
    
    best_checkpoint = torch.load(save_dir / f'{model_name}.pth', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    mse, mae, preds, targets = test_model(model, dls.test, revin, args, device)
    print(f'测试结果: MSE = {mse:.6f}, MAE = {mae:.6f}')
    
    # 保存结果
    results_df = pd.DataFrame({
        'metric': ['MSE', 'MAE'],
        'value': [mse, mae]
    })
    results_df.to_csv(save_dir / f'{model_name}_results.csv', index=False)
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, args.n_epochs + 1),
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
