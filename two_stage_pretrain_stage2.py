"""
阶段2预训练: Next Token Prediction (NTP)
- 加载带有码本的模型
- 冻结 embedding layer 和码本
- 训练因果 Transformer 预测下一个码本索引
- Loss: Cross Entropy
"""

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from pathlib import Path
import json

from src.models.two_stage_pretrain import TwoStagePretrainModel, compute_stage2_loss
from src.models.layers.revin import RevIN
from src.basics import set_device
from datautils import get_dls


def parse_args():
    parser = argparse.ArgumentParser(description='阶段2预训练: Next Token Prediction')
    
    # 数据集参数
    parser.add_argument('--dset', type=str, default='ettm1', help='数据集名称')
    parser.add_argument('--context_points', type=int, default=512, help='输入序列长度')
    parser.add_argument('--target_points', type=int, default=96, help='预测长度 (不使用)')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--scaler', type=str, default='standard', help='数据缩放方式')
    parser.add_argument('--features', type=str, default='M', help='特征类型')
    
    # 模型参数
    parser.add_argument('--model_with_codebook', type=str, required=True, 
                        help='带有码本的模型路径 (来自 build_codebook)')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/two_stage/', help='模型保存路径')
    parser.add_argument('--model_id', type=int, default=1, help='模型ID')
    
    return parser.parse_args()


def freeze_for_stage2(model):
    """
    Stage2 冻结策略:
    - 冻结: patch_embedding, pos_encoding, mlm_head, mask_token, codebook, pred_head
    - 解冻: transformer, ntp_head
    """
    # 冻结 patch_embedding
    for param in model.patch_embedding.parameters():
        param.requires_grad = False
    
    # 冻结位置编码
    for param in model.pos_encoding.parameters():
        param.requires_grad = False
    
    # 冻结 MLM head 和 mask token (阶段2不使用)
    for param in model.mlm_head.parameters():
        param.requires_grad = False
    model.mask_token.requires_grad = False
    
    # 冻结 codebook (使用聚类中心，不再训练)
    for param in model.codebook.parameters():
        param.requires_grad = False
    
    # 冻结 pred_head (微调时才训练)
    for param in model.pred_head.parameters():
        param.requires_grad = False
    
    print('冻结了: patch_embedding, pos_encoding, mlm_head, mask_token, codebook, pred_head')
    print('可训练: transformer, ntp_head')


def train_epoch(model, dataloader, optimizer, revin, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)
        
        if revin:
            batch_x = revin(batch_x, 'norm')
        
        # 前向传播
        logits, targets = model.forward_stage2(batch_x)
        
        # 计算 loss
        loss = compute_stage2_loss(logits, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate_epoch(model, dataloader, revin, device):
    """验证一个 epoch"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            logits, targets = model.forward_stage2(batch_x)
            loss = compute_stage2_loss(logits, targets)
            
            # 计算准确率
            B, C, seq_len, codebook_size = logits.shape
            pred = logits.argmax(dim=-1)  # [B, C, seq_len]
            correct = (pred == targets).sum().item()
            total_correct += correct
            total_tokens += B * C * seq_len
            
            total_loss += loss.item()
            n_batches += 1
    
    accuracy = total_correct / total_tokens
    return total_loss / n_batches, accuracy


def main():
    args = parse_args()
    print('Args:', args)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载带有码本的模型
    print(f'\n加载模型: {args.model_with_codebook}')
    checkpoint = torch.load(args.model_with_codebook, map_location=device)
    config = checkpoint['config']
    
    model = TwoStagePretrainModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'模型加载成功，码本大小: {config.get("codebook_size", "N/A")}')
    
    # Stage2 冻结策略
    freeze_for_stage2(model)
    
    # 打印可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}, Trainable: {trainable_params:,}')
    
    # 创建保存目录
    save_dir = Path(args.save_path) / args.dset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型文件名
    base_name = Path(args.model_with_codebook).stem
    model_name = f'{base_name}_ntp_model{args.model_id}'
    
    # 获取数据
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Train batches: {len(dls.train)}, Valid batches: {len(dls.valid)}')
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 检查码本利用率
    with torch.no_grad():
        sample_batch = next(iter(dls.train))[0].to(device)
        if revin:
            sample_batch = revin(sample_batch, 'norm')
        usage, _ = model.get_codebook_usage(sample_batch)
        print(f'初始码本利用率: {usage*100:.1f}%')
    
    # 优化器和调度器 (只优化可训练参数)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    
    # 训练
    best_val_loss = float('inf')
    train_losses, valid_losses, accuracies = [], [], []
    
    print(f'\n开始阶段2预训练 (NTP)，共 {args.n_epochs} 个 epoch')
    print('=' * 80)
    
    for epoch in range(args.n_epochs):
        # 训练
        train_loss = train_epoch(model, dls.train, optimizer, revin, device)
        
        # 验证
        val_loss, accuracy = validate_epoch(model, dls.valid, revin, device)
        
        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        accuracies.append(accuracy)
        
        scheduler.step()
        
        # 打印进度
        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} | "
              f"Accuracy: {accuracy*100:.2f}%", end="")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'args': vars(args),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'accuracy': accuracy,
            }
            torch.save(checkpoint, save_dir / f'{model_name}.pth')
            print(f" | *Best*")
        else:
            print()
        
        # 每20个epoch检查码本利用率
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                usage, _ = model.get_codebook_usage(sample_batch)
                print(f"  -> Codebook usage: {usage*100:.1f}%")
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, args.n_epochs + 1),
        'train_loss': train_losses,
        'valid_loss': valid_losses,
        'accuracy': accuracies,
    })
    history_df.to_csv(save_dir / f'{model_name}_history.csv', index=False)
    
    print('=' * 80)
    print(f'阶段2预训练完成！')
    print(f'最佳验证损失: {best_val_loss:.4f}')
    print(f'模型保存至: {save_dir / model_name}.pth')
    print(f'\n下一步: 运行 two_stage_finetune.py 进行微调')


if __name__ == '__main__':
    set_device()
    main()
