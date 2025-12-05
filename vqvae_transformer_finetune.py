"""
VQVAE + Transformer 微调脚本
使用标准 PyTorch 训练循环，不使用 Learner
在微调完成后直接输出测试集结果
"""

import numpy as np
import pandas as pd
import os
import random
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from src.models.vqvae_transformer import VQVAETransformerPretrain, VQVAETransformerFinetune
from src.models.layers.revin import RevIN
from src.basics import set_device
from src.utils_vqvae import load_vqvae_config
from datautils import get_dls

import argparse

parser = argparse.ArgumentParser()
# Pretraining and Finetuning
parser.add_argument('--is_finetune', type=int, default=1, help='do finetuning or not')
# Dataset and dataloader
parser.add_argument('--dset_finetune', type=str, default='ettm1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Pretrained model paths
parser.add_argument('--pretrained_model', type=str, required=True,
                   help='Path to pretrained VQVAE+Transformer model checkpoint')
parser.add_argument('--vqvae_config_path', type=str, required=True,
                   help='Path to VQVAE config file')
parser.add_argument('--vqvae_checkpoint', type=str, default=None,
                   help='Path to VQVAE checkpoint (deprecated, not needed for new architecture)')
parser.add_argument('--transformer_config_path', type=str, default='', help='Path to transformer config file')
# Optimization args
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--n_epochs_head_only', type=int, default=10, help='number of epochs to train only prediction head (freeze transformer)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for full finetuning')
parser.add_argument('--lr_head_only', type=float, default=1e-3, help='learning rate for head-only training phase (usually higher)')
# Model architecture args
parser.add_argument('--aggregation', type=str, default='mean', 
                   choices=['mean', 'max', 'last', 'attention'],
                   help='aggregation method for embeddings: mean, max, last, or attention')
parser.add_argument('--head_type', type=str, default='mlp', choices=['mlp', 'linear'],
                   help='prediction head type: mlp or linear')
parser.add_argument('--head_dropout', type=float, default=0.1, help='dropout rate for prediction head')
parser.add_argument('--individual', type=int, default=0, help='use individual prediction head for each channel')
# model id to keep track of the number of models saved
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='vqvae_transformer', help='model type for saving')

args = parser.parse_args()
print('args:', args)
args.save_path = 'saved_models/' + args.dset_finetune + '/vqvae_transformer_finetune/' + args.model_type + '/'
if not os.path.exists(args.save_path): 
    os.makedirs(args.save_path)

suffix_name = (
    '_cw' + str(args.context_points) 
    + '_tw' + str(args.target_points) 
    + '_epochs-finetune' + str(args.n_epochs_finetune) 
    + '_model' + str(args.finetuned_model_id)
)
args.save_finetuned_model = args.dset_finetune + '_vqvae_transformer_finetuned' + suffix_name

# get available GPU device
set_device()



def load_json_config(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)


def load_transformer_config(args, pretrained_checkpoint=None, device='cpu'):
    """
    统一加载 transformer_config 的辅助函数
    优先从 checkpoint 加载，其次从配置文件加载
    
    Args:
        args: 命令行参数
        pretrained_checkpoint: 已加载的 checkpoint（可选，避免重复加载）
        device: 设备
    """
    # 如果未提供 checkpoint，则加载
    if pretrained_checkpoint is None:
        print(f"加载预训练Transformer模型: {args.pretrained_model}")
        pretrained_checkpoint = torch.load(args.pretrained_model, map_location=device)
    
    if isinstance(pretrained_checkpoint, dict) and "transformer_config" in pretrained_checkpoint:
        transformer_config = pretrained_checkpoint["transformer_config"]
        print("从 checkpoint 加载 transformer_config")
    elif hasattr(args, "transformer_config_path") and args.transformer_config_path and os.path.exists(args.transformer_config_path):
        transformer_config = load_json_config(args.transformer_config_path)
        print(f"从文件加载 transformer_config: {args.transformer_config_path}")
    else:
        raise ValueError(
            "找不到 Transformer 配置！请确保 checkpoint 包含 transformer_config，"
            "或通过 --transformer_config_path 指定配置文件。"
        )
    
    print("Transformer 配置:", transformer_config)
    return transformer_config


def get_model(c_in, args, vqvae_config, device='cpu'):
    """
    加载预训练模型 + 构建 Finetune 模型（新架构，不需要 decoder）
    c_in: number of input variables
    """
    print(f"加载预训练Transformer模型: {args.pretrained_model}")
    pretrained_checkpoint = torch.load(args.pretrained_model, map_location=device)

    # --------------------------
    # 1) 获取 transformer_config
    # --------------------------
    transformer_config = load_transformer_config(args, pretrained_checkpoint, device)

    # --------------------------
    # 2) 构建预训练模型骨架
    # --------------------------
    pretrained_model = VQVAETransformerPretrain(vqvae_config, transformer_config)

    # --------------------------
    # 3) 加载 state_dict
    # --------------------------
    if isinstance(pretrained_checkpoint, dict) and 'model_state_dict' in pretrained_checkpoint:
        pretrained_model.load_state_dict(pretrained_checkpoint['model_state_dict'])
    elif isinstance(pretrained_checkpoint, dict):
        pretrained_model.load_state_dict(pretrained_checkpoint)
    else:
        if hasattr(pretrained_checkpoint, 'state_dict'):
            pretrained_model.load_state_dict(pretrained_checkpoint.state_dict())
        else:
            print("警告: checkpoint格式异常，尝试直接使用")
            pretrained_model = pretrained_checkpoint

    pretrained_model.eval()
    print("预训练Transformer模型已成功加载")

    # --------------------------
    # 4) 构建 Finetune 模型（新架构，不需要 decoder）
    # --------------------------
    finetune_model = VQVAETransformerFinetune(
        pretrained_model,
        vqvae_config,
        freeze_encoder=True,      # 冻结 VQVAE encoder
        freeze_vq=True,           # 冻结 VQ 层
        freeze_transformer=False,  # 允许微调 Transformer
        head_type=args.head_type,          # 预测头类型
        head_dropout=args.head_dropout,    # 预测头 dropout
        individual=bool(args.individual),   # 是否使用独立预测头
        aggregation=args.aggregation       # 聚合方法
    )

    print('number of trainable params:',
          sum(p.numel() for p in finetune_model.parameters() if p.requires_grad))

    return finetune_model


def find_lr():
    """简单的学习率查找：使用一个小的训练循环"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1) dataloader
    dls = get_dls(args)

    # 2) 加载 VQVAE config
    vqvae_config = load_vqvae_config(args.vqvae_config_path)

    # 3) 构建 model
    model = get_model(dls.vars, args, vqvae_config)
    model = model.to(device)
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False) if args.revin else None
    if revin:
        revin = revin.to(device)
    
    # Loss
    loss_func = nn.MSELoss(reduction='mean')
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # 简单的学习率查找：尝试几个不同的学习率
    best_lr = args.lr
    best_loss = float('inf')
    
    for test_lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
        optimizer.param_groups[0]['lr'] = test_lr
        model.train()
        
        # 使用一个 batch 测试
        for batch_x, batch_y in dls.train:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # RevIN normalization
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            optimizer.zero_grad()
            pred = model(batch_x, args.target_points)
            
            # RevIN denormalization
            if revin:
                pred = revin(pred, 'denorm')
            
            loss = loss_func(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_lr = test_lr
            break
    
    print(f"suggested_lr = {best_lr}")
    return best_lr


def finetune_func(lr=args.lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('end-to-end finetuning')
    # get dataloader
    dls = get_dls(args)
    
    # 加载配置
    vqvae_config = load_vqvae_config(args.vqvae_config_path)
    
    # get model (transformer_config 会在 get_model 内部自动加载)
    model = get_model(dls.vars, args, vqvae_config)
    model = model.to(device)
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False) if args.revin else None
    if revin:
        revin = revin.to(device)
    
    # 冻结 VQVAE encoder、VQ 和 decoder 参数（在预训练模型中）
    print("冻结 VQVAE encoder、VQ 和 decoder 参数")
    if hasattr(model, 'pretrained_model'):
        if hasattr(model.pretrained_model, 'vqvae_encoder'):
            for param in model.pretrained_model.vqvae_encoder.parameters():
                param.requires_grad = False
        if hasattr(model.pretrained_model, 'vq'):
            for param in model.pretrained_model.vq.parameters():
                param.requires_grad = False
        # 冻结 decoder（finetune 阶段不使用）
        if hasattr(model.pretrained_model, 'decoder'):
            for param in model.pretrained_model.decoder.parameters():
                param.requires_grad = False
            print(f"  Decoder 参数数量: {sum(p.numel() for p in model.pretrained_model.decoder.parameters())}")
    
    # 第一阶段：冻结 Transformer，只训练预测头
    if args.n_epochs_head_only > 0:
        print(f"\n{'='*60}")
        print(f"第一阶段：只训练预测头（冻结 Transformer）")
        print(f"训练轮数: {args.n_epochs_head_only}, 学习率: {args.lr_head_only}")
        print(f"{'='*60}")
        
        # 冻结 Transformer 参数
        if hasattr(model, 'pretrained_model'):
            for param in model.pretrained_model.transformer_layers.parameters():
                param.requires_grad = False
            for param in model.pretrained_model.projection.parameters():
                param.requires_grad = False
            for param in model.pretrained_model.codebook_head.parameters():
                param.requires_grad = False
            if hasattr(model, 'attention_aggregation') and model.attention_aggregation is not None:
                for param in model.attention_aggregation.parameters():
                    param.requires_grad = False
                if hasattr(model, '_attention_query'):
                    model._attention_query.requires_grad = False
        
        # 只优化预测头参数
        head_params = [p for p in model.parameters() if p.requires_grad]
        print(f"第一阶段可训练参数数量: {sum(p.numel() for p in head_params)}")
        
        # 创建第一阶段的 optimizer 和 scheduler
        head_optimizer = Adam(head_params, lr=args.lr_head_only, weight_decay=1e-5)
        head_total_steps = len(dls.train) * args.n_epochs_head_only
        head_scheduler = OneCycleLR(head_optimizer, max_lr=args.lr_head_only, 
                                   total_steps=head_total_steps, pct_start=0.3)
    
    # Loss
    loss_func = nn.MSELoss(reduction='mean')
    
    # 清空 torch 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Training loop
    train_losses = []
    valid_losses = []
    best_val_loss = float('inf')
    current_epoch = 0
    
    # 第一阶段：只训练预测头
    if args.n_epochs_head_only > 0:
        for epoch in range(args.n_epochs_head_only):
            current_epoch = epoch + 1
            # Training
            model.train()
            epoch_train_losses = []
            
            for batch_idx, (batch_x, batch_y) in enumerate(dls.train):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # RevIN normalization
                if revin:
                    batch_x = revin(batch_x, 'norm')
                
                head_optimizer.zero_grad()
                pred = model(batch_x, args.target_points)
                
                # RevIN denormalization
                if revin:
                    pred = revin(pred, 'denorm')
                
                loss = loss_func(pred, batch_y)
                
                # 检查 loss 是否为 NaN 或 Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: Epoch {current_epoch}, Batch {batch_idx}, Loss is NaN/Inf: {loss.item()}!")
                    continue
                
                loss.backward()
                
                # 随机打印梯度信息（用于调试）
                if random.random() < 0.005:  # 10% 的概率打印
                    print(f"\n[阶段1 - Epoch {current_epoch}, Batch {batch_idx}] 梯度信息:")
                    has_grad = False
                    no_grad_count = 0
                    grad_stats = []
                    
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if param.grad is not None:
                                grad_norm = param.grad.data.norm(2).item()
                                grad_mean = param.grad.data.mean().item()
                                grad_max = param.grad.data.max().item()
                                grad_min = param.grad.data.min().item()
                                
                                if grad_norm > 1e-8:  # 只打印有意义的梯度
                                    print(f"  {name}:")
                                    print(f"    norm={grad_norm:.6e}, mean={grad_mean:.6e}, max={grad_max:.6e}, min={grad_min:.6e}")
                                    print(f"    shape={param.shape}, numel={param.numel()}")
                                    has_grad = True
                                    grad_stats.append({
                                        'name': name,
                                        'norm': grad_norm,
                                        'mean': grad_mean,
                                        'max': grad_max,
                                        'min': grad_min
                                    })
                                    if len(grad_stats) >= 5:  # 只打印前5个有梯度的参数
                                        break
                            else:
                                no_grad_count += 1
                                if no_grad_count <= 3:  # 只打印前3个没有梯度的参数
                                    print(f"  {name}: grad is None!")
                    
                    if not has_grad:
                        print("  警告: 没有检测到任何有意义的梯度！")
                    else:
                        print(f"  共找到 {len(grad_stats)} 个有梯度的参数，{no_grad_count} 个参数梯度为 None")
                    
                    print(f"  Loss: {loss.item():.6f}\n")
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(head_params, max_norm=100)
                
                head_optimizer.step()
                head_scheduler.step()
                
                epoch_train_losses.append(loss.item())
            
            avg_train_loss = np.mean(epoch_train_losses)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            epoch_valid_losses = []
            
            with torch.no_grad():
                for batch_x, batch_y in dls.valid:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    # RevIN normalization
                    if revin:
                        batch_x = revin(batch_x, 'norm')
                    
                    pred = model(batch_x, args.target_points)
                    
                    # RevIN denormalization
                    if revin:
                        pred = revin(pred, 'denorm')
                    
                    loss = loss_func(pred, batch_y)
                    epoch_valid_losses.append(loss.item())
            
            avg_valid_loss = np.mean(epoch_valid_losses)
            valid_losses.append(avg_valid_loss)
            
            # 第一阶段也检查是否有改善
            if avg_valid_loss < best_val_loss:
                best_val_loss = avg_valid_loss
                print(f"阶段1 - Epoch {current_epoch}/{args.n_epochs_head_only} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f} | *Improved*")
            else:
                print(f"阶段1 - Epoch {current_epoch}/{args.n_epochs_head_only} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f}")
            
            # 如果 valid loss 完全不变，可能是验证集问题，打印警告
            if current_epoch > 1 and abs(avg_valid_loss - valid_losses[-2]) < 1e-8:
                if current_epoch == 2:
                    print(f"  警告: Valid loss 似乎没有变化，可能是验证集问题或模型输出固定")
        
        print(f"\n第一阶段训练完成！")
        # 重置 best_val_loss，让第二阶段从零开始
        best_val_loss = float('inf')
    
    # 第二阶段：解冻 Transformer 和 VQVAE，全量微调
    print(f"\n{'='*60}")
    print(f"第二阶段：全量微调（解冻 Transformer 和 VQVAE）")
    print(f"训练轮数: {args.n_epochs_finetune - args.n_epochs_head_only}, 学习率: {lr}")
    print(f"{'='*60}")
    
    # 解冻 Transformer 参数
    if hasattr(model, 'pretrained_model'):
        for param in model.pretrained_model.transformer_layers.parameters():
            param.requires_grad = True
        for param in model.pretrained_model.projection.parameters():
            param.requires_grad = True
        for param in model.pretrained_model.codebook_head.parameters():
            param.requires_grad = True
        if hasattr(model, 'attention_aggregation') and model.attention_aggregation is not None:
            for param in model.attention_aggregation.parameters():
                param.requires_grad = True
            if hasattr(model, '_attention_query'):
                model._attention_query.requires_grad = True
        
        # 解冻 VQVAE encoder 和 VQ 参数（decoder 保持冻结，因为 finetune 阶段不使用）
        print("解冻 VQVAE encoder 和 VQ 参数（decoder 保持冻结）")
        if hasattr(model.pretrained_model, 'vqvae_encoder'):
            for param in model.pretrained_model.vqvae_encoder.parameters():
                param.requires_grad = True
        if hasattr(model.pretrained_model, 'vq'):
            for param in model.pretrained_model.vq.parameters():
                param.requires_grad = True
        # decoder 保持冻结（finetune 阶段不使用 decoder）
    
    # 创建第二阶段的 optimizer 和 scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"第二阶段可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = Adam(trainable_params, lr=lr, weight_decay=1e-5)
    full_finetune_steps = args.n_epochs_finetune - args.n_epochs_head_only
    total_steps = len(dls.train) * full_finetune_steps
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.3)
    
    # 第二阶段训练
    for epoch in range(full_finetune_steps):
        # Training
        model.train()
        epoch_train_losses = []
        
        for batch_idx, (batch_x, batch_y) in enumerate(dls.train):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # RevIN normalization
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            optimizer.zero_grad()
            pred = model(batch_x, args.target_points)
            
            # RevIN denormalization
            if revin:
                pred = revin(pred, 'denorm')
            
            loss = loss_func(pred, batch_y)
            
            # 检查 loss 是否为 NaN 或 Inf
            current_epoch = args.n_epochs_head_only + epoch + 1
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: Epoch {current_epoch}, Batch {batch_idx}, Loss is NaN/Inf: {loss.item()}!")
                continue
            
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=100)
            
            optimizer.step()
            scheduler.step()
            
            epoch_train_losses.append(loss.item())
        
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_valid_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in dls.valid:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # RevIN normalization
                if revin:
                    batch_x = revin(batch_x, 'norm')
                
                pred = model(batch_x, args.target_points)
                
                # RevIN denormalization
                if revin:
                    pred = revin(pred, 'denorm')
                
                loss = loss_func(pred, batch_y)
                epoch_valid_losses.append(loss.item())
        
        avg_valid_loss = np.mean(epoch_valid_losses)
        valid_losses.append(avg_valid_loss)
        
        # Save best model
        current_epoch = args.n_epochs_head_only + epoch + 1
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            torch.save(model.state_dict(), os.path.join(args.save_path, args.save_finetuned_model + '.pth'))
            print(f"阶段2 - Epoch {current_epoch}/{args.n_epochs_finetune} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f} | *Best Model Saved*")
        else:
            print(f"阶段2 - Epoch {current_epoch}/{args.n_epochs_finetune} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f}")
    
    # Save loss history
    df = pd.DataFrame(data={'train_loss': train_losses, 'valid_loss': valid_losses})
    df.to_csv(args.save_path + args.save_finetuned_model + '_losses.csv', float_format='%.6f', index=False)
    print(f"微调完成！损失历史已保存到 {args.save_path + args.save_finetuned_model + '_losses.csv'}")


def test_func(weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # get dataloader
    dls = get_dls(args)
    
    # 加载配置
    vqvae_config = load_vqvae_config(args.vqvae_config_path)
    
    # get model (transformer_config 会在 get_model 内部自动加载)
    model = get_model(dls.vars, args, vqvae_config)
    model = model.to(device)
    
    # Load weights
    model.load_state_dict(torch.load(weight_path + '.pth', map_location=device))
    model.eval()
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False) if args.revin else None
    if revin:
        revin = revin.to(device)
    
    # Test
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dls.test:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # RevIN normalization
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            pred = model(batch_x, args.target_points)
            
            # RevIN denormalization
            if revin:
                pred = revin(pred, 'denorm')
            
            all_preds.append(pred.cpu())
            all_targets.append(batch_y.cpu())
    
    # Concatenate all predictions and targets
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    # Calculate metrics
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    
    scores = [mse, mae]
    print(f'score: MSE={mse:.6f}, MAE={mae:.6f}')
    
    # save results
    pd.DataFrame(np.array(scores).reshape(1, -1), columns=['mse', 'mae']).to_csv(
        args.save_path + args.save_finetuned_model + '_acc.csv', 
        float_format='%.6f', 
        index=False
    )
    
    return [preds, targets, scores]


if __name__ == '__main__':
    if args.is_finetune:
        args.dset = args.dset_finetune
        # Finetune
        suggested_lr = find_lr()
        finetune_func(suggested_lr)
        print('finetune completed')
        # Test - 直接输出测试集结果
        out = test_func(args.save_path + args.save_finetuned_model)
        print('----------- Complete! -----------')
    else:
        args.dset = args.dset_finetune
        weight_path = args.save_path + args.save_finetuned_model
        # Test only
        out = test_func(weight_path)
        print('----------- Complete! -----------')
