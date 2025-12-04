"""
VQVAE + Transformer 微调脚本
使用标准 PyTorch 训练循环，不使用 Learner
在微调完成后直接输出测试集结果
"""

import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from src.models.vqvae_transformer import VQVAETransformerPretrain, VQVAETransformerFinetune
from src.models.vqvae import vqvae, Decoder
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
parser.add_argument('--vqvae_checkpoint', type=str, required=True,
                   help='Path to VQVAE checkpoint (for decoder)')
parser.add_argument('--transformer_config_path', type=str, default='', help='Path to transformer config file')
# Optimization args
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
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
    加载预训练模型 + VQVAE decoder + 构建 Finetune 模型
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
    # 4) 加载 VQVAE decoder
    # --------------------------
    print(f"加载VQVAE Decoder: {args.vqvae_checkpoint}")
    vqvae_model = torch.load(args.vqvae_checkpoint, map_location=device)
    decoder = vqvae_model.decoder
    print("VQVAE Decoder已加载")

    # --------------------------
    # 5) Finetune 模型
    # --------------------------
    finetune_model = VQVAETransformerFinetune(
        pretrained_model,
        decoder,
        vqvae_config,
        freeze_transformer=False,
        freeze_decoder=True,
        add_finetune_head=False,
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
    
    # 冻结 VQVAE encoder 和 VQ 参数（在预训练模型中）
    print("冻结 VQVAE encoder 和 VQ 参数")
    if hasattr(model, 'pretrained_model'):
        if hasattr(model.pretrained_model, 'vqvae_encoder'):
            for param in model.pretrained_model.vqvae_encoder.parameters():
                param.requires_grad = False
        if hasattr(model.pretrained_model, 'vq'):
            for param in model.pretrained_model.vq.parameters():
                param.requires_grad = False
    
    # Loss
    loss_func = nn.MSELoss(reduction='mean')
    
    # Optimizer and scheduler
    # 只优化需要梯度的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr, weight_decay=1e-5)
    total_steps = len(dls.train) * args.n_epochs_finetune
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.3)
    
    # 清空 torch 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Training loop
    train_losses = []
    valid_losses = []
    best_val_loss = float('inf')
    
    print(f"开始微调，共 {args.n_epochs_finetune} 个 epoch")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    for epoch in range(args.n_epochs_finetune):
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
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: Epoch {epoch+1}, Batch {batch_idx}, Loss is NaN/Inf: {loss.item()}!")
                continue
            
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_train_losses.append(loss.item())
            
            # 每 1000 个 batch 打印一次梯度信息（用于调试）
            if batch_idx % 1000 == 0 and batch_idx > 0:
                total_grad_norm = 0
                for p in trainable_params:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** (1. / 2)
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Batch {batch_idx}, Loss: {loss.item():.6f}, Grad Norm: {total_grad_norm:.6f}, LR: {current_lr:.2e}")
        
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
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            torch.save(model.state_dict(), os.path.join(args.save_path, args.save_finetuned_model + '.pth'))
            print(f"Epoch {epoch+1}/{args.n_epochs_finetune} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f} | *Best Model Saved*")
        else:
            print(f"Epoch {epoch+1}/{args.n_epochs_finetune} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f}")
    
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
