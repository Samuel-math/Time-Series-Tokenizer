"""
VQVAE + Transformer 微调脚本
使用标准 PyTorch 训练循环，参考 TOTEM 设计
"""

import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

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
parser.add_argument('--init_mode', type=str, default='vqvae_transformer',
                   choices=['random', 'vqvae_only', 'vqvae_transformer'],
                   help='Initialization mode')
parser.add_argument('--pretrained_model', type=str, default=None,
                   help='Path to pretrained VQVAE+Transformer model checkpoint')
parser.add_argument('--vqvae_config_path', type=str, required=True, help='Path to VQVAE config file')
parser.add_argument('--vqvae_checkpoint', type=str, default=None, help='Path to VQVAE checkpoint')
parser.add_argument('--transformer_config_path', type=str, default='', help='Path to transformer config file')
# Optimization args
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--n_epochs_head_only', type=int, default=0, help='number of epochs to train only prediction head')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for full finetuning')
parser.add_argument('--lr_head_only', type=float, default=1e-3, help='learning rate for head-only training phase')
# Model architecture args
parser.add_argument('--head_type', type=str, default='mlp', choices=['mlp', 'linear'], help='prediction head type')
parser.add_argument('--head_dropout', type=float, default=0.1, help='dropout rate for prediction head')
parser.add_argument('--individual', type=int, default=0, help='use individual prediction head for each channel')
# TOTEM-style training args
parser.add_argument('--scheme', type=int, default=2, choices=[1, 2],
                   help='prediction scheme: 1 predicts mu/std separately, 2 uses RevIN denorm')
parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'smoothl1'], help='loss function type')
parser.add_argument('--beta', type=float, default=0.1, help='beta for smoothl1 loss')
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

set_device()


class MuStdModel(nn.Module):
    """MuStdModel: 预测未来序列的均值和标准差（参考 TOTEM）"""
    def __init__(self, Tin, Tout, hidden_dims=[512, 512], dropout=0.2, is_mlp=True):
        super().__init__()
        if is_mlp:
            layers = []
            input_dim = Tin
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, 2))
            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Linear(Tin, 2)
    
    def forward(self, x):
        """x: [B*C, Tin] -> [B*C, 2]"""
        return self.model(x)


def load_json_config(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)


def load_transformer_config(args, pretrained_checkpoint=None, device='cpu'):
    """统一加载 transformer_config 的辅助函数"""
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
        raise ValueError("找不到 Transformer 配置！")
    
    print("Transformer 配置:", transformer_config)
    return transformer_config


def get_model(c_in, args, vqvae_config, device='cpu'):
    """根据初始化模式构建 Finetune 模型"""
    print(f"\n初始化模式: {args.init_mode}")
    
    # 获取 transformer_config
    transformer_config = None
    if args.transformer_config_path and os.path.exists(args.transformer_config_path):
        transformer_config = load_json_config(args.transformer_config_path)
    else:
        transformer_config = {
            'd_model': 128, 'n_layers': 3, 'n_heads': 8, 'd_ff': 256,
            'dropout': 0.1, 'attn_dropout': 0.1
        }
        print("使用默认 transformer_config")
    
    # 构建模型
    if args.init_mode == 'random':
        finetune_model = VQVAETransformerFinetune(
            vqvae_config, transformer_config, pretrained_model=None,
            freeze_encoder=False, freeze_vq=False, freeze_transformer=False,
            head_type=args.head_type, head_dropout=args.head_dropout,
            individual=bool(args.individual), load_vqvae_weights=False,
            vqvae_checkpoint_path=None, device=device
        )
    elif args.init_mode == 'vqvae_only':
        if args.vqvae_checkpoint is None:
            raise ValueError("vqvae_only 模式需要提供 --vqvae_checkpoint")
        finetune_model = VQVAETransformerFinetune(
            vqvae_config, transformer_config, pretrained_model=None,
            freeze_encoder=False, freeze_vq=False, freeze_transformer=False,
            head_type=args.head_type, head_dropout=args.head_dropout,
            individual=bool(args.individual), load_vqvae_weights=True,
            vqvae_checkpoint_path=args.vqvae_checkpoint, device=device
        )
    elif args.init_mode == 'vqvae_transformer':
        if args.pretrained_model is None:
            raise ValueError("vqvae_transformer 模式需要提供 --pretrained_model")
        
        pretrained_checkpoint = torch.load(args.pretrained_model, map_location=device)
        pretrained_model = VQVAETransformerPretrain(
            vqvae_config, transformer_config, load_vqvae_weights=False,
            vqvae_checkpoint_path=None, device=device
        )
        
        try:
            if isinstance(pretrained_checkpoint, dict) and 'model_state_dict' in pretrained_checkpoint:
                pretrained_model.load_state_dict(pretrained_checkpoint['model_state_dict'], strict=False)
            elif isinstance(pretrained_checkpoint, dict):
                pretrained_model.load_state_dict(pretrained_checkpoint, strict=False)
            else:
                if hasattr(pretrained_checkpoint, 'state_dict'):
                    pretrained_model.load_state_dict(pretrained_checkpoint.state_dict(), strict=False)
        except Exception as e:
            print(f"加载预训练模型权重时出错: {e}")
            if args.vqvae_checkpoint is not None:
                pretrained_model._load_vqvae_weights(args.vqvae_checkpoint, device)
            else:
                raise ValueError("无法加载预训练权重，且未提供 --vqvae_checkpoint")
        
        if args.vqvae_checkpoint is not None:
            pretrained_model._load_vqvae_weights(args.vqvae_checkpoint, device)
        
        pretrained_model.eval()
        finetune_model = VQVAETransformerFinetune(
            vqvae_config, transformer_config, pretrained_model=pretrained_model,
            freeze_encoder=False, freeze_vq=False, freeze_transformer=False,
            head_type=args.head_type, head_dropout=args.head_dropout,
            individual=bool(args.individual), load_vqvae_weights=False,
            vqvae_checkpoint_path=None, device=device
        )

    print('可训练参数数量:', sum(p.numel() for p in finetune_model.parameters() if p.requires_grad))
    return finetune_model


def setup_models_and_revin(dls, args, device):
    """设置模型和 RevIN"""
    vqvae_config = load_vqvae_config(args.vqvae_config_path)
    model = get_model(dls.vars, args, vqvae_config)
    model = model.to(device)
    
    # 只使用一个 RevIN（用于输入归一化和输出反归一化）
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # Scheme 1 才需要 MuStdModel
    model_mustd = None
    if args.scheme == 1:
        model_mustd = MuStdModel(
            Tin=args.context_points, Tout=args.target_points,
            hidden_dims=[512, 512], dropout=0.2, is_mlp=True
        ).to(device)
        # Scheme 1 需要两个 RevIN（输入和输出分别归一化）
        revin_in = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
        revin_out = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
        model_mustd.revin_in = revin_in
        model_mustd.revin_out = revin_out
    
    return model, revin, model_mustd, vqvae_config


def compute_loss(pred_norm, batch_x, batch_y, norm_y, revin, model_mustd, loss_func, args, n_vars):
    """计算损失（统一处理 scheme 1 和 2）"""
    if args.scheme == 1:
        # Scheme 1: 预测 mu 和 std（需要 MuStdModel）
        times = batch_x.permute(0, 2, 1).reshape(-1, batch_x.shape[1])  # [B*C, Tin]
        ymeanstd = model_mustd(times).reshape(batch_x.shape[0], n_vars, 2).permute(0, 2, 1)  # [B, 2, C]
        ymean, ystd = ymeanstd[:, 0:1, :], ymeanstd[:, 1:2, :]  # [B, 1, C]
        
        loss_mu = loss_func(model_mustd.revin_out.mean - model_mustd.revin_in.mean, ymean)
        loss_std = loss_func(model_mustd.revin_out.stdev - model_mustd.revin_in.stdev, ystd)
        loss_decode = loss_func(pred_norm, norm_y)
        loss_all = loss_func(
            pred_norm * (ystd.detach() + model_mustd.revin_in.stdev) + 
            (ymean.detach() + model_mustd.revin_in.mean),
            batch_y
        )
        return loss_decode + loss_mu + loss_std + loss_all
    else:
        # Scheme 2: 直接使用 RevIN 反归一化（不需要 MuStdModel）
        pred = revin(pred_norm, 'denorm') if revin else pred_norm
        return loss_func(pred, batch_y)


def forward_pass(batch_x, batch_y, model, revin, model_mustd, args):
    """前向传播：归一化和预测"""
    if args.scheme == 1:
        # Scheme 1: 使用两个 RevIN
        if model_mustd.revin_in:
            _ = model_mustd.revin_in(batch_x, 'norm')
        norm_y = model_mustd.revin_out(batch_y, 'norm') if model_mustd.revin_out else batch_y
    else:
        # Scheme 2: 只使用一个 RevIN
        if revin:
            _ = revin(batch_x, 'norm')
        norm_y = batch_y  # Scheme 2 不需要对 target 归一化
    
    pred_norm = model(batch_x, args.target_points)
    return pred_norm, norm_y


def train_epoch(model, revin, model_mustd, dataloader, optimizer, scheduler, loss_func, args, device, n_vars):
    """训练一个 epoch"""
    model.train()
    if model_mustd:
        model_mustd.train()
    losses = []
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred_norm, norm_y = forward_pass(batch_x, batch_y, model, revin, model_mustd, args)
        
        loss = compute_loss(pred_norm, batch_x, batch_y, norm_y, revin, model_mustd, loss_func, args, n_vars)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=100)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        losses.append(loss.item())
    
    return np.mean(losses) if losses else float('inf')


def validate_epoch(model, revin, model_mustd, dataloader, loss_func, args, device, n_vars):
    """验证一个 epoch"""
    model.eval()
    if model_mustd:
        model_mustd.eval()
    losses = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_norm, norm_y = forward_pass(batch_x, batch_y, model, revin, model_mustd, args)
            loss = compute_loss(pred_norm, batch_x, batch_y, norm_y, revin, model_mustd, loss_func, args, n_vars)
            losses.append(loss.item())
    
    return np.mean(losses) if losses else float('inf')


def freeze_transformer(model, freeze=True):
    """冻结或解冻 Transformer 参数"""
    for param in model.transformer_layers.parameters():
        param.requires_grad = not freeze
    for param in model.projection.parameters():
        param.requires_grad = not freeze
    for param in model.codebook_head.parameters():
        param.requires_grad = not freeze


def finetune_func(lr=args.lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dls = get_dls(args)
    model, revin, model_mustd, vqvae_config = setup_models_and_revin(dls, args, device)
    
    # 损失函数
    loss_func = nn.MSELoss() if args.loss_type == 'mse' else nn.SmoothL1Loss(beta=args.beta)
    
    # 第一阶段：只训练预测头
    train_losses, valid_losses = [], []
    best_val_loss = float('inf')
    
    if args.n_epochs_head_only > 0:
        print(f"\n{'='*60}\n第一阶段：只训练预测头（冻结 Transformer）\n{'='*60}")
        freeze_transformer(model, freeze=True)
        head_params = [p for p in model.parameters() if p.requires_grad]
        if model_mustd:
            head_params += list(model_mustd.parameters())
        print(f"可训练参数数量: {sum(p.numel() for p in head_params)}")
        
        optimizer = Adam(head_params, lr=args.lr_head_only, weight_decay=1e-5)
        scheduler = OneCycleLR(optimizer, max_lr=args.lr_head_only, 
                               total_steps=len(dls.train) * args.n_epochs_head_only, pct_start=0.3)
        
        for epoch in range(args.n_epochs_head_only):
            train_loss = train_epoch(model, revin, model_mustd, dls.train, optimizer, scheduler, 
                                     loss_func, args, device, dls.vars)
            valid_loss = validate_epoch(model, revin, model_mustd, dls.valid, loss_func, args, device, dls.vars)
            
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                print(f"阶段1 - Epoch {epoch+1}/{args.n_epochs_head_only} | "
                      f"Train: {train_loss:.6f} | Valid: {valid_loss:.6f} | *Improved*")
            else:
                print(f"阶段1 - Epoch {epoch+1}/{args.n_epochs_head_only} | "
                      f"Train: {train_loss:.6f} | Valid: {valid_loss:.6f}")
        
        print("第一阶段训练完成！")
        best_val_loss = float('inf')
    
    # 第二阶段：全量微调
    print(f"\n{'='*60}\n第二阶段：全量微调（解冻 Transformer）\n{'='*60}")
    freeze_transformer(model, freeze=False)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if model_mustd:
        trainable_params += list(model_mustd.parameters())
    print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = Adam(trainable_params, lr=lr, weight_decay=1e-5)
    full_finetune_steps = args.n_epochs_finetune - args.n_epochs_head_only
    scheduler = OneCycleLR(optimizer, max_lr=lr, 
                          total_steps=len(dls.train) * full_finetune_steps, pct_start=0.3)
    
    for epoch in range(full_finetune_steps):
        train_loss = train_epoch(model, revin, model_mustd, dls.train, optimizer, scheduler, 
                                loss_func, args, device, dls.vars)
        valid_loss = validate_epoch(model, revin, model_mustd, dls.valid, loss_func, args, device, dls.vars)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        current_epoch = args.n_epochs_head_only + epoch + 1
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            checkpoint = {'model_state_dict': model.state_dict(), 'args': args}
            # 保存 prediction_head 的配置信息（如果已初始化）
            if model.prediction_head is not None:
                checkpoint['prediction_head_config'] = {
                    'n_vars': dls.vars,
                    'target_len': args.target_points,
                    'head_type': args.head_type,
                    'head_dropout': args.head_dropout,
                    'individual': bool(args.individual)
                }
            if model_mustd:
                checkpoint['mustd_state_dict'] = model_mustd.state_dict()
            torch.save(checkpoint, os.path.join(args.save_path, args.save_finetuned_model + '.pth'))
            print(f"阶段2 - Epoch {current_epoch}/{args.n_epochs_finetune} | "
                  f"Train: {train_loss:.6f} | Valid: {valid_loss:.6f} | *Best Model Saved*")
        else:
            print(f"阶段2 - Epoch {current_epoch}/{args.n_epochs_finetune} | "
                  f"Train: {train_loss:.6f} | Valid: {valid_loss:.6f}")
    
    # 保存损失历史
    df = pd.DataFrame({'train_loss': train_losses, 'valid_loss': valid_losses})
    df.to_csv(args.save_path + args.save_finetuned_model + '_losses.csv', float_format='%.6f', index=False)
    print(f"微调完成！损失历史已保存")


def find_lr():
    """简单的学习率查找"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dls = get_dls(args)
    model, revin, model_mustd, _ = setup_models_and_revin(dls, args, device)
    
    loss_func = nn.MSELoss() if args.loss_type == 'mse' else nn.SmoothL1Loss(beta=args.beta)
    params = list(model.parameters())
    if model_mustd:
        params += list(model_mustd.parameters())
    optimizer = Adam(params, lr=args.lr)
    
    best_lr, best_loss = args.lr, float('inf')
    
    for test_lr in [1e-4, 5e-4, 1e-3]:
        optimizer.param_groups[0]['lr'] = test_lr
        model.train()
        if model_mustd:
            model_mustd.train()
        
        for batch_x, batch_y in dls.train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_norm, norm_y = forward_pass(batch_x, batch_y, model, revin, model_mustd, args)
            loss = compute_loss(pred_norm, batch_x, batch_y, norm_y, revin, model_mustd, loss_func, args, dls.vars)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_lr = test_lr
            break
    
    print(f"suggested_lr = {best_lr}")
    return best_lr


def test_func(weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dls = get_dls(args)
    model, revin, model_mustd, _ = setup_models_and_revin(dls, args, device)
    
    # 加载权重
    checkpoint = torch.load(weight_path + '.pth', map_location=device)
    
    # 在加载权重之前，先初始化 prediction_head（如果 checkpoint 中包含它）
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # 检查是否有 prediction_head 的权重
        has_prediction_head = any(key.startswith('prediction_head') for key in state_dict.keys())
        
        if has_prediction_head and model.prediction_head is None:
            # 优先使用保存的配置信息初始化
            if 'prediction_head_config' in checkpoint:
                config = checkpoint['prediction_head_config']
                model._build_prediction_head(
                    config['n_vars'], 
                    config['target_len'], 
                    num_tokens=1
                )
                model.prediction_head = model.prediction_head.to(device)
                print(f"根据 checkpoint 配置初始化 prediction_head: n_vars={config['n_vars']}, target_len={config['target_len']}")
            else:
                # 如果没有配置信息，使用 dummy forward 初始化（向后兼容）
                print("警告: checkpoint 中没有 prediction_head_config，使用 dummy forward 初始化")
                dummy_x = torch.zeros(1, args.context_points, dls.vars).to(device)
                _ = model(dummy_x, args.target_points)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 使用 strict=False 以处理可能的键不匹配（向后兼容）
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
            print(f"警告: 加载权重时缺少以下键: {missing_keys}")
        if unexpected_keys:
            print(f"警告: 加载权重时发现意外的键: {unexpected_keys}")
        if model_mustd and 'mustd_state_dict' in checkpoint:
            model_mustd.load_state_dict(checkpoint['mustd_state_dict'])
    else:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        if missing_keys:
            print(f"警告: 加载权重时缺少以下键: {missing_keys}")
        if unexpected_keys:
            print(f"警告: 加载权重时发现意外的键: {unexpected_keys}")
    
    model.eval()
    if model_mustd:
        model_mustd.eval()
    
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in dls.test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_norm, _ = forward_pass(batch_x, batch_y, model, revin, model_mustd, args)
            
            # 根据 scheme 进行 denormalization
            if args.scheme == 1:
                # Scheme 1: 使用 MuStdModel 预测的 mean/std
                times = batch_x.permute(0, 2, 1).reshape(-1, batch_x.shape[1])
                ymeanstd = model_mustd(times).reshape(batch_x.shape[0], dls.vars, 2).permute(0, 2, 1)
                ymean, ystd = ymeanstd[:, 0:1, :] + model_mustd.revin_in.mean, ymeanstd[:, 1:2, :] + model_mustd.revin_in.stdev
                pred = pred_norm * ystd + ymean
            else:
                # Scheme 2: 直接使用 RevIN 反归一化（更简单高效）
                pred = revin(pred_norm, 'denorm') if revin else pred_norm
            
            all_preds.append(pred.cpu())
            all_targets.append(batch_y.cpu())
    
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    
    print(f'score: MSE={mse:.6f}, MAE={mae:.6f}')
    
    pd.DataFrame([[mse, mae]], columns=['mse', 'mae']).to_csv(
        args.save_path + args.save_finetuned_model + '_acc.csv', 
        float_format='%.6f', index=False
    )
    
    return [preds, targets, [mse, mae]]


if __name__ == '__main__':
    if args.is_finetune:
        args.dset = args.dset_finetune
        suggested_lr = find_lr()
        finetune_func(suggested_lr)
        print('finetune completed')
        out = test_func(args.save_path + args.save_finetuned_model)
        print('----------- Complete! -----------')
    else:
        args.dset = args.dset_finetune
        out = test_func(args.save_path + args.save_finetuned_model)
        print('----------- Complete! -----------')
