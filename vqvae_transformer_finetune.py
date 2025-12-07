"""
VQVAE + Transformer 微调脚本 (优化版)
简化架构，提高效率，减少冗余代码
"""

import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import json
from pathlib import Path

from src.models.vqvae_transformer import VQVAETransformerPretrain, VQVAETransformerFinetune
from src.models.layers.revin import RevIN
from src.basics import set_device
from src.utils_vqvae import load_vqvae_config
from datautils import get_dls

import argparse

class ModelManager:
    """模型管理和权重加载管理器"""

    def __init__(self, args):
        self.args = args
        self.vqvae_config = load_vqvae_config(args.vqvae_config_path)
        self.transformer_config = self._load_transformer_config()

    def _load_transformer_config(self):
        """加载Transformer配置"""
        if self.args.transformer_config_path and os.path.exists(self.args.transformer_config_path):
            with open(self.args.transformer_config_path, 'r') as f:
                return json.load(f)
        return {
            'd_model': 128, 'n_layers': 3, 'n_heads': 8,
            'd_ff': 256, 'dropout': 0.1, 'attn_dropout': 0.1
        }

    def create_model(self, n_vars, device='cpu'):
        """创建微调模型"""
        print(f"初始化模式: {self.args.init_mode}")

        pretrained_model = None
        if self.args.init_mode == 'vqvae_transformer':
            pretrained_model = self._load_pretrained_model(device)
        elif self.args.init_mode == 'vqvae_only' and self.args.vqvae_checkpoint is None:
            raise ValueError("vqvae_only 模式需要提供 --vqvae_checkpoint")

        model = VQVAETransformerFinetune(
            self.vqvae_config, self.transformer_config,
            patch_size=self.args.patch_size, stride=self.args.stride,
            pretrained_model=pretrained_model,
            freeze_vq=False, freeze_transformer=False,
            head_type=self.args.head_type, head_dropout=self.args.head_dropout,
            individual=bool(self.args.individual),
            load_vqvae_weights=(self.args.init_mode == 'vqvae_only'),
            vqvae_checkpoint_path=self.args.vqvae_checkpoint,
            device=device
        )

        print(f'可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        return model

    def _load_pretrained_model(self, device):
        """加载预训练模型"""
        if not self.args.pretrained_model:
            raise ValueError("vqvae_transformer 模式需要提供 --pretrained_model")

        checkpoint = torch.load(self.args.pretrained_model, map_location=device)
        pretrained_model = VQVAETransformerPretrain(
            self.vqvae_config, self.transformer_config,
            load_vqvae_weights=False, vqvae_checkpoint_path=None, device=device
        )

        # 统一加载权重逻辑
        state_dict = self._extract_state_dict(checkpoint)
        pretrained_model.load_state_dict(state_dict, strict=False)
        print(f"成功从预训练模型加载权重: {self.args.pretrained_model}")

        pretrained_model.eval()
        return pretrained_model

    @staticmethod
    def _extract_state_dict(checkpoint):
        """从checkpoint中提取state_dict"""
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict):
            return checkpoint
        elif hasattr(checkpoint, 'state_dict'):
            return checkpoint.state_dict()
        else:
            raise ValueError("无法识别的checkpoint格式")

class CheckpointManager:
    """Checkpoint保存和加载管理器"""

    @staticmethod
    def save_checkpoint(model, path, args, dls, **kwargs):
        """保存checkpoint"""
        checkpoint = {'model_state_dict': model.state_dict(), 'args': args}
        if model.prediction_head is not None:
            checkpoint['prediction_head_config'] = {
                'n_vars': dls.vars,
                'target_len': args.target_points,
                'head_type': args.head_type,
                'head_dropout': args.head_dropout,
                'individual': bool(args.individual)
            }
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(model, path, device):
        """加载checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        state_dict = CheckpointManager._extract_state_dict(checkpoint)

        # 处理prediction_head初始化
        if any(key.startswith('prediction_head') for key in state_dict.keys()):
            if model.prediction_head is None and 'prediction_head_config' in checkpoint:
                config = checkpoint['prediction_head_config']
                model._build_prediction_head(config['n_vars'], config['target_len'])
                model.prediction_head = model.prediction_head.to(device)
                print(f"根据checkpoint配置初始化prediction_head")

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"警告: 缺少键 {missing_keys}")
        if unexpected_keys:
            print(f"警告: 意外键 {unexpected_keys}")

        return checkpoint

    @staticmethod
    def _extract_state_dict(checkpoint):
        """提取state_dict"""
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        return checkpoint

class Trainer:
    """训练管理器"""

    def __init__(self, model, revin, dls, args, device):
        self.model = model
        self.revin = revin
        self.dls = dls
        self.args = args
        self.device = device
        self.loss_func = nn.MSELoss()

    def freeze_transformer(self, freeze=True):
        """冻结/解冻Transformer参数"""
        for param in self.model.cross_attention_layers.parameters():
            param.requires_grad = not freeze
        for param in self.model.patch_projection.parameters():
            param.requires_grad = not freeze
        if hasattr(self.model, 'codebook_projection'):
            for param in self.model.codebook_projection.parameters():
                param.requires_grad = not freeze

    def train_epoch(self, optimizer, scheduler=None):
        """训练一个epoch"""
        self.model.train()
        losses = []

        for batch_x, batch_y in self.dls.train:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            pred_norm = self._forward_pass(batch_x)
            loss = self._compute_loss(pred_norm, batch_y)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100)
            optimizer.step()
            if scheduler:
                scheduler.step()

            losses.append(loss.item())

        return np.mean(losses) if losses else float('inf')

    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch_x, batch_y in self.dls.test:  # 使用test作为validation
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_norm = self._forward_pass(batch_x)
                loss = self._compute_loss(pred_norm, batch_y)
                losses.append(loss.item())

        return np.mean(losses) if losses else float('inf')

    def _forward_pass(self, batch_x):
        """前向传播"""
        if self.revin:
            _ = self.revin(batch_x, 'norm')
        return self.model(batch_x, self.args.target_points)

    def _compute_loss(self, pred_norm, batch_y):
        """计算损失"""
        pred = self.revin(pred_norm, 'denorm') if self.revin else pred_norm
        return self.loss_func(pred, batch_y)

def parse_args():
    """参数解析"""
    parser = argparse.ArgumentParser()

    # 数据集和数据加载器
    parser.add_argument('--dset_finetune', type=str, default='ettm1', help='数据集名称')
    parser.add_argument('--context_points', type=int, default=512, help='序列长度')
    parser.add_argument('--target_points', type=int, default=96, help='预测长度')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader工作线程数')
    parser.add_argument('--scaler', type=str, default='standard', help='数据缩放方式')
    parser.add_argument('--features', type=str, default='M', help='单变量或多变量模型')

    # 模型配置
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN')
    parser.add_argument('--init_mode', type=str, default='vqvae_transformer',
                       choices=['random', 'vqvae_only', 'vqvae_transformer'],
                       help='初始化模式')
    parser.add_argument('--vqvae_config_path', type=str, required=True, help='VQVAE配置文件路径')
    parser.add_argument('--vqvae_checkpoint', type=str, default=None, help='VQVAE checkpoint路径')
    parser.add_argument('--pretrained_model', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--transformer_config_path', type=str, default='', help='Transformer配置文件路径')

    # 训练参数
    parser.add_argument('--n_epochs_finetune', type=int, default=20, help='微调总轮数')
    parser.add_argument('--n_epochs_head_only', type=int, default=0, help='仅训练预测头轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='全量微调学习率')
    parser.add_argument('--lr_head_only', type=float, default=1e-3, help='仅训练预测头学习率')

    # 模型架构参数
    parser.add_argument('--patch_size', type=int, default=16, help='时间序列patch大小')
    parser.add_argument('--stride', type=int, default=None, help='patch步长')
    parser.add_argument('--head_type', type=str, default='mlp', choices=['mlp', 'linear'], help='预测头类型')
    parser.add_argument('--head_dropout', type=float, default=0.1, help='预测头dropout率')
    parser.add_argument('--individual', type=int, default=0, help='是否为每个通道使用独立预测头')

    # 模型保存参数
    parser.add_argument('--finetuned_model_id', type=int, default=1, help='保存模型ID')
    parser.add_argument('--model_type', type=str, default='vqvae_transformer', help='模型类型')

    # 运行模式
    parser.add_argument('--is_finetune', type=int, default=1, help='是否进行微调')

    return parser.parse_args()

args = parse_args()
print('args:', args)

# 设置保存路径
args.save_path = f'saved_models/{args.dset_finetune}/vqvae_transformer_finetune/{args.model_type}/'
Path(args.save_path).mkdir(parents=True, exist_ok=True)

# 构建保存文件名
suffix_name = f'_cw{args.context_points}_tw{args.target_points}_epochs-finetune{args.n_epochs_finetune}_model{args.finetuned_model_id}'
args.save_finetuned_model = f'{args.dset_finetune}_vqvae_transformer_finetuned{suffix_name}'

set_device()




def finetune_func(lr=args.lr):
    """优化后的微调函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dls = get_dls(args)

    # 初始化模型管理器和模型
    model_manager = ModelManager(args)
    model = model_manager.create_model(dls.vars, device)
    model = model.to(device)

    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None

    # 创建训练器
    trainer = Trainer(model, revin, dls, args, device)

    train_losses, valid_losses = [], []
    best_val_loss = float('inf')

    # 第一阶段：只训练预测头
    if args.n_epochs_head_only > 0:
        print(f"\n{'='*60}\n第一阶段：只训练预测头（冻结 Transformer）\n{'='*60}")
        trainer.freeze_transformer(freeze=True)
        head_params = [p for p in model.parameters() if p.requires_grad]
        print(f"可训练参数数量: {sum(p.numel() for p in head_params)}")

        optimizer = Adam(head_params, lr=args.lr_head_only, weight_decay=1e-4)
        scheduler = OneCycleLR(optimizer, max_lr=args.lr_head_only,
                               total_steps=len(dls.train) * args.n_epochs_head_only, pct_start=0.3)

        for epoch in range(args.n_epochs_head_only):
            train_loss = trainer.train_epoch(optimizer, scheduler)
            valid_loss = trainer.validate_epoch()

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
    trainer.freeze_transformer(freeze=False)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")

    optimizer = Adam(trainable_params, lr=lr, weight_decay=1e-4)
    full_finetune_steps = args.n_epochs_finetune - args.n_epochs_head_only
    scheduler = OneCycleLR(optimizer, max_lr=lr,
                          total_steps=len(dls.train) * full_finetune_steps, pct_start=0.3)

    for epoch in range(full_finetune_steps):
        train_loss = trainer.train_epoch(optimizer, scheduler)
        valid_loss = trainer.validate_epoch()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        current_epoch = args.n_epochs_head_only + epoch + 1
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            checkpoint_path = os.path.join(args.save_path, args.save_finetuned_model + '.pth')
            CheckpointManager.save_checkpoint(model, checkpoint_path, args, dls)
            print(f"阶段2 - Epoch {current_epoch}/{args.n_epochs_finetune} | "
                  f"Train: {train_loss:.6f} | Valid: {valid_loss:.6f} | *Best Model Saved*")
        else:
            print(f"阶段2 - Epoch {current_epoch}/{args.n_epochs_finetune} | "
                  f"Train: {train_loss:.6f} | Valid: {valid_loss:.6f}")

    # 保存损失历史
    df = pd.DataFrame({'train_loss': train_losses, 'valid_loss': valid_losses})
    df.to_csv(args.save_path + args.save_finetuned_model + '_losses.csv', float_format='%.6f', index=False)
    print("微调完成！损失历史已保存")


def test_func(weight_path):
    """优化后的测试函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dls = get_dls(args)

    # 初始化模型
    model_manager = ModelManager(args)
    model = model_manager.create_model(dls.vars, device)
    model = model.to(device)

    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None

    # 加载checkpoint
    checkpoint_path = weight_path + '.pth'
    CheckpointManager.load_checkpoint(model, checkpoint_path, device)

    # 创建训练器用于推理
    trainer = Trainer(model, revin, dls, args, device)

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_x, batch_y in dls.test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_norm = trainer._forward_pass(batch_x)
            pred = revin(pred_norm, 'denorm') if revin else pred_norm

            all_preds.append(pred.cpu())
            all_targets.append(batch_y.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))

    print(f'测试结果: MSE={mse:.6f}, MAE={mae:.6f}')

    # 保存结果
    pd.DataFrame([[mse, mae]], columns=['mse', 'mae']).to_csv(
        args.save_path + args.save_finetuned_model + '_acc.csv',
        float_format='%.6f', index=False
    )

    return [preds, targets, [mse, mae]]


if __name__ == '__main__':
    args.dset = args.dset_finetune

    if args.is_finetune:
        print("开始微调流程...")

        # 执行微调，使用指定的学习率
        finetune_func(args.lr)
        print('微调完成')

        # 测试最佳模型
        out = test_func(args.save_path + args.save_finetuned_model)
        print('----------- 微调流程完成! -----------')
    else:
        print("仅执行测试...")
        out = test_func(args.save_path + args.save_finetuned_model)
        print('----------- 测试完成! -----------')
