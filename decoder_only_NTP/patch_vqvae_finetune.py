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
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
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
    parser.add_argument('--run_id', type=int, default=None, help='运行ID（用于多次运行同一参数组合）')

    # 训练 loss 类型（验证/测试始终用 MSE 报告，保持与 benchmark 可比）
    parser.add_argument('--train_loss', type=str, default='mse',
                        choices=['mse', 'huber', 'smooth_l1'],
                        help='训练反向传播使用的 loss 类型：mse | huber | smooth_l1')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                        help='Huber / Smooth-L1 的 delta（beta）阈值。RevIN 空间建议 0.3~1.0')

    # Decoder 解冻（finetune 阶段做任务精调；不会改动预训练 checkpoint 文件）
    parser.add_argument('--unfreeze_decoder', type=int, default=0,
                        help='1 = 解冻 VQVAE decoder 参与 finetune（用更小 lr + 更大 weight decay 保护）'
                             '；0 = 保持冻结（默认，向后兼容）')
    parser.add_argument('--decoder_lr_ratio', type=float, default=0.1,
                        help='解冻 decoder 时使用的 lr 相对比例：decoder_lr = main_lr * ratio（默认 0.1）')
    parser.add_argument('--decoder_wd_ratio', type=float, default=5.0,
                        help='解冻 decoder 时使用的 weight_decay 放大倍数（默认 5.0，加强过拟合抑制）')
    
    # Gumbel-Softmax参数（微调阶段的码本查找）
    parser.add_argument('--use_gumbel_softmax', type=int, default=1, help='是否使用Gumbel-Softmax（1启用，0使用普通Softmax）')
    parser.add_argument('--gumbel_temperature', type=float, default=1.0, help='Gumbel-Softmax温度（越小越接近argmax）')
    parser.add_argument('--gumbel_hard', type=int, default=0, help='是否使用Straight-Through Gumbel（前向硬采样，反向软梯度）')
    
    # 自回归预测参数
    parser.add_argument('--ar_step_size', type=int, default=None, help='自回归步长（每步预测的patch数）。None表示非自回归（一次预测所有）')
    parser.add_argument('--pred_len', type=int, default=None,
                        help='每次 Transformer forward 预测的 patch 数 N（默认 None = 等于 ar_step_size）。'
                             'N > M 时产生 overlapping chunk，同一未来位置的多个 logit 在概率层面融合后再 argmax。')

    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/patch_vqvae_finetune/', help='模型保存路径')
    parser.add_argument('--model_id', type=int, default=1, help='模型ID')
    
    return parser.parse_args()


def load_pretrained_model(checkpoint_path, device, n_channels=None, args=None):
    """加载预训练模型
    
    Args:
        checkpoint_path: checkpoint路径
        device: 设备
        n_channels: 通道数（未使用，保留以兼容旧代码）
        args: 命令行参数（用于Gumbel-Softmax配置）
    
    Returns:
        model: 加载的模型
        config: 模型配置（原始config，用于保存checkpoint）
        model_config: 模型配置（包含Gumbel-Softmax，用于创建模型）
        pretrain_args: 预训练时的参数（用于获取step_size等）
    """
    print(f'加载预训练模型: {checkpoint_path}')
    # PyTorch 2.6+ 兼容性：设置 weights_only=False 以支持包含 numpy 对象的 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 保存原始config（用于保存checkpoint，确保架构一致性）
    import copy
    config = copy.deepcopy(checkpoint['config'])  # 深拷贝，避免修改原始config
    state_dict = checkpoint['model_state_dict']
    pretrain_args = checkpoint.get('args', {})  # 获取预训练时的参数
    
    # 打印预训练时的关键参数，用于调试
    print(f'预训练checkpoint中的关键参数:')
    print(f'  num_residual_hiddens: {config.get("num_residual_hiddens", "NOT FOUND")}')
    print(f'  num_hiddens: {config.get("num_hiddens", "NOT FOUND")}')
    print(f'  num_residual_layers: {config.get("num_residual_layers", "NOT FOUND")}')
    print(f'  n_layers: {config.get("n_layers", "NOT FOUND")}')
    print(f'  n_heads: {config.get("n_heads", "NOT FOUND")}')
    
    # 创建模型配置（添加Gumbel-Softmax配置，但不影响架构）
    import copy
    model_config = copy.deepcopy(config)  # 深拷贝原始config
    if args is not None:
        model_config['use_gumbel_softmax'] = bool(getattr(args, 'use_gumbel_softmax', 1))
        model_config['gumbel_temperature'] = getattr(args, 'gumbel_temperature', 1.0)
        model_config['gumbel_hard'] = bool(getattr(args, 'gumbel_hard', 0))
        print(f'Gumbel-Softmax配置: use={model_config["use_gumbel_softmax"]}, temp={model_config["gumbel_temperature"]}, hard={model_config["gumbel_hard"]}')
    
    # 创建模型（使用model_config，包含Gumbel-Softmax配置）
    model = PatchVQVAETransformer(model_config).to(device)
    
    # 直接加载所有权重，使用strict=False允许架构差异
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"警告: 以下权重未加载: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"警告: 以下权重未加载: {missing_keys}")
    if unexpected_keys:
        print(f"警告: 以下权重未使用: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"警告: 以下权重未使用: {unexpected_keys}")
    
    print(f'预训练模型配置: {config}')
    print(f'预训练验证损失: {checkpoint.get("val_loss", "N/A")}')
    
    # 打印预训练时的step_size（如果存在）
    if 'progressive_step_size' in pretrain_args:
        print(f'预训练step_size: {pretrain_args["progressive_step_size"]}')
    
    # 返回原始config（用于保存checkpoint），确保架构一致性
    return model, config, pretrain_args


def freeze_encoder_vq(model, unfreeze_decoder=False):
    """冻结 VQVAE 相关组件（Encoder + VQ 始终冻结；Decoder 可选是否解冻）

    注意：这里仅修改 requires_grad / EMA 标志，不会触碰模型权重本身。
    因此加载的 VQVAE 预训练 checkpoint 文件始终保持原样，未被任何读写操作修改。

    Args:
        model: PatchVQVAETransformer
        unfreeze_decoder: True 时解冻 decoder，让它在 finetune 中参与梯度更新
    """
    components = ['Encoder', 'VQ']
    if not unfreeze_decoder:
        components.append('Decoder')
    model.freeze_vqvae(components=components)


def _compute_train_loss(pred, target, args):
    """根据 args.train_loss 计算训练反向 loss（始终 fp32）。

    - mse:       F.mse_loss
    - huber:     F.huber_loss(delta=args.huber_delta)
    - smooth_l1: F.smooth_l1_loss(beta=args.huber_delta)

    |e| < delta 时近似 L2，|e| > delta 时近似 L1，对 outlier 鲁棒。
    """
    pred_f = pred.float()
    loss_type = getattr(args, 'train_loss', 'mse')
    delta = float(getattr(args, 'huber_delta', 1.0))
    if loss_type == 'huber':
        return F.huber_loss(pred_f, target, reduction='mean', delta=delta)
    if loss_type == 'smooth_l1':
        return F.smooth_l1_loss(pred_f, target, reduction='mean', beta=delta)
    return F.mse_loss(pred_f, target, reduction='mean')


def train_batch(model, batch_x, batch_y, optimizer, revin, args, device, scaler):
    """训练一个batch"""
    batch_x = batch_x.to(device)  # [B, context_points, C]
    batch_y = batch_y.to(device)  # [B, target_points, C]
    
    # RevIN归一化
    if revin:
        batch_x = revin(batch_x, 'norm')
    
    with amp.autocast(enabled=scaler.is_enabled()):
        # 前向传播: 预测码本索引 -> 解码（支持自回归步长 + overlapping chunk 融合）
        pred, _ = model.forward_finetune(
            batch_x, args.target_points,
            step_size=args.ar_step_size, pred_len=args.pred_len,
        )

    # RevIN 反归一化 + loss 放到 autocast 外（fp32），与 val/test 一致，
    # 避免在 fp16 下对较大的 stdev 做乘法带来精度损失。
    if revin:
        pred = revin(pred, 'denorm')

    # 用于反向传播的训练 loss（可选 mse/huber/smooth_l1；val/test 始终用 MSE 报告）
    loss = _compute_train_loss(pred, batch_y, args)

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
    """验证一个epoch

    注意：验证始终在 FP32 下进行，避免 FP16 的 softmax/码本查表/RevIN denorm
    的精度损失导致 best epoch 选择不稳。参数 use_amp 仅为了兼容调用方签名。
    """
    model.eval()
    total_loss = 0
    total_token_acc = 0.0
    layer_acc_sum = None
    metric_batches = 0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if revin:
                batch_x = revin(batch_x, 'norm')
                batch_y_for_token = revin._normalize(batch_y)
            else:
                batch_y_for_token = batch_y

            # 验证强制 FP32（忽略 use_amp）
            pred, _, token_metrics = model.forward_finetune(
                batch_x, args.target_points,
                step_size=args.ar_step_size, pred_len=args.pred_len,
                target=batch_y_for_token, return_token_metrics=True,
            )

            if revin:
                pred = revin(pred, 'denorm')

            mse_loss = F.mse_loss(pred.float(), batch_y, reduction='mean')
            total_loss += mse_loss.item()
            if token_metrics.get('token_acc') is not None:
                total_token_acc += token_metrics['token_acc']
                if layer_acc_sum is None:
                    layer_acc_sum = [0.0] * len(token_metrics.get('layer_acc', []))
                for i, acc in enumerate(token_metrics.get('layer_acc', [])):
                    layer_acc_sum[i] += acc
                metric_batches += 1
            n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return {
        'loss': avg_loss,
        'token_acc': total_token_acc / metric_batches if metric_batches > 0 else None,
        'layer_acc': [v / metric_batches for v in layer_acc_sum] if metric_batches > 0 else [],
    }


def test_model(model, dataloader, revin, args, device, use_amp):
    """测试模型

    注意：测试始终在 FP32 下进行。FP16 推理在 softmax/码本查表/RevIN denorm
    等算子上有 micro 精度损失，对 SOTA 级别的 MSE 比较敏感。参数 use_amp
    仅为了兼容调用方签名，实际被忽略。
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_token_acc = 0.0
    layer_acc_sum = None
    metric_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if revin:
                batch_x = revin(batch_x, 'norm')
                batch_y_for_token = revin._normalize(batch_y)
            else:
                batch_y_for_token = batch_y

            # 测试强制 FP32（忽略 use_amp）
            pred, _, token_metrics = model.forward_finetune(
                batch_x, args.target_points,
                step_size=args.ar_step_size, pred_len=args.pred_len,
                target=batch_y_for_token, return_token_metrics=True,
            )

            assert pred.shape[1] == batch_y.shape[1] == args.target_points, \
                f"预测长度 {pred.shape[1]} 与目标长度 {batch_y.shape[1]} 或 args.target_points {args.target_points} 不匹配"

            if revin:
                pred = revin(pred, 'denorm')

            if token_metrics.get('token_acc') is not None:
                total_token_acc += token_metrics['token_acc']
                if layer_acc_sum is None:
                    layer_acc_sum = [0.0] * len(token_metrics.get('layer_acc', []))
                for i, acc in enumerate(token_metrics.get('layer_acc', [])):
                    layer_acc_sum[i] += acc
                metric_batches += 1

            all_preds.append(pred.float().cpu())
            all_targets.append(batch_y.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    token_metrics = {
        'token_acc': total_token_acc / metric_batches if metric_batches > 0 else None,
        'layer_acc': [v / metric_batches for v in layer_acc_sum] if metric_batches > 0 else [],
    }

    return mse, mae, preds, targets, token_metrics


def main():
    args = parse_args()
    print('Args:', args)
    
    # CausalTransformer 现已走 is_causal=True 的 fused SDP 路径，无需禁用 flash/mem-efficient attention

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('medium')
    
    # 创建保存目录
    save_dir = Path(args.save_path) / args.dset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 先获取数据
    args.dset_finetune = args.dset
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Train batches: {len(dls.train)}, Valid batches: {len(dls.valid)}, Test batches: {len(dls.test)}')
    
    # 加载预训练模型（传入args以配置Gumbel-Softmax）
    model, config, pretrain_args = load_pretrained_model(args.pretrained_model, device, n_channels=dls.vars, args=args)
    
    # 验证config完整性（确保所有必要的配置项都存在）
    required_config_keys = ['patch_size', 'embedding_dim', 'compression_factor', 'codebook_size', 
                           'n_layers', 'n_heads', 'd_ff', 'dropout', 'num_hiddens', 
                           'num_residual_layers', 'num_residual_hiddens']
    missing_keys = [key for key in required_config_keys if key not in config]
    if missing_keys:
        raise ValueError(f"配置不完整，缺少以下键: {missing_keys}")
    
    print(f'✓ 模型配置验证通过: {config}')
    
    # 自动继承预训练的step_size（如果finetune时未指定ar_step_size）
    if args.ar_step_size is None and 'progressive_step_size' in pretrain_args:
        args.ar_step_size = pretrain_args['progressive_step_size']
        print(f'✓ 自动继承预训练step_size: {args.ar_step_size}')

    # 自动继承预训练的 pred_len（如果 finetune 时未指定，且预训练启用了 overlapping chunk）
    if args.pred_len is None and 'pred_len' in pretrain_args and pretrain_args['pred_len'] is not None:
        args.pred_len = pretrain_args['pred_len']
        print(f'✓ 自动继承预训练 pred_len: {args.pred_len}')

    # 冻结 Encoder + VQ；根据 --unfreeze_decoder 决定 Decoder 是否参与 finetune
    unfreeze_dec = bool(getattr(args, 'unfreeze_decoder', 0))
    freeze_encoder_vq(model, unfreeze_decoder=unfreeze_dec)
    if unfreeze_dec:
        print(f'✓ Decoder 已解冻（decoder_lr = main_lr × {args.decoder_lr_ratio}, '
              f'decoder_wd = main_wd × {args.decoder_wd_ratio}）')
    else:
        print('✓ Decoder 保持冻结（向后兼容行为）')
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    
    print(f'\n模型参数统计:')
    print(f'  总参数: {total_params:,}')
    print(f'  可训练参数: {trainable_params:,}')
    print(f'  冻结参数: {frozen_params:,}')
    
    # AMP
    use_amp = bool(args.amp) and device.type == 'cuda'
    scaler = amp.GradScaler(enabled=use_amp)
    print(f'AMP enabled: {use_amp}')
    if args.train_loss == 'mse':
        print('训练 loss: MSE（验证/测试也用 MSE）')
    else:
        print(f'训练 loss: {args.train_loss.upper()} (delta/beta={args.huber_delta})'
              f'  | 验证/测试仍用 MSE')
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 模型文件名
    # 如果提供了 run_id，则使用它；否则尝试从预训练模型路径中提取
    run_id = args.run_id
    if run_id is None:
        # 尝试从预训练模型路径中提取 run_id
        import re
        pretrained_model_name = Path(args.pretrained_model).stem
        match = re.search(r'_run(\d+)_', pretrained_model_name)
        if match:
            run_id = int(match.group(1))
    
    if run_id is not None:
        model_name = f'patch_vqvae_finetune_cw{args.context_points}_tw{args.target_points}_run{run_id}_model{args.model_id}'
    else:
        model_name = f'patch_vqvae_finetune_cw{args.context_points}_tw{args.target_points}_model{args.model_id}'
    
    # 优化器和调度器
    # 解冻 decoder 时分组：decoder 用更小 lr + 更大 weight_decay，
    # 保护 VQ-VAE 在 pretrain 中学到的几何结构，避免过拟合 ETTh1 这种小数据集。
    if unfreeze_dec:
        decoder_params = [p for n, p in model.named_parameters()
                          if n.startswith('decoder.') and p.requires_grad]
        other_params   = [p for n, p in model.named_parameters()
                          if (not n.startswith('decoder.')) and p.requires_grad]
        dec_lr = args.lr * float(args.decoder_lr_ratio)
        dec_wd = args.weight_decay * float(args.decoder_wd_ratio)
        optimizer = AdamW(
            [
                {'params': other_params,   'lr': args.lr, 'weight_decay': args.weight_decay},
                {'params': decoder_params, 'lr': dec_lr,  'weight_decay': dec_wd},
            ],
        )
        n_dec = sum(p.numel() for p in decoder_params)
        n_oth = sum(p.numel() for p in other_params)
        print(f'  Optimizer groups: main={n_oth:,} params (lr={args.lr}, wd={args.weight_decay})'
              f' | decoder={n_dec:,} params (lr={dec_lr}, wd={dec_wd})')
    else:
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    
    # 训练
    best_val_loss = float('inf')
    train_losses, valid_losses, valid_token_accs = [], [], []
    no_improve_epochs = 0  # 连续无改善的epoch数
    early_stop_patience = 5  # 连续5个epoch无下降就停止
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
        val_metrics = validate_epoch(model, dls.valid, revin, args, device, use_amp)
        val_loss = val_metrics['loss']
        
        train_losses.append(avg_train_loss)
        valid_losses.append(val_loss)
        valid_token_accs.append(val_metrics.get('token_acc'))
        
        total_time = time.time() - start_time
        
        # 检查当前epoch是否改善了最佳验证损失
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0  # 重置计数器
            
            # 保存最佳模型
            # 确保使用原始config（从预训练checkpoint加载的），确保架构一致性
            import copy
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': copy.deepcopy(config),  # 深拷贝，确保config不被后续修改影响
                'args': vars(args),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat(),
                'total_training_time_seconds': total_time,
            }
            # 验证保存的config与预训练时一致
            print(f'保存checkpoint时的config验证: num_residual_hiddens={checkpoint["config"].get("num_residual_hiddens")}')
            torch.save(checkpoint, save_dir / f'{model_name}.pth')
            status = "*Best*"
        else:
            no_improve_epochs += 1
            status = ""
        
        # 打印进度
        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | Valid Loss: {val_loss:.6f} | "
              f"Time: {total_time/60:.1f}min {status}")
        if val_metrics.get('token_acc') is not None:
            layer_text = ', '.join(
                f'L{i}:{acc * 100:.1f}%' for i, acc in enumerate(val_metrics.get('layer_acc', []))
            )
            print(f"  └─ Forecast Token Acc: Val {val_metrics['token_acc'] * 100:.2f}%"
                  + (f" ({layer_text})" if layer_text else ""))
        
        if not is_best:
            print(f"  -> 无改善 (当前最佳: epoch {best_epoch+1}, val_loss: {best_val_loss:.6f}, "
                  f"连续 {no_improve_epochs} 个epoch无改善)")
        
        # 早停检查：连续10个epoch无改善
        if no_improve_epochs >= early_stop_patience:
            print(f"\n>>> 早停: 连续 {early_stop_patience} 个 epoch 无改善")
            # 保存当前模型（10个epoch无改善时的模型）
            import copy
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': copy.deepcopy(config),  # 深拷贝，确保config不被后续修改影响
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
    # PyTorch 2.6+ 兼容性：设置 weights_only=False 以支持包含 numpy 对象的 checkpoint
    best_checkpoint = torch.load(save_dir / f'{model_name}.pth', map_location=device, weights_only=False)
    
    # 始终使用checkpoint中的config重新创建模型，确保架构完全一致
    import copy
    checkpoint_config = copy.deepcopy(best_checkpoint.get('config', {}))
    if not checkpoint_config:
        raise ValueError(f"Checkpoint中缺少config！文件: {save_dir / f'{model_name}.pth'}")
    
    print(f"使用checkpoint中的config创建模型:")
    print(f"  num_residual_hiddens: {checkpoint_config.get('num_residual_hiddens')}")
    print(f"  num_hiddens: {checkpoint_config.get('num_hiddens')}")
    print(f"  num_residual_layers: {checkpoint_config.get('num_residual_layers')}")
    print(f"  n_layers: {checkpoint_config.get('n_layers')}")
    print(f"  n_heads: {checkpoint_config.get('n_heads')}")
    
    # 确保config包含必要的字段
    checkpoint_config['n_channels'] = dls.vars  # 确保通道数正确
    if args is not None:
        checkpoint_config['use_gumbel_softmax'] = bool(getattr(args, 'use_gumbel_softmax', 1))
        checkpoint_config['gumbel_temperature'] = getattr(args, 'gumbel_temperature', 1.0)
        checkpoint_config['gumbel_hard'] = bool(getattr(args, 'gumbel_hard', 0))
    
    # 使用checkpoint中的config重新创建模型（确保架构完全一致）
    model = PatchVQVAETransformer(checkpoint_config).to(device)
    # 测试阶段只跑 no_grad，冻/不冻不影响结果；这里保持与训练阶段一致以防混淆
    freeze_encoder_vq(model, unfreeze_decoder=bool(getattr(args, 'unfreeze_decoder', 0)))
    print("✓ 已使用checkpoint config重新创建模型")
    
    # 加载权重
    try:
        model.load_state_dict(best_checkpoint['model_state_dict'], strict=True)
        print("✓ 权重加载成功（strict=True）")
    except RuntimeError as e:
        print(f"警告: strict=True加载失败，尝试strict=False: {e}")
        missing_keys, unexpected_keys = model.load_state_dict(best_checkpoint['model_state_dict'], strict=False)
        if missing_keys:
            print(f"警告: 以下权重未加载: {missing_keys[:10]}..." if len(missing_keys) > 10 else f"警告: 以下权重未加载: {missing_keys}")
        if unexpected_keys:
            print(f"警告: 以下权重未使用: {unexpected_keys[:10]}..." if len(unexpected_keys) > 10 else f"警告: 以下权重未使用: {unexpected_keys}")
    
    mse, mae, preds, targets, test_token_metrics = test_model(model, dls.test, revin, args, device, use_amp)
    print(f'测试结果: MSE = {mse:.6f}, MAE = {mae:.6f}')
    if test_token_metrics.get('token_acc') is not None:
        layer_text = ', '.join(
            f'L{i}:{acc * 100:.1f}%' for i, acc in enumerate(test_token_metrics.get('layer_acc', []))
        )
        print(f"测试 Token Acc: {test_token_metrics['token_acc'] * 100:.2f}%"
              + (f" ({layer_text})" if layer_text else ""))
    
    # 保存结果
    result_metrics = ['MSE', 'MAE']
    result_values = [mse, mae]
    if test_token_metrics.get('token_acc') is not None:
        result_metrics.append('TokenAcc')
        result_values.append(test_token_metrics['token_acc'])
        for i, acc in enumerate(test_token_metrics.get('layer_acc', [])):
            result_metrics.append(f'TokenAcc_L{i}')
            result_values.append(acc)
    results_df = pd.DataFrame({
        'metric': result_metrics,
        'value': result_values,
    })
    results_df.to_csv(save_dir / f'{model_name}_results.csv', index=False)
    
    # 保存训练历史 (基于epoch)
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'valid_loss': valid_losses,
        'valid_token_acc': valid_token_accs,
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
