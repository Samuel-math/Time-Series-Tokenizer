"""
Patch-based VQVAE + Transformer 模型评估脚本
直接加载微调好的模型并进行评估
"""

import numpy as np
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda import amp
import argparse

# 添加根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.patch_vqvae_transformer import PatchVQVAETransformer
from src.models.layers.revin import RevIN
from src.basics import set_device
from datautils import get_dls


def parse_args():
    parser = argparse.ArgumentParser(description='评估微调好的模型')
    
    # 数据集参数
    parser.add_argument('--dset', type=str, required=True, help='数据集名称')
    parser.add_argument('--context_points', type=int, default=512, help='输入序列长度')
    parser.add_argument('--target_points', type=int, required=True, help='预测长度')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--features', type=str, default='M', help='特征类型')
    
    # 模型路径
    parser.add_argument('--model_path', type=str, required=True, help='微调好的模型路径')
    
    # 评估参数
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN（需要与训练时一致）')
    parser.add_argument('--amp', type=int, default=1, help='是否使用混合精度')
    
    return parser.parse_args()


def evaluate_model(model, dataloader, revin, args, device, use_amp):
    """评估模型"""
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
            
            # 验证预测长度与目标长度一致
            assert pred.shape[1] == batch_y.shape[1] == args.target_points, \
                f"预测长度 {pred.shape[1]} 与目标长度 {batch_y.shape[1]} 或 args.target_points {args.target_points} 不匹配"
            
            if revin:
                pred = revin(pred, 'denorm')
            
            all_preds.append(pred.cpu())
            all_targets.append(batch_y.cpu())
    
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    # 计算指标
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)
    
    # 计算每个通道的指标
    n_channels = targets.shape[-1]
    channel_metrics = {}
    for c in range(n_channels):
        channel_mse = np.mean((preds[:, :, c] - targets[:, :, c]) ** 2)
        channel_mae = np.mean(np.abs(preds[:, :, c] - targets[:, :, c]))
        channel_rmse = np.sqrt(channel_mse)
        channel_metrics[f'channel_{c}'] = {
            'MSE': channel_mse,
            'MAE': channel_mae,
            'RMSE': channel_rmse
        }
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'channel_metrics': channel_metrics,
        'preds': preds,
        'targets': targets
    }


def main():
    args = parse_args()
    print('Args:', args)
    
    # PyTorch 2.7+ 兼容性修复：禁用 flash attention 和 memory-efficient attention
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
        print('✓ 已禁用 flash/memory-efficient attention（PyTorch 2.7+ 兼容性修复）')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    print(f'\n加载模型: {args.model_path}')
    # PyTorch 2.6+ 兼容性：设置 weights_only=False 以支持包含 numpy 对象的 checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    
    # 获取数据以知道通道数
    args.dset_eval = args.dset
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Test batches: {len(dls.test)}')
    
    # 创建模型
    config['n_channels'] = dls.vars
    model = PatchVQVAETransformer(config).to(device)
    
    # 加载权重
    model.load_state_dict(state_dict)
    print('✓ 模型加载成功')
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    if revin:
        print('✓ RevIN已启用')
    
    # AMP
    use_amp = bool(args.amp) and device.type == 'cuda'
    print(f'AMP enabled: {use_amp}')
    
    # 评估
    print('\n' + '=' * 80)
    print('开始评估...')
    print('=' * 80)
    
    results = evaluate_model(model, dls.test, revin, args, device, use_amp)
    
    # 打印结果
    print('\n评估结果:')
    print('=' * 80)
    print(f'MSE:  {results["MSE"]:.6f}')
    print(f'MAE:  {results["MAE"]:.6f}')
    print(f'RMSE: {results["RMSE"]:.6f}')
    print('\n各通道指标:')
    for channel_name, metrics in results['channel_metrics'].items():
        print(f'  {channel_name}:')
        print(f'    MSE:  {metrics["MSE"]:.6f}')
        print(f'    MAE:  {metrics["MAE"]:.6f}')
        print(f'    RMSE: {metrics["RMSE"]:.6f}')
    
    print('\n' + '=' * 80)
    print('评估完成！')
    print('=' * 80)


if __name__ == '__main__':
    main()
