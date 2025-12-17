"""
中间步骤: 构建码本
- 加载阶段1预训练的模型
- 使用 embedding layer 获取所有 patch 的 embedding
- 对 embeddings 进行聚类
- 将聚类中心作为码本
"""

import numpy as np
import os
import sys
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
import argparse
from pathlib import Path
import json
from tqdm import tqdm

# 添加根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.mlm_encoder import MLMEncoder
from src.models.layers.revin import RevIN
from src.basics import set_device
from datautils import get_dls


def parse_args():
    parser = argparse.ArgumentParser(description='构建码本 (聚类)')
    
    # 数据集参数
    parser.add_argument('--dset', type=str, default='ettm1', help='数据集名称')
    parser.add_argument('--context_points', type=int, default=512, help='输入序列长度')
    parser.add_argument('--target_points', type=int, default=0, help='预测长度 (不使用)')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--scaler', type=str, default='standard', help='数据缩放方式')
    parser.add_argument('--features', type=str, default='M', help='特征类型')
    
    # 模型参数
    parser.add_argument('--mlm_model', type=str, required=True, help='MLM预训练模型路径')
    parser.add_argument('--codebook_size', type=int, default=256, help='码本大小')
    parser.add_argument('--cluster_method', type=str, default='minibatch_kmeans', 
                        choices=['kmeans', 'minibatch_kmeans'], help='聚类方法')
    parser.add_argument('--max_samples', type=int, default=100000, 
                       help='最大采样数量（用于聚类）')
    
    # 数据采样参数
    parser.add_argument('--train_sample_ratio', type=float, default=1.0, 
                       help='训练集采样比例 (0.0-1.0)')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/codebook/', help='模型保存路径')
    parser.add_argument('--revin', type=int, default=1, help='是否使用RevIN')
    
    return parser.parse_args()


def collect_embeddings(model, dataloader, revin, device, max_samples=100000):
    """
    收集所有 patch 的 embedding
    """
    model.eval()
    all_embeddings = []
    total_samples = 0
    
    print("收集 embeddings...")
    with torch.no_grad():
        for batch_x, _ in tqdm(dataloader):
            batch_x = batch_x.to(device)
            
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            # 获取 embeddings
            embeddings = model.get_embeddings(batch_x)  # [B*num_patches*C, d_model]
            all_embeddings.append(embeddings.cpu())
            
            total_samples += embeddings.size(0)
            if total_samples >= max_samples:
                break
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"收集到 {all_embeddings.size(0)} 个 embeddings, 维度: {all_embeddings.size(1)}")
    
    # 如果太多，随机采样
    if all_embeddings.size(0) > max_samples:
        perm = torch.randperm(all_embeddings.size(0))[:max_samples]
        all_embeddings = all_embeddings[perm]
        print(f"随机采样到 {max_samples} 个 embeddings")
    
    return all_embeddings


def build_codebook(embeddings, codebook_size, method='minibatch_kmeans'):
    """
    使用聚类构建码本
    """
    print(f"\n使用 {method} 聚类，码本大小: {codebook_size}")
    embeddings_np = embeddings.numpy()
    
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=codebook_size, random_state=42, n_init=10, verbose=1)
        kmeans.fit(embeddings_np)
    elif method == 'minibatch_kmeans':
        kmeans = MiniBatchKMeans(n_clusters=codebook_size, random_state=42, 
                                  batch_size=1024, n_init=10, verbose=1)
        kmeans.fit(embeddings_np)
    
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    
    # 统计每个聚类的样本数
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n聚类统计:")
    print(f"  - 使用的聚类数: {len(unique)}/{codebook_size}")
    print(f"  - 最大聚类大小: {counts.max()}")
    print(f"  - 最小聚类大小: {counts.min()}")
    print(f"  - 平均聚类大小: {counts.mean():.1f}")
    
    return centroids, labels


def main():
    args = parse_args()
    print('Args:', args)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载MLM模型
    print(f'\n加载MLM模型: {args.mlm_model}')
    checkpoint = torch.load(args.mlm_model, map_location=device)
    config = checkpoint['config']
    
    model = MLMEncoder(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'MLM模型加载成功，验证损失: {checkpoint.get("val_loss", "N/A")}')
    
    # 码本大小（用于后续NTP模型）
    codebook_size = args.codebook_size
    
    # 获取数据
    args.dset_pretrain = args.dset
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Train batches: {len(dls.train)}')
    
    # 对训练集进行采样（如果指定了采样比例）
    if args.train_sample_ratio < 1.0:
        from torch.utils.data import Subset, DataLoader
        
        train_dataset = dls.train.dataset
        train_size = len(train_dataset)
        sample_size = int(train_size * args.train_sample_ratio)
        indices = torch.randperm(train_size)[:sample_size].tolist()
        train_subset = Subset(train_dataset, indices)
        dls.train = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=getattr(dls.train, 'collate_fn', None)
        )
        print(f'训练集采样: {sample_size}/{train_size} ({args.train_sample_ratio*100:.1f}%)')
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 收集 embeddings
    embeddings = collect_embeddings(model, dls.train, revin, device, args.max_samples)
    
    # 构建码本
    centroids, labels = build_codebook(embeddings, codebook_size, args.cluster_method)
    
    # 保存VQ_encoder和codebook（这是MLM阶段唯一需要保存的两个组件）
    save_dir = Path(args.save_path) / args.dset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 从MLM模型名提取信息
    mlm_name = Path(args.mlm_model).stem
    codebook_name = f'{mlm_name}_codebook_cb{codebook_size}'
    
    # 保存VQ_encoder（patch_embedding）和codebook（centroids）
    vq_codebook_checkpoint = {
        'vq_encoder_state_dict': model.patch_embedding.state_dict(),  # VQ_encoder
        'centroids': centroids,  # codebook
        'config': config,
        'mlm_args': checkpoint.get('args', {}),
        'codebook_size': codebook_size,
        'cluster_method': args.cluster_method,
        'd_model': config['d_model'],
        'patch_size': config['patch_size'],
    }
    torch.save(vq_codebook_checkpoint, save_dir / f'{codebook_name}.pth')
    
    # 保存码本和聚类信息
    np.savez(save_dir / f'{codebook_name}_codebook.npz',
             centroids=centroids.numpy(),
             labels=labels)
    
    print('=' * 80)
    print(f'VQ_encoder和codebook构建完成！')
    print(f'保存内容:')
    print(f'  - VQ_encoder (patch_embedding): {save_dir / codebook_name}.pth')
    print(f'  - Codebook (centroids): {save_dir / codebook_name}.pth')
    print(f'  - 聚类数据: {save_dir / codebook_name}_codebook.npz')
    print(f'\n下一步: 运行 decoder-only/patch_vqvae_pretrain.py 进行 NTP 预训练')
    print(f'使用参数: --mlm_codebook_path {save_dir / codebook_name}.pth')
    print(f'文件路径: {save_dir / codebook_name}.pth')


if __name__ == '__main__':
    set_device()
    main()
