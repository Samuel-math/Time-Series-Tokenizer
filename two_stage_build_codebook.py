"""
中间步骤: 构建码本
- 加载阶段1预训练的模型
- 使用 embedding layer 获取所有 patch 的 embedding
- 对 embeddings 进行聚类
- 将聚类中心作为码本
"""

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
import argparse
from pathlib import Path
import json
from tqdm import tqdm

from src.models.two_stage_pretrain import TwoStagePretrainModel
from src.models.layers.revin import RevIN
from src.basics import set_device
from datautils import get_dls


def parse_args():
    parser = argparse.ArgumentParser(description='构建码本 (聚类)')
    
    # 数据集参数
    parser.add_argument('--dset', type=str, default='ettm1', help='数据集名称')
    parser.add_argument('--context_points', type=int, default=512, help='输入序列长度')
    parser.add_argument('--target_points', type=int, default=96, help='预测长度 (不使用)')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--scaler', type=str, default='standard', help='数据缩放方式')
    parser.add_argument('--features', type=str, default='M', help='特征类型')
    
    # 模型参数
    parser.add_argument('--stage1_model', type=str, required=True, help='阶段1预训练模型路径')
    parser.add_argument('--codebook_size', type=int, default=256, help='码本大小')
    parser.add_argument('--cluster_method', type=str, default='minibatch_kmeans', 
                        choices=['kmeans', 'minibatch_kmeans'], help='聚类方法')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, default='saved_models/two_stage/', help='模型保存路径')
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
    
    # 加载阶段1模型
    print(f'\n加载阶段1模型: {args.stage1_model}')
    checkpoint = torch.load(args.stage1_model, map_location=device)
    config = checkpoint['config']
    
    # 更新码本大小
    config['codebook_size'] = args.codebook_size
    
    model = TwoStagePretrainModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'阶段1模型加载成功，验证损失: {checkpoint.get("val_loss", "N/A")}')
    
    # 获取数据
    dls = get_dls(args)
    print(f'Number of channels: {dls.vars}')
    print(f'Train batches: {len(dls.train)}')
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
    
    # 收集 embeddings
    embeddings = collect_embeddings(model, dls.train, revin, device)
    
    # 构建码本
    centroids, labels = build_codebook(embeddings, args.codebook_size, args.cluster_method)
    
    # 初始化模型的码本
    model.codebook.init_from_centroids(centroids.to(device))
    
    # 保存带有码本的模型
    save_dir = Path(args.save_path) / args.dset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 从阶段1模型名提取信息
    stage1_name = Path(args.stage1_model).stem
    model_name = f'{stage1_name}_cb{args.codebook_size}'
    
    checkpoint_with_codebook = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'stage1_args': checkpoint.get('args', {}),
        'codebook_size': args.codebook_size,
        'cluster_method': args.cluster_method,
    }
    torch.save(checkpoint_with_codebook, save_dir / f'{model_name}.pth')
    
    # 保存码本和聚类信息
    np.savez(save_dir / f'{model_name}_codebook.npz',
             centroids=centroids.numpy(),
             labels=labels)
    
    print('=' * 80)
    print(f'码本构建完成！')
    print(f'模型保存至: {save_dir / model_name}.pth')
    print(f'码本数据保存至: {save_dir / model_name}_codebook.npz')
    print(f'\n下一步: 运行 two_stage_pretrain_stage2.py 进行 NTP 预训练')


if __name__ == '__main__':
    set_device()
    main()
