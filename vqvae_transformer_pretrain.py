"""
VQVAE + Transformer 预训练脚本
使用标准 PyTorch 训练循环，不使用 Learner
"""

import numpy as np
import pandas as pd
import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.models.vqvae_transformer import VQVAETransformerPretrain
from src.models.layers.revin import RevIN
from src.basics import set_device
from src.utils_vqvae import load_vqvae_config
from datautils import get_dls

import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=str, default='ettm1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# VQVAE config
parser.add_argument('--vqvae_config_path', type=str, 
                   default='saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/configs/config_file.json')
parser.add_argument('--vqvae_checkpoint', type=str, default=None,
                   help='Path to pretrained VQVAE checkpoint (optional)')
parser.add_argument('--beta', type=float, default=0.0,
                   help='Weight for reconstruction loss (0.0 means no reconstruction loss)')
# Transformer config
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=8, help='number of Transformer heads')
parser.add_argument('--d_ff', type=int, default=256, help='Transformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='Transformer dropout')
parser.add_argument('--attn_dropout', type=float, default=0.1, help='attention dropout')
parser.add_argument('--transformer_parameter_path', type=str, default='model_config/', help='to store transformer parameter')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=50, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='vqvae_transformer', help='model type for saving')

args = parser.parse_args()
print('args:', args)
args.save_pretrained_model = (
    'vqvae_transformer_pretrained'
    + '_cw' + str(args.context_points)
    + '_d' + str(args.d_model)
    + '_l' + str(args.n_layers)
    + '_h' + str(args.n_heads)
    + '_epochs-pretrain' + str(args.n_epochs_pretrain)
    + '_model' + str(args.pretrained_model_id)
)
args.save_path = 'saved_models/' + args.dset_pretrain + '/' + args.model_type + '/'
if not os.path.exists(args.save_path): 
    os.makedirs(args.save_path)

# get available GPU device
set_device()

def save_transformer_config(args, filename="transformer_config.json"):
    """
    将 Transformer 相关参数保存为 JSON 文件
    """
    # 1) 需要保存的 Transformer 参数
    transformer_config = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "attn_dropout": args.attn_dropout,
    }

    # 2) 创建路径（如果不存在）
    save_dir = args.transformer_parameter_path
    os.makedirs(save_dir, exist_ok=True)

    # 3) 文件完整路径
    save_path = os.path.join(save_dir, filename)

    # 4) 保存为 JSON
    with open(save_path, "w") as f:
        json.dump(transformer_config, f, indent=4)

    print(f"Transformer 参数已保存到 {save_path}")


def visualize_codebook(model, save_path, args):
    """可视化码本向量并保存图像"""
    if not hasattr(model, 'vq'):
        print("模型不包含VQ层，跳过码本可视化")
        return

    codebook = model.vq._embedding.weight.data.cpu().numpy()  # [num_embeddings, embedding_dim]
    num_embeddings, embedding_dim = codebook.shape

    print(f"\n=== 最终码本可视化 ===")
    print(f"码本大小: {num_embeddings}, 嵌入维度: {embedding_dim}")

    # 计算码本统计信息
    norms = np.linalg.norm(codebook, axis=1)
    print(f"码本向量范数 - 平均: {norms.mean():.4f}, 最大: {norms.max():.4f}, 最小: {norms.min():.4f}")

    # 降维可视化
    if embedding_dim == 2:
        # 2D直接可视化
        reduced_codebook = codebook
        method = "原始2D"
    elif embedding_dim > 2:
        # 使用t-SNE降维到2D（如果码本不大）
        if num_embeddings <= 1000:
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, num_embeddings-1))
                reduced_codebook = tsne.fit_transform(codebook)
                method = "t-SNE"
            except:
                # t-SNE失败时使用PCA
                pca = PCA(n_components=2, random_state=42)
                reduced_codebook = pca.fit_transform(codebook)
                method = "PCA"
        else:
            # 大码本使用PCA
            pca = PCA(n_components=2, random_state=42)
            reduced_codebook = pca.fit_transform(codebook)
            method = "PCA"
    else:
        print("嵌入维度小于2，跳过可视化")
        return

    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 子图1：码本向量分布
    scatter = axes[0].scatter(reduced_codebook[:, 0], reduced_codebook[:, 1],
                             c=norms, cmap='viridis', alpha=0.7, s=50)
    axes[0].set_title(f'码本向量分布 ({method}降维)')
    axes[0].set_xlabel('维度1')
    axes[0].set_ylabel('维度2')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='向量范数')

    # 添加向量索引标签（只对小码本）
    if num_embeddings <= 50:
        for i, (x, y) in enumerate(reduced_codebook):
            axes[0].annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points',
                           fontsize=8, alpha=0.8)

    # 子图2：码本向量范数分布直方图
    axes[1].hist(norms, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1].set_title('码本向量范数分布')
    axes[1].set_xlabel('向量范数')
    axes[1].set_ylabel('频次')
    axes[1].grid(True, alpha=0.3)

    # 添加统计信息文本
    stats_text = f"""
数据集: {args.dset_pretrain}
码本大小: {num_embeddings}
嵌入维度: {embedding_dim}
降维方法: {method}
平均范数: {norms.mean():.4f}
范数标准差: {norms.std():.4f}
最大范数: {norms.max():.4f}
最小范数: {norms.min():.4f}
"""
    fig.suptitle('VQVAE Transformer 码本分析', fontsize=14, fontweight='bold')
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # 保存图像
    viz_path = os.path.join(save_path, f'{args.save_pretrained_model}_codebook_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"码本可视化已保存到: {viz_path}")

    # 保存码本数据为numpy文件（用于后续分析）
    npz_path = os.path.join(save_path, f'{args.save_pretrained_model}_codebook.npz')
    np.savez(npz_path, codebook=codebook, norms=norms, reduced_codebook=reduced_codebook)
    print(f"码本数据已保存到: {npz_path}")

    plt.close()


def compute_next_token_loss(model, x, compression_factor, device='cuda', beta=0.0):
    """
    计算 next token prediction 损失
    
    Args:
        model: VQVAETransformerPretrain 模型
        x: [B, L, C] 输入时间序列
        compression_factor: VQVAE 压缩因子
        device: 设备
        beta: 重构损失的权重（已废弃，不再使用）
    
    Returns:
        loss: 标量损失值
    """
    B, L, C = x.shape
    T_compressed = L // compression_factor
    
    # 创建 causal mask
    causal_mask = torch.triu(torch.ones(T_compressed, T_compressed, device=device), diagonal=1)
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
    
    # 获取真实的 codebook indices（通过 VQVAE encoder + VQ）
    target_indices = []
    with torch.no_grad():
        for ch in range(C):
            x_ch = x[:, :, ch].view(B, L)
            # 使用 VQVAE encoder 和 VQ 获取真实 indices
            z = model.vqvae_encoder(x_ch, compression_factor)
            z = z.permute(0, 2, 1)  # [B, T/compressed, embedding_dim]
            _, _, _, _, encoding_indices, _ = model.vq(z.permute(0, 2, 1).contiguous())
            # encoding_indices: [B*T/compressed, 1]
            indices = encoding_indices.squeeze(-1).view(B, T_compressed)
            target_indices.append(indices)
    
    target_indices = torch.stack(target_indices, dim=2)  # [B, T/compressed, C]
    
    # 前向传播（使用 causal mask）
    model._current_mask = causal_mask
    logits = model(x)  # [B, codebook_size, T/compression_factor, C]
    
    # 计算损失（next token prediction）
    
    ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    loss = 0
    for ch in range(C):
        logits_ch = logits[:, :, :, ch].permute(0, 2, 1)  # [B, T/compressed, codebook_size]
        target_ch = target_indices[:, :, ch]  # [B, T/compressed]
        
        # Next token prediction: 预测位置 i+1 的 token，使用位置 i 的 context
        pred_logits = logits_ch[:, :-1, :].reshape(-1, logits_ch.shape[-1])  # [B*(T/compressed-1), codebook_size]
        target_tokens = target_ch[:, 1:].reshape(-1)  # [B*(T/compressed-1)]
        
        loss += ce_criterion(pred_logits, target_tokens)
    
    loss = loss / C
    
    # 不再计算重构损失（decoder 已移除）
    # 如果 beta > 0，会打印警告但不会计算重构损失
    
    return loss


def get_model(c_in, args, vqvae_config, transformer_config, device='cpu'):
    """
    c_in: number of variables
    """
    # 创建VQVAE+Transformer模型
    model = VQVAETransformerPretrain(
        vqvae_config,
        transformer_config,
        load_vqvae_weights=args.vqvae_checkpoint is not None,
        vqvae_checkpoint_path=args.vqvae_checkpoint,
        device=device
    )
    
    # 对 Transformer 部分进行更好的初始化
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # 使用 Xavier 初始化
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)
    
    # 只对 Transformer 部分初始化（如果 VQVAE 是预训练的，不要重新初始化）
    if args.vqvae_checkpoint is None:
        # 如果没有加载 VQVAE 权重，初始化所有层
        model.apply(init_weights)
    else:
        # 只初始化 Transformer 部分
        model.projection.apply(init_weights)
        model.transformer_layers.apply(init_weights)
        model.codebook_head.apply(init_weights)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数数量: {total_params}, 可训练参数: {trainable_params}')
    return model


def find_lr():
    """简单的学习率查找：使用一个小的训练循环"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # get dataloader
    dls = get_dls(args)
    
    # 加载VQVAE配置
    vqvae_config = load_vqvae_config(args.vqvae_config_path)
    
    # Transformer配置
    transformer_config = {
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'attn_dropout': args.attn_dropout
    }
    
    model = get_model(dls.vars, args, vqvae_config, transformer_config)
    model = model.to(device)
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False) if args.revin else None
    if revin:
        revin = revin.to(device)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # 简单的学习率查找：尝试几个不同的学习率（从更大的学习率开始）
    best_lr = args.lr
    best_loss = float('inf')
    
    # 尝试更大的学习率范围
    for test_lr in [5e-5, 1e-4, 5e-4, 1e-3, 2e-3]:
        optimizer.param_groups[0]['lr'] = test_lr
        model.train()
        
        # 使用几个 batch 测试，取平均
        losses = []
        for batch_idx, (batch_x, _) in enumerate(dls.train):
            if batch_idx >= 5:  # 只测试前5个batch
                break
            batch_x = batch_x.to(device)
            
            # RevIN normalization
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            optimizer.zero_grad()
            loss = compute_next_token_loss(model, batch_x, vqvae_config['compression_factor'], device, beta=args.beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            losses.append(loss.item())
        
        avg_loss = np.mean(losses) if losses else float('inf')
        if avg_loss < best_loss and not np.isnan(avg_loss):
            best_loss = avg_loss
            best_lr = test_lr
    
    # 如果找到的学习率太小，使用默认值或稍微大一点的值
    if best_lr < 1e-4:
        best_lr = max(1e-4, args.lr * 2)
    
    print(f'suggested_lr = {best_lr}')
    return best_lr


def pretrain_func(lr=args.lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # get dataloader
    dls = get_dls(args)
    
    # 加载VQVAE配置
    vqvae_config = load_vqvae_config(args.vqvae_config_path)
    
    # Transformer配置
    transformer_config = {
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'attn_dropout': args.attn_dropout
    }
    
    # get model
    model = get_model(dls.vars, args, vqvae_config, transformer_config)
    model = model.to(device)
    
    # RevIN
    revin = RevIN(dls.vars, eps=1e-5, affine=False) if args.revin else None
    if revin:
        revin = revin.to(device)
    
    # Optimizer and scheduler
    # 使用更大的学习率和调整 betas 以加速训练
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
    total_steps = len(dls.train) * args.n_epochs_pretrain
    # 减少 warmup 时间，更快达到最大学习率
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1, div_factor=10.0)
    
    # 清空 torch 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Training loop
    train_losses = []
    valid_losses = []
    best_val_loss = float('inf')
    
    print(f"开始预训练，共 {args.n_epochs_pretrain} 个 epoch")
    print(f"总参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    for epoch in range(args.n_epochs_pretrain):
        # Training
        model.train()
        epoch_train_losses = []
        
        for batch_idx, (batch_x, _) in enumerate(dls.train):
            batch_x = batch_x.to(device)
            
            # RevIN normalization
            if revin:
                batch_x = revin(batch_x, 'norm')
            
            optimizer.zero_grad()
            loss = compute_next_token_loss(model, batch_x, vqvae_config['compression_factor'], device)
            
            # 检查 loss 是否为 NaN 或 Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: Epoch {epoch+1}, Batch {batch_idx}, Loss is NaN/Inf: {loss.item()}!")
                continue
            
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_train_losses.append(loss.item())
            
            # 每 1000 个 batch 打印一次梯度信息（用于调试）
            if batch_idx % 1000 == 0 and batch_idx > 0:
                total_grad_norm = 0
                for p in model.parameters():
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
            for batch_x, _ in dls.valid:
                batch_x = batch_x.to(device)
                
                # RevIN normalization
                if revin:
                    batch_x = revin(batch_x, 'norm')
                
                loss = compute_next_token_loss(model, batch_x, vqvae_config['compression_factor'], device)
                epoch_valid_losses.append(loss.item())
        
        avg_valid_loss = np.mean(epoch_valid_losses)
        valid_losses.append(avg_valid_loss)
        
        # Save best model
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'transformer_config': transformer_config,
                'vqvae_config': vqvae_config,
                'epoch': epoch,
                'valid_loss': avg_valid_loss
            }
            torch.save(checkpoint, os.path.join(args.save_path, args.save_pretrained_model + '.pth'))
            print(f"Epoch {epoch+1}/{args.n_epochs_pretrain} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f} | *Best Model Saved*")
        else:
            print(f"Epoch {epoch+1}/{args.n_epochs_pretrain} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f}")
    
    # Save loss history
    df = pd.DataFrame(data={'train_loss': train_losses, 'valid_loss': valid_losses})
    df.to_csv(args.save_path + args.save_pretrained_model + '_losses.csv', float_format='%.6f', index=False)
    print(f"训练完成！损失历史已保存到 {args.save_path + args.save_pretrained_model + '_losses.csv'}")

    # 可视化最终码本
    try:
        visualize_codebook(model, args.save_path, args)
    except Exception as e:
        print(f"码本可视化失败: {e}")


if __name__ == '__main__':
    args.dset = args.dset_pretrain
    # 保存 transformer 配置
    save_transformer_config(args, f'{args.dset_pretrain}_transformer_config.json')
    # 查找学习率
    suggested_lr = find_lr()
    # Pretrain
    pretrain_func(suggested_lr)
    print('pretraining completed')
