"""
Patch-based VQVAE + Transformer 模型架构

架构说明：
1. 输入: [B, T, C] 时间序列
2. Overlapping Patch划分 (stride 控制重叠) -> [B, num_patches, patch_size, C]
3. Patch Encoder: 线性投影 [B, num_patches, patch_size * C] -> [B, num_patches, d_model]
4. VQ 量化到码本 -> [B, num_patches] indices
5. Decoder-only Transformer: 预测下一个码本索引
6. Patch Decoder: [B, num_patches, d_model] -> [B, num_patches, patch_size, C]
7. 预训练: NTP loss + Reconstruction loss
8. 微调: 预测未来 patch -> MSE loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEncoder(nn.Module):
    """
    Patch Encoder - 线性投影
    
    将 patch 展平后通过 MLP 编码为 d_model 维向量
    
    输入: [B, num_patches, patch_size, C]
    输出: [B, num_patches, d_model]
    """
    def __init__(self, patch_size, n_channels, d_model, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.d_model = d_model
        self.input_dim = patch_size * n_channels
        
        # MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, num_patches, patch_size, C]
        Returns:
            out: [B, num_patches, d_model]
        """
        B, num_patches, patch_size, C = x.shape
        
        # 展平 patch: [B, num_patches, patch_size * C]
        x = x.reshape(B, num_patches, -1)
        
        # 编码: [B, num_patches, d_model]
        out = self.encoder(x)
        
        return out


class PatchDecoder(nn.Module):
    """
    Patch Decoder - 线性投影
    
    将 d_model 维向量解码回 patch
    
    输入: [B, num_patches, d_model]
    输出: [B, num_patches, patch_size, C]
    """
    def __init__(self, patch_size, n_channels, d_model, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.d_model = d_model
        self.output_dim = patch_size * n_channels
        
        # MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, self.output_dim)
        )
    
    def forward(self, z):
        """
        Args:
            z: [B, num_patches, d_model]
        Returns:
            out: [B, num_patches, patch_size, C]
        """
        B, num_patches, _ = z.shape
        
        # 解码: [B, num_patches, patch_size * C]
        out = self.decoder(z)
        
        # reshape: [B, num_patches, patch_size, C]
        out = out.reshape(B, num_patches, self.patch_size, self.n_channels)
        
        return out


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with Straight-Through Estimator
    支持 EMA 更新码本 (更稳定的训练)
    """
    def __init__(self, codebook_size, d_model, commitment_cost=0.25, use_ema=True, ema_decay=0.99):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        # 码本: [codebook_size, d_model]
        self.embedding = nn.Embedding(codebook_size, d_model)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        
        if use_ema:
            # EMA 相关的 buffer
            self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
            self.register_buffer('ema_embedding_avg', self.embedding.weight.data.clone())
    
    def forward(self, z):
        """
        Args:
            z: [B, num_patches, d_model]
        Returns:
            loss, quantized, indices
        """
        input_shape = z.shape
        flat = z.reshape(-1, self.d_model)  # [N, d_model]
        
        # 计算距离
        distances = (
            torch.sum(flat ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.matmul(flat, self.embedding.weight.t())
        )
        
        # 找最近的码本向量
        indices = torch.argmin(distances, dim=1)
        
        # One-hot encoding
        encodings = F.one_hot(indices, self.codebook_size).float()
        
        # 量化
        quantized = self.embedding(indices)
        
        if self.use_ema and self.training:
            # EMA 更新
            with torch.no_grad():
                # 更新聚类大小
                cluster_size = encodings.sum(0)
                self.ema_cluster_size.data.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
                
                # 更新嵌入平均值
                dw = encodings.t() @ flat
                self.ema_embedding_avg.data.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
                
                # 归一化得到新的码本向量
                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
                self.embedding.weight.data.copy_(self.ema_embedding_avg / cluster_size.unsqueeze(1))
            
            # EMA 模式下只需要 commitment loss
            loss = self.commitment_cost * F.mse_loss(quantized.detach(), flat)
        else:
            # 原始 VQ Loss
            e_latent_loss = F.mse_loss(quantized.detach(), flat)
            q_latent_loss = F.mse_loss(quantized, flat.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = flat + (quantized - flat).detach()
        
        # Reshape back
        quantized = quantized.view(input_shape)
        indices = indices.view(input_shape[:-1])  # [B, num_patches]
        
        return loss, quantized, indices
    
    def get_embedding(self, indices):
        return self.embedding(indices)
    
    def reset_unused_codes(self, z, threshold=1.0):
        """
        重置未使用的码本向量
        将使用频率低于 threshold 的码本向量重置为随机的 encoder 输出
        """
        with torch.no_grad():
            flat = z.reshape(-1, self.d_model)
            
            # 找到使用频率低的码本
            if self.use_ema:
                usage = self.ema_cluster_size
            else:
                # 计算当前 batch 的使用情况
                distances = (
                    torch.sum(flat ** 2, dim=1, keepdim=True) +
                    torch.sum(self.embedding.weight ** 2, dim=1) -
                    2 * torch.matmul(flat, self.embedding.weight.t())
                )
                indices = torch.argmin(distances, dim=1)
                usage = torch.bincount(indices, minlength=self.codebook_size).float()
            
            # 重置使用率低的码本
            unused_mask = usage < threshold
            n_unused = unused_mask.sum().item()
            
            if n_unused > 0:
                # 随机选择 encoder 输出来重置
                random_indices = torch.randperm(flat.size(0))[:int(n_unused)]
                self.embedding.weight.data[unused_mask] = flat[random_indices]
                
                if self.use_ema:
                    self.ema_cluster_size.data[unused_mask] = 1.0
                    self.ema_embedding_avg.data[unused_mask] = flat[random_indices]
            
            return n_unused


class CausalTransformer(nn.Module):
    """Decoder-only Causal Transformer"""
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        
        # 位置编码
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, d_model]
        """
        B, T, _ = x.shape
        
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)
        x = self.drop(x)
        
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.norm(x)


class PatchVQTransformer(nn.Module):
    """
    Patch-based VQVAE + Transformer
    
    架构:
    - PatchEncoder: [B, num_patches, patch_size, C] -> [B, num_patches, d_model]
    - VQ: [B, num_patches, d_model] -> indices + z_q
    - Transformer: NTP 预训练
    - PatchDecoder: [B, num_patches, d_model] -> [B, num_patches, patch_size, C]
    """
    def __init__(self, config):
        super().__init__()
        
        # ========== Patch 配置 ==========
        self.patch_size = config.get('patch_size', 16)
        self.stride = config.get('stride', 16)
        self.n_channels = config.get('n_channels', 7)
        
        # ========== 模型配置 ==========
        self.d_model = config.get('d_model', 128)
        self.codebook_size = config.get('codebook_size', 256)
        self.commitment_cost = config.get('commitment_cost', 0.25)
        self.use_ema = config.get('use_ema', True)  # 默认使用 EMA
        
        # ========== Transformer 配置 ==========
        self.n_layers = config.get('n_layers', 4)
        self.n_heads = config.get('n_heads', 8)
        self.d_ff = config.get('d_ff', 512)
        self.dropout = config.get('dropout', 0.1)
        
        # ========== 模块 ==========
        
        # 1. Patch Encoder (线性投影)
        self.encoder = PatchEncoder(
            patch_size=self.patch_size,
            n_channels=self.n_channels,
            d_model=self.d_model,
            dropout=self.dropout
        )
        
        # 2. Vector Quantizer (支持 EMA)
        self.vq = VectorQuantizer(
            codebook_size=self.codebook_size,
            d_model=self.d_model,
            commitment_cost=self.commitment_cost,
            use_ema=self.use_ema
        )
        
        # 3. Causal Transformer
        self.transformer = CausalTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout
        )
        
        # 4. Output Head: 预测码本索引
        self.output_head = nn.Linear(self.d_model, self.codebook_size)
        
        # 5. Patch Decoder (线性投影)
        self.decoder = PatchDecoder(
            patch_size=self.patch_size,
            n_channels=self.n_channels,
            d_model=self.d_model,
            dropout=self.dropout
        )
    
    def _update_n_channels(self, n_channels):
        """动态更新通道数"""
        if n_channels != self.n_channels:
            self.n_channels = n_channels
            device = next(self.parameters()).device
            
            # 重新创建 encoder
            self.encoder = PatchEncoder(
                patch_size=self.patch_size,
                n_channels=n_channels,
                d_model=self.d_model,
                dropout=self.dropout
            ).to(device)
            
            # 重新创建 decoder
            self.decoder = PatchDecoder(
                patch_size=self.patch_size,
                n_channels=n_channels,
                d_model=self.d_model,
                dropout=self.dropout
            ).to(device)
    
    def create_patches(self, x):
        """
        创建 overlapping patches
        
        Args:
            x: [B, T, C]
        Returns:
            patches: [B, num_patches, patch_size, C]
        """
        B, T, C = x.shape
        num_patches = (T - self.patch_size) // self.stride + 1
        
        x = x.permute(0, 2, 1)  # [B, C, T]
        patches = x.unfold(dimension=2, size=self.patch_size, step=self.stride)
        patches = patches.permute(0, 2, 3, 1)  # [B, num_patches, patch_size, C]
        
        return patches
    
    def encode(self, x):
        """
        编码输入序列
        
        Args:
            x: [B, T, C]
        Returns:
            indices: [B, num_patches]
            vq_loss: scalar
            z_q: [B, num_patches, d_model]
        """
        B, T, C = x.shape
        self._update_n_channels(C)
        
        # 1. 创建 patches
        patches = self.create_patches(x)  # [B, num_patches, patch_size, C]
        
        # 2. Patch Encoder
        z = self.encoder(patches)  # [B, num_patches, d_model]
        
        # 3. VQ 量化
        vq_loss, z_q, indices = self.vq(z)
        
        return indices, vq_loss, z_q
    
    def decode(self, z_q):
        """
        从量化向量解码为 patches
        
        Args:
            z_q: [B, num_patches, d_model]
        Returns:
            patches: [B, num_patches, patch_size, C]
        """
        return self.decoder(z_q)
    
    def patches_to_sequence(self, patches, target_len=None):
        """
        将 patches 合并为序列 (处理重叠区域用平均)
        
        Args:
            patches: [B, num_patches, patch_size, C]
            target_len: 目标序列长度
        Returns:
            x: [B, T, C]
        """
        B, num_patches, patch_size, C = patches.shape
        
        if self.stride == self.patch_size:
            # 无重叠，直接拼接
            x = patches.reshape(B, -1, C)
        else:
            # 有重叠，使用加权平均
            T = (num_patches - 1) * self.stride + patch_size
            x = torch.zeros(B, T, C, device=patches.device, dtype=patches.dtype)
            count = torch.zeros(B, T, C, device=patches.device, dtype=patches.dtype)
            
            for i in range(num_patches):
                start = i * self.stride
                x[:, start:start+patch_size, :] += patches[:, i, :, :]
                count[:, start:start+patch_size, :] += 1
            
            x = x / count.clamp(min=1)
        
        if target_len is not None:
            x = x[:, :target_len, :]
        
        return x
    
    def forward_pretrain(self, x):
        """
        预训练: Next Token Prediction + Reconstruction
        
        Args:
            x: [B, T, C]
        Returns:
            logits: [B, num_patches-1, codebook_size]
            targets: [B, num_patches-1]
            vq_loss: scalar
            recon_loss: scalar
        """
        B, T, C = x.shape
        
        # 编码
        indices, vq_loss, z_q = self.encode(x)
        num_patches = indices.shape[1]
        
        # 重构损失
        decoded_patches = self.decode(z_q)  # [B, num_patches, patch_size, C]
        x_recon = self.patches_to_sequence(decoded_patches, target_len=T)
        recon_loss = F.mse_loss(x_recon, x[:, :x_recon.shape[1], :])
        
        # Transformer for NTP
        h = self.transformer(z_q)  # [B, num_patches, d_model]
        logits = self.output_head(h)  # [B, num_patches, codebook_size]
        
        # NTP: 用位置 i 预测位置 i+1
        return logits[:, :-1], indices[:, 1:], vq_loss, recon_loss
    
    def forward_finetune(self, x, target_len):
        """
        微调: 预测未来序列
        
        Args:
            x: [B, T, C]
            target_len: 预测长度
        Returns:
            pred: [B, target_len, C]
            vq_loss: scalar
        """
        B, T, C = x.shape
        num_pred_patches = (target_len + self.stride - 1) // self.stride
        
        # 编码输入
        indices, vq_loss, z_q = self.encode(x)
        
        # 自回归预测
        current_z = z_q
        
        for _ in range(num_pred_patches):
            h = self.transformer(current_z)
            logits = self.output_head(h[:, -1, :])  # [B, codebook_size]
            
            # 预测下一个 token
            pred_idx = torch.argmax(logits, dim=-1)  # [B]
            pred_code = self.vq.get_embedding(pred_idx)  # [B, d_model]
            
            current_z = torch.cat([current_z, pred_code.unsqueeze(1)], dim=1)
        
        # 获取预测的码本向量
        pred_z = current_z[:, -num_pred_patches:, :]
        
        # 解码
        pred_patches = self.decode(pred_z)
        pred = self.patches_to_sequence(pred_patches, target_len=target_len)
        
        return pred, vq_loss
    
    def forward(self, x, target_len=None, mode='pretrain'):
        if mode == 'pretrain':
            return self.forward_pretrain(x)
        else:
            return self.forward_finetune(x, target_len)
    
    @torch.no_grad()
    def get_codebook_usage(self, x):
        """获取码本使用率"""
        indices, _, _ = self.encode(x)
        unique = torch.unique(indices.reshape(-1))
        return len(unique) / self.codebook_size, unique


# 为了向后兼容
PatchVQVAETransformer = PatchVQTransformer


# ============ 工具函数 ============

def get_model_config(args):
    """从命令行参数构建模型配置"""
    return {
        # Patch
        'patch_size': args.patch_size,
        'stride': getattr(args, 'stride', args.patch_size),
        'n_channels': getattr(args, 'n_channels', 7),
        
        # Model
        'd_model': args.d_model,
        'codebook_size': args.codebook_size,
        'commitment_cost': getattr(args, 'commitment_cost', 0.25),
        'use_ema': getattr(args, 'use_ema', True),  # 默认使用 EMA
        
        # Transformer
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
    }
