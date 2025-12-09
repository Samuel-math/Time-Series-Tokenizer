"""
Patch-based VQ + Transformer 模型架构 (v2)

架构说明：
1. 输入: [B, T, C] 时间序列
2. Overlapping Patch划分 (stride 控制重叠) -> [B, num_patches, patch_size, C]
3. Intra-Patch Attention (Encoder): learnable query cross-attention -> [B, num_patches, d_model]
4. VQ 量化到码本 -> [B, num_patches] indices
5. Decoder-only Transformer: 预测下一个码本索引
6. Intra-Patch Attention (Decoder): d_model -> [B, num_patches, patch_size, C]
7. 预训练: NTP loss + Reconstruction loss
8. 微调: 预测未来 patch -> MSE loss

参考: https://arxiv.org/pdf/2402.05956 (Intra-Patch Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class IntraPatchEncoder(nn.Module):
    """
    Intra-Patch Attention Encoder
    
    使用可学习的 query 与 patch 元素做 cross-attention，将 patch 编码为 d_model 维向量
    
    输入: [B, num_patches, patch_size, C]
    输出: [B, num_patches, d_model]
    """
    def __init__(self, patch_size, n_channels, d_model, n_heads=2, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        
        # 可学习的 Query: [1, d_model]
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # K, V 投影: 将 patch 中每个时间步投影到 d_model
        self.k_proj = nn.Linear(n_channels, d_model)
        self.v_proj = nn.Linear(n_channels, d_model)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, num_patches, patch_size, C]
        Returns:
            out: [B, num_patches, d_model]
        """
        B, num_patches, patch_size, C = x.shape
        N = B * num_patches
        
        # 重组为 [B * num_patches, patch_size, C]
        x = x.reshape(N, patch_size, C)
        
        # Query: [N, 1, d_model]
        q = self.query.expand(N, -1, -1)
        
        # Key, Value: [N, patch_size, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Multi-Head Cross-Attention
        q = q.view(N, 1, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(N, patch_size, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(N, patch_size, self.n_heads, self.head_size).transpose(1, 2)
        
        # Attention: [N, n_heads, 1, patch_size]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Output: [N, n_heads, 1, head_size] -> [N, 1, d_model]
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(N, 1, self.d_model)
        out = self.out_proj(out)
        
        # Residual + LayerNorm
        out = self.norm1(self.query.expand(N, -1, -1) + self.dropout(out))
        
        # FFN
        residual = out
        out = self.ffn(out)
        out = self.norm2(residual + out)
        
        # [N, 1, d_model] -> [B, num_patches, d_model]
        out = out.squeeze(1).view(B, num_patches, self.d_model)
        
        return out


class IntraPatchDecoder(nn.Module):
    """
    Intra-Patch Attention Decoder
    
    使用可学习的 queries (patch_size 个) 与编码向量做 cross-attention，解码为 patch
    
    输入: [B, num_patches, d_model]
    输出: [B, num_patches, patch_size, C]
    """
    def __init__(self, patch_size, n_channels, d_model, n_heads=2, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        
        # 可学习的 Queries: [patch_size, d_model]
        # 每个 query 负责生成一个时间步的输出
        self.queries = nn.Parameter(torch.randn(1, patch_size, d_model) * 0.02)
        
        # K, V 投影: 对编码向量做投影
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 最终输出层: d_model -> n_channels
        self.output_layer = nn.Linear(d_model, n_channels)
    
    def forward(self, z):
        """
        Args:
            z: [B, num_patches, d_model]
        Returns:
            out: [B, num_patches, patch_size, C]
        """
        B, num_patches, _ = z.shape
        N = B * num_patches
        
        # 重组为 [N, 1, d_model] (每个 patch 的编码)
        z = z.reshape(N, 1, self.d_model)
        
        # Queries: [N, patch_size, d_model]
        q = self.queries.expand(N, -1, -1)
        
        # Key, Value from encoded vector: [N, 1, d_model]
        k = self.k_proj(z)
        v = self.v_proj(z)
        
        # Multi-Head Cross-Attention
        q = q.view(N, self.patch_size, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(N, 1, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(N, 1, self.n_heads, self.head_size).transpose(1, 2)
        
        # Attention: [N, n_heads, patch_size, 1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Output: [N, n_heads, patch_size, head_size] -> [N, patch_size, d_model]
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(N, self.patch_size, self.d_model)
        out = self.out_proj(out)
        
        # Residual + LayerNorm
        out = self.norm1(self.queries.expand(N, -1, -1) + self.dropout(out))
        
        # FFN
        residual = out
        out = self.ffn(out)
        out = self.norm2(residual + out)
        
        # 输出层: [N, patch_size, d_model] -> [N, patch_size, C]
        out = self.output_layer(out)
        
        # [N, patch_size, C] -> [B, num_patches, patch_size, C]
        out = out.view(B, num_patches, self.patch_size, self.n_channels)
        
        return out


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with Straight-Through Estimator
    """
    def __init__(self, codebook_size, d_model, commitment_cost=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.commitment_cost = commitment_cost
        
        # 码本: [codebook_size, d_model]
        self.embedding = nn.Embedding(codebook_size, d_model)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
    
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
        quantized = self.embedding(indices)
        
        # VQ Loss
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
    Patch-based VQ + Transformer (v2)
    
    架构:
    - IntraPatchEncoder: [B, num_patches, patch_size, C] -> [B, num_patches, d_model]
    - VQ: [B, num_patches, d_model] -> indices + z_q
    - Transformer: NTP 预训练
    - IntraPatchDecoder: [B, num_patches, d_model] -> [B, num_patches, patch_size, C]
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
        
        # ========== Transformer 配置 ==========
        self.n_layers = config.get('n_layers', 4)
        self.n_heads = config.get('n_heads', 8)
        self.d_ff = config.get('d_ff', 512)
        self.dropout = config.get('dropout', 0.1)
        
        # ========== Intra-Patch Attention 配置 ==========
        self.intra_n_heads = config.get('intra_n_heads', 2)
        
        # ========== 模块 ==========
        
        # 1. Intra-Patch Encoder
        self.encoder = IntraPatchEncoder(
            patch_size=self.patch_size,
            n_channels=self.n_channels,
            d_model=self.d_model,
            n_heads=self.intra_n_heads,
            dropout=self.dropout
        )
        
        # 2. Vector Quantizer
        self.vq = VectorQuantizer(
            codebook_size=self.codebook_size,
            d_model=self.d_model,
            commitment_cost=self.commitment_cost
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
        
        # 5. Intra-Patch Decoder
        self.decoder = IntraPatchDecoder(
            patch_size=self.patch_size,
            n_channels=self.n_channels,
            d_model=self.d_model,
            n_heads=self.intra_n_heads,
            dropout=self.dropout
        )
    
    def _update_n_channels(self, n_channels):
        """动态更新通道数"""
        if n_channels != self.n_channels:
            self.n_channels = n_channels
            device = next(self.parameters()).device
            
            # 重新创建 encoder
            self.encoder = IntraPatchEncoder(
                patch_size=self.patch_size,
                n_channels=n_channels,
                d_model=self.d_model,
                n_heads=self.intra_n_heads,
                dropout=self.dropout
            ).to(device)
            
            # 重新创建 decoder
            self.decoder = IntraPatchDecoder(
                patch_size=self.patch_size,
                n_channels=n_channels,
                d_model=self.d_model,
                n_heads=self.intra_n_heads,
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
        
        # 2. Intra-Patch Encoder
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
        
        # Transformer
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        
        # Intra-Patch Attention
        'intra_n_heads': getattr(args, 'intra_n_heads', 2),
    }
