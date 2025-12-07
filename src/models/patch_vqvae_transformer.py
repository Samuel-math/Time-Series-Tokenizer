"""
Patch-based VQVAE + Transformer 模型架构

架构说明：
1. 输入: [B, T, C] 时间序列
2. Patch划分: [B, num_patches, C, patch_size]，num_patches = T / patch_size
3. VQVAE Encoder (复用vqvae.py): 对每个patch编码得到 [B, num_patches, C, compressed_len, embedding_dim]
4. Flatten + VQ: 将每个patch映射到码本
   码本大小: [codebook_size, embedding_dim * compressed_len]
5. Transformer (Decoder-only): 使用causal attention进行next token prediction
6. 预训练: NTP loss (CrossEntropy)
7. 微调: 解码 + MSE loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# 复用 vqvae.py 中的组件
from .vqvae import Encoder, Decoder, VectorQuantizer


class PatchVectorQuantizer(nn.Module):
    """
    Patch-level Vector Quantizer
    码本大小: [codebook_size, embedding_dim * compressed_len]
    每个patch整体映射到一个码本向量
    """
    def __init__(self, codebook_size, embedding_dim, compressed_len, commitment_cost=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.compressed_len = compressed_len
        self.code_dim = embedding_dim * compressed_len  # 码本向量的维度
        self.commitment_cost = commitment_cost
        
        # 码本: [codebook_size, code_dim]
        self.codebook = nn.Embedding(codebook_size, self.code_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
    
    def forward(self, z):
        """
        Args:
            z: [B, embedding_dim, compressed_len] 编码后的patch表示 (来自vqvae Encoder)
        Returns:
            loss: VQ loss
            quantized: [B, embedding_dim, compressed_len] 量化后的表示
            indices: [B] 码本索引
        """
        B = z.shape[0]
        
        # Flatten: [B, code_dim]
        z_flat = z.reshape(B, -1)
        
        # 计算距离: [B, codebook_size]
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) +
            torch.sum(self.codebook.weight ** 2, dim=1) -
            2 * torch.matmul(z_flat, self.codebook.weight.t())
        )
        
        # 找到最近的码本向量
        indices = torch.argmin(distances, dim=1)  # [B]
        
        # 获取量化向量
        quantized_flat = self.codebook(indices)  # [B, code_dim]
        quantized = quantized_flat.reshape(B, self.embedding_dim, self.compressed_len)
        
        # 计算损失
        z_for_loss = z.reshape(B, -1)
        e_latent_loss = F.mse_loss(quantized_flat.detach(), z_for_loss)
        q_latent_loss = F.mse_loss(quantized_flat, z_for_loss.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        return loss, quantized, indices


class CausalSelfAttention(nn.Module):
    """Causal Self-Attention for Decoder-only Transformer"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, d_model]
            mask: [T, T] causal mask (optional)
        """
        B, T, _ = x.shape
        
        # QKV projection
        qkv = self.W_qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    """Transformer Block with Causal Self-Attention"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class PatchVQVAETransformer(nn.Module):
    """
    Patch-based VQVAE + Transformer 模型
    复用 vqvae.py 中的 Encoder 和 Decoder
    """
    def __init__(self, config):
        """
        Args:
            config: 配置字典，包含：
                - patch_size: patch大小
                - embedding_dim: VQVAE embedding维度
                - compression_factor: 压缩因子
                - codebook_size: 码本大小
                - d_model: Transformer隐藏维度
                - n_layers: Transformer层数
                - n_heads: 注意力头数
                - d_ff: FFN维度
                - dropout: dropout率
                - num_hiddens: VQVAE隐藏层维度
                - num_residual_layers: 残差层数
                - num_residual_hiddens: 残差隐藏层维度
                - commitment_cost: VQ commitment cost
        """
        super().__init__()
        
        # 配置参数
        self.patch_size = config.get('patch_size', 16)
        self.embedding_dim = config.get('embedding_dim', 64)
        self.compression_factor = config.get('compression_factor', 4)
        self.codebook_size = config.get('codebook_size', 512)
        self.d_model = config.get('d_model', 256)
        self.n_layers = config.get('n_layers', 6)
        self.n_heads = config.get('n_heads', 8)
        self.d_ff = config.get('d_ff', 1024)
        self.dropout = config.get('dropout', 0.1)
        self.commitment_cost = config.get('commitment_cost', 0.25)
        
        # VQVAE 配置
        self.num_hiddens = config.get('num_hiddens', 128)
        self.num_residual_layers = config.get('num_residual_layers', 2)
        self.num_residual_hiddens = config.get('num_residual_hiddens', 64)
        
        self.compressed_len = self.patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len
        
        # 复用 vqvae.py 中的 Encoder
        self.encoder = Encoder(
            in_channels=1,
            num_hiddens=self.num_hiddens,
            num_residual_layers=self.num_residual_layers,
            num_residual_hiddens=self.num_residual_hiddens,
            embedding_dim=self.embedding_dim,
            compression_factor=self.compression_factor
        )
        
        # 复用 vqvae.py 中的 Decoder
        self.decoder = Decoder(
            in_channels=self.embedding_dim,
            num_hiddens=self.num_hiddens,
            num_residual_layers=self.num_residual_layers,
            num_residual_hiddens=self.num_residual_hiddens,
            compression_factor=self.compression_factor
        )
        
        # Patch-level Vector Quantizer
        self.vq = PatchVectorQuantizer(
            self.codebook_size, self.embedding_dim,
            self.compressed_len, self.commitment_cost
        )
        
        # Token Embedding: 从码本索引到d_model
        self.token_embedding = nn.Embedding(self.codebook_size, self.d_model)
        
        # Positional Embedding
        self.max_seq_len = 512
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        
        # Transformer Blocks (Decoder-only)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # 输出层: 预测下一个token的codebook索引
        self.output_head = nn.Linear(self.d_model, self.codebook_size)
        
        # 预测头: 用于微调时预测未来序列 (延迟初始化)
        self.prediction_head = None
        
        self.norm = nn.LayerNorm(self.d_model)
        self.drop = nn.Dropout(self.dropout)
    
    def encode_patch(self, patch):
        """
        使用 vqvae Encoder 编码单个 patch
        
        Args:
            patch: [B, patch_size] 单个patch
        Returns:
            z: [B, embedding_dim, compressed_len]
        """
        # Encoder 期望输入 [B, patch_size]，内部会 reshape 为 [B, 1, patch_size]
        z = self.encoder(patch, self.compression_factor)  # [B, embedding_dim, compressed_len]
        return z
    
    def decode_patch(self, z_q):
        """
        使用 vqvae Decoder 解码
        
        Args:
            z_q: [B, embedding_dim, compressed_len]
        Returns:
            recon: [B, patch_size]
        """
        recon = self.decoder(z_q, self.compression_factor)  # [B, patch_size]
        return recon
    
    def encode_patches(self, x):
        """
        将输入序列编码为patch tokens
        
        Args:
            x: [B, T, C] 输入时间序列
        Returns:
            indices: [B, num_patches, C] 码本索引
            vq_loss: VQ损失
            z_q: [B, num_patches, C, embedding_dim, compressed_len] 量化后的表示
        """
        B, T, C = x.shape
        num_patches = T // self.patch_size
        
        # 划分patches: [B, num_patches, C, patch_size]
        x = x[:, :num_patches * self.patch_size, :]
        x = x.reshape(B, num_patches, self.patch_size, C).permute(0, 1, 3, 2)
        
        # 对每个patch-channel进行编码和量化
        all_indices = []
        all_z_q = []
        total_vq_loss = 0
        
        for p in range(num_patches):
            patch_indices = []
            patch_z_q = []
            for c in range(C):
                patch = x[:, p, c, :]  # [B, patch_size]
                
                # 使用 vqvae Encoder
                z = self.encode_patch(patch)  # [B, embedding_dim, compressed_len]
                
                # 使用 Patch-level VQ
                vq_loss, z_q, indices = self.vq(z)
                total_vq_loss += vq_loss
                
                patch_indices.append(indices)
                patch_z_q.append(z_q)
            
            all_indices.append(torch.stack(patch_indices, dim=1))  # [B, C]
            all_z_q.append(torch.stack(patch_z_q, dim=1))  # [B, C, embedding_dim, compressed_len]
        
        indices = torch.stack(all_indices, dim=1)  # [B, num_patches, C]
        z_q = torch.stack(all_z_q, dim=1)  # [B, num_patches, C, embedding_dim, compressed_len]
        avg_vq_loss = total_vq_loss / (num_patches * C)
        
        return indices, avg_vq_loss, z_q
    
    def decode_patches(self, z_q):
        """
        将量化后的表示解码回原始序列
        
        Args:
            z_q: [B, num_patches, C, embedding_dim, compressed_len]
        Returns:
            x_recon: [B, T, C]
        """
        B, num_patches, C, embedding_dim, compressed_len = z_q.shape
        
        all_patches = []
        for p in range(num_patches):
            channel_patches = []
            for c in range(C):
                z = z_q[:, p, c, :, :]  # [B, embedding_dim, compressed_len]
                patch_recon = self.decode_patch(z)  # [B, patch_size]
                channel_patches.append(patch_recon)
            all_patches.append(torch.stack(channel_patches, dim=2))  # [B, patch_size, C]
        
        x_recon = torch.cat(all_patches, dim=1)  # [B, T, C]
        return x_recon
    
    def forward_pretrain(self, x):
        """
        预训练前向传播: Next Token Prediction
        
        Args:
            x: [B, T, C] 输入时间序列
        Returns:
            logits: [B, num_patches-1, C, codebook_size] 预测的codebook分布
            targets: [B, num_patches-1, C] 目标codebook索引
            vq_loss: VQ损失
            recon_loss: 重构损失
        """
        B, T, C = x.shape
        
        # 编码为patch tokens
        indices, vq_loss, z_q = self.encode_patches(x)
        num_patches = indices.shape[1]
        
        # 计算重构损失
        x_recon = self.decode_patches(z_q)
        recon_loss = F.mse_loss(x_recon, x[:, :x_recon.shape[1], :])
        
        # 对每个channel独立处理 (channel-independent)
        all_logits = []
        for c in range(C):
            channel_indices = indices[:, :, c]  # [B, num_patches]
            
            # Token + Position Embedding
            token_emb = self.token_embedding(channel_indices)
            positions = torch.arange(num_patches, device=x.device).unsqueeze(0).expand(B, -1)
            pos_emb = self.pos_embedding(positions)
            
            h = self.drop(token_emb + pos_emb)
            
            # Transformer blocks
            for block in self.transformer_blocks:
                h = block(h)
            
            h = self.norm(h)
            logits = self.output_head(h)  # [B, num_patches, codebook_size]
            all_logits.append(logits)
        
        logits = torch.stack(all_logits, dim=2)  # [B, num_patches, C, codebook_size]
        
        # NTP: 用位置i预测位置i+1的token
        logits = logits[:, :-1, :, :]  # [B, num_patches-1, C, codebook_size]
        targets = indices[:, 1:, :]    # [B, num_patches-1, C]
        
        return logits, targets, vq_loss, recon_loss
    
    def forward_finetune(self, x, target_len):
        """
        微调前向传播: 预测未来序列
        
        Args:
            x: [B, T, C] 输入时间序列
            target_len: 预测长度
        Returns:
            pred: [B, target_len, C] 预测结果
            vq_loss: VQ损失
        """
        B, T, C = x.shape
        
        # 编码为patch tokens
        indices, vq_loss, z_q = self.encode_patches(x)
        num_patches = indices.shape[1]
        
        # 对每个channel独立处理
        all_features = []
        for c in range(C):
            channel_indices = indices[:, :, c]
            
            # Token + Position Embedding
            token_emb = self.token_embedding(channel_indices)
            positions = torch.arange(num_patches, device=x.device).unsqueeze(0).expand(B, -1)
            pos_emb = self.pos_embedding(positions)
            
            h = self.drop(token_emb + pos_emb)
            
            for block in self.transformer_blocks:
                h = block(h)
            
            h = self.norm(h)
            all_features.append(h[:, -1, :])  # [B, d_model]
        
        features = torch.stack(all_features, dim=1)  # [B, C, d_model]
        
        # 延迟初始化预测头
        if self.prediction_head is None:
            self.prediction_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, target_len)
            ).to(x.device)
        
        # 预测每个channel
        preds = [self.prediction_head(features[:, c, :]) for c in range(C)]
        pred = torch.stack(preds, dim=2)  # [B, target_len, C]
        
        return pred, vq_loss
    
    def forward(self, x, target_len=None, mode='pretrain'):
        """统一的前向传播接口"""
        if mode == 'pretrain':
            return self.forward_pretrain(x)
        else:
            return self.forward_finetune(x, target_len)
    
    @torch.no_grad()
    def get_codebook_usage(self, x):
        """统计码本使用情况"""
        indices, _, _ = self.encode_patches(x)
        indices = indices.reshape(-1)
        unique_indices = torch.unique(indices)
        usage = len(unique_indices) / self.codebook_size
        return usage, unique_indices
    
    def load_vqvae_weights(self, vqvae_checkpoint_path, device='cpu'):
        """
        从预训练的 VQVAE 模型加载 Encoder 和 Decoder 权重
        
        Args:
            vqvae_checkpoint_path: vqvae_pretrain.py 保存的模型路径
            device: 设备
        """
        import os
        if not os.path.exists(vqvae_checkpoint_path):
            print(f"警告: VQVAE checkpoint不存在: {vqvae_checkpoint_path}")
            return False
        
        try:
            vqvae_model = torch.load(vqvae_checkpoint_path, map_location=device)
            
            # 加载 Encoder 权重
            if hasattr(vqvae_model, 'encoder'):
                self.encoder.load_state_dict(vqvae_model.encoder.state_dict())
                print("成功加载 VQVAE Encoder 权重")
            
            # 加载 Decoder 权重
            if hasattr(vqvae_model, 'decoder'):
                self.decoder.load_state_dict(vqvae_model.decoder.state_dict())
                print("成功加载 VQVAE Decoder 权重")
            
            return True
        except Exception as e:
            print(f"加载 VQVAE 权重失败: {e}")
            return False
