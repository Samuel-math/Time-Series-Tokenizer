"""
Patch-based VQVAE + Transformer 模型架构 (优化版)

架构说明：
1. 输入: [B, T, C] 时间序列
2. Patch划分 + VQVAE Encoder -> 展平为 [B, num_patches, C, code_dim]
3. VQ 量化后的表示直接作为 Transformer 输入 (无需 token embedding)
4. Transformer (Decoder-only): 预测下一个码本向量
5. 预训练: NTP loss (预测码本索引)
6. 微调: 预测未来patch -> 解码 -> MSE loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .vqvae import Encoder, Decoder


class FlattenedVectorQuantizer(nn.Module):
    """
    展平的 Vector Quantizer
    码本维度 = embedding_dim * compressed_len
    """
    def __init__(self, codebook_size, code_dim, commitment_cost=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        
        # 码本: [codebook_size, code_dim]
        self.embedding = nn.Embedding(codebook_size, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
    
    def forward(self, z_flat):
        """
        Args:
            z_flat: [N, code_dim]
        Returns:
            loss, quantized, indices
        """
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices)
        
        e_latent_loss = F.mse_loss(quantized.detach(), z_flat)
        q_latent_loss = F.mse_loss(quantized, z_flat.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = z_flat + (quantized - z_flat).detach()
        
        return loss, quantized, indices
    
    def get_embedding(self, indices):
        return self.embedding(indices)


class CausalTransformer(nn.Module):
    """轻量级 Causal Transformer，输入维度为 code_dim"""
    def __init__(self, code_dim, n_heads, n_layers, d_ff, dropout=0.1, max_len=512):
        super().__init__()
        self.code_dim = code_dim
        
        # 位置编码，维度与 code_dim 一致
        self.pos_embedding = nn.Embedding(max_len, code_dim)
        self.drop = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=code_dim, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(code_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, code_dim] 直接是量化后的码本向量
        """
        B, T, _ = x.shape
        
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)
        x = self.drop(x)
        
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.norm(x)


class PatchVQVAETransformer(nn.Module):
    """
    Patch-based VQVAE + Transformer
    直接使用展平的码本向量作为 Transformer 输入
    """
    def __init__(self, config):
        super().__init__()
        
        # 配置
        self.patch_size = config.get('patch_size', 16)
        self.embedding_dim = config.get('embedding_dim', 32)
        self.compression_factor = config.get('compression_factor', 4)
        self.codebook_size = config.get('codebook_size', 256)
        self.n_layers = config.get('n_layers', 4)
        self.n_heads = config.get('n_heads', 4)
        self.d_ff = config.get('d_ff', 256)
        self.dropout = config.get('dropout', 0.1)
        self.commitment_cost = config.get('commitment_cost', 0.25)
        
        # VQVAE 配置
        self.num_hiddens = config.get('num_hiddens', 64)
        self.num_residual_layers = config.get('num_residual_layers', 2)
        self.num_residual_hiddens = config.get('num_residual_hiddens', 32)
        
        self.compressed_len = self.patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len  # Transformer 输入维度
        
        # VQVAE Encoder/Decoder
        self.encoder = Encoder(
            in_channels=1,
            num_hiddens=self.num_hiddens,
            num_residual_layers=self.num_residual_layers,
            num_residual_hiddens=self.num_residual_hiddens,
            embedding_dim=self.embedding_dim,
            compression_factor=self.compression_factor
        )
        
        self.decoder = Decoder(
            in_channels=self.embedding_dim,
            num_hiddens=self.num_hiddens,
            num_residual_layers=self.num_residual_layers,
            num_residual_hiddens=self.num_residual_hiddens,
            compression_factor=self.compression_factor
        )
        
        # VQ (码本维度 = code_dim)
        self.vq = FlattenedVectorQuantizer(
            self.codebook_size, self.code_dim, self.commitment_cost
        )
        
        # Transformer (输入维度 = code_dim，无需 token embedding)
        self.transformer = CausalTransformer(
            self.code_dim, self.n_heads, self.n_layers, 
            self.d_ff, self.dropout
        )
        
        # 输出头: code_dim -> codebook_size (预测码本索引)
        self.output_head = nn.Linear(self.code_dim, self.codebook_size)
    
    def encode_to_indices(self, x):
        """
        编码为码本索引和量化向量
        
        Args:
            x: [B, T, C]
        Returns:
            indices: [B, num_patches, C]
            vq_loss: scalar
            z_q: [B, num_patches, C, code_dim]
        """
        B, T, C = x.shape
        num_patches = T // self.patch_size
        
        # 重组为 patches 并批量编码
        x = x[:, :num_patches * self.patch_size, :]
        x = x.reshape(B, num_patches, self.patch_size, C)
        x = x.permute(0, 1, 3, 2).reshape(-1, self.patch_size)  # [B*num_patches*C, patch_size]
        
        # VQVAE Encoder
        z = self.encoder(x, self.compression_factor)  # [B*num_patches*C, embedding_dim, compressed_len]
        z_flat = z.reshape(z.shape[0], -1)  # [B*num_patches*C, code_dim]
        
        # VQ
        vq_loss, z_q_flat, indices = self.vq(z_flat)
        
        # Reshape
        indices = indices.reshape(B, num_patches, C)
        z_q = z_q_flat.reshape(B, num_patches, C, self.code_dim)
        
        return indices, vq_loss, z_q
    
    def decode_from_codes(self, z_q):
        """
        从量化向量解码
        
        Args:
            z_q: [B, num_patches, C, code_dim]
        Returns:
            x_recon: [B, num_patches * patch_size, C]
        """
        B, num_patches, C, _ = z_q.shape
        
        # Reshape for decoder
        z_q = z_q.reshape(-1, self.embedding_dim, self.compressed_len)
        
        # VQVAE Decoder
        x_recon = self.decoder(z_q, self.compression_factor)
        
        # Reshape back
        x_recon = x_recon.reshape(B, num_patches, C, self.patch_size)
        x_recon = x_recon.permute(0, 1, 3, 2).reshape(B, -1, C)
        
        return x_recon
    
    def forward_pretrain(self, x):
        """
        预训练: NTP
        直接用量化后的码本向量 z_q 作为 Transformer 输入
        """
        B, T, C = x.shape
        
        # 编码
        indices, vq_loss, z_q = self.encode_to_indices(x)  # z_q: [B, num_patches, C, code_dim]
        num_patches = indices.shape[1]
        
        # 重构损失
        x_recon = self.decode_from_codes(z_q)
        recon_loss = F.mse_loss(x_recon, x[:, :x_recon.shape[1], :])
        
        # Channel-independent Transformer
        # 直接使用 z_q 作为输入，不需要 token embedding
        all_logits = []
        for c in range(C):
            z_c = z_q[:, :, c, :]  # [B, num_patches, code_dim]
            h = self.transformer(z_c)  # [B, num_patches, code_dim]
            logits = self.output_head(h)  # [B, num_patches, codebook_size]
            all_logits.append(logits)
        
        logits = torch.stack(all_logits, dim=2)  # [B, num_patches, C, codebook_size]
        
        # NTP: 用位置i预测位置i+1
        return logits[:, :-1], indices[:, 1:], vq_loss, recon_loss
    
    def forward_finetune(self, x, target_len):
        """
        微调: 预测未来序列
        
        1. 编码输入为量化向量
        2. Transformer 自回归预测未来 patch 的码本索引
        3. 从码本获取向量并解码
        """
        B, T, C = x.shape
        num_pred_patches = (target_len + self.patch_size - 1) // self.patch_size
        
        # 编码输入
        indices, vq_loss, z_q = self.encode_to_indices(x)  # z_q: [B, num_patches, C, code_dim]
        
        # 自回归预测
        current_z = z_q  # [B, num_patches, C, code_dim]
        pred_codes_list = []
        
        for _ in range(num_pred_patches):
            next_codes = []
            for c in range(C):
                z_c = current_z[:, :, c, :]  # [B, seq_len, code_dim]
                h = self.transformer(z_c)  # [B, seq_len, code_dim]
                logits = self.output_head(h[:, -1, :])  # [B, codebook_size]
                
                # 使用 softmax + 加权求和 替代 argmax，保持可微分
                weights = F.softmax(logits, dim=-1)  # [B, codebook_size]
                codebook = self.vq.embedding.weight  # [codebook_size, code_dim]
                pred_code = torch.matmul(weights, codebook)  # [B, code_dim]
                next_codes.append(pred_code)
            
            next_codes = torch.stack(next_codes, dim=1)  # [B, C, code_dim]
            pred_codes_list.append(next_codes)
            
            # 更新序列
            current_z = torch.cat([current_z, next_codes.unsqueeze(1)], dim=1)
        
        # 获取预测的码本向量
        pred_codes = torch.stack(pred_codes_list, dim=1)  # [B, num_pred_patches, C, code_dim]
        
        # 解码
        pred = self.decode_from_codes(pred_codes)  # [B, num_pred_patches*patch_size, C]
        pred = pred[:, :target_len, :]
        
        return pred, vq_loss
    
    def forward(self, x, target_len=None, mode='pretrain'):
        if mode == 'pretrain':
            return self.forward_pretrain(x)
        else:
            return self.forward_finetune(x, target_len)
    
    @torch.no_grad()
    def get_codebook_usage(self, x):
        indices, _, _ = self.encode_to_indices(x)
        unique = torch.unique(indices.reshape(-1))
        return len(unique) / self.codebook_size, unique
    
    def load_vqvae_weights(self, checkpoint_path, device='cpu'):
        import os
        if not os.path.exists(checkpoint_path):
            print(f"警告: checkpoint不存在: {checkpoint_path}")
            return False
        
        try:
            vqvae_model = torch.load(checkpoint_path, map_location=device)
            if hasattr(vqvae_model, 'encoder'):
                self.encoder.load_state_dict(vqvae_model.encoder.state_dict())
                print("加载 Encoder 权重成功")
            if hasattr(vqvae_model, 'decoder'):
                self.decoder.load_state_dict(vqvae_model.decoder.state_dict())
                print("加载 Decoder 权重成功")
            return True
        except Exception as e:
            print(f"加载权重失败: {e}")
            return False


# ============ 工具函数 ============

def get_model_config(args):
    """构建模型配置"""
    # code_dim = embedding_dim * (patch_size / compression_factor)
    code_dim = args.embedding_dim * (args.patch_size // args.compression_factor)
    print(f"Transformer 输入维度 (code_dim) = {code_dim}")
    
    return {
        'patch_size': args.patch_size,
        'embedding_dim': args.embedding_dim,
        'compression_factor': args.compression_factor,
        'codebook_size': args.codebook_size,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'commitment_cost': args.commitment_cost,
        # VQVAE Encoder/Decoder 配置
        'num_hiddens': args.num_hiddens,
        'num_residual_layers': args.num_residual_layers,
        'num_residual_hiddens': args.num_residual_hiddens,
    }
