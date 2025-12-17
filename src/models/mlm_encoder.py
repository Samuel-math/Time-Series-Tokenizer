"""
MLM Encoder模型
仅用于MLM预训练阶段，生成embeddings用于聚类构建码本
后续不会再用到这个模型的transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """将 patch 映射到 d_model 维度"""
    def __init__(self, patch_size, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(patch_size, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: [B, num_patches, C, patch_size]
        output: [B, C, num_patches, d_model]
        """
        x = self.proj(x)  # [B, num_patches, C, d_model]
        x = x.permute(0, 2, 1, 3)  # [B, C, num_patches, d_model]
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: [B, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer编码器（双向，用于MLM）"""
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        x: [B, seq_len, d_model]
        output: [B, seq_len, d_model]
        """
        x = self.encoder(x)
        return self.norm(x)


class MLMHead(nn.Module):
    """Masked Language Model Head: d_model -> patch_size"""
    def __init__(self, d_model, patch_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_size)
    
    def forward(self, x):
        """
        x: [B, C, num_patches, d_model]
        output: [B, num_patches, C, patch_size]
        """
        x = x.permute(0, 2, 1, 3)  # [B, num_patches, C, d_model]
        x = self.linear(self.dropout(x))  # [B, num_patches, C, patch_size]
        return x


class MLMEncoder(nn.Module):
    """
    MLM Encoder模型
    仅用于MLM预训练，生成embeddings用于聚类
    """
    def __init__(self, config):
        super().__init__()
        
        # 配置
        self.patch_size = config.get('patch_size', 16)
        self.d_model = config.get('d_model', 128)
        self.n_heads = config.get('n_heads', 8)
        self.n_layers = config.get('n_layers', 4)
        self.d_ff = config.get('d_ff', 256)
        self.dropout = config.get('dropout', 0.1)
        self.mask_ratio = config.get('mask_ratio', 0.4)
        
        # Patch Embedding
        self.patch_embedding = PatchEmbedding(self.patch_size, self.d_model, self.dropout)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, dropout=self.dropout)
        
        # Transformer Encoder（双向，用于MLM）
        self.transformer = TransformerEncoder(
            self.d_model, self.n_heads, self.n_layers, self.d_ff, self.dropout
        )
        
        # MLM Head
        self.mlm_head = MLMHead(self.d_model, self.patch_size, self.dropout)
        
        # Mask token (可学习)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
    
    def create_patch(self, x):
        """
        将输入序列划分为 patches
        x: [B, T, C]
        output: [B, num_patches, C, patch_size]
        """
        B, T, C = x.shape
        num_patches = T // self.patch_size
        x = x[:, :num_patches * self.patch_size, :]  # 截断
        x = x.reshape(B, num_patches, self.patch_size, C)
        x = x.permute(0, 1, 3, 2)  # [B, num_patches, C, patch_size]
        return x
    
    def random_mask(self, x, mask_ratio=None):
        """
        随机 mask patches
        x: [B, C, num_patches, d_model]
        output: masked_x, mask (True = masked)
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        B, C, num_patches, d_model = x.shape
        
        # 每个 channel 独立 mask
        num_mask = int(num_patches * mask_ratio)
        
        # 生成 mask
        noise = torch.rand(B, C, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=2)
        mask = torch.zeros(B, C, num_patches, device=x.device, dtype=torch.bool)
        
        for b in range(B):
            for c in range(C):
                mask[b, c, ids_shuffle[b, c, :num_mask]] = True
        
        # 应用 mask
        masked_x = x.clone()
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, d_model)
        masked_x[mask_expanded] = self.mask_token.expand(B, C, num_patches, d_model)[mask_expanded]
        
        return masked_x, mask
    
    def forward(self, x, mask_ratio=None):
        """
        MLM前向传播
        x: [B, T, C] 原始序列
        output: recon [B, num_patches, C, patch_size], mask, target
        """
        # 创建 patches
        patches = self.create_patch(x)  # [B, num_patches, C, patch_size]
        target = patches.clone()
        
        B, num_patches, C, patch_size = patches.shape
        
        # Patch embedding
        embedded = self.patch_embedding(patches)  # [B, C, num_patches, d_model]
        
        # Random mask
        masked_embedded, mask = self.random_mask(embedded, mask_ratio)  # [B, C, num_patches, d_model]
        
        # 展平处理 (channel-independent)
        masked_embedded = masked_embedded.reshape(B * C, num_patches, self.d_model)
        
        # 位置编码
        masked_embedded = self.pos_encoding(masked_embedded)
        
        # Transformer Encoder (双向)
        output = self.transformer(masked_embedded)  # [B*C, num_patches, d_model]
        
        # Reshape back
        output = output.reshape(B, C, num_patches, self.d_model)
        
        # MLM Head
        recon = self.mlm_head(output)  # [B, num_patches, C, patch_size]
        
        return recon, mask, target
    
    def get_embeddings(self, x):
        """
        获取所有 patch 的 embedding (用于聚类)
        x: [B, T, C]
        output: [B * num_patches * C, d_model]
        """
        patches = self.create_patch(x)  # [B, num_patches, C, patch_size]
        B, num_patches, C, patch_size = patches.shape
        
        embedded = self.patch_embedding(patches)  # [B, C, num_patches, d_model]
        
        # 展平
        embeddings = embedded.permute(0, 2, 1, 3).reshape(-1, self.d_model)  # [B*num_patches*C, d_model]
        return embeddings


def compute_mlm_loss(recon, mask, target):
    """
    计算MLM的 reconstruction loss (只计算 masked 部分)
    recon: [B, num_patches, C, patch_size]
    mask: [B, C, num_patches] (True = masked)
    target: [B, num_patches, C, patch_size]
    """
    # 调整 mask 的维度
    mask = mask.permute(0, 2, 1)  # [B, num_patches, C]
    mask = mask.unsqueeze(-1).expand_as(recon)  # [B, num_patches, C, patch_size]
    
    # 只计算 masked 部分的 loss
    loss = F.mse_loss(recon[mask], target[mask])
    return loss
