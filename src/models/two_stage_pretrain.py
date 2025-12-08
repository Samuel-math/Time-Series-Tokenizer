"""
两阶段预训练模型 (共享 Transformer 参数)

阶段1: Masked Reconstruction (类似 PatchTST)
- 输入: patched_X (B, num_patches, C, patch_size)
- Embedding: Linear projection → (B, C, num_patches, d_model)
- 共享 Transformer backbone (无 causal mask, 双向)
- MLM Head: 重建 masked patches
- Loss: Reconstruction loss (MSE)

中间步骤: 码本训练
- 使用阶段1的 embedding layer
- 对所有 patch 的 embedding 进行聚类
- 聚类中心作为码本

阶段2: Next Token Prediction
- 使用阶段1的 embedding layer + 中间步骤的 codebook
- 离散化 embedding 到 codebook indices
- 共享 Transformer backbone (有 causal mask, 因果)
- NTP Head: 预测下一个 codebook index
- Loss: Cross entropy

关键: 阶段1和阶段2共享同一个 Transformer，仅有无 causal mask 的区别

微调: 预测未来序列 → MSE loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


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


class SharedTransformer(nn.Module):
    """
    共享的 Transformer 模块
    - 阶段1 (MLM): 使用双向 attention (causal=False)
    - 阶段2 (NTP): 使用因果 attention (causal=True)
    """
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, causal=False):
        """
        x: [B, seq_len, d_model]
        causal: 是否使用因果掩码
        """
        if causal:
            seq_len = x.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            x = self.encoder(x, mask=causal_mask, is_causal=True)
        else:
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


class NTPHead(nn.Module):
    """Next Token Prediction Head: d_model -> codebook_size"""
    def __init__(self, d_model, codebook_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, codebook_size)
    
    def forward(self, x):
        """
        x: [B, seq_len, d_model]
        output: [B, seq_len, codebook_size]
        """
        return self.linear(self.dropout(x))


class Codebook(nn.Module):
    """码本模块: 存储聚类中心"""
    def __init__(self, codebook_size, d_model):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.embedding = nn.Embedding(codebook_size, d_model)
    
    def init_from_centroids(self, centroids):
        """从聚类中心初始化码本"""
        assert centroids.shape == (self.codebook_size, self.d_model)
        self.embedding.weight.data.copy_(centroids)
    
    def quantize(self, z):
        """
        将连续 embedding 量化为最近的码本索引
        z: [N, d_model]
        output: indices [N], quantized [N, d_model]
        """
        # 计算距离
        distances = (
            torch.sum(z ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.matmul(z, self.embedding.weight.t())
        )
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices)
        return indices, quantized
    
    def get_embedding(self, indices):
        return self.embedding(indices)


class TwoStagePretrainModel(nn.Module):
    """
    两阶段预训练模型 (共享 Transformer 参数)
    
    阶段1 (MLM): 使用共享 Transformer, causal=False (双向)
    阶段2 (NTP): 使用共享 Transformer, causal=True (因果)
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
        self.codebook_size = config.get('codebook_size', 256)
        self.mask_ratio = config.get('mask_ratio', 0.4)
        
        # Patch Embedding (阶段1和阶段2共享)
        self.patch_embedding = PatchEmbedding(self.patch_size, self.d_model, self.dropout)
        
        # 位置编码 (共享)
        self.pos_encoding = PositionalEncoding(self.d_model, dropout=self.dropout)
        
        # 共享 Transformer (阶段1和阶段2共享，仅通过 causal 参数控制)
        self.transformer = SharedTransformer(
            self.d_model, self.n_heads, self.n_layers, self.d_ff, self.dropout
        )
        
        # 阶段1: MLM Head
        self.mlm_head = MLMHead(self.d_model, self.patch_size, self.dropout)
        
        # Mask token (可学习)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        
        # 阶段2: NTP Head
        self.ntp_head = NTPHead(self.d_model, self.codebook_size, self.dropout)
        
        # 码本 (在中间步骤初始化)
        self.codebook = Codebook(self.codebook_size, self.d_model)
        
        # 预测 Head (微调用)
        self.pred_head = nn.Linear(self.d_model, self.patch_size)
    
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
    
    def forward_stage1(self, x, mask_ratio=None):
        """
        阶段1: Masked Reconstruction
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
        
        # 共享 Transformer (双向, causal=False)
        output = self.transformer(masked_embedded, causal=False)  # [B*C, num_patches, d_model]
        
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
    
    def forward_stage2(self, x):
        """
        阶段2: Next Token Prediction
        x: [B, T, C] 原始序列
        output: logits [B, C, num_patches-1, codebook_size], targets [B, C, num_patches-1]
        """
        patches = self.create_patch(x)  # [B, num_patches, C, patch_size]
        B, num_patches, C, patch_size = patches.shape
        
        # Patch embedding
        embedded = self.patch_embedding(patches)  # [B, C, num_patches, d_model]
        
        # 量化为码本索引
        embedded_flat = embedded.permute(0, 2, 1, 3).reshape(-1, self.d_model)  # [B*num_patches*C, d_model]
        indices, _ = self.codebook.quantize(embedded_flat)  # [B*num_patches*C]
        indices = indices.reshape(B, num_patches, C).permute(0, 2, 1)  # [B, C, num_patches]
        
        # 获取量化后的 embedding
        quantized = self.codebook.get_embedding(indices)  # [B, C, num_patches, d_model]
        
        # Channel-independent 处理
        quantized = quantized.reshape(B * C, num_patches, self.d_model)
        
        # 位置编码
        quantized = self.pos_encoding(quantized)
        
        # 共享 Transformer (因果, causal=True)
        output = self.transformer(quantized, causal=True)  # [B*C, num_patches, d_model]
        
        # NTP Head (预测下一个 token)
        logits = self.ntp_head(output)  # [B*C, num_patches, codebook_size]
        logits = logits.reshape(B, C, num_patches, self.codebook_size)
        
        # 目标是下一个 token 的索引
        # logits[:, :, :-1] 预测 indices[:, :, 1:]
        pred_logits = logits[:, :, :-1, :]  # [B, C, num_patches-1, codebook_size]
        targets = indices[:, :, 1:]  # [B, C, num_patches-1]
        
        return pred_logits, targets
    
    def forward_finetune(self, x, target_len):
        """
        微调: 预测未来序列
        x: [B, T, C] 输入序列
        target_len: 预测长度
        output: pred [B, target_len, C]
        """
        patches = self.create_patch(x)  # [B, num_patches, C, patch_size]
        B, num_patches, C, patch_size = patches.shape
        num_pred_patches = (target_len + self.patch_size - 1) // self.patch_size
        
        # Patch embedding
        embedded = self.patch_embedding(patches)  # [B, C, num_patches, d_model]
        
        # 自回归预测
        pred_patches_list = []
        current_embed = embedded  # [B, C, num_patches, d_model]
        
        for _ in range(num_pred_patches):
            pred_per_channel = []
            for c in range(C):
                z_c = current_embed[:, c, :, :]  # [B, seq_len, d_model]
                z_c = self.pos_encoding(z_c)
                # 共享 Transformer (因果, causal=True)
                h = self.transformer(z_c, causal=True)  # [B, seq_len, d_model]
                logits = self.ntp_head(h[:, -1, :])  # [B, codebook_size]
                
                # Soft attention to codebook (可微分)
                weights = F.softmax(logits, dim=-1)  # [B, codebook_size]
                pred_embed = torch.matmul(weights, self.codebook.embedding.weight)  # [B, d_model]
                pred_per_channel.append(pred_embed)
            
            # [B, C, d_model]
            pred_embed_all = torch.stack(pred_per_channel, dim=1)
            pred_patches_list.append(pred_embed_all)
            
            # 更新序列
            current_embed = torch.cat([current_embed, pred_embed_all.unsqueeze(2)], dim=2)
        
        # [B, num_pred_patches, C, d_model]
        pred_embeds = torch.stack(pred_patches_list, dim=1)
        
        # 解码为 patch
        pred_embeds = pred_embeds.permute(0, 2, 1, 3)  # [B, C, num_pred_patches, d_model]
        pred_patches = self.pred_head(pred_embeds)  # [B, C, num_pred_patches, patch_size]
        
        # Reshape to sequence
        pred = pred_patches.permute(0, 2, 3, 1)  # [B, num_pred_patches, patch_size, C]
        pred = pred.reshape(B, -1, C)  # [B, num_pred_patches*patch_size, C]
        pred = pred[:, :target_len, :]  # [B, target_len, C]
        
        return pred
    
    def init_codebook_from_data(self, embeddings, method='kmeans'):
        """
        从数据初始化码本
        embeddings: [N, d_model] 所有 patch 的 embedding
        method: 'kmeans' 或 'random'
        """
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            embeddings_np = embeddings.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=self.codebook_size, random_state=42, n_init=10)
            kmeans.fit(embeddings_np)
            centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            self.codebook.init_from_centroids(centroids.to(embeddings.device))
            print(f"码本已从 KMeans 聚类初始化 (codebook_size={self.codebook_size})")
        elif method == 'random':
            # 随机选择 N 个样本作为初始码本
            perm = torch.randperm(embeddings.size(0))[:self.codebook_size]
            centroids = embeddings[perm]
            self.codebook.init_from_centroids(centroids)
            print(f"码本已从随机样本初始化 (codebook_size={self.codebook_size})")
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_codebook_usage(self, x):
        """获取码本利用率"""
        patches = self.create_patch(x)
        B, num_patches, C, patch_size = patches.shape
        
        embedded = self.patch_embedding(patches)
        embedded_flat = embedded.permute(0, 2, 1, 3).reshape(-1, self.d_model)
        indices, _ = self.codebook.quantize(embedded_flat)
        
        unique = torch.unique(indices)
        return len(unique) / self.codebook_size, unique


def compute_stage1_loss(recon, mask, target):
    """
    计算阶段1的 reconstruction loss (只计算 masked 部分)
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


def compute_stage2_loss(logits, targets):
    """
    计算阶段2的 NTP loss
    logits: [B, C, num_patches-1, codebook_size]
    targets: [B, C, num_patches-1]
    """
    B, C, seq_len, codebook_size = logits.shape
    logits = logits.reshape(-1, codebook_size)
    targets = targets.reshape(-1)
    loss = F.cross_entropy(logits, targets)
    return loss
