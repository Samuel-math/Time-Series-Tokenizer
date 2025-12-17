"""
MLM NTP模型
用于NTP预训练和微调
使用码本进行离散化，然后使用独立的Transformer进行NTP
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


class CausalTransformer(nn.Module):
    """因果Transformer编码器（用于NTP）"""
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
        使用因果掩码
        """
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.encoder(x, mask=causal_mask, is_causal=True)
        return self.norm(x)


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


class MLMNTPModel(nn.Module):
    """
    MLM NTP模型
    用于NTP预训练和微调
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
        
        # Patch Embedding（用于将输入转换为embedding，然后量化）
        self.patch_embedding = PatchEmbedding(self.patch_size, self.d_model, self.dropout)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, dropout=self.dropout)
        
        # 码本
        self.codebook = Codebook(self.codebook_size, self.d_model)
        
        # 因果Transformer（用于NTP）
        self.transformer = CausalTransformer(
            self.d_model, self.n_heads, self.n_layers, self.d_ff, self.dropout
        )
        
        # NTP Head
        self.ntp_head = NTPHead(self.d_model, self.codebook_size, self.dropout)
        
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
    
    def forward_ntp(self, x):
        """
        NTP前向传播
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
        
        # 因果Transformer
        output = self.transformer(quantized)  # [B*C, num_patches, d_model]
        
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
        微调: 预测未来序列（非自回归版本）
        
        1. 编码输入为量化向量
        2. 创建占位符位置（pred_len / patch_size个）
        3. Transformer 处理整个序列（输入 + 占位符），一次性预测所有未来patches
        4. 从码本获取向量并解码
        """
        patches = self.create_patch(x)  # [B, num_patches, C, patch_size]
        B, num_patches, C, patch_size = patches.shape
        num_pred_patches = (target_len + self.patch_size - 1) // self.patch_size
        
        # Patch embedding
        embedded = self.patch_embedding(patches)  # [B, C, num_patches, d_model]
        
        # 量化为码本索引（用于获取量化后的embedding）
        embedded_flat = embedded.permute(0, 2, 1, 3).reshape(-1, self.d_model)  # [B*num_patches*C, d_model]
        indices, _ = self.codebook.quantize(embedded_flat)  # [B*num_patches*C]
        indices = indices.reshape(B, num_patches, C).permute(0, 2, 1)  # [B, C, num_patches]
        
        # 获取量化后的 embedding
        quantized = self.codebook.get_embedding(indices)  # [B, C, num_patches, d_model]
        
        # Channel-independent 处理: [B, C, num_patches, d_model] -> [B*C, num_patches, d_model]
        quantized_flat = quantized.reshape(B * C, num_patches, self.d_model)
        
        # 创建占位符（零向量）: [B*C, num_pred_patches, d_model]
        placeholder = torch.zeros(B * C, num_pred_patches, self.d_model, device=quantized_flat.device, dtype=quantized_flat.dtype)
        
        # 拼接输入序列和占位符: [B*C, num_patches + num_pred_patches, d_model]
        full_sequence = torch.cat([quantized_flat, placeholder], dim=1)
        
        # 位置编码
        full_sequence = self.pos_encoding(full_sequence)
        
        # Transformer处理整个序列（causal mask确保占位符只能看到输入序列，不能看到其他占位符）
        h_full = self.transformer(full_sequence)  # [B*C, num_patches + num_pred_patches, d_model]
        
        # 只取占位符位置的输出: [B*C, num_pred_patches, d_model]
        h_pred = h_full[:, num_patches:, :]  # [B*C, num_pred_patches, d_model]
        
        # NTP Head: [B*C, num_pred_patches, codebook_size]
        logits = self.ntp_head(h_pred)  # [B*C, num_pred_patches, codebook_size]
        
        # 使用 softmax + 加权求和 替代 argmax，保持可微分
        weights = F.softmax(logits, dim=-1)  # [B*C, num_pred_patches, codebook_size]
        codebook = self.codebook.embedding.weight  # [codebook_size, d_model]
        pred_embeds_flat = torch.matmul(weights, codebook)  # [B*C, num_pred_patches, d_model]
        
        # 恢复形状: [B*C, num_pred_patches, d_model] -> [B, C, num_pred_patches, d_model]
        pred_embeds = pred_embeds_flat.reshape(B, C, num_pred_patches, self.d_model)
        
        # 解码为 patch
        pred_patches = self.pred_head(pred_embeds)  # [B, C, num_pred_patches, patch_size]
        
        # Reshape to sequence
        pred = pred_patches.permute(0, 2, 3, 1)  # [B, num_pred_patches, patch_size, C]
        pred = pred.reshape(B, -1, C)  # [B, num_pred_patches*patch_size, C]
        pred = pred[:, :target_len, :]  # [B, target_len, C]
        
        return pred


def compute_ntp_loss(logits, targets):
    """
    计算NTP的 loss
    logits: [B, C, num_patches-1, codebook_size]
    targets: [B, C, num_patches-1]
    """
    B, C, seq_len, codebook_size = logits.shape
    logits = logits.reshape(-1, codebook_size)
    targets = targets.reshape(-1)
    loss = F.cross_entropy(logits, targets)
    return loss
