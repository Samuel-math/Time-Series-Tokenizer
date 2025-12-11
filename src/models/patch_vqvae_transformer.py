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


class FlattenedVectorQuantizerEMA(nn.Module):
    """
    使用 EMA 更新码本的 Vector Quantizer
    码本维度 = embedding_dim * compressed_len
    """
    def __init__(self, codebook_size, code_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        
        # 码本权重与EMA状态
        embed = torch.randn(codebook_size, code_dim)
        embed.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.embedding = nn.Embedding(codebook_size, code_dim)
        self.embedding.weight.data.copy_(embed)
        self.embedding.weight.requires_grad = False
        
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_w', embed.clone())
    
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
        
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.codebook_size).type(z_flat.dtype)
                
                # EMA 累积
                self.ema_cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
                dw = torch.matmul(one_hot.t(), z_flat)
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                
                # 归一化避免小簇消失
                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
                embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
                self.embedding.weight.data.copy_(embed_normalized)
        
        # 只有commitment项
        e_latent_loss = F.mse_loss(z_flat, quantized.detach())
        loss = self.commitment_cost * e_latent_loss
        
        quantized = z_flat + (quantized - z_flat).detach()
        return loss, quantized, indices
    
    def get_embedding(self, indices):
        return self.embedding(indices)


class PatchTCN(nn.Module):
    """
    使用TCN（Temporal Convolutional Network）处理patch内时间信息
    输入: [B*num_patches, patch_size, C]
    输出: [B*num_patches, patch_size, C]
    使用因果卷积和膨胀卷积捕捉时序依赖
    """
    def __init__(self, patch_size, n_channels, dropout=0.1, num_layers=2, kernel_size=3, hidden_dim=None):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim or n_channels
        
        # TCN层：多层因果卷积，每层dilation递增
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i  # dilation: 1, 2, 4, 8, ...
            in_channels = n_channels if i == 0 else self.hidden_dim
            out_channels = self.hidden_dim
            
            # 因果卷积：padding = (kernel_size - 1) * dilation
            # 这样确保输出长度 = 输入长度（不考虑padding）
            padding = (kernel_size - 1) * dilation
            
            tcn_block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.tcn_layers.append(tcn_block)
        
        # 输出投影层：hidden_dim -> n_channels
        if self.hidden_dim != n_channels:
            self.output_proj = nn.Conv1d(self.hidden_dim, n_channels, kernel_size=1)
        else:
            self.output_proj = None
    
    def forward(self, x):
        """
        Args:
            x: [B*num_patches, patch_size, C]
        Returns:
            out: [B*num_patches, patch_size, C]
        """
        B, T, C = x.shape  # B = B*num_patches, T = patch_size
        
        # 保存原始输入用于残差连接
        residual = x
        
        # Conv1d需要 [B, C, T] 格式
        x = x.permute(0, 2, 1)  # [B, C, patch_size]
        
        # 通过TCN层
        for i, tcn_layer in enumerate(self.tcn_layers):
            x_out = tcn_layer(x)  # [B, hidden_dim, patch_size + padding]
            
            # 裁剪padding，保持输出长度与输入相同
            # 由于使用了因果padding，输出长度可能略大于输入，需要裁剪
            x_out = x_out[:, :, :T]  # [B, hidden_dim, patch_size]
            
            # 残差连接（如果维度匹配）
            if x.shape[1] == x_out.shape[1] and x.shape[2] == x_out.shape[2]:
                x = x + x_out
            else:
                x = x_out
        
        # 输出投影
        if self.output_proj is not None:
            x = self.output_proj(x)  # [B, C, patch_size]
        
        # 转回 [B, patch_size, C]
        x = x.permute(0, 2, 1)  # [B, patch_size, C]
        
        # 整体残差连接
        out = residual + x
        
        return out


class PatchSelfAttention(nn.Module):
    """
    处理patch内时间信息的Self-Attention层
    对所有通道一起在patch_size个时间步之间应用self-attention
    输入: [B*num_patches, patch_size, C]
    输出: [B*num_patches, patch_size, C]
    在时间步之间做attention，同时可以看到所有通道的信息，捕捉时序和通道间交互
    """
    def __init__(self, patch_size, n_channels, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        
        # 位置编码（patch内的时间位置）
        self.pos_embedding = nn.Embedding(patch_size, n_channels)
        
        # 使用nn.MultiheadAttention，embed_dim = n_channels
        # 自动调整num_heads以确保embed_dim能被num_heads整除
        num_heads = 1
        if n_channels >= 4:
            # 尝试找到合适的num_heads（不超过4，且能整除n_channels）
            for h in [4, 2, 1]:
                if n_channels % h == 0:
                    num_heads = h
                    break
        
        self.attention = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.norm1 = nn.LayerNorm(n_channels)
        self.norm2 = nn.LayerNorm(n_channels)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(n_channels, n_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_channels * 2, n_channels),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B*num_patches, patch_size, C]
        Returns:
            out: [B*num_patches, patch_size, C]
        """
        B, T, C = x.shape  # B = B*num_patches, T = patch_size
        
        # 保存原始输入用于残差连接
        residual = x
        
        # 位置编码
        positions = torch.arange(self.patch_size, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)  # [B, patch_size, C]
        
        # 添加位置编码
        x_pos = x + pos_emb
        
        # Self-attention（在时间步之间，对所有通道一起）
        attn_out, _ = self.attention(x_pos, x_pos, x_pos)  # [B, patch_size, C]
        
        # 残差连接和Layer Norm
        x_normed = self.norm1(x + self.dropout(attn_out))  # [B, patch_size, C]
        
        # FFN（带残差连接）
        ffn_out = self.ffn(x_normed)  # [B, patch_size, C]
        x_out = self.norm2(x_normed + ffn_out)  # [B, patch_size, C]
        
        # 整体残差连接：输入 + 处理后的输出
        out = residual + x_out
        
        return out


class PatchCrossAttention(nn.Module):
    """
    处理patch内时间信息的Cross-Attention层
    使用可学习的query，key和value来自输入
    输入: [B*num_patches, patch_size, C]
    输出: [B*num_patches, patch_size, C]
    使用可学习的query tokens来查询输入序列，捕捉时序信息
    """
    def __init__(self, patch_size, n_channels, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        
        # 可学习的query tokens: [patch_size, n_channels]
        self.learnable_query = nn.Parameter(torch.randn(patch_size, n_channels) * 0.02)
        
        # 位置编码（patch内的时间位置，用于key/value）
        self.pos_embedding = nn.Embedding(patch_size, n_channels)
        
        # 使用nn.MultiheadAttention，embed_dim = n_channels
        # 自动调整num_heads以确保embed_dim能被num_heads整除
        num_heads = 1
        if n_channels >= 4:
            # 尝试找到合适的num_heads（不超过4，且能整除n_channels）
            for h in [4, 2, 1]:
                if n_channels % h == 0:
                    num_heads = h
                    break
        
        self.attention = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.norm1 = nn.LayerNorm(n_channels)
        self.norm2 = nn.LayerNorm(n_channels)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(n_channels, n_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_channels * 2, n_channels),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B*num_patches, patch_size, C]
        Returns:
            out: [B*num_patches, patch_size, C]
        """
        B, T, C = x.shape  # B = B*num_patches, T = patch_size
        
        # 保存原始输入用于残差连接
        residual = x
        
        # 位置编码（用于key/value）
        positions = torch.arange(self.patch_size, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)  # [B, patch_size, C]
        
        # 添加位置编码到输入（作为key/value）
        x_pos = x + pos_emb  # [B, patch_size, C]
        
        # 可学习的query: [patch_size, n_channels] -> [B, patch_size, C]
        learnable_q = self.learnable_query.unsqueeze(0).expand(B, -1, -1)  # [B, patch_size, C]
        
        # Cross-attention: query来自可学习参数，key/value来自输入
        attn_out, _ = self.attention(learnable_q, x_pos, x_pos)  # [B, patch_size, C]
        
        # 残差连接和Layer Norm（注意：这里残差连接的是query的输出，而不是原始输入）
        x_normed = self.norm1(learnable_q + self.dropout(attn_out))  # [B, patch_size, C]
        
        # FFN（带残差连接）
        ffn_out = self.ffn(x_normed)  # [B, patch_size, C]
        x_out = self.norm2(x_normed + ffn_out)  # [B, patch_size, C]
        
        # 整体残差连接：输入 + 处理后的输出
        out = residual + x_out
        
        return out


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
        self.use_codebook_ema = config.get('codebook_ema', False)
        self.ema_decay = config.get('ema_decay', 0.99)
        self.ema_eps = config.get('ema_eps', 1e-5)
        
        # VQVAE 配置
        self.num_hiddens = config.get('num_hiddens', 64)
        self.num_residual_layers = config.get('num_residual_layers', 2)
        self.num_residual_hiddens = config.get('num_residual_hiddens', 32)
        
        self.compressed_len = self.patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len  # Transformer 输入维度
        
        # Patch内时序建模配置（支持TCN、Self-Attention和Cross-Attention）
        self.use_patch_attention = config.get('use_patch_attention', False)
        patch_attention_type = config.get('patch_attention_type', 'tcn')  # 'tcn', 'attention', 或 'cross_attention'
        n_channels = config.get('n_channels', None)  # 如果提供了通道数，立即初始化
        
        if self.use_patch_attention and n_channels is not None:
            if patch_attention_type == 'tcn':
                tcn_num_layers = config.get('tcn_num_layers', 2)
                tcn_kernel_size = config.get('tcn_kernel_size', 3)
                tcn_hidden_dim = config.get('tcn_hidden_dim', None)
                self.patch_attention = PatchTCN(
                    patch_size=self.patch_size,
                    n_channels=n_channels,
                    dropout=self.dropout,
                    num_layers=tcn_num_layers,
                    kernel_size=tcn_kernel_size,
                    hidden_dim=tcn_hidden_dim
                )
            elif patch_attention_type == 'cross_attention':
                self.patch_attention = PatchCrossAttention(
                    patch_size=self.patch_size,
                    n_channels=n_channels,
                    dropout=self.dropout
                )
            else:  # 'attention' (self-attention)
                self.patch_attention = PatchSelfAttention(
                    patch_size=self.patch_size,
                    n_channels=n_channels,
                    dropout=self.dropout
                )
        else:
            self.patch_attention = None
        
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
        if self.use_codebook_ema:
            self.vq = FlattenedVectorQuantizerEMA(
                self.codebook_size, self.code_dim, self.commitment_cost,
                decay=self.ema_decay, eps=self.ema_eps
            )
        else:
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
    
    def encode_to_indices(self, x, return_processed_patches=False):
        """
        编码为码本索引和量化向量
        
        Args:
            x: [B, T, C]
            return_processed_patches: 是否返回经过时序建模处理后的patches
        Returns:
            indices: [B, num_patches, C]
            vq_loss: scalar
            z_q: [B, num_patches, C, code_dim]
            x_patches_processed: [B, num_patches, patch_size, C] (如果return_processed_patches=True)
        """
        B, T, C = x.shape
        num_patches = T // self.patch_size
        
        # 重组为 patches: [B, num_patches, patch_size, C]
        x = x[:, :num_patches * self.patch_size, :]
        x_patches = x.reshape(B, num_patches, self.patch_size, C)
        
        # 应用Patch内时序建模（TCN或Attention，如果启用）
        if self.use_patch_attention:
            if self.patch_attention is None:
                raise RuntimeError(
                    "patch_attention 未初始化。请在创建模型时通过 config['n_channels'] 提供通道数，"
                    "或使用 load_vqvae_weights 方法时提供 n_channels 参数。"
                )
            
            # 对每个patch，在patch_size × C上应用时序建模
            # [B, num_patches, patch_size, C] -> [B*num_patches, patch_size, C]
            x_patches_flat = x_patches.reshape(B * num_patches, self.patch_size, C)
            
            # 应用时序建模（TCN或Attention）: [B*num_patches, patch_size, C] -> [B*num_patches, patch_size, C]
            x_out = self.patch_attention(x_patches_flat)
            
            # 恢复形状: [B*num_patches, patch_size, C] -> [B, num_patches, patch_size, C]
            x_patches = x_out.reshape(B, num_patches, self.patch_size, C)
        
        # 保存处理后的patches（用于重构损失）
        x_patches_processed = x_patches if return_processed_patches else None
        
        # 重组为 [B*num_patches*C, patch_size] 用于编码
        x = x_patches.permute(0, 1, 3, 2).reshape(-1, self.patch_size)  # [B*num_patches*C, patch_size]
        
        # VQVAE Encoder
        z = self.encoder(x, self.compression_factor)  # [B*num_patches*C, embedding_dim, compressed_len]
        z_flat = z.reshape(z.shape[0], -1)  # [B*num_patches*C, code_dim]
        
        # VQ
        vq_loss, z_q_flat, indices = self.vq(z_flat)
        
        # Reshape
        indices = indices.reshape(B, num_patches, C)
        z_q = z_q_flat.reshape(B, num_patches, C, self.code_dim)
        
        if return_processed_patches:
            return indices, vq_loss, z_q, x_patches_processed
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
        
        # 编码（返回处理后的patches用于重构损失）
        if self.use_patch_attention:
            indices, vq_loss, z_q, x_patches_processed = self.encode_to_indices(x, return_processed_patches=True)
        else:
            indices, vq_loss, z_q = self.encode_to_indices(x, return_processed_patches=False)
            x_patches_processed = None
        num_patches = indices.shape[1]
        
        # 重构损失
        x_recon = self.decode_from_codes(z_q)  # [B, num_patches * patch_size, C]
        
        # 如果使用了patch时序建模，计算处理后的patches与重构结果的损失
        if self.use_patch_attention and x_patches_processed is not None:
            # 将处理后的patches reshape为 [B, num_patches * patch_size, C]
            x_processed = x_patches_processed.reshape(B, num_patches * self.patch_size, C)
            recon_loss = F.mse_loss(x_recon, x_processed[:, :x_recon.shape[1], :])
        else:
            # 否则使用原始输入
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
        微调: 预测未来序列（非自回归版本）
        
        1. 编码输入为量化向量
        2. 创建占位符位置（pred_len / patch_size个）
        3. Transformer 处理整个序列（输入 + 占位符），一次性预测所有未来patches
        4. 从码本获取向量并解码
        
        非自回归：先留出对应数量的位置，然后对应预测，最终解码
        """
        B, T, C = x.shape
        num_pred_patches = (target_len + self.patch_size - 1) // self.patch_size
        
        # 编码输入
        indices, vq_loss, z_q = self.encode_to_indices(x)  # z_q: [B, num_patches, C, code_dim]
        num_input_patches = z_q.shape[1]
        
        # 批量处理所有channels: [B, num_patches, C, code_dim] -> [B*C, num_patches, code_dim]
        z_flat = z_q.permute(0, 2, 1, 3).reshape(B * C, num_input_patches, self.code_dim)
        
        # 创建占位符（零向量）: [B*C, num_pred_patches, code_dim]
        placeholder = torch.zeros(B * C, num_pred_patches, self.code_dim, device=z_flat.device, dtype=z_flat.dtype)
        
        # 拼接输入序列和占位符: [B*C, num_patches + num_pred_patches, code_dim]
        full_sequence = torch.cat([z_flat, placeholder], dim=1)  # [B*C, num_patches + num_pred_patches, code_dim]
        
        # Transformer处理整个序列（causal mask确保占位符只能看到输入序列，不能看到其他占位符）
        h_full = self.transformer(full_sequence)  # [B*C, num_patches + num_pred_patches, code_dim]
        
        # 只取占位符位置的输出: [B*C, num_pred_patches, code_dim]
        h_pred = h_full[:, num_input_patches:, :]  # [B*C, num_pred_patches, code_dim]
        
        # 输出头: [B*C, num_pred_patches, codebook_size]
        logits = self.output_head(h_pred)  # [B*C, num_pred_patches, codebook_size]
        
        # 使用 softmax + 加权求和 替代 argmax，保持可微分
        weights = F.softmax(logits, dim=-1)  # [B*C, num_pred_patches, codebook_size]
        codebook = self.vq.embedding.weight  # [codebook_size, code_dim]
        pred_codes_flat = torch.matmul(weights, codebook)  # [B*C, num_pred_patches, code_dim]
        
        # 恢复形状: [B*C, num_pred_patches, code_dim] -> [B, num_pred_patches, C, code_dim]
        pred_codes = pred_codes_flat.reshape(B, C, num_pred_patches, self.code_dim).permute(0, 2, 1, 3)
        
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
        result = self.encode_to_indices(x, return_processed_patches=False)
        if isinstance(result, tuple) and len(result) == 4:
            indices, _, _, _ = result
        else:
            indices, _, _ = result
        unique = torch.unique(indices.reshape(-1))
        return len(unique) / self.codebook_size, unique
    
    def load_vqvae_weights(self, checkpoint_path, device='cpu', load_vq=True, n_channels=None):
        """
        加载预训练的VQVAE权重（包括encoder、decoder、VQ和patch_attention/TCN）
        
        Args:
            checkpoint_path: checkpoint路径
            device: 设备
            load_vq: 是否加载VQ层权重
            n_channels: 通道数（如果提供且patch_attention未初始化，会立即初始化）
        """
        import os
        if not os.path.exists(checkpoint_path):
            print(f"警告: checkpoint不存在: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 提取state_dict
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint)
            elif hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
            else:
                state_dict = None
            
            loaded_components = []
            
            # 辅助函数：加载模块权重
            def load_module_weights(module, prefix, module_name):
                if state_dict is not None:
                    module_dict = {k.replace(f'{prefix}.', ''): v for k, v in state_dict.items() 
                                  if k.startswith(f'{prefix}.')}
                    if module_dict:
                        module.load_state_dict(module_dict, strict=False)
                        return True
                elif hasattr(checkpoint, prefix):
                    module.load_state_dict(getattr(checkpoint, prefix).state_dict(), strict=False)
                    return True
                return False
            
            # 加载encoder和decoder
            if load_module_weights(self.encoder, 'encoder', 'Encoder'):
                loaded_components.append('Encoder')
            if load_module_weights(self.decoder, 'decoder', 'Decoder'):
                loaded_components.append('Decoder')
            
            # 加载VQ层
            if load_vq:
                if state_dict is not None:
                    vq_keys = [k for k in state_dict.keys() if 'vq' in k.lower() and 'embedding' in k.lower()]
                    for key in vq_keys:
                        if ('weight' in key or 'embedding' in key) and hasattr(self.vq, 'embedding'):
                            try:
                                self.vq.embedding.weight.data.copy_(state_dict[key])
                                loaded_components.append('VQ')
                                break
                            except:
                                continue
                elif hasattr(checkpoint, 'vq'):
                    vq = checkpoint.vq
                    if hasattr(vq, 'embedding'):
                        self.vq.embedding.weight.data.copy_(vq.embedding.weight.data)
                        loaded_components.append('VQ')
                    elif hasattr(vq, '_embedding'):
                        self.vq.embedding.weight.data.copy_(vq._embedding.weight.data)
                        loaded_components.append('VQ')
            
            # 加载Patch Attention/TCN权重
            patch_attention_dict = None
            if state_dict is not None:
                patch_attention_dict = {k: v for k, v in state_dict.items() if 'patch_attention' in k}
            elif hasattr(checkpoint, 'patch_attention'):
                patch_attention_dict = checkpoint.patch_attention.state_dict()
            
            if patch_attention_dict:
                # 如果patch_attention未初始化，尝试初始化
                if self.patch_attention is None:
                    if n_channels is not None and self.use_patch_attention:
                        # 根据权重键名判断类型
                        # TCN的键名包含'tcn_layers'
                        # Cross-Attention的键名包含'learnable_query'
                        # Self-Attention的键名包含'attention'或'pos_embedding'（但没有'learnable_query'）
                        has_tcn_keys = any('tcn_layers' in k for k in patch_attention_dict.keys())
                        has_learnable_query = any('learnable_query' in k for k in patch_attention_dict.keys())
                        has_attention_keys = any('attention' in k or 'pos_embedding' in k for k in patch_attention_dict.keys())
                        
                        if has_tcn_keys:
                            # 尝试从权重推断TCN参数
                            tcn_num_layers = len([k for k in patch_attention_dict.keys() if 'tcn_layers' in k and 'weight' in k]) // 2
                            tcn_num_layers = max(1, tcn_num_layers) if tcn_num_layers > 0 else 2
                            self.patch_attention = PatchTCN(
                                patch_size=self.patch_size,
                                n_channels=n_channels,
                                dropout=self.dropout,
                                num_layers=tcn_num_layers
                            ).to(device)
                        elif has_learnable_query:
                            self.patch_attention = PatchCrossAttention(
                                patch_size=self.patch_size,
                                n_channels=n_channels,
                                dropout=self.dropout
                            ).to(device)
                        elif has_attention_keys:
                            self.patch_attention = PatchSelfAttention(
                                patch_size=self.patch_size,
                                n_channels=n_channels,
                                dropout=self.dropout
                            ).to(device)
                        else:
                            # 默认使用TCN
                            self.patch_attention = PatchTCN(
                                patch_size=self.patch_size,
                                n_channels=n_channels,
                                dropout=self.dropout
                            ).to(device)
                    else:
                        raise RuntimeError(
                            "无法加载 Patch Attention/TCN 权重：patch_attention 未初始化且未提供 n_channels。"
                            "请通过 config['n_channels'] 或 load_vqvae_weights 的 n_channels 参数提供通道数。"
                        )
                
                # 加载权重
                try:
                    clean_dict = {k.replace('model.patch_attention.', '').replace('patch_attention.', ''): v 
                                 for k, v in patch_attention_dict.items()}
                    self.patch_attention.load_state_dict(clean_dict, strict=False)
                    # 判断模块类型
                    if hasattr(self.patch_attention, 'tcn_layers'):
                        module_type = 'Patch TCN'
                    elif hasattr(self.patch_attention, 'learnable_query'):
                        module_type = 'Patch Cross-Attention'
                    else:
                        module_type = 'Patch Self-Attention'
                    loaded_components.append(module_type)
                except Exception as e:
                    print(f"加载 Patch Attention/TCN 权重时出错: {e}")
            
            # 打印加载结果
            if loaded_components:
                print(f"成功加载: {', '.join(loaded_components)}")
            
            return len(loaded_components) > 0
            
        except Exception as e:
            print(f"加载权重失败: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============ 工具函数 ============

def get_model_config(args):
    """构建模型配置
    
    注意: n_channels 需要从数据加载器获取，应在调用此函数后添加到 config 中:
        config = get_model_config(args)
        config['n_channels'] = dls.vars  # 从数据加载器获取通道数
    """
    # code_dim = embedding_dim * (patch_size / compression_factor)
    code_dim = args.embedding_dim * (args.patch_size // args.compression_factor)
    print(f"Transformer 输入维度 (code_dim) = {code_dim}")
    
    config = {
        'patch_size': args.patch_size,
        'embedding_dim': args.embedding_dim,
        'compression_factor': args.compression_factor,
        'codebook_size': args.codebook_size,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'commitment_cost': args.commitment_cost,
        'codebook_ema': bool(args.codebook_ema),
        'ema_decay': args.ema_decay,
        'ema_eps': args.ema_eps,
        # VQVAE Encoder/Decoder 配置
        'num_hiddens': args.num_hiddens,
        'num_residual_layers': args.num_residual_layers,
        'num_residual_hiddens': args.num_residual_hiddens,
    }
    
    # Patch内时序建模配置（支持TCN、Self-Attention和Cross-Attention）
    if hasattr(args, 'use_patch_attention'):
        config['use_patch_attention'] = bool(args.use_patch_attention)
        config['patch_attention_type'] = getattr(args, 'patch_attention_type', 'tcn')  # 'tcn', 'attention', 或 'cross_attention'
        config['tcn_num_layers'] = getattr(args, 'tcn_num_layers', 2)
        config['tcn_kernel_size'] = getattr(args, 'tcn_kernel_size', 3)
        config['tcn_hidden_dim'] = getattr(args, 'tcn_hidden_dim', None)
    
    # 注意: n_channels 需要从数据加载器获取，应在调用此函数后添加:
    # config['n_channels'] = dls.vars
    
    return config
