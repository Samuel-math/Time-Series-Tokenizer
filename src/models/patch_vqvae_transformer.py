"""
Patch-based VQVAE + Transformer 模型架构 (Channel-Independent版本)

架构说明：
1. 输入: [B, T, C] 时间序列
2. Channel-independent处理: 每个通道独立进行 Patch划分 + VQVAE Encoder -> [B, num_patches, C, code_dim]
3. VQ 量化后的表示直接作为 Transformer 输入 (无需 token embedding)
4. Transformer (Decoder-only): 对每个通道独立预测下一个码本向量
5. 预训练: NTP loss (预测码本索引) - logits: [B, num_patches-1, C, codebook_size]
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
    def __init__(self, codebook_size, code_dim, commitment_cost=0.25, init_method='random'):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.init_method = init_method
        
        # 码本: [codebook_size, code_dim]
        self.embedding = nn.Embedding(codebook_size, code_dim)
        
        # 初始化方法
        if init_method == 'random':
            # 完全随机初始化
            nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        elif init_method == 'normal':
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        elif init_method == 'xavier':
            nn.init.xavier_uniform_(self.embedding.weight)
        elif init_method == 'kaiming':
            nn.init.kaiming_uniform_(self.embedding.weight)
        else:
            # 默认完全随机
            nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
    
    def init_from_data(self, z_samples, method='kmeans'):
        """
        从数据初始化码本（数据驱动初始化）
        
        Args:
            z_samples: [N, code_dim] encoder输出的样本
            method: 'kmeans' 或 'random_sample'
        """
        if method == 'kmeans':
            try:
                from sklearn.cluster import KMeans
                z_np = z_samples.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=self.codebook_size, n_init=10, max_iter=300)
                kmeans.fit(z_np)
                centroids = torch.tensor(kmeans.cluster_centers_, dtype=z_samples.dtype, device=z_samples.device)
                self.embedding.weight.data.copy_(centroids)
                print(f"✓ 码本已从K-means聚类初始化 (codebook_size={self.codebook_size}, 样本数={len(z_samples)})")
            except ImportError:
                print("警告: sklearn未安装，使用随机采样初始化")
                method = 'random_sample'
        
        if method == 'random_sample':
            # 随机采样N个样本作为码本中心（不使用种子，允许随机性）
            N = z_samples.size(0)
            if N >= self.codebook_size:
                perm = torch.randperm(N)[:self.codebook_size]
                centroids = z_samples[perm]
            else:
                # 如果样本数不足，使用重复采样
                indices = torch.randint(0, N, (self.codebook_size,))
                centroids = z_samples[indices]
            self.embedding.weight.data.copy_(centroids)
            print(f"✓ 码本已从随机采样初始化 (codebook_size={self.codebook_size}, 样本数={N})")
    
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
    def __init__(self, codebook_size, code_dim, commitment_cost=0.25, decay=0.99, eps=1e-5, init_method='random'):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        self.init_method = init_method
        
        # 码本权重与EMA状态
        if init_method == 'random':
            # 完全随机初始化
            embed = torch.randn(codebook_size, code_dim)
        elif init_method == 'normal':
            embed = torch.randn(codebook_size, code_dim) * 0.02
        elif init_method == 'xavier':
            embed = torch.empty(codebook_size, code_dim)
            nn.init.xavier_uniform_(embed)
        elif init_method == 'kaiming':
            embed = torch.empty(codebook_size, code_dim)
            nn.init.kaiming_uniform_(embed)
        else:
            # 默认完全随机
            embed = torch.randn(codebook_size, code_dim)
        
        self.embedding = nn.Embedding(codebook_size, code_dim)
        self.embedding.weight.data.copy_(embed)
        self.embedding.weight.requires_grad = False
        
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_w', embed.clone())
        # 标志：是否禁用EMA更新（当VQ被冻结时）
        self._disable_ema_update = False
    
    def init_from_data(self, z_samples, method='kmeans'):
        """
        从数据初始化码本（数据驱动初始化）
        
        Args:
            z_samples: [N, code_dim] encoder输出的样本
            method: 'kmeans' 或 'random_sample'
        """
        if method == 'kmeans':
            try:
                from sklearn.cluster import KMeans
                z_np = z_samples.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=self.codebook_size, n_init=10, max_iter=300)
                kmeans.fit(z_np)
                centroids = torch.tensor(kmeans.cluster_centers_, dtype=z_samples.dtype, device=z_samples.device)
                self.embedding.weight.data.copy_(centroids)
                self.ema_w.data.copy_(centroids)
                print(f"✓ 码本已从K-means聚类初始化 (codebook_size={self.codebook_size}, 样本数={len(z_samples)})")
            except ImportError:
                print("警告: sklearn未安装，使用随机采样初始化")
                method = 'random_sample'
        
        if method == 'random_sample':
            # 随机采样N个样本作为码本中心（不使用种子，允许随机性）
            N = z_samples.size(0)
            if N >= self.codebook_size:
                perm = torch.randperm(N)[:self.codebook_size]
                centroids = z_samples[perm]
            else:
                # 如果样本数不足，使用重复采样
                indices = torch.randint(0, N, (self.codebook_size,))
                centroids = z_samples[indices]
            self.embedding.weight.data.copy_(centroids)
            self.ema_w.data.copy_(centroids)
            print(f"✓ 码本已从随机采样初始化 (codebook_size={self.codebook_size}, 样本数={N})")
    
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
        
        # EMA更新：只在训练模式且未禁用时执行
        if self.training and not self._disable_ema_update:
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
    """轻量级 Causal Transformer，支持独立的 hidden_dim 参数"""
    def __init__(self, code_dim, n_heads, n_layers, d_ff, dropout=0.1, max_len=512, hidden_dim=None):
        super().__init__()
        
        # PyTorch 2.7+ 兼容性修复：禁用 flash attention 和 memory-efficient attention
        # 当使用 mask 时，这些优化可能导致 CUDA 错误
        # 强制使用标准的数学实现
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
        
        self.code_dim = code_dim
        # 如果未指定 hidden_dim，默认使用 code_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else code_dim
        
        # 如果 hidden_dim 与 code_dim 不同，需要输入和输出投影层
        if self.hidden_dim != self.code_dim:
            self.input_proj = nn.Linear(self.code_dim, self.hidden_dim)
            self.output_proj = nn.Linear(self.hidden_dim, self.code_dim)
        else:
            self.input_proj = None
            self.output_proj = None
        
        # 位置编码，维度与 hidden_dim 一致
        self.pos_embedding = nn.Embedding(max_len, self.hidden_dim)
        self.drop = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(self.hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, code_dim] 直接是量化后的码本向量
        Returns:
            output: [B, T, code_dim] 输出维度与输入一致
        """
        B, T, _ = x.shape
        
        # 输入投影（如果需要）
        if self.input_proj is not None:
            x = self.input_proj(x)  # [B, T, code_dim] -> [B, T, hidden_dim]
        
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)
        x = self.drop(x)
        
        # 创建 causal mask（上三角矩阵，对角线以上为 True）
        # 只使用 mask 参数，避免 is_causal=True 在某些 PyTorch 版本中的兼容性问题
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        x = self.transformer(x, mask=mask)
        x = self.norm(x)  # [B, T, hidden_dim]
        
        # 输出投影（如果需要）
        if self.output_proj is not None:
            x = self.output_proj(x)  # [B, T, hidden_dim] -> [B, T, code_dim]
        
        return x


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
        
        # code_dim = embedding_dim * compressed_len
        # Channel-independent: 每个通道独立处理，使用单通道Encoder/Decoder
        self.compressed_len = self.patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len  # Transformer 输入维度
        
        # Transformer的hidden_dim（用于Transformer内部维度，默认使用code_dim）
        self.transformer_hidden_dim = config.get('transformer_hidden_dim', None)
        
        # Patch内时序建模配置（支持TCN、Self-Attention和Cross-Attention）
        # Patch attention 已移除
        
        # VQ (码本维度 = code_dim)
        vq_init_method = config.get('vq_init_method', 'random')
        if self.use_codebook_ema:
            self.vq = FlattenedVectorQuantizerEMA(
                self.codebook_size, self.code_dim, self.commitment_cost,
                decay=self.ema_decay, eps=self.ema_eps,
                init_method=vq_init_method
            )
        else:
            self.vq = FlattenedVectorQuantizer(
                self.codebook_size, self.code_dim, self.commitment_cost,
                init_method=vq_init_method
            )
        
        # Transformer (输入维度 = code_dim，内部维度 = transformer_hidden_dim)
        self.transformer = CausalTransformer(
            self.code_dim, self.n_heads, self.n_layers, 
            self.d_ff, self.dropout, hidden_dim=self.transformer_hidden_dim
        )
        
        # 输出头: code_dim -> codebook_size (预测码本索引)
        # 注意：Transformer输出始终是code_dim（通过输出投影），所以output_head输入维度是code_dim
        self.output_head = nn.Linear(self.code_dim, self.codebook_size)

        # 获取通道数（从config中获取，如果不存在则为None，稍后通过load_vqvae_weights或直接设置）
        n_channels = config.get('n_channels', None)
        self._n_channels = n_channels
        # Channel-independent: 每个通道独立处理，使用单通道Encoder/Decoder
        self.encoder = Encoder(
            in_channels=1,  # 单通道输入
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
            compression_factor=self.compression_factor,
            out_channels=1  # 单通道输出
        )
        
        # Channel Attention 已移除
    
    def encode_to_indices(self, x):
        """
        编码为码本索引和量化向量（channel-independent版本）
        
        Args:
            x: [B, T, C] 输入序列
        Returns:
            indices: [B, num_patches, C]
            vq_loss: scalar
            z_q: [B, num_patches, C, code_dim]
        """
        B, T, C = x.shape
        
        num_patches = T // self.patch_size
        
        # 重组为 patches: [B, num_patches, patch_size, C]
        x = x[:, :num_patches * self.patch_size, :]
        x_patches = x.reshape(B, num_patches, self.patch_size, C)
        
        # Channel-independent: 对每个通道独立编码
        z_list = []
        
        for c in range(C):
            # 提取第c个通道的patches: [B, num_patches, patch_size]
            x_c = x_patches[:, :, :, c]  # [B, num_patches, patch_size]
            x_c_flat = x_c.reshape(B * num_patches, self.patch_size)  # [B*num_patches, patch_size]
            x_c_flat = x_c_flat.unsqueeze(1)  # [B*num_patches, 1, patch_size] (单通道输入)
            
            # VQVAE Encoder (单通道输入)
            z = self.encoder(x_c_flat, self.compression_factor)  # [B*num_patches, embedding_dim, compressed_len]
            z_flat = z.reshape(B * num_patches, -1)  # [B*num_patches, code_dim]
            z_c = z_flat.reshape(B, num_patches, self.code_dim)  # [B, num_patches, code_dim]
            
            z_list.append(z_c)
        
        # 合并所有通道: [B, num_patches, C, code_dim]
        z_all = torch.stack(z_list, dim=2)  # [B, num_patches, C, code_dim]
        
        # VQ量化（对每个通道独立进行）
        indices_list = []
        z_q_list = []
        vq_loss_sum = 0
        
        for c in range(C):
            z_c = z_all[:, :, c, :]  # [B, num_patches, code_dim]
            z_c_flat = z_c.reshape(B * num_patches, self.code_dim)  # [B*num_patches, code_dim]
            
            # VQ
            vq_loss_c, z_q_flat_c, indices_c = self.vq(z_c_flat)
            vq_loss_sum += vq_loss_c
            
            # Reshape: [B*num_patches] -> [B, num_patches]
            indices_c = indices_c.reshape(B, num_patches)  # [B, num_patches]
            z_q_c = z_q_flat_c.reshape(B, num_patches, self.code_dim)  # [B, num_patches, code_dim]
            
            indices_list.append(indices_c)
            z_q_list.append(z_q_c)
        
        # 合并所有通道: [B, num_patches, C] 和 [B, num_patches, C, code_dim]
        indices = torch.stack(indices_list, dim=2)  # [B, num_patches, C]
        z_q = torch.stack(z_q_list, dim=2)  # [B, num_patches, C, code_dim]
        vq_loss = vq_loss_sum / C  # 平均VQ损失
        
        return indices, vq_loss, z_q
    
    def decode_from_codes(self, z_q):
        """
        从量化向量解码（channel-independent版本）
        
        Args:
            z_q: [B, num_patches, C, code_dim]
        Returns:
            x_recon: [B, num_patches * patch_size, C]
        """
        B, num_patches, C, code_dim = z_q.shape
        
        # Channel-independent: 对每个通道独立解码
        x_recon_list = []
        
        for c in range(C):
            # 提取第c个通道的量化向量: [B, num_patches, code_dim]
            z_q_c = z_q[:, :, c, :]  # [B, num_patches, code_dim]
            
            # Reshape for decoder: [B*num_patches, embedding_dim, compressed_len]
            z_q_c_flat = z_q_c.reshape(B * num_patches, self.embedding_dim, self.compressed_len)
            
            # VQVAE Decoder (单通道输出)
            x_recon_c = self.decoder(z_q_c_flat, self.compression_factor)  # [B*num_patches, patch_size]
            x_recon_c = x_recon_c.reshape(B, num_patches, self.patch_size)  # [B, num_patches, patch_size]
            
            x_recon_list.append(x_recon_c)
        
        # 合并所有通道: [B, num_patches, patch_size, C]
        x_recon = torch.stack(x_recon_list, dim=3)  # [B, num_patches, patch_size, C]
        x_recon = x_recon.reshape(B, -1, C)  # [B, num_patches * patch_size, C]
        
        return x_recon
    
    def forward_progressive_pretrain(self, x_full, step_size, max_stages=None):
        """
        渐进式预训练: 使用不同长度的上下文预测固定长度的未来tokens
        
        例如，如果 step_size=a，max_stages=3：
        - 阶段1: 使用前 a 个tokens 预测接下来的 a 个tokens
        - 阶段2: 使用前 2a 个tokens 预测接下来的 a 个tokens (从位置 a 开始)
        - 阶段3: 使用前 3a 个tokens 预测接下来的 a 个tokens (从位置 2a 开始)
        
        Args:
            x_full: [B, total_len, C] 完整序列
            step_size: int, 每个阶段的步长（以patches为单位）
            max_stages: int, 最大阶段数。如果为None，则使用所有可能的阶段
        
        Returns:
            all_logits: List of [B, num_target_patches, C, codebook_size] 每个阶段的预测logits
            all_target_indices: List of [B, num_target_patches, C] 每个阶段的目标索引
            vq_loss: VQ损失（所有阶段的平均）
            recon_loss: 重构损失（所有阶段的平均）
        """
        B, total_len, C = x_full.shape
        
        # 编码完整序列
        full_indices, vq_loss_full, z_q_full = self.encode_to_indices(x_full)
        
        num_total_patches = full_indices.shape[1]
        
        # 计算最大阶段数
        if max_stages is None:
            max_stages = (num_total_patches - step_size) // step_size
        else:
            max_stages = min(max_stages, (num_total_patches - step_size) // step_size)
        
        if max_stages <= 0:
            raise ValueError(f"序列长度不足：总patches={num_total_patches}, step_size={step_size}, 无法创建任何阶段")
        
        all_logits = []
        all_target_indices = []
        all_vq_losses = []
        all_recon_losses = []
        
        # Channel-independent处理: [B, num_patches, C, code_dim] -> [B*C, num_patches, code_dim]
        B, num_patches, C, code_dim = z_q_full.shape
        z_q_full_flat = z_q_full.permute(0, 2, 1, 3).reshape(B * C, num_patches, code_dim)
        
        # 遍历每个阶段
        for stage in range(1, max_stages + 1):
            context_size = stage * step_size  # 当前阶段的上下文长度
            target_start = (stage - 1) * step_size  # 目标序列的起始位置
            target_end = stage * step_size  # 目标序列的结束位置
            
            if target_end > num_patches:
                break
            
            # 提取当前阶段的上下文和目标（直接使用已编码的完整序列）
            z_q_context = z_q_full_flat[:, :context_size, :]  # [B*C, context_size, code_dim]
            target_indices_stage = full_indices[:, target_start:target_end, :]  # [B, step_size, C]
            
            # 重构损失（仅使用上下文序列）
            # 从已编码的完整序列中提取上下文部分，避免重复编码
            z_q_context_for_recon = z_q_full[:, :context_size, :, :]  # [B, context_size, C, code_dim]
            x_recon_context = self.decode_from_codes(z_q_context_for_recon)
            
            # 计算上下文对应的原始序列长度
            context_len = context_size * self.patch_size
            x_context = x_full[:, :context_len, :]  # [B, context_len, C]
            
            # 重构损失
            recon_loss_stage = F.mse_loss(
                x_recon_context, 
                x_context[:, :x_recon_context.shape[1], :]
            )
            
            # VQ损失（使用完整序列的VQ损失，简化处理）
            vq_loss_context = vq_loss_full
            
            # 创建占位符用于预测目标序列
            num_target_patches = step_size
            placeholder = torch.zeros(
                B * C, num_target_patches, code_dim,
                device=z_q_context.device, dtype=z_q_context.dtype
            )
            
            # 拼接上下文和占位符
            full_sequence_stage = torch.cat([z_q_context, placeholder], dim=1)
            
            # Transformer处理完整序列
            h_full_stage = self.transformer(full_sequence_stage)
            
            # 只取占位符位置的输出
            h_target_stage = h_full_stage[:, context_size:, :]
            
            # 输出头: 预测目标序列的索引概率分布
            logits_flat_stage = self.output_head(h_target_stage)  # [B*C, num_target_patches, codebook_size]
            
            # Reshape回通道分离格式
            logits_stage = logits_flat_stage.reshape(B, C, num_target_patches, -1).permute(
                0, 2, 1, 3
            )  # [B, num_target_patches, C, codebook_size]
            
            all_logits.append(logits_stage)
            all_target_indices.append(target_indices_stage)
            all_vq_losses.append(vq_loss_context)
            all_recon_losses.append(recon_loss_stage)
        
        # 计算平均损失
        vq_loss = sum(all_vq_losses) / len(all_vq_losses)
        recon_loss = sum(all_recon_losses) / len(all_recon_losses)
        
        return all_logits, all_target_indices, vq_loss, recon_loss
    
    def forward_finetune(self, x, target_len):
        """
        微调: 预测未来序列（非自回归版本，channel-independent）
        
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
        
        # Channel-independent: 批量处理所有通道以加速
        # z_q: [B, num_patches, C, code_dim] -> [B*C, num_patches, code_dim]
        B, num_patches, C, code_dim = z_q.shape
        z_q_flat = z_q.permute(0, 2, 1, 3).reshape(B * C, num_patches, code_dim)  # [B*C, num_patches, code_dim]
        
        # 创建占位符（零向量）: [B*C, num_pred_patches, code_dim]
        placeholder = torch.zeros(B * C, num_pred_patches, self.code_dim, device=z_q_flat.device, dtype=z_q_flat.dtype)
        
        # 拼接输入序列和占位符: [B*C, num_patches + num_pred_patches, code_dim]
        full_sequence = torch.cat([z_q_flat, placeholder], dim=1)  # [B*C, num_patches + num_pred_patches, code_dim]
        
        # Transformer处理整个序列（causal mask确保占位符只能看到输入序列，不能看到其他占位符）
        h_full = self.transformer(full_sequence)  # [B*C, num_patches + num_pred_patches, code_dim]
        
        # 只取占位符位置的输出: [B*C, num_pred_patches, code_dim]
        h_pred = h_full[:, num_input_patches:, :]  # [B*C, num_pred_patches, code_dim]
        
        # 输出头: [B*C, num_pred_patches, codebook_size]
        logits = self.output_head(h_pred)  # [B*C, num_pred_patches, codebook_size]
        
        # 使用 softmax + 加权求和 替代 argmax，保持可微分
        weights = F.softmax(logits, dim=-1)  # [B*C, num_pred_patches, codebook_size]
        codebook = self.vq.embedding.weight  # [codebook_size, code_dim]
        pred_codes = torch.matmul(weights, codebook)  # [B*C, num_pred_patches, code_dim]
        
        # Reshape回通道分离格式: [B*C, num_pred_patches, code_dim] -> [B, num_pred_patches, C, code_dim]
        pred_codes = pred_codes.reshape(B, C, num_pred_patches, code_dim).permute(0, 2, 1, 3)  # [B, num_pred_patches, C, code_dim]
        
        # 解码
        pred = self.decode_from_codes(pred_codes)  # [B, num_pred_patches*patch_size, C]
        pred = pred[:, :target_len, :]  # [B, target_len, C]
        
        # 确保输出长度与目标长度一致
        assert pred.shape[1] == target_len, f"预测长度 {pred.shape[1]} 与目标长度 {target_len} 不匹配"
        
        return pred, vq_loss
    
    def forward(self, x, target=None, target_len=None, mode='pretrain'):
        if mode == 'pretrain':
            if target is None:
                raise ValueError("pretrain mode requires target argument")
            return self.forward_pretrain(x, target)
        else:
            if target_len is None:
                raise ValueError("finetune mode requires target_len argument")
            return self.forward_finetune(x, target_len)
    
    @torch.no_grad()
    def get_codebook_usage(self, x):
        indices, _, _ = self.encode_to_indices(x)
        # indices: [B, num_patches, C] (channel-independent)
        unique = torch.unique(indices.reshape(-1))
        return len(unique) / self.codebook_size, unique
    
    def load_vqvae_weights(self, checkpoint_path, device='cpu', load_vq=True, freeze=False):
        """
        加载预训练的VQVAE权重（包括encoder、decoder、VQ）
        
        Args:
            checkpoint_path: checkpoint路径
            device: 设备
            load_vq: 是否加载VQ层权重
            freeze: 是否在加载后冻结VQVAE组件（encoder、decoder、VQ）
        """
        import os
        if not os.path.exists(checkpoint_path):
            print(f"警告: checkpoint不存在: {checkpoint_path}")
            return False
        
        try:
            # PyTorch 2.6+ 兼容性：设置 weights_only=False 以支持包含 numpy 对象的 checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # 提取state_dict
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint)
            elif hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
            else:
                state_dict = None
            
            loaded_components = []
            
            # 辅助函数：尝试加载模块权重（支持两种格式）
            def try_load_module(module, name, state_dict_key=None, prefix=None):
                # 格式1: codebook_pretrain格式 (encoder_state_dict, decoder_state_dict, vq_state_dict)
                if state_dict_key and state_dict_key in checkpoint:
                    try:
                        module.load_state_dict(checkpoint[state_dict_key], strict=False)
                        return True
                    except Exception as e:
                        print(f"加载{name}权重失败 ({state_dict_key}): {e}")
                
                # 格式2: 标准格式 (model_state_dict中包含prefix.)
                if prefix and state_dict is not None:
                    module_dict = {k.replace(f'{prefix}.', ''): v for k, v in state_dict.items() 
                                  if k.startswith(f'{prefix}.')}
                    if module_dict:
                        try:
                            module.load_state_dict(module_dict, strict=False)
                            return True
                        except Exception as e:
                            print(f"加载{name}权重失败 ({prefix}): {e}")
                
                # 格式3: checkpoint对象属性
                if hasattr(checkpoint, name.lower()):
                    try:
                        module.load_state_dict(getattr(checkpoint, name.lower()).state_dict(), strict=False)
                        return True
                    except Exception as e:
                        print(f"加载{name}权重失败 (checkpoint.{name.lower()}): {e}")
                
                return False
            
            # 加载encoder、decoder、VQ
            # 注意：TFCPatchVQVAE使用time_encoder，需要兼容两种前缀
            if try_load_module(self.encoder, 'Encoder', 'encoder_state_dict', 'encoder'):
                loaded_components.append('Encoder')
            elif try_load_module(self.encoder, 'Encoder', None, 'time_encoder'):
                # 兼容TFCPatchVQVAE的time_encoder
                loaded_components.append('Encoder')
            
            if try_load_module(self.decoder, 'Decoder', 'decoder_state_dict', 'decoder'):
                loaded_components.append('Decoder')
            
            if load_vq:
                if try_load_module(self.vq, 'VQ', 'vq_state_dict', 'vq'):
                    loaded_components.append('VQ')
                elif state_dict is not None:
                    # 尝试直接加载embedding权重
                    for key in state_dict.keys():
                        if 'vq' in key.lower() and 'embedding' in key.lower() and 'weight' in key.lower():
                            if hasattr(self.vq, 'embedding'):
                                try:
                                    self.vq.embedding.weight.data.copy_(state_dict[key])
                                    loaded_components.append('VQ')
                                    break
                                except:
                                    continue
            
            if loaded_components:
                print(f"成功加载: {', '.join(loaded_components)}")
                
                # 如果指定冻结，则冻结已加载的组件
                if freeze:
                    self.freeze_vqvae(components=loaded_components)
            
            return len(loaded_components) > 0
            
        except Exception as e:
            print(f"加载权重失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def freeze_vqvae(self, components=None):
        """
        冻结VQVAE组件（encoder、decoder、VQ）
        
        Args:
            components: 要冻结的组件列表，如 ['Encoder', 'Decoder', 'VQ']。
                       如果为None，则冻结所有VQVAE组件
        """
        if components is None:
            components = ['Encoder', 'Decoder', 'VQ']
        
        frozen = []
        if 'Encoder' in components:
            for param in self.encoder.parameters():
                param.requires_grad = False
            frozen.append('Encoder')
        
        if 'Decoder' in components:
            for param in self.decoder.parameters():
                param.requires_grad = False
            frozen.append('Decoder')
        
        if 'VQ' in components:
            for param in self.vq.parameters():
                param.requires_grad = False
            # 如果使用EMA codebook，禁用EMA更新
            if isinstance(self.vq, FlattenedVectorQuantizerEMA):
                self.vq._disable_ema_update = True
            frozen.append('VQ')
        
        if frozen:
            print(f"✓ 已冻结: {', '.join(frozen)}")
        
        return frozen
    
    def unfreeze_vqvae(self, components=None):
        """
        解冻VQVAE组件（encoder、decoder、VQ）
        
        Args:
            components: 要解冻的组件列表，如 ['Encoder', 'Decoder', 'VQ']。
                       如果为None，则解冻所有VQVAE组件
        """
        if components is None:
            components = ['Encoder', 'Decoder', 'VQ']
        
        unfrozen = []
        if 'Encoder' in components:
            for param in self.encoder.parameters():
                param.requires_grad = True
            unfrozen.append('Encoder')
        
        if 'Decoder' in components:
            for param in self.decoder.parameters():
                param.requires_grad = True
            unfrozen.append('Decoder')
        
        if 'VQ' in components:
            for param in self.vq.parameters():
                param.requires_grad = True
            # 如果使用EMA codebook，重新启用EMA更新
            if isinstance(self.vq, FlattenedVectorQuantizerEMA):
                self.vq._disable_ema_update = False
            unfrozen.append('VQ')
        
        if unfrozen:
            print(f"✓ 已解冻: {', '.join(unfrozen)}")
        
        return unfrozen


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
        # Transformer hidden_dim（可选，默认使用code_dim）
        'transformer_hidden_dim': getattr(args, 'transformer_hidden_dim', None),
    }
    
    # Patch attention 已移除
    
    # Channel Attention 已移除
    
    # 注意: n_channels 需要从数据加载器获取，应在调用此函数后添加:
    # config['n_channels'] = dls.vars
    
    return config
