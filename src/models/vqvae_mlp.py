"""
MLP-based VQ-VAE for Time Series

使用 MLP 替代 CNN 作为编码器和解码器。
适用于 patch-based 时间序列处理，每个 patch 独立编码。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPResidualBlock(nn.Module):
    """MLP 残差块"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(F.gelu(x + self.block(x)))


class MLPEncoder(nn.Module):
    """
    MLP 编码器
    
    输入: [B, patch_size] 或 [B, 1, patch_size]
    输出: [B, embedding_dim, compressed_len]
    
    Args:
        patch_size: 输入 patch 的长度
        embedding_dim: 输出的 embedding 维度
        compression_factor: 压缩因子，compressed_len = patch_size // compression_factor
        num_hiddens: 隐藏层维度
        num_layers: MLP 层数
        dropout: dropout 概率
    """
    def __init__(
        self, 
        patch_size, 
        embedding_dim, 
        compression_factor,
        num_hiddens=128,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.compression_factor = compression_factor
        self.compressed_len = patch_size // compression_factor
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(patch_size, num_hiddens),
            nn.LayerNorm(num_hiddens),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 残差 MLP 块
        self.residual_blocks = nn.ModuleList([
            MLPResidualBlock(num_hiddens, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影: 投影到 [embedding_dim * compressed_len]
        output_dim = embedding_dim * self.compressed_len
        self.output_proj = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.LayerNorm(num_hiddens),
            nn.GELU(),
            nn.Linear(num_hiddens, output_dim),
        )
    
    def forward(self, x, compression_factor=None):
        """
        Args:
            x: [B, patch_size] 或 [B, 1, patch_size]
            compression_factor: 未使用，保持接口兼容
        
        Returns:
            z: [B, embedding_dim, compressed_len]
        """
        # 处理输入形状
        if x.dim() == 3:
            # [B, 1, patch_size] -> [B, patch_size]
            x = x.squeeze(1)
        
        # 输入投影
        h = self.input_proj(x)  # [B, num_hiddens]
        
        # 残差块
        for block in self.residual_blocks:
            h = block(h)
        
        # 输出投影
        z = self.output_proj(h)  # [B, embedding_dim * compressed_len]
        
        # Reshape 为 [B, embedding_dim, compressed_len]
        z = z.view(-1, self.embedding_dim, self.compressed_len)
        
        return z


class MLPDecoder(nn.Module):
    """
    MLP 解码器
    
    输入: [B, embedding_dim, compressed_len]
    输出: [B, patch_size]
    
    Args:
        patch_size: 输出 patch 的长度
        embedding_dim: 输入的 embedding 维度
        compression_factor: 压缩因子
        num_hiddens: 隐藏层维度
        num_layers: MLP 层数
        dropout: dropout 概率
    """
    def __init__(
        self,
        patch_size,
        embedding_dim,
        compression_factor,
        num_hiddens=128,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.compression_factor = compression_factor
        self.compressed_len = patch_size // compression_factor
        
        # 输入维度
        input_dim = embedding_dim * self.compressed_len
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, num_hiddens),
            nn.LayerNorm(num_hiddens),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 残差 MLP 块
        self.residual_blocks = nn.ModuleList([
            MLPResidualBlock(num_hiddens, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.LayerNorm(num_hiddens),
            nn.GELU(),
            nn.Linear(num_hiddens, patch_size),
        )
    
    def forward(self, z, compression_factor=None):
        """
        Args:
            z: [B, embedding_dim, compressed_len]
            compression_factor: 未使用，保持接口兼容
        
        Returns:
            x_recon: [B, patch_size]
        """
        # Flatten: [B, embedding_dim, compressed_len] -> [B, embedding_dim * compressed_len]
        h = z.view(z.shape[0], -1)
        
        # 输入投影
        h = self.input_proj(h)  # [B, num_hiddens]
        
        # 残差块
        for block in self.residual_blocks:
            h = block(h)
        
        # 输出投影
        x_recon = self.output_proj(h)  # [B, patch_size]
        
        return x_recon


class VectorQuantizerMLP(nn.Module):
    """
    Vector Quantizer (与 CNN 版本相同)
    
    输入: [B, embedding_dim, compressed_len]
    输出: quantized [B, embedding_dim, compressed_len], loss, indices
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
    
    @property
    def embedding(self):
        return self._embedding
    
    def forward(self, inputs):
        """
        Args:
            inputs: [B, embedding_dim, compressed_len]
        
        Returns:
            loss: VQ loss
            quantized: [B, embedding_dim, compressed_len]
            perplexity: codebook perplexity
            embedding_weight: codebook weights
            encoding_indices: [B * compressed_len, 1]
            encodings: one-hot encodings
        """
        # [B, embedding_dim, compressed_len] -> [B, compressed_len, embedding_dim]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten: [B * compressed_len, embedding_dim]
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # 计算距离
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True) +
            torch.sum(self._embedding.weight ** 2, dim=1) -
            2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        
        # 编码
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # 损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # [B, compressed_len, embedding_dim] -> [B, embedding_dim, compressed_len]
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        return loss, quantized, perplexity, self._embedding.weight, encoding_indices, encodings


class VectorQuantizerEMAMLP(nn.Module):
    """
    Vector Quantizer with EMA (Exponential Moving Average) updates
    
    使用 EMA 更新码本，训练更稳定
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, eps=1e-5):
        super().__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._eps = eps
        
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.normal_()
        
        # EMA 相关 buffers
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self._embedding.weight.data.clone())
        
        self._update_ema = True  # 控制是否更新 EMA
    
    @property
    def embedding(self):
        return self._embedding
    
    def disable_ema_update(self):
        """禁用 EMA 更新"""
        self._update_ema = False
    
    def enable_ema_update(self):
        """启用 EMA 更新"""
        self._update_ema = True
    
    def forward(self, inputs):
        """
        Args:
            inputs: [B, embedding_dim, compressed_len]
        
        Returns:
            loss: VQ loss (commitment loss only)
            quantized: [B, embedding_dim, compressed_len]
            perplexity: codebook perplexity
            embedding_weight: codebook weights
            encoding_indices: [B * compressed_len, 1]
            encodings: one-hot encodings
        """
        # [B, embedding_dim, compressed_len] -> [B, compressed_len, embedding_dim]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten: [B * compressed_len, embedding_dim]
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # 计算距离
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True) +
            torch.sum(self._embedding.weight ** 2, dim=1) -
            2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        
        # 编码
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # EMA 更新（仅在训练时且启用时）
        if self.training and self._update_ema:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing
            n = torch.sum(self._ema_cluster_size)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._eps) /
                (n + self._num_embeddings * self._eps) * n
            )
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw
            
            self._embedding.weight.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)
        
        # 损失（仅 commitment loss）
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # [B, compressed_len, embedding_dim] -> [B, embedding_dim, compressed_len]
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        return loss, quantized, perplexity, self._embedding.weight, encoding_indices, encodings


class VQVAE_MLP(nn.Module):
    """
    MLP-based VQ-VAE
    
    完整的 VQ-VAE 模型，使用 MLP 作为编码器和解码器
    
    Args:
        patch_size: patch 长度
        embedding_dim: embedding 维度
        num_embeddings: 码本大小
        compression_factor: 压缩因子
        commitment_cost: commitment loss 权重
        num_hiddens: MLP 隐藏层维度
        num_layers: MLP 层数
        dropout: dropout 概率
        use_ema: 是否使用 EMA 更新码本
        ema_decay: EMA 衰减系数
        ema_eps: EMA 平滑项
    """
    def __init__(
        self,
        patch_size,
        embedding_dim,
        num_embeddings,
        compression_factor,
        commitment_cost=0.25,
        num_hiddens=128,
        num_layers=3,
        dropout=0.1,
        use_ema=True,
        ema_decay=0.99,
        ema_eps=1e-5
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.compression_factor = compression_factor
        self.compressed_len = patch_size // compression_factor
        
        # Encoder
        self.encoder = MLPEncoder(
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            compression_factor=compression_factor,
            num_hiddens=num_hiddens,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Vector Quantizer
        if use_ema:
            self.vq = VectorQuantizerEMAMLP(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=ema_decay,
                eps=ema_eps
            )
        else:
            self.vq = VectorQuantizerMLP(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost
            )
        
        # Decoder
        self.decoder = MLPDecoder(
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            compression_factor=compression_factor,
            num_hiddens=num_hiddens,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def encode(self, x):
        """编码"""
        return self.encoder(x, self.compression_factor)
    
    def quantize(self, z):
        """量化"""
        return self.vq(z)
    
    def decode(self, z_q):
        """解码"""
        return self.decoder(z_q, self.compression_factor)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, patch_size] 或 [B, 1, patch_size]
        
        Returns:
            x_recon: [B, patch_size]
            vq_loss: VQ loss
            perplexity: codebook perplexity
            indices: encoding indices
        """
        # Encode
        z = self.encode(x)  # [B, embedding_dim, compressed_len]
        
        # Quantize
        vq_loss, z_q, perplexity, _, encoding_indices, _ = self.quantize(z)
        
        # Decode
        x_recon = self.decode(z_q)  # [B, patch_size]
        
        return x_recon, vq_loss, perplexity, encoding_indices
    
    def get_codebook_indices(self, x):
        """获取码本索引"""
        z = self.encode(x)
        _, _, _, _, encoding_indices, _ = self.quantize(z)
        return encoding_indices
    
    def decode_from_indices(self, indices):
        """从索引解码"""
        # indices: [B * compressed_len, 1]
        B_times_L = indices.shape[0]
        
        # 获取量化向量
        z_q = self.vq.embedding.weight[indices.squeeze(1)]  # [B * compressed_len, embedding_dim]
        
        # 推断 B 和 compressed_len
        z_q = z_q.view(-1, self.compressed_len, self.embedding_dim)  # [B, compressed_len, embedding_dim]
        z_q = z_q.permute(0, 2, 1)  # [B, embedding_dim, compressed_len]
        
        # 解码
        x_recon = self.decode(z_q)
        
        return x_recon


# 便捷函数：获取 MLP VQ-VAE 配置
def get_mlp_vqvae_config(args):
    """从 args 构建 MLP VQ-VAE 配置"""
    return {
        'patch_size': args.patch_size,
        'embedding_dim': args.embedding_dim,
        'num_embeddings': args.codebook_size,
        'compression_factor': args.compression_factor,
        'commitment_cost': args.commitment_cost,
        'num_hiddens': getattr(args, 'mlp_hiddens', 128),
        'num_layers': getattr(args, 'mlp_layers', 3),
        'dropout': getattr(args, 'mlp_dropout', 0.1),
        'use_ema': bool(getattr(args, 'codebook_ema', 1)),
        'ema_decay': getattr(args, 'ema_decay', 0.99),
        'ema_eps': getattr(args, 'ema_eps', 1e-5),
    }

