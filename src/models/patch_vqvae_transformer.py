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
    """展平的 Vector Quantizer"""
    def __init__(self, codebook_size, code_dim, commitment_cost=0.25, init_method='uniform'):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.init_method = init_method
        
        self.embedding = nn.Embedding(codebook_size, code_dim)
        
        if init_method == 'uniform':
            self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        elif init_method == 'normal':
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        elif init_method == 'xavier':
            nn.init.xavier_uniform_(self.embedding.weight)
        elif init_method == 'kaiming':
            nn.init.kaiming_uniform_(self.embedding.weight)
        else:
            self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
    
    def init_from_data(self, z_samples, method='kmeans', random_state=42):
        """
        从数据初始化码本（数据驱动初始化）
        
        Args:
            z_samples: [N, code_dim] encoder输出的样本
            method: 'kmeans' 或 'random_sample'
            random_state: 随机种子
        """
        if method == 'kmeans':
            try:
                from sklearn.cluster import KMeans
                z_np = z_samples.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=self.codebook_size, random_state=random_state, n_init=10, max_iter=300)
                kmeans.fit(z_np)
                centroids = torch.tensor(kmeans.cluster_centers_, dtype=z_samples.dtype, device=z_samples.device)
                self.embedding.weight.data.copy_(centroids)
                print(f"✓ 码本已从K-means聚类初始化 (codebook_size={self.codebook_size}, 样本数={len(z_samples)})")
            except ImportError:
                print("警告: sklearn未安装，使用随机采样初始化")
                method = 'random_sample'
        
        if method == 'random_sample':
            # 随机采样N个样本作为码本中心
            N = z_samples.size(0)
            if N >= self.codebook_size:
                perm = torch.randperm(N, generator=torch.Generator().manual_seed(random_state))[:self.codebook_size]
                centroids = z_samples[perm]
            else:
                # 如果样本数不足，使用重复采样
                indices = torch.randint(0, N, (self.codebook_size,), generator=torch.Generator().manual_seed(random_state))
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


class ResidualVectorQuantizer(nn.Module):
    """残差向量量化器：使用多层码本逐步拟合残差，减少量化误差"""
    def __init__(self, codebook_size, code_dim, commitment_cost=0.25, num_layers=2, init_method='uniform', combine_method='sum'):
        super().__init__()
        # codebook_size可以是整数（所有层相同）或列表（每层不同）
        if isinstance(codebook_size, (list, tuple)):
            if len(codebook_size) != num_layers:
                raise ValueError(f"codebook_size列表长度({len(codebook_size)})必须等于num_layers({num_layers})")
            self.codebook_sizes = list(codebook_size)
            self.codebook_size = codebook_size[0]  # 保留第一个作为默认值（用于兼容性）
        else:
            self.codebook_sizes = [codebook_size] * num_layers
            self.codebook_size = codebook_size
        
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.num_layers = num_layers
        self.combine_method = combine_method  # 'sum' 或 'concat'
        
        # 创建多层码本（每层可以使用不同的大小）
        self.codebooks = nn.ModuleList([
            FlattenedVectorQuantizer(self.codebook_sizes[i], code_dim, commitment_cost, init_method)
            for i in range(num_layers)
        ])
    
    def forward(self, z_flat):
        """
        Args:
            z_flat: [N, code_dim] 输入向量
        Returns:
            total_loss: 总损失（所有层的损失之和）
            quantized: [N, code_dim] 最终量化结果
            indices_list: List[[N]] 每层的码本索引
        """
        residual = z_flat
        total_loss = 0
        indices_list = []
        quantized_residuals_for_recon = []  # 收集STE-enabled的量化残差用于重建
        
        for codebook in self.codebooks:
            loss, quantized_residual_ste, indices = codebook(residual)
            total_loss += loss
            # 对于下一层，使用detached的残差以保持残差学习特性
            residual = residual - quantized_residual_ste.detach()
            indices_list.append(indices)
            # 保存STE-enabled的输出用于重建，确保梯度可以回传到encoder
            quantized_residuals_for_recon.append(quantized_residual_ste)
        
        # 最终的量化向量应该是所有STE-enabled量化残差的和
        # 这样recon_loss的梯度可以通过每层的STE路径回传到encoder
        quantized = sum(quantized_residuals_for_recon)
        
        return total_loss, quantized, indices_list
    
    def get_embedding(self, indices_list):
        """
        从多层索引获取量化向量
        
        Args:
            indices_list: List[[N]] 每层的码本索引
        Returns:
            quantized: [N, code_dim] (sum) 或 [N, code_dim * num_layers] (concat)
        """
        embeddings = [self.codebooks[i].get_embedding(indices_list[i]) for i in range(self.num_layers)]
        
        if self.combine_method == 'concat':
            quantized = torch.cat(embeddings, dim=1)  # [N, code_dim * num_layers]
        else:  # 'sum'
            quantized = embeddings[0]
            for emb in embeddings[1:]:
                quantized = quantized + emb
        
        return quantized


class ResidualVectorQuantizerEMA(nn.Module):
    """残差向量量化器（EMA版本）：使用EMA更新多层码本"""
    def __init__(self, codebook_size, code_dim, commitment_cost=0.25, decay=0.99, eps=1e-5, 
                 num_layers=2, init_method='uniform', combine_method='sum'):
        super().__init__()
        # codebook_size可以是整数（所有层相同）或列表（每层不同）
        if isinstance(codebook_size, (list, tuple)):
            if len(codebook_size) != num_layers:
                raise ValueError(f"codebook_size列表长度({len(codebook_size)})必须等于num_layers({num_layers})")
            self.codebook_sizes = list(codebook_size)
            self.codebook_size = codebook_size[0]  # 保留第一个作为默认值（用于兼容性）
        else:
            self.codebook_sizes = [codebook_size] * num_layers
            self.codebook_size = codebook_size
        
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        self.num_layers = num_layers
        self.combine_method = combine_method  # 'sum' 或 'concat'
        
        # 创建多层EMA码本（每层可以使用不同的大小）
        self.codebooks = nn.ModuleList([
            FlattenedVectorQuantizerEMA(self.codebook_sizes[i], code_dim, commitment_cost, decay, eps, init_method)
            for i in range(num_layers)
        ])
    
    def forward(self, z_flat):
        """
        Args:
            z_flat: [N, code_dim] 输入向量
        Returns:
            total_loss: 总损失（所有层的损失之和）
            quantized: [N, code_dim] 最终量化结果
            indices_list: List[[N]] 每层的码本索引
        """
        residual = z_flat
        total_loss = 0
        indices_list = []
        quantized_residuals_for_recon = []  # 收集STE-enabled的量化残差用于重建
        
        for codebook in self.codebooks:
            loss, quantized_residual_ste, indices = codebook(residual)
            total_loss += loss
            # 对于下一层，使用detached的残差以保持残差学习特性
            residual = residual - quantized_residual_ste.detach()
            indices_list.append(indices)
            # 保存STE-enabled的输出用于重建，确保梯度可以回传到encoder
            quantized_residuals_for_recon.append(quantized_residual_ste)
        
        # 最终的量化向量应该是所有STE-enabled量化残差的和
        # 这样recon_loss的梯度可以通过每层的STE路径回传到encoder
        quantized = sum(quantized_residuals_for_recon)
        
        return total_loss, quantized, indices_list
    
    def get_embedding(self, indices_list):
        """
        从多层索引获取量化向量
        
        Args:
            indices_list: List[[N]] 每层的码本索引
        Returns:
            quantized: [N, code_dim] (sum) 或 [N, code_dim * num_layers] (concat)
        """
        embeddings = [self.codebooks[i].get_embedding(indices_list[i]) for i in range(self.num_layers)]
        
        if self.combine_method == 'concat':
            quantized = torch.cat(embeddings, dim=1)  # [N, code_dim * num_layers]
        else:  # 'sum'
            quantized = embeddings[0]
            for emb in embeddings[1:]:
                quantized = quantized + emb
        
        return quantized


class FlattenedVectorQuantizerEMA(nn.Module):
    """使用 EMA 更新码本的 Vector Quantizer"""
    def __init__(self, codebook_size, code_dim, commitment_cost=0.25, decay=0.99, eps=1e-5, init_method='uniform'):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        self.init_method = init_method
        
        if init_method == 'uniform':
            embed = torch.randn(codebook_size, code_dim)
            embed.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        elif init_method == 'normal':
            embed = torch.randn(codebook_size, code_dim) * 0.02
        elif init_method == 'xavier':
            embed = torch.empty(codebook_size, code_dim)
            nn.init.xavier_uniform_(embed)
        elif init_method == 'kaiming':
            embed = torch.empty(codebook_size, code_dim)
            nn.init.kaiming_uniform_(embed)
        else:
            embed = torch.randn(codebook_size, code_dim)
            embed.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        
        self.embedding = nn.Embedding(codebook_size, code_dim)
        self.embedding.weight.data.copy_(embed)
        self.embedding.weight.requires_grad = False
        
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_w', embed.clone())
        self._disable_ema_update = False
    
    def init_from_data(self, z_samples, method='kmeans', random_state=42):
        """
        从数据初始化码本（数据驱动初始化）
        
        Args:
            z_samples: [N, code_dim] encoder输出的样本
            method: 'kmeans' 或 'random_sample'
            random_state: 随机种子
        """
        if method == 'kmeans':
            try:
                from sklearn.cluster import KMeans
                z_np = z_samples.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=self.codebook_size, random_state=random_state, n_init=10, max_iter=300)
                kmeans.fit(z_np)
                centroids = torch.tensor(kmeans.cluster_centers_, dtype=z_samples.dtype, device=z_samples.device)
                self.embedding.weight.data.copy_(centroids)
                self.ema_w.data.copy_(centroids)
                print(f"✓ 码本已从K-means聚类初始化 (codebook_size={self.codebook_size}, 样本数={len(z_samples)})")
            except ImportError:
                print("警告: sklearn未安装，使用随机采样初始化")
                method = 'random_sample'
        
        if method == 'random_sample':
            N = z_samples.size(0)
            if N >= self.codebook_size:
                perm = torch.randperm(N, generator=torch.Generator().manual_seed(random_state))[:self.codebook_size]
                centroids = z_samples[perm]
            else:
                indices = torch.randint(0, N, (self.codebook_size,), generator=torch.Generator().manual_seed(random_state))
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
        
        # VQ损失：包含commitment loss和codebook loss
        # commitment loss: 让encoder输出接近quantized（codebook更新通过EMA，不需要梯度）
        e_latent_loss = F.mse_loss(quantized.detach(), z_flat)
        # codebook loss: 让quantized接近encoder输出（通过straight-through estimator传递梯度）
        q_latent_loss = F.mse_loss(quantized, z_flat.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = z_flat + (quantized - z_flat).detach()
        return loss, quantized, indices
    
    def get_embedding(self, indices):
        return self.embedding(indices)


class CausalTransformer(nn.Module):
    """轻量级 Causal Transformer，支持独立的 hidden_dim 参数"""
    def __init__(self, code_dim, n_heads, n_layers, d_ff, dropout=0.1, max_len=512, hidden_dim=None):
        super().__init__()
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
        
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask, is_causal=True)
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
        
        # VQ (码本维度 = code_dim)
        vq_init_method = config.get('vq_init_method', 'uniform')
        use_residual_vq = config.get('use_residual_vq', False)
        residual_vq_layers = config.get('residual_vq_layers', 2)
        residual_vq_combine_method = config.get('residual_vq_combine_method', 'sum')  # 'sum' 或 'concat'
        # 支持每层不同的codebook大小：可以是整数（所有层相同）或列表（每层不同）
        residual_vq_codebook_sizes = config.get('residual_vq_codebook_sizes', None)
        if residual_vq_codebook_sizes is None:
            residual_vq_codebook_sizes = self.codebook_size  # 默认使用统一的codebook_size
        elif isinstance(residual_vq_codebook_sizes, str):
            # 如果是字符串（如 "256,128"），解析为列表
            residual_vq_codebook_sizes = [int(x.strip()) for x in residual_vq_codebook_sizes.split(',')]
        
        if use_residual_vq:
            # 使用残差量化（多层码本）
            if self.use_codebook_ema:
                self.vq = ResidualVectorQuantizerEMA(
                    residual_vq_codebook_sizes, self.code_dim, self.commitment_cost,
                    decay=self.ema_decay, eps=self.ema_eps,
                    num_layers=residual_vq_layers, init_method=vq_init_method,
                    combine_method=residual_vq_combine_method
                )
            else:
                self.vq = ResidualVectorQuantizer(
                    residual_vq_codebook_sizes, self.code_dim, self.commitment_cost,
                    num_layers=residual_vq_layers, init_method=vq_init_method,
                    combine_method=residual_vq_combine_method
                )
        else:
            # 使用单层量化（原始方式）
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
        
        # Transformer 和输出头
        # 如果使用残差量化，为每一层创建独立的transformer和output_head
        if use_residual_vq:
            self.use_multi_layer_transformer = True
            self.residual_vq_layers = residual_vq_layers
            
            # 获取每一层的codebook大小
            if isinstance(residual_vq_codebook_sizes, (list, tuple)):
                codebook_sizes_list = residual_vq_codebook_sizes
            else:
                codebook_sizes_list = [residual_vq_codebook_sizes] * residual_vq_layers
            
            # 为每一层创建独立的transformer和output_head
            self.transformers = nn.ModuleList([
                CausalTransformer(
                    self.code_dim, self.n_heads, self.n_layers,
                    self.d_ff, self.dropout, hidden_dim=self.transformer_hidden_dim
                ) for _ in range(residual_vq_layers)
            ])
            
            self.output_heads = nn.ModuleList([
                nn.Linear(self.code_dim, codebook_sizes_list[i])
                for i in range(residual_vq_layers)
            ])
            
            # 为了兼容性，保留单层的transformer和output_head（使用第一层）
            self.transformer = self.transformers[0]
            self.output_head = self.output_heads[0]
        else:
            self.use_multi_layer_transformer = False
            self.residual_vq_layers = 1
            
            # Transformer (输入维度 = code_dim，内部维度 = transformer_hidden_dim)
            self.transformer = CausalTransformer(
                self.code_dim, self.n_heads, self.n_layers,
                self.d_ff, self.dropout, hidden_dim=self.transformer_hidden_dim
            )
            
            # 输出头: code_dim -> codebook_size (预测码本索引)
            self.output_head = nn.Linear(self.code_dim, self.codebook_size)

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
    
    def encode_to_indices(self, x):
        """
        编码为码本索引和量化向量（channel-independent版本）
        
        Args:
            x: [B, T, C]
        Returns:
            indices: [B, num_patches, C, num_layers] 或 [B, num_patches, C] (单层)
            vq_loss: scalar
            z_q: [B, num_patches, C, code_dim]
        """
        B, T, C = x.shape
        
        num_patches = T // self.patch_size
        
        x = x[:, :num_patches * self.patch_size, :]
        x_patches = x.reshape(B, num_patches, self.patch_size, C)
        
        indices_list = []
        z_q_list = []
        vq_loss_sum = 0
        
        for c in range(C):
            x_c = x_patches[:, :, :, c]
            x_c_flat = x_c.reshape(B * num_patches, self.patch_size).unsqueeze(1)
            
            z = self.encoder(x_c_flat, self.compression_factor)
            z_flat = z.reshape(B * num_patches, -1)
            
            vq_result = self.vq(z_flat)
            if isinstance(vq_result[2], list):
                vq_loss_c, z_q_flat_c, indices_list_c = vq_result
                indices_c = torch.stack(indices_list_c, dim=0).t()
            else:
                vq_loss_c, z_q_flat_c, indices_c = vq_result
                indices_c = indices_c.unsqueeze(1)
            
            vq_loss_sum += vq_loss_c
            num_layers = indices_c.shape[1]
            indices_c = indices_c.reshape(B, num_patches, num_layers)
            z_q_c = z_q_flat_c.reshape(B, num_patches, self.code_dim)
            
            indices_list.append(indices_c)
            z_q_list.append(z_q_c)
        
        indices = torch.stack(indices_list, dim=2)
        z_q = torch.stack(z_q_list, dim=2)
        vq_loss = vq_loss_sum / C
        
        return indices, vq_loss, z_q
    
    def decode_from_indices(self, indices):
        """
        从码本索引恢复量化向量（支持单层和残差量化）
        
        Args:
            indices: [B, num_patches, C, num_layers] 或 [B, num_patches, C] (单层)
        Returns:
            z_q: [B, num_patches, C, code_dim] (sum模式) 或 [B, num_patches, C, code_dim * num_layers] (concat模式)
        """
        B, num_patches, C = indices.shape[:3]
        
        # 检查是否为残差量化（多层索引）
        if indices.dim() == 4:
            num_layers = indices.shape[3]
            use_residual = num_layers > 1
        else:
            num_layers = 1
            use_residual = False
        
        z_q_list = []
        
        for c in range(C):
            indices_c = indices[:, :, c, :] if use_residual else indices[:, :, c].unsqueeze(-1)  # [B, num_patches, num_layers]
            indices_c_flat = indices_c.reshape(B * num_patches, num_layers)  # [B*num_patches, num_layers]
            
            # 从多层索引恢复量化向量
            if use_residual and hasattr(self.vq, 'get_embedding'):
                # 残差量化：需要将多层索引转换为列表
                indices_list_c = [indices_c_flat[:, i] for i in range(num_layers)]
                z_q_flat_c = self.vq.get_embedding(indices_list_c)
                # 如果使用concat模式，维度是 [B*num_patches, code_dim * num_layers]
                # 如果使用sum模式，维度是 [B*num_patches, code_dim]
                output_dim = z_q_flat_c.shape[1]
            else:
                # 单层量化：直接使用第一层索引
                z_q_flat_c = self.vq.get_embedding(indices_c_flat[:, 0])
                output_dim = self.code_dim
            
            z_q_c = z_q_flat_c.reshape(B, num_patches, output_dim)
            z_q_list.append(z_q_c)
        
        # 合并所有通道: [B, num_patches, C, code_dim]
        z_q = torch.stack(z_q_list, dim=2)
        return z_q
    
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
    
    def forward_pretrain(self, x, target):
        """
        预训练: 基于input的index预测target的index概率分布 (channel-independent版本)
        
        Args:
            x: [B, input_len, C] 输入序列
            target: [B, target_len, C] 目标序列
        Returns:
            logits: [B, num_target_patches, C, codebook_size] (单层) 或 [B, num_target_patches, C, num_layers, codebook_size] (残差量化多层)
            target_indices: [B, num_target_patches, C] (单层) 或 [B, num_target_patches, C, num_layers] (残差量化多层)
            vq_loss: VQ损失
            recon_loss: 重构损失
        """
        B, input_len, C = x.shape
        _, target_len, C_target = target.shape
        assert C == C_target, "输入和目标序列的通道数必须相同"
        
        # 1. 编码输入序列
        input_indices, vq_loss_input, z_q_input = self.encode_to_indices(x)
        
        # 2. 编码目标序列（用于计算损失）
        target_indices, vq_loss_target, z_q_target = self.encode_to_indices(target)
        
        # VQ损失（输入和目标序列的平均）
        vq_loss = (vq_loss_input + vq_loss_target) / 2
        
        # 重构损失（仅使用输入序列）
        num_input_patches = input_indices.shape[1]
        x_recon = self.decode_from_codes(z_q_input)  # [B, num_input_patches * patch_size, C]
        recon_loss = F.mse_loss(x_recon, x[:, :x_recon.shape[1], :])
        
        # 3. Channel-independent Transformer处理
        B, num_input_patches, C, code_dim = z_q_input.shape
        z_q_input_flat = z_q_input.permute(0, 2, 1, 3).reshape(B * C, num_input_patches, code_dim)
        
        # 4. 创建占位符并拼接
        num_target_patches = target_indices.shape[1]
        placeholder = torch.zeros(B * C, num_target_patches, code_dim, device=z_q_input_flat.device, dtype=z_q_input_flat.dtype)
        full_sequence = torch.cat([z_q_input_flat, placeholder], dim=1)
        
        # 5. Transformer处理并预测
        if self.use_multi_layer_transformer and target_indices.dim() == 4:
            # 残差量化：为每一层分别预测
            num_layers = target_indices.shape[3]
            logits_list = []
            target_indices_list = []
            
            for layer_idx in range(num_layers):
                # 使用对应层的transformer和output_head
                transformer = self.transformers[layer_idx]
                output_head = self.output_heads[layer_idx]
                
                # Transformer处理
                h_full = transformer(full_sequence)
                h_target = h_full[:, num_input_patches:, :]
                
                # 预测该层的logits
                logits_flat = output_head(h_target)  # [B*C, num_target_patches, codebook_size_i]
                logits = logits_flat.reshape(B, C, num_target_patches, -1).permute(0, 2, 1, 3)  # [B, num_target_patches, C, codebook_size_i]
                logits_list.append(logits)
                
                # 提取该层的target indices
                target_indices_layer = target_indices[:, :, :, layer_idx]  # [B, num_target_patches, C]
                target_indices_list.append(target_indices_layer)
            
            # 合并所有层的logits: [B, num_target_patches, C, num_layers, codebook_size_i]
            # 注意：每层的codebook_size可能不同，所以不能直接stack
            # 返回logits_list和target_indices_list，让训练代码处理
            # 为了兼容性，将logits_list stack成一个tensor（如果所有层codebook_size相同）
            # 否则返回列表
            if len(set([logits.shape[-1] for logits in logits_list])) == 1:
                # 所有层codebook_size相同，可以stack
                logits = torch.stack(logits_list, dim=3)  # [B, num_target_patches, C, num_layers, codebook_size]
            else:
                # codebook_size不同，返回列表
                logits = logits_list
            
            # target_indices已经是 [B, num_target_patches, C, num_layers]
            target_indices_flat = target_indices
        else:
            # 单层量化：使用单个transformer和output_head
            h_full = self.transformer(full_sequence)
            h_target = h_full[:, num_input_patches:, :]
            logits_flat = self.output_head(h_target)
            logits = logits_flat.reshape(B, C, num_target_patches, -1).permute(0, 2, 1, 3)
            
            # 对于残差量化但只使用第一层的情况
            if target_indices.dim() == 4:
                target_indices_flat = target_indices[:, :, :, 0]
            else:
                target_indices_flat = target_indices
        
        return logits, target_indices_flat, vq_loss, recon_loss
    
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
        
        # Transformer处理并预测
        if self.use_multi_layer_transformer:
            # 残差量化：为每一层分别预测，然后合并
            pred_codes_list = []
            
            for layer_idx in range(self.residual_vq_layers):
                transformer = self.transformers[layer_idx]
                output_head = self.output_heads[layer_idx]
                
                # Transformer处理
                h_full = transformer(full_sequence)
                h_pred = h_full[:, num_input_patches:, :]  # [B*C, num_pred_patches, code_dim]
                
                # 输出头预测
                logits = output_head(h_pred)  # [B*C, num_pred_patches, codebook_size_i]
                
                # 使用 softmax + 加权求和 替代 argmax，保持可微分
                weights = F.softmax(logits, dim=-1)  # [B*C, num_pred_patches, codebook_size_i]
                # 获取对应层的codebook
                codebook = self.vq.codebooks[layer_idx].embedding.weight  # [codebook_size_i, code_dim]
                pred_codes_layer = torch.matmul(weights, codebook)  # [B*C, num_pred_patches, code_dim]
                pred_codes_list.append(pred_codes_layer)
            
            # 合并所有层的预测（相加）
            pred_codes = sum(pred_codes_list)  # [B*C, num_pred_patches, code_dim]
            
            # Reshape回通道分离格式: [B*C, num_pred_patches, code_dim] -> [B, num_pred_patches, C, code_dim]
            pred_codes = pred_codes.reshape(B, C, num_pred_patches, code_dim).permute(0, 2, 1, 3)
        else:
            # 单层量化
            h_full = self.transformer(full_sequence)
            h_pred = h_full[:, num_input_patches:, :]
            logits = self.output_head(h_pred)
            
            # 使用 softmax + 加权求和 替代 argmax，保持可微分
            weights = F.softmax(logits, dim=-1)
            codebook = self.vq.embedding.weight
            pred_codes = torch.matmul(weights, codebook)
            
            # Reshape回通道分离格式
            pred_codes = pred_codes.reshape(B, C, num_pred_patches, code_dim).permute(0, 2, 1, 3)
        
        # 解码
        pred = self.decode_from_codes(pred_codes)  # [B, num_pred_patches*patch_size, C]
        pred = pred[:, :target_len, :]  # [B, target_len, C]
        
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
        result = self.encode_to_indices(x, return_processed_patches=False)
        if isinstance(result, tuple) and len(result) == 4:
            indices, _, _, _ = result
        else:
            indices, _, _ = result
        # indices: [B, num_patches, C, num_layers] 或 [B, num_patches, C]
        # 对于残差量化，只使用第一层索引计算利用率
        if indices.dim() == 4:
            indices = indices[:, :, :, 0]  # [B, num_patches, C]
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
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
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
            if try_load_module(self.encoder, 'Encoder', 'encoder_state_dict', 'encoder'):
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
        # 残差量化配置
        'use_residual_vq': getattr(args, 'use_residual_vq', False),
        'residual_vq_layers': getattr(args, 'residual_vq_layers', 2),
        'vq_init_method': getattr(args, 'vq_init_method', 'uniform'),
        'residual_vq_combine_method': getattr(args, 'residual_vq_combine_method', 'sum'),  # 'sum' 或 'concat'
        'residual_vq_codebook_sizes': getattr(args, 'residual_vq_codebook_sizes', None),  # 每层codebook大小，如 "256,128"
    }
    return config
