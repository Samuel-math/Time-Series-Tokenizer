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

try:
    from .vqvae import build_encoder, build_decoder
except ImportError:
    from src.models.vqvae import build_encoder, build_decoder


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


class ResidualVQ(nn.Module):
    """
    残差向量量化（Residual Vector Quantization，RVQ）
    逐层量化上一层的残差，每层有独立码本。
    n_rq_layers=1 时退化为普通 VQ，与原有行为完全一致。
    """
    def __init__(self, n_layers, make_single_vq_fn):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([make_single_vq_fn() for _ in range(n_layers)])

    @property
    def codebook_size(self):
        return self.layers[0].codebook_size

    @property
    def code_dim(self):
        return self.layers[0].code_dim

    # 为兼容旧代码中 `vq.embedding.weight` 的访问（n_rq_layers=1 时）
    @property
    def embedding(self):
        return self.layers[0].embedding

    def forward(self, z_flat, return_per_layer=False):
        """
        Args:
            z_flat: [N, code_dim]
            return_per_layer: 若为 True，额外返回 per_layer_z_q: List[L] of [N, code_dim]
        Returns:
            total_loss: scalar
            z_q_sum: [N, code_dim]  各层量化之和
            all_indices: List[[N]]  每层的索引
            per_layer_z_q (optional): List[L] of [N, code_dim]，每层单独输出的量化向量
        """
        residual = z_flat
        z_q_sum = torch.zeros_like(z_flat)
        total_loss = z_flat.new_tensor(0.0)
        all_indices = []
        per_layer_z_q = [] if return_per_layer else None

        for vq in self.layers:
            loss, z_q, idx = vq(residual)
            residual = residual - z_q.detach()
            z_q_sum = z_q_sum + z_q
            total_loss = total_loss + loss
            all_indices.append(idx)
            if return_per_layer:
                per_layer_z_q.append(z_q)

        if return_per_layer:
            return total_loss, z_q_sum, all_indices, per_layer_z_q
        return total_loss, z_q_sum, all_indices

    def quantize_separate(self, z_parts, return_per_layer=False):
        """
        显式分量量化：z_parts[i] 直接交给第 i 层码本。
        用于 low/high 频率拆分；n_layers=1 或未传够分量时仍可退化到残差量化。
        """
        if len(z_parts) == 0:
            raise ValueError("z_parts must contain at least one tensor")

        z_ref = torch.stack(z_parts, dim=0).sum(dim=0)
        z_q_sum = torch.zeros_like(z_ref)
        total_loss = z_ref.new_tensor(0.0)
        all_indices = []
        per_layer_z_q = [] if return_per_layer else None

        n_direct = min(len(z_parts), self.n_layers)
        for i in range(n_direct):
            loss, z_q, idx = self.layers[i](z_parts[i])
            z_q_sum = z_q_sum + z_q
            total_loss = total_loss + loss
            all_indices.append(idx)
            if return_per_layer:
                per_layer_z_q.append(z_q)

        residual = z_ref - z_q_sum.detach()
        for i in range(n_direct, self.n_layers):
            loss, z_q, idx = self.layers[i](residual)
            residual = residual - z_q.detach()
            z_q_sum = z_q_sum + z_q
            total_loss = total_loss + loss
            all_indices.append(idx)
            if return_per_layer:
                per_layer_z_q.append(z_q)

        if return_per_layer:
            return total_loss, z_q_sum, all_indices, per_layer_z_q
        return total_loss, z_q_sum, all_indices

    def get_embedding(self, indices_list):
        """
        indices_list: List[[N]] (length = n_rq_layers)
        Returns: [N, code_dim]，各层 embedding 之和
        """
        dtype = self.layers[0].embedding.weight.dtype
        z_q = torch.zeros(indices_list[0].shape[0], self.code_dim,
                          device=indices_list[0].device, dtype=dtype)
        for i, idx in enumerate(indices_list):
            z_q = z_q + self.layers[i].get_embedding(idx)
        return z_q

    def init_from_data(self, z_samples, method='kmeans'):
        """逐层用残差初始化码本"""
        residual = z_samples.detach().clone()
        for layer in self.layers:
            layer.init_from_data(residual, method)
            with torch.no_grad():
                distances = (
                    torch.sum(residual ** 2, dim=1, keepdim=True)
                    + torch.sum(layer.embedding.weight ** 2, dim=1)
                    - 2 * torch.matmul(residual, layer.embedding.weight.t())
                )
                idx = torch.argmin(distances, dim=1)
                z_q = layer.embedding(idx)
                residual = residual - z_q

    def init_from_data_separate(self, z_samples_parts, method='kmeans'):
        """按分量初始化码本；额外层继续初始化总表示残差。"""
        if len(z_samples_parts) == 0:
            raise ValueError("z_samples_parts must contain at least one tensor")

        z_ref = torch.stack([z.detach() for z in z_samples_parts], dim=0).sum(dim=0)
        z_q_sum = torch.zeros_like(z_ref)
        n_direct = min(len(z_samples_parts), self.n_layers)

        for i in range(n_direct):
            layer = self.layers[i]
            layer.init_from_data(z_samples_parts[i], method)
            with torch.no_grad():
                distances = (
                    torch.sum(z_samples_parts[i] ** 2, dim=1, keepdim=True)
                    + torch.sum(layer.embedding.weight ** 2, dim=1)
                    - 2 * torch.matmul(z_samples_parts[i], layer.embedding.weight.t())
                )
                idx = torch.argmin(distances, dim=1)
                z_q_sum = z_q_sum + layer.embedding(idx)

        residual = z_ref - z_q_sum
        for i in range(n_direct, self.n_layers):
            layer = self.layers[i]
            layer.init_from_data(residual, method)
            with torch.no_grad():
                distances = (
                    torch.sum(residual ** 2, dim=1, keepdim=True)
                    + torch.sum(layer.embedding.weight ** 2, dim=1)
                    - 2 * torch.matmul(residual, layer.embedding.weight.t())
                )
                idx = torch.argmin(distances, dim=1)
                residual = residual - layer.embedding(idx)


class CausalTransformer(nn.Module):
    """轻量级 Causal Transformer，支持独立的 hidden_dim 参数

    性能说明：
        使用 is_causal=True 让 attention 走 fused SDP 路径
        （flash / memory-efficient attention），比显式 triu 布尔 mask 快 2~4×。
        语义上与三角 mask 完全等价。
    """
    def __init__(self, code_dim, n_heads, n_layers, d_ff, dropout=0.1, max_len=512, hidden_dim=None):
        super().__init__()

        self.code_dim = code_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else code_dim

        if self.hidden_dim != self.code_dim:
            self.input_proj  = nn.Linear(self.code_dim, self.hidden_dim)
            self.output_proj = nn.Linear(self.hidden_dim, self.code_dim)
        else:
            self.input_proj = None
            self.output_proj = None

        self.pos_embedding = nn.Embedding(max_len, self.hidden_dim)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        """
        Args:
            x: [B, T, code_dim]
        Returns:
            output: [B, T, code_dim]
        """
        B, T, _ = x.shape

        if self.input_proj is not None:
            x = self.input_proj(x)

        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)
        x = self.drop(x)

        # 同时给 mask 和 is_causal=True：
        # - fast-path（推理 / 无 dropout 等满足条件）→ 走 fused SDP（flash / mem-efficient attention）
        # - slow-path（训练 + dropout / 旧内部实现）→ 用 mask 兜底，避免
        #   "Need attn_mask if specifying the is_causal hint" 的报错
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        if self.output_proj is not None:
            x = self.output_proj(x)
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
        
        # 微调阶段的码本分布配置。
        # 历史参数名保留为 use_gumbel_softmax/gumbel_temperature，但这里不再采样 Gumbel 噪声；
        # use_gumbel_softmax=True 表示使用 temperature softmax 做确定性 sharpen。
        self.use_gumbel_softmax = config.get('use_gumbel_softmax', True)
        self.gumbel_temperature = config.get('gumbel_temperature', 1.0)
        self.gumbel_hard = config.get('gumbel_hard', False)
        
        # code_dim = embedding_dim * compressed_len
        # Channel-independent: 每个通道独立处理，使用单通道Encoder/Decoder
        self.compressed_len = self.patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len  # Transformer 输入维度
        
        # Transformer的hidden_dim（用于Transformer内部维度，默认使用code_dim）
        self.transformer_hidden_dim = config.get('transformer_hidden_dim', None)
        
        # VQ (码本维度 = code_dim)
        vq_init_method = config.get('vq_init_method', 'random')
        self.per_channel_codebook = config.get('per_channel_codebook', False)
        self.n_rq_layers = config.get('n_rq_layers', 1)
        n_channels = config.get('n_channels', None)
        self._n_channels = n_channels

        def _make_single_vq():
            if self.use_codebook_ema:
                return FlattenedVectorQuantizerEMA(
                    self.codebook_size, self.code_dim, self.commitment_cost,
                    decay=self.ema_decay, eps=self.ema_eps,
                    init_method=vq_init_method
                )
            return FlattenedVectorQuantizer(
                self.codebook_size, self.code_dim, self.commitment_cost,
                init_method=vq_init_method
            )

        def _make_rvq():
            return ResidualVQ(self.n_rq_layers, _make_single_vq)

        if self.per_channel_codebook:
            if n_channels is None:
                raise ValueError("per_channel_codebook=True 需要在 config 中提供 n_channels")
            self.vqs = nn.ModuleList([_make_rvq() for _ in range(n_channels)])
        else:
            self.vq = _make_rvq()

        # Transformer (输入维度 = code_dim，内部维度 = transformer_hidden_dim)
        self.transformer = CausalTransformer(
            self.code_dim, self.n_heads, self.n_layers,
            self.d_ff, self.dropout, hidden_dim=self.transformer_hidden_dim
        )

        # 每层 RVQ 独立一个预测头；n_rq_layers=1 时等价于原来的 output_head
        self.output_heads = nn.ModuleList([
            nn.Linear(self.code_dim, self.codebook_size) for _ in range(self.n_rq_layers)
        ])

        # NMPP 模式 (use_raw_input=True): 将原始 patch 投影到 Transformer 输入维度
        self.patch_embedding = nn.Linear(self.patch_size, self.code_dim)

        # Forecasting 时 overlap 聚合用的 per-coverer-rank gate（方案 X4）。
        # 对绝对位置 p，被多个 chunk 覆盖；按 "新 → 旧" 排序后，rank k = floor(offset / step_size)：
        #   k=0 : 最新的 coverer（当前 chunk fresh 区，offset ∈ [0, M)）
        #   k=1 : 上一个 chunk 的覆盖（offset ∈ [M, 2M)）
        #   k=j : offset ∈ [jM, (j+1)M) 的覆盖
        # 每个 rank 一个可学 sigmoid 权重，参数量 = ceil(N/M) 上限 = chunk_gate_max_rank。
        # 初始化全 0 → sigmoid(0)=0.5 → 起始行为等价于等权平均（向后兼容）。
        self.chunk_gate_max_rank = config.get('chunk_gate_max_rank', 16)
        self.chunk_gate_logits = nn.Parameter(torch.zeros(self.chunk_gate_max_rank))

        # Channel-independent: 每个通道独立处理，使用单通道Encoder/Decoder
        self.encoder = build_encoder(config, in_channels=1)
        self.decoder = build_decoder(config, in_channels=self.embedding_dim, out_channels=1)
        
        # Channel Attention 已移除
    
    def _get_vq(self, c: int):
        """返回通道 c 对应的 VQ 模块（per_channel_codebook=True 时每通道独立，否则共享）"""
        return self.vqs[c] if self.per_channel_codebook else self.vq

    @property
    def uses_frequency_codebooks(self):
        return self.n_rq_layers == 2

    def _split_low_high(self, x_c):
        """固定 moving-average 低通 + 残差高频，不引入额外参数。"""
        kernel = min(5, self.patch_size)
        if kernel % 2 == 0:
            kernel -= 1
        if kernel <= 1:
            low = x_c
        else:
            low = F.avg_pool1d(
                x_c, kernel_size=kernel, stride=1,
                padding=kernel // 2, count_include_pad=False,
            )
        return low, x_c - low

    def _encode_frequency_parts(self, x_c):
        low, high = self._split_low_high(x_c)
        return [
            self.encoder(low, self.compression_factor).reshape(x_c.shape[0], self.code_dim),
            self.encoder(high, self.compression_factor).reshape(x_c.shape[0], self.code_dim),
        ]

    def _raw_patch_embedding_bc(self, x, B, num_patches, C):
        """原始 patch 线性投影，输出 [B*C, num_patches, code_dim]（NMPP 模式使用）"""
        x = x[:, :num_patches * self.patch_size, :]
        # [B, num_patches, patch_size, C] -> [B, C, num_patches, patch_size]
        x_patches = x.reshape(B, num_patches, self.patch_size, C).permute(0, 3, 1, 2)
        flat = x_patches.reshape(B * C, num_patches, self.patch_size)
        return self.patch_embedding(flat)  # [B*C, num_patches, code_dim]

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
        
        # Channel-independent: 对每个通道独立编码 + 量化
        indices_list = []
        z_q_list = []
        vq_loss_sum = 0

        for c in range(C):
            # 提取第c个通道的patches: [B, num_patches, patch_size]
            x_c = x_patches[:, :, :, c]  # [B, num_patches, patch_size]
            x_c_flat = x_c.reshape(B * num_patches, self.patch_size)  # [B*num_patches, patch_size]
            x_c_flat = x_c_flat.unsqueeze(1)  # [B*num_patches, 1, patch_size] (单通道输入)

            # RVQ（每通道独立码本或共享码本）
            if self.uses_frequency_codebooks:
                z_parts = self._encode_frequency_parts(x_c_flat)
                vq_loss_c, z_q_sum_c, all_idx_c = self._get_vq(c).quantize_separate(z_parts)
            else:
                z = self.encoder(x_c_flat, self.compression_factor)
                z_c_flat = z.reshape(B * num_patches, self.code_dim)
                vq_loss_c, z_q_sum_c, all_idx_c = self._get_vq(c)(z_c_flat)
            vq_loss_sum += vq_loss_c

            # all_idx_c: List[n_rq_layers] of [B*num_patches]
            # Stack → [B*num_patches, n_rq_layers] → [B, num_patches, n_rq_layers]
            indices_c = torch.stack(all_idx_c, dim=1).reshape(B, num_patches, self.n_rq_layers)
            z_q_c = z_q_sum_c.reshape(B, num_patches, self.code_dim)

            indices_list.append(indices_c)
            z_q_list.append(z_q_c)

        # indices: [B, num_patches, C, n_rq_layers]
        # z_q:     [B, num_patches, C, code_dim]
        indices = torch.stack(indices_list, dim=2)
        z_q = torch.stack(z_q_list, dim=2)
        vq_loss = vq_loss_sum / C

        return indices, vq_loss, z_q
    
    def decode_from_codes(self, z_q):
        """
        从量化向量解码（channel-independent版本，批量化优化）
        
        Args:
            z_q: [B, num_patches, C, code_dim]
        Returns:
            x_recon: [B, num_patches * patch_size, C]
        """
        B, num_patches, C, code_dim = z_q.shape
        
        # ============ 批量化优化：将所有通道合并为一个大batch ============
        # [B, num_patches, C, code_dim] -> [B*C, num_patches, code_dim]
        z_q_flat = z_q.permute(0, 2, 1, 3).reshape(B * C, num_patches, code_dim)
        
        # Reshape for decoder: [B*C*num_patches, embedding_dim, compressed_len]
        z_q_for_decoder = z_q_flat.reshape(B * C * num_patches, self.embedding_dim, self.compressed_len)
        
        # 单次decoder调用（关键优化点！）
        x_recon_flat = self.decoder(z_q_for_decoder, self.compression_factor)  # [B*C*num_patches, patch_size]
        
        # Reshape回原始格式
        # [B*C*num_patches, patch_size] -> [B, C, num_patches, patch_size]
        x_recon_reshaped = x_recon_flat.reshape(B, C, num_patches, self.patch_size)
        
        # [B, C, num_patches, patch_size] -> [B, num_patches, patch_size, C] -> [B, num_patches*patch_size, C]
        x_recon = x_recon_reshaped.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        return x_recon
    
    def forward_progressive_pretrain(self, x_full, step_size, max_stages=None,
                                      compute_recon_loss=True, use_raw_input=False,
                                      pred_len=None):
        """
        渐进式预训练：每 stage 的预测长度（pred_len=N）与推进步长（step_size=M）解耦。

        Stage s（1-indexed）：
          - context = patches[0 : s*M]
          - target  = patches[s*M : s*M + N]      ← 预测 context 之后的 N 个 patch
          - 当 M < N 时，相邻 stage 的预测区间重叠，重叠长度 = N - M

        当 pred_len = step_size（默认）时退化为原始不重叠逻辑，行为完全一致。

        Args:
            x_full:             [B, total_len, C]
            step_size:          M，推进步长（patches）
            max_stages:         最大阶段数，None 表示全部
            compute_recon_loss: 是否计算 VQ 重构损失
            use_raw_input:      NMPP 模式
            pred_len:           N，每 stage 预测的 patch 数；None 时等于 step_size

        Returns:
            all_logits:         List[stage] of List[rq_layer] of [B, N, C, K]
            all_target_indices: List[stage] of List[rq_layer] of [B, N, C]
            vq_loss:            scalar
            recon_loss:         scalar
        """
        # pred_len=None → 与 step_size 相同，保持后向兼容
        if pred_len is None:
            pred_len = step_size

        B, total_len, C = x_full.shape
        num_patches = total_len // self.patch_size
        x_full = x_full[:, :num_patches * self.patch_size, :]

        if use_raw_input:
            with torch.no_grad():
                full_indices, _, _ = self.encode_to_indices(x_full)
            vq_loss_full = x_full.new_tensor(0.0)
            seq_full = self._raw_patch_embedding_bc(x_full, B, num_patches, C)
        else:
            full_indices, vq_loss_full, z_q_full = self.encode_to_indices(x_full)
            B, num_patches, C, code_dim = z_q_full.shape
            seq_full = z_q_full.permute(0, 2, 1, 3).reshape(B * C, num_patches, code_dim)

        num_total_patches = full_indices.shape[1]

        # Stage s 需要 s*M + N <= num_total_patches
        if max_stages is None:
            max_stages = (num_total_patches - pred_len) // step_size
        else:
            max_stages = min(max_stages, (num_total_patches - pred_len) // step_size)

        if max_stages <= 0:
            raise ValueError(
                f"序列长度不足：总patches={num_total_patches}, "
                f"step_size={step_size}, pred_len={pred_len}, 无法创建任何阶段"
            )

        n_rq = len(self.output_heads)

        # ── 逐 stage for-loop 版本 ────────────────────────────────────────────
        # 对 stage s：
        #   context  = seq_full[:, 0 : s*M]
        #   placeholder = zeros(M_or_N_patches)，放在 [s*M : s*M + N]
        #   target   = full_indices[:, s*M : s*M + N]
        # 模型在零输入位置、仅凭 causal 上下文预测未来 N 个 token。
        all_logits, all_target_indices = [], []
        code_dim_inner = seq_full.shape[2]

        for stage in range(1, max_stages + 1):
            context_size = stage * step_size
            target_start = stage * step_size
            target_end   = target_start + pred_len

            if target_end > num_total_patches:
                break

            placeholder = seq_full.new_zeros(B * C, pred_len, code_dim_inner)
            full_sequence_stage = torch.cat(
                [seq_full[:, :context_size, :], placeholder], dim=1
            )  # [B*C, context_size + pred_len, D]

            h_full   = self.transformer(full_sequence_stage)
            h_target = h_full[:, context_size:context_size + pred_len, :]  # [B*C, N, hidden]

            target_indices_stage = full_indices[:, target_start:target_end, :, :]

            logits_layers, tgt_layers = [], []
            for l, head in enumerate(self.output_heads):
                logits_flat = head(h_target)                                      # [B*C, N, K]
                logits_l = logits_flat.reshape(B, C, pred_len, -1).permute(0, 2, 1, 3)  # [B, N, C, K]
                logits_layers.append(logits_l)
                tgt_layers.append(target_indices_stage[:, :, :, l])              # [B, N, C]

            all_logits.append(logits_layers)
            all_target_indices.append(tgt_layers)

        if not all_logits:
            raise ValueError(
                f"序列长度不足：总patches={num_total_patches}, "
                f"step_size={step_size}, pred_len={pred_len}, 无法创建任何阶段"
            )

        if compute_recon_loss and not use_raw_input:
            x_recon_full = self.decode_from_codes(z_q_full)
            recon_loss = F.mse_loss(x_recon_full, x_full[:, :x_recon_full.shape[1], :])
        else:
            recon_loss = x_full.new_tensor(0.0)

        return all_logits, all_target_indices, vq_loss_full, recon_loss
    
    def forward_finetune(self, x, target_len, step_size=None, use_raw_input=False,
                          pred_len=None, target=None, return_token_metrics=False):
        """
        微调: 预测未来序列（支持 overlapping chunk prediction + 概率级融合）

        Args:
            x:            [B, T, C]
            target_len:   int，目标预测时间步数
            step_size:    M，自回归步长（每步提交的 patch 数）；None 表示非自回归
            use_raw_input: 与 NMPP 预训练一致时设 True
            pred_len:     N，每次 Transformer forward 预测的 patch 数；
                          None 时等于 step_size（无重叠，退化为旧行为）
                          M < N 时产生 overlapping chunk，同一位置的多个 logit 在
                          argmax 之前做均值融合（概率级融合）
            target:       可选，[B, target_len, C]。提供时统计预测 token 与 target VQ id 的准确率
            return_token_metrics: True 时额外返回 token accuracy 诊断信息

        Returns:
            pred:    [B, target_len, C]
            vq_loss: scalar
            token_metrics: 可选 dict，包含 avg/layer token accuracy
        """
        B, T, C = x.shape
        num_pred_patches = (target_len + self.patch_size - 1) // self.patch_size
        num_input_patches = T // self.patch_size
        x_aligned = x[:, :num_input_patches * self.patch_size, :]

        target_indices = None
        if return_token_metrics and target is not None:
            with torch.no_grad():
                target_indices, _, _ = self.encode_to_indices(target)

        if use_raw_input:
            context_flat = self._raw_patch_embedding_bc(x_aligned, B, num_input_patches, C)
            vq_loss = x.new_tensor(0.0)
            code_dim = self.code_dim
        else:
            indices, vq_loss, z_q = self.encode_to_indices(x)
            num_input_patches = z_q.shape[1]
            B, num_patches, C, code_dim = z_q.shape
            context_flat = z_q.permute(0, 2, 1, 3).reshape(B * C, num_patches, code_dim)

        # ── 预先准备所有 RVQ 层的码本（修复：finetune 也走全部 L 层并累加，
        #     让 decoder 看到 z_q_sum，与 VQ-VAE 预训练时的输入分布保持一致）──
        n_rq = len(self.output_heads)
        if self.per_channel_codebook:
            # list[L] of [C, K, code_dim]
            stacked_codebooks_layers = [
                torch.stack(
                    [self.vqs[c].layers[l].embedding.weight for c in range(C)], dim=0
                )
                for l in range(n_rq)
            ]
        else:
            # list[L] of [K, code_dim]
            shared_codebooks_layers = [
                self.vq.layers[l].embedding.weight for l in range(n_rq)
            ]

        def _lookup_codebook_at_layer(weights_bc, l):
            """weights_bc: [B*C, P, K] → [B*C, P, code_dim]，查第 l 层 RVQ 码本"""
            if self.per_channel_codebook:
                w = weights_bc.reshape(B, C, -1, self.codebook_size)
                out = torch.einsum('bcpk,ckd->bcpd', w, stacked_codebooks_layers[l])
                return out.reshape(B * C, -1, code_dim)
            return torch.matmul(weights_bc, shared_codebooks_layers[l])

        def _lookup_hard_codebook_at_layer(indices_bc, l):
            """indices_bc: [B*C] → [B*C, 1, code_dim]，按 argmax id 查第 l 层 RVQ 码本"""
            if self.per_channel_codebook:
                idx = indices_bc.reshape(B, C)
                emb = stacked_codebooks_layers[l].unsqueeze(0).expand(B, -1, -1, -1)
                gather_idx = idx[:, :, None, None].expand(-1, -1, 1, code_dim)
                return emb.gather(dim=2, index=gather_idx).reshape(B * C, 1, code_dim)
            return shared_codebooks_layers[l][indices_bc].unsqueeze(1)

        def _softmax_or_gumbel(logits, keep_extra_dim=False):
            """对 logits 做确定性的 temperature softmax。
            logits 形状：[B*C, P, K] 或 [B*C, K]
            若最后一维前没有 P 维，使用 keep_extra_dim=True 会先 unsqueeze(1)。"""
            x = logits.unsqueeze(1) if keep_extra_dim else logits
            tau = self.gumbel_temperature if self.use_gumbel_softmax else 1.0
            tau = max(float(tau), 1e-6)
            return F.softmax(x / tau, dim=-1)

        def _decode_h_pred_all_layers(h_pred):
            """h_pred: [B*C, P, hidden] → z_q [B*C, P, code_dim]（累加所有 RVQ 层）"""
            z_q_total = None
            for l, head in enumerate(self.output_heads):
                logits_l = head(h_pred)                               # [B*C, P, K]
                w_l = _softmax_or_gumbel(logits_l)                    # [B*C, P, K]
                z_q_l = _lookup_codebook_at_layer(w_l, l)             # [B*C, P, code_dim]
                z_q_total = z_q_l if z_q_total is None else z_q_total + z_q_l
            return z_q_total

        def _decode_avg_logits_per_layer(avg_logits_list):
            """avg_logits_list: list[L] of [B*C, K] → [B*C, 1, code_dim]（累加所有层）"""
            z_q_total = None
            for l, avg_l in enumerate(avg_logits_list):
                w_l = _softmax_or_gumbel(avg_l, keep_extra_dim=True)  # [B*C, 1, K]
                z_q_l = _lookup_codebook_at_layer(w_l, l)             # [B*C, 1, code_dim]
                z_q_total = z_q_l if z_q_total is None else z_q_total + z_q_l
            return z_q_total

        def _decode_hard_logits_per_layer(logits_list):
            """list[L] of [B*C, K] → [B*C, 1, code_dim]，用于自回归 context 的 hard first-hit 回填"""
            z_q_total = None
            for l, logits_l in enumerate(logits_list):
                idx_l = logits_l.argmax(dim=-1)
                z_q_l = _lookup_hard_codebook_at_layer(idx_l, l)
                z_q_total = z_q_l if z_q_total is None else z_q_total + z_q_l
            return z_q_total

        # 如果没有指定 step_size 或 step_size >= num_pred_patches，使用非自回归模式
        if step_size is None or step_size >= num_pred_patches:
            # 非自回归：一次性预测所有 patches
            placeholder = torch.zeros(B * C, num_pred_patches, code_dim,
                                      device=context_flat.device, dtype=context_flat.dtype)
            full_sequence = torch.cat([context_flat, placeholder], dim=1)

            h_full = self.transformer(full_sequence)
            h_pred = h_full[:, num_input_patches:, :]
            all_pred_codes = _decode_h_pred_all_layers(h_pred)        # 累加所有 L 层
            if return_token_metrics:
                pred_idx_layers = [
                    head(h_pred).argmax(dim=-1) for head in self.output_heads
                ]  # list[L] of [B*C, P]
        else:
            # ── 批量自回归（支持 overlapping chunk prediction + per-coverer-rank gate）──
            # pred_len=None 时默认等于 step_size（无重叠，退化为旧逻辑）
            eff_pred_len = pred_len if pred_len is not None else step_size

            # 配置合法性检查：step_size (M) > pred_len (N) 会漏预测位置 → 禁止
            if step_size > eff_pred_len:
                raise ValueError(
                    f"非法配置：step_size(M)={step_size} > pred_len(N)={eff_pred_len}。"
                    f"M 必须 <= N，否则每步前进量超过单次预测量，会有位置未被覆盖。"
                    f"请调整 --ar_step_size 或 --pred_len。"
                )

            # 本次 forecasting 用到的最大 rank 数 = ceil(N/M)
            max_rank_needed = (eff_pred_len + step_size - 1) // step_size
            if max_rank_needed > self.chunk_gate_max_rank:
                # 超过预设最大 rank 数：末尾用最后一个 gate pad
                gate_logits = torch.cat([
                    self.chunk_gate_logits,
                    self.chunk_gate_logits[-1:].expand(max_rank_needed - self.chunk_gate_max_rank)
                ])
            else:
                gate_logits = self.chunk_gate_logits[:max_rank_needed]
            gate = torch.sigmoid(gate_logits)  # [max_rank_needed]

            # 每个 chunk 先 logits -> sharpen softmax -> soft codebook vector，
            # 再对同一未来位置的 soft code 向量做加权平均。
            # pos_code_sum[p]   : [B*C, 1, code_dim]
            # pos_weight_sum[p] : scalar tensor
            pos_code_sum = {}
            pos_weight_sum       = {}
            pos_first_hard_code = {}

            current_context = context_flat
            committed_list  = []   # 每项 [B*C, 1, code_dim]
            pred_idx_committed_layers = [[] for _ in range(n_rq)] if return_token_metrics else None

            step = 0
            while step * step_size < num_pred_patches:
                abs_start = step * step_size   # 本 chunk 覆盖的第一个未来 patch 位置

                # Transformer forward：context + N 个 placeholder
                n_ctx = current_context.shape[1]
                placeholder = torch.zeros(
                    B * C, eff_pred_len, code_dim,
                    device=current_context.device, dtype=current_context.dtype,
                )
                h_full = self.transformer(torch.cat([current_context, placeholder], dim=1))
                h_chunk = h_full[:, n_ctx:n_ctx + eff_pred_len, :]   # [B*C, N, hidden]
                # list[L] of [B*C, N, K]
                logits_chunk_layers = [head(h_chunk) for head in self.output_heads]

                # 把本 chunk 的每个位置先查成 soft code，再加权累积到全局融合表。
                # 关键：本 chunk 在 offset o 是位置 p=abs_start+o 的第 floor(o/M) 个 coverer。
                # gate 按 rank 分组，不依赖层号。
                for offset in range(eff_pred_len):
                    p = abs_start + offset
                    if p >= num_pred_patches:
                        break
                    rank = offset // step_size
                    w    = gate[rank]                                # scalar (learnable)
                    logits_at_p = [
                        logits_chunk_layers[l][:, offset, :] for l in range(n_rq)
                    ]
                    code_at_p = _decode_avg_logits_per_layer(logits_at_p)  # [B*C, 1, code_dim]
                    if p not in pos_code_sum:
                        pos_code_sum[p] = code_at_p * w
                        pos_weight_sum[p] = w
                        pos_first_hard_code[p] = _decode_hard_logits_per_layer(logits_at_p).detach()
                    else:
                        pos_code_sum[p] = pos_code_sum[p] + code_at_p * w
                        pos_weight_sum[p] = pos_weight_sum[p] + w

                # 提交本步的 step_size 个位置。
                # 最终输出仍使用带梯度的 fused code；喂回下一步 context 的版本使用
                # 该位置第一次被预测到时的 hard code，保证自回归输入始终落在码本元素上。
                commit_len = min(step_size, eff_pred_len)
                new_codes = []
                for offset in range(commit_len):
                    p = abs_start + offset
                    if p >= num_pred_patches:
                        break
                    denom = pos_weight_sum[p] + 1e-8
                    if return_token_metrics:
                        # token 诊断仍使用当前提交 chunk 在该位置的原始 logits argmax。
                        for l in range(n_rq):
                            pred_idx_committed_layers[l].append(
                                logits_chunk_layers[l][:, offset, :].argmax(dim=-1, keepdim=True)
                            )
                    code = pos_code_sum[p] / denom  # [B*C, 1, code_dim]
                    new_codes.append(pos_first_hard_code[p])
                    committed_list.append(code)

                # 用本步提交的 hard detached codes 延伸 context（供下一步使用）
                if new_codes:
                    current_context = torch.cat(
                        [current_context, torch.cat(new_codes, dim=1)], dim=1
                    )
                step += 1

            all_pred_codes = torch.cat(committed_list, dim=1)  # [B*C, num_pred_patches, code_dim]
            if return_token_metrics:
                pred_idx_layers = [
                    torch.cat(pred_idx_committed_layers[l], dim=1) for l in range(n_rq)
                ]  # list[L] of [B*C, P]

        # Reshape 回通道分离格式: [B*C, num_pred_patches, code_dim] -> [B, num_pred_patches, C, code_dim]
        pred_codes = all_pred_codes.reshape(B, C, num_pred_patches, code_dim).permute(0, 2, 1, 3)

        # 解码（优化后的批量解码）
        pred = self.decode_from_codes(pred_codes)  # [B, num_pred_patches*patch_size, C]
        pred = pred[:, :target_len, :]             # [B, target_len, C]

        # 确保输出长度与目标长度一致
        assert pred.shape[1] == target_len, f"预测长度 {pred.shape[1]} 与目标长度 {target_len} 不匹配"

        if return_token_metrics:
            metrics = {'token_acc': None, 'layer_acc': []}
            if target_indices is not None:
                common_patches = min(num_pred_patches, target_indices.shape[1])
                total_correct = 0
                total_count = 0
                for l, pred_idx_l in enumerate(pred_idx_layers):
                    pred_l = (
                        pred_idx_l[:, :common_patches]
                        .reshape(B, C, common_patches)
                        .permute(0, 2, 1)
                    )
                    tgt_l = target_indices[:, :common_patches, :, l]
                    correct_l = (pred_l == tgt_l).sum().item()
                    count_l = tgt_l.numel()
                    metrics['layer_acc'].append(correct_l / count_l if count_l > 0 else 0.0)
                    total_correct += correct_l
                    total_count += count_l
                metrics['token_acc'] = total_correct / total_count if total_count > 0 else 0.0
            return pred, vq_loss, metrics

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
        # indices: [B, num_patches, C, n_rq_layers]
        B, num_patches, C, n_rq = indices.shape
        all_usages = []
        for l in range(n_rq):
            if self.per_channel_codebook:
                for c in range(C):
                    unique_c = torch.unique(indices[:, :, c, l])
                    all_usages.append(len(unique_c) / self.codebook_size)
            else:
                unique = torch.unique(indices[:, :, :, l].reshape(-1))
                all_usages.append(len(unique) / self.codebook_size)
        avg_usage = sum(all_usages) / len(all_usages)
        return avg_usage, torch.tensor(all_usages)
    
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
                if self.per_channel_codebook:
                    # 尝试加载 per-channel RVQ（格式：vqs.{c}.）
                    any_vq_loaded = False
                    for c, rvq_mod in enumerate(self.vqs):
                        per_ch_prefix = f'vqs.{c}'
                        if try_load_module(rvq_mod, f'VQ[{c}]', None, per_ch_prefix):
                            any_vq_loaded = True

                    if not any_vq_loaded:
                        # 回退：从共享 VQ checkpoint 初始化所有通道码本
                        shared_loaded = False
                        for rvq_mod in self.vqs:
                            if try_load_module(rvq_mod, 'VQ', 'vq_state_dict', 'vq'):
                                shared_loaded = True
                        if shared_loaded:
                            loaded_components.append('VQ')
                    else:
                        loaded_components.append('VQ')
                else:
                    if try_load_module(self.vq, 'VQ', 'vq_state_dict', 'vq'):
                        loaded_components.append('VQ')
                    elif state_dict is not None:
                        # 尝试直接加载 layers.0.embedding 权重（兼容旧单层格式）
                        for key in state_dict.keys():
                            if 'vq' in key.lower() and 'embedding' in key.lower() and 'weight' in key.lower():
                                target = self.vq.layers[0]
                                if hasattr(target, 'embedding'):
                                    try:
                                        target.embedding.weight.data.copy_(state_dict[key])
                                        loaded_components.append('VQ')
                                        break
                                    except Exception:
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
            vq_modules = list(self.vqs) if self.per_channel_codebook else [self.vq]
            for rvq_mod in vq_modules:
                for param in rvq_mod.parameters():
                    param.requires_grad = False
                for single_vq in rvq_mod.layers:
                    if isinstance(single_vq, FlattenedVectorQuantizerEMA):
                        single_vq._disable_ema_update = True
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
            vq_modules = list(self.vqs) if self.per_channel_codebook else [self.vq]
            for rvq_mod in vq_modules:
                for param in rvq_mod.parameters():
                    param.requires_grad = True
                for single_vq in rvq_mod.layers:
                    if isinstance(single_vq, FlattenedVectorQuantizerEMA):
                        single_vq._disable_ema_update = False
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
        # VQVAE backbone: mlp 保持旧结构；tcn 启用 Conv1d/TCN codec
        'vqvae_backbone': getattr(args, 'vqvae_backbone', 'mlp'),
        'vqvae_tcn_kernel_size': int(getattr(args, 'vqvae_tcn_kernel_size', 5)),
        'vqvae_chunk_size': int(getattr(args, 'vqvae_chunk_size', 2)),
        'decoder_lowpass': bool(getattr(args, 'decoder_lowpass', 0)),
        # Transformer hidden_dim（可选，默认使用code_dim）
        'transformer_hidden_dim': getattr(args, 'transformer_hidden_dim', None),
        # 每通道独立码本（默认False，与旧行为兼容）
        'per_channel_codebook': bool(getattr(args, 'per_channel_codebook', False)),
        # RVQ 层数（默认1，与旧行为兼容）
        'n_rq_layers': int(getattr(args, 'n_rq_layers', 1)),
    }
    
    # 注意: n_channels 需要从数据加载器获取，应在调用此函数后添加:
    # config['n_channels'] = dls.vars
    
    return config
