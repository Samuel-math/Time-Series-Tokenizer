"""
轻量级码本模型：只包含Encoder、VQ和Decoder
用于码本预训练，不包含Transformer等重型模块

CodebookModel          — 所有通道共享一个 VQ
PerChannelCodebookModel — 每个通道拥有独立 VQ（nn.ModuleList self.vqs），
                          编码器/解码器仍然共享
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vqvae import Encoder, Decoder, SparseNet, TrendExtractor
from .patch_vqvae_transformer import (
    FlattenedVectorQuantizer, FlattenedVectorQuantizerEMA, ResidualVQ,
)


class CodebookModel(nn.Module):
    """
    轻量级码本模型：只包含Encoder、VQ和Decoder
    用于码本预训练，不包含Transformer等重型模块
    """
    def __init__(self, config, n_channels):
        super().__init__()
        self.patch_size = config['patch_size']
        self.embedding_dim = config['embedding_dim']
        self.compression_factor = config['compression_factor']
        self.codebook_size = config['codebook_size']
        self.commitment_cost = config['commitment_cost']
        
        # 计算code_dim
        self.compressed_len = self.patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len
        
        # Encoder和Decoder（单通道，channel-independent）
        self.encoder = Encoder(
            in_channels=1,
            num_hiddens=config['num_hiddens'],
            num_residual_layers=config['num_residual_layers'],
            num_residual_hiddens=config['num_residual_hiddens'],
            embedding_dim=self.embedding_dim,
            compression_factor=self.compression_factor
        )
        self.decoder = Decoder(
            in_channels=self.embedding_dim,
            num_hiddens=config['num_hiddens'],
            num_residual_layers=config['num_residual_layers'],
            num_residual_hiddens=config['num_residual_hiddens'],
            compression_factor=self.compression_factor,
            out_channels=1
        )
        
        # VQ（包裹在 ResidualVQ 中；n_rq_layers=1 时等价于原来的单层 VQ）
        self.n_rq_layers = config.get('n_rq_layers', 1)
        init_method = config.get('vq_init_method', 'random')

        def _make_single_vq():
            if config.get('codebook_ema', False):
                return FlattenedVectorQuantizerEMA(
                    self.codebook_size, self.code_dim, self.commitment_cost,
                    decay=config.get('ema_decay', 0.99), eps=config.get('ema_eps', 1e-5),
                    init_method=init_method,
                )
            return FlattenedVectorQuantizer(
                self.codebook_size, self.code_dim, self.commitment_cost,
                init_method=init_method,
            )

        self.vq = ResidualVQ(self.n_rq_layers, _make_single_vq)

        # Robust VQVAE: 稀疏异常分量网络（sparse_weight=0 时不实例化）
        self.sparse_net: SparseNet | None = None
        if config.get('sparse_weight', 0) > 0:
            self.sparse_net = SparseNet(
                patch_size=self.patch_size,
                num_hiddens=config['num_hiddens'],
                amplitude=config.get('sparse_amplitude', 0.5),
            )

    def _apply_sparse(self, x_c: torch.Tensor):
        """
        对单通道 patch 做稀疏分解。

        Args:
            x_c: [N, 1, patch_size]
        Returns:
            x_clean: [N, 1, patch_size]  — 去掉稀疏分量后送 Encoder
            s:       [N, patch_size] or None
        """
        if self.sparse_net is None:
            return x_c, None
        s = self.sparse_net(x_c)              # [N, 1, patch_size]
        return x_c - s, s.squeeze(1)          # x_clean [N,1,P], s [N,P]

    def init_codebook_from_data(self, dataloader, device, num_samples=10000, method='kmeans', revin=None):
        """
        从数据初始化码本（数据驱动初始化）
        
        Args:
            dataloader: 数据加载器
            device: 设备
            num_samples: 收集的样本数量
            method: 'kmeans' 或 'random_sample'
            revin: RevIN归一化器（可选）
        """
        self.eval()
        z_samples_list = []
        n_collected = 0
        
        print(f"\n收集encoder输出用于码本初始化（目标样本数: {num_samples}）...")
        
        with torch.no_grad():
            for batch_x, _ in dataloader:
                if n_collected >= num_samples:
                    break
                
                batch_x = batch_x.to(device)
                
                # RevIN归一化（如果使用）
                if revin is not None:
                    batch_x = revin(batch_x, 'norm')
                
                B, T, C = batch_x.shape
                num_patches = T // self.patch_size
                x = batch_x[:, :num_patches * self.patch_size, :]
                x_patches = x.reshape(B, num_patches, self.patch_size, C)
                
                # 收集所有通道的encoder输出
                for c in range(C):
                    x_c = x_patches[:, :, :, c]  # [B, num_patches, patch_size]
                    x_c_flat = x_c.reshape(B * num_patches, self.patch_size)
                    x_c_flat = x_c_flat.unsqueeze(1)  # [B*num_patches, 1, patch_size]
                    
                    # Robust: 用 x_clean 送 Encoder
                    x_c_clean, _ = self._apply_sparse(x_c_flat)
                    # Encoder输出
                    z = self.encoder(x_c_clean, self.compression_factor)  # [B*num_patches, embedding_dim, compressed_len]
                    z_flat = z.reshape(B * num_patches, -1)  # [B*num_patches, code_dim]
                    
                    z_samples_list.append(z_flat)
                    n_collected += z_flat.size(0)
                    
                    if n_collected >= num_samples:
                        break
        
        # 合并所有样本
        z_samples = torch.cat(z_samples_list, dim=0)  # [N, code_dim]
        if z_samples.size(0) > num_samples:
            z_samples = z_samples[:num_samples]
        
        print(f"已收集 {z_samples.size(0)} 个encoder输出样本")
        
        # 初始化码本
        self.vq.init_from_data(z_samples, method=method)
        
        self.train()  # 恢复训练模式
    
    def encode_to_indices(self, x, return_distances=False, return_sparse=False):
        """
        Args:
            x: [B, T, C]
            return_distances: 是否返回到码本第 0 层的距离（用于软索引计算）
            return_sparse: 是否返回稀疏分量 s [B, num_patches*patch_size, C]
        Returns:
            indices: [B, num_patches, C, n_rq_layers]
            vq_loss: scalar
            z_q: [B, num_patches, C, code_dim]
            distances (optional)
            s_tensor (optional): [B, num_patches*patch_size, C] or None
        """
        B, T, C = x.shape
        num_patches = T // self.patch_size

        x = x[:, :num_patches * self.patch_size, :]
        x_patches = x.reshape(B, num_patches, self.patch_size, C)

        z_list, s_list = [], []
        for c in range(C):
            x_c = x_patches[:, :, :, c].reshape(B * num_patches, self.patch_size).unsqueeze(1)
            x_c_clean, s_c = self._apply_sparse(x_c)
            z = self.encoder(x_c_clean, self.compression_factor)
            z_flat = z.reshape(B * num_patches, self.code_dim)
            z_list.append(z_flat.reshape(B, num_patches, self.code_dim))
            if return_sparse and s_c is not None:
                s_list.append(s_c.reshape(B, num_patches, self.patch_size))

        z_all = torch.stack(z_list, dim=2)  # [B, num_patches, C, code_dim]

        indices_list, z_q_list, distances_list = [], [], []
        vq_loss_sum = 0

        for c in range(C):
            z_c_flat = z_all[:, :, c, :].reshape(B * num_patches, self.code_dim)

            if return_distances:
                w0 = self.vq.layers[0].embedding.weight
                distances_c = (
                    torch.sum(z_c_flat ** 2, dim=1, keepdim=True)
                    + torch.sum(w0 ** 2, dim=1)
                    - 2 * torch.matmul(z_c_flat, w0.t())
                )
                distances_list.append(distances_c)

            vq_loss_c, z_q_sum_c, all_idx_c = self.vq(z_c_flat)
            vq_loss_sum += vq_loss_c

            # [B*num_patches, n_rq_layers] → [B, num_patches, n_rq_layers]
            indices_c = torch.stack(all_idx_c, dim=1).reshape(B, num_patches, self.n_rq_layers)
            z_q_c = z_q_sum_c.reshape(B, num_patches, self.code_dim)

            indices_list.append(indices_c)
            z_q_list.append(z_q_c)

        indices = torch.stack(indices_list, dim=2)  # [B, num_patches, C, n_rq_layers]
        z_q = torch.stack(z_q_list, dim=2)          # [B, num_patches, C, code_dim]
        vq_loss = vq_loss_sum / C

        # 组装稀疏分量张量
        if return_sparse:
            if s_list:
                s_tensor = torch.stack(s_list, dim=3)   # [B, num_patches, patch_size, C]
                s_tensor = s_tensor.reshape(B, -1, C)   # [B, num_patches*patch_size, C]
            else:
                s_tensor = None
            if return_distances:
                distances = torch.cat(distances_list, dim=0)
                return indices, vq_loss, z_q, distances, s_tensor
            return indices, vq_loss, z_q, s_tensor

        if return_distances:
            distances = torch.cat(distances_list, dim=0)
            return indices, vq_loss, z_q, distances

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


class PerChannelCodebookModel(nn.Module):
    """
    Per-channel 码本模型：共享 Encoder/Decoder，每个通道拥有独立 VQ。

    VQ 模块存放在 self.vqs = nn.ModuleList(...)，保存时的 state_dict 键名为
    vqs.0.*, vqs.1.*, ...，与 PatchVQVAETransformer(per_channel_codebook=True)
    的命名完全一致，因此可以直接被 load_vqvae_weights() 加载。
    """

    def __init__(self, config, n_channels):
        super().__init__()
        self.patch_size = config['patch_size']
        self.embedding_dim = config['embedding_dim']
        self.compression_factor = config['compression_factor']
        self.codebook_size = config['codebook_size']
        self.commitment_cost = config['commitment_cost']
        self.n_channels = n_channels

        self.compressed_len = self.patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len

        # 共享 Encoder / Decoder（channel-independent：单通道输入/输出）
        self.encoder = Encoder(
            in_channels=1,
            num_hiddens=config['num_hiddens'],
            num_residual_layers=config['num_residual_layers'],
            num_residual_hiddens=config['num_residual_hiddens'],
            embedding_dim=self.embedding_dim,
            compression_factor=self.compression_factor,
        )
        self.decoder = Decoder(
            in_channels=self.embedding_dim,
            num_hiddens=config['num_hiddens'],
            num_residual_layers=config['num_residual_layers'],
            num_residual_hiddens=config['num_residual_hiddens'],
            compression_factor=self.compression_factor,
            out_channels=1,
        )

        # 每个通道独立的 RVQ（键名 vqs.{c}.* 与 PatchVQVAETransformer 保持一致）
        self.n_rq_layers = config.get('n_rq_layers', 1)
        init_method = config.get('vq_init_method', 'random')

        def _make_single_vq():
            if config.get('codebook_ema', False):
                return FlattenedVectorQuantizerEMA(
                    self.codebook_size, self.code_dim, self.commitment_cost,
                    decay=config.get('ema_decay', 0.99),
                    eps=config.get('ema_eps', 1e-5),
                    init_method=init_method,
                )
            return FlattenedVectorQuantizer(
                self.codebook_size, self.code_dim, self.commitment_cost,
                init_method=init_method,
            )

        self.vqs = nn.ModuleList([
            ResidualVQ(self.n_rq_layers, _make_single_vq) for _ in range(n_channels)
        ])

        # Robust VQVAE: 稀疏异常分量网络（与共享 Encoder 搭配，所有通道共用一个 SparseNet）
        self.sparse_net: SparseNet | None = None
        if config.get('sparse_weight', 0) > 0:
            self.sparse_net = SparseNet(
                patch_size=self.patch_size,
                num_hiddens=config['num_hiddens'],
                amplitude=config.get('sparse_amplitude', 0.5),
            )

    def _apply_sparse(self, x_c: torch.Tensor):
        """见 CodebookModel._apply_sparse。"""
        if self.sparse_net is None:
            return x_c, None
        s = self.sparse_net(x_c)
        return x_c - s, s.squeeze(1)

    # ------------------------------------------------------------------

    def init_codebook_from_data(self, dataloader, device, num_samples=10000,
                                method='kmeans', revin=None):
        """从数据分通道初始化各自的码本。"""
        self.eval()
        # 每个通道单独收集 encoder 输出
        z_per_channel = [[] for _ in range(self.n_channels)]
        n_collected = [0] * self.n_channels
        target = num_samples

        print(f"\n收集 encoder 输出用于 per-channel 码本初始化（每通道目标: {target}）...")

        with torch.no_grad():
            for batch_x, _ in dataloader:
                if all(n >= target for n in n_collected):
                    break
                batch_x = batch_x.to(device)
                if revin is not None:
                    batch_x = revin(batch_x, 'norm')

                B, T, C = batch_x.shape
                num_patches = T // self.patch_size
                x = batch_x[:, :num_patches * self.patch_size, :]
                x_patches = x.reshape(B, num_patches, self.patch_size, C)

                for c in range(min(C, self.n_channels)):
                    if n_collected[c] >= target:
                        continue
                    x_c = x_patches[:, :, :, c].reshape(B * num_patches, self.patch_size)
                    x_c = x_c.unsqueeze(1)  # [B*P, 1, patch_size]
                    x_c_clean, _ = self._apply_sparse(x_c)
                    z = self.encoder(x_c_clean, self.compression_factor)
                    z_flat = z.reshape(B * num_patches, -1)
                    z_per_channel[c].append(z_flat.cpu())
                    n_collected[c] += z_flat.size(0)

        for c in range(self.n_channels):
            if not z_per_channel[c]:
                print(f"  警告: 通道 {c} 未收集到样本，跳过初始化")
                continue
            z_samples = torch.cat(z_per_channel[c], dim=0)[:target].to(device)
            print(f"  通道 {c}: {z_samples.size(0)} 个样本")
            self.vqs[c].init_from_data(z_samples, method=method)

        self.train()

    # ------------------------------------------------------------------
    # 前向接口（与 CodebookModel 完全相同，方便 train_epoch 复用）
    # ------------------------------------------------------------------

    def encode_to_indices(self, x, return_sparse=False):
        """
        Args:
            x: [B, T, C]
            return_sparse: 是否返回稀疏分量 s [B, num_patches*patch_size, C]
        Returns:
            indices: [B, num_patches, C, n_rq_layers]
            vq_loss: scalar
            z_q: [B, num_patches, C, code_dim]
            s_tensor (optional): [B, num_patches*patch_size, C] or None
        """
        B, T, C = x.shape
        num_patches = T // self.patch_size
        x = x[:, :num_patches * self.patch_size, :]
        x_patches = x.reshape(B, num_patches, self.patch_size, C)

        z_list, s_list = [], []
        for c in range(C):
            x_c = x_patches[:, :, :, c].reshape(B * num_patches, self.patch_size).unsqueeze(1)
            x_c_clean, s_c = self._apply_sparse(x_c)
            z = self.encoder(x_c_clean, self.compression_factor)
            z_flat = z.reshape(B * num_patches, self.code_dim)
            z_list.append(z_flat.reshape(B, num_patches, self.code_dim))
            if return_sparse and s_c is not None:
                s_list.append(s_c.reshape(B, num_patches, self.patch_size))

        z_all = torch.stack(z_list, dim=2)  # [B, num_patches, C, code_dim]

        indices_list, z_q_list = [], []
        vq_loss_sum = 0.0
        for c in range(C):
            z_c_flat = z_all[:, :, c, :].reshape(B * num_patches, self.code_dim)
            vq_loss_c, z_q_sum_c, all_idx_c = self.vqs[c](z_c_flat)
            vq_loss_sum += vq_loss_c
            indices_c = torch.stack(all_idx_c, dim=1).reshape(B, num_patches, self.n_rq_layers)
            indices_list.append(indices_c)
            z_q_list.append(z_q_sum_c.reshape(B, num_patches, self.code_dim))

        indices = torch.stack(indices_list, dim=2)  # [B, num_patches, C, n_rq_layers]
        z_q = torch.stack(z_q_list, dim=2)          # [B, num_patches, C, code_dim]
        vq_loss = vq_loss_sum / C

        if return_sparse:
            if s_list:
                s_tensor = torch.stack(s_list, dim=3)  # [B, num_patches, patch_size, C]
                s_tensor = s_tensor.reshape(B, -1, C)
            else:
                s_tensor = None
            return indices, vq_loss, z_q, s_tensor

        return indices, vq_loss, z_q

    def decode_from_codes(self, z_q):
        """
        Args:
            z_q: [B, num_patches, C, code_dim]
        Returns:
            x_recon: [B, num_patches * patch_size, C]
        """
        B, num_patches, C, _ = z_q.shape
        x_recon_list = []
        for c in range(C):
            z_q_c = z_q[:, :, c, :].reshape(B * num_patches, self.embedding_dim, self.compressed_len)
            x_c = self.decoder(z_q_c, self.compression_factor)   # [B*P, 1, patch_size]
            x_recon_list.append(x_c.reshape(B, num_patches, self.patch_size))

        x_recon = torch.stack(x_recon_list, dim=3)  # [B, num_patches, patch_size, C]
        return x_recon.reshape(B, -1, C)


class TrendResidualCodebookModel(nn.Module):
    """
    基于低通滤波的趋势-残差分解双码本模型。

    分解方式（确定性，不可学习）:
        X_patch = X_trend + X_residual
        X_trend   = LowPass(X_patch)       # 低通滤波（移动平均）
        X_residual= X_patch - X_trend

    两条独立支路，各自的 Encoder/Decoder 与 VQ 码本：
        X_trend    ──→ Enc_T ──→ VQ_T ──→ Dec_T ──→ X_trend_hat
        X_residual
            ├── SparseNet → s            （L1 稀疏约束）
            └── (X_residual − s) ─→ Enc_R ─→ VQ_R ─→ Dec_R ─→ r_clean_hat

        X_hat = X_trend_hat + r_clean_hat + s

    损失:
        L_trend_rec = MSE(X_trend_hat, X_trend)
        L_res_rec   = MSE(r_clean_hat + s, X_residual)
        L_vq        = vq_loss_T + vq_loss_R
        L_sparse    = λ · mean(|s|)

    两个码本语义明确:
        - Codebook_T : 低频趋势原子
        - Codebook_R : 高频/振荡残差原子（稀疏异常已剥离）
    """

    def __init__(self, config, n_channels):
        super().__init__()
        self.patch_size = config['patch_size']
        self.embedding_dim = config['embedding_dim']
        self.compression_factor = config['compression_factor']
        self.commitment_cost = config['commitment_cost']
        self.compressed_len = self.patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len
        self.n_rq_layers = config.get('n_rq_layers', 1)
        self.n_channels = n_channels

        # 允许两个码本使用不同大小；缺省值回落到共享的 codebook_size
        default_cb = config['codebook_size']
        self.codebook_size_trend = int(config.get('codebook_size_trend') or default_cb)
        self.codebook_size_res = int(config.get('codebook_size_res') or default_cb)

        # 低通滤波器（learnable=False 固定 MA，无参数；learnable=True 可学习权重，
        # 通过 softmax 归一化保持非负且和为 1 的低通性质）
        self.trend_extractor = TrendExtractor(
            kernel_size=int(config.get('trend_kernel_size', 5)),
            learnable=bool(config.get('trend_learnable_filter', False)),
        )

        enc_kw = dict(
            in_channels=1,
            num_hiddens=config['num_hiddens'],
            num_residual_layers=config['num_residual_layers'],
            num_residual_hiddens=config['num_residual_hiddens'],
            embedding_dim=self.embedding_dim,
            compression_factor=self.compression_factor,
        )
        dec_kw = dict(
            in_channels=self.embedding_dim,
            num_hiddens=config['num_hiddens'],
            num_residual_layers=config['num_residual_layers'],
            num_residual_hiddens=config['num_residual_hiddens'],
            compression_factor=self.compression_factor,
            out_channels=1,
        )

        # 趋势支路
        self.trend_encoder = Encoder(**enc_kw)
        self.trend_decoder = Decoder(**dec_kw)
        self.trend_vq = self._build_rvq(self.codebook_size_trend, config)

        # 残差支路
        self.res_encoder = Encoder(**enc_kw)
        self.res_decoder = Decoder(**dec_kw)
        self.res_vq = self._build_rvq(self.codebook_size_res, config)

        # 稀疏噪声网络（作用于残差）
        self.sparse_net: SparseNet | None = None
        if config.get('sparse_weight', 0) > 0:
            self.sparse_net = SparseNet(
                patch_size=self.patch_size,
                num_hiddens=config['num_hiddens'],
                amplitude=config.get('sparse_amplitude', 0.5),
            )

    def _build_rvq(self, codebook_size, config):
        init_method = config.get('vq_init_method', 'random')
        cd = self.code_dim
        cc = self.commitment_cost

        def _make():
            if config.get('codebook_ema', False):
                return FlattenedVectorQuantizerEMA(
                    codebook_size, cd, cc,
                    decay=config.get('ema_decay', 0.99),
                    eps=config.get('ema_eps', 1e-5),
                    init_method=init_method,
                )
            return FlattenedVectorQuantizer(
                codebook_size, cd, cc, init_method=init_method,
            )
        return ResidualVQ(self.n_rq_layers, _make)

    # ------------------------------------------------------------------
    # 内部单通道前向：返回该通道的所有中间量
    # ------------------------------------------------------------------
    def _forward_channel(self, x_c: torch.Tensor):
        """
        Args:
            x_c: [N, 1, patch_size]
        Returns:
            dict: 该通道的所有输出与损失分量（标量 loss 为该通道平均）
        """
        N = x_c.size(0)

        # 1. 趋势分解（低通滤波，无梯度）
        x_trend = self.trend_extractor(x_c)          # [N, 1, P]
        r = x_c - x_trend                            # [N, 1, P]

        # 2. 趋势支路
        z_t = self.trend_encoder(x_trend, self.compression_factor)
        z_t_flat = z_t.reshape(N, self.code_dim)
        t_vq_loss, z_qt, t_idx_layers = self.trend_vq(z_t_flat)
        x_trend_hat = self.trend_decoder(
            z_qt.reshape(N, self.embedding_dim, self.compressed_len),
            self.compression_factor,
        )                                            # [N, P]

        # 3. 稀疏噪声（作用于残差）
        if self.sparse_net is not None:
            s = self.sparse_net(r)                   # [N, 1, P]
            r_clean = r - s
        else:
            s = None
            r_clean = r

        # 4. 残差支路
        z_r = self.res_encoder(r_clean, self.compression_factor)
        z_r_flat = z_r.reshape(N, self.code_dim)
        r_vq_loss, z_qr, r_idx_layers = self.res_vq(z_r_flat)
        r_clean_hat = self.res_decoder(
            z_qr.reshape(N, self.embedding_dim, self.compressed_len),
            self.compression_factor,
        )                                            # [N, P]

        # 5. 损失分量
        x_trend_2d = x_trend.squeeze(1)              # [N, P]
        r_2d = r.squeeze(1)                          # [N, P]
        trend_rec = F.mse_loss(x_trend_hat, x_trend_2d)
        if s is not None:
            s_2d = s.squeeze(1)                      # [N, P]
            res_rec = F.mse_loss(r_clean_hat + s_2d, r_2d)
            sparse_norm = s_2d.abs().mean()
            x_hat = x_trend_hat + r_clean_hat + s_2d
        else:
            res_rec = F.mse_loss(r_clean_hat, r_2d)
            sparse_norm = x_c.new_tensor(0.0)
            x_hat = x_trend_hat + r_clean_hat

        return {
            'trend_rec': trend_rec,
            'res_rec': res_rec,
            'trend_vq': t_vq_loss,
            'res_vq': r_vq_loss,
            'sparse_norm': sparse_norm,
            'x_hat': x_hat,                          # [N, P]
            'z_qt': z_qt,                            # [N, code_dim]
            'z_qr': z_qr,                            # [N, code_dim]
            't_idx_layers': t_idx_layers,            # list of [N]
            'r_idx_layers': r_idx_layers,            # list of [N]
        }

    # ------------------------------------------------------------------
    # 统一前向接口（训练时使用）
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, C]
        Returns:
            dict:
                trend_recon_loss, res_recon_loss  — 各通道平均
                trend_vq_loss,    res_vq_loss
                sparse_norm                         — 各通道平均 L1
                trend_indices [B, P, C, n_rq]
                res_indices   [B, P, C, n_rq]
                z_q_trend     [B, P, C, code_dim]
                z_q_res       [B, P, C, code_dim]
                x_recon       [B, P*patch_size, C]
        """
        B, T, C = x.shape
        P = T // self.patch_size
        x = x[:, :P * self.patch_size, :]
        x_patches = x.reshape(B, P, self.patch_size, C)

        trend_rec = res_rec = trend_vq = res_vq = sparse_norm = x.new_tensor(0.0)
        t_idx_all, r_idx_all = [], []
        zqt_all, zqr_all = [], []
        recon_channels = []

        for c in range(C):
            x_c = x_patches[:, :, :, c].reshape(B * P, self.patch_size).unsqueeze(1)
            out = self._forward_channel(x_c)

            trend_rec = trend_rec + out['trend_rec']
            res_rec = res_rec + out['res_rec']
            trend_vq = trend_vq + out['trend_vq']
            res_vq = res_vq + out['res_vq']
            sparse_norm = sparse_norm + out['sparse_norm']

            recon_channels.append(out['x_hat'].reshape(B, P, self.patch_size))
            t_idx_all.append(
                torch.stack(out['t_idx_layers'], dim=1).reshape(B, P, self.n_rq_layers)
            )
            r_idx_all.append(
                torch.stack(out['r_idx_layers'], dim=1).reshape(B, P, self.n_rq_layers)
            )
            zqt_all.append(out['z_qt'].reshape(B, P, self.code_dim))
            zqr_all.append(out['z_qr'].reshape(B, P, self.code_dim))

        x_recon = torch.stack(recon_channels, dim=3).reshape(B, -1, C)

        return {
            'trend_recon_loss': trend_rec / C,
            'res_recon_loss': res_rec / C,
            'trend_vq_loss': trend_vq / C,
            'res_vq_loss': res_vq / C,
            'sparse_norm': sparse_norm / C,
            'trend_indices': torch.stack(t_idx_all, dim=2),
            'res_indices': torch.stack(r_idx_all, dim=2),
            'z_q_trend': torch.stack(zqt_all, dim=2),
            'z_q_res': torch.stack(zqr_all, dim=2),
            'x_recon': x_recon,
        }

    # ------------------------------------------------------------------
    # 推理接口（返回两套码本索引与量化向量）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, C]
        Returns:
            trend_indices [B, P, C, n_rq], res_indices [B, P, C, n_rq],
            z_q_trend [B, P, C, code_dim], z_q_res [B, P, C, code_dim]
        """
        out = self.forward(x)
        return (
            out['trend_indices'], out['res_indices'],
            out['z_q_trend'], out['z_q_res'],
        )

    def decode_from_codes(self, z_q_trend: torch.Tensor, z_q_res: torch.Tensor,
                          s: torch.Tensor | None = None):
        """
        Args:
            z_q_trend: [B, P, C, code_dim]
            z_q_res:   [B, P, C, code_dim]
            s:         [B, P*patch_size, C] 或 None（通常推理时不提供）
        Returns:
            x_recon: [B, P*patch_size, C]
        """
        B, P, C, _ = z_q_trend.shape
        recon_channels = []
        for c in range(C):
            zqt = z_q_trend[:, :, c, :].reshape(B * P, self.embedding_dim, self.compressed_len)
            zqr = z_q_res[:, :, c, :].reshape(B * P, self.embedding_dim, self.compressed_len)
            x_trend_hat = self.trend_decoder(zqt, self.compression_factor)  # [B*P, P]
            r_clean_hat = self.res_decoder(zqr, self.compression_factor)    # [B*P, P]
            x_hat_c = x_trend_hat + r_clean_hat
            recon_channels.append(x_hat_c.reshape(B, P, self.patch_size))

        x_recon = torch.stack(recon_channels, dim=3).reshape(B, -1, C)
        if s is not None:
            recon_len = x_recon.shape[1]
            x_recon = x_recon + s[:, :recon_len, :]
        return x_recon

    # ------------------------------------------------------------------
    # 数据驱动码本初始化：两个码本分别收集
    # ------------------------------------------------------------------
    def init_codebook_from_data(self, dataloader, device, num_samples=10000,
                                method='kmeans', revin=None):
        """分别从趋势/残差 encoder 输出初始化两个码本。"""
        self.eval()
        z_trend_list, z_res_list = [], []
        n_t = n_r = 0

        print(f"\n收集 trend/res encoder 输出用于码本初始化（目标样本数: {num_samples}）...")
        with torch.no_grad():
            for batch_x, _ in dataloader:
                if n_t >= num_samples and n_r >= num_samples:
                    break
                batch_x = batch_x.to(device)
                if revin is not None:
                    batch_x = revin(batch_x, 'norm')

                B, T, C = batch_x.shape
                P = T // self.patch_size
                x = batch_x[:, :P * self.patch_size, :]
                x_patches = x.reshape(B, P, self.patch_size, C)

                for c in range(C):
                    x_c = x_patches[:, :, :, c].reshape(B * P, self.patch_size).unsqueeze(1)
                    x_trend = self.trend_extractor(x_c)
                    r = x_c - x_trend
                    if self.sparse_net is not None:
                        r = r - self.sparse_net(r)

                    if n_t < num_samples:
                        zt = self.trend_encoder(x_trend, self.compression_factor)
                        z_trend_list.append(zt.reshape(B * P, -1).cpu())
                        n_t += zt.size(0)
                    if n_r < num_samples:
                        zr = self.res_encoder(r, self.compression_factor)
                        z_res_list.append(zr.reshape(B * P, -1).cpu())
                        n_r += zr.size(0)

                    if n_t >= num_samples and n_r >= num_samples:
                        break

        if z_trend_list:
            zt_all = torch.cat(z_trend_list, dim=0)[:num_samples].to(device)
            print(f"  趋势码本初始化: {zt_all.size(0)} 个样本")
            self.trend_vq.init_from_data(zt_all, method=method)
        if z_res_list:
            zr_all = torch.cat(z_res_list, dim=0)[:num_samples].to(device)
            print(f"  残差码本初始化: {zr_all.size(0)} 个样本")
            self.res_vq.init_from_data(zr_all, method=method)

        self.train()
