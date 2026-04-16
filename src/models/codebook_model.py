"""
轻量级码本模型：只包含Encoder、VQ和Decoder
用于码本预训练，不包含Transformer等重型模块

CodebookModel              — 标准 VQVAE（所有通道共享一个 VQ）
DecomposedCodebookModel    — Trend–Stochastic–Oscillatory 三分量分解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vqvae import Encoder, Decoder, SparseNet, TrendExtractor, StochasticVAE
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


class DecomposedCodebookModel(nn.Module):
    """
    Trend–Stochastic–Oscillatory 三分量分解码本模型。

    数据流:
        x_patch → TrendExtractor → x_trend ──→ Enc_T → VQ_T → Dec_T → x̂_trend
                                    │
                               r = x − x_trend
                                    │
                                    ├─→ StochasticVAE → ŝ  (低维 VAE)
                                    │
                               o = r − ŝ
                                    │
                                    └─→ Enc_O → VQ_O → Dec_O → ô

        最终重构: x̂ = x̂_trend + ŝ + ô

    损失: L = L_trend_rec + L_rec + λ₁·L_KL + λ₂·L_shape + λ₃·L_vq
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

        self.codebook_size_trend = config.get('codebook_size_trend', config['codebook_size'])
        self.codebook_size_osc = config.get('codebook_size_osc', config['codebook_size'])
        self.shape_alpha = config.get('shape_alpha', 0.5)

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

        # ── Trend path ───────────────────────────────────────────────
        self.trend_extractor = TrendExtractor(
            kernel_size=config.get('trend_kernel_size', 5)
        )
        self.trend_encoder = Encoder(**enc_kw)
        self.trend_decoder = Decoder(**dec_kw)
        self.trend_vq = self._build_rvq(self.codebook_size_trend, config)

        # ── Stochastic VAE ───────────────────────────────────────────
        self.stochastic_vae = StochasticVAE(
            patch_size=self.patch_size,
            latent_dim=config.get('stochastic_latent_dim', 4),
            num_hiddens=config['num_hiddens'],
        )

        # ── Oscillatory path ─────────────────────────────────────────
        self.osc_encoder = Encoder(**enc_kw)
        self.osc_decoder = Decoder(**dec_kw)
        self.osc_vq = self._build_rvq(self.codebook_size_osc, config)

    # ------------------------------------------------------------------
    def _build_rvq(self, codebook_size, config):
        init_method = config.get('vq_init_method', 'random')
        def _make():
            if config.get('codebook_ema', False):
                return FlattenedVectorQuantizerEMA(
                    codebook_size, self.code_dim, self.commitment_cost,
                    decay=config.get('ema_decay', 0.99),
                    eps=config.get('ema_eps', 1e-5),
                    init_method=init_method,
                )
            return FlattenedVectorQuantizer(
                codebook_size, self.code_dim, self.commitment_cost,
                init_method=init_method,
            )
        return ResidualVQ(self.n_rq_layers, _make)

    # ------------------------------------------------------------------
    def init_codebook_from_data(self, dataloader, device,
                                num_samples=10000, method='kmeans', revin=None):
        """从数据初始化 trend_vq 和 osc_vq 两个码本。"""
        self.eval()
        z_trend_list, z_osc_list = [], []
        n_collected = 0

        print(f"\n收集 encoder 输出用于双码本初始化（目标: {num_samples}）...")
        with torch.no_grad():
            for batch_x, _ in dataloader:
                if n_collected >= num_samples:
                    break
                batch_x = batch_x.to(device)
                if revin is not None:
                    batch_x = revin(batch_x, 'norm')

                B, T, C = batch_x.shape
                P = T // self.patch_size
                x_patches = batch_x[:, :P * self.patch_size, :].reshape(
                    B, P, self.patch_size, C
                )
                for c in range(C):
                    if n_collected >= num_samples:
                        break
                    x_c = x_patches[:, :, :, c].reshape(B * P, self.patch_size).unsqueeze(1)

                    x_trend = self.trend_extractor(x_c)
                    r = x_c - x_trend

                    z_t = self.trend_encoder(x_trend, self.compression_factor)
                    z_trend_list.append(z_t.reshape(B * P, -1))

                    s_hat, _, _ = self.stochastic_vae(r)
                    o = r - s_hat
                    z_o = self.osc_encoder(o, self.compression_factor)
                    z_osc_list.append(z_o.reshape(B * P, -1))

                    n_collected += B * P

        z_trend_all = torch.cat(z_trend_list, dim=0)[:num_samples]
        z_osc_all = torch.cat(z_osc_list, dim=0)[:num_samples]
        print(f"  Trend  码本: {z_trend_all.size(0)} 个样本")
        self.trend_vq.init_from_data(z_trend_all, method=method)
        print(f"  Oscillatory 码本: {z_osc_all.size(0)} 个样本")
        self.osc_vq.init_from_data(z_osc_all, method=method)

        self.train()

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        完整前向传播（训练用）。逐通道处理以保持 EMA 更新语义一致。

        Args:
            x: [B, T, C]
        Returns:
            dict: trend_indices, osc_indices, z_q_trend, z_q_osc,
                  trend_vq_loss, osc_vq_loss, kl_loss, recon_loss,
                  trend_recon_loss, shape_loss
        """
        B, T, C = x.shape
        P = T // self.patch_size
        x = x[:, :P * self.patch_size, :]
        x_patches = x.reshape(B, P, self.patch_size, C)

        t_idx_l, o_idx_l, zqt_l, zqo_l = [], [], [], []
        t_vq = osc_vq = kl = rec = t_rec = shape = x.new_tensor(0.0)

        for c in range(C):
            xc = x_patches[:, :, :, c].reshape(B * P, self.patch_size).unsqueeze(1)

            # 1. Trend
            x_trend = self.trend_extractor(xc)
            r = xc - x_trend

            z_t = self.trend_encoder(x_trend, self.compression_factor)
            z_t_flat = z_t.reshape(B * P, self.code_dim)
            tvl, zqt, tidx = self.trend_vq(z_t_flat)
            x_trend_hat = self.trend_decoder(
                zqt.reshape(B * P, self.embedding_dim, self.compressed_len),
                self.compression_factor,
            )
            t_vq = t_vq + tvl
            t_rec = t_rec + F.mse_loss(x_trend_hat, x_trend)

            # 2. Stochastic VAE
            s_hat, mu, logvar = self.stochastic_vae(r)
            kl = kl + (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))

            # 3. Oscillatory
            o = r - s_hat.detach()
            z_o = self.osc_encoder(o, self.compression_factor)
            z_o_flat = z_o.reshape(B * P, self.code_dim)
            ovl, zqo, oidx = self.osc_vq(z_o_flat)
            o_hat = self.osc_decoder(
                zqo.reshape(B * P, self.embedding_dim, self.compressed_len),
                self.compression_factor,
            )
            osc_vq = osc_vq + ovl

            # 4. Losses
            rec = rec + F.mse_loss(o_hat + s_hat, r)
            s_flat = s_hat.reshape(-1)
            shape = shape + (
                self.shape_alpha * s_flat.abs().mean()
                + (1 - self.shape_alpha) * s_flat.pow(2).mean()
            )

            # collect
            t_idx_l.append(torch.stack(tidx, dim=1).reshape(B, P, self.n_rq_layers))
            o_idx_l.append(torch.stack(oidx, dim=1).reshape(B, P, self.n_rq_layers))
            zqt_l.append(zqt.reshape(B, P, self.code_dim))
            zqo_l.append(zqo.reshape(B, P, self.code_dim))

        return {
            'trend_indices': torch.stack(t_idx_l, dim=2),       # [B, P, C, n_rq]
            'osc_indices': torch.stack(o_idx_l, dim=2),         # [B, P, C, n_rq]
            'z_q_trend': torch.stack(zqt_l, dim=2),             # [B, P, C, cd]
            'z_q_osc': torch.stack(zqo_l, dim=2),               # [B, P, C, cd]
            'trend_vq_loss': t_vq / C,
            'osc_vq_loss': osc_vq / C,
            'kl_loss': kl / C,
            'recon_loss': rec / C,
            'trend_recon_loss': t_rec / C,
            'shape_loss': shape / C,
        }

    # ------------------------------------------------------------------
    def decode_trend(self, z_q_trend):
        """z_q_trend: [B, P, C, code_dim] → [B, P*ps, C]"""
        B, P, C, _ = z_q_trend.shape
        out = []
        for c in range(C):
            zt = z_q_trend[:, :, c, :].reshape(B * P, self.embedding_dim, self.compressed_len)
            out.append(self.trend_decoder(zt, self.compression_factor).reshape(B, P, self.patch_size))
        return torch.stack(out, dim=3).reshape(B, -1, C)

    def decode_osc(self, z_q_osc):
        """z_q_osc: [B, P, C, code_dim] → [B, P*ps, C]"""
        B, P, C, _ = z_q_osc.shape
        out = []
        for c in range(C):
            zo = z_q_osc[:, :, c, :].reshape(B * P, self.embedding_dim, self.compressed_len)
            out.append(self.osc_decoder(zo, self.compression_factor).reshape(B, P, self.patch_size))
        return torch.stack(out, dim=3).reshape(B, -1, C)

    def sample_stochastic(self, B, num_patches, C, device):
        """从先验 N(0,I) 采样 stochastic 分量 → [B, P*ps, C]"""
        N = B * num_patches * C
        z_s = torch.randn(N, self.stochastic_vae.latent_dim, device=device)
        s = self.stochastic_vae.decode(z_s)  # [N, 1, ps]
        return (
            s.squeeze(1)
             .reshape(B, num_patches, C, self.patch_size)
             .permute(0, 1, 3, 2)
             .reshape(B, -1, C)
        )
