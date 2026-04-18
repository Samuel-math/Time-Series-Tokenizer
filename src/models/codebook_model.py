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
from .vqvae import Encoder, Decoder, SparseNet
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
    
    def encode_to_indices(self, x, return_distances=False, return_sparse=False,
                          return_per_layer=False):
        """
        Args:
            x: [B, T, C]
            return_distances: 是否返回到码本第 0 层的距离（用于软索引计算）
            return_sparse: 是否返回稀疏分量 s [B, num_patches*patch_size, C]
            return_per_layer: 是否返回 per_layer_z_q: List[L] of [B, num_patches, C, code_dim]
                             （用于频率分工正则等需要 per-layer 量化向量的场景）
        Returns:
            indices: [B, num_patches, C, n_rq_layers]
            vq_loss: scalar
            z_q: [B, num_patches, C, code_dim]
            distances (optional)
            s_tensor (optional): [B, num_patches*patch_size, C] or None
            per_layer_z_q (optional): List[L] of [B, num_patches, C, code_dim]
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
        # per_layer_per_channel[c] = List[L] of [B, num_patches, code_dim]
        per_layer_per_channel = [] if return_per_layer else None
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

            if return_per_layer:
                vq_loss_c, z_q_sum_c, all_idx_c, per_layer_c = self.vq(
                    z_c_flat, return_per_layer=True
                )
                per_layer_per_channel.append([
                    zl.reshape(B, num_patches, self.code_dim) for zl in per_layer_c
                ])
            else:
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

        # 组装 per-layer z_q: List[L] of [B, num_patches, C, code_dim]
        if return_per_layer:
            per_layer_z_q = []
            for l in range(self.n_rq_layers):
                # stack across channels → [B, num_patches, C, code_dim]
                per_layer_z_q.append(torch.stack(
                    [per_layer_per_channel[c][l] for c in range(C)], dim=2
                ))
        else:
            per_layer_z_q = None

        # 组装稀疏分量张量
        if return_sparse:
            if s_list:
                s_tensor = torch.stack(s_list, dim=3)   # [B, num_patches, patch_size, C]
                s_tensor = s_tensor.reshape(B, -1, C)   # [B, num_patches*patch_size, C]
            else:
                s_tensor = None
            if return_distances and return_per_layer:
                distances = torch.cat(distances_list, dim=0)
                return indices, vq_loss, z_q, distances, s_tensor, per_layer_z_q
            if return_distances:
                distances = torch.cat(distances_list, dim=0)
                return indices, vq_loss, z_q, distances, s_tensor
            if return_per_layer:
                return indices, vq_loss, z_q, s_tensor, per_layer_z_q
            return indices, vq_loss, z_q, s_tensor

        if return_distances and return_per_layer:
            distances = torch.cat(distances_list, dim=0)
            return indices, vq_loss, z_q, distances, per_layer_z_q
        if return_distances:
            distances = torch.cat(distances_list, dim=0)
            return indices, vq_loss, z_q, distances
        if return_per_layer:
            return indices, vq_loss, z_q, per_layer_z_q

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

    def encode_to_indices(self, x, return_sparse=False, return_per_layer=False):
        """
        Args:
            x: [B, T, C]
            return_sparse: 是否返回稀疏分量 s [B, num_patches*patch_size, C]
            return_per_layer: 是否返回 per_layer_z_q: List[L] of [B, num_patches, C, code_dim]
        Returns:
            indices: [B, num_patches, C, n_rq_layers]
            vq_loss: scalar
            z_q: [B, num_patches, C, code_dim]
            s_tensor (optional)
            per_layer_z_q (optional)
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
        per_layer_per_channel = [] if return_per_layer else None
        vq_loss_sum = 0.0
        for c in range(C):
            z_c_flat = z_all[:, :, c, :].reshape(B * num_patches, self.code_dim)
            if return_per_layer:
                vq_loss_c, z_q_sum_c, all_idx_c, per_layer_c = self.vqs[c](
                    z_c_flat, return_per_layer=True
                )
                per_layer_per_channel.append([
                    zl.reshape(B, num_patches, self.code_dim) for zl in per_layer_c
                ])
            else:
                vq_loss_c, z_q_sum_c, all_idx_c = self.vqs[c](z_c_flat)
            vq_loss_sum += vq_loss_c
            indices_c = torch.stack(all_idx_c, dim=1).reshape(B, num_patches, self.n_rq_layers)
            indices_list.append(indices_c)
            z_q_list.append(z_q_sum_c.reshape(B, num_patches, self.code_dim))

        indices = torch.stack(indices_list, dim=2)  # [B, num_patches, C, n_rq_layers]
        z_q = torch.stack(z_q_list, dim=2)          # [B, num_patches, C, code_dim]
        vq_loss = vq_loss_sum / C

        if return_per_layer:
            per_layer_z_q = [
                torch.stack([per_layer_per_channel[c][l] for c in range(C)], dim=2)
                for l in range(self.n_rq_layers)
            ]
        else:
            per_layer_z_q = None

        if return_sparse:
            if s_list:
                s_tensor = torch.stack(s_list, dim=3)  # [B, num_patches, patch_size, C]
                s_tensor = s_tensor.reshape(B, -1, C)
            else:
                s_tensor = None
            if return_per_layer:
                return indices, vq_loss, z_q, s_tensor, per_layer_z_q
            return indices, vq_loss, z_q, s_tensor

        if return_per_layer:
            return indices, vq_loss, z_q, per_layer_z_q
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
