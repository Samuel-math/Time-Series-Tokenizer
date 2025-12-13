"""
轻量级码本模型：只包含Encoder、VQ和Decoder
用于码本预训练，不包含Transformer等重型模块
"""

import torch
import torch.nn as nn
from .vqvae import Encoder, Decoder
from .patch_vqvae_transformer import FlattenedVectorQuantizer, FlattenedVectorQuantizerEMA


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
        
        # VQ
        if config.get('codebook_ema', False):
            self.vq = FlattenedVectorQuantizerEMA(
                self.codebook_size, self.code_dim, self.commitment_cost,
                decay=config.get('ema_decay', 0.99), eps=config.get('ema_eps', 1e-5)
            )
        else:
            self.vq = FlattenedVectorQuantizer(
                self.codebook_size, self.code_dim, self.commitment_cost
            )
    
    def encode_to_indices(self, x):
        """
        编码为码本索引和量化向量（channel-independent版本）
        
        Args:
            x: [B, T, C]
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
        indices_list = []
        z_q_list = []
        vq_loss_sum = 0
        
        for c in range(C):
            # 提取第c个通道的patches: [B, num_patches, patch_size]
            x_c = x_patches[:, :, :, c]  # [B, num_patches, patch_size]
            x_c_flat = x_c.reshape(B * num_patches, self.patch_size)  # [B*num_patches, patch_size]
            x_c_flat = x_c_flat.unsqueeze(1)  # [B*num_patches, 1, patch_size] (单通道输入)
            
            # VQVAE Encoder (单通道输入)
            z = self.encoder(x_c_flat, self.compression_factor)  # [B*num_patches, embedding_dim, compressed_len]
            z_flat = z.reshape(B * num_patches, -1)  # [B*num_patches, code_dim]
            
            # VQ
            vq_loss_c, z_q_flat_c, indices_c = self.vq(z_flat)
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
