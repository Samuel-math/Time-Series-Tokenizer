"""
轻量级码本模型：只包含Encoder、VQ和Decoder
用于码本预训练，不包含Transformer等重型模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vqvae import Encoder, Decoder
from .patch_vqvae_transformer import FlattenedVectorQuantizer, FlattenedVectorQuantizerEMA


class ChannelAttention(nn.Module):
    """
    Channel Attention模块
    在通道维度上进行注意力机制，增强通道间的交互
    """
    def __init__(self, code_dim, n_channels, dropout=0.1):
        super().__init__()
        self.code_dim = code_dim
        self.n_channels = n_channels
        
        # 通道注意力：使用self-attention在通道维度
        # Query, Key, Value投影
        self.q_proj = nn.Linear(code_dim, code_dim)
        self.k_proj = nn.Linear(code_dim, code_dim)
        self.v_proj = nn.Linear(code_dim, code_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(code_dim, code_dim)
        
        # Layer norm和dropout
        self.norm = nn.LayerNorm(code_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = code_dim ** -0.5
    
    def forward(self, z):
        """
        Args:
            z: [B, num_patches, C, code_dim] encoder输出的所有通道表示
        Returns:
            z_attn: [B, num_patches, C, code_dim] 经过channel attention的表示
        """
        B, num_patches, C, code_dim = z.shape
        
        # Reshape: [B, num_patches, C, code_dim] -> [B*num_patches, C, code_dim]
        z_flat = z.reshape(B * num_patches, C, code_dim)
        
        # Query, Key, Value: [B*num_patches, C, code_dim]
        q = self.q_proj(z_flat)
        k = self.k_proj(z_flat)
        v = self.v_proj(z_flat)
        
        # 注意力计算: [B*num_patches, C, code_dim] x [B*num_patches, code_dim, C] -> [B*num_patches, C, C]
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B*num_patches, C, C]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力: [B*num_patches, C, C] x [B*num_patches, C, code_dim] -> [B*num_patches, C, code_dim]
        z_attn = torch.bmm(attn_weights, v)  # [B*num_patches, C, code_dim]
        
        # 输出投影和残差连接
        z_attn = self.out_proj(z_attn)  # [B*num_patches, C, code_dim]
        z_attn = z_attn + z_flat  # 残差连接
        z_attn = self.norm(z_attn)  # Layer norm
        
        # Reshape back: [B*num_patches, C, code_dim] -> [B, num_patches, C, code_dim]
        z_attn = z_attn.reshape(B, num_patches, C, code_dim)
        
        return z_attn


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
        
        # Channel Attention模块（在VQ之前）
        self.use_channel_attention = config.get('use_channel_attention', False)
        if self.use_channel_attention:
            self.channel_attention = ChannelAttention(
                code_dim=self.code_dim,
                n_channels=n_channels,
                dropout=config.get('channel_attention_dropout', 0.1)
            )
        else:
            self.channel_attention = None
        
        # VQ
        init_method = config.get('vq_init_method', 'uniform')
        if config.get('codebook_ema', False):
            self.vq = FlattenedVectorQuantizerEMA(
                self.codebook_size, self.code_dim, self.commitment_cost,
                decay=config.get('ema_decay', 0.99), eps=config.get('ema_eps', 1e-5),
                init_method=init_method
            )
        else:
            self.vq = FlattenedVectorQuantizer(
                self.codebook_size, self.code_dim, self.commitment_cost,
                init_method=init_method
            )
    
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
                    
                    # Encoder输出
                    z = self.encoder(x_c_flat, self.compression_factor)  # [B*num_patches, embedding_dim, compressed_len]
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
        
        # Channel Attention（如果启用）
        if self.use_channel_attention and self.channel_attention is not None:
            z_all = self.channel_attention(z_all)  # [B, num_patches, C, code_dim]
        
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
