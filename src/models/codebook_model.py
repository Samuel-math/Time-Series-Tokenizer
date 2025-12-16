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
        init_method = config.get('vq_init_method', 'uniform')
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
            from .patch_vqvae_transformer import ResidualVectorQuantizer, ResidualVectorQuantizerEMA
            if config.get('codebook_ema', False):
                self.vq = ResidualVectorQuantizerEMA(
                    residual_vq_codebook_sizes, self.code_dim, self.commitment_cost,
                    decay=config.get('ema_decay', 0.99), eps=config.get('ema_eps', 1e-5),
                    num_layers=residual_vq_layers, init_method=init_method,
                    combine_method=residual_vq_combine_method
                )
            else:
                self.vq = ResidualVectorQuantizer(
                    residual_vq_codebook_sizes, self.code_dim, self.commitment_cost,
                    num_layers=residual_vq_layers, init_method=init_method,
                    combine_method=residual_vq_combine_method
                )
        else:
            # 使用单层量化（原始方式）
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
            
            # VQ（支持单层和残差量化）
            vq_result = self.vq(z_flat)
            if isinstance(vq_result[2], list):
                # 残差量化：返回多层索引列表
                vq_loss_c, z_q_flat_c, indices_list_c = vq_result
                # 将多层索引合并为单个tensor: [num_layers, B*num_patches] -> [B*num_patches, num_layers]
                indices_c = torch.stack(indices_list_c, dim=0).t()  # [B*num_patches, num_layers]
            else:
                # 单层量化：返回单个索引
                vq_loss_c, z_q_flat_c, indices_c = vq_result
                # 添加维度以保持一致性: [B*num_patches] -> [B*num_patches, 1]
                indices_c = indices_c.unsqueeze(1)  # [B*num_patches, 1]
            
            vq_loss_sum += vq_loss_c
            
            # Reshape: [B*num_patches, num_layers] -> [B, num_patches, num_layers]
            num_layers = indices_c.shape[1]
            indices_c = indices_c.reshape(B, num_patches, num_layers)  # [B, num_patches, num_layers]
            z_q_c = z_q_flat_c.reshape(B, num_patches, self.code_dim)  # [B, num_patches, code_dim]
            
            indices_list.append(indices_c)
            z_q_list.append(z_q_c)
        
        # 合并所有通道: [B, num_patches, C, num_layers] 和 [B, num_patches, C, code_dim]
        indices = torch.stack(indices_list, dim=2)  # [B, num_patches, C, num_layers]
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
