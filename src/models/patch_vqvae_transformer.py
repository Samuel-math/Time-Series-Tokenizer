"""
Patch-based VQVAE + Transformer 模型架构 (优化版)

架构说明：
1. 输入: [B, T, C] 时间序列
2. Patch划分 + VQVAE Encoder -> 展平为 [B, num_patches, C, code_dim]
3. VQ 量化后的表示直接作为 Transformer 输入 (无需 token embedding)
4. Transformer (Decoder-only): 预测下一个码本向量
5. 预训练: NTP loss (预测码本索引)
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
    def __init__(self, codebook_size, code_dim, commitment_cost=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        
        # 码本: [codebook_size, code_dim]
        self.embedding = nn.Embedding(codebook_size, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
    
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
    def __init__(self, codebook_size, code_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        
        # 码本权重与EMA状态
        embed = torch.randn(codebook_size, code_dim)
        embed.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.embedding = nn.Embedding(codebook_size, code_dim)
        self.embedding.weight.data.copy_(embed)
        self.embedding.weight.requires_grad = False
        
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_w', embed.clone())
    
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
        
        if self.training:
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


class PatchSelfAttention(nn.Module):
    """
    处理patch内时间信息的Self-Attention层
    对每个patch内的patch_size个时间步应用self-attention
    输入: [B*num_patches, patch_size, C]
    输出: [B*num_patches, patch_size, C]
    attention在patch_size个时间步之间进行，每个时间步有C个特征
    """
    def __init__(self, patch_size, n_channels, n_heads=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_heads = n_heads
        
        # 位置编码（patch内的时间位置），维度为C
        self.pos_embedding = nn.Embedding(patch_size, n_channels)
        
        # Self-attention层，embed_dim = C
        self.attention = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm和dropout
        self.norm1 = nn.LayerNorm(n_channels)
        self.norm2 = nn.LayerNorm(n_channels)
        self.dropout = nn.Dropout(dropout)
        
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
        # 位置编码
        positions = torch.arange(self.patch_size, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        x_pos = x + self.pos_embedding(positions)
        
        # Self-attention (bidirectional, 因为patch内的时间信息应该可以互相看到)
        attn_out, _ = self.attention(x_pos, x_pos, x_pos)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class CausalTransformer(nn.Module):
    """轻量级 Causal Transformer，输入维度为 code_dim"""
    def __init__(self, code_dim, n_heads, n_layers, d_ff, dropout=0.1, max_len=512):
        super().__init__()
        self.code_dim = code_dim
        
        # 位置编码，维度与 code_dim 一致
        self.pos_embedding = nn.Embedding(max_len, code_dim)
        self.drop = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=code_dim, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(code_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, code_dim] 直接是量化后的码本向量
        """
        B, T, _ = x.shape
        
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)
        x = self.drop(x)
        
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.norm(x)


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
        
        self.compressed_len = self.patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len  # Transformer 输入维度
        
        # Patch内Self-Attention配置
        self.use_patch_attention = config.get('use_patch_attention', False)
        self.patch_attention_heads = config.get('patch_attention_heads', 4)
        self.patch_attention_n_channels = None  # 延迟初始化，在第一次forward时确定
        self.patch_attention = None
        
        # VQVAE Encoder/Decoder
        self.encoder = Encoder(
            in_channels=1,
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
            compression_factor=self.compression_factor
        )
        
        # VQ (码本维度 = code_dim)
        if self.use_codebook_ema:
            self.vq = FlattenedVectorQuantizerEMA(
                self.codebook_size, self.code_dim, self.commitment_cost,
                decay=self.ema_decay, eps=self.ema_eps
            )
        else:
            self.vq = FlattenedVectorQuantizer(
                self.codebook_size, self.code_dim, self.commitment_cost
            )
        
        # Transformer (输入维度 = code_dim，无需 token embedding)
        self.transformer = CausalTransformer(
            self.code_dim, self.n_heads, self.n_layers, 
            self.d_ff, self.dropout
        )
        
        # 输出头: code_dim -> codebook_size (预测码本索引)
        self.output_head = nn.Linear(self.code_dim, self.codebook_size)
    
    def encode_to_indices(self, x, return_processed_patches=False):
        """
        编码为码本索引和量化向量
        
        Args:
            x: [B, T, C]
            return_processed_patches: 是否返回经过self-attention处理后的patches
        Returns:
            indices: [B, num_patches, C]
            vq_loss: scalar
            z_q: [B, num_patches, C, code_dim]
            x_patches_processed: [B, num_patches, patch_size, C] (如果return_processed_patches=True)
        """
        B, T, C = x.shape
        num_patches = T // self.patch_size
        
        # 重组为 patches: [B, num_patches, patch_size, C]
        x = x[:, :num_patches * self.patch_size, :]
        x_patches = x.reshape(B, num_patches, self.patch_size, C)
        
        # 应用Patch内Self-Attention（如果启用）
        if self.use_patch_attention:
            # 对每个patch，在patch_size × C上做attention
            # [B, num_patches, patch_size, C] -> [B*num_patches, patch_size, C]
            x_patches_flat = x_patches.reshape(B * num_patches, self.patch_size, C)
            
            # 如果patch_attention还未初始化，现在初始化（因为需要知道C的值）
            if self.patch_attention is None or self.patch_attention_n_channels != C:
                self.patch_attention_n_channels = C
                self.patch_attention = PatchSelfAttention(
                    patch_size=self.patch_size,
                    n_channels=C,
                    n_heads=self.patch_attention_heads,
                    dropout=self.dropout
                ).to(x_patches.device)
            
            # 应用self-attention: [B*num_patches, patch_size, C] -> [B*num_patches, patch_size, C]
            x_out = self.patch_attention(x_patches_flat)
            
            # 恢复形状: [B*num_patches, patch_size, C] -> [B, num_patches, patch_size, C]
            x_patches = x_out.reshape(B, num_patches, self.patch_size, C)
        
        # 保存处理后的patches（用于重构损失）
        x_patches_processed = x_patches if return_processed_patches else None
        
        # 重组为 [B*num_patches*C, patch_size] 用于编码
        x = x_patches.permute(0, 1, 3, 2).reshape(-1, self.patch_size)  # [B*num_patches*C, patch_size]
        
        # VQVAE Encoder
        z = self.encoder(x, self.compression_factor)  # [B*num_patches*C, embedding_dim, compressed_len]
        z_flat = z.reshape(z.shape[0], -1)  # [B*num_patches*C, code_dim]
        
        # VQ
        vq_loss, z_q_flat, indices = self.vq(z_flat)
        
        # Reshape
        indices = indices.reshape(B, num_patches, C)
        z_q = z_q_flat.reshape(B, num_patches, C, self.code_dim)
        
        if return_processed_patches:
            return indices, vq_loss, z_q, x_patches_processed
        return indices, vq_loss, z_q
    
    def decode_from_codes(self, z_q):
        """
        从量化向量解码
        
        Args:
            z_q: [B, num_patches, C, code_dim]
        Returns:
            x_recon: [B, num_patches * patch_size, C]
        """
        B, num_patches, C, _ = z_q.shape
        
        # Reshape for decoder
        z_q = z_q.reshape(-1, self.embedding_dim, self.compressed_len)
        
        # VQVAE Decoder
        x_recon = self.decoder(z_q, self.compression_factor)
        
        # Reshape back
        x_recon = x_recon.reshape(B, num_patches, C, self.patch_size)
        x_recon = x_recon.permute(0, 1, 3, 2).reshape(B, -1, C)
        
        return x_recon
    
    def forward_pretrain(self, x):
        """
        预训练: NTP
        直接用量化后的码本向量 z_q 作为 Transformer 输入
        """
        B, T, C = x.shape
        
        # 编码（返回处理后的patches用于重构损失）
        if self.use_patch_attention:
            indices, vq_loss, z_q, x_patches_processed = self.encode_to_indices(x, return_processed_patches=True)
        else:
            indices, vq_loss, z_q = self.encode_to_indices(x, return_processed_patches=False)
            x_patches_processed = None
        num_patches = indices.shape[1]
        
        # 重构损失
        x_recon = self.decode_from_codes(z_q)  # [B, num_patches * patch_size, C]
        
        # 如果使用了patch attention，计算处理后的patches与重构结果的损失
        if self.use_patch_attention and x_patches_processed is not None:
            # 将处理后的patches reshape为 [B, num_patches * patch_size, C]
            x_processed = x_patches_processed.reshape(B, num_patches * self.patch_size, C)
            recon_loss = F.mse_loss(x_recon, x_processed[:, :x_recon.shape[1], :])
        else:
            # 否则使用原始输入
            recon_loss = F.mse_loss(x_recon, x[:, :x_recon.shape[1], :])
        
        # Channel-independent Transformer
        # 直接使用 z_q 作为输入，不需要 token embedding
        all_logits = []
        for c in range(C):
            z_c = z_q[:, :, c, :]  # [B, num_patches, code_dim]
            h = self.transformer(z_c)  # [B, num_patches, code_dim]
            logits = self.output_head(h)  # [B, num_patches, codebook_size]
            all_logits.append(logits)
        
        logits = torch.stack(all_logits, dim=2)  # [B, num_patches, C, codebook_size]
        
        # NTP: 用位置i预测位置i+1
        return logits[:, :-1], indices[:, 1:], vq_loss, recon_loss
    
    def forward_finetune(self, x, target_len):
        """
        微调: 预测未来序列（优化版）
        
        1. 编码输入为量化向量
        2. Transformer 自回归预测未来 patch 的码本索引
        3. 从码本获取向量并解码
        
        优化：
        - 批量处理所有channels，而不是逐个处理
        - 只计算最后一个位置的transformer输出
        """
        B, T, C = x.shape
        num_pred_patches = (target_len + self.patch_size - 1) // self.patch_size
        
        # 编码输入
        indices, vq_loss, z_q = self.encode_to_indices(x)  # z_q: [B, num_patches, C, code_dim]
        num_input_patches = z_q.shape[1]
        
        # 自回归预测（优化版：批量处理channels）
        current_z = z_q  # [B, num_patches, C, code_dim]
        pred_codes_list = []
        
        for _ in range(num_pred_patches):
            # 批量处理所有channels: [B, seq_len, C, code_dim] -> [B*C, seq_len, code_dim]
            B_p, seq_len, C_p, code_dim = current_z.shape
            z_flat = current_z.permute(0, 2, 1, 3).reshape(B_p * C_p, seq_len, code_dim)
            
            # Transformer处理: [B*C, seq_len, code_dim]
            h = self.transformer(z_flat)  # [B*C, seq_len, code_dim]
            
            # 只取最后一个位置: [B*C, code_dim]
            h_last = h[:, -1, :]  # [B*C, code_dim]
            
            # 输出头: [B*C, codebook_size]
            logits = self.output_head(h_last)  # [B*C, codebook_size]
            
            # 使用 softmax + 加权求和 替代 argmax，保持可微分
            weights = F.softmax(logits, dim=-1)  # [B*C, codebook_size]
            codebook = self.vq.embedding.weight  # [codebook_size, code_dim]
            pred_code_flat = torch.matmul(weights, codebook)  # [B*C, code_dim]
            
            # 恢复形状: [B*C, code_dim] -> [B, C, code_dim]
            next_codes = pred_code_flat.reshape(B_p, C_p, code_dim)
            
            pred_codes_list.append(next_codes)
            
            # 更新序列: [B, C, code_dim] -> [B, 1, C, code_dim]
            current_z = torch.cat([current_z, next_codes.unsqueeze(1)], dim=1)
        
        # 获取预测的码本向量: [B, num_pred_patches, C, code_dim]
        pred_codes = torch.stack(pred_codes_list, dim=1)
        
        # 解码
        pred = self.decode_from_codes(pred_codes)  # [B, num_pred_patches*patch_size, C]
        pred = pred[:, :target_len, :]
        
        return pred, vq_loss
    
    def forward(self, x, target_len=None, mode='pretrain'):
        if mode == 'pretrain':
            return self.forward_pretrain(x)
        else:
            return self.forward_finetune(x, target_len)
    
    @torch.no_grad()
    def get_codebook_usage(self, x):
        result = self.encode_to_indices(x, return_processed_patches=False)
        if isinstance(result, tuple) and len(result) == 4:
            indices, _, _, _ = result
        else:
            indices, _, _ = result
        unique = torch.unique(indices.reshape(-1))
        return len(unique) / self.codebook_size, unique
    
    def load_vqvae_weights(self, checkpoint_path, device='cpu', load_vq=True):
        """
        加载预训练的VQVAE权重
        
        Args:
            checkpoint_path: checkpoint路径
            device: 设备
            load_vq: 是否加载VQ层权重
        """
        import os
        if not os.path.exists(checkpoint_path):
            print(f"警告: checkpoint不存在: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 处理checkpoint格式：可能是模型对象或字典
            if isinstance(checkpoint, dict):
                # 字典格式：尝试从model_state_dict加载
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                # 模型对象格式
                if hasattr(checkpoint, 'state_dict'):
                    state_dict = checkpoint.state_dict()
                else:
                    state_dict = None
            
            # 加载encoder
            encoder_loaded = False
            if state_dict is not None:
                encoder_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() 
                               if k.startswith('encoder.')}
                if encoder_dict:
                    self.encoder.load_state_dict(encoder_dict, strict=False)
                    encoder_loaded = True
                    print("✓ 加载 Encoder 权重成功")
            elif hasattr(checkpoint, 'encoder'):
                self.encoder.load_state_dict(checkpoint.encoder.state_dict(), strict=False)
                encoder_loaded = True
                print("✓ 加载 Encoder 权重成功")
            
            # 加载decoder
            decoder_loaded = False
            if state_dict is not None:
                decoder_dict = {k.replace('decoder.', ''): v for k, v in state_dict.items() 
                               if k.startswith('decoder.')}
                if decoder_dict:
                    self.decoder.load_state_dict(decoder_dict, strict=False)
                    decoder_loaded = True
                    print("✓ 加载 Decoder 权重成功")
            elif hasattr(checkpoint, 'decoder'):
                self.decoder.load_state_dict(checkpoint.decoder.state_dict(), strict=False)
                decoder_loaded = True
                print("✓ 加载 Decoder 权重成功")
            
            # 加载VQ层
            if load_vq:
                vq_loaded = False
                if state_dict is not None:
                    # 尝试加载vq.embedding或vq._embedding
                    vq_keys = [k for k in state_dict.keys() if 'vq' in k.lower() and 'embedding' in k.lower()]
                    if vq_keys:
                        # 找到embedding权重
                        for key in vq_keys:
                            if 'weight' in key or 'embedding' in key:
                                try:
                                    vq_weight = state_dict[key]
                                    if hasattr(self.vq, 'embedding'):
                                        self.vq.embedding.weight.data.copy_(vq_weight)
                                        vq_loaded = True
                                        print("✓ 加载 VQ 码本权重成功")
                                        break
                                except Exception as e:
                                    continue
                elif hasattr(checkpoint, 'vq'):
                    try:
                        if hasattr(checkpoint.vq, 'embedding'):
                            self.vq.embedding.weight.data.copy_(checkpoint.vq.embedding.weight.data)
                            vq_loaded = True
                            print("✓ 加载 VQ 码本权重成功")
                        elif hasattr(checkpoint.vq, '_embedding'):
                            self.vq.embedding.weight.data.copy_(checkpoint.vq._embedding.weight.data)
                            vq_loaded = True
                            print("✓ 加载 VQ 码本权重成功")
                    except Exception as e:
                        print(f"⚠ 加载 VQ 权重时出错: {e}")
            
            return encoder_loaded or decoder_loaded
        except Exception as e:
            print(f"✗ 加载权重失败: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============ 工具函数 ============

def get_model_config(args):
    """构建模型配置"""
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
    }
    
    # Patch内Self-Attention配置
    if hasattr(args, 'use_patch_attention'):
        config['use_patch_attention'] = bool(args.use_patch_attention)
        config['patch_attention_heads'] = getattr(args, 'patch_attention_heads', 4)
    
    return config
