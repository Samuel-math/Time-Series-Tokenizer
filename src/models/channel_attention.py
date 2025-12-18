"""
Channel Attention模块
在通道维度上进行注意力机制，增强通道间的交互
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # 输出投影
        z_attn = self.out_proj(z_attn)
        
        # Residual connection
        z_attn = z_attn + z_flat
        
        # Layer norm
        z_attn = self.norm(z_attn)
        
        # Reshape back: [B*num_patches, C, code_dim] -> [B, num_patches, C, code_dim]
        z_attn = z_attn.reshape(B, num_patches, C, code_dim)
        
        return z_attn
