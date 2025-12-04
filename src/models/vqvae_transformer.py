"""
VQVAE + Transformer 模型架构
用于预训练和微调任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .vqvae import vqvae, Encoder, VectorQuantizer
from .layers.attention import MultiheadAttention
from .layers.basics import *
from .layers.pos_encoding import *


class VQVAETransformerPretrain(nn.Module):
    """
    预训练模型：VQVAE Encoder + Transformer (mask attention) + Codebook Prediction Head
    
    输入: [B, T, C] 时间序列
    输出: [B, codebook_size, T/compression_factor, C] codebook概率分布
    """
    def __init__(self, vqvae_config, transformer_config, 
                 load_vqvae_weights=False, vqvae_checkpoint_path=None, device='cpu'):
        """
        Args:
            vqvae_config: VQVAE配置字典
            transformer_config: Transformer配置字典
            load_vqvae_weights: 是否加载预训练的VQVAE权重
            vqvae_checkpoint_path: VQVAE模型checkpoint路径（如果load_vqvae_weights=True）
            device: 加载模型时的设备
        """
        super().__init__()
        
        # VQVAE配置
        self.embedding_dim = vqvae_config['embedding_dim']
        self.num_embeddings = vqvae_config['num_embeddings']
        self.compression_factor = vqvae_config['compression_factor']
        
        # Transformer配置
        self.d_model = transformer_config.get('d_model', 128)
        self.n_layers = transformer_config.get('n_layers', 3)
        self.n_heads = transformer_config.get('n_heads', 8)
        self.d_ff = transformer_config.get('d_ff', 256)
        self.dropout = transformer_config.get('dropout', 0.1)
        self.attn_dropout = transformer_config.get('attn_dropout', 0.1)
        
        # 为每个通道创建VQVAE encoder（channel independent）
        # 注意：这里我们共享同一个encoder，因为每个通道独立处理
        self.vqvae_encoder = Encoder(
            1, 
            vqvae_config['block_hidden_size'],
            vqvae_config['num_residual_layers'],
            vqvae_config['res_hidden_size'],
            self.embedding_dim,
            self.compression_factor
        )
        
        # VQ层（用于获取codebook，但预训练时可能冻结）
        self.vq = VectorQuantizer(
            self.num_embeddings,
            self.embedding_dim,
            vqvae_config['commitment_cost']
        )
        
        # 加载预训练的VQVAE权重
        if load_vqvae_weights and vqvae_checkpoint_path is not None:
            self._load_vqvae_weights(vqvae_checkpoint_path, device)
        
        # 投影层：将embedding_dim投影到d_model
        self.projection = nn.Linear(self.embedding_dim, self.d_model)
        
        # 位置编码（会在forward中动态创建）
        self.pe = 'zeros'
        self.learn_pe = True
        
        # Transformer Encoder（mask attention）
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout
            ) for _ in range(self.n_layers)
        ])
        
        # Codebook预测头
        self.codebook_head = nn.Linear(self.d_model, self.num_embeddings)
        
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def _load_vqvae_weights(self, checkpoint_path, device='cpu'):
        """
        从checkpoint加载预训练的VQVAE权重
        
        Args:
            checkpoint_path: VQVAE模型checkpoint路径（可以是vqvae_pretrain.py保存的完整模型）
            device: 加载模型时的设备
        """
        import os
        
        if not os.path.exists(checkpoint_path):
            print(f"警告: VQVAE checkpoint路径不存在: {checkpoint_path}")
            print("将使用随机初始化的VQVAE权重")
            return
        
        try:
            # 加载VQVAE模型
            # vqvae_pretrain.py保存的是完整的模型对象: torch.save(model, path)
            vqvae_model = torch.load(checkpoint_path, map_location=device)
            
            # 如果checkpoint是字典格式（包含state_dict），需要提取模型
            if isinstance(vqvae_model, dict):
                if 'model_state_dict' in vqvae_model:
                    # 如果是保存的state_dict格式，需要重新构建模型
                    from .vqvae import vqvae as VQVAE
                    # 从当前encoder获取配置信息
                    temp_model = VQVAE({
                        'block_hidden_size': getattr(self.vqvae_encoder, '_conv_1', None) and self.vqvae_encoder._conv_1.out_channels * 2 or 128,
                        'num_residual_layers': 2,
                        'res_hidden_size': 64,
                        'embedding_dim': self.embedding_dim,
                        'num_embeddings': self.num_embeddings,
                        'commitment_cost': 0.25,
                        'compression_factor': self.compression_factor
                    })
                    temp_model.load_state_dict(vqvae_model['model_state_dict'])
                    vqvae_model = temp_model
                elif 'encoder' in vqvae_model:
                    # 如果字典中直接包含encoder和vq
                    vqvae_model = type('VQVAE', (), vqvae_model)()
                else:
                    print("警告: checkpoint格式可能不正确，尝试直接加载...")
                    return
            
            # 检查是否是vqvae模型对象（有encoder和vq属性）
            # vqvae_pretrain.py使用torch.save(model, path)保存完整模型对象
            if not hasattr(vqvae_model, 'encoder') or not hasattr(vqvae_model, 'vq'):
                print("警告: checkpoint中未找到encoder或vq属性")
                print(f"checkpoint类型: {type(vqvae_model)}")
                print("将使用随机初始化的VQVAE权重")
                return
            
            # 加载encoder权重
            try:
                encoder_state_dict = vqvae_model.encoder.state_dict()
                self.vqvae_encoder.load_state_dict(encoder_state_dict, strict=True)
                print(f"✓ VQVAE Encoder权重已成功加载 (从 {checkpoint_path})")
            except Exception as e:
                print(f"✗ VQVAE Encoder权重加载失败: {e}")
                print("将使用随机初始化的Encoder权重")
            
            # 加载VQ（codebook）权重
            try:
                vq_state_dict = vqvae_model.vq.state_dict()
                self.vq.load_state_dict(vq_state_dict, strict=True)
                print(f"✓ VQ Codebook权重已成功加载 (从 {checkpoint_path})")
            except Exception as e:
                print(f"✗ VQ Codebook权重加载失败: {e}")
                print("将使用随机初始化的Codebook权重")
                
        except Exception as e:
            print(f"加载VQVAE权重时出错: {e}")
            import traceback
            traceback.print_exc()
            print("将使用随机初始化的VQVAE权重")
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, C] 输入时间序列
            mask: [B, T/compression_factor] 可选，mask attention mask
                  如果为None，会尝试从模型属性中获取（用于Learner框架兼容）
        
        Returns:
            logits: [B, codebook_size, T/compression_factor, C] codebook概率logits
        """
        # 如果mask为None，尝试从模型属性获取（用于Learner框架）
        if mask is None and hasattr(self, '_current_mask'):
            mask = self._current_mask
        
        B, T, C = x.shape
        
        # 对每个通道分别编码
        encoded_features = []
        for ch in range(C):
            x_ch = x[:, :, ch]  # [B, T]
            x_ch = x_ch.view(B, T)
            
            # VQVAE encoder
            z = self.vqvae_encoder(x_ch, self.compression_factor)  # [B, embedding_dim, T/compression_factor]
            z = z.permute(0, 2, 1)  # [B, T/compression_factor, embedding_dim]
            
            encoded_features.append(z)
        
        # Stack: [B, T/compression_factor, C, embedding_dim]
        z = torch.stack(encoded_features, dim=2)  # [B, T/compression_factor, C, embedding_dim]
        
        # 投影到d_model
        z = self.projection(z)  # [B, T/compression_factor, C, d_model]
        
        # Channel independent处理：对每个通道分别通过transformer
        T_compressed = z.shape[1]
        outputs = []
        
        for ch in range(C):
            z_ch = z[:, :, ch, :]  # [B, T/compression_factor, d_model]
            
            # 位置编码
            W_pos = positional_encoding(self.pe, self.learn_pe, T_compressed, self.d_model)
            z_ch = z_ch + W_pos.to(z_ch.device)
            z_ch = self.dropout_layer(z_ch)
            
            # Transformer layers with mask attention
            for layer in self.transformer_layers:
                z_ch = layer(z_ch, mask=mask)
            
            outputs.append(z_ch)
        
        # Stack: [B, T/compression_factor, C, d_model]
        z = torch.stack(outputs, dim=2)
        
        # Codebook预测
        logits = self.codebook_head(z)  # [B, T/compression_factor, C, codebook_size]
        logits = logits.permute(0, 3, 1, 2)  # [B, codebook_size, T/compression_factor, C]
        
        # 应用softmax得到每个码本元素的概率
        probs = F.softmax(logits, dim=1)  # [B, codebook_size, T/compression_factor, C]
        
        return probs


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with mask attention support"""
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model, n_heads, 
            attn_dropout=attn_dropout, 
            proj_dropout=dropout
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, d_model]
            mask: [B, T] or [T, T] attention mask
        """
        # Self-attention with mask
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:  # [B, T]
                # 转换为causal mask或padding mask
                attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            elif mask.dim() == 3:  # [B, T, T]
                attn_mask = mask
        
        x2, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        # Feed-forward
        x2 = self.ff(x)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        
        return x


class VQVAETransformerFinetune(nn.Module):
    """
    微调模型：预训练的Transformer + VQVAE Decoder
    
    输入: [B, T, C] 时间序列
    输出: [B, target_len, C] 预测结果
    """
    def __init__(self, pretrained_model, vqvae_decoder, vqvae_config, 
                 freeze_transformer=True, freeze_decoder=True, add_finetune_head=False):
        super().__init__()
        
        # 使用预训练的transformer部分
        self.pretrained_model = pretrained_model
        
        # VQVAE decoder
        self.decoder = vqvae_decoder
        
        self.compression_factor = vqvae_config['compression_factor']
        self.num_embeddings = vqvae_config['num_embeddings']
        self.embedding_dim = vqvae_config['embedding_dim']
        
        # 冻结参数
        if freeze_transformer:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        
        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
        
        # 可选：添加微调头
        self.add_finetune_head = add_finetune_head
        if add_finetune_head:
            self.finetune_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
        
    def forward(self, x, target_len):
        """
        Args:
            x: [B, T, C] 输入时间序列
            target_len: 预测长度
        
        Returns:
            pred: [B, target_len, C] 预测结果
        """
        B, T, C = x.shape
        
        # 获取codebook logits
        with torch.no_grad():
            logits = self.pretrained_model(x)  # [B, codebook_size, T/compression_factor, C]
        
        # 选择最可能的codebook索引
        codebook_indices = torch.argmax(logits, dim=1)  # [B, T/compression_factor, C]
        
        # 从VQ层获取codebook embeddings
        codebook_embeddings = self.pretrained_model.vq._embedding.weight  # [num_embeddings, embedding_dim]
        
        # 对每个通道分别解码
        T_compressed = codebook_indices.shape[1]
        decoded = []
        
        for ch in range(C):
            indices_ch = codebook_indices[:, :, ch]  # [B, T/compressed]
            
            # 获取对应的embeddings
            indices_flat = indices_ch.view(-1)  # [B*T/compressed]
            quantized_ch = codebook_embeddings[indices_flat]  # [B*T/compressed, embedding_dim]
            quantized_ch = quantized_ch.view(B, -1, self.embedding_dim)  # [B, T/compressed, embedding_dim]
            
            # 如果添加了finetune head，通过它处理
            if self.add_finetune_head:
                quantized_ch = self.finetune_head(quantized_ch)
            
            # 转换为decoder需要的格式 [B, embedding_dim, T/compressed]
            quantized_ch = quantized_ch.permute(0, 2, 1)
            
            # 通过decoder解码
            decoded_ch = self.decoder(quantized_ch, self.compression_factor)  # [B, T]
            decoded.append(decoded_ch)
        
        # Stack: [B, T, C]
        pred = torch.stack(decoded, dim=2)  # [B, T, C]
        
        # 只返回target_len长度的预测
        if pred.shape[1] > target_len:
            pred = pred[:, :target_len, :]
        elif pred.shape[1] < target_len:
            # 如果预测长度不够，进行插值或重复最后一个值
            last_val = pred[:, -1:, :].repeat(1, target_len - pred.shape[1], 1)
            pred = torch.cat([pred, last_val], dim=1)
        
        return pred

