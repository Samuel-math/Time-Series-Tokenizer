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
                print(f"VQVAE Encoder权重已成功加载 (从 {checkpoint_path})")
            except Exception as e:
                print(f"VQVAE Encoder权重加载失败: {e}")
                print("将使用随机初始化的Encoder权重")
            
            # 加载VQ（codebook）权重
            try:
                vq_state_dict = vqvae_model.vq.state_dict()
                self.vq.load_state_dict(vq_state_dict, strict=True)
                print(f"VQ Codebook权重已成功加载 (从 {checkpoint_path})")
            except Exception as e:
                print(f"VQ Codebook权重加载失败: {e}")
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
            probs: [B, codebook_size, T/compression_factor, C] codebook概率
            encoding_indices: [B, T/compression_factor, C] 对应每个位置的码本索引（可选返回）
        """
        # 如果mask为None，尝试从模型属性获取（用于Learner框架）
        if mask is None and hasattr(self, '_current_mask'):
            mask = self._current_mask

        B, T, C = x.shape
        
        # 对每个通道分别编码
        encoded_features = []
        encoding_indices = []
        for ch in range(C):
            x_ch = x[:, :, ch]  # [B, T]
            x_ch = x_ch.view(B, T)
            
            # VQVAE encoder
            z = self.vqvae_encoder(x_ch, self.compression_factor)  # [B, embedding_dim, T/compression_factor]
            z = z.permute(0, 2, 1)  # [B, T/compression_factor, embedding_dim]
            
            # 找到码本（VQ）最近的索引
            with torch.no_grad():
                # z [B, T/compression_factor, embedding_dim] -> [B, embedding_dim, T/compression_factor]
                z_for_vq = z.permute(0, 2, 1).contiguous() # [B, embedding_dim, T/compression_factor]
                _, _, _, _, ind, _ = self.vq(z_for_vq)
                # ind: [B * T/compression_factor, 1]
                ind = ind.view(B, -1)  # [B, T/compression_factor]
                encoding_indices.append(ind)
            
            encoded_features.append(z)
        
        # Stack: [B, T/compression_factor, C, embedding_dim]
        z = torch.stack(encoded_features, dim=2)  # [B, T/compression_factor, C, embedding_dim]
        encoding_indices = torch.stack(encoding_indices, dim=2)  # [B, T/compression_factor, C]
        
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
        
        # 返回概率和对应的编码索引（如果直接用于训练可只返回probs，根据需要可返回encoding_indices）
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
            if mask.dim() == 2:  # [T, T]
                # 转换为causal mask
                attn_mask = mask.unsqueeze(0)  # [1, T, T]
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
        device = x.device
        
        # Step 1: 通过 VQVAE encoder 生成 [B, codebook_size, T/compression_factor, C] 的概率分布
        with torch.no_grad():
            # 使用预训练模型的 encoder 部分获取 codebook 概率分布
            codebook_probs = self.pretrained_model(x)  # [B, codebook_size, T/compression_factor, C]
        
        # Step 2: 对每个通道独立，使用 decoder-only 生成模式生成未来 tokens
        T_compressed = codebook_probs.shape[2]
        target_T_compressed = target_len // self.compression_factor
        
        # 创建 causal mask 用于生成（decoder only 模式）
        total_len = T_compressed + target_T_compressed
        causal_mask = torch.triu(torch.ones(total_len, total_len, device=device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        
        # 对每个通道分别生成
        generated_probs = []
        
        for ch in range(C):
            # 当前通道的 codebook 概率分布: [B, codebook_size, T/compression_factor]
            probs_ch = codebook_probs[:, :, :, ch]  # [B, codebook_size, T/compression_factor]
            
            # 将概率分布转换为 logits（取对数）
            logits_ch = torch.log(probs_ch + 1e-10)  # [B, codebook_size, T/compression_factor]
            logits_ch = logits_ch.permute(0, 2, 1)  # [B, T/compression_factor, codebook_size]
            
            # 投影到 d_model 维度
            # 使用预训练模型的 projection 层
            # 首先需要将 codebook indices 转换为 embeddings
            # 选择最可能的 codebook index
            codebook_indices = torch.argmax(probs_ch, dim=1)  # [B, T/compression_factor]
            
            # 从 VQ 层获取 codebook embeddings
            codebook_embeddings = self.pretrained_model.vq._embedding.weight  # [num_embeddings, embedding_dim]
            
            # 获取对应的 embeddings
            indices_flat = codebook_indices.view(-1)  # [B*T/compression_factor]
            z_ch = codebook_embeddings[indices_flat]  # [B*T/compression_factor, embedding_dim]
            z_ch = z_ch.view(B, T_compressed, self.embedding_dim)  # [B, T/compression_factor, embedding_dim]
            
            # 投影到 d_model
            z_ch = self.pretrained_model.projection(z_ch)  # [B, T/compression_factor, d_model]
            
            # 位置编码
            W_pos = positional_encoding(
                self.pretrained_model.pe, 
                self.pretrained_model.learn_pe, 
                total_len, 
                self.pretrained_model.d_model
            )
            z_ch = z_ch + W_pos[:T_compressed].to(device)
            z_ch = self.pretrained_model.dropout_layer(z_ch)
            
            # 自回归生成未来的 tokens
            generated_tokens = []
            current_z = z_ch  # [B, T/compression_factor, d_model]
            
            for step in range(target_T_compressed):
                # 对当前整个序列（包括已生成的）通过 Transformer layers
                current_seq_len = current_z.shape[1]
                z_processed = current_z
                
                # 创建当前序列的 causal mask
                seq_mask = causal_mask[:current_seq_len, :current_seq_len]
                
                # 通过所有 Transformer layers（decoder only，使用 causal mask）
                for layer in self.pretrained_model.transformer_layers:
                    z_processed = layer(z_processed, mask=seq_mask)
                
                # 获取最后一个位置的表示（用于预测下一个 token）
                last_z = z_processed[:, -1:, :]  # [B, 1, d_model]
                
                # 更新 current_z 为处理后的序列（用于下一次迭代，避免重复处理）
                current_z = z_processed
                
                # 预测下一个 codebook token 的概率分布
                next_logits = self.pretrained_model.codebook_head(last_z)  # [B, 1, codebook_size]
                next_probs = F.softmax(next_logits, dim=-1)  # [B, 1, codebook_size]
                
                # 选择最可能的 codebook index
                next_index = torch.argmax(next_probs, dim=-1)  # [B, 1]
                
                # 获取对应的 embedding
                next_index_flat = next_index.view(-1)  # [B]
                next_embedding = codebook_embeddings[next_index_flat]  # [B, embedding_dim]
                next_embedding = next_embedding.unsqueeze(1)  # [B, 1, embedding_dim]
                
                # 投影到 d_model
                next_z = self.pretrained_model.projection(next_embedding)  # [B, 1, d_model]
                
                # 添加位置编码
                pos_idx = T_compressed + step
                next_z = next_z + W_pos[pos_idx:pos_idx+1].to(device)
                next_z = self.pretrained_model.dropout_layer(next_z)
                
                # 添加到序列中（用于下一次迭代）
                current_z = torch.cat([current_z, next_z], dim=1)  # [B, current_seq_len+1, d_model]
                generated_tokens.append(next_probs)
            
            # 合并所有生成的概率分布
            if generated_tokens:
                generated_probs_ch = torch.cat(generated_tokens, dim=1)  # [B, target_T/compression_factor, codebook_size]
                generated_probs_ch = generated_probs_ch.permute(0, 2, 1)  # [B, codebook_size, target_T/compression_factor]
            else:
                # 如果没有生成任何 token，创建一个空的
                generated_probs_ch = torch.zeros(B, self.num_embeddings, target_T_compressed, device=device)
            
            generated_probs.append(generated_probs_ch)
        
        # Stack: [B, codebook_size, target_T/compression_factor, C]
        generated_probs = torch.stack(generated_probs, dim=3)  # [B, codebook_size, target_T/compression_factor, C]
        
        # Step 3: 从生成的概率分布中选择最可能的 codebook indices
        codebook_indices = torch.argmax(generated_probs, dim=1)  # [B, target_T/compression_factor, C]
        
        # Step 4: 通过 VQVAE decoder 解码得到最终预测
        codebook_embeddings = self.pretrained_model.vq._embedding.weight  # [num_embeddings, embedding_dim]
        decoded = []
        
        for ch in range(C):
            indices_ch = codebook_indices[:, :, ch]  # [B, target_T/compression_factor]
            
            # 获取对应的 embeddings
            indices_flat = indices_ch.view(-1)  # [B*target_T/compression_factor]
            quantized_ch = codebook_embeddings[indices_flat]  # [B*target_T/compression_factor, embedding_dim]
            quantized_ch = quantized_ch.view(B, -1, self.embedding_dim)  # [B, target_T/compression_factor, embedding_dim]
            
            # 如果添加了 finetune head，通过它处理
            if self.add_finetune_head:
                quantized_ch = self.finetune_head(quantized_ch)
            
            # 转换为 decoder 需要的格式 [B, embedding_dim, target_T/compression_factor]
            quantized_ch = quantized_ch.permute(0, 2, 1)
            
            # 通过 decoder 解码
            decoded_ch = self.decoder(quantized_ch, self.compression_factor)  # [B, target_len]
            decoded.append(decoded_ch)
        
        # Stack: [B, target_len, C]
        pred = torch.stack(decoded, dim=2)  # [B, target_len, C]
        
        return pred

