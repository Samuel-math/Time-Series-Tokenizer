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
        
        # 加载预训练的VQVAE权重（不包括 decoder，因为不需要）
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
        
        return logits


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


class CrossAttentionLayer(nn.Module):
    """Cross-Attention Layer: Query from input patches, Key and Value from codebook"""
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.cross_attn = MultiheadAttention(
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
        
    def forward(self, query, key_value):
        """
        Args:
            query: [B, T_q, d_model] - patch后的时间序列
            key_value: [B, num_embeddings, d_model] - 码本embeddings
        """
        # Cross-attention: Q from query, K and V from codebook
        x2, _ = self.cross_attn(query, key_value, key_value)
        query = query + self.dropout1(x2)
        query = self.norm1(query)
        
        # Feed-forward
        x2 = self.ff(query)
        query = query + self.dropout2(x2)
        query = self.norm2(query)
        
        return query


class VQVAETransformerFinetune(nn.Module):
    """
    微调模型：使用 Cross-Attention 架构，将训练出的码本作为 K 和 V，与 patch 后的时间序列做 cross-attention
    
    输入: [B, T, C] 时间序列
    输出: [B, target_len, C] 预测结果
    """
    def __init__(self, vqvae_config, transformer_config,
                 patch_size=16, stride=None,
                 pretrained_model=None,
                 freeze_vq=True, freeze_transformer=False,
                 head_type='mlp', head_dropout=0.1, individual=False,
                 load_vqvae_weights=False, vqvae_checkpoint_path=None,
                 device='cpu'):
        """
        Args:
            vqvae_config: VQVAE配置字典（主要用于获取码本）
            transformer_config: Transformer配置字典
            patch_size: patch 大小
            stride: patch 步长，默认等于 patch_size（非重叠）
            pretrained_model: 可选的预训练模型（如果提供，将使用其 VQ 层）
            freeze_vq: 是否冻结 VQ 层（包括 codebook）
            freeze_transformer: 是否冻结 Transformer 层
            head_type: 预测头类型 ('mlp' 或 'linear')
            head_dropout: 预测头的 dropout 率
            individual: 是否为每个通道使用独立的预测头
            load_vqvae_weights: 是否加载预训练的VQVAE权重
            vqvae_checkpoint_path: VQVAE模型checkpoint路径
            device: 加载模型时的设备
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.num_embeddings = vqvae_config['num_embeddings']
        self.embedding_dim = vqvae_config['embedding_dim']
        self.d_model = transformer_config.get('d_model', 128)
        self.n_layers = transformer_config.get('n_layers', 3)
        self.n_heads = transformer_config.get('n_heads', 8)
        self.d_ff = transformer_config.get('d_ff', 256)
        self.dropout = transformer_config.get('dropout', 0.1)
        self.attn_dropout = transformer_config.get('attn_dropout', 0.1)
        self.individual = individual
        
        # VQ层（用于获取码本）
        if pretrained_model is not None:
            self.vq = pretrained_model.vq
        else:
            self.vq = VectorQuantizer(
                self.num_embeddings, self.embedding_dim,
                vqvae_config.get('commitment_cost', 0.25)
            )
            if load_vqvae_weights and vqvae_checkpoint_path is not None:
                self._load_vq_weights(vqvae_checkpoint_path, device)
        
        # Patch投影层（参考PatchTST）
        self.patch_projection = nn.Linear(self.patch_size, self.d_model)
        
        # 码本投影层
        self.codebook_projection = nn.Linear(self.embedding_dim, self.d_model)
        
        # 位置编码
        self.W_pos = None  # 延迟初始化
        self.pe = 'zeros'
        self.learn_pe = True
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Cross-Attention Layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(self.d_model, self.n_heads, self.d_ff,
                              self.dropout, self.attn_dropout)
            for _ in range(self.n_layers)
        ])
        
        # 如果提供了预训练模型，加载其transformer层的K和V参数
        if pretrained_model is not None:
            self._load_transformer_kv_weights(pretrained_model)
        
        # 冻结参数
        if freeze_vq:
            for param in self.vq.parameters():
                param.requires_grad = False
        if freeze_transformer:
            for param in self.cross_attention_layers.parameters():
                param.requires_grad = False
            for param in self.patch_projection.parameters():
                param.requires_grad = False
            for param in self.codebook_projection.parameters():
                param.requires_grad = False
        
        # 预测头（延迟初始化）
        self.head_type = head_type
        self.head_dropout = head_dropout
        self.prediction_head = None
    
    def _load_vq_weights(self, checkpoint_path, device='cpu'):
        """从checkpoint加载预训练的VQ（码本）权重"""
        import os
        if not os.path.exists(checkpoint_path):
            print(f"警告: VQVAE checkpoint路径不存在: {checkpoint_path}")
            return
        
        try:
            vqvae_model = torch.load(checkpoint_path, map_location=device)
            if isinstance(vqvae_model, dict) and 'model_state_dict' in vqvae_model:
                from .vqvae import vqvae as VQVAE
                temp_model = VQVAE({
                    'block_hidden_size': 128, 'num_residual_layers': 2, 'res_hidden_size': 64,
                    'embedding_dim': self.embedding_dim, 'num_embeddings': self.num_embeddings,
                    'commitment_cost': 0.25, 'compression_factor': 4
                })
                temp_model.load_state_dict(vqvae_model['model_state_dict'])
                vqvae_model = temp_model
            
            if hasattr(vqvae_model, 'vq'):
                self.vq.load_state_dict(vqvae_model.vq.state_dict(), strict=True)
                print(f"VQ Codebook权重已成功加载 (从 {checkpoint_path})")
        except Exception as e:
            print(f"加载VQ权重时出错: {e}")
    
    def _load_transformer_kv_weights(self, pretrained_model):
        """
        从预训练模型加载transformer层的K和V参数到cross_attention_layers
        Q参数保持随机初始化
        
        Args:
            pretrained_model: VQVAETransformerPretrain 模型实例
        """
        if not hasattr(pretrained_model, 'transformer_layers'):
            print("警告: 预训练模型中没有 transformer_layers，跳过K和V参数加载")
            return
        
        pretrained_layers = pretrained_model.transformer_layers
        num_layers = min(len(pretrained_layers), len(self.cross_attention_layers))
        
        loaded_count = 0
        for i in range(num_layers):
            try:
                # 获取预训练模型的self-attention层
                pretrained_attn = pretrained_layers[i].self_attn
                # 获取当前模型的cross-attention层
                current_attn = self.cross_attention_layers[i].cross_attn
                
                # 检查维度是否匹配
                if (pretrained_attn.W_K.weight.shape == current_attn.W_K.weight.shape and
                    pretrained_attn.W_V.weight.shape == current_attn.W_V.weight.shape):
                    
                    # 加载K和V的权重和偏置
                    current_attn.W_K.weight.data.copy_(pretrained_attn.W_K.weight.data)
                    if pretrained_attn.W_K.bias is not None and current_attn.W_K.bias is not None:
                        current_attn.W_K.bias.data.copy_(pretrained_attn.W_K.bias.data)
                    
                    current_attn.W_V.weight.data.copy_(pretrained_attn.W_V.weight.data)
                    if pretrained_attn.W_V.bias is not None and current_attn.W_V.bias is not None:
                        current_attn.W_V.bias.data.copy_(pretrained_attn.W_V.bias.data)
                    
                    loaded_count += 1
                else:
                    print(f"警告: 第 {i} 层维度不匹配，跳过K和V参数加载")
                    print(f"  预训练 K: {pretrained_attn.W_K.weight.shape}, 当前 K: {current_attn.W_K.weight.shape}")
                    print(f"  预训练 V: {pretrained_attn.W_V.weight.shape}, 当前 V: {current_attn.W_V.weight.shape}")
            except Exception as e:
                print(f"警告: 加载第 {i} 层的K和V参数时出错: {e}")
        
        if loaded_count > 0:
            print(f"成功加载 {loaded_count}/{num_layers} 层的K和V参数（Q参数保持随机初始化）")
        else:
            print("警告: 未能加载任何K和V参数")
    
    def _create_patches(self, x):
        """
        参考PatchTST的patch方法：使用unfold创建patches
        
        Args:
            x: [B, T] 输入序列
        
        Returns:
            patches: [B, num_patches, patch_size] patch 序列
        """
        B, T = x.shape
        patch_len = self.patch_size
        stride = self.stride
        
        # 计算patch数量（参考PatchTST）
        num_patches = (max(T, patch_len) - patch_len) // stride + 1
        tgt_len = patch_len + stride * (num_patches - 1)
        s_begin = max(0, T - tgt_len)  # 确保非负
        
        # 从末尾截取（确保最后一个patch包含最新数据）
        x = x[:, s_begin:]  # [B, tgt_len]
        
        # 如果序列太短，进行padding
        if x.shape[1] < patch_len:
            x = F.pad(x, (0, patch_len - x.shape[1]), mode='constant', value=0)
        
        # 使用unfold创建patches（参考PatchTST）
        patches = x.unfold(dimension=1, size=patch_len, step=stride)  # [B, num_patches, patch_len]
        
        return patches
    
    def _get_codebook_kv(self, B):
        """获取码本作为 K 和 V"""
        codebook_emb = self.vq._embedding.weight  # [num_embeddings, embedding_dim]
        codebook_kv = self.codebook_projection(codebook_emb)  # [num_embeddings, d_model]
        return codebook_kv.unsqueeze(0).expand(B, -1, -1)  # [B, num_embeddings, d_model]
    
    def _encode_with_cross_attention(self, x):
        """
        使用 patch + Cross-Attention 处理输入序列（参考PatchTST的处理方式）
        
        Args:
            x: [B, T, C] 输入时间序列
        
        Returns:
            output: [B, num_patches, C, d_model] cross-attention 输出
        """
        B, T, C = x.shape
        
        # 创建patches（参考PatchTST）
        patches_list = []
        for ch in range(C):
            patches = self._create_patches(x[:, :, ch])  # [B, num_patches, patch_size]
            patches_list.append(patches)
        
        # Stack: [B, num_patches, C, patch_size]
        patches = torch.stack(patches_list, dim=2)
        
        # 投影到d_model（参考PatchTST的W_P）
        z = self.patch_projection(patches)  # [B, num_patches, C, d_model]
        
        # 获取码本作为K和V
        codebook_kv = self._get_codebook_kv(B)  # [B, num_embeddings, d_model]
        
        # Channel independent处理
        num_patches = z.shape[1]
        if self.W_pos is None or self.W_pos.shape[0] != num_patches:
            self.W_pos = positional_encoding(self.pe, self.learn_pe, num_patches, self.d_model)
        
        outputs = []
        for ch in range(C):
            z_ch = z[:, :, ch, :]  # [B, num_patches, d_model]
            z_ch = z_ch + self.W_pos.to(z_ch.device)
            z_ch = self.dropout_layer(z_ch)
            
            # Cross-Attention
            for layer in self.cross_attention_layers:
                z_ch = layer(z_ch, codebook_kv)
            
            outputs.append(z_ch)
        
        return torch.stack(outputs, dim=2)  # [B, num_patches, C, d_model]
    
    def _build_prediction_head(self, n_vars, target_len):
        """构建预测头（参考PatchTST的PredictionHead）"""
        if self.prediction_head is not None:
            return
        
        if self.head_type == 'linear':
            if self.individual:
                self.prediction_head = nn.ModuleList([
                    nn.Sequential(nn.Dropout(self.head_dropout), nn.Linear(self.d_model, target_len))
                    for _ in range(n_vars)
                ])
            else:
                self.prediction_head = nn.Sequential(
                    nn.Dropout(self.head_dropout), nn.Linear(self.d_model, target_len)
                )
        else:  # 'mlp'
            if self.individual:
                self.prediction_head = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.d_model, self.d_model), nn.GELU(),
                        nn.Dropout(self.head_dropout), nn.Linear(self.d_model, target_len)
                    ) for _ in range(n_vars)
                ])
            else:
                self.prediction_head = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model), nn.GELU(),
                    nn.Dropout(self.head_dropout), nn.Linear(self.d_model, target_len)
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
        
        # Patch + Cross-Attention
        cross_attn_output = self._encode_with_cross_attention(x)  # [B, num_patches, C, d_model]
        last_features = cross_attn_output[:, -1, :, :]  # [B, C, d_model]
        
        # 延迟初始化预测头
        if self.prediction_head is None:
            self._build_prediction_head(C, target_len)
            self.prediction_head = self.prediction_head.to(x.device)
        
        # 生成预测
        if self.individual:
            preds = [self.prediction_head[ch](last_features[:, ch, :]) for ch in range(C)]
        else:
            preds = [self.prediction_head(last_features[:, ch, :]) for ch in range(C)]
        
        return torch.stack(preds, dim=2)  # [B, target_len, C]

