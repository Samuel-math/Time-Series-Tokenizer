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
        
        # VQVAE Decoder（用于重构loss计算）
        from .vqvae import Decoder
        self.decoder = Decoder(
            self.embedding_dim,
            vqvae_config['block_hidden_size'],
            vqvae_config['num_residual_layers'],
            vqvae_config['res_hidden_size'],
            self.compression_factor
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
            
            # 检查是否是vqvae模型对象（有encoder、vq和decoder属性）
            # vqvae_pretrain.py使用torch.save(model, path)保存完整模型对象
            if not hasattr(vqvae_model, 'encoder') or not hasattr(vqvae_model, 'vq'):
                print("警告: checkpoint中未找到encoder或vq属性")
                print(f"checkpoint类型: {type(vqvae_model)}")
                print("将使用随机初始化的VQVAE权重")
                return
            
            # 检查是否有decoder（可选，如果没有则使用随机初始化）
            if not hasattr(vqvae_model, 'decoder'):
                print("警告: checkpoint中未找到decoder属性，将使用随机初始化的Decoder权重")
            
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
            
            # 加载Decoder权重
            try:
                decoder_state_dict = vqvae_model.decoder.state_dict()
                self.decoder.load_state_dict(decoder_state_dict, strict=True)
                print(f"VQVAE Decoder权重已成功加载 (从 {checkpoint_path})")
            except Exception as e:
                print(f"VQVAE Decoder权重加载失败: {e}")
                print("将使用随机初始化的Decoder权重")
                
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


class VQVAETransformerFinetune(nn.Module):
    """
    微调模型：使用预训练的 VQVAE Encoder + Transformer 获取码本索引，直接预测目标序列
    
    设计思路：
    1. 使用预训练的 VQVAE encoder + Transformer 获取码本索引 [B, T/compression_factor, C]
    2. 直接使用预训练模型的 codebook (vq._embedding) 将索引转换为向量表示
    3. 通过简单的预测网络（MLP）直接从索引向量预测目标序列 [B, target_len, C]
    4. 不需要 decoder，不需要新的 embedding 层，架构更简单高效
    
    输入: [B, T, C] 时间序列
    输出: [B, target_len, C] 预测结果
    """
    def __init__(self, pretrained_model, vqvae_config, 
                 freeze_encoder=True, freeze_vq=True, freeze_transformer=False,
                 head_type='mlp', head_dropout=0.1, individual=False,
                 aggregation='mean'):
        """
        Args:
            pretrained_model: 预训练的 VQVAETransformerPretrain 模型
            vqvae_config: VQVAE配置字典
            freeze_encoder: 是否冻结 VQVAE encoder
            freeze_vq: 是否冻结 VQ 层（包括 codebook）
            freeze_transformer: 是否冻结 Transformer 层
            head_type: 预测头类型 ('mlp' 或 'linear')
            head_dropout: 预测头的 dropout 率
            individual: 是否为每个通道使用独立的预测头
            aggregation: 聚合方式 ('mean', 'max', 'last', 'attention')
        """
        super().__init__()
        
        self.pretrained_model = pretrained_model
        self.compression_factor = vqvae_config['compression_factor']
        self.num_embeddings = vqvae_config['num_embeddings']
        self.embedding_dim = vqvae_config['embedding_dim']  # 使用预训练模型的 embedding_dim
        self.individual = individual
        self.aggregation = aggregation
        self.n_vars = None  # 将在第一次 forward 时确定
        
        # 冻结参数
        if freeze_encoder:
            for param in self.pretrained_model.vqvae_encoder.parameters():
                param.requires_grad = False
        
        if freeze_vq:
            for param in self.pretrained_model.vq.parameters():
                param.requires_grad = False
        
        if freeze_transformer:
            for param in self.pretrained_model.transformer_layers.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.projection.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.codebook_head.parameters():
                param.requires_grad = False
        
        # 如果使用注意力聚合，需要创建注意力层
        if aggregation == 'attention':
            self.attention_aggregation = nn.MultiheadAttention(
                self.embedding_dim, num_heads=8, dropout=head_dropout, batch_first=True
            )
            # 注册可学习的 query 参数
            self._attention_query = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        else:
            self.attention_aggregation = None
            self._attention_query = None
        
        # 预测头（延迟初始化，因为需要知道通道数和目标长度）
        self.head_type = head_type
        self.head_dropout = head_dropout
        self.prediction_head = None
    
    def _build_prediction_head(self, n_vars, target_len):
        """
        构建预测头（使用聚合后的 embedding，输入维度为 embedding_dim）
        
        Args:
            n_vars: 通道数
            target_len: 预测长度
        """
        if self.prediction_head is not None:
            return
        
        # 使用聚合后的 embedding_dim 作为输入（而不是 T_compressed * embedding_dim）
        input_dim = self.embedding_dim
        
        if self.head_type == 'linear':
            # 简单线性头：从聚合后的 embedding 直接映射到 target_len
            if self.individual:
                self.prediction_head = nn.ModuleList([
                    nn.Sequential(
                        nn.Dropout(self.head_dropout),
                        nn.Linear(input_dim, target_len)
                    ) for _ in range(n_vars)
                ])
            else:
                self.prediction_head = nn.Sequential(
                    nn.Dropout(self.head_dropout),
                    nn.Linear(input_dim, target_len)
                )
        else:  # 'mlp'
            # MLP 头：更强大的表达能力
            if self.individual:
                self.prediction_head = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_dim, self.embedding_dim),
                        nn.GELU(),
                        nn.Dropout(self.head_dropout),
                        nn.Linear(self.embedding_dim, target_len)
                    ) for _ in range(n_vars)
                ])
            else:
                self.prediction_head = nn.Sequential(
                    nn.Linear(input_dim, self.embedding_dim),
                    nn.GELU(),
                    nn.Dropout(self.head_dropout),
                    nn.Linear(self.embedding_dim, target_len)
                )
        
        # 初始化预测头权重（使用 Xavier 初始化）
        self._init_prediction_head()
    
    def _init_prediction_head(self):
        """初始化预测头权重"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # 使用 Xavier 初始化
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        if self.prediction_head is not None:
            if isinstance(self.prediction_head, nn.ModuleList):
                for head in self.prediction_head:
                    head.apply(init_weights)
            else:
                self.prediction_head.apply(init_weights)
    
    def _aggregate_embeddings(self, embeddings):
        """
        对时间维度的 embeddings 进行聚合
        
        Args:
            embeddings: [B, T/compression_factor, C, embedding_dim] 或 [B, T/compression_factor, embedding_dim]
        
        Returns:
            aggregated: [B, C, embedding_dim] 或 [B, embedding_dim]
        """
        if embeddings.dim() == 4:
            # [B, T/compression_factor, C, embedding_dim]
            B, T_compressed, C, embedding_dim = embeddings.shape
            
            if self.aggregation == 'mean':
                # 平均池化
                aggregated = embeddings.mean(dim=1)  # [B, C, embedding_dim]
            elif self.aggregation == 'max':
                # 最大池化
                aggregated = embeddings.max(dim=1)[0]  # [B, C, embedding_dim]
            elif self.aggregation == 'last':
                # 取最后一个时间步
                aggregated = embeddings[:, -1, :, :]  # [B, C, embedding_dim]
            elif self.aggregation == 'attention':
                # 注意力聚合
                aggregated_list = []
                for ch in range(C):
                    embed_ch = embeddings[:, :, ch, :]  # [B, T_compressed, embedding_dim]
                    # 使用可学习的 query
                    query = self._attention_query.expand(B, 1, embedding_dim)
                    attn_out, _ = self.attention_aggregation(query, embed_ch, embed_ch)
                    aggregated_list.append(attn_out.squeeze(1))  # [B, embedding_dim]
                aggregated = torch.stack(aggregated_list, dim=1)  # [B, C, embedding_dim]
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        else:
            # [B, T/compression_factor, embedding_dim]
            B, T_compressed, embedding_dim = embeddings.shape
            
            if self.aggregation == 'mean':
                aggregated = embeddings.mean(dim=1)  # [B, embedding_dim]
            elif self.aggregation == 'max':
                aggregated = embeddings.max(dim=1)[0]  # [B, embedding_dim]
            elif self.aggregation == 'last':
                aggregated = embeddings[:, -1, :]  # [B, embedding_dim]
            elif self.aggregation == 'attention':
                # 注意力聚合
                query = self._attention_query.expand(B, 1, embedding_dim)
                attn_out, _ = self.attention_aggregation(query, embeddings, embeddings)
                aggregated = attn_out.squeeze(1)  # [B, embedding_dim]
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        return aggregated
    
    def _get_codebook_embeddings_soft(self, x):
        """
        使用预训练模型获取码本 embeddings（可微分版本）
        
        Args:
            x: [B, T, C] 输入时间序列
        
        Returns:
            embeddings: [B, T/compression_factor, C, embedding_dim] 软分配的 codebook embeddings
        """
        # 使用预训练模型获取 logits
        logits = self.pretrained_model(x, mask=None)  # [B, codebook_size, T/compression_factor, C]
        
        # 使用 softmax 获取概率分布（可微分）
        # logits: [B, codebook_size, T/compression_factor, C]
        probs = F.softmax(logits, dim=1)  # [B, codebook_size, T/compression_factor, C]
        
        # 获取预训练模型的 codebook embeddings
        codebook_embedding = self.pretrained_model.vq._embedding  # [num_embeddings, embedding_dim]
        
        # 使用概率分布加权求和 codebook embeddings（可微分）
        # probs: [B, codebook_size, T/compression_factor, C]
        # codebook_embedding: [num_embeddings, embedding_dim]
        B, codebook_size, T_compressed, C = probs.shape
        embedding_dim = codebook_embedding.weight.shape[1]
        
        # 对每个通道分别处理
        embeddings = []
        for ch in range(C):
            probs_ch = probs[:, :, :, ch]  # [B, codebook_size, T/compression_factor]
            probs_ch = probs_ch.permute(0, 2, 1)  # [B, T/compression_factor, codebook_size]
            # 加权求和: [B, T/compression_factor, codebook_size] @ [codebook_size, embedding_dim]
            embed_ch = torch.matmul(probs_ch, codebook_embedding.weight)  # [B, T/compression_factor, embedding_dim]
            embeddings.append(embed_ch)
        
        # Stack: [B, T/compression_factor, C, embedding_dim]
        embeddings = torch.stack(embeddings, dim=2)
        
        return embeddings
    
    def forward(self, x, target_len):
        """
        Args:
            x: [B, T, C] 输入时间序列
            target_len: 预测长度
        
        Returns:
            pred: [B, target_len, C] 预测结果
        """
        B, T, C = x.shape
        
        # 使用可微分的 soft assignment 获取 codebook embeddings
        # 这样可以保持梯度流，允许微调 transformer
        embeddings = self._get_codebook_embeddings_soft(x)  # [B, T/compression_factor, C, embedding_dim]
        
        # 对时间维度进行聚合，减少参数
        aggregated = self._aggregate_embeddings(embeddings)  # [B, C, embedding_dim]
        
        # 延迟初始化预测头（第一次 forward 时）
        if self.prediction_head is None:
            self._build_prediction_head(C, target_len)
            self.prediction_head = self.prediction_head.to(x.device)
            if self.attention_aggregation is not None:
                self.attention_aggregation = self.attention_aggregation.to(x.device)
        
        # 通过预测头生成预测
        if self.individual:
            # 每个通道独立预测
            preds = []
            for ch in range(C):
                embed_ch = aggregated[:, ch, :]  # [B, embedding_dim]
                pred_ch = self.prediction_head[ch](embed_ch)  # [B, target_len]
                preds.append(pred_ch)
            pred = torch.stack(preds, dim=2)  # [B, target_len, C]
        else:
            # 共享预测头
            # 对每个通道分别处理
            preds = []
            for ch in range(C):
                embed_ch = aggregated[:, ch, :]  # [B, embedding_dim]
                pred_ch = self.prediction_head(embed_ch)  # [B, target_len]
                preds.append(pred_ch)
            pred = torch.stack(preds, dim=2)  # [B, target_len, C]
        
        return pred

