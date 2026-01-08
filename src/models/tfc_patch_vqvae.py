"""
TF-C (Time-Frequency Consistency) Patch-based VQVAE Transformer

架构说明：
1. 双编码器架构：时域编码器 + 频域编码器
2. 共享VQ码本：时频特征在同一码本空间对齐
3. 投影头：将特征映射到对比学习空间
4. 损失函数：L_total = α·L_recon + β·L_vq + γ·L_tfc

参考：TF-C (Time-Frequency Consistency) 对比学习框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .vqvae import Encoder, Decoder
from .patch_vqvae_transformer import (
    FlattenedVectorQuantizer,
    FlattenedVectorQuantizerEMA,
    CausalTransformer
)


class FrequencyEncoder(nn.Module):
    """
    频域编码器：从频域幅值中提取特征
    
    支持两种架构：
    - 'mlp': 3层MLP
    - 'cnn': 1D-CNN
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256, encoder_type='mlp', dropout=0.1):
        super().__init__()
        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if encoder_type == 'mlp':
            # 3层MLP架构
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        elif encoder_type == 'cnn':
            # 1D-CNN架构
            self.encoder = nn.Sequential(
                # 第一层卷积
                nn.Conv1d(1, hidden_dim // 4, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                
                # 第二层卷积
                nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                
                # 第三层卷积
                nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                
                # 全局平均池化
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                
                # 输出投影
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Use 'mlp' or 'cnn'.")
    
    def forward(self, x_f):
        """
        Args:
            x_f: [B, freq_len] 频域幅值（已归一化）
        Returns:
            z_f: [B, output_dim] 频域特征
        """
        if self.encoder_type == 'cnn':
            # CNN需要 [B, 1, freq_len] 格式
            x_f = x_f.unsqueeze(1)
        
        z_f = self.encoder(x_f)
        return z_f


class ProjectionHead(nn.Module):
    """
    投影头：将特征映射到对比学习空间
    
    使用非线性投影: Linear -> ReLU -> Linear
    目的：解耦重构空间和对比空间，防止对比学习干扰码本的聚类稳定性
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            p: [B, output_dim] L2归一化后的投影
        """
        p = self.projector(x)
        # L2归一化用于对比学习
        p = F.normalize(p, p=2, dim=-1)
        return p


class InfoNCELoss(nn.Module):
    """
    InfoNCE对比损失
    
    正样本对：来自同一个样本的 (p_t, p_f)
    负样本对：Batch内的其他样本
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, p_t, p_f):
        """
        计算InfoNCE损失（使用L2距离）
        
        Args:
            p_t: [B, D] 时域投影（已L2归一化）
            p_f: [B, D] 频域投影（已L2归一化）
        
        Returns:
            loss: scalar InfoNCE损失
        """
        batch_size = p_t.shape[0]
        device = p_t.device
        
        # 计算L2距离矩阵 [B, B]
        # dist[i,j] = ||p_t[i] - p_f[j]||_2^2
        # 使用公式: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        p_t_sq = torch.sum(p_t ** 2, dim=1, keepdim=True)  # [B, 1]
        p_f_sq = torch.sum(p_f ** 2, dim=1, keepdim=True)  # [B, 1]
        
        # t->f 距离矩阵
        dist_t2f = p_t_sq + p_f_sq.t() - 2 * torch.matmul(p_t, p_f.t())  # [B, B]
        # f->t 距离矩阵
        dist_f2t = p_f_sq + p_t_sq.t() - 2 * torch.matmul(p_f, p_t.t())  # [B, B]
        
        # 转换为相似度：使用负距离（距离越小，相似度越高）
        sim_t2f = -dist_t2f / self.temperature  # [B, B]
        sim_f2t = -dist_f2t / self.temperature  # [B, B]
        
        # 正样本在对角线上，标签为 [0, 1, 2, ..., B-1]
        labels = torch.arange(batch_size, device=device)
        
        # 交叉熵损失：对角线应该最大（即距离最小）
        # t->f方向：给定时域，找对应频域
        loss_t2f = F.cross_entropy(sim_t2f, labels)
        # f->t方向：给定频域，找对应时域
        loss_f2t = F.cross_entropy(sim_f2t, labels)
        
        # 对称损失
        loss = (loss_t2f + loss_f2t) / 2
        
        return loss


class TFCPatchVQVAE(nn.Module):
    """
    TF-C (Time-Frequency Consistency) Patch-based VQVAE
    
    核心创新：
    1. 双编码器：时域+频域
    2. 共享VQ码本：时频语义对齐
    3. 对比学习：增强码本的语义表达能力
    """
    def __init__(self, config):
        super().__init__()
        
        # ============ 基础配置 ============
        self.patch_size = config.get('patch_size', 16)
        self.embedding_dim = config.get('embedding_dim', 32)
        self.compression_factor = config.get('compression_factor', 4)
        self.codebook_size = config.get('codebook_size', 256)
        self.commitment_cost = config.get('commitment_cost', 0.25)
        self.use_codebook_ema = config.get('codebook_ema', False)
        self.ema_decay = config.get('ema_decay', 0.99)
        self.ema_eps = config.get('ema_eps', 1e-5)
        
        # VQVAE Encoder/Decoder 配置
        self.num_hiddens = config.get('num_hiddens', 64)
        self.num_residual_layers = config.get('num_residual_layers', 2)
        self.num_residual_hiddens = config.get('num_residual_hiddens', 32)
        
        # code_dim = embedding_dim * compressed_len
        self.compressed_len = self.patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len
        
        # ============ TF-C 特定配置 ============
        # 频域编码器配置
        self.freq_encoder_type = config.get('freq_encoder_type', 'mlp')  # 'mlp' or 'cnn'
        self.freq_encoder_hidden = config.get('freq_encoder_hidden', 256)
        
        # 投影头配置
        self.proj_hidden_dim = config.get('proj_hidden_dim', 256)
        self.proj_output_dim = config.get('proj_output_dim', 128)  # 对比空间维度
        
        # InfoNCE损失配置
        self.temperature = config.get('temperature', 0.07)
        
        # 损失权重配置
        self.alpha = config.get('alpha', 1.0)  # 重构损失权重
        self.beta = config.get('beta', 1.0)    # VQ损失权重
        self.gamma = config.get('gamma', 0.5)  # TF-C对比损失权重
        
        # ============ 时域编码器（复用现有Encoder） ============
        self.time_encoder = Encoder(
            in_channels=1,  # 单通道输入
            num_hiddens=self.num_hiddens,
            num_residual_layers=self.num_residual_layers,
            num_residual_hiddens=self.num_residual_hiddens,
            embedding_dim=self.embedding_dim,
            compression_factor=self.compression_factor
        )
        
        # ============ 频域编码器（新增） ============
        # 频域输入维度：rfft输出长度 = patch_size // 2 + 1
        self.freq_input_dim = self.patch_size // 2 + 1
        self.freq_encoder = FrequencyEncoder(
            input_dim=self.freq_input_dim,
            output_dim=self.code_dim,  # 输出维度与时域编码器一致
            hidden_dim=self.freq_encoder_hidden,
            encoder_type=self.freq_encoder_type,
            dropout=config.get('dropout', 0.1)
        )
        
        # ============ 共享VQ码本 ============
        vq_init_method = config.get('vq_init_method', 'random')
        if self.use_codebook_ema:
            self.vq = FlattenedVectorQuantizerEMA(
                self.codebook_size, self.code_dim, self.commitment_cost,
                decay=self.ema_decay, eps=self.ema_eps,
                init_method=vq_init_method
            )
        else:
            self.vq = FlattenedVectorQuantizer(
                self.codebook_size, self.code_dim, self.commitment_cost,
                init_method=vq_init_method
            )
        
        # ============ 时域解码器 ============
        self.decoder = Decoder(
            in_channels=self.embedding_dim,
            num_hiddens=self.num_hiddens,
            num_residual_layers=self.num_residual_layers,
            num_residual_hiddens=self.num_residual_hiddens,
            compression_factor=self.compression_factor,
            out_channels=1  # 单通道输出
        )
        
        # ============ 投影头（用于对比学习） ============
        self.proj_time = ProjectionHead(
            input_dim=self.code_dim,
            hidden_dim=self.proj_hidden_dim,
            output_dim=self.proj_output_dim
        )
        self.proj_freq = ProjectionHead(
            input_dim=self.code_dim,
            hidden_dim=self.proj_hidden_dim,
            output_dim=self.proj_output_dim
        )
        
        # ============ InfoNCE损失模块 ============
        self.infonce_loss = InfoNCELoss(temperature=self.temperature)
        
        # ============ Transformer（可选，用于预测任务） ============
        self.n_layers = config.get('n_layers', 4)
        self.n_heads = config.get('n_heads', 4)
        self.d_ff = config.get('d_ff', 256)
        self.dropout = config.get('dropout', 0.1)
        self.transformer_hidden_dim = config.get('transformer_hidden_dim', None)
        
        self.transformer = CausalTransformer(
            self.code_dim, self.n_heads, self.n_layers,
            self.d_ff, self.dropout, hidden_dim=self.transformer_hidden_dim
        )
        
        # 输出头
        self.output_head = nn.Linear(self.code_dim, self.codebook_size)
    
    def compute_fft_magnitude(self, x_t):
        """
        计算时间序列的FFT幅值
        
        Args:
            x_t: [B, L] 时域信号
        
        Returns:
            x_f: [B, L//2+1] 归一化的频域幅值
        """
        # 实数FFT
        fft_result = torch.fft.rfft(x_t, dim=-1)  # [B, L//2+1] (complex)
        
        # 取幅值
        magnitude = torch.abs(fft_result)  # [B, L//2+1]
        
        # 归一化：使用log-magnitude + 标准化
        # 加1e-8防止log(0)
        log_magnitude = torch.log(magnitude + 1e-8)
        
        # 标准化到零均值单位方差
        mean = log_magnitude.mean(dim=-1, keepdim=True)
        std = log_magnitude.std(dim=-1, keepdim=True) + 1e-8
        x_f = (log_magnitude - mean) / std
        
        return x_f
    
    def encode_time_domain(self, x_patches):
        """
        时域编码
        
        Args:
            x_patches: [B, num_patches, patch_size, C] 时域patches
        
        Returns:
            z_t: [B, num_patches, C, code_dim] 时域特征
        """
        B, num_patches, patch_size, C = x_patches.shape
        
        z_t_list = []
        for c in range(C):
            # 提取通道 [B, num_patches, patch_size]
            x_c = x_patches[:, :, :, c]
            x_c_flat = x_c.reshape(B * num_patches, patch_size)
            x_c_flat = x_c_flat.unsqueeze(1)  # [B*num_patches, 1, patch_size]
            
            # 时域编码器
            z = self.time_encoder(x_c_flat, self.compression_factor)
            z_flat = z.reshape(B * num_patches, -1)  # [B*num_patches, code_dim]
            z_c = z_flat.reshape(B, num_patches, self.code_dim)
            
            z_t_list.append(z_c)
        
        z_t = torch.stack(z_t_list, dim=2)  # [B, num_patches, C, code_dim]
        return z_t
    
    def encode_freq_domain(self, x_patches):
        """
        频域编码
        
        Args:
            x_patches: [B, num_patches, patch_size, C] 时域patches
        
        Returns:
            z_f: [B, num_patches, C, code_dim] 频域特征
        """
        B, num_patches, patch_size, C = x_patches.shape
        
        z_f_list = []
        for c in range(C):
            # 提取通道 [B, num_patches, patch_size]
            x_c = x_patches[:, :, :, c]
            x_c_flat = x_c.reshape(B * num_patches, patch_size)
            
            # FFT转换
            x_f = self.compute_fft_magnitude(x_c_flat)  # [B*num_patches, freq_len]
            
            # 频域编码器
            z_f_flat = self.freq_encoder(x_f)  # [B*num_patches, code_dim]
            z_f_c = z_f_flat.reshape(B, num_patches, self.code_dim)
            
            z_f_list.append(z_f_c)
        
        z_f = torch.stack(z_f_list, dim=2)  # [B, num_patches, C, code_dim]
        return z_f
    
    def quantize(self, z):
        """
        向量量化
        
        Args:
            z: [B, num_patches, C, code_dim] 编码器输出
        
        Returns:
            z_q: [B, num_patches, C, code_dim] 量化后的向量
            vq_loss: scalar VQ损失
            indices: [B, num_patches, C] 码本索引
        """
        B, num_patches, C, code_dim = z.shape
        
        indices_list = []
        z_q_list = []
        vq_loss_sum = 0
        
        for c in range(C):
            z_c = z[:, :, c, :]  # [B, num_patches, code_dim]
            z_c_flat = z_c.reshape(B * num_patches, code_dim)
            
            # VQ
            vq_loss_c, z_q_flat_c, indices_c = self.vq(z_c_flat)
            vq_loss_sum += vq_loss_c
            
            indices_c = indices_c.reshape(B, num_patches)
            z_q_c = z_q_flat_c.reshape(B, num_patches, code_dim)
            
            indices_list.append(indices_c)
            z_q_list.append(z_q_c)
        
        indices = torch.stack(indices_list, dim=2)  # [B, num_patches, C]
        z_q = torch.stack(z_q_list, dim=2)  # [B, num_patches, C, code_dim]
        vq_loss = vq_loss_sum / C
        
        return z_q, vq_loss, indices
    
    def decode(self, z_q):
        """
        从量化向量解码
        
        Args:
            z_q: [B, num_patches, C, code_dim]
        
        Returns:
            x_recon: [B, num_patches * patch_size, C]
        """
        B, num_patches, C, code_dim = z_q.shape
        
        x_recon_list = []
        for c in range(C):
            z_q_c = z_q[:, :, c, :]  # [B, num_patches, code_dim]
            z_q_c_flat = z_q_c.reshape(B * num_patches, self.embedding_dim, self.compressed_len)
            
            x_recon_c = self.decoder(z_q_c_flat, self.compression_factor)
            x_recon_c = x_recon_c.reshape(B, num_patches, self.patch_size)
            
            x_recon_list.append(x_recon_c)
        
        x_recon = torch.stack(x_recon_list, dim=3)  # [B, num_patches, patch_size, C]
        x_recon = x_recon.reshape(B, -1, C)  # [B, num_patches * patch_size, C]
        
        return x_recon
    
    def compute_tfc_loss(self, z_t, z_f):
        """
        计算时频对比损失
        
        Args:
            z_t: [B, num_patches, C, code_dim] 时域特征
            z_f: [B, num_patches, C, code_dim] 频域特征
        
        Returns:
            tfc_loss: scalar 对比损失
        """
        B, num_patches, C, code_dim = z_t.shape
        
        # 将所有patches和channels展平
        # [B, num_patches, C, code_dim] -> [B*num_patches*C, code_dim]
        z_t_flat = z_t.reshape(-1, code_dim)
        z_f_flat = z_f.reshape(-1, code_dim)
        
        # 投影到对比空间
        p_t = self.proj_time(z_t_flat)  # [B*num_patches*C, proj_output_dim]
        p_f = self.proj_freq(z_f_flat)  # [B*num_patches*C, proj_output_dim]
        
        # 计算InfoNCE损失
        tfc_loss = self.infonce_loss(p_t, p_f)
        
        return tfc_loss
    
    def forward_tfc_pretrain(self, x):
        """
        TF-C预训练前向传播
        
        Args:
            x: [B, T, C] 输入时间序列
        
        Returns:
            x_recon: [B, T', C] 重构序列
            recon_loss: scalar 重构损失
            vq_loss: scalar VQ损失
            tfc_loss: scalar 时频对比损失
            total_loss: scalar 总损失
            info_dict: dict 包含额外信息
        """
        B, T, C = x.shape
        num_patches = T // self.patch_size
        
        # 截取整数个patch
        x = x[:, :num_patches * self.patch_size, :]
        x_patches = x.reshape(B, num_patches, self.patch_size, C)
        
        # ============ 双编码器 ============
        z_t = self.encode_time_domain(x_patches)  # [B, num_patches, C, code_dim]
        z_f = self.encode_freq_domain(x_patches)  # [B, num_patches, C, code_dim]
        
        # ============ 共享VQ（时域和频域都量化） ============
        z_q_t, vq_loss_t, indices_t = self.quantize(z_t)
        z_q_f, vq_loss_f, indices_f = self.quantize(z_f)
        
        # VQ损失取平均
        vq_loss = (vq_loss_t + vq_loss_f) / 2
        
        # ============ 解码（仅使用时域量化向量） ============
        x_recon = self.decode(z_q_t)  # [B, num_patches * patch_size, C]
        
        # ============ 计算损失 ============
        # 重构损失
        x_target = x[:, :x_recon.shape[1], :]
        recon_loss = F.mse_loss(x_recon, x_target)
        
        # 时频对比损失（使用量化前的特征）
        tfc_loss = self.compute_tfc_loss(z_t, z_f)
        
        # 总损失
        total_loss = self.alpha * recon_loss + self.beta * vq_loss + self.gamma * tfc_loss
        
        # 额外信息
        info_dict = {
            'z_t': z_t,
            'z_f': z_f,
            'z_q_t': z_q_t,
            'z_q_f': z_q_f,
            'indices_t': indices_t,
            'indices_f': indices_f,
            'recon_loss': recon_loss.item(),
            'vq_loss': vq_loss.item(),
            'tfc_loss': tfc_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return x_recon, recon_loss, vq_loss, tfc_loss, total_loss, info_dict
    
    def forward(self, x, mode='tfc_pretrain', **kwargs):
        """
        统一的前向传播接口
        
        Args:
            x: [B, T, C] 输入序列
            mode: 'tfc_pretrain' | 'pretrain' | 'finetune'
        """
        if mode == 'tfc_pretrain':
            return self.forward_tfc_pretrain(x)
        elif mode == 'pretrain':
            return self.forward_pretrain(x, **kwargs)
        elif mode == 'finetune':
            return self.forward_finetune(x, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def encode_to_indices(self, x, use_freq=False):
        """
        编码为码本索引（用于预测任务）
        
        Args:
            x: [B, T, C] 输入序列
            use_freq: 是否使用频域编码器
        
        Returns:
            indices: [B, num_patches, C]
            vq_loss: scalar
            z_q: [B, num_patches, C, code_dim]
        """
        B, T, C = x.shape
        num_patches = T // self.patch_size
        
        x = x[:, :num_patches * self.patch_size, :]
        x_patches = x.reshape(B, num_patches, self.patch_size, C)
        
        if use_freq:
            z = self.encode_freq_domain(x_patches)
        else:
            z = self.encode_time_domain(x_patches)
        
        z_q, vq_loss, indices = self.quantize(z)
        
        return indices, vq_loss, z_q
    
    def forward_pretrain(self, x, target):
        """
        标准预训练（Next Token Prediction）
        与原始PatchVQVAETransformer兼容
        """
        # 编码为码本索引
        indices, vq_loss, z_q = self.encode_to_indices(x)
        
        B, num_patches, C, code_dim = z_q.shape
        
        # 展平通道维度用于Transformer处理
        z_q_flat = z_q.permute(0, 2, 1, 3).reshape(B * C, num_patches, code_dim)
        
        # Transformer
        h = self.transformer(z_q_flat)  # [B*C, num_patches, code_dim]
        
        # 输出头
        logits = self.output_head(h[:, :-1, :])  # [B*C, num_patches-1, codebook_size]
        
        # 重组
        logits = logits.reshape(B, C, num_patches - 1, -1).permute(0, 2, 1, 3)
        
        # 目标索引
        target_indices = indices[:, 1:, :]  # [B, num_patches-1, C]
        
        return logits, target_indices, vq_loss
    
    def forward_finetune(self, x, target_len):
        """
        微调：预测未来序列
        """
        B, T, C = x.shape
        num_pred_patches = (target_len + self.patch_size - 1) // self.patch_size
        
        # 编码
        indices, vq_loss, z_q = self.encode_to_indices(x)
        num_input_patches = z_q.shape[1]
        
        # 展平
        B, num_patches, C, code_dim = z_q.shape
        z_q_flat = z_q.permute(0, 2, 1, 3).reshape(B * C, num_patches, code_dim)
        
        # 占位符
        placeholder = torch.zeros(B * C, num_pred_patches, self.code_dim, 
                                 device=z_q_flat.device, dtype=z_q_flat.dtype)
        
        # 拼接
        full_sequence = torch.cat([z_q_flat, placeholder], dim=1)
        
        # Transformer
        h_full = self.transformer(full_sequence)
        h_pred = h_full[:, num_input_patches:, :]
        
        # 预测
        logits = self.output_head(h_pred)
        weights = F.softmax(logits, dim=-1)
        codebook = self.vq.embedding.weight
        pred_codes = torch.matmul(weights, codebook)
        
        # 重组
        pred_codes = pred_codes.reshape(B, C, num_pred_patches, code_dim).permute(0, 2, 1, 3)
        
        # 解码
        pred = self.decode(pred_codes)
        pred = pred[:, :target_len, :]
        
        return pred, vq_loss
    
    @torch.no_grad()
    def get_codebook_usage(self, x, use_freq=False):
        """获取码本使用率"""
        indices, _, _ = self.encode_to_indices(x, use_freq=use_freq)
        unique = torch.unique(indices.reshape(-1))
        return len(unique) / self.codebook_size, unique
    
    def freeze_vqvae(self, components=None):
        """冻结VQVAE组件"""
        if components is None:
            components = ['TimeEncoder', 'FreqEncoder', 'Decoder', 'VQ']
        
        frozen = []
        if 'TimeEncoder' in components:
            for param in self.time_encoder.parameters():
                param.requires_grad = False
            frozen.append('TimeEncoder')
        
        if 'FreqEncoder' in components:
            for param in self.freq_encoder.parameters():
                param.requires_grad = False
            frozen.append('FreqEncoder')
        
        if 'Decoder' in components:
            for param in self.decoder.parameters():
                param.requires_grad = False
            frozen.append('Decoder')
        
        if 'VQ' in components:
            for param in self.vq.parameters():
                param.requires_grad = False
            if isinstance(self.vq, FlattenedVectorQuantizerEMA):
                self.vq._disable_ema_update = True
            frozen.append('VQ')
        
        if frozen:
            print(f"✓ 已冻结: {', '.join(frozen)}")
        
        return frozen
    
    def unfreeze_vqvae(self, components=None):
        """解冻VQVAE组件"""
        if components is None:
            components = ['TimeEncoder', 'FreqEncoder', 'Decoder', 'VQ']
        
        unfrozen = []
        if 'TimeEncoder' in components:
            for param in self.time_encoder.parameters():
                param.requires_grad = True
            unfrozen.append('TimeEncoder')
        
        if 'FreqEncoder' in components:
            for param in self.freq_encoder.parameters():
                param.requires_grad = True
            unfrozen.append('FreqEncoder')
        
        if 'Decoder' in components:
            for param in self.decoder.parameters():
                param.requires_grad = True
            unfrozen.append('Decoder')
        
        if 'VQ' in components:
            for param in self.vq.parameters():
                param.requires_grad = True
            if isinstance(self.vq, FlattenedVectorQuantizerEMA):
                self.vq._disable_ema_update = False
            unfrozen.append('VQ')
        
        if unfrozen:
            print(f"✓ 已解冻: {', '.join(unfrozen)}")
        
        return unfrozen


def get_tfc_model_config(args):
    """
    构建TF-C模型配置
    
    在args基础上添加TF-C特定配置
    """
    config = {
        # 基础VQVAE配置
        'patch_size': getattr(args, 'patch_size', 16),
        'embedding_dim': getattr(args, 'embedding_dim', 32),
        'compression_factor': getattr(args, 'compression_factor', 4),
        'codebook_size': getattr(args, 'codebook_size', 256),
        'commitment_cost': getattr(args, 'commitment_cost', 0.25),
        'codebook_ema': getattr(args, 'codebook_ema', False),
        'ema_decay': getattr(args, 'ema_decay', 0.99),
        'ema_eps': getattr(args, 'ema_eps', 1e-5),
        
        # Encoder/Decoder配置
        'num_hiddens': getattr(args, 'num_hiddens', 64),
        'num_residual_layers': getattr(args, 'num_residual_layers', 2),
        'num_residual_hiddens': getattr(args, 'num_residual_hiddens', 32),
        
        # Transformer配置
        'n_layers': getattr(args, 'n_layers', 4),
        'n_heads': getattr(args, 'n_heads', 4),
        'd_ff': getattr(args, 'd_ff', 256),
        'dropout': getattr(args, 'dropout', 0.1),
        'transformer_hidden_dim': getattr(args, 'transformer_hidden_dim', None),
        
        # TF-C特定配置
        'freq_encoder_type': getattr(args, 'freq_encoder_type', 'mlp'),
        'freq_encoder_hidden': getattr(args, 'freq_encoder_hidden', 256),
        'proj_hidden_dim': getattr(args, 'proj_hidden_dim', 256),
        'proj_output_dim': getattr(args, 'proj_output_dim', 128),
        'temperature': getattr(args, 'temperature', 0.07),
        
        # 损失权重
        'alpha': getattr(args, 'alpha', 1.0),
        'beta': getattr(args, 'beta', 1.0),
        'gamma': getattr(args, 'gamma', 0.5),
        
        # VQ初始化方法
        'vq_init_method': getattr(args, 'vq_init_method', 'random'),
    }
    
    return config

