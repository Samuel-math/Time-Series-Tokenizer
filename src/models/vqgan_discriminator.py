"""
VQGAN Discriminator for Time Series
用于区分真实和重构的时间序列
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesDiscriminator(nn.Module):
    """
    时间序列判别器
    输入: [B, T, C] 时间序列
    输出: [B] 真实概率
    """
    def __init__(self, n_channels, patch_size=16, num_hiddens=64, num_layers=3):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        
        # 特征提取层
        layers = []
        in_dim = n_channels
        
        for i in range(num_layers):
            out_dim = num_hiddens * (2 ** i)
            layers.extend([
                nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_dim = out_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, C] 时间序列
        Returns:
            prob: [B] 真实概率
        """
        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        
        # 特征提取
        features = self.feature_extractor(x)  # [B, hidden_dim, T']
        
        # 分类
        prob = self.classifier(features).squeeze(-1)  # [B]
        
        return prob


class PerceptualLoss(nn.Module):
    """
    感知损失：使用简单的特征提取器计算特征空间的距离
    对于时间序列，使用多层卷积特征
    """
    def __init__(self, n_channels, num_layers=2):
        super().__init__()
        layers = []
        in_dim = n_channels
        
        for i in range(num_layers):
            out_dim = 64 * (i + 1)
            layers.extend([
                nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = out_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 冻结特征提取器（只用于提取特征，不更新）
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, x_real, x_recon):
        """
        Args:
            x_real: [B, T, C] 真实时间序列
            x_recon: [B, T, C] 重构时间序列
        Returns:
            loss: 感知损失
        """
        # [B, T, C] -> [B, C, T]
        x_real = x_real.transpose(1, 2)
        x_recon = x_recon.transpose(1, 2)
        
        # 提取特征
        feat_real = self.feature_extractor(x_real)
        feat_recon = self.feature_extractor(x_recon)
        
        # 计算特征空间的MSE损失
        loss = F.mse_loss(feat_real, feat_recon)
        
        return loss

