import torch
import torch.nn as nn
import torch.nn.functional as F




from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def configure_optimizers(self, lr=1e-3):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=lr)  # adds weight decay
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        return optimizer


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, compression_factor):
        super(Encoder, self).__init__()
        if compression_factor == 4:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 8:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 12:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=3, padding=1)
            self._conv_4 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 16:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_B = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

    def forward(self, inputs, compression_factor):
        # 如果输入是2D [B, L]，reshape为 [B, 1, L]
        # 如果输入已经是3D [B, C, L]，直接使用
        if inputs.dim() == 2:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
        else:
            x = inputs  # [B, C, L]
        
        if compression_factor == 4:

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 8:
            if inputs.dim() == 2:
                x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
            else:
                x = inputs

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_A(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 12:
            if inputs.dim() == 2:
                x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
            else:
                x = inputs

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = F.relu(x)

            x = self._conv_4(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 16:
            if inputs.dim() == 2:
                x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
            else:
                x = inputs

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_A(x)
            x = F.relu(x)

            x = self._conv_B(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, compression_factor, out_channels=1):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        if compression_factor == 4:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=out_channels,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 8:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._conv_trans_A = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=out_channels,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 12:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            # To get the correct shape back the kernel size has to be 5 not 4
            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=5,
                                                    stride=3, padding=1)

            self._conv_trans_3 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_4 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=out_channels,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 16:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._conv_trans_A = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_B = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=out_channels,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

    def forward(self, inputs, compression_factor):
        if compression_factor == 4:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            # 如果out_channels=1，squeeze以保持向后兼容；否则返回多通道
            if self.out_channels == 1:
                return torch.squeeze(x)
            else:
                return x

        elif compression_factor == 8:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_A(x)
            x = F.relu(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            if self.out_channels == 1:
                return torch.squeeze(x)
            else:
                return x

        elif compression_factor == 12:
            x = self._conv_1(inputs)
            x = self._residual_stack(x)

            x = self._conv_trans_2(x)
            x = F.relu(x)

            x = self._conv_trans_3(x)
            x = F.relu(x)

            x = self._conv_trans_4(x)

            if self.out_channels == 1:
                return torch.squeeze(x)
            else:
                return x

        elif compression_factor == 16:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_A(x)
            x = F.relu(x)

            x = self._conv_trans_B(x)
            x = F.relu(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            if self.out_channels == 1:
                return torch.squeeze(x)
            else:
                return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BLH -> BHL
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2, dim=1) - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, self._embedding.weight, encoding_indices, encodings


class vqvae(BaseModel):
    def __init__(self, vqvae_config):
        super().__init__()
        num_hiddens = vqvae_config['block_hidden_size']
        num_residual_layers = vqvae_config['num_residual_layers']
        num_residual_hiddens = vqvae_config['res_hidden_size']
        embedding_dim = vqvae_config['embedding_dim']
        num_embeddings = vqvae_config['num_embeddings']
        commitment_cost = vqvae_config['commitment_cost']
        self.compression_factor = vqvae_config['compression_factor']

        self.vq = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost
        )
        self.encoder = Encoder(
            1, num_hiddens, num_residual_layers,
            num_residual_hiddens, embedding_dim,
            self.compression_factor
        )
        self.decoder = Decoder(
            embedding_dim, num_hiddens, num_residual_layers,
            num_residual_hiddens, self.compression_factor
        )

    def shared_eval(self, batch, optimizer, mode):
        """mode: 'train' / 'val' / 'test'"""

        if mode == 'train':
            optimizer.zero_grad()

            z = self.encoder(batch, self.compression_factor)
            vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
            
            data_recon = self.decoder(quantized, self.compression_factor)
            recon_error = F.mse_loss(data_recon, batch)

            loss = recon_error + vq_loss
            loss.backward()
            optimizer.step()

        elif mode in ['val', 'test']:
            with torch.no_grad():
                z = self.encoder(batch, self.compression_factor)
                vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)

                data_recon = self.decoder(quantized, self.compression_factor)
                recon_error = F.mse_loss(data_recon, batch)

                loss = recon_error + vq_loss

        return (
            loss, vq_loss, recon_error, data_recon,
            perplexity, embedding_weight, encoding_indices, encodings
        )


class SparseNet(nn.Module):
    """
    轻量稀疏分量预测网络（Robust VQVAE 输入分解模块）

    输入:  x_patch [N, 1, patch_size]
    输出:  s       [N, 1, patch_size]  — 稀疏异常分量

    x_clean = x_patch - s  送入 Encoder，
    重构时: recon = Decoder(z_q) + s

    tanh + amplitude 上界防止网络学走主体结构：
        s = tanh(f(x_patch)) × amplitude
    """
    def __init__(self, patch_size: int, num_hiddens: int, amplitude: float = 0.5):
        super().__init__()
        self.amplitude = amplitude
        hidden = max(1, num_hiddens // 4)
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=3, padding=1),
        )
        # 初始化偏向零，让初始 s ≈ 0（等价于标准 VQVAE 起点）
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, 1, patch_size]
        Returns:
            s: [N, 1, patch_size]，值域 (-amplitude, amplitude)
        """
        return torch.tanh(self.net(x)) * self.amplitude


# =====================================================================
# Trend-Stochastic-Oscillatory 分解组件
# =====================================================================

class TrendExtractor(nn.Module):
    """
    可学习因果低通滤波器：提取 patch 内的局部趋势。

    权重经 softmax 归一化为正且和为 1，等效于加权移动平均。
    因果（左填充）保证不泄露未来信息。
    """
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.raw_weights = nn.Parameter(torch.ones(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x: [N, 1, patch_size]
        Returns: x_trend: [N, 1, patch_size]
        """
        w = F.softmax(self.raw_weights, dim=0).view(1, 1, -1)
        x_pad = F.pad(x, (self.kernel_size - 1, 0), mode='replicate')
        return F.conv1d(x_pad, w)


class StochasticVAE(nn.Module):
    """
    微型 VAE：将残差 r 映射到极低维随机隐变量 z_s，
    再解码为随机分量 s_hat。

    训练时使用重参数化技巧采样，推理时直接使用均值。
    """
    def __init__(self, patch_size: int, latent_dim: int = 4, num_hiddens: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        hidden = max(8, num_hiddens // 2)

        self.enc = nn.Sequential(
            nn.Conv1d(1, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, hidden * patch_size)
        self.dec_conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, 3, padding=1),
        )
        self._hidden = hidden

        nn.init.zeros_(self.dec_conv[-1].weight)
        nn.init.zeros_(self.dec_conv[-1].bias)

    def encode(self, x: torch.Tensor):
        """x: [N, 1, P] → mu, logvar: [N, latent_dim]"""
        h = self.enc(x).squeeze(-1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            return mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [N, latent_dim] → [N, 1, P]"""
        h = self.dec_fc(z).reshape(-1, self._hidden, self.patch_size)
        return self.dec_conv(h)

    def forward(self, x: torch.Tensor):
        """
        Args:  x: [N, 1, P]
        Returns: s_hat [N, 1, P], mu [N, D], logvar [N, D]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
