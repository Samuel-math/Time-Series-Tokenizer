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


class ResidualMLPBlock(nn.Module):
    def __init__(self, width, hidden_width):
        super().__init__()
        self._block = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )
        nn.init.zeros_(self._block[-1].weight)
        nn.init.zeros_(self._block[-1].bias)

    def forward(self, x):
        return x + self._block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 embedding_dim, compression_factor, patch_size=None):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.embedding_dim = embedding_dim
        self.compression_factor = compression_factor
        self.patch_size = patch_size
        self.compressed_len = None
        self.code_dim = None

        self._input_proj = None
        self._residual_mlp = None
        self._output_proj = None
        if patch_size is not None:
            self._build(patch_size)

    def _build(self, patch_size):
        if patch_size % self.compression_factor != 0:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by compression_factor={self.compression_factor}"
            )
        self.patch_size = patch_size
        self.compressed_len = patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len
        input_dim = self.in_channels * patch_size

        self._input_proj = nn.Sequential(
            nn.Linear(input_dim, self.num_hiddens),
            nn.SiLU(),
        )
        self._residual_mlp = nn.Sequential(*[
            ResidualMLPBlock(self.num_hiddens, self.num_residual_hiddens)
            for _ in range(self.num_residual_layers)
        ])
        self._output_proj = nn.Linear(self.num_hiddens, self.code_dim)

    def forward(self, inputs, compression_factor):
        # 如果输入是2D [B, L]，reshape为 [B, 1, L]
        # 如果输入已经是3D [B, C, L]，直接使用
        if inputs.dim() == 2:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
        else:
            x = inputs  # [B, C, L]

        if compression_factor != self.compression_factor:
            raise ValueError(
                f"Encoder was initialized for compression_factor={self.compression_factor}, "
                f"got {compression_factor}"
            )
        if self._input_proj is None:
            self._build(x.shape[-1])
            self.to(device=x.device, dtype=x.dtype)

        x = x.flatten(start_dim=1)
        x = self._input_proj(x)
        x = self._residual_mlp(x)
        x = self._output_proj(x)
        return x.view(x.shape[0], self.embedding_dim, self.compressed_len)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 compression_factor, out_channels=1, patch_size=None):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.compression_factor = compression_factor
        self.patch_size = patch_size
        self.compressed_len = None

        self._input_proj = None
        self._residual_mlp = None
        self._output_proj = None
        if patch_size is not None:
            self._build(patch_size)

    def _build(self, patch_size):
        if patch_size % self.compression_factor != 0:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by compression_factor={self.compression_factor}"
            )
        self.patch_size = patch_size
        self.compressed_len = patch_size // self.compression_factor
        input_dim = self.in_channels * self.compressed_len
        output_dim = self.out_channels * patch_size

        self._input_proj = nn.Sequential(
            nn.Linear(input_dim, self.num_hiddens),
            nn.SiLU(),
        )
        self._residual_mlp = nn.Sequential(*[
            ResidualMLPBlock(self.num_hiddens, self.num_residual_hiddens)
            for _ in range(self.num_residual_layers)
        ])
        self._output_proj = nn.Linear(self.num_hiddens, output_dim)

    def forward(self, inputs, compression_factor):
        if compression_factor != self.compression_factor:
            raise ValueError(
                f"Decoder was initialized for compression_factor={self.compression_factor}, "
                f"got {compression_factor}"
            )
        if self._input_proj is None:
            self._build(inputs.shape[-1] * compression_factor)
            self.to(device=inputs.device, dtype=inputs.dtype)

        x = inputs.flatten(start_dim=1)
        x = self._input_proj(x)
        x = self._residual_mlp(x)
        x = self._output_proj(x)
        x = x.view(x.shape[0], self.out_channels, self.patch_size)
        if self.out_channels == 1:
            return x.squeeze(1)
        return x


class LinearEncoder(nn.Module):
    """Single linear projection from a flattened patch to the VQ code space."""
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 embedding_dim, compression_factor, patch_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.compression_factor = compression_factor
        self.patch_size = patch_size
        self.compressed_len = None
        self.code_dim = None
        self.proj = None
        if patch_size is not None:
            self._build(patch_size)

    def _build(self, patch_size):
        if patch_size % self.compression_factor != 0:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by compression_factor={self.compression_factor}"
            )
        self.patch_size = patch_size
        self.compressed_len = patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len
        self.proj = nn.Linear(self.in_channels * patch_size, self.code_dim)

    def forward(self, inputs, compression_factor):
        if inputs.dim() == 2:
            x = inputs.view(inputs.shape[0], 1, inputs.shape[-1])
        else:
            x = inputs

        if compression_factor != self.compression_factor:
            raise ValueError(
                f"LinearEncoder was initialized for compression_factor={self.compression_factor}, "
                f"got {compression_factor}"
            )
        if self.proj is None:
            self._build(x.shape[-1])
            self.to(device=x.device, dtype=x.dtype)

        x = x.flatten(start_dim=1)
        x = self.proj(x)
        return x.view(x.shape[0], self.embedding_dim, self.compressed_len)


class LinearDecoder(nn.Module):
    """Single linear projection from the VQ code space back to a patch."""
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 compression_factor, out_channels=1, patch_size=None):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.compression_factor = compression_factor
        self.patch_size = patch_size
        self.compressed_len = None
        self.proj = None
        if patch_size is not None:
            self._build(patch_size)

    def _build(self, patch_size):
        if patch_size % self.compression_factor != 0:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by compression_factor={self.compression_factor}"
            )
        self.patch_size = patch_size
        self.compressed_len = patch_size // self.compression_factor
        input_dim = self.in_channels * self.compressed_len
        output_dim = self.out_channels * patch_size
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, inputs, compression_factor):
        if compression_factor != self.compression_factor:
            raise ValueError(
                f"LinearDecoder was initialized for compression_factor={self.compression_factor}, "
                f"got {compression_factor}"
            )
        if self.proj is None:
            self._build(inputs.shape[-1] * compression_factor)
            self.to(device=inputs.device, dtype=inputs.dtype)

        x = inputs.flatten(start_dim=1)
        x = self.proj(x)
        x = x.view(x.shape[0], self.out_channels, self.patch_size)
        if self.out_channels == 1:
            return x.squeeze(1)
        return x


class ConvLinearEncoder(nn.Module):
    """One local Conv1d layer followed by pooling/projection into VQ code space."""
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 embedding_dim, compression_factor, patch_size=None, kernel_size=5):
        super().__init__()
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.embedding_dim = embedding_dim
        self.compression_factor = compression_factor
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.compressed_len = None
        self.code_dim = None

        self.local_conv = None
        self.output_proj = None
        if patch_size is not None:
            self._build(patch_size)

    def _build(self, patch_size):
        if patch_size % self.compression_factor != 0:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by compression_factor={self.compression_factor}"
            )
        if self.kernel_size % 2 == 0:
            raise ValueError("ConvLinearEncoder kernel_size must be odd")
        self.patch_size = patch_size
        self.compressed_len = patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len
        padding = (self.kernel_size - 1) // 2
        self.local_conv = nn.Sequential(
            nn.Conv1d(self.in_channels, self.num_hiddens, self.kernel_size, padding=padding),
            nn.SiLU(),
        )
        self.output_proj = nn.Conv1d(self.num_hiddens, self.embedding_dim, kernel_size=1)

    def forward(self, inputs, compression_factor):
        if inputs.dim() == 2:
            x = inputs.view(inputs.shape[0], 1, inputs.shape[-1])
        else:
            x = inputs

        if compression_factor != self.compression_factor:
            raise ValueError(
                f"ConvLinearEncoder was initialized for compression_factor={self.compression_factor}, "
                f"got {compression_factor}"
            )
        if self.local_conv is None:
            self._build(x.shape[-1])
            self.to(device=x.device, dtype=x.dtype)

        x = self.local_conv(x)
        x = F.adaptive_avg_pool1d(x, self.compressed_len)
        return self.output_proj(x)


class ConvLinearDecoder(nn.Module):
    """One upsampling step and one local Conv1d layer back to the patch space."""
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 compression_factor, out_channels=1, patch_size=None, kernel_size=5):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.compression_factor = compression_factor
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.compressed_len = None

        self.input_proj = None
        self.output_proj = None
        if patch_size is not None:
            self._build(patch_size)

    def _build(self, patch_size):
        if patch_size % self.compression_factor != 0:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by compression_factor={self.compression_factor}"
            )
        if self.kernel_size % 2 == 0:
            raise ValueError("ConvLinearDecoder kernel_size must be odd")
        self.patch_size = patch_size
        self.compressed_len = patch_size // self.compression_factor
        padding = (self.kernel_size - 1) // 2
        self.input_proj = nn.Sequential(
            nn.Conv1d(self.in_channels, self.num_hiddens, kernel_size=1),
            nn.SiLU(),
        )
        self.output_proj = nn.Conv1d(self.num_hiddens, self.out_channels, self.kernel_size, padding=padding)

    def forward(self, inputs, compression_factor):
        if compression_factor != self.compression_factor:
            raise ValueError(
                f"ConvLinearDecoder was initialized for compression_factor={self.compression_factor}, "
                f"got {compression_factor}"
            )
        if self.input_proj is None:
            self._build(inputs.shape[-1] * compression_factor)
            self.to(device=inputs.device, dtype=inputs.dtype)

        x = self.input_proj(inputs)
        x = F.interpolate(x, size=self.patch_size, mode='linear', align_corners=False)
        x = self.output_proj(x)
        if self.out_channels == 1:
            return x.squeeze(1)
        return x


class TCNResidualBlock(nn.Module):
    def __init__(self, width, hidden_width, kernel_size=3, dilation=1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("TCNResidualBlock requires an odd kernel_size")
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.SiLU(),
            nn.Conv1d(width, hidden_width, kernel_size, padding=padding, dilation=dilation),
            nn.SiLU(),
            nn.Conv1d(hidden_width, width, kernel_size, padding=padding, dilation=dilation),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, x):
        return x + self.block(x)


class TCNEncoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 embedding_dim, compression_factor, patch_size=None, kernel_size=5):
        super().__init__()
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.embedding_dim = embedding_dim
        self.compression_factor = compression_factor
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.compressed_len = None
        self.code_dim = None

        self.stem = None
        self.residual_tcn = None
        self.output_proj = None
        if patch_size is not None:
            self._build(patch_size)

    def _build(self, patch_size):
        if patch_size % self.compression_factor != 0:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by compression_factor={self.compression_factor}"
            )
        if self.kernel_size % 2 == 0:
            raise ValueError("TCN encoder kernel_size must be odd")
        self.patch_size = patch_size
        self.compressed_len = patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len
        padding = (self.kernel_size - 1) // 2
        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, self.num_hiddens, self.kernel_size, padding=padding),
            nn.SiLU(),
        )
        self.residual_tcn = nn.Sequential(*[
            TCNResidualBlock(
                self.num_hiddens,
                self.num_residual_hiddens,
                kernel_size=self.kernel_size,
                dilation=2 ** (i % 3),
            )
            for i in range(self.num_residual_layers)
        ])
        self.output_proj = nn.Conv1d(self.num_hiddens, self.embedding_dim, kernel_size=1)

    def forward(self, inputs, compression_factor):
        if inputs.dim() == 2:
            x = inputs.view(inputs.shape[0], 1, inputs.shape[-1])
        else:
            x = inputs

        if compression_factor != self.compression_factor:
            raise ValueError(
                f"TCNEncoder was initialized for compression_factor={self.compression_factor}, "
                f"got {compression_factor}"
            )
        if self.stem is None:
            self._build(x.shape[-1])
            self.to(device=x.device, dtype=x.dtype)

        x = self.stem(x)
        x = self.residual_tcn(x)
        x = F.adaptive_avg_pool1d(x, self.compressed_len)
        return self.output_proj(x)


class TCNDecoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 compression_factor, out_channels=1, patch_size=None, kernel_size=5):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.compression_factor = compression_factor
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.compressed_len = None

        self.input_proj = None
        self.residual_tcn = None
        self.output_proj = None
        if patch_size is not None:
            self._build(patch_size)

    def _build(self, patch_size):
        if patch_size % self.compression_factor != 0:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by compression_factor={self.compression_factor}"
            )
        if self.kernel_size % 2 == 0:
            raise ValueError("TCN decoder kernel_size must be odd")
        self.patch_size = patch_size
        self.compressed_len = patch_size // self.compression_factor
        padding = (self.kernel_size - 1) // 2
        self.input_proj = nn.Sequential(
            nn.Conv1d(self.in_channels, self.num_hiddens, kernel_size=1),
            nn.SiLU(),
        )
        self.residual_tcn = nn.Sequential(*[
            TCNResidualBlock(
                self.num_hiddens,
                self.num_residual_hiddens,
                kernel_size=self.kernel_size,
                dilation=2 ** (i % 3),
            )
            for i in range(self.num_residual_layers)
        ])
        self.output_proj = nn.Conv1d(self.num_hiddens, self.out_channels, self.kernel_size, padding=padding)

    def forward(self, inputs, compression_factor):
        if compression_factor != self.compression_factor:
            raise ValueError(
                f"TCNDecoder was initialized for compression_factor={self.compression_factor}, "
                f"got {compression_factor}"
            )
        if self.input_proj is None:
            self._build(inputs.shape[-1] * compression_factor)
            self.to(device=inputs.device, dtype=inputs.dtype)

        x = self.input_proj(inputs)
        x = F.interpolate(x, size=self.patch_size, mode='linear', align_corners=False)
        x = self.residual_tcn(x)
        x = self.output_proj(x)
        if self.out_channels == 1:
            return x.squeeze(1)
        return x


class ChunkMLPEncoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 embedding_dim, compression_factor, patch_size=None, chunk_size=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.embedding_dim = embedding_dim
        self.compression_factor = compression_factor
        self.patch_size = patch_size
        self.chunk_size = chunk_size
        self.compressed_len = None
        self.code_dim = None

        self.local_proj = None
        self.fuse_proj = None
        self.residual_mlp = None
        self.output_proj = None
        if patch_size is not None:
            self._build(patch_size)

    def _build(self, patch_size):
        if patch_size % self.compression_factor != 0:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by compression_factor={self.compression_factor}"
            )
        if patch_size % self.chunk_size != 0:
            raise ValueError(f"patch_size={patch_size} must be divisible by chunk_size={self.chunk_size}")
        self.patch_size = patch_size
        self.compressed_len = patch_size // self.compression_factor
        self.code_dim = self.embedding_dim * self.compressed_len
        self.num_chunks = patch_size // self.chunk_size
        local_hidden = max(4, self.num_hiddens // max(1, self.num_chunks))

        self.local_proj = nn.Sequential(
            nn.Linear(self.in_channels * self.chunk_size, local_hidden),
            nn.SiLU(),
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(self.num_chunks * local_hidden, self.num_hiddens),
            nn.SiLU(),
        )
        self.residual_mlp = nn.Sequential(*[
            ResidualMLPBlock(self.num_hiddens, self.num_residual_hiddens)
            for _ in range(self.num_residual_layers)
        ])
        self.output_proj = nn.Linear(self.num_hiddens, self.code_dim)

    def forward(self, inputs, compression_factor):
        if inputs.dim() == 2:
            x = inputs.view(inputs.shape[0], 1, inputs.shape[-1])
        else:
            x = inputs

        if compression_factor != self.compression_factor:
            raise ValueError(
                f"ChunkMLPEncoder was initialized for compression_factor={self.compression_factor}, "
                f"got {compression_factor}"
            )
        if self.local_proj is None:
            self._build(x.shape[-1])
            self.to(device=x.device, dtype=x.dtype)

        bsz = x.shape[0]
        x = x.reshape(bsz, self.in_channels, self.num_chunks, self.chunk_size)
        x = x.permute(0, 2, 1, 3).reshape(bsz, self.num_chunks, self.in_channels * self.chunk_size)
        x = self.local_proj(x).flatten(start_dim=1)
        x = self.fuse_proj(x)
        x = self.residual_mlp(x)
        x = self.output_proj(x)
        return x.view(bsz, self.embedding_dim, self.compressed_len)


class ChunkMLPDecoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 compression_factor, out_channels=1, patch_size=None, chunk_size=2):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.compression_factor = compression_factor
        self.patch_size = patch_size
        self.chunk_size = chunk_size
        self.compressed_len = None

        self.input_proj = None
        self.residual_mlp = None
        self.chunk_proj = None
        if patch_size is not None:
            self._build(patch_size)

    def _build(self, patch_size):
        if patch_size % self.compression_factor != 0:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by compression_factor={self.compression_factor}"
            )
        if patch_size % self.chunk_size != 0:
            raise ValueError(f"patch_size={patch_size} must be divisible by chunk_size={self.chunk_size}")
        self.patch_size = patch_size
        self.compressed_len = patch_size // self.compression_factor
        self.num_chunks = patch_size // self.chunk_size
        input_dim = self.in_channels * self.compressed_len
        local_hidden = max(4, self.num_hiddens // max(1, self.num_chunks))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.num_hiddens),
            nn.SiLU(),
        )
        self.residual_mlp = nn.Sequential(*[
            ResidualMLPBlock(self.num_hiddens, self.num_residual_hiddens)
            for _ in range(self.num_residual_layers)
        ])
        self.chunk_proj = nn.Linear(self.num_hiddens, self.num_chunks * local_hidden)
        self.local_out = nn.Linear(local_hidden, self.out_channels * self.chunk_size)

    def forward(self, inputs, compression_factor):
        if compression_factor != self.compression_factor:
            raise ValueError(
                f"ChunkMLPDecoder was initialized for compression_factor={self.compression_factor}, "
                f"got {compression_factor}"
            )
        if self.input_proj is None:
            self._build(inputs.shape[-1] * compression_factor)
            self.to(device=inputs.device, dtype=inputs.dtype)

        bsz = inputs.shape[0]
        x = inputs.flatten(start_dim=1)
        x = self.input_proj(x)
        x = self.residual_mlp(x)
        x = self.chunk_proj(x).view(bsz, self.num_chunks, -1)
        x = self.local_out(x).view(bsz, self.num_chunks, self.out_channels, self.chunk_size)
        x = x.permute(0, 2, 1, 3).reshape(bsz, self.out_channels, self.patch_size)
        if self.out_channels == 1:
            return x.squeeze(1)
        return x


def _build_mlp_encoder(config, common):
    return Encoder(**common)


def _build_mlp_decoder(config, common):
    return Decoder(**common)


def _build_linear_encoder(config, common):
    return LinearEncoder(**common)


def _build_linear_decoder(config, common):
    return LinearDecoder(**common)


def _build_conv_linear_encoder(config, common):
    return ConvLinearEncoder(**common, kernel_size=int(config.get('vqvae_tcn_kernel_size', 5)))


def _build_conv_linear_decoder(config, common):
    return ConvLinearDecoder(**common, kernel_size=int(config.get('vqvae_tcn_kernel_size', 5)))


def _build_tcn_encoder(config, common):
    return TCNEncoder(**common, kernel_size=int(config.get('vqvae_tcn_kernel_size', 5)))


def _build_tcn_decoder(config, common):
    return TCNDecoder(**common, kernel_size=int(config.get('vqvae_tcn_kernel_size', 5)))


def _build_chunk_mlp_encoder(config, common):
    return ChunkMLPEncoder(**common, chunk_size=int(config.get('vqvae_chunk_size', 2)))


def _build_chunk_mlp_decoder(config, common):
    return ChunkMLPDecoder(**common, chunk_size=int(config.get('vqvae_chunk_size', 2)))


VQVAE_ENCODER_BUILDERS = {
    'mlp': _build_mlp_encoder,
    'linear': _build_linear_encoder,
    'conv_linear': _build_conv_linear_encoder,
    'tcn': _build_tcn_encoder,
    'chunk_mlp': _build_chunk_mlp_encoder,
}

VQVAE_DECODER_BUILDERS = {
    'mlp': _build_mlp_decoder,
    'linear': _build_linear_decoder,
    'conv_linear': _build_conv_linear_decoder,
    'tcn': _build_tcn_decoder,
    'chunk_mlp': _build_chunk_mlp_decoder,
}


def build_encoder(config, in_channels=1):
    backbone = str(config.get('vqvae_backbone', 'mlp')).lower()
    common = dict(
        in_channels=in_channels,
        num_hiddens=config['num_hiddens'],
        num_residual_layers=config['num_residual_layers'],
        num_residual_hiddens=config['num_residual_hiddens'],
        embedding_dim=config['embedding_dim'],
        compression_factor=config['compression_factor'],
        patch_size=config.get('patch_size'),
    )
    try:
        return VQVAE_ENCODER_BUILDERS[backbone](config, common)
    except KeyError as exc:
        supported = ', '.join(sorted(VQVAE_ENCODER_BUILDERS))
        raise ValueError(f"Unsupported vqvae_backbone={backbone!r}; supported: {supported}") from exc


def build_decoder(config, in_channels=None, out_channels=1):
    backbone = str(config.get('vqvae_backbone', 'mlp')).lower()
    decoder_in_channels = config['embedding_dim'] if in_channels is None else in_channels
    common = dict(
        in_channels=decoder_in_channels,
        num_hiddens=config['num_hiddens'],
        num_residual_layers=config['num_residual_layers'],
        num_residual_hiddens=config['num_residual_hiddens'],
        compression_factor=config['compression_factor'],
        out_channels=out_channels,
        patch_size=config.get('patch_size'),
    )
    try:
        return VQVAE_DECODER_BUILDERS[backbone](config, common)
    except KeyError as exc:
        supported = ', '.join(sorted(VQVAE_DECODER_BUILDERS))
        raise ValueError(f"Unsupported vqvae_backbone={backbone!r}; supported: {supported}") from exc


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
        patch_size = vqvae_config.get('patch_size', None)

        self.vq = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost
        )
        codec_config = {
            'patch_size': patch_size,
            'embedding_dim': embedding_dim,
            'compression_factor': self.compression_factor,
            'num_hiddens': num_hiddens,
            'num_residual_layers': num_residual_layers,
            'num_residual_hiddens': num_residual_hiddens,
            'vqvae_backbone': vqvae_config.get('vqvae_backbone', 'mlp'),
            'vqvae_tcn_kernel_size': vqvae_config.get('vqvae_tcn_kernel_size', 5),
            'vqvae_chunk_size': vqvae_config.get('vqvae_chunk_size', 2),
        }
        self.encoder = build_encoder(codec_config, in_channels=1)
        self.decoder = build_decoder(codec_config, in_channels=embedding_dim, out_channels=1)

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

