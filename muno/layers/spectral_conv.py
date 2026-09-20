"""
spectral_conv.py - simple 2D spectral convolution (Fourier layer).

Ported from the AirfRANS experiments (airrans FNO/DNO/Geo-FNO models).
Unlike neuralop's SpectralConv (factorized tensors, channel MLP etc.),
this is a minimal implementation: rfft2 -> mode truncation with complex
weights -> irfft2.
"""

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """2D spectral convolution (Fourier layer)."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Fourier modes to keep along dim 1 (H)
        self.modes2 = modes2  # Fourier modes to keep along dim 2 (W)

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                    dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                    dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        """Complex multiplication via einsum."""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """x: [B, C, H, W] -> [B, C_out, H, W]"""
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Low modes (first modes along H)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # High modes (last modes along H)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
