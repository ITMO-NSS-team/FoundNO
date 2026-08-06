"""
rno.py - 2D Riesz Neural Operator for AirfRANS.

Ported from the AirfRANS experiments (airrans RNO/model.py).
Architecture based on the RNO paper (OpenReview Vjw7q1quNt):

    1. Lifting (CoordToRiesz): input -> hidden space
    2. Riesz Conductors: spectral mixer with Riesz transform
       - global spectral weight (like FNO)
       - direction-aware weights (Riesz transform along X and Y)
       - zeta parameter suppressing grid noise
    3. Projection (RieszToCoord): hidden space -> output

Key differences from FNO:
  - Riesz transform: in addition to the plain spectral multiplication,
    normalized gradients (j * k_i / ||k||) are computed, letting the model
    capture field change directions.
  - Direction-aware mixing: each layer mixes the global spectrum and the
    spectral gradients along each direction.
  - zeta limiter: scales the Riesz contribution, limiting noise.

Works on channel-last tensors [B, H, W, C_in] -> [B, H, W, C_out].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RieszConductor2d(nn.Module):
    """
    Riesz Conductor - spectral layer with Riesz transform.

    Computes:
      out = IFFT( W_global * Q + zeta * W_x * (R_x * Q) + zeta * W_y * (R_y * Q) )

    where:
      Q = FFT(q) - input spectrum
      R_x = j * k_x / ||k|| - Riesz multiplier for X
      R_y = j * k_y / ||k|| - Riesz multiplier for Y
      zeta - trainable scaling factor (bounded from above)
      W_global, W_x, W_y - complex weights (truncated to modes, like FNO)
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # modes along H
        self.modes2 = modes2  # modes along W (rfft -> W//2+1)

        self.scale = 1.0 / (in_channels * out_channels)

        # Weights: global spectrum (like FNO) + 2 directions (X, Y)
        # Low and high modes along H, low modes along W (rfft)
        self.w_global_lo = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                    dtype=torch.cfloat)
        )
        self.w_global_hi = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                    dtype=torch.cfloat)
        )
        self.w_dir_x_lo = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                    dtype=torch.cfloat)
        )
        self.w_dir_x_hi = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                    dtype=torch.cfloat)
        )
        self.w_dir_y_lo = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                    dtype=torch.cfloat)
        )
        self.w_dir_y_hi = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                    dtype=torch.cfloat)
        )

        # zeta - trainable noise limiter (Formula 11)
        # Initialized small, zeta < pi/6 ~ 0.524
        self.zeta_raw = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1) ~ 0.27

    @property
    def zeta(self):
        """zeta bounded from above via sigmoid: zeta in (0, pi/6)."""
        return torch.sigmoid(self.zeta_raw) * (np.pi / 6.0)

    def _compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def _build_riesz_kernels(self, H, W, device):
        """Build Riesz multipliers R_x, R_y in the frequency domain."""
        k_y = torch.fft.fftfreq(H, device=device).view(-1, 1)   # [H, 1]
        k_x = torch.fft.rfftfreq(W, device=device).view(1, -1)  # [1, W//2+1]

        k_norm = torch.sqrt(k_y ** 2 + k_x ** 2) + 1e-8         # [H, W//2+1]

        j = torch.tensor(1j, device=device, dtype=torch.cfloat)

        riesz_x = (j * k_x) / k_norm
        riesz_y = (j * k_y) / k_norm
        return riesz_x, riesz_y

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: [B, C_out, H, W]
        """
        B, C, H, W = x.shape
        device = x.device

        # 1. FFT of the input
        x_ft = torch.fft.rfft2(x)  # [B, C, H, W//2+1]

        # 2. Riesz multipliers
        riesz_x, riesz_y = self._build_riesz_kernels(H, W, device)

        # 3. Spectral gradients (Riesz transform)
        R_x = x_ft * riesz_x.unsqueeze(0)
        R_y = x_ft * riesz_y.unsqueeze(0)

        # 4. Output tensor in the frequency domain
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=device)

        z = self.zeta
        sl_lo = slice(None, self.modes1)
        sl_w = slice(None, self.modes2)

        # Global spectrum + Riesz X/Y (low modes along H)
        out_ft[:, :, sl_lo, sl_w] += self._compl_mul2d(
            x_ft[:, :, sl_lo, sl_w], self.w_global_lo
        )
        out_ft[:, :, sl_lo, sl_w] += z * self._compl_mul2d(
            R_x[:, :, sl_lo, sl_w], self.w_dir_x_lo
        )
        out_ft[:, :, sl_lo, sl_w] += z * self._compl_mul2d(
            R_y[:, :, sl_lo, sl_w], self.w_dir_y_lo
        )

        # High modes along H
        sl_hi = slice(-self.modes1, None)
        out_ft[:, :, sl_hi, sl_w] += self._compl_mul2d(
            x_ft[:, :, sl_hi, sl_w], self.w_global_hi
        )
        out_ft[:, :, sl_hi, sl_w] += z * self._compl_mul2d(
            R_x[:, :, sl_hi, sl_w], self.w_dir_x_hi
        )
        out_ft[:, :, sl_hi, sl_w] += z * self._compl_mul2d(
            R_y[:, :, sl_hi, sl_w], self.w_dir_y_hi
        )

        # 5. Inverse FFT
        out = torch.fft.irfft2(out_ft, s=(H, W))
        return out


class RNO2d(nn.Module):
    """
    2D Riesz Neural Operator for AirfRANS.

    Architecture:
      1. Lifting: Linear(in_channels + grid -> width)
      2. n_layers x RieszConductor2d + bypass Conv1x1 + GELU
      3. Projection: Linear(width -> 128) -> GELU -> Linear(128 -> out_channels)

    Input:  [B, H, W, C_in] (+ 2 grid channels if use_grid=True)
    Output: [B, H, W, C_out]
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=4,
                 modes=16,
                 width=32,
                 n_layers=4,
                 use_grid=True):
        super().__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        self.n_layers = n_layers
        self.use_grid = use_grid

        # Lifting (CoordToRiesz)
        grid_channels = 2 if use_grid else 0
        self.fc0 = nn.Linear(in_channels + grid_channels, self.width)

        # Riesz Conductors + bypass
        self.riesz_conductors = nn.ModuleList()
        self.bypass_convs = nn.ModuleList()

        for _ in range(n_layers):
            self.riesz_conductors.append(
                RieszConductor2d(self.width, self.width, self.modes1, self.modes2)
            )
            self.bypass_convs.append(nn.Conv2d(self.width, self.width, 1))

        # Projection (RieszToCoord)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def get_grid(self, x):
        """Coordinate grid [B, H, W, 2] in [0, 1]."""
        batchsize, H, W = x.shape[0], x.shape[1], x.shape[2]
        grid_x = torch.linspace(0, 1, W, device=x.device).reshape(1, 1, W, 1)
        grid_y = torch.linspace(0, 1, H, device=x.device).reshape(1, H, 1, 1)
        grid = torch.cat([
            grid_x.expand(batchsize, H, W, 1),
            grid_y.expand(batchsize, H, W, 1)
        ], dim=-1)
        return grid

    def forward(self, x, mask=None, grid_mesh=None):
        """
        x: [B, H, W, C_in]
        mask: [B, H, W, 1] or None
        grid_mesh: ignored (kept for a uniform interface with DNO)
        returns: [B, H, W, C_out]
        """
        # Append grid coordinates
        if self.use_grid:
            grid = self.get_grid(x)
            x = torch.cat([x, grid], dim=-1)

        # Lifting
        x = self.fc0(x)                        # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)             # [B, width, H, W]

        # Riesz layers
        for riesz, bypass in zip(self.riesz_conductors, self.bypass_convs):
            x_riesz = riesz(x)
            x_bypass = bypass(x)
            x = F.gelu(x_riesz + x_bypass)

        # Projection
        x = x.permute(0, 2, 3, 1)             # [B, H, W, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)                        # [B, H, W, C_out]

        if mask is not None:
            x = x * mask

        return x
