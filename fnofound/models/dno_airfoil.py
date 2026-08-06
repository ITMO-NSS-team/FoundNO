"""
dno_airfoil.py - DNO (Diffeomorphism Neural Operator) for AirfRANS.

Exact port of the AirfRANS DNO (airrans DNO/model.py), checkpoint-compatible
with the trained models in airrans (DNO/runs/*/model_best.pth).

Architecture:
    fc0 (lifting) -> N x (SpectralConv2d + w(x) + b(grid) + c(grid_mesh))
    -> fc1(128) -> fc2

Works on channel-last tensors [B, H, W, C_in] -> [B, H, W, C_out].

Differences from fnofound.models.dno.DNO (darcy/fluid/reservoir):
  - own simple SpectralConv2d (not neuralop)
  - projection head fc1(128) -> fc2 (not Linear(256) -> GELU -> Linear)
  - geometry (grid_mesh) passed explicitly to forward(), not taken from
    input channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fnofound.layers.spectral_conv import SpectralConv2d


class DNO(nn.Module):
    """
    Diffeomorphism Neural Operator (FNO2d + geometry terms in every layer).

    Parameters
    ----------
    in_channels : int
        Input channels (3: vx_in, vy_in, mask).
    out_channels : int
        Output channels (4: vel_x, vel_y, pressure, nu_t).
    modes : int
        Fourier modes (same for both axes).
    width : int
        Hidden width.
    n_layers : int
        Number of FNO blocks.
    use_grid : bool
        Append the universal grid (xi, eta) to the fc0 input.
    use_geom : bool
        Add geometry terms b(grid) + c(grid_mesh) in every layer.
        When False, the model is a plain FNO - compatible with the early
        airrans checkpoints (DNO/runs/*) trained without geometry.
    padding : int
        Zero-padding buffer against FFT boundary artifacts (0 = none).

    Input:  x [B, H, W, C_in], mask [B, H, W, 1] or None,
            grid_mesh [B, H, W, 2] physical coordinates (grid_x, grid_y)
            or None - then geometry terms are disabled
    Output: [B, H, W, C_out]
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=4,
                 modes=16,
                 width=32,
                 n_layers=4,
                 use_grid=True,
                 use_geom=True,
                 padding=0):
        super().__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        self.n_layers = n_layers
        self.use_grid = use_grid
        self.use_geom = use_geom
        self.padding = padding

        # Input projection: channels + optional universal grid (xi, eta)
        grid_channels = 2 if use_grid else 0
        self.fc0 = nn.Linear(in_channels + grid_channels, self.width)

        # FNO blocks: spectral + w(x) + (optionally) b(grid) + c(grid_mesh)
        self.spectral_convs = nn.ModuleList()
        self.conv_ws = nn.ModuleList()          # w(x) - local residual
        self.conv_grids = nn.ModuleList()       # b(grid) - universal grid
        self.conv_meshes = nn.ModuleList()      # c(grid_mesh) - geometry

        for _ in range(n_layers):
            self.spectral_convs.append(
                SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            )
            self.conv_ws.append(nn.Conv2d(self.width, self.width, 1))
            if use_geom:
                self.conv_grids.append(nn.Conv2d(2, self.width, 1))
                self.conv_meshes.append(nn.Conv2d(2, self.width, 1))

        # Output layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def get_grid(self, B, H, W, device):
        """Universal grid [0, 1]^2 -> [B, H, W, 2] (channel-last)."""
        grid_x = torch.linspace(0, 1, W, device=device).reshape(1, 1, W, 1)
        grid_y = torch.linspace(0, 1, H, device=device).reshape(1, H, 1, 1)
        return torch.cat([
            grid_x.expand(B, H, W, 1),
            grid_y.expand(B, H, W, 1)
        ], dim=-1)

    def forward(self, x, mask=None, grid_mesh=None):
        """
        x:         [B, H, W, C_in] - fields on the regular (universal) grid
        mask:      [B, H, W, 1] or None - mask of valid points
        grid_mesh: [B, H, W, 2] physical coordinates (geometry) or None
        """
        B, H, W, _ = x.shape
        device = x.device

        # Universal grid (xi, eta) -> fc0 input
        if self.use_grid:
            grid = self.get_grid(B, H, W, device)
            x = torch.cat([x, grid], dim=-1)        # [B, H, W, C_in + 2]

        # Lift -> hidden space
        x = self.fc0(x)                             # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)                   # [B, width, H, W]

        # Grids for b(grid), c(grid_mesh) terms - channel-first
        grid_c = self.get_grid(B, H, W, device).permute(0, 3, 1, 2)
        mesh_c = None
        if self.use_geom and grid_mesh is not None:
            mesh_c = grid_mesh.permute(0, 3, 1, 2)

        # Padding buffer for non-periodic boundaries (zeros)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
            grid_c = F.pad(grid_c, [0, self.padding, 0, self.padding])
            if mesh_c is not None:
                mesh_c = F.pad(mesh_c, [0, self.padding, 0, self.padding])

        # FNO blocks with geometry
        for k in range(self.n_layers):
            y = self.spectral_convs[k](x) + self.conv_ws[k](x)
            if self.use_geom:
                y = y + self.conv_grids[k](grid_c)
                if mesh_c is not None:
                    y = y + self.conv_meshes[k](mesh_c)
            x = F.gelu(y)

        # Crop padding
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        # Decode
        x = x.permute(0, 2, 3, 1)                   # [B, H, W, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)                             # [B, H, W, C_out]

        # Mask
        if mask is not None:
            x = x * mask

        return x
