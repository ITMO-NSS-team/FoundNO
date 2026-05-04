"""
DNO — Deep Neural Operator with Geometry Injection.

Architecture:
  1. GridEmbedding — append normalized logical coordinates (ξ, η)
  2. Lifting — Linear(C_in + 2, hidden)
  3. N spectral layers — each with skip + geometry bias terms
  4. Projection — Linear(hidden, 256) → GELU → Linear(256, C_out)

Geometry Injection:
  b_layers — bias from the logical grid (universal [0,1]²)
  c_layers — bias from the physical mesh (X_map, Y_map)

Per-case channel layout:
  # darcy:     [C, X_map, Y_map, Mask]        → geometry_channels=(1, 2)
  # fluid:     [X_map, Y_map, Re]             → geometry_channels=(0, 1)
  # reservoir: [X_map, Y_map, P0, S0, Src...] → geometry_channels=(0, 1)
"""

from fnofound.layers.embeddings import GridEmbeddingND
from neuralop.layers.spectral_convolution import SpectralConv
import torch.nn as nn
import torch.nn.functional as F


class DNO(nn.Module):
    def __init__(self, n_modes, in_channels, out_channels,
                 hidden_channels, n_layers=4,
                 geometry_channels=(0, 1)):
        """
        Parameters
        ----------
        geometry_channels : tuple (int, int)
            Indices of X_map and Y_map channels in the input tensor.
            darcy:     (1, 2)  — [C, X, Y, Mask]
            fluid:     (0, 1)  — [X, Y, Re]
            reservoir: (0, 1)  — [X, Y, P0, S0, Src...]
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.geometry_channels = geometry_channels

        # 1. Lifting layer: project input (with grid) to hidden dim
        #    +2 for the grid coordinates appended by GridEmbeddingND
        self.lifting = nn.Linear(in_channels + 2, hidden_channels)

        # 2. Spectral convolution layers with skip connections
        self.convs = nn.ModuleList([
            SpectralConv(hidden_channels, hidden_channels, n_modes)
            for _ in range(n_layers)
        ])
        self.skips = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, 1)
            for _ in range(n_layers)
        ])

        # 3. Geometry Injection (bias layers)
        self.b_layers = nn.ModuleList([
            nn.Conv2d(2, hidden_channels, 1) for _ in range(n_layers)
        ])
        self.c_layers = nn.ModuleList([
            nn.Conv2d(2, hidden_channels, 1) for _ in range(n_layers)
        ])

        # 4. Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, 256),
            nn.GELU(),
            nn.Linear(256, out_channels),
        )

        self.grid_emb = GridEmbeddingND()

    def forward(self, x):
        # x: [B, H, W, C] 
        # Extract geometry channels (X_map, Y_map) for c_layers bias
        g0, g1 = self.geometry_channels
        grid_mesh = x[..., [g0, g1]].permute(0, 3, 1, 2)  # [B, 2, H, W]

        x = self.grid_emb(x)                     # [B, H, W, C+2]
        x = self.lifting(x).permute(0, 3, 1, 2)  # [B, hidden, H, W]

        # Logical grid [0, 1] for b_layers — in NCHW format
        grid = self.grid_emb.grid                 # [1, H, W, 2]
        logical_grid = grid.permute(0, 3, 1, 2).expand(
            x.shape[0], -1, -1, -1)              # [B, 2, H, W]

        for i in range(len(self.convs)):
            res = self.convs[i](x)
            skip = self.skips[i](x)
            # Geometry injection: bias from logical + physical grids
            x = (res + skip
                 + self.b_layers[i](logical_grid)
                 + self.c_layers[i](grid_mesh))
            x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)  # back to NHWC
        return self.projection(x)
