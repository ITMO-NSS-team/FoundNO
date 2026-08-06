"""
fno2d.py - 2D Fourier Neural Operator for AirfRANS (regular grid).

Ported from the AirfRANS experiments (airrans FNO/model.py + Geo_FNO/model.py).

Architecture:
    fc0 (lifting) -> N x (SpectralConv2d + 1x1 conv, GELU) -> fc1(128) -> fc2

Works on channel-last tensors [B, H, W, C_in] -> [B, H, W, C_out].
`use_grid=True` appends the logical grid (xi, eta) in [0, 1]^2 to the input.

The `padding` parameter covers the Geo-FNO case: zero-padding before the
Fourier blocks and cropping after absorbs FFT boundary artifacts on
non-periodic grids (e.g. the unfolded C-grid). With padding=0 this is the
plain FNO; with padding>0 it is the Geo-FNO model (the geometry itself is
baked into the C-grid data).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fnofound.layers.spectral_conv import SpectralConv2d


def remap_geofno_keys(state_dict: dict) -> dict:
    """Remap Geo-FNO checkpoint keys (blocks.N.spectral.*, blocks.N.conv_w.*)
    to FNO2d keys (spectral_convs.N.*, conv_ws.N.*).

    Geo-FNO models (airrans Geo_FNO/model.py) wrap each Fourier layer in an
    FNOBlock: blocks.N.spectral (SpectralConv2d) + blocks.N.conv_w (1x1 conv).
    Our FNO2d stores them as flat ModuleLists, so a plain load_state_dict
    would fail. fc0/fc1/fc2 keys are identical and pass through.
    """
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith('blocks.'):
            parts = k.split('.')
            layer_idx = parts[1]
            sub = parts[2]
            if sub == 'spectral':
                new_k = f'spectral_convs.{layer_idx}.{".".join(parts[3:])}'
            elif sub == 'conv_w':
                new_k = f'conv_ws.{layer_idx}.{".".join(parts[3:])}'
            else:
                new_k = k
        else:
            new_k = k
        remapped[new_k] = v
    return remapped


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator on a regular grid [B, H, W, C].

    Parameters
    ----------
    in_channels : int
        Number of input channels (3: vx_in, vy_in, mask; grid coords are
        appended automatically when use_grid=True).
    out_channels : int
        Number of output channels (4: vel_x, vel_y, pressure, nu_t).
    modes : int
        Number of Fourier modes (same for both dimensions).
    width : int
        Hidden width.
    n_layers : int
        Number of Fourier layers.
    use_grid : bool
        Append grid coordinates as extra input channels.
    padding : int
        Zero-padding buffer against FFT boundary artifacts
        (0 = none; >0 = Geo-FNO style on the unfolded C-grid).
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=4,
                 modes=16,
                 width=32,
                 n_layers=4,
                 use_grid=True,
                 padding=0):
        super().__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        self.n_layers = n_layers
        self.use_grid = use_grid
        self.padding = padding

        # Input projection
        grid_channels = 2 if use_grid else 0
        self.fc0 = nn.Linear(in_channels + grid_channels, self.width)

        # Fourier layers
        self.spectral_convs = nn.ModuleList()
        self.conv_ws = nn.ModuleList()  # W(x) - residual connection

        for _ in range(n_layers):
            self.spectral_convs.append(
                SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            )
            self.conv_ws.append(nn.Conv2d(self.width, self.width, 1))

        # Output layers
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
        x: [B, H, W, C_in] - fields on the regular grid
        mask: [B, H, W, 1] or None - mask of valid points
        grid_mesh: ignored (kept for a uniform interface with DNO)
        returns: [B, H, W, C_out]
        """
        # Append grid coordinates
        if self.use_grid:
            grid = self.get_grid(x)
            x = torch.cat([x, grid], dim=-1)  # [B, H, W, C_in + 2]

        # Lift to width dimension
        x = self.fc0(x)                        # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)             # [B, width, H, W]

        # Padding buffer for non-periodic boundaries (zeros)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        # Fourier layers
        for spectral_conv, conv_w in zip(self.spectral_convs, self.conv_ws):
            x1 = spectral_conv(x)
            x2 = conv_w(x)
            x = F.gelu(x1 + x2)

        # Crop padding
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        # Decode
        x = x.permute(0, 2, 3, 1)             # [B, H, W, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)                        # [B, H, W, C_out]

        # Apply mask if given
        if mask is not None:
            x = x * mask

        return x
