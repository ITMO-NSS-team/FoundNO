import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def windowed_sinc_1d(factor: int = 2, half: int = 2) -> torch.Tensor:
    """Ханн-оконный sinc для пере-дискретизации в `factor` раз; сумма = 1."""
    K = 2 * half * factor + 1
    n = torch.arange(K, dtype=torch.float64) - (K - 1) / 2.0
    arg = math.pi * (1.0 / factor) * n
    sinc = torch.where(arg.abs() < 1e-12, torch.ones_like(arg), torch.sin(arg) / arg)
    win = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(K, dtype=torch.float64) / (K - 1))
    h = sinc * win
    return (h / h.sum()).float()


class LowPass3d(nn.Module):
    """Depthwise windowed-sinc фильтр по (T, H, W).

    axes — по каким осям фильтровать/прореживать. Для коротких осей (T)
    прореживание можно отключить, передав stride=1 по этой оси.
    """

    def __init__(self, channels, factor=2, half=2):
        super().__init__()
        self.channels = channels
        h = windowed_sinc_1d(factor, half)
        k3 = h[:, None, None] * h[None, :, None] * h[None, None, :]
        self.register_buffer("kernel", k3[None, None].repeat(channels, 1, 1, 1, 1))
        self.pad = (k3.shape[-1] - 1) // 2

    def forward(self, x, stride=(1, 1, 1)):
        return F.conv3d(x, self.kernel, stride=stride, padding=self.pad,
                        groups=self.channels)


class LowPass3dAutoreg(nn.Module):
    """Depthwise windowed-sinc фильтр по (1, H, W), адаптированный для авторегрессионных НО.

    axes — по каким осям фильтровать/прореживать. Для коротких осей (T)
    прореживание можно отключить, передав stride=1 по этой оси.
    """

    def __init__(self, channels, factor=2, half=2):
        super().__init__()
        self.channels = channels
        h = windowed_sinc_1d(factor, half)
        k3 = h[:, None] * h[None, :] 
        self.register_buffer("kernel", k3[None, None].repeat(channels, 1, 1, 1))
        self.pad = (k3.shape[-1] - 1) // 2

    def forward(self, x: torch.Tensor, stride=(1, 1)):
        inp_dim = x.ndim
        if inp_dim == 5:
            assert x.shape[2] == 1, 'LowPass3dAutoreg has to be applied to input with 1 previous time frame'
            x = x.squeeze(2)
        elif inp_dim != 4:
            raise RuntimeError(f'LowPass3dAutoreg arg. has to be 4 or 5 - dimensional, instead got {inp_dim}')

        x = F.conv2d(x, self.kernel, stride=stride, padding=self.pad,
                     groups=self.channels)
        if inp_dim == 5:
            x = x.unsqueeze(2) # Revert to the original shape
        return x


class AAact(nn.Module):
    """Alias-free активация; act_up=1 — обычная GELU."""

    def __init__(self, channels, act_up=1, half=2, autoregressive_mode: bool = False):
        super().__init__()
        self._autoreg_mode = autoregressive_mode
        self.act_up = act_up
        self.act = nn.GELU()
        if act_up > 1:
            if autoregressive_mode:
                self.lp = LowPass3dAutoreg(channels, factor=act_up, half=half)
            else:
                self.lp = LowPass3d(channels, factor=act_up, half=half)

    def forward(self, x):
        if self.act_up == 1:
            return self.act(x)
        s = x.shape[-3:]
        x = F.interpolate(x, scale_factor=self.act_up, mode="nearest")
        if x.ndim == 5:
            stride, s_multipl = (1, 1, 1), 3
        else:
            stride, s_multipl = (1, 1,), 2

        x = self.act(self.lp(x, stride = stride))
        x = self.lp(x, stride = (self.act_up,) * s_multipl)
        if x.shape[-s_multipl:] != s:
            x = F.interpolate(x, size=s, mode="trilinear", align_corners=False)
        return x


class ConvBlock(nn.Module):
    """conv3x3x3 -> alias-free активация, residual."""

    def __init__(self, channels, act_up=1, half=2, autoregressive_mode: bool = False):
        super().__init__()
        self._autoregressive_mode = autoregressive_mode
        if autoregressive_mode:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        else:  
            self.conv = nn.Conv3d(channels, channels, 3, padding=1)
        self.act = AAact(channels, act_up, half,
                         autoregressive_mode=autoregressive_mode)

    def forward(self, x):
        if self._autoregressive_mode and x.ndim == 5:
            to_unsqueeze = True
            x = x.squeeze(2)
        else:
            to_unsqueeze = False

        x = x + self.act(self.conv(x))

        if to_unsqueeze:
            x = x.unsqueeze(2)
        return x


def _stride_for(shape, min_size=4):
    """Прореживаем ось только если после этого останется >= min_size."""
    return tuple(2 if s // 2 >= min_size else 1 for s in shape)


class CNOBranch(nn.Module):
    """U-образная CNO-ветка в пространстве скрытых признаков.

    Вход/выход: [B, width, T, H, W] — размерность сохраняется, так что ветку
    можно ставить параллельно спектральным блокам или вместо них.
    """

    def __init__(self, width, levels=2, blocks=2, act_up=1, half=2,
                 min_size=4, autoregressive_mode: bool = False):
        super().__init__()
        self._autoregressive_mode = autoregressive_mode
        if self._autoregressive_mode:
            LowPassCls = LowPass3dAutoreg
            ConvCls = nn.Conv2d
        else:
            LowPassCls = LowPass3d
            ConvCls = nn.Conv3d

        self.levels = levels
        self.min_size = min_size
        chs = [width * (2 ** l) for l in range(levels + 1)]

        self.enc = nn.ModuleList()
        self.lp_down = nn.ModuleList()
        self.down_proj = nn.ModuleList()
        for l in range(levels):
            self.enc.append(nn.Sequential(*[ConvBlock(chs[l], act_up, half, autoregressive_mode=autoregressive_mode)
                                            for _ in range(blocks)]))
            self.lp_down.append(LowPassCls(chs[l], factor=2, half=half))
            self.down_proj.append(ConvCls(chs[l], chs[l + 1], 1))

        self.mid = nn.Sequential(*[ConvBlock(chs[levels], act_up, half, autoregressive_mode=autoregressive_mode)
                                   for _ in range(blocks)])

        self.up_proj = nn.ModuleList()
        self.lp_up = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.dec = nn.ModuleList()
        for l in reversed(range(levels)):
            self.up_proj.append(ConvCls(chs[l + 1], chs[l], 1))
            self.lp_up.append(LowPassCls(chs[l], factor=2, half=half))
            self.fuse.append(ConvCls(2 * chs[l], chs[l], 1))
            self.dec.append(nn.Sequential(*[ConvBlock(chs[l], act_up, half, autoregressive_mode=autoregressive_mode)
                                            for _ in range(blocks)]))

    def forward(self, x):
        if x.ndim == 5 and self._autoregressive_mode:
            to_unsqueeze = True
            x = x.squeeze(2)

        if self._autoregressive_mode:
            dim_idx = 2
            stride = (1, 1)
        else:
            dim_idx = 3
            stride = (1, 1, 1)
        
        skips, sizes = [], []
        for enc, lp, proj in zip(self.enc, self.lp_down, self.down_proj):
            x = enc(x)
            skips.append(x)
            sizes.append(x.shape[-dim_idx:])
            st = _stride_for(x.shape[-dim_idx:], self.min_size)
            x = proj(lp(x, stride=st))
        x = self.mid(x)
        for proj, lp, fuse, dec, skip, size in zip(
                self.up_proj, self.lp_up, self.fuse, self.dec,
                reversed(skips), reversed(sizes)):
            x = proj(x)
            if x.shape[-dim_idx:] != tuple(size):
                x = F.interpolate(x, size=tuple(size), mode="nearest")
                x = lp(x, stride=stride)
            x = dec(fuse(torch.cat([x, skip], dim=1)))

        if to_unsqueeze:
            x = x.unsqueeze(2)

        return x