# -*- coding: utf-8 -*-
"""
tnav_cno.py — Convolutional Neural Operator (Raonic et al., NeurIPS 2023) для
сетки tnav [B, C, T, H, W], без FFT.

Зачем здесь. FNO с модами (12,48,48) на сетке 134x113 физически не может
представить дельта-функцию в одной ячейке — а скважина это точечный источник,
и именно её уставка задаёт амплитуду давления. Плюс диагностика показала, что
провальные кейсы отличаются узкими высокопроницаемыми каналами, то есть
высокочастотной структурой. CNO работает свёртками с band-limited
пере-дискретизацией: локальное восприятие есть, алиасинга нет.

Два CNO-специфичных ингредиента против обычного U-Net:
  1) вместо maxpool/интерполяции — windowed-sinc фильтр (Ханн) как depthwise
     свёртка, то есть честная низкочастотная фильтрация перед прореживанием;
  2) alias-free активация: upsample -> sigma -> low-pass + subsample.
     act_up=1 сводит её к обычной активации (легковесный режим, по умолчанию);
     act_up>=2 — точный вариант, но в 3D стоит act_up^3 объёма активаций.

Отличия от исходного 3D-скрипта, продиктованные нашими данными:
  * сетка НЕ кубическая (T=21, H=134, W=113) и T мало — прореживание по
    времени отключается, когда размер по оси становится меньше 4 (иначе
    sinc-ядро длиной 9 съедает всю ось);
  * ветка работает в пространстве скрытых признаков (width каналов),
    поэтому lifting/projection остаются внешними.
"""

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


class AAact(nn.Module):
    """Alias-free активация; act_up=1 — обычная GELU."""

    def __init__(self, channels, act_up=1, half=2):
        super().__init__()
        self.act_up = act_up
        self.act = nn.GELU()
        if act_up > 1:
            self.lp = LowPass3d(channels, factor=act_up, half=half)

    def forward(self, x):
        if self.act_up == 1:
            return self.act(x)
        s = x.shape[-3:]
        x = F.interpolate(x, scale_factor=self.act_up, mode="nearest")
        x = self.act(self.lp(x, stride=(1, 1, 1)))
        x = self.lp(x, stride=(self.act_up,) * 3)
        if x.shape[-3:] != s:
            x = F.interpolate(x, size=s, mode="trilinear", align_corners=False)
        return x


class ConvBlock(nn.Module):
    """conv3x3x3 -> alias-free активация, residual."""

    def __init__(self, channels, act_up=1, half=2):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)
        self.act = AAact(channels, act_up, half)

    def forward(self, x):
        return x + self.act(self.conv(x))


def _stride_for(shape, min_size=4):
    """Прореживаем ось только если после этого останется >= min_size."""
    return tuple(2 if s // 2 >= min_size else 1 for s in shape)


class CNOBranch(nn.Module):
    """U-образная CNO-ветка в пространстве скрытых признаков.

    Вход/выход: [B, width, T, H, W] — размерность сохраняется, так что ветку
    можно ставить параллельно спектральным блокам или вместо них.
    """

    def __init__(self, width, levels=2, blocks=2, act_up=1, half=2,
                 min_size=4):
        super().__init__()
        self.levels = levels
        self.min_size = min_size
        chs = [width * (2 ** l) for l in range(levels + 1)]

        self.enc = nn.ModuleList()
        self.lp_down = nn.ModuleList()
        self.down_proj = nn.ModuleList()
        for l in range(levels):
            self.enc.append(nn.Sequential(*[ConvBlock(chs[l], act_up, half)
                                            for _ in range(blocks)]))
            self.lp_down.append(LowPass3d(chs[l], factor=2, half=half))
            self.down_proj.append(nn.Conv3d(chs[l], chs[l + 1], 1))

        self.mid = nn.Sequential(*[ConvBlock(chs[levels], act_up, half)
                                   for _ in range(blocks)])

        self.up_proj = nn.ModuleList()
        self.lp_up = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.dec = nn.ModuleList()
        for l in reversed(range(levels)):
            self.up_proj.append(nn.Conv3d(chs[l + 1], chs[l], 1))
            self.lp_up.append(LowPass3d(chs[l], factor=2, half=half))
            self.fuse.append(nn.Conv3d(2 * chs[l], chs[l], 1))
            self.dec.append(nn.Sequential(*[ConvBlock(chs[l], act_up, half)
                                            for _ in range(blocks)]))

    def forward(self, x):
        skips, sizes = [], []
        for enc, lp, proj in zip(self.enc, self.lp_down, self.down_proj):
            x = enc(x)
            skips.append(x)
            sizes.append(x.shape[-3:])
            st = _stride_for(x.shape[-3:], self.min_size)
            x = proj(lp(x, stride=st))
        x = self.mid(x)
        for proj, lp, fuse, dec, skip, size in zip(
                self.up_proj, self.lp_up, self.fuse, self.dec,
                reversed(skips), reversed(sizes)):
            x = proj(x)
            if x.shape[-3:] != tuple(size):
                x = F.interpolate(x, size=tuple(size), mode="nearest")
                x = lp(x, stride=(1, 1, 1))
            x = dec(fuse(torch.cat([x, skip], dim=1)))
        return x
