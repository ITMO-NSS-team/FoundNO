# -*- coding: utf-8 -*-
"""
tnav_model.py — Temporal-SSM (Mamba по времени) + FNO3D для схемы v6.

Тензорная раскладка везде [B, C, T, H, W]; временной SSM каузален
(прошлое -> будущее) и стоит после lifting, до domain padding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO

MAMBA_AVAILABLE = False
MambaClass = None
try:
    import mamba_ssm as _mamba_pkg
    if hasattr(_mamba_pkg, "Mamba"):
        MambaClass = _mamba_pkg.Mamba
        MAMBA_AVAILABLE = True
except Exception:
    MAMBA_AVAILABLE = False


class CausalTemporalConv(nn.Module):
    """Фолбэк: каузальная gated depthwise-свёртка по времени. Токены [B, L, C]."""

    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.k = kernel_size
        self.dw = nn.Conv1d(dim, dim, kernel_size, groups=dim)
        self.gate = nn.Conv1d(dim, dim, 1)
        self.out = nn.Conv1d(dim, dim, 1)
        for m in (self.dw, self.gate, self.out):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, tok):
        x = tok.transpose(1, 2)                              # [B, C, L]
        y = torch.tanh(self.dw(F.pad(x, (self.k - 1, 0))))   # pad слева => каузально
        g = torch.sigmoid(self.gate(x))
        y = self.out((1 - g) * x + g * y)
        return y.transpose(1, 2)


class GatedRasterConv(nn.Module):
    """Фолбэк для пространственного скана: некаузальная gated depthwise-свёртка
    вдоль растровой последовательности (аналог SimpleSSM из исходного
    PostLiftMambaFNO). Токены [B, L, C]."""

    def __init__(self, dim, kernel_size=9):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=pad, groups=dim)
        self.gate = nn.Conv1d(dim, dim, 1)
        for m in (self.dw, self.gate):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, tok):
        x = tok.transpose(1, 2)                              # [B, C, L]
        y = torch.tanh(self.dw(x))
        g = torch.sigmoid(self.gate(x))
        return ((1 - g) * x + g * y).transpose(1, 2)


class TemporalProcessor(nn.Module):
    """SSM строго вдоль времени: для каждой точки (h, w) — последовательность
    длины T, батч = B*H*W. Вход/выход: [B, C, T, H, W]."""

    def __init__(self, channels, kind='auto', d_state=16, d_conv=4,
                 fallback_kernel=5):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.kind = 'conv'
        if kind in ('auto', 'mamba') and MAMBA_AVAILABLE:
            try:
                self.ssm = MambaClass(d_model=channels, d_state=d_state,
                                      d_conv=d_conv, expand=2)
                self.kind = 'mamba'
            except Exception as e:
                print(f"Mamba init failed ({e}); fallback -> causal conv")
                self.ssm = CausalTemporalConv(channels, fallback_kernel)
        else:
            if kind == 'mamba':
                print("mamba_ssm недоступна — использую каузальную свёртку по времени")
            self.ssm = CausalTemporalConv(channels, fallback_kernel)
        print(f"TemporalProcessor: {self.kind} (axis=time)")

    def forward(self, x):
        B, C, T, H, W = x.shape
        tokens = x.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, T, C)
        y = self.ssm(self.norm(tokens))
        y = y.view(B, H, W, T, C).permute(0, 4, 3, 1, 2).contiguous()
        return x + y


class SpatialProcessor(nn.Module):
    """PostLift-скан по ПРОСТРАНСТВУ — вариант исходного PostLiftMambaFNO3D
    с исправленными осями: для каждого timestep — растровая последовательность
    длины H*W, батч = B*T. Скан некаузален по смыслу (порядок растра
    произволен); фолбэк — gated-свёртка вдоль растра (аналог SimpleSSM).
    Вход/выход: [B, C, T, H, W]."""

    def __init__(self, channels, kind='auto', d_state=16, d_conv=4,
                 fallback_kernel=9):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.kind = 'conv'
        if kind in ('auto', 'mamba') and MAMBA_AVAILABLE:
            try:
                self.ssm = MambaClass(d_model=channels, d_state=d_state,
                                      d_conv=d_conv, expand=2)
                self.kind = 'mamba'
            except Exception as e:
                print(f"Mamba init failed ({e}); fallback -> raster conv")
                self.ssm = GatedRasterConv(channels, fallback_kernel)
        else:
            self.ssm = GatedRasterConv(channels, fallback_kernel)
        print(f"SpatialProcessor: {self.kind} (axis=raster H*W)")

    def forward(self, x):
        B, C, T, H, W = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).contiguous().view(B * T, H * W, C)
        y = self.ssm(self.norm(tokens))
        y = y.view(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return x + y


class GlobalToken(nn.Module):
    """Глобальный агрегат по всему пласту.

    Мотивация: спектральная свёртка собирает каждую точку выхода из
    ограниченного набора мод — механизма «вычислить одно число на весь пласт
    и сдвинуть им уровень» в архитектуре нет. Диагностика показала, что
    ошибка в провальных кейсах именно такая: почти постоянное отношение
    pred/target по всему полю.

    Здесь: маскированное среднее скрытых признаков по (H, W) на каждый
    timestep -> MLP по каналам -> вектор, прибавляемый ко ВСЕМ позициям.
    Последний слой инициализирован нулями, поэтому на старте модуль —
    тождество, и модель стартует точно как базовая.
    """

    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or 2 * channels
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Conv1d(channels, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, channels, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h, mask=None):
        B, C, T, H, W = h.shape
        if mask is not None:
            m = mask.view(B, 1, 1, H, W)
            area = m.sum(dim=(3, 4)).clamp_min(1.0)          # [B,1,1]
            pooled = (h * m).sum(dim=(3, 4)) / area           # [B,C,T]
        else:
            pooled = h.mean(dim=(3, 4))
        z = self.norm(pooled.transpose(1, 2)).transpose(1, 2)  # LayerNorm по C
        g = self.fc2(F.gelu(self.fc1(z)))                      # [B,C,T]
        return h + g.unsqueeze(-1).unsqueeze(-1)


GLOBAL_TOKEN_MODES = ('off', 'post', 'pre', 'both')
ARCHS = ('fno', 'temporal', 'spatial', 'spatial-temporal')
LOCAL_MODES = ('off', 'parallel', 'only')
DIFF_MODES = ('off', 'parallel')


class TnavTemporalFNO3D(FNO):
    def __init__(self, in_channels, out_channels, modes=(12, 32, 32), width=64,
                 n_layers=4, rank=0.10, domain_padding=0.15, temporal='auto',
                 d_state=16, d_conv=4, dropout=0.1, residual_u0=True,
                 arch='temporal', global_token='off', local_branch='off',
                 cno_levels=2, cno_blocks=2, cno_act_up=1,
                 diff_kernels='off', diff_kernel_size=3,
                 diff_padding='replicate', aux_out=0):
        super().__init__(
            n_modes=tuple(modes),
            hidden_channels=width,
            in_channels=in_channels,
            out_channels=out_channels,
            factorization='tucker',
            rank=rank,
            implementation='factorized',
            n_layers=n_layers,
            use_channel_mlp=True,
            channel_mlp_dropout=dropout,
            positional_embedding=None,
            domain_padding=domain_padding,
        )
        assert arch in ARCHS, f"arch must be one of {ARCHS}"
        self.arch = arch
        self.spatial = SpatialProcessor(width, kind=temporal, d_state=d_state,
                                        d_conv=d_conv) \
            if arch in ('spatial', 'spatial-temporal') else None
        self.temporal = TemporalProcessor(width, kind=temporal,
                                          d_state=d_state, d_conv=d_conv) \
            if arch in ('temporal', 'spatial-temporal') else None
        if arch == 'fno':
            print("Arch: чистый FNO3D (без SSM-блоков)")
        # --- CNO-ветка: локальное восприятие, которого нет у спектральных
        # блоков с усечёнными модами (скважина = точечный источник, узкие
        # высокопроницаемые каналы = высокие частоты).
        #   parallel — параллельно спектральному стеку, сумма с обучаемым
        #              масштабом alpha (нулевая инициализация => старт как
        #              у базовой модели, сравнение честное);
        #   only     — вместо спектрального стека (чистый CNO, FFT-free).
        assert local_branch in LOCAL_MODES
        self.local_branch = local_branch
        if local_branch != 'off':
            from tnav_cno import CNOBranch
            self.cno = CNOBranch(width, levels=cno_levels, blocks=cno_blocks,
                                 act_up=cno_act_up)
            self.cno_alpha = (torch.nn.Parameter(torch.zeros(1))
                              if local_branch == 'parallel' else None)
            n = sum(p.numel() for p in self.cno.parameters())
            print(f"CNO-ветка: {local_branch} "
                  f"(levels={cno_levels}, blocks={cno_blocks}, "
                  f"act_up={cno_act_up}, {n/1e6:.2f}M параметров"
                  f"{', alpha=0 на старте' if local_branch == 'parallel' else ''})")
        else:
            self.cno = None
            self.cno_alpha = None

        # --- дифференциальные ядра (LocalNO, Liu-Schiaffini et al. 2024) ---
        # Свёрточный слой, отнормированный так, что при измельчении сетки он
        # сходится к дифференциальному оператору (по мотивам конечно-разностных
        # шаблонов). Ставится ПАРАЛЛЕЛЬНО каждому спектральному блоку, как в
        # LocalNO: out = spectral(x) + alpha_i * diff(x, h). Наша задача —
        # диффузия давления, то есть дифференциальный оператор; в статье это
        # ровно тот режим, где такие ядра дают наибольший выигрыш, а локальные
        # интегральные (DISCO) — почти нет.
        # alpha инициализирован нулями => старт совпадает с базовой моделью.
        assert diff_kernels in DIFF_MODES
        self.diff_kernels = diff_kernels
        if diff_kernels != 'off':
            from neuralop.layers.differential_conv import (
                FiniteDifferenceConvolution as _FD)
            self.diff = nn.ModuleList([
                _FD(in_channels=width, out_channels=width, n_dim=3,
                    kernel_size=diff_kernel_size, groups=1,
                    padding=diff_padding)
                for _ in range(n_layers)])
            self.diff_alpha = nn.Parameter(torch.zeros(n_layers))
            n = sum(p.numel() for p in self.diff.parameters())
            print(f"Диф. ядра (LocalNO): {diff_kernels}, k={diff_kernel_size}, "
                  f"padding={diff_padding}, {n/1e3:.1f}k параметров, alpha=0")
        else:
            self.diff = None
            self.diff_alpha = None

        # --- вспомогательная голова: фактические дебиты в completion-ячейках ---
        # Дебит у скважины = проводимость * градиент давления, то есть та
        # прискважинная физика, где сидит хвост ошибки. Голова обязана её
        # предсказать => скрытое представление вынуждено разрешать
        # окрестность скважины честно (multi-task). На инференс основной
        # выход не влияет: forward() возвращает только поле, дебиты —
        # через forward_with_aux(), который использует только обучение.
        self.aux_out = int(aux_out)
        if self.aux_out > 0:
            self.aux_head = nn.Sequential(
                nn.Conv3d(width, width, 1), nn.GELU(),
                nn.Conv3d(width, self.aux_out, 1))
            print(f"Aux-голова (дебиты): {self.aux_out} канала, "
                  f"{sum(p.numel() for p in self.aux_head.parameters())/1e3:.1f}k параметров")
        else:
            self.aux_head = None

        assert global_token in GLOBAL_TOKEN_MODES
        self.global_token = global_token
        self.gt_pre = (GlobalToken(width)
                       if global_token in ('pre', 'both') else None)
        self.gt_post = (GlobalToken(width)
                        if global_token in ('post', 'both') else None)
        if global_token != 'off':
            n = sum(p.numel() for p in
                    list(self.gt_pre.parameters() if self.gt_pre else [])
                    + list(self.gt_post.parameters() if self.gt_post else []))
            print(f"GlobalToken: {global_token} ({n/1e3:.1f}k параметров, "
                  f"нулевая инициализация)")
        self.residual_u0 = residual_u0
        self.n_out = out_channels

    def forward(self, x):
        # x: [B, IN_CH, T, H, W]; каналы 0:n_out — u0 (нормализованные)
        u0 = x[:, :self.n_out]
        # канал 2 входа — активная маска пласта (0/1), постоянна во времени
        gmask = x[:, 2, 0] if x.shape[1] > 2 else None
        h = self.lifting(x)
        if self.spatial is not None:
            h = self.spatial(h)
        if self.temporal is not None:
            h = self.temporal(h)
        if self.gt_pre is not None:
            h = self.gt_pre(h, gmask)

        h_local = self.cno(h) if self.cno is not None else None

        if self.local_branch == 'only':
            h = h_local
        else:
            h_in = h
            if self.domain_padding is not None:
                h = self.domain_padding.pad(h)
            # grid_width: шаг сетки в нормированных координатах [0,1].
            # Спектральная часть его не видит, а дифференциальному ядру он
            # нужен для корректной нормировки (сходимость при h->0).
            gw = 1.0 / max(h.shape[-2], h.shape[-1])
            for i in range(self.n_layers):
                h_in = h
                h = self.fno_blocks(h, i)
                if self.diff is not None:
                    h = h + self.diff_alpha[i] * self.diff[i](h_in, gw)
            if self.domain_padding is not None:
                h = self.domain_padding.unpad(h)
            if h_local is not None:
                h = h + self.cno_alpha * h_local
        if self.gt_post is not None:
            h = self.gt_post(h, gmask)
        out = self.projection(h)
        if self.residual_u0:
            out = out + u0
        self._last_hidden = h if self.aux_head is not None else None
        return out

    def forward_with_aux(self, x):
        """(поле, aux) — aux [B, aux_out, T, H, W] из того же скрытого h."""
        out = self.forward(x)
        aux = self.aux_head(self._last_hidden) if self.aux_head is not None else None
        self._last_hidden = None
        return out, aux


def build_model(cfg, in_channels, out_channels):
    return TnavTemporalFNO3D(
        in_channels=in_channels, out_channels=out_channels,
        modes=tuple(cfg['modes']), width=cfg['width'],
        n_layers=cfg['n_layers'], rank=cfg['rank'],
        domain_padding=cfg['domain_padding'], temporal=cfg['temporal_ssm'],
        d_state=cfg['d_state'], d_conv=cfg['d_conv'],
        dropout=cfg['dropout'], residual_u0=cfg['residual_u0'],
        arch=cfg.get('arch', 'temporal'),
        global_token=cfg.get('global_token', 'off'),
        local_branch=cfg.get('local_branch', 'off'),
        cno_levels=cfg.get('cno_levels', 2),
        cno_blocks=cfg.get('cno_blocks', 2),
        cno_act_up=cfg.get('cno_act_up', 1),
        diff_kernels=cfg.get('diff_kernels', 'off'),
        diff_kernel_size=cfg.get('diff_kernel_size', 3),
        diff_padding=cfg.get('diff_padding', 'replicate'),
        aux_out=cfg.get('aux_out', 0),
    )
