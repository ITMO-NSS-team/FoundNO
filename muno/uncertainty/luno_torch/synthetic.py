"""Синтетический Muno (лифтинг -> FNO-ядро -> проекция) со случайными весами.

Нужен там, где важна структура сети, а не её качество: тесты (маленькие размеры,
плотный эталон) и ``bench_luno.py`` (реальные размеры, замер времени) -- без
чекпойнтов и данных.
"""
from __future__ import annotations

import inspect

import torch
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.spectral_convolution import SpectralConv

from .adapter import TorchFNOWrapper, TorchLiftingWrapper
from .affine import AffineLastBlock

if "enforce_hermitian_symmetry" in inspect.signature(SpectralConv.__init__).parameters:
    _Conv = SpectralConv
else:
    class _Conv(SpectralConv):
        """Релизный neuralop старше, чем ждёт muno.layers.fno_block: глотаем лишний аргумент."""

        def __init__(self, *args, enforce_hermitian_symmetry=None, **kwargs):
            super().__init__(*args, **kwargs)


class SyntheticMuno:
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_channels: int = 4,
        n_modes: tuple[int, ...] = (2, 4, 4),
        n_layers: int = 2,
        grid: tuple[int, ...] = (4, 6, 6),
        adapter_hidden: int = 5,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ):
        from muno.models.fno import FNO

        torch.manual_seed(seed)
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        n_dim = len(n_modes)
        self.in_channels, self.grid, self.dtype = in_channels, tuple(grid), dtype
        self.device = torch.device(device)
        self.lifting = ChannelMLP(
            in_channels=in_channels, out_channels=hidden_channels,
            hidden_channels=adapter_hidden, n_layers=2, n_dim=n_dim,
        ).to(device=self.device, dtype=dtype)
        self.core = FNO(
            n_modes=tuple(n_modes), in_channels=hidden_channels,
            out_channels=hidden_channels, hidden_channels=hidden_channels,
            n_layers=n_layers, disable_lifting_and_projection=True,
            conv_module=_Conv,
        ).to(self.device)
        # Не core.to(dtype): это отбросило бы мнимую часть спектральных весов.
        for p in self.core.parameters():
            p.data = p.data.to(cdtype if p.is_complex() else dtype)
        self.projection = ChannelMLP(
            in_channels=hidden_channels, out_channels=out_channels,
            hidden_channels=adapter_hidden, n_layers=2, n_dim=n_dim,
        ).to(device=self.device, dtype=dtype)
        # Нулевые после инициализации bias делают часть якобиана тривиальной.
        with torch.no_grad():
            for mod in (self.lifting, self.core, self.projection):
                for name, p in mod.named_parameters():
                    if "bias" in name:
                        p.normal_(std=0.3)

        self.wrapper = TorchFNOWrapper(self.core, dtype=dtype)
        self.w0 = self.wrapper.w0
        self.affine = AffineLastBlock(
            self.core, self.projection, self.wrapper.keys, dtype=dtype
        )
        self.lift_wrapper = TorchLiftingWrapper(
            self.lifting, self.core, self.projection, dtype=dtype
        )

    def model_fn(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Полная модель с весами последнего блока ``w`` (как ``make_model_fn``)."""
        sd = self.wrapper.reconstruct(w)
        z = self.lifting(x.to(dtype=self.dtype, device=self.device))
        out = torch.func.functional_call(self.core, sd, (z,))
        return self.projection(out)

    def xs(self, n: int, seed: int = 1) -> list[torch.Tensor]:
        g = torch.Generator().manual_seed(seed)
        return [
            torch.randn(1, self.in_channels, *self.grid, dtype=self.dtype, generator=g)
            for _ in range(n)
        ]
