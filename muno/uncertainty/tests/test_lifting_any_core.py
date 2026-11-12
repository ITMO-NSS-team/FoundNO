"""Лифтинг-UQ не зависит от архитектуры ядра.

Ядро -- обычная CNN без FNO-весов, лифтинг -- ``PostLiftMambaLifting`` (в
pure-PyTorch варианте ``SimpleSSM``, как в чекпойнтах checkpoints_mamba).
Эталон -- плотный якобиан ``torch.func.jacrev``.
"""
from __future__ import annotations

import argparse

import pytest
import torch
from neuralop.layers.channel_mlp import ChannelMLP

from luno_torch.adapter import TorchLiftingWrapper
from luno_torch.ggn import LiftingGGNMatvec, to_dense
from luno_torch.jacobian import LiftingWeightJacobian, check_jacobian

from .conftest import close, dense_ggn

DTYPE = torch.float64
GRID = (3, 5, 5)


class TinyCNNCore(torch.nn.Module):
    """Ядро без спектральных весов: две 3D-свёртки с GELU."""

    def __init__(self, ch):
        super().__init__()
        self.out_channels = ch
        self.c1 = torch.nn.Conv3d(ch, ch, 3, padding=1, padding_mode="circular")
        self.c2 = torch.nn.Conv3d(ch, ch, 3, padding=1, padding_mode="circular")

    def forward(self, x):
        return x + self.c2(torch.nn.functional.gelu(self.c1(x)))


def _mamba_lifting(in_ch, hid):
    from muno.utils.model_factory import _post_lift_mamba_lifting

    cls = _post_lift_mamba_lifting()
    lift = cls(in_channels=in_ch, out_channels=hid, padding=0, n_dim=3,
               use_mamba_kwargs=None, mamba_fallback_kernel=3)
    # Тест про архитектуру, а не про mamba_ssm: фиксируем pure-PyTorch вариант.
    for m in lift.modules():
        if hasattr(m, "_use_mamba"):
            assert not m._use_mamba, "ожидался SimpleSSM-фолбэк без CUDA-ядер"
    return lift.to(DTYPE)


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    hid, in_ch, out_ch = 4, 2, 2
    lifting = _mamba_lifting(in_ch, hid)
    core = TinyCNNCore(hid).to(DTYPE)
    projection = ChannelMLP(in_channels=hid, out_channels=out_ch,
                            hidden_channels=5, n_layers=2, n_dim=3).to(DTYPE)
    return TorchLiftingWrapper(lifting, core, projection, dtype=DTYPE)


def _xs(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 2, *GRID, dtype=DTYPE, generator=g) for _ in range(n)]


def test_wrapper_metadata(net):
    assert net.num_output_channels == 2  # проекция, а не скрытая ширина ядра
    assert not net.lifting.training
    names = [n for n, *_ in net.param_layout()]
    assert any("post_lift_ssm" in n for n in names)  # SSM-часть тоже линеаризуется


def test_lifting_ggn_matches_dense_on_cnn_core(net):
    xs = _xs(3)
    G = dense_ggn(net.model_fn, xs, net.w0)
    mv = LiftingGGNMatvec(net.model_fn, net.w0, xs, fwd_batch=4)
    assert close(to_dense(mv), G, tol=1e-9)


def test_jacobian_check_passes(net):
    ok, rel = check_jacobian(net.model_fn, _xs(1)[0], net.w0,
                             jacobian_cls=LiftingWeightJacobian, fwd_batch=2)
    assert ok, rel


def test_script_builds_lifting_setup_without_fno(net, monkeypatch):
    run = pytest.importorskip("run_luno_ggn_calibrate")
    monkeypatch.setattr(run, "DTYPE", DTYPE)
    args = argparse.Namespace(ggn_target="lifting", jv_mode="forward",
                              ggn_method="randomized", prefix_cache=True)
    setup = run.build_ggn_setup(args, net.core, None, None, net.lifting, net.projection)
    assert setup.target == "lifting" and setup.w0.numel() == net.w0.numel()
    x = _xs(1)[0]
    assert close(setup.model_fn(x, setup.w0), net.model_fn(x, net.w0))
    with pytest.raises(ValueError, match="last-block"):
        run.build_ggn_setup(argparse.Namespace(**{**vars(args), "ggn_target": "last-block"}),
                            net.core, None, None, net.lifting, net.projection)
