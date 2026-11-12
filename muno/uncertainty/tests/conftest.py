"""Маленький Muno (лифтинг -> FNO-ядро -> проекция) для тестов luno_torch на CPU.

Размеры подобраны так, чтобы якобиан и GGN материализовались плотно: эталоном
служит ``torch.func.jacrev``, а не код, который тестируется.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest
import torch

UNCERTAINTY_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = UNCERTAINTY_DIR.parents[1]
for _p in (str(PROJECT_ROOT), str(UNCERTAINTY_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from luno_torch.synthetic import SyntheticMuno  # noqa: E402


def TinyMuno(seed: int = 0):
    return SyntheticMuno(seed=seed)


def dense_jacobian(model_fn, x, w0):
    return torch.func.jacrev(lambda w: model_fn(x, w).reshape(-1))(w0)


def dense_ggn(model_fn, xs, w0):
    G = torch.zeros(w0.numel(), w0.numel(), dtype=w0.dtype)
    for x in xs:
        J = dense_jacobian(model_fn, x, w0)
        G += 2.0 * J.T @ J
    return G


def close(a, b, tol: float = 1e-5) -> bool:
    """``max|a - b| <= tol * max|b|``.

    Допуск уровня float32, хотя веса float64: ``SpectralConv`` в neuralop пишет
    спектр в буфер ``complex64``, поэтому сам forward точен лишь до ~1e-7.
    Настоящие ошибки (не тот модуль, потерянный множитель) дают O(1).
    """
    a, b = a.detach(), b.detach()
    return bool((a - b).abs().max() <= tol * b.abs().max())


@pytest.fixture(autouse=True)
def _quiet_stdout():
    # В forward блока FNO стоит отладочный print: глушим, чтобы вывод тестов читался.
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@pytest.fixture(scope="module")
def tiny():
    with contextlib.redirect_stdout(io.StringIO()):
        return TinyMuno()
