from __future__ import annotations

from dataclasses import dataclass

import torch

from .jacobian import LastFNOBlockWeightJacobian
from .lino_ops import LinearOperator


@dataclass
class LowRankTerms:
    """Low-rank Gaussian-Newton curvature terms: A ~= U diag(S) U^T."""

    U: torch.Tensor
    S: torch.Tensor
    scalar: torch.Tensor | float = 0.0


class GGNMatvec(LinearOperator):
    """Generalized Gauss-Newton matvec of the last FNO block weights.

    G v = factor * sum_i J_i^T H J_i v, where for ``loss_fn == "mse"`` the pointwise
    Hessian is H = 2 I (laplax convention). ``model_fn(x, w)`` should be the wrapped
    last-block-parameter forward; ``xs`` is a list of per-sample inputs.
    """

    def __init__(
        self,
        model_fn: callable,
        w0: torch.Tensor,
        xs: list[torch.Tensor],
        loss_fn: str = "mse",
        factor: float | torch.Tensor = 1.0,
        num_output_channels: int | None = None,
        vjp_chunk: int = 64,
    ):
        self._model_fn = model_fn
        self._w0 = w0
        self._xs = [x.detach().to(dtype=w0.dtype, device=w0.device) for x in xs]
        self._loss_fn = loss_fn
        if isinstance(factor, torch.Tensor):
            self._factor = factor.to(dtype=w0.dtype, device=w0.device)
        else:
            self._factor = torch.tensor(factor, dtype=w0.dtype, device=w0.device)
        self._jacobians = [
            LastFNOBlockWeightJacobian(
                model_fn,
                x,
                w0,
                num_output_channels=num_output_channels,
                vjp_chunk=vjp_chunk,
            )
            for x in self._xs
        ]

    def shape(self) -> tuple[int, int]:
        d = self._w0.numel()
        return d, d

    @property
    def dtype(self) -> torch.dtype:
        return self._w0.dtype

    @property
    def device(self) -> torch.device:
        return self._w0.device

    def _matmul(self, v: torch.Tensor) -> torch.Tensor:
        v = v.to(self._w0)
        single = v.dim() == 1
        vv = v.unsqueeze(-1) if single else v
        g = torch.zeros_like(vv)
        for jac in self._jacobians:
            Jv = jac._matmul(vv)
            cot = 2.0 * Jv
            g = g + jac._vjp_cols(cot)
        g = self._factor * g
        return g.squeeze(-1) if single else g


def randomized_eigh(
    mv: LinearOperator,
    n: int,
    rank: int,
    oversample: int = 10,
    power_iter: int = 2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomized symmetric eigendecomposition (Halko) with power iterations.

    Returns ``(U, S)`` with orthonormal columns, ``U diag(S) U^T ~= A``.
    """
    q = min(rank + oversample, n)
    gen = None
    if mv.device.type == "cpu":
        gen = torch.Generator(device="cpu")
    else:
        gen = torch.Generator(device=mv.device)
    gen.manual_seed(seed)

    Q = torch.randn(n, q, dtype=mv.dtype, device=mv.device, generator=gen)
    for _ in range(power_iter):
        Y = mv._matmul(Q)
        Q, _ = torch.linalg.qr(Y)
    Y = mv._matmul(Q)
    Q, _ = torch.linalg.qr(Y)

    T = mv._matmul(Q)
    B = Q.T @ T
    B = (B + B.T) / 2.0
    evals, evecs = torch.linalg.eigh(B)
    order = torch.argsort(evals, descending=True)
    U = Q @ evecs[:, order[:rank]]
    S = evals[order[:rank]]
    return U, S


def skerch_low_rank(
    mv: LinearOperator,
    rank: int = 100,
    inner_rank: int | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> LowRankTerms:
    """Low-rank estimate of a symmetric positive matvec using skerch if available.

    Falls back to a randomized eigendecomposition when skerch is not installed.
    Supports multiple skerch APIs:
      * legacy ``skerch.decompositions.seigh`` (returns a 3-tuple (Q, U, S));
      * modern ``skerch.algorithms.seigh`` (returns ``(Lambda, Q)`` with
        ``A ~= Q diag(Lambda) Q^H``, keyword ``outer_dims``);
      * ``skerch.seigh`` (same numpy-style modern API).
    """
    if inner_rank is None:
        inner_rank = rank
    try:
        import skerch.linops as _ll  # noqa: F401 — ensures skerch is importable

        class TorchOp:
            def __init__(self, shape, dtype_):
                self.shape = shape
                self.dtype = dtype_
                self.device = torch.device(device)

            def __matmul__(self, x):
                return mv._matmul(x.to(device=mv.device))

            def __rmatmul__(self, x):
                return x.to(device=mv.device) @ mv

        op = TorchOp(mv.shape(), dtype)
        try:
            from skerch.algorithms import seigh

            res = seigh(
                op,
                lop_device=device,
                lop_dtype=dtype,
                outer_dims=rank
            )
            U = res[0] @ res[1]
            S = res[2]
            U, S = U.to(dtype=dtype), S.to(dtype=dtype)
        except ImportError:
            try:
                from skerch.algorithms import seigh
            except ImportError:
                from skerch import seigh
            try:
                res = seigh(
                    op,
                    op_device=device,
                    op_dtype=dtype,
                    outer_dims=rank,
                )
            except TypeError:
                res = seigh(
                    op,
                    op_device=device,
                    op_dtype=dtype,
                    outer_dims=rank,
                    inner_dims=inner_rank,
                    recovery_type="nystrom",
                )
            try:
                evals, evecs = res
            except (TypeError, ValueError):
                evals, evecs = res[1], res[0]
            S = evals[:, :rank] if evals.dim() == 2 else evals[:rank]
            U = evecs[:, :rank]
        S = S.clamp_min(0.0)
        return LowRankTerms(U=U, S=S, scalar=0.0)
    except Exception as e:  # noqa: BLE001
        import warnings

        warnings.warn(
            f"skerch unavailable ({type(e).__name__}: {e}); "
            "falling back to randomized_eigh.",
            stacklevel=2,
        )
        n, _ = mv.shape()
        U, S = randomized_eigh(
            mv,
            n=n,
            rank=rank,
        )
        return LowRankTerms(U=U, S=S, scalar=0.0)


def low_rank_curvature(
    model_fn: callable,
    w0: torch.Tensor,
    xs: list[torch.Tensor],
    rank: int = 100,
    loss_fn: str = "mse",
    factor: float | torch.Tensor = 1.0,
    method: str = "randomized",
    seed: int = 0,
) -> LowRankTerms:
    """One-shot low-rank GGN of the last FNO block over the data samples ``xs``."""
    mv = GGNMatvec(
        model_fn,
        w0,
        xs,
        loss_fn=loss_fn,
        factor=factor,
    )
    if method == "randomized":
        n, _ = mv.shape()
        U, S = randomized_eigh(mv, n=n, rank=rank, seed=seed)
        return LowRankTerms(U=U, S=S, scalar=0.0)
    if method == "skerch":
        return skerch_low_rank(mv, rank=rank)
    raise ValueError(f"unknown method: {method}")


def to_dense(mv: LinearOperator) -> torch.Tensor:
    """Materialize a (small) LinearOperator as a dense matrix."""
    n, _ = mv.shape()
    eye = torch.eye(n, dtype=mv.dtype, device=mv.device)
    return torch.stack([mv._matmul(eye[:, i]) for i in range(n)], dim=1)