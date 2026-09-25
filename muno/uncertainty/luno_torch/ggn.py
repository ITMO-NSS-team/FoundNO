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

def _print_mem(tag, device=None):
    """Печатает CUDA-память: allocated / reserved / свободно от total."""
    if not torch.cuda.is_available():
        return
    if device is None:
        device = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    total = torch.cuda.get_device_properties(device).total_memory / 1024**2
    print(f"[mem:{tag}] allocated={alloc:.1f} MiB, reserved={reserved:.1f} MiB, "
          f"free_from_total={total - alloc:.1f} MiB")
    
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

    Uses the modern ``skerch.algorithms.seigh`` API, which returns the pair
    ``(Lambda, Q)`` with ``A ~= Q diag(Lambda) Q^H``. Falls back to a
    randomized eigendecomposition only when skerch is not installed.
    """
    if inner_rank is None:
        inner_rank = rank
    try:
        from skerch.algorithms import seigh

        class TorchOp:
            def __init__(self, shape):
                self.shape = shape
                self.dtype = dtype
                self.device = torch.device(device)

            def __matmul__(self, x):
                y = mv._matmul(x.to(device=mv.device, dtype=mv.dtype))
                return y.to(dtype=self.dtype, device=self.device)

            def __rmatmul__(self, x):
                # Hermitian operator: x @ A == (A @ x.H).H
                y = mv._matmul(x.conj().T.to(device=mv.device, dtype=mv.dtype))
                return y.conj().T.to(dtype=self.dtype, device=self.device)

        op = TorchOp(mv.shape())
        lam, qq = seigh(
            op,
            lop_device=device,
            lop_dtype=dtype,
            outer_dims=rank,
            recovery_type = "nystrom"
        )
        S = lam[:rank].clamp_min(0.0).to(dtype=dtype)
        U = qq[:, :rank].to(dtype=dtype, device=torch.device(device))
        return LowRankTerms(U=U, S=S, scalar=0.0)
    except ImportError as e:
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