from __future__ import annotations

import torch


class LinearOperator:
    """Minimal torch view of a real linear operator (no materialized matrix)."""

    def _matmul(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _rmatmul(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, LinearOperator):
            raise NotImplementedError
        return self._matmul(x)

    def transpose(self) -> "LinearOperator":
        raise NotImplementedError

    @property
    def T(self) -> "LinearOperator":
        return self.transpose()

    def shape(self) -> tuple[int, int]:
        raise NotImplementedError

    def todense(self) -> torch.Tensor:
        rows, cols = self.shape()
        eye = torch.eye(cols, dtype=self.dtype, device=self.device)
        return torch.stack([self._matmul(eye[:, i]) for i in range(cols)], dim=1)

    @property
    def dtype(self) -> torch.dtype:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        raise NotImplementedError


class Diagonal(LinearOperator):
    def __init__(self, diag: torch.Tensor):
        self._diag = diag

    @property
    def diag(self) -> torch.Tensor:
        return self._diag

    def _matmul(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return self._diag * x
        return self._diag.unsqueeze(-1) * x

    def transpose(self) -> "Diagonal":
        return self

    def shape(self) -> tuple[int, int]:
        n = self._diag.shape[0]
        return n, n

    @property
    def dtype(self) -> torch.dtype:
        return self._diag.dtype

    @property
    def device(self) -> torch.device:
        return self._diag.device

    def todense(self) -> torch.Tensor:
        return torch.diag(self._diag)


class ScaledLinearOperator(LinearOperator):
    def __init__(self, scalar: torch.Tensor, operator: LinearOperator):
        self._scalar = scalar
        self._operator = operator

    @property
    def scalar(self) -> torch.Tensor:
        return self._scalar

    @property
    def operator(self) -> LinearOperator:
        return self._operator

    def _matmul(self, x: torch.Tensor) -> torch.Tensor:
        return self._scalar * self._operator._matmul(x)

    def transpose(self) -> "ScaledLinearOperator":
        return ScaledLinearOperator(self._scalar, self._operator.transpose())

    def shape(self) -> tuple[int, int]:
        return self._operator.shape()

    @property
    def dtype(self) -> torch.dtype:
        return self._operator.dtype

    @property
    def device(self) -> torch.device:
        return self._operator.device


class SymmetricLowRank(LinearOperator):
    r"""A = U diag(S) U^T."""

    def __init__(self, U: torch.Tensor, S: torch.Tensor | None = None):
        self._U = U
        if S is None:
            S = torch.ones(U.shape[-1], dtype=U.dtype, device=U.device)
        self._S = S

    @property
    def U(self) -> torch.Tensor:
        return self._U

    @property
    def S(self) -> torch.Tensor:
        return self._S

    def _matmul(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return self._U @ (self._S * (self._U.T @ x))
        proj = self._U.T @ x
        return self._U @ (self._S.unsqueeze(-1) * proj)

    def transpose(self) -> "SymmetricLowRank":
        return self

    def shape(self) -> tuple[int, int]:
        n = self._U.shape[-2]
        return n, n

    @property
    def dtype(self) -> torch.dtype:
        return self._U.dtype

    @property
    def device(self) -> torch.device:
        return self._U.device

    def todense(self) -> torch.Tensor:
        return self._U @ torch.diag(self._S) @ self._U.T


class IsotropicScalingPlusSymmetricLowRank(LinearOperator):
    r"""A = scalar * I + U diag(S) U^T."""

    def __init__(self, scalar: torch.Tensor, U: torch.Tensor, S: torch.Tensor):
        self._scalar = scalar
        self._U = U
        self._S = S

    @property
    def scalar(self) -> torch.Tensor:
        return self._scalar

    @property
    def U(self) -> torch.Tensor:
        return self._U

    @property
    def S(self) -> torch.Tensor:
        return self._S

    @property
    def operator_list(self):
        ident = Diagonal(
            torch.ones(self._U.shape[-2], dtype=self._U.dtype, device=self._U.device)
        )
        return [
            ScaledLinearOperator(self._scalar, ident),
            SymmetricLowRank(self._U, self._S),
        ]

    def _matmul(self, x: torch.Tensor) -> torch.Tensor:
        return sum(s._matmul(x) for s in self.operator_list)

    def transpose(self) -> "IsotropicScalingPlusSymmetricLowRank":
        return self

    def shape(self) -> tuple[int, int]:
        n = self._U.shape[-2]
        return n, n

    @property
    def dtype(self) -> torch.dtype:
        return self._U.dtype

    @property
    def device(self) -> torch.device:
        return self._U.device

    def todense(self) -> torch.Tensor:
        return self._scalar * torch.eye(
            self.shape()[0], dtype=self.dtype, device=self.device
        ) + self._U @ torch.diag(self._S) @ self._U.T


class PositiveDiagonalPlusSymmetricLowRank(LinearOperator):
    r"""A = D + low_rank_scale * U diag(S) U^T."""

    def __init__(
        self,
        diagonal: Diagonal,
        low_rank: SymmetricLowRank,
        low_rank_scale: torch.Tensor | float = 1.0,
    ):
        self._diagonal = diagonal
        self._low_rank = low_rank
        if not isinstance(low_rank_scale, torch.Tensor):
            low_rank_scale = torch.tensor(
                low_rank_scale, dtype=diagonal.dtype, device=diagonal.device
            )
        self._low_rank_scale = low_rank_scale

    @property
    def diagonal(self) -> Diagonal:
        return self._diagonal

    @property
    def low_rank(self) -> SymmetricLowRank:
        return self._low_rank

    @property
    def low_rank_scale(self) -> torch.Tensor:
        return self._low_rank_scale

    @property
    def operator_list(self):
        return [
            self._diagonal,
            ScaledLinearOperator(self._low_rank_scale, self._low_rank),
        ]

    def _matmul(self, x: torch.Tensor) -> torch.Tensor:
        return sum(s._matmul(x) for s in self.operator_list)

    def transpose(self) -> "PositiveDiagonalPlusSymmetricLowRank":
        return self

    def shape(self) -> tuple[int, int]:
        return self._diagonal.shape()

    @property
    def dtype(self) -> torch.dtype:
        return self._diagonal.dtype

    @property
    def device(self) -> torch.device:
        return self._diagonal.device

    def _id_plus_low_rank(self) -> IsotropicScalingPlusSymmetricLowRank:
        U, S = self._low_rank.U, self._low_rank.S
        d = self._diagonal.diag
        M = (U * torch.sqrt(S)[None, :]) / torch.sqrt(d[:, None])
        U2, sqrt_S2, _ = torch.linalg.svd(M, full_matrices=False)
        return IsotropicScalingPlusSymmetricLowRank(
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
            U2,
            self._low_rank_scale * sqrt_S2**2,
        )


class ProductLinearOperator(LinearOperator):
    def __init__(self, factors: list[LinearOperator]):
        self._factors = factors

    @property
    def factors(self):
        return self._factors

    def _matmul(self, x: torch.Tensor) -> torch.Tensor:
        for f in reversed(self._factors):
            x = f._matmul(x)
        return x

    def transpose(self) -> "ProductLinearOperator":
        return ProductLinearOperator([f.transpose() for f in reversed(self._factors)])

    def shape(self) -> tuple[int, int]:
        m, _ = self._factors[0].shape()
        _, n = self._factors[-1].shape()
        return m, n

    @property
    def dtype(self) -> torch.dtype:
        return self._factors[0].dtype

    @property
    def device(self) -> torch.device:
        return self._factors[0].device


class CongruenceTransform(LinearOperator):
    r"""A B A^T."""

    def __init__(self, A: LinearOperator, B: LinearOperator):
        self._A = A
        self._B = B

    @property
    def A(self) -> LinearOperator:
        return self._A

    @property
    def B(self) -> LinearOperator:
        return self._B

    def _matmul(self, x: torch.Tensor) -> torch.Tensor:
        return self._A._matmul(self._B._matmul(self._A.T._matmul(x)))

    def transpose(self) -> "CongruenceTransform":
        return CongruenceTransform(self._A.T, self._B.T)

    def shape(self) -> tuple[int, int]:
        m, _ = self._A.shape()
        _, n = self._A.T.shape()
        return m, n

    @property
    def dtype(self) -> torch.dtype:
        return self._A.dtype

    @property
    def device(self) -> torch.device:
        return self._A.device


class CircularlySymmetricDiagonal(Diagonal):
    def __init__(self, R_real: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None):
        self._R_real = R_real
        self._W = W
        self._b = b
        diag = (
            self._R_real.reshape(-1),
            self._R_real.reshape(-1),
            self._W.reshape(-1),
        )
        if self._b is not None:
            diag = diag + (self._b.reshape(-1),)
        super().__init__(torch.cat(diag, dim=0))

    @property
    def R_real(self) -> torch.Tensor:
        return self._R_real

    @property
    def W(self) -> torch.Tensor:
        return self._W

    @property
    def b(self) -> torch.Tensor | None:
        return self._b


def diagonal(a: LinearOperator) -> torch.Tensor:
    if isinstance(a, Diagonal):
        return a.diag.detach()
    if isinstance(a, SymmetricLowRank):
        return torch.sum(a.U**2 * a.S, dim=-1).detach()
    if isinstance(a, ScaledLinearOperator):
        return a.scalar * diagonal(a.operator)
    if isinstance(a, IsotropicScalingPlusSymmetricLowRank):
        return (
            a.scalar
            + torch.sum(a.U**2 * a.S, dim=-1)
        ).detach()
    if isinstance(a, PositiveDiagonalPlusSymmetricLowRank):
        return (
            diagonal(a.diagonal)
            + a.low_rank_scale * torch.sum(a.low_rank.U**2 * a.low_rank.S, dim=-1)
        ).detach()
    raise NotImplementedError(f"diagonal not implemented for {type(a)}")


def congruence_transform(A: LinearOperator, B: LinearOperator) -> LinearOperator:
    if isinstance(B, ScaledLinearOperator):
        return B.scalar * congruence_transform(A, B.operator)
    if isinstance(B, SymmetricLowRank):
        return SymmetricLowRank(A._matmul(B.U), B.S)
    return CongruenceTransform(A, B)


def linverse(a: LinearOperator) -> LinearOperator:
    if isinstance(a, Diagonal):
        return Diagonal(1.0 / a.diag)
    if isinstance(a, IsotropicScalingPlusSymmetricLowRank):
        scalar_inv = 1.0 / a.scalar
        return IsotropicScalingPlusSymmetricLowRank(
            scalar_inv,
            a.U,
            -a.S / (a.scalar * (a.S + a.scalar)),
        )
    if isinstance(a, PositiveDiagonalPlusSymmetricLowRank):
        D_inv = linverse(a.diagonal)
        U, S = a.low_rank.U, a.low_rank.S
        alpha = a.low_rank_scale
        D_inv_U = D_inv.diag.unsqueeze(-1) * U
        schur = (
            alpha * (U.T @ D_inv_U)
            + torch.diag(1.0 / S)
        )
        schur_eigvals, schur_eigvecs = torch.linalg.eigh(schur)
        M = D_inv_U @ schur_eigvecs / torch.sqrt(schur_eigvals)
        U2, sqrt_S2, _ = torch.linalg.svd(M, full_matrices=False)
        return PositiveDiagonalPlusSymmetricLowRank(
            D_inv,
            SymmetricLowRank(U2, sqrt_S2**2),
            low_rank_scale=-alpha,
        )
    raise NotImplementedError(f"linverse not implemented for {type(a)}")


def lsqrt(a: LinearOperator) -> LinearOperator:
    if isinstance(a, Diagonal):
        return Diagonal(torch.sqrt(a.diag))
    if isinstance(a, CircularlySymmetricDiagonal):
        R_sqrt = torch.sqrt(a.R_real)
        W_sqrt = torch.sqrt(a.W)
        b_sqrt = None if a.b is None else torch.sqrt(a.b)
        return CircularlySymmetricDiagonal(R_sqrt, W_sqrt, b_sqrt)
    if isinstance(a, IsotropicScalingPlusSymmetricLowRank):
        sc = torch.as_tensor(a.scalar, dtype=a.U.dtype, device=a.U.device)
        scalar_sqrt = torch.sqrt(sc)
        return IsotropicScalingPlusSymmetricLowRank(
            scalar_sqrt,
            a.U,
            scalar_sqrt * (torch.sqrt(a.S / sc + 1.0) - 1.0),
        )
    if isinstance(a, PositiveDiagonalPlusSymmetricLowRank):
        return ProductLinearOperator(
            [lsqrt(a.diagonal), lsqrt(a._id_plus_low_rank())]
        )
    raise NotImplementedError(f"lsqrt not implemented for {type(a)}")


def stats(
    a: LinearOperator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Numeric eigendecomposition of a small linear operator."""
    dense = a.todense()
    eigvals, eigvecs = torch.linalg.eigh((dense + dense.T) / 2.0)
    return eigvals, eigvecs