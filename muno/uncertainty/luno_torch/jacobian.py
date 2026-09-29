from __future__ import annotations

import torch

from .lino_ops import (
    Diagonal,
    IsotropicScalingPlusSymmetricLowRank,
    LinearOperator,
    PositiveDiagonalPlusSymmetricLowRank,
    SymmetricLowRank,
)
from .progress import tqdm, write

def _print_mem(tag, device=None):
    """Печатает CUDA-память: allocated / reserved / свободно от total."""
    if not torch.cuda.is_available():
        return
    if device is None:
        device = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    total = torch.cuda.get_device_properties(device).total_memory / 1024**2
    write(f"[mem:{tag}] allocated={alloc:.1f} MiB, reserved={reserved:.1f} MiB, "
          f"free_from_total={total - alloc:.1f} MiB")
    
class LastFNOBlockWeightJacobian(LinearOperator):
    """Linearized map w -> f(x; w) for the last FNO block weights.

    Shape ``(m, d)`` with ``d = 2*|R| + |W| + |b|`` and ``m`` the flattened model
    output size. J itself is never materialized; both applications use reverse-mode:
    Jv (matvec) is assembled from chunked reverse passes over the output basis and
    J^T u (transpose) uses ``torch.func.vjp``. Forward-mode AD was avoided because
    PyTorch's forward gradients are numerically wrong on this model's graph (they
    disagree with finite differences on the block bias columns).
    """

    def __init__(
        self,
        model_fn: callable,
        x: torch.Tensor,
        w0: torch.Tensor,
        num_output_channels: int | None = None,
        output_grid_shape: tuple[int, ...] | None = None,
        vjp_chunk: int = 64,
        vjp_refresh_every: int = 50,
    ):
        self._model_fn = model_fn
        self._x = x
        self._w0 = w0
        self._num_output_channels = num_output_channels
        self._output_grid_shape = output_grid_shape
        self._vjp_chunk = vjp_chunk
        self._vjp_refresh_every = vjp_refresh_every
        self._diag_JJT_cache = None

        # Ленивая сборка: forward/VJP строятся только при первом использовании,
        # а после — освобождаются (_release_vjp). Это JAX-подобная семантика
        # пересчёта: граф одного сэмпла не удерживается в памяти между матвеками,
        # иначе max_num_samples * (размер графа) упирается в CUDA-память.
        self._flat_fn = None
        self._fx = None
        self._vjp_fn = None
        self._vjp_rows = None
        self._vjp_cols = None

        super().__init__()

    def _ensure_vjp(self):
        """Строит forward и VJP-примитивы лениво (no-op, если уже построены)."""
        if self._vjp_fn is not None:
            return self
        self._flat_fn = lambda w: self._model_fn(self._x, w).reshape(-1)
        self._fx = self._flat_fn(self._w0)
        self._vjp_fn = torch.func.vjp(self._flat_fn, self._w0)[1]
        self._vjp_rows = torch.vmap(
            lambda c: self._vjp_fn(c, create_graph=False)[0], in_dims=0
        )
        self._vjp_cols = torch.vmap(
            lambda c: self._vjp_fn(c, create_graph=False)[0], in_dims=-1, out_dims=-1
        )
        return self

    def _release_vjp(self):
        """Освобождает удержанный forward-граф; следующий вызов пересчитает его."""
        self._fx = None
        self._vjp_fn = None
        self._vjp_rows = None
        self._vjp_cols = None
        torch.cuda.empty_cache()
        return self

    def _rebuild_vjp(self):
        """Пересоздаёт forward-граф и VJP-примитивы (сбрасывает удержанную память)."""
        return self._release_vjp()._ensure_vjp()

    @property
    def fx(self) -> torch.Tensor:
        return self._ensure_vjp()._fx

    def shape(self) -> tuple[int, int]:
        d = self._w0.numel()
        m = self._ensure_vjp()._fx.numel()
        return m, d

    def _chunked_rows(self):
        """Yield (slice, rows) with rows = J[slice, :] built via reverse-mode."""
        self._ensure_vjp()
        m = self._fx.numel()
        nchunks_total = (m + self._vjp_chunk - 1) // self._vjp_chunk
        bar = tqdm(
            total=nchunks_total, desc="чанки", unit="chunk", leave=False, position=1
        )
        nchunks = 0
        try:
            for c0 in range(0, m, self._vjp_chunk):
                c1 = min(c0 + self._vjp_chunk, m)
                cnt = c1 - c0
                cot = torch.zeros(cnt, m, dtype=self._w0.dtype, device=self._w0.device)
                if cnt == m:
                    cot = torch.eye(m, dtype=self._w0.dtype, device=self._w0.device)
                else:
                    cot[torch.arange(cnt), torch.arange(c0, c1)] = 1.0
                nchunks += 1
                bar.update(1)
                if nchunks % self._vjp_refresh_every == 0:
                    self._rebuild_vjp()
                yield slice(c0, c1), self._vjp_rows(cot)
                #_print_mem(f"_chunked_rows {c0}", self._w0.device)
        finally:
            bar.close()

    def _matmul(self, weights: torch.Tensor) -> torch.Tensor:
        """J w or J @ W (W shaped (d, k)) via chunked reverse passes."""
        weights = weights.to(self._w0)
        self._ensure_vjp()
        m = self._fx.numel()
        if weights.dim() == 1:
            out = torch.empty(m, dtype=self._w0.dtype, device=self._w0.device)
            for sl, rows in self._chunked_rows():
                out[sl] = rows @ weights
                del rows; torch.cuda.empty_cache()
            return out
        k = weights.shape[-1]
        out = torch.empty(
            m, k, dtype=self._w0.dtype, device=self._w0.device
        )
        for sl, rows in self._chunked_rows():
            out[sl] = rows @ weights
            del rows; torch.cuda.empty_cache()
        return out

    def transpose(self) -> "LastFNOBlockTransposeWeightJacobian":
        return LastFNOBlockTransposeWeightJacobian(self)

    def diag_JJT(self) -> torch.Tensor:
        """Row-wise squared norms of J, i.e. diag(J J^T)."""
        if self._diag_JJT_cache is not None:
            return self._diag_JJT_cache
        norms = []
        for sl, rows in self._chunked_rows():
            norms.append((rows**2).sum(dim=-1))
        result = torch.cat(norms)
        self._diag_JJT_cache = result
        return result

    def diag_JJT_times(self, diag: torch.Tensor) -> torch.Tensor:
        """Row-wise ``diag(J D J^T)`` for a weight-space diagonal D."""
        diag = diag.to(self._w0)
        self._ensure_vjp()
        out = torch.empty(self._fx.numel(), dtype=self._w0.dtype, device=self._w0.device)
        for sl, rows in self._chunked_rows():
            out[sl] = (rows**2) @ diag
        return out


class LastFNOBlockTransposeWeightJacobian(LinearOperator):
    def __init__(self, jacobian: LastFNOBlockWeightJacobian):
        self._jacobian = jacobian

    def shape(self) -> tuple[int, int]:
        m, d = self._jacobian.shape()
        return d, m

    def _matmul(self, outputs: torch.Tensor) -> torch.Tensor:
        self._jacobian._ensure_vjp()
        if outputs.dim() == 1:
            return self._jacobian._vjp_fn(outputs.to(self._jacobian._w0), create_graph=False)[0]
        return self._jacobian._vjp_cols(outputs.to(self._jacobian._w0))

    def transpose(self) -> LastFNOBlockWeightJacobian:
        return self._jacobian


def var_of_congruence(
    J: LastFNOBlockWeightJacobian, Sigma: LinearOperator
) -> torch.Tensor:
    """diag(J Sigma J^T) as in luno._linox, split over Sigma's operator_list.

    Uses ``diag(J J^T)`` (reverse-mode over output basis) for the diagonal part and
    ``(J U)^2 S`` (forward-mode over the rank basis) for the low-rank part.
    """
    if isinstance(Sigma, IsotropicScalingPlusSymmetricLowRank):
        scalar_part = Sigma.scalar * J.diag_JJT()
        JU = J._matmul(Sigma.U)
        low_rank_part = torch.sum(JU**2 * Sigma.S, dim=-1)
        return scalar_part + low_rank_part
    if isinstance(Sigma, PositiveDiagonalPlusSymmetricLowRank):
        diag_part = J.diag_JJT_times(Sigma.diagonal.diag)
        JU = J._matmul(Sigma.low_rank.U)
        low_rank_part = Sigma.low_rank_scale * torch.sum(
            JU**2 * Sigma.low_rank.S, dim=-1
        )
        return diag_part + low_rank_part
    if isinstance(Sigma, Diagonal):
        return Sigma.diag * J.diag_JJT()
    if isinstance(Sigma, SymmetricLowRank):
        JU = J._matmul(Sigma.U)
        return torch.sum(JU**2 * Sigma.S, dim=-1)
    raise NotImplementedError(f"var_of_congruence not implemented for {type(Sigma)}")