from __future__ import annotations

from dataclasses import dataclass

import torch

from .jacobian import LastFNOBlockWeightJacobian
from .lino_ops import LinearOperator
from .progress import tqdm

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
        affine=None,
        mode: str = "forward",
        fwd_batch: int = 4,
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
                affine=affine,
                mode=mode,
                fwd_batch=fwd_batch,
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

    def _accum(self, vv: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Суммирует по сэмплам ``sum_s J_s^T (2 J_s v)`` для блока направлений ``vv``.

        Пишет результат в переданный ``out`` (взгляд в буфер вызывающего кода).
        Так одновременно живут только ``vv``, один ``Jv``, один результат VJP и
        ``out`` -- без ``torch.zeros_like`` на весь блок и без ``2.0 * Jv``.
        При d = 27.6M каждый лишний полноразмерный буфер стоит ~0.1 ГиБ на
        столбец, что при rank = 50 и есть разница между OOM и запуском.
        """
        out.zero_()
        bar = tqdm(
            total=len(self._jacobians), desc="[GGN] сэмплы", unit="smp",
            leave=False, position=0,
        )
        for jac in self._jacobians:
            # Порядок важен для пика памяти: сначала Jv выбранным прямым режимом
            # (обратный граф для этого не нужен), затем -- граф только ради J^T.
            # После сэмпла всё освобождается: в памяти живёт не более одного
            # удержанного forward-графа за матвек (иначе max_num_samples графов ~ OOM).
            Jv = jac._matmul(vv)
            jac._ensure_vjp()
            gv = jac._vjp_cols(Jv)
            del Jv
            out.add_(gv)
            del gv
            jac._release_vjp()
            bar.update(1)
        bar.close()
        # Множитель MSE 2 и внешний factor применяются один раз к результату,
        # чтобы не материализовать отдельный тензор 2.0 * Jv на (d, fwd_batch).
        out *= 2.0 * self._factor
        return out

    def _matmul(self, v: torch.Tensor) -> torch.Tensor:
        v = v.to(self._w0)
        d = self._w0.numel()
        if v.dim() == 1:
            out = torch.empty(d, 1, dtype=self._w0.dtype, device=self._w0.device)
            self._accum(v.unsqueeze(-1), out)
            return out.squeeze(-1)
        # Скетч low-rank GGN -- это блок (d, q) направлений, и при d = 27.6M
        # каждый буфер (d, k) стоит ~0.1 ГиБ на столбец. Результат выделяется
        # один раз целиком, а чанки пишутся в его столбцы: список чанков с
        # последующим ``torch.cat`` держал бы одновременно и чанки (d, q), и
        # результат (d, q), то есть удваивал пик на ~5 ГиБ при rank = 50.
        k = v.shape[-1]
        out = torch.empty(d, k, dtype=self._w0.dtype, device=self._w0.device)
        step = max(1, int(self._jacobians[0]._fwd_batch))
        for c0 in range(0, k, step):
            self._accum(v[..., c0:c0 + step], out[..., c0:c0 + step])
        return out


def _orthonormalize(Y: torch.Tensor) -> torch.Tensor:
    """Ортонормировать столбцы ``Y`` с пиком памяти 2x размера ``Y``.

    ``torch.linalg.qr`` на матрице ``(n, q)`` держит пик ~3x (вход + выход +
    внутренний workspace), что при ``q = rank + oversample`` не помещается в
    16 ГиБ начиная с rank ~50. Здесь используется Cholesky-QR: скалярные
    нормы столбцов и Gram считаются в float64 и живут только как матрицы
    ``(q, q)``, а ортогонализация -- один GEMM ``Y @ M`` (пик 2x).

    Базис span(Q) совпадает с базисом Householder-QR с точностью до вращения
    внутри подпространства, поэтому на результат randomized_eigh это не влияет.
    При вырожденном ``Y`` (Cholesky не сходится) используется обычный QR.
    """
    q = Y.shape[1]
    if q == 0:
        return Y
    nrm = Y.norm(dim=0).clamp_min(1e-30).to(torch.float64)
    G = (Y.T @ Y).to(torch.float64)
    G = 0.5 * (G + G.T)
    G = G / nrm.unsqueeze(0) / nrm.unsqueeze(1)
    try:
        R = torch.linalg.cholesky(G, upper=True)
        M = torch.linalg.solve_triangular(
            R, torch.eye(q, device=Y.device, dtype=G.dtype), upper=True)
        M = (M / nrm.unsqueeze(1)).to(Y.dtype)
    except Exception:
        del G, nrm
        return torch.linalg.qr(Y)[0]
    del G, R, nrm
    return Y @ M


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
        del Q
        Q = _orthonormalize(Y)
        del Y
    Y = mv._matmul(Q)
    del Q
    Q = _orthonormalize(Y)
    del Y

    T = mv._matmul(Q)
    B = Q.T @ T
    del T
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
    sketch_blocksize: int | None = None,
) -> LowRankTerms:
    """Low-rank estimate of a symmetric positive matvec using skerch if available.

    ВНИМАНИЕ: ``sketch_blocksize`` -> ``meas_blocksize`` НЕ ограничивает главный
    буфер. ``seigh`` всегда предварительно материализует ``ro_sketch`` формы
    ``(d, outer_dims)`` целиком, а блок ограничивает только генерацию шума. Пик
    памяти -- ``3 * rank`` полноразмерных столбцов (``ro_sketch`` + QR-копия +
    ``lop @ Q``), поэтому при rank >= 50 на GPU ~16 ГиБ скетч не помещается ни
    при каком ``sketch_blocksize``. Для rank ~50 используйте
    :func:`low_rank_ggn` с ``method="randomized"``.
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
        kw = {}
        if sketch_blocksize is not None:
            kw["meas_blocksize"] = sketch_blocksize
        lam, qq = seigh(
            op,
            lop_device=device,
            lop_dtype=dtype,
            outer_dims=rank,
            recovery_type="nystrom",
            **kw,
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


def plan_low_rank_ggn(
    method: str,
    shape: int | tuple[int, int],
    rank: int,
    oversample: int = 10,
    fwd_batch: int = 4,
    vjp_chunk: int = 16,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    mem_fraction: float = 1.0,
    strict: bool = False,
) -> dict:
    """Оценка пиковой памяти GGN-скетча и максимального допустимого rank.

    Один столбец ``(d,)`` занимает ``d * itemsize`` байт. Для ``randomized``
    пик -- ``2q + fwd_batch`` столбцов (``q = rank + oversample``): скетч плюс
    результат matvec/ортогонализации. Для ``skerch`` -- ``3q + vjp_chunk``:
    ``ro_sketch`` + QR-копия + ``lop @ Q`` + блок шума.

    Бросает ``RuntimeError`` только при ``strict=True``; по умолчанию план --
    информационный, чтобы не блокировать расчёт на GPU, где часть памяти уже
    занята соседними процессами (оценка считается от фактически свободной).
    """
    d = int(shape[0]) if isinstance(shape, (tuple, list)) else int(shape)
    if method not in ("randomized", "skerch"):
        raise ValueError(f"unknown method: {method}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())

    col_bytes = d * torch.empty((), dtype=dtype).element_size()
    if method == "randomized":
        q = min(rank + oversample, d)
        cols = 2 * q + fwd_batch
    else:
        q = rank
        cols = 3 * q + vjp_chunk
    peak_bytes = cols * col_bytes

    info = {
        "method": method, "rank": rank, "oversample": oversample, "q": q,
        "cols": cols, "col_bytes": col_bytes, "peak_bytes": peak_bytes,
        "budget_bytes": None, "max_rank": None,
    }

    if dev.type != "cuda":
        info["message"] = (
            f"[GGN-план] method={method} rank={rank} q={q}: оценка пика "
            f"{peak_bytes / 1024 ** 3:.2f} GiB (план по VRAM не применяется на CPU)."
        )
        return info

    free, _total = torch.cuda.mem_get_info(dev)
    budget = mem_fraction * free
    per_rank = 2 if method == "randomized" else 3
    fixed = fwd_batch if method == "randomized" else vjp_chunk
    max_rank = int((budget - fixed * col_bytes) // (per_rank * col_bytes))
    if method == "randomized":
        max_rank -= oversample
    info["budget_bytes"] = budget
    info["max_rank"] = max_rank

    head = (f"[GGN-план] method={method} rank={rank} oversample={oversample} q={q}: "
            f"пик ~{peak_bytes / 1024 ** 3:.2f} GiB, бюджет "
            f"{budget / 1024 ** 3:.2f} GiB (свободно {free / 1024 ** 3:.2f} GiB, "
            f"один столбец {col_bytes / 1024 ** 3:.4f} GiB)")
    if peak_bytes <= budget:
        info["message"] = head + f" -- укладывается (макс. rank здесь ~{max_rank})."
        return info

    if method == "skerch":
        msg = (
            head + ". ВНИМАНИЕ: skerch держит 3 полноразмерных буфера (ro_sketch + "
            "QR-копия + lop@Q), blocksize на это не влияет. Надежнее "
            "--ggn-method randomized."
        )
    else:
        msg = (
            head + ". ВНИМАНИЕ: оценка может не поместиться (пик обычно выше "
            "оценки на ~1 ГиБ из-за обратного графа VJP). Если вылезет CUDA OOM -- "
            "снизьте --max-rank или --ggn-oversample, уменьшите --fwd-batch."
        )
    info["message"] = msg
    if strict:
        raise RuntimeError(msg)
    return info


def low_rank_ggn(
    mv: LinearOperator,
    rank: int = 50,
    oversample: int = 10,
    method: str = "randomized",
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    fwd_batch: int = 4,
    vjp_chunk: int = 16,
    power_iter: int = 2,
    seed: int = 0,
    mem_fraction: float = 1.0,
    plan: bool = True,
    strict: bool = False,
) -> LowRankTerms:
    """Низкоранговая GGN-аппроксимация для ``LinearOperator`` ``mv``.

    ``randomized`` (по умолчанию) держит пик ``2q + fwd_batch`` полноразмерных
    столбцов и доходит до rank ~50 на GPU ~16 ГиБ. ``skerch`` (``seigh``)
    держит ``3q`` плюс блок шума и на rank >= 50 не помещается ни при каком
    размере блока, поэтому отвергается заранее с actionable-сообщением.
    """
    n, _ = mv.shape()
    device = mv.device if device is None else torch.device(device)
    dtype = mv.dtype if dtype is None else dtype

    if plan:
        print(plan_low_rank_ggn(
            method, mv.shape(), rank, oversample, fwd_batch, vjp_chunk,
            device=device, dtype=dtype, mem_fraction=mem_fraction,
            strict=strict,
        )["message"])

    if method == "randomized":
        U, S = randomized_eigh(
            mv, n=n, rank=rank, oversample=oversample,
            power_iter=power_iter, seed=seed,
        )
        return LowRankTerms(U=U, S=S, scalar=0.0)

    terms = skerch_low_rank(
        mv, rank=min(rank + oversample, n), inner_rank=rank,
        device=str(device), dtype=dtype,
    )
    if terms.U.shape[1] > rank:
        terms.U = terms.U[:, :rank].contiguous()
        terms.S = terms.S[:rank].contiguous()
    return terms


def low_rank_curvature(
    model_fn: callable,
    w0: torch.Tensor,
    xs: list[torch.Tensor],
    affine=None,
    mode: str = "forward",
    fwd_batch: int = 4,
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
        affine=affine,
        mode=mode,
        fwd_batch=fwd_batch,
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