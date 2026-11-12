from __future__ import annotations

from dataclasses import dataclass

import torch

from .jacobian import LastFNOBlockWeightJacobian, LiftingWeightJacobian
from .lino_ops import LinearOperator
import time

from .progress import tqdm, write

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

    Subclasses change only *which* parameters are linearized by overriding
    :attr:`jacobian_cls` (e.g. :class:`LiftingGGNMatvec` for the lifting head).
    """

    jacobian_cls = LastFNOBlockWeightJacobian

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
        sample_batch_size: int = 1,
        bind: callable | None = None,
    ):
        # Подпись внутреннего прогресс-бара; randomized_eigh пишет сюда номер
        # прохода ("[GGN] проход 2/3"), чтобы было видно, где идёт расчёт.
        self.progress_label = "[GGN]"
        # ``bind(x) -> (model_fn, state)``: необязательная пер-сэмпловая привязка.
        # Для last-block это срез замороженного префикса (лифтинг + блоки до
        # последнего считаются один раз на сэмпл, см. affine.LastBlockSuffix).
        self._bind_fn = bind
        self._model_fn = model_fn
        self._w0 = w0
        self._affine = affine
        self._mode = mode
        self._fwd_batch = int(fwd_batch)
        self._num_output_channels = num_output_channels
        self._vjp_chunk = vjp_chunk
        self._sample_batch_size = max(1, int(sample_batch_size))
        self._loss_fn = loss_fn
        if isinstance(factor, torch.Tensor):
            self._factor = factor.to(dtype=w0.dtype, device=w0.device)
        else:
            self._factor = torch.tensor(factor, dtype=w0.dtype, device=w0.device)
        # Храним сэмплы на CPU, чтобы не занимать всю GPU память под все входы
        self._xs_cpu = []
        for x in xs:
            if torch.is_tensor(x):
                self._xs_cpu.append(x.detach().to(device="cpu", dtype=w0.dtype))
            else:
                self._xs_cpu.append(x)
        self._xs = self._xs_cpu
        # Для обратной совместимости, если где-то используется
        self._jacobians = []

    def shape(self) -> tuple[int, int]:
        d = self._w0.numel()
        return d, d

    @property
    def dtype(self) -> torch.dtype:
        return self._w0.dtype

    @property
    def device(self) -> torch.device:
        return self._w0.device

    def _bind(self, x: torch.Tensor):
        """``(model_fn, x)`` для одного сэмпла; ``bind`` может срезать замороженный префикс."""
        if self._bind_fn is None:
            return self._model_fn, x
        return self._bind_fn(x)

    def _matmul(self, v: torch.Tensor) -> torch.Tensor:
        """``G v`` или ``G @ V`` (``V`` формы ``(d, k)``) за ОДИН проход по сэмплам.

        Сэмпл -- снаружи, чанки колонок -- внутри: на сэмпл строится один якобиан
        и один обратный граф, который переиспользуется для всех ``k`` колонок.
        Порядок "чанк снаружи, сэмплы внутри" строил бы граф заново на каждую
        пару (сэмпл, чанк), то есть ``k / fwd_batch`` раз на сэмпл.
        """
        v = v.to(self._w0)
        squeeze = v.dim() == 1
        V = v.unsqueeze(-1) if squeeze else v
        d = self._w0.numel()
        k = V.shape[-1]
        device = self._w0.device
        out = torch.zeros(d, k, dtype=self._w0.dtype, device=device)
        step = max(1, int(self._fwd_batch))
        n_smp = len(self._xs_cpu)
        n_chunks = -(-k // step)
        # Единица работы -- чанк колонок одного сэмпла (JVP + VJP на fwd_batch
        # колонок). Так бар двигается и внутри сэмпла, и видно, сколько колонок
        # ранга уже посчитано.
        bar = tqdm(
            total=n_smp * n_chunks, desc=f"{self.progress_label} колонки",
            unit="chunk", leave=False, position=1,
        )
        try:
            for i_smp, x_c in enumerate(self._xs_cpu):
                x_g = x_c.to(device=device) if torch.is_tensor(x_c) else x_c
                fn, state = self._bind(x_g)
                jac = self.jacobian_cls(
                    fn,
                    state,
                    self._w0,
                    affine=self._affine,
                    mode=self._mode,
                    fwd_batch=self._fwd_batch,
                    num_output_channels=self._num_output_channels,
                    vjp_chunk=self._vjp_chunk,
                )
                try:
                    jac._ensure_vjp()
                    for c0 in range(0, k, step):
                        c1 = min(c0 + step, k)
                        Jv = jac._matmul(V[:, c0:c1])
                        out[:, c0:c1].add_(jac._vjp_cols(Jv))
                        del Jv
                        bar.set_postfix_str(
                            f"сэмпл {i_smp + 1}/{n_smp}, колонки {c1}/{k}",
                            refresh=False,
                        )
                        bar.update(1)
                finally:
                    jac._release_vjp(collect=False)
                del jac, fn, state, x_g
        finally:
            bar.close()
        out *= 2.0 * self._factor
        return out.squeeze(-1) if squeeze else out


class LiftingGGNMatvec(GGNMatvec):
    """GGN matvec over the lifting-adapter weights.

    Identical to :class:`GGNMatvec`, but ``J`` is taken with respect to all
    parameters of the lifting module while the core and projection are frozen
    (see :class:`luno_torch.adapter.TorchLiftingWrapper`).
    """

    jacobian_cls = LiftingWeightJacobian


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


def _nystrom_eigh(
    Q: torch.Tensor, Y: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Собственное разложение ранга ``rank`` по скетчу ``Y = A Q`` (Nystrom).

    Для PSD-оператора ``A`` и ортонормированного ``Q`` приближение
    ``A ~= Y (Q^T Y)^+ Y^T`` не требует ещё одного умножения на ``A``: проекция
    Рэлея-Ритца ``Q^T A Q`` стоила бы отдельного прохода по всем сэмплам, а
    здесь ``Q^T Y`` -- это уже посчитанное ``Q^T A Q``.

    Как в Tropp et al. (2017, алг. 3): сдвиг ``nu`` стабилизирует обращение
    ``Q^T Y`` и вычитается из собственных значений в конце. Все разложения --
    над матрицами ``(q, q)`` в float64; полноразмерные операции -- два Gram'а и
    один GEMM ``Y @ M``. ``Q`` и ``Y`` потребляются (``Y`` сдвигается на месте).
    """
    q = Y.shape[1]
    f64 = torch.float64
    nu = float(torch.finfo(Y.dtype).eps) * float(torch.linalg.matrix_norm(Y))
    Y.add_(Q, alpha=nu)
    B = (Q.T @ Y).to(f64)
    del Q
    B = 0.5 * (B + B.T)
    # Псевдообратный корень B через eigh: Cholesky падает, как только ранг A
    # меньше q (хвост спектра B уходит в шум округления).
    lam_b, V_b = torch.linalg.eigh(B)
    keep = lam_b > lam_b.max().clamp_min(0.0) * q * torch.finfo(Y.dtype).eps
    M1 = V_b[:, keep] / torch.sqrt(lam_b[keep])
    W = (Y.T @ Y).to(f64)
    W = M1.T @ (0.5 * (W + W.T)) @ M1
    sig2, V = torch.linalg.eigh(0.5 * (W + W.T))
    order = torch.argsort(sig2, descending=True)[:rank]
    sig2 = sig2[order].clamp_min(torch.finfo(f64).tiny)
    M = (M1 @ V[:, order]) / torch.sqrt(sig2)
    U = Y @ M.to(Y.dtype)
    S = (sig2 - nu).clamp_min(0.0).to(Y.dtype)
    if U.shape[1] < rank:
        # Ранг A меньше запрошенного: дополняем нулевыми направлениями, чтобы
        # форма (d, rank) не зависела от данных (S там равно нулю).
        pad = rank - U.shape[1]
        U = torch.cat([U, U.new_zeros(U.shape[0], pad)], dim=1)
        S = torch.cat([S, S.new_zeros(pad)])
    return U, S


def _nystrom_values(Q: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Собственные значения Nystrom-приближения ``Y (Q^T Y)^+ Y^T`` по убыванию.

    Та же арифметика, что в :func:`_nystrom_eigh` (сдвиг ``nu``, обрезка
    вырожденного хвоста ``Q^T Y``), но без полноразмерных буферов: только
    матрицы ``(q, q)``; ``Y`` не меняется.
    """
    q = Y.shape[1]
    f64 = torch.float64
    eps = torch.finfo(Y.dtype).eps
    nu = float(eps) * float(torch.linalg.matrix_norm(Y))
    QtY = (Q.T @ Y).to(f64)
    B = 0.5 * (QtY + QtY.T) + nu * torch.eye(q, dtype=f64, device=QtY.device)
    lam_b, V_b = torch.linalg.eigh(B)
    keep = lam_b > lam_b.max().clamp_min(0.0) * q * eps
    M1 = V_b[:, keep] / torch.sqrt(lam_b[keep])
    # (Y + nu Q)^T (Y + nu Q) = Y^T Y + nu (Q^T Y + Y^T Q) + nu^2 I
    G = (Y.T @ Y).to(f64)
    G = 0.5 * (G + G.T) + nu * (QtY + QtY.T) + nu ** 2 * torch.eye(q, dtype=f64, device=G.device)
    W = M1.T @ G @ M1
    sig2 = torch.linalg.eigvalsh(0.5 * (W + W.T)).flip(0)
    return (sig2 - nu).clamp_min(0.0)


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

    Стоит ``power_iter + 1`` умножений на ``A`` (проходов по данным): финальная
    проекция берётся из того же скетча в форме Nystrom, см.
    :func:`_nystrom_eigh`. ``A`` предполагается симметричным PSD (GGN).
    ``power_iter=2`` (по умолчанию) даёт подпространство ``A^3 Omega`` -- как
    прежняя схема Рэлея-Ритца, которой на это нужно было 4 прохода, а здесь 3.
    ``power_iter=1`` -- 2 прохода; ведущие, хорошо отделённые собственные пары
    остаются точными, но в плоском хвосте спектра ошибка значений растёт до
    процентов (см. tests/test_luno_fast.py).
    """
    q = min(rank + oversample, n)
    gen = None
    if mv.device.type == "cpu":
        gen = torch.Generator(device="cpu")
    else:
        gen = torch.Generator(device=mv.device)
    gen.manual_seed(seed)

    n_pass = power_iter + 1
    write(f"[GGN] скетч: rank={rank}, q=rank+oversample={q}, d={n:,}, "
          f"проходов по сэмплам={n_pass} (power_iter={power_iter} + Nystrom)")
    progress = _SketchProgress(mv, rank, n_pass)

    Q = _orthonormalize(
        torch.randn(n, q, dtype=mv.dtype, device=mv.device, generator=gen)
    )
    try:
        for _ in range(power_iter):
            Y = progress.matmul(Q)
            del Q
            Q = _orthonormalize(Y)
            del Y
        Y = progress.matmul(Q)
    finally:
        progress.close()
    return _nystrom_eigh(Q, Y, rank)


class _SketchProgress:
    """Прогресс скетча: бар по проходам + строка с оценкой спектра после каждого.

    После каждого прохода ``Y = A Q`` печатаются собственные значения
    Nystrom-приближения ``Y (Q^T Y)^+ Y^T`` -- ровно то, что вернёт
    :func:`_nystrom_eigh`, если остановиться на этом проходе (на последнем они
    совпадают с итоговым ``S``). Стоит один Gram ``(q, q)`` по ``Y``.
    Относительное изменение ведущих ``rank`` значений между проходами
    показывает, сошёлся ли скетч. Оценки Ритца ``eig(Q^T Y)`` здесь не годятся:
    на первом проходе ``Q`` -- случайное подпространство, и они занижены на
    порядки. Матвек ``mv`` -- любой ``LinearOperator``; подпись внутреннего
    бара выставляется, только если у него есть ``progress_label``.
    """

    def __init__(self, mv, rank: int, n_pass: int):
        self.mv, self.rank, self.n_pass = mv, rank, n_pass
        self.i = 0
        self.prev = None
        self.t0 = time.perf_counter()
        self.bar = tqdm(total=n_pass, desc="[GGN] проходы скетча", unit="pass",
                        leave=False, position=0)

    def matmul(self, Q: torch.Tensor) -> torch.Tensor:
        self.i += 1
        if hasattr(self.mv, "progress_label"):
            self.mv.progress_label = f"[GGN] проход {self.i}/{self.n_pass}"
        t = time.perf_counter()
        Y = self.mv._matmul(Q)
        dt = time.perf_counter() - t
        self.bar.update(1)
        self._log(Q, Y, dt)
        return Y

    def _log(self, Q: torch.Tensor, Y: torch.Tensor, dt: float) -> None:
        ev = _nystrom_values(Q, Y)[: self.rank].cpu()
        shown = ", ".join(f"{v:.3e}" for v in ev[:5].tolist())
        more = f" ... {ev[-1].item():.3e}" if ev.numel() > 5 else ""
        conv = ""
        if self.prev is not None and self.prev.numel() == ev.numel():
            rel = ((ev - self.prev).abs() / ev.clamp_min(1e-30)).max().item()
            conv = f"; max отн. изменение за проход={rel:.1e}"
        self.prev = ev
        elapsed = time.perf_counter() - self.t0
        left = (self.n_pass - self.i) * dt
        write(f"[GGN] проход {self.i}/{self.n_pass}: {dt:.0f} с "
              f"(всего {elapsed:.0f} с, осталось ~{left:.0f} с); "
              f"собств. значения GGN (Nystrom, {self.rank} шт.): "
              f"{shown}{more}{conv}")

    def close(self) -> None:
        self.bar.close()


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
    sample_batch_size: int = 1,
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
        sample_batch_size=sample_batch_size,
    )
    if method == "randomized":
        n, _ = mv.shape()
        U, S = randomized_eigh(mv, n=n, rank=rank, seed=seed)
        return LowRankTerms(U=U, S=S, scalar=0.0)
    if method == "skerch":
        return skerch_low_rank(mv, rank=rank)
    raise ValueError(f"unknown method: {method}")


def low_rank_lifting_ggn(
    model_fn: callable,
    w0: torch.Tensor,
    xs: list[torch.Tensor],
    rank: int = 50,
    oversample: int = 10,
    fwd_batch: int = 4,
    vjp_chunk: int = 64,
    loss_fn: str = "mse",
    factor: float | torch.Tensor = 1.0,
    method: str = "randomized",
    power_iter: int = 2,
    seed: int = 0,
    mem_fraction: float = 1.0,
    plan: bool = True,
    strict: bool = False,
    sample_batch_size: int = 1,
) -> LowRankTerms:
    """One-shot low-rank GGN of the lifting adapter over the samples ``xs``.

    ``model_fn`` / ``w0`` come from
    :class:`luno_torch.adapter.TorchLiftingWrapper`: the lifting weights are
    linearized and the core/projection are frozen. ``Jv`` is computed in
    forward mode (``torch.func.jvp``), exactly like ``GGNMatvec`` for the last
    FNO block. All the rank/sketch/memory options behave as in
    :func:`low_rank_ggn`.
    """
    mv = LiftingGGNMatvec(
        model_fn,
        w0,
        xs,
        mode="forward",
        fwd_batch=fwd_batch,
        loss_fn=loss_fn,
        factor=factor,
        vjp_chunk=vjp_chunk,
        sample_batch_size=sample_batch_size,
    )
    return low_rank_ggn(
        mv,
        rank=rank,
        oversample=oversample,
        method=method,
        fwd_batch=fwd_batch,
        vjp_chunk=vjp_chunk,
        power_iter=power_iter,
        seed=seed,
        mem_fraction=mem_fraction,
        plan=plan,
        strict=strict,
    )


def auto_ggn_mc(d: int, n_samples: int, fwd_batch: int = 1, oversample: float = 1.1) -> int:
    """Число случайных котангенсов на сэмпл, чтобы ``n_samples * n_mc >= 1.1 d``.

    Ранг сэмплированной GGN не больше ``n_samples * n_mc``; меньший ранг
    оставил бы часть направлений весов без кривизны (там работал бы только
    prior). Округляется вверх до кратного ``fwd_batch`` (котангенсы идут
    чанками), минимум 8.
    """
    n_mc = max(8, -(-int(oversample * d) // max(1, n_samples)))
    step = max(1, int(fwd_batch))
    return -(-n_mc // step) * step


def _eigh_psd_full(G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``eigh`` симметричной PSD-матрицы ``d x d`` по убыванию; ``(U, S)``.

    На GPU ``syevd`` держит матрицу, собственные векторы и рабочий буфер
    (~3-4 матрицы); при нехватке памяти разложение уходит на CPU, а векторы
    возвращаются на исходное устройство. Используется нижний треугольник ``G``,
    симметризация не нужна. Отрицательный шум округления обрезается до 0.
    """
    device = G.device
    try:
        S, U = torch.linalg.eigh(G)
    except torch.cuda.OutOfMemoryError:
        write(f"[GGN] eigh {tuple(G.shape)} не помещается в GPU -- считаю на CPU.")
        G_cpu = G.cpu()
        del G
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        S, U = torch.linalg.eigh(G_cpu)
        del G_cpu
        U = U.to(device)
        S = S.to(device)
    return U.flip(1), S.flip(0).clamp_min(0.0)


def sampled_full_ggn(
    model_fn: callable,
    w0: torch.Tensor,
    xs: list,
    jacobian_cls,
    n_mc: int,
    fwd_batch: int = 2,
    vjp_chunk: int = 16,
    bind: callable | None = None,
    factor: float = 1.0,
    seed: int = 0,
) -> LowRankTerms:
    """Полная GGN ``d x d`` сэмплированием выхода и её собственное разложение.

    ``G = factor * sum_s J_s^T (2I) J_s`` (MSE, как в :class:`GGNMatvec`)
    оценивается несмещённо::

        G ~= factor * sum_s (1/n_mc) sum_j g_sj g_sj^T,
        g_sj = J_s^T eps_j,  eps_j ~ N(0, 2 I)

    т.к. ``E[J^T eps eps^T J] = 2 J^T J``. На сэмпл -- один forward под
    обратный граф и ``n_mc / fwd_batch`` обратных проходов; ни одного JVP и ни
    одной строки ``J`` (выход ``m ~ 1e6`` точек не перебирается). Ранг оценки --
    ``min(d, N * n_mc)``, поэтому для полной кривизны нужно ``N * n_mc >= d``
    (см. :func:`auto_ggn_mc`).

    Возвращает ``LowRankTerms(U, S)`` с ПОЛНЫМ базисом: ``U`` -- ``(d, d)``,
    ``S`` -- все ``d`` собственных значений. Дальше пайплайн не отличает его от
    низкорангового: ``(prior_prec I + U S U^T)^{-1}`` точна, так как базис
    полный. Память -- две матрицы ``d x d`` (накопление и векторы), для
    лифтинга (``d ~ 3e4``) это ~3.5 ГБ каждая во float32.
    """
    d = w0.numel()
    device, dtype = w0.device, w0.dtype
    if d > 60000:
        raise ValueError(
            f"sampled_full_ggn: d={d} слишком велико для матрицы d x d "
            f"({d * d * 4 / 1024 ** 3:.0f} ГБ во float32); метод -- для лифтинга."
        )
    G = torch.zeros(d, d, dtype=dtype, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    step = max(1, int(fwd_batch))
    n_chunks = -(-n_mc // step)
    alpha = float(factor) / n_mc
    write(f"[GGN] полная сэмплированная GGN: d={d:,}, сэмплов={len(xs)}, "
          f"n_mc={n_mc} (ранг оценки <= {len(xs) * n_mc:,}), "
          f"обратных проходов={len(xs) * n_mc:,}")
    bar = tqdm(total=len(xs) * n_chunks, desc="[GGN] сэмпл. котангенсы",
               unit="chunk", leave=False, position=0)
    t0 = time.perf_counter()
    try:
        for i_smp, x in enumerate(xs):
            x = x.to(device=device) if torch.is_tensor(x) else x
            fn, state = bind(x) if bind is not None else (model_fn, x)
            jac = jacobian_cls(fn, state, w0, fwd_batch=fwd_batch, vjp_chunk=vjp_chunk)
            try:
                jac._ensure_vjp()
                m = jac._m
                for c0 in range(0, n_mc, step):
                    c = min(step, n_mc - c0)
                    eps = torch.randn(m, c, dtype=dtype, device=device, generator=gen)
                    eps.mul_(2.0 ** 0.5)
                    g = jac._vjp_cols(eps)  # (d, c) = J^T eps
                    G.addmm_(g, g.transpose(0, 1), alpha=alpha)
                    del eps, g
                    bar.set_postfix_str(f"сэмпл {i_smp + 1}/{len(xs)}", refresh=False)
                    bar.update(1)
            finally:
                jac._release_vjp(collect=False)
            del jac, fn, state, x
    finally:
        bar.close()
    write(f"[GGN] накопление: {time.perf_counter() - t0:.0f} с; eigh {d}x{d}...")
    t1 = time.perf_counter()
    U, S = _eigh_psd_full(G)
    del G
    tiny = float(S[0]) * d * torch.finfo(dtype).eps if S.numel() else 0.0
    n_pos = int((S > tiny).sum())
    write(f"[GGN] eigh: {time.perf_counter() - t1:.0f} с; собств. значения "
          f"{float(S[0]):.3e} .. {float(S[min(n_pos, d) - 1]):.3e}, "
          f"значимых {n_pos:,} из {d:,}")
    return LowRankTerms(U=U, S=S, scalar=0.0)


def to_dense(mv: LinearOperator) -> torch.Tensor:
    """Materialize a (small) LinearOperator as a dense matrix."""
    n, _ = mv.shape()
    eye = torch.eye(n, dtype=mv.dtype, device=mv.device)
    return torch.stack([mv._matmul(eye[:, i]) for i in range(n)], dim=1)