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

# Во сколько раз максимум |J v| может отличаться от максимума |rows @ v|,
# не считая ошибкой. Обрыв tangent-пути обычно даёт полный ноль; окно generous'ное,
# т.к. в float32 отдельные пробы гуляют на ~30%.
_MAG_WINDOW = 100.0

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
    output size. J itself is never materialized.

    ``Jv`` / ``J @ W`` (единственный путь для прямых произведений) считаются
    forward-mode AD через ``torch.func.jvp`` -- это режим по умолчанию (``mode="forward"``).
    Одна касательная на всё направление = один прямой проход по сети; несколько
    направлений считаются батчем по ``fwd_batch`` колонок через ``torch.vmap``.
    Точная производная, а не приближение, строки J не материализуются.

    Альтернатива ``mode="affine"``: :class:`luno_torch.affine.AffineLastBlock`
    прогоняет спектральный conv и skip последнего блока с *касательными* весами
    (обычный forward без AD) и протягивает результат через замороженный хвост.
    Тот же точный результат, но fwAD не дифференцирует спектральный conv вообще --
    полезно, если fwAD через in-place запись в комплексный буфер FFT
    (``out_fft[slices_x] = ...``) окажется неточен на конкретной версии torch/CUDA.

    ``J^T u`` и ``diag(JJ^T)`` всегда остаются на reverse-mode (``torch.func.vjp``)
    с чанками строк по ``vjp_chunk``.

    В режиме ``affine`` требуется ``affine`` (:class:`luno_torch.affine.AffineLastBlock`).
    """

    def __init__(
        self,
        model_fn: callable,
        x: torch.Tensor,
        w0: torch.Tensor,
        affine=None,
        mode: str = "forward",
        fwd_batch: int = 4,
        num_output_channels: int | None = None,
        output_grid_shape: tuple[int, ...] | None = None,
        vjp_chunk: int = 64,
        vjp_refresh_every: int = 50,
        diag_probes: int = 0,
    ):
        if mode not in ("forward", "affine"):
            raise ValueError(
                f"Неизвестный mode={mode!r}: допустимо 'forward' или 'affine'."
            )
        self._model_fn = model_fn
        self._x = x
        self._w0 = w0
        self._affine = affine
        self._mode = mode
        self._fwd_batch = int(fwd_batch)
        self._num_output_channels = num_output_channels
        self._output_grid_shape = output_grid_shape
        self._vjp_chunk = vjp_chunk
        self._vjp_refresh_every = vjp_refresh_every
        # 0 -- точный diag(J J^T) обратным режимом (m строк по vjp_chunk за чанк);
        # >0 -- оценка Хатчинсона на diag_probes проб, forward-режимом.
        self._diag_probes = int(diag_probes)
        self._diag_JJT_cache = None

        # Ленивая сборка: forward/VJP строятся только при первом использовании,
        # а после — освобождаются (_release_vjp). Это JAX-подобная семантика
        # пересчёта: граф одного сэмпла не удерживается в памяти между матвеками,
        # иначе max_num_samples * (размер графа) упирается в CUDA-память.
        # Аффинное состояние (z, s0, a0) кэшируется отдельно и тоже освобождается.
        self._flat_fn = None
        self._fx = None
        self._vjp_fn = None
        self._vjp_rows = None
        self._vjp_cols = None
        self._bound = None

        super().__init__()

    def _ensure_flat(self):
        """Строит только ``_flat_fn`` -- достаточно для forward-режима (``torch.func.jvp``).

        Обратные примитивы при этом не строятся, поэтому чистый ``Jv``/``J @ W``
        не держит обратный граф и не тратит на него память.

        Замыкание захватывает ТОЛЬКО локальные ``model_fn``/``x``, а не ``self``:
        лямбда со ссылкой на ``self`` замыкает объект в цикл
        ``self -> self._flat_fn -> cell -> self``, из-за чего сборщик мусора не
        освобождает удержанный обратный граф (и его активации на GPU) до конца
        эпохи. Для цикла из 500 eval-сэмплов это и есть накопление памяти.
        """
        if self._flat_fn is None:
            model_fn, x = self._model_fn, self._x
            self._flat_fn = lambda w: model_fn(x, w).reshape(-1)
        return self

    def _ensure_vjp(self):
        """Строит forward и VJP-примитивы лениво (no-op, если уже построены)."""
        self._ensure_flat()
        if self._vjp_fn is not None:
            return self
        self._fx = self._flat_fn(self._w0)
        self._vjp_fn = torch.func.vjp(self._flat_fn, self._w0)[1]
        self._vjp_rows = torch.vmap(
            lambda c: self._vjp_fn(c, create_graph=False)[0], in_dims=0
        )
        self._vjp_cols = torch.vmap(
            lambda c: self._vjp_fn(c, create_graph=False)[0], in_dims=-1, out_dims=-1
        )
        return self

    def _ensure_affine(self):
        """Строит аффинное состояние (z, s0, a0) лениво (no-op, если уже построено)."""
        if self._bound is None:
            if self._affine is None:
                raise ValueError(
                    "LastFNOBlockWeightJacobian требует affine (luno_torch.affine."
                    "AffineLastBlock) для mode='affine'."
                )
            self._bound = self._affine.bind(self._x, self._model_fn, self._w0)
        return self._bound

    def _release_vjp(self):
        """Освобождает удержанный forward-граф и аффинное состояние; следующий вызов пересчитает."""
        self._fx = None
        self._vjp_fn = None
        self._vjp_rows = None
        self._vjp_cols = None
        self._bound = None
        torch.cuda.empty_cache()
        return self

    def _rebuild_vjp(self):
        """Пересоздаёт forward-граф и VJP-примитивы (сбрасывает удержанную память)."""
        return self._release_vjp()._ensure_vjp()

    @property
    def fx(self) -> torch.Tensor:
        # fx — это f(x, w0), достаточно одного прямого прохода; обратный граф для
        # этого не нужен, поэтому берём его без удержания графа.
        if self._bound is not None:
            return self._bound.fx
        self._ensure_flat()
        return self._flat_fn(self._w0).detach()

    def shape(self) -> tuple[int, int]:
        d = self._w0.numel()
        m = self.fx.numel()
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

    def _matmul_forward(self, weights: torch.Tensor) -> torch.Tensor:
        """J v или J @ W через forward-mode (``torch.func.jvp``), без строк J.

        Одна касательная = один прямой проход сети. Для W формы (d, k) колонки
        считаются батчем по ``fwd_batch`` через ``torch.vmap``: каждая колонка
        получает свою касательную, vmap проходит по ней как по батч-измерению.

        Результат пишется в заранее выделенный буфер ``out`` размера (m, k), а не
        собирается списком чанков с последующим ``torch.cat``: конкатенация
        держит ОДНОВРЕМЕННО и список чанков (m, k), и результат (m, k), то есть
        удваивает пик. При d = 27.6M и k = rank это лишние ~5 ГиБ на rank = 50,
        которых не хватает вместе со скетчем.

        Возвращаемое значение всегда ``requires_grad=False``. При включённом grad
        mode тангенс ``torch.func.jvp`` наследует град от параметров сети, а
        присваивание его в ``out`` делает ``out`` grad-трекаемым с ``grad_fn =
        CopySlices``. Такой тензор удерживает ПОЛНЫЙ прямой граф сети (~0.41 ГиБ
        при d = 27.6M), и каждый ``J @ W`` привязывает к результату свой граф:
        запись в накопитель проб (см. ``sample_congruence_probes``) копила их по
        0.41 ГиБ на чанк и уходила в OOM уже на 35-м из 50. ``.detach()``
        отбрасывает граф без копирования -- значения побитово те же.
        """
        flat_fn = self._flat_fn
        w0 = self._w0
        if weights.dim() == 1:
            return torch.func.jvp(flat_fn, (w0,), (weights,))[1].detach()
        k = weights.shape[-1]
        m = self.fx.numel()
        if k == 0:
            return torch.zeros(m, 0, dtype=w0.dtype, device=w0.device)
        out = torch.empty(m, k, dtype=w0.dtype, device=w0.device)
        step = max(1, self._fwd_batch)
        for c0 in range(0, k, step):
            c1 = min(c0 + step, k)
            # (d, b) -> (b, d): каждая строка — отдельное направление для vmap.
            tangents = weights[:, c0:c1].transpose(0, 1).contiguous()
            outs = torch.vmap(
                lambda t: torch.func.jvp(flat_fn, (w0,), (t,))[1],
                in_dims=0, out_dims=0,
            )(tangents)  # (b, m)
            out[:, c0:c1] = outs.transpose(0, 1).detach()
            del tangents, outs
        return out

    def _matmul(self, weights: torch.Tensor) -> torch.Tensor:
        """J v или J @ W (W формы (d, k)) выбранным режимом, без строк J."""
        weights = weights.to(self._w0)
        if self._mode == "affine":
            return self._ensure_affine().jmatmul(weights)
        self._ensure_flat()  # для jvp нужен только _flat_fn; VJP-граф не строится
        return self._matmul_forward(weights)

    def transpose(self) -> "LastFNOBlockTransposeWeightJacobian":
        return LastFNOBlockTransposeWeightJacobian(self)

    def diag_JJT(self) -> torch.Tensor:
        """Row-wise squared norms of J, i.e. diag(J J^T).

        При ``diag_probes > 0`` возвращается безсмещённая оценка Хатчинсона
        forward-режимом, иначе -- точное значение обратным режимом (по строкам).
        """
        if self._diag_JJT_cache is not None:
            return self._diag_JJT_cache
        if self._diag_probes > 0:
            result = self._diag_JJT_hutchinson(self._diag_probes)
        else:
            result = self._diag_JJT_exact()
        self._diag_JJT_cache = result
        return result

    def _diag_JJT_exact(self) -> torch.Tensor:
        """diag(J J^T) построчно: одна обратная прохода на каждую строку J."""
        norms = []
        try:
            for sl, rows in self._chunked_rows():
                norms.append((rows**2).sum(dim=-1))
                del rows
            result = torch.cat(norms)
        finally:
            # Граф последнего чанка не переживает выход из диагонали: иначе он
            # удерживается до следующего вызова и живёт вместе со всеми
            # оставшимися сэмплами eval-цикла.
            norms.clear()
            self._release_vjp()
        return result

    def _diag_JJT_hutchinson(self, n_probe: int) -> torch.Tensor:
        """Оценка Хатчинсона для diag(J J^T), полностью в прямом режиме.

        Для ``Z ∈ {±1}^{d×c}`` по элементам с независимыми `E[Z_ik²] = 1`:

            E[(J Z)_ik²] = Σ_j J_ij² · E[Z_jk²] = Σ_j J_ij² = (J J^T)_ii

        то есть усреднение ``(J Z)²`` по пробам даёт безсмещённую оценку искомой
        диагонали. Стоимость -- ``n_probe`` прямых проходов вместо
        ``ceil(m / vjp_chunk)`` обратных (при m = 65536 и vjp_chunk = 16 это 16
        против 4096, т.е. ~256x), память на пробу ограничена чанком
        ``fwd_batch``: ``fwd_batch * d * 4`` байт.

        Оценка относится к самой диагонали ``(J J^T)_ii``. Если дальше эта
        диагональ вычитается из сопоставимого слагаемого, шум окажется
        усиленным -- в этом случае оценивать надо всю величину, см.
        :func:`_var_of_congruence_hutchinson`.

        Ошибка на один элемент ~ ``sqrt(2/n_probe)`` (на пробе chi2_1), то есть
        n_probe = 16 даёт ~20%, n_probe = 64 -- ~6%. ``n_probe`` -- это обмен
        скорости на точность; калибровка, val и test делят одно значение флага
        ``--diag-hutchinson``, а число проб не влияет на смещение оценщика var
        (оно ``~2/p``), только на его разброс.

        Обратный граф не строится, поэтому ``_release_vjp`` не требуется.
        """
        m = self._fx_numel()
        w0 = self._w0
        acc = torch.zeros(m, dtype=w0.dtype, device=w0.device)
        step = max(1, self._fwd_batch)
        for c0 in range(0, n_probe, step):
            c = min(step, n_probe - c0)
            # Bernoulli in-place -> {-1, +1} без второго буфера под Z.
            Z = torch.empty(w0.numel(), c, dtype=w0.dtype, device=w0.device)
            Z.bernoulli_(0.5).mul_(2).sub_(1)
            acc.add_((self._matmul(Z) ** 2).sum(dim=1))
            del Z
        return acc / n_probe

    def diag_JJT_times(self, diag: torch.Tensor) -> torch.Tensor:
        """Row-wise ``diag(J D J^T)`` for a weight-space diagonal D."""
        diag = diag.to(self._w0)
        out = torch.empty(self._fx_numel(), dtype=self._w0.dtype, device=self._w0.device)
        try:
            for sl, rows in self._chunked_rows():
                out[sl] = (rows**2) @ diag
                del rows
        finally:
            self._release_vjp()
        return out

    def _fx_numel(self) -> int:
        """Размер выхода f(x, w0) без построения и удержания обратного графа."""
        if self._bound is not None:
            return self._bound.fx.numel()
        self._ensure_flat()
        return self._flat_fn(self._w0).detach().numel()


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


def _cov_sqrt_isotropic_lowrank(
    Sigma: IsotropicScalingPlusSymmetricLowRank,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Факторизация ``Sigma^{1/2}`` для ``IsotropicScalingPlusSymmetricLowRank``.

    Та же формула, что ``linox.lsqrt``:

        ``Sigma = s I + U diag(S) U^T``
        ``Sigma^{1/2} = sqrt(s) I + U diag( sqrt(s)(sqrt(S/s + 1) - 1) ) U^T``

    Возвращает ``(scale, U, coeff)`` с ``Sigma^{1/2} = scale*(I + U diag(coeff) U^T)``.
    ``S`` может быть отрицательным (после ``linverse``), но ``S/s + 1 = 1 - r > 0``,
    поэтому корень определён.
    """
    U = Sigma.U
    scalar = Sigma.scalar
    if not isinstance(scalar, torch.Tensor):
        scalar = torch.tensor(scalar, dtype=U.dtype, device=U.device)
    coeff = torch.sqrt(Sigma.S / scalar + 1.0) - 1.0
    return torch.sqrt(scalar), U, coeff


def sample_congruence_probes(
    J: LastFNOBlockWeightJacobian,
    U: torch.Tensor,
    n_probe: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Пробы Хатчинсона, не зависящие от ``Sigma``: ``(A, B, W) = (J@Z, J@U, U^T Z)``.

    ``A`` формы ``(m, n_probe)``, ``B`` -- ``(m, k)``, ``W`` -- ``(k, n_probe)``,
    где ``Z ∈ {±1}^{d×n_probe}`` выбирается один раз на весь прогон и чанкуется
    по ``fwd_batch``: полный ``Z`` весил бы ``d * n_probe * 4`` байт, то есть
    22 ГиБ при ``d = 27.6M`` и ``n_probe = 200``.

    Кроме ``U`` здесь ничего не зависит от ``Sigma``, поэтому однажды посчитанные
    величины переиспользуются для всего грида ``prior_prec``: на каждом кандидате
    меняются только ``scalar``/``coeff`` (``O(k)``), а ``n_probe`` прямых проходов
    не повторяются. Одинаковый ``Z`` на всех кандидатах даёт common random numbers
    -- цель по ``prior_prec`` получается гладкой, а не зашумлённой независимым
    шумом на каждой итерации.

    Память -- ``(m, n_probe) + (m, k)`` байт (~52 МБ при ``m = 65536`` и
    ``n_probe = 200``, против ``U`` на 553 МБ при rank 5) плюс обычный рабочий
    буфер прямого режима ``fwd_batch * d * 4`` байт.
    """
    if n_probe < 1:
        raise ValueError(f"n_probe должен быть >= 1, получено {n_probe}")
    w0 = J._w0
    m = J._fx_numel()
    k = U.shape[1]
    dtype, device = w0.dtype, w0.device
    A = torch.empty(m, n_probe, dtype=dtype, device=device)
    B = J._matmul(U)
    W = torch.empty(k, n_probe, dtype=dtype, device=device)
    step = max(1, J._fwd_batch)
    for c0 in range(0, n_probe, step):
        c1 = min(c0 + step, n_probe)
        # Bernoulli in-place -> {-1, +1} без второго буфера под Z.
        Z = torch.empty(w0.numel(), c1 - c0, dtype=dtype, device=device)
        Z.bernoulli_(0.5).mul_(2).sub_(1)
        A[:, c0:c1] = J._matmul(Z)
        W[:, c0:c1] = U.transpose(0, 1) @ Z
        del Z
    return A, B, W


def var_from_congruence_probes(
    A: torch.Tensor,
    B: torch.Tensor,
    W: torch.Tensor,
    Sigma: IsotropicScalingPlusSymmetricLowRank,
) -> torch.Tensor:
    """``diag(J Sigma J^T)`` из проб, взятых в :func:`sample_congruence_probes`.

    ``T = scale * (A + (B*coeff) @ W)`` -- это ровно ``J Sigma^{1/2} Z``, поэтому
    ``mean_p T_ip^2`` -- безсмещённая оценка искомой диагонали и сумма квадратов,
    то есть неотрицательна по построению (см. :func:`_var_of_congruence_hutchinson`).
    """
    scale, _, coeff = _cov_sqrt_isotropic_lowrank(Sigma)
    T = scale * (A + (B * coeff) @ W)
    return (T ** 2).sum(dim=1) / A.shape[1]


def _var_of_congruence_hutchinson(
    J: LastFNOBlockWeightJacobian,
    Sigma: IsotropicScalingPlusSymmetricLowRank,
    n_probe: int,
) -> torch.Tensor:
    """Оценка Хатчинсона для ``diag(J Sigma J^T)`` целиком, прямым режимом.

    Для ``Z ∈ {±1}^{d×c}`` с независимыми ``E[Z z^T] = I``:

        ``E[(J Sigma^{1/2} Z)_ik^2] = (J Sigma J^T)_ii``

    Оценивается **вся** диагональ конгруэнции одним семейством проб, а не два
    слагаемых ``scalar*diag(JJ^T)`` и ``sum((JU)^2 * S)`` по отдельности.

    Разделение неприемлемо, потому что после ``linverse`` ``S < 0``: оба слагаемых
    одного порядка и почти вычитают друг друга (``var = scalar*(|g|^2 - sum r_k (g.u_k)^2)``,
    ``r_max ≈ 0.59``). Шум Хатчинсона на первом слагаемом приводит к тому, что у
    доли строк оценка уходит в минус, ``torch.sqrt`` даёт NaN, и один NaN
    отравляет ``nll.mean()``/``chi2.sum()`` целиком. При построении через
    ``Sigma^{1/2}`` результат -- сумма квадратов, поэтому он неотрицателен по
    построению, а относительная ошибка ``~sqrt(2/n_probe)`` относится к самому
    ``var_i``, а не к большому ``scalar*diag_i``.

    Стоимость прежняя: один ``J @ U`` (нужен и точному пути) плюс ``n_probe``
    прямых проходов на ``J @ Z``; ``U^T Z`` и ``(k, c)``-умножения пренебрежимо дёшевы.

    Обёртка над :func:`sample_congruence_probes` + :func:`var_from_congruence_probes`:
    ради переиспользования проб на гриде ``prior_prec`` они вынесены наружу, а сами
    оценки для одного и того же ``Z`` совпадают байт-в-байт.
    """
    A, B, W = sample_congruence_probes(J, Sigma.U, n_probe)
    return var_from_congruence_probes(A, B, W, Sigma)


def var_of_congruence(
    J: LastFNOBlockWeightJacobian, Sigma: LinearOperator, *, diag_probes: int = 0
) -> torch.Tensor:
    """diag(J Sigma J^T) as in luno._linox, split over Sigma's operator_list.

    Uses ``diag(J J^T)`` (reverse-mode over output basis) for the diagonal part and
    ``(J U)^2 S`` (forward-mode over rank basis) for the low-rank part.

    ``diag_probes > 0`` включает оценку Хатчинсона прямым режимом вместо обратного.
    Для ``IsotropicScalingPlusSymmetricLowRank`` оценивается вся конгруэнция через
    ``Sigma^{1/2}`` (см. :func:`_var_of_congruence_hutchinson`): частичная оценка
    недопустима из-за почти полного вычитания ``scalar*diag(JJ^T)`` и отрицательного
    низкорангового слагаемого. Остальные типы не оцениваются шумно: их
    низкоранговая часть точная, а ``Sigma.diag`` неотрицателен.
    """
    if diag_probes > 0 and isinstance(Sigma, IsotropicScalingPlusSymmetricLowRank):
        return _var_of_congruence_hutchinson(J, Sigma, diag_probes)
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


def default_check_tol(dtype: torch.dtype) -> float:
    """Стартовый допуск self-check Jv против reverse-строк, разный для 32/64.

    Метрика self-check -- ошибка ``|J v - (rows @ v)|``, нормированная на
    ``||row||_1 * max|v|``, то есть на масштаб слагаемых в ``rows @ v``, а НЕ на
    ``|J v|``. При ``d = 27 651 660`` строка J каскадно сокращается (измеренное
    отношение суммы модулей к результату ~1e2), поэтому нормировка на результат
    в float32 измеряет не ошибку ``Jv``, а шум суммирования: разные, корректно
    согласованные пути расходятся на ~3e-1. Нормировка на слагаемые этого
    убирает.

    Измерено на реальном весе (3 пробы, reverse-строки из ``torch.func.vjp``):
    float64 -> 7.8e-15, float32 -> 7e-4. Допуски выбраны с запасом относительно
    измеренного шума, но заведомо ниже ошибок, которые ловит гейт (обрыв
    tangent-пути, потерянная мнимая часть complex-веса или неверно подключённый
    модуль блока дают 1e-1..1e0 по этой шкале).
    """
    if dtype == torch.float64:
        return 1e-6
    return 5e-2


def check_jacobian(
    model_fn: callable,
    x: torch.Tensor,
    w0: torch.Tensor,
    affine=None,
    mode: str = "forward",
    fwd_batch: int = 4,
    num_probes: int = 3,
    tol: float | None = None,
) -> tuple[bool, float]:
    """Проверяет J*v выбранного режима против reverse-строк, возвращает ``(ok, rel)``.

    Эталон: ``num_probes`` строк J через reverse-mode (``_vjp_rows``), домноженных
    на случайный ``v``; сравнивается с ``J v`` по тем же выходным координатам.
    Используется как стартовый гейт пайплайна: любое расхождение (зацепление за
    неверный модуль блока, неучтённая нормализация, потеря мнимой части при
    кастинге) должно падать явно, а не молча давать кривую кривизну.

    Ошибка нормируется на масштаб слагаемых в ``rows @ v``
    (``||row||_1 * max|v|``), а не на ``|J v|``: при ``d ~ 2.8e7`` строка J
    сокращается каскадно, и нормировка на результат в float32 даёт ~3e-1 даже
    для заведомо корректных forward/reverse (см. :func:`default_check_tol`).

    Дополнительно проверяется, что ``J v`` не вырожден: обрыв tangent-пути даёт
    ``|J v|``, кратно меньший ``|rows @ v|``, и на шкале слагаемых это всего
    ``1/1e2``, то есть в float32 проскочило бы. Поэтому максимумы ``|Jv|`` и
    ``|rows @ v|`` по пробам должны совпадать с точностью до ``_MAG_WINDOW``
    (100x); при нарушении в отчётное число подставляется 1.0 и гейт падает при
    любом dtype.

    ``tol=None`` -> :func:`default_check_tol` для точности ``w0`` (float32 и
    float64 допускаются разные).
    """
    if tol is None:
        tol = default_check_tol(w0.dtype)
    jac = LastFNOBlockWeightJacobian(
        model_fn, x, w0, affine=affine, mode=mode, fwd_batch=fwd_batch,
    )
    jac._ensure_vjp()
    m = jac._fx.numel()
    p = min(num_probes, m)
    d = w0.numel()
    dev = w0.device

    v = torch.randn(d, dtype=w0.dtype, device=dev)
    cot = torch.zeros(p, m, dtype=w0.dtype, device=dev)
    cot[torch.arange(p), torch.arange(p)] = 1.0
    rows = jac._vjp_rows(cot)
    rev = rows @ v  # (p,)
    fwd = jac._matmul(v)[:p]  # (p,)

    # Масштаб слагаемых: |row_j * v_j| <= ||row||_1 * max|v|.
    tiny = torch.finfo(w0.dtype).tiny
    term = (rows.abs().sum(dim=1) * v.abs().max()).clamp_min(tiny)
    rel = ((rev - fwd).abs() / term).max().item()

    # Отдельный тест на вырожденность Jv (обрыв tangent-пути): |Jv| кратно
    # меньше эталонных значений. Сравниваем максимумы по пробам, а не пары
    # "probe k против probe k": в float32 отдельная проба гуляет на ~30%, а
    # максимум по нескольким пробам устойчив.
    mrev = rev.abs().max()
    mfwd = fwd.abs().max()
    ratio = float((mfwd / mrev.clamp_min(tiny)).item())
    if not (1.0 / _MAG_WINDOW <= ratio <= _MAG_WINDOW):
        # Отчётное число делаем заведомо проигрышным, чтобы и сообщение об
        # ошибке не выглядело как "проверка прошла с запасом".
        rel = max(rel, 1.0)

    jac._release_vjp()
    return rel < tol, rel


# Историческое имя: affine-режим self-check. Оставлено для совместимости вызовов.
def check_affine_jacobian(
    model_fn: callable,
    x: torch.Tensor,
    w0: torch.Tensor,
    affine=None,
    num_probes: int = 3,
    tol: float | None = None,
) -> tuple[bool, float]:
    return check_jacobian(
        model_fn, x, w0, affine=affine, mode="affine", num_probes=num_probes, tol=tol,
    )