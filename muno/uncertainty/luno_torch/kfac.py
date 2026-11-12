"""KFAC по Фурье-модам для весов последнего FNO-блока.

Альтернатива низкоранговому GGN (``luno_torch.ggn``): вместо ``rank`` ведущих
направлений в пространстве весов размера ``d`` -- полноранговая, но
факторизованная кривизна. Стоимость -- один проход по train-сэмплам, на сэмпл
один forward и ``n_mc`` backward через хвост сети; буферов ``(d, rank)`` нет.

Откуда факторизация
-------------------
Спектральный conv по каждой моде ``k`` -- отдельный линейный слой::

    s_hat[o, k] = sum_i R[i, o, k] * z_hat[i, k]

поэтому градиент скаляра по ``R[:, :, k]`` -- внешнее произведение
``g_hat[o, k] * conj(z_hat[i, k])``, где ``g_hat`` -- градиент по выходу
контракции. Блок GGN моды ``k`` (в комплексной записи, действует на
``v = Re + i Im``) приближается произведением Кронекера::

    G_k ~= 1/(2N) * A_k (x) B_k,
    A_k = sum_s conj(z_hat) conj(z_hat)^H      (in x in)
    B_k = sum_s E_eps[g_hat g_hat^H]           (out x out),  eps ~ N(0, 2 I)

``2 I`` -- гессиан MSE (как множитель 2 в ``GGNMatvec``), ``N`` -- число сэмплов.
Skip-вес ``W`` (точечный линейный слой) и bias ``b`` -- обычный KFAC с суммой по
пространству.

Что здесь приближение (в отличие от low-rank GGN, это ДРУГАЯ аппроксимация):

* блоки разных мод, а также блоки R / W / b считаются независимыми;
* сумма по сэмплам произведений Кронекера заменена произведением сумм;
* комплексный вес считается циркулярно-симметричным (отброшена
  псевдоковариация ``E[c c^T]``); для мод, где ``z_hat`` вещественный (нулевая
  частота), это перераспределяет кривизну между Re и Im;
* для ``W`` градиенты в разных точках пространства считаются некоррелированными.

Предиктивная дисперсия
----------------------
Возмущение весов нужно только через возмущение суммы блока. Для моды ``k``
``delta s_hat = delta R_k z_hat`` при ``delta R_k ~ N(0, H_k^{-1})`` -- это
гауссов вектор размера ``out`` с ковариацией ``Q_B diag(v_k) Q_B^H``::

    v_k[b] = sum_a |zeta_a|^2 / (prior_prec + a_a * b_b / (2N)),
    zeta = Q_A^T z_hat

то есть на пробу сэмплируется ``out * K`` чисел, а не ``in * out * K``, и через
сеть идёт один JVP по замороженному хвосту (``channel_mlp`` + проекции) -- ни
блоки ядра, ни тангенты размера ``d`` не участвуют. Это точный сэмпл из
KFAC-апостериора, а не дополнительное приближение.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .affine import AffineLastBlock, BoundAffine
from .progress import tqdm


@dataclass
class KFACTerms:
    """Собственные разложения KFAC-факторов последнего блока (всё на одном устройстве)."""

    QA: torch.Tensor  # (K, in, in)   complex, собственные векторы A_k
    a: torch.Tensor  # (K, in)        real >= 0
    QB: torch.Tensor  # (K, out, out) complex
    b: torch.Tensor  # (K, out)       real >= 0
    QAW: torch.Tensor  # (in, in)     real, skip-вес W: фактор входов
    aW: torch.Tensor  # (in,)
    QBW: torch.Tensor  # (out, out)   real, skip-вес W: фактор градиентов
    bW: torch.Tensor  # (out,)
    Qb: torch.Tensor | None  # (out, out) real, bias
    bb: torch.Tensor | None  # (out,)
    mode_shape: tuple[int, ...]  # форма удержанных мод, prod == K
    n_samples: int
    n_points: int  # число точек сетки на сэмпл (нормировка фактора W)

    def to(self, device=None, dtype=None) -> "KFACTerms":
        def mv(t):
            if t is None:
                return None
            if dtype is not None:
                cd = torch.complex128 if dtype == torch.float64 else torch.complex64
                t = t.to(cd if t.is_complex() else dtype)
            return t.to(device=device) if device is not None else t

        return KFACTerms(
            QA=mv(self.QA), a=mv(self.a), QB=mv(self.QB), b=mv(self.b),
            QAW=mv(self.QAW), aW=mv(self.aW), QBW=mv(self.QBW), bW=mv(self.bW),
            Qb=mv(self.Qb), bb=mv(self.bb), mode_shape=tuple(self.mode_shape),
            n_samples=self.n_samples, n_points=self.n_points,
        )

    @property
    def num_modes(self) -> int:
        return int(self.a.shape[0])


def _eigh_psd(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``eigh`` эрмитовой PSD-матрицы (батчем) с обрезкой отрицательного шума."""
    M = 0.5 * (M + M.mH)
    evals, evecs = torch.linalg.eigh(M)
    return evecs, evals.clamp_min(0.0)


class KFACLastBlock:
    """Контекст KFAC: модули последнего блока + ``model_fn`` конкретной задачи.

    Структуру блока проверяет :class:`luno_torch.affine.AffineLastBlock` (плотный
    комплексный вес, postactivation, без norm, линейный skip без bias).
    ``model_fn`` / ``w0`` -- те же, что у low-rank пути; они нужны только для
    reference-прохода с хуками (:class:`BoundAffine`).
    """

    def __init__(self, affine: AffineLastBlock, model_fn, w0: torch.Tensor):
        self.p = affine
        self._model_fn = model_fn
        self._w0 = w0

    # ------------------------------------------------------------------ utils
    def _conv_with(self, x: torch.Tensor, contract):
        """``conv(x)`` с подменённой контракцией ``contract(x_hat, weight, orig)``.

        FFT, выбор мод, сдвиги и нормировка остаются родными для ``SpectralConv``:
        подменяется только произведение на вес, поэтому всё, что через это
        измеряется или подаётся, живёт в той же конвенции, что и сам вес.
        """
        conv = self.p.conv
        orig = conv._contract

        def patched(x_hat, weight, separable=False):
            return contract(x_hat, weight, orig)

        conv._contract = patched
        try:
            return conv(x)
        finally:
            conv._contract = orig

    def _tail(self, s: torch.Tensor, cskip: torch.Tensor) -> torch.Tensor:
        p = self.p
        return p.projection(p.core_tail(p.cmlp(s) + cskip))

    def _zhat(self, z: torch.Tensor) -> torch.Tensor:
        """Спектр входа блока на удержанных модах: ``(B, in, *mode_shape)``."""
        rec = {}

        def contract(x_hat, weight, orig):
            rec["zhat"] = x_hat
            return orig(x_hat, weight, separable=False)

        with torch.no_grad():
            self._conv_with(z, contract)
        return rec["zhat"]

    # --------------------------------------------------------------- факторы
    def new_accumulator(self) -> dict:
        return {"n": 0}

    def accumulate(
        self,
        x: torch.Tensor,
        acc: dict,
        n_mc: int = 8,
        generator: torch.Generator | None = None,
        exact: bool = False,
    ) -> None:
        """Добавляет вклад одного сэмпла ``x`` в факторы.

        ``exact=True`` заменяет ``n_mc`` случайных котангенсов полным базисом
        выходов (``m`` backward) -- только для тестов на маленьких моделях.
        """
        p = self.p
        bound = BoundAffine(p, x, self._model_fn, self._w0)
        z, cskip = bound.z, bound.cskip
        rec = {}

        def contract(x_hat, weight, orig):
            out = orig(x_hat, weight, separable=False)
            rec["zhat"] = x_hat.detach()
            rec["d"] = torch.zeros_like(out, requires_grad=True)
            return out + rec["d"]

        with torch.enable_grad():
            s_conv = self._conv_with(z, contract)
            ds = torch.zeros_like(s_conv, requires_grad=True)
            s = s_conv + p.conv.transform(p.skip(z), output_shape=None) + ds
            out = self._tail(s, cskip)
            m = out.numel()
            n_cot = m if exact else int(n_mc)
            weight = 1.0 if exact else 1.0 / n_cot

            zh = rec["zhat"][0].flatten(1)  # (in, K)
            zs = z[0].flatten(1)  # (in, X)
            n_out = rec["d"].shape[1]
            if "A" not in acc:
                K, n_in = zh.shape[1], zh.shape[0]
                acc["A"] = torch.zeros(K, n_in, n_in, dtype=zh.dtype, device=zh.device)
                acc["B"] = torch.zeros(K, n_out, n_out, dtype=zh.dtype, device=zh.device)
                acc["AW"] = torch.zeros(n_in, n_in, dtype=zs.dtype, device=zs.device)
                acc["BW"] = torch.zeros(n_out, n_out, dtype=zs.dtype, device=zs.device)
                acc["Bb"] = torch.zeros(n_out, n_out, dtype=zs.dtype, device=zs.device)
                acc["mode_shape"] = tuple(rec["zhat"].shape[2:])
                acc["n_points"] = zs.shape[1]

            acc["A"] += torch.einsum("ik,jk->kij", zh.conj(), zh)
            acc["AW"] += zs @ zs.T

            for j in range(n_cot):
                if exact:
                    eps = torch.zeros(m, dtype=out.dtype, device=out.device)
                    eps[j] = 2.0 ** 0.5
                    eps = eps.reshape(out.shape)
                else:
                    eps = (2.0 ** 0.5) * torch.randn(
                        out.shape, dtype=out.dtype, device=out.device,
                        generator=generator,
                    )
                g_hat, g_s = torch.autograd.grad(
                    out, [rec["d"], ds], grad_outputs=eps,
                    retain_graph=j + 1 < n_cot,
                )
                gh = g_hat[0].flatten(1)  # (out, K)
                gs = g_s[0].flatten(1).to(zs.dtype)  # (out, X)
                acc["B"].add_(torch.einsum("ok,pk->kop", gh, gh.conj()), alpha=weight)
                acc["BW"].add_(gs @ gs.T, alpha=weight)
                gsum = gs.sum(dim=1)
                acc["Bb"].add_(torch.outer(gsum, gsum), alpha=weight)
        acc["n"] += 1

    def finalize(self, acc: dict) -> KFACTerms:
        if not acc.get("n"):
            raise ValueError("KFAC: не накоплено ни одного сэмпла.")
        QA, a = _eigh_psd(acc["A"])
        QB, b = _eigh_psd(acc["B"])
        QAW, aW = _eigh_psd(acc["AW"])
        QBW, bW = _eigh_psd(acc["BW"])
        Qb = bb = None
        if self.p.b_key is not None:
            Qb, bb = _eigh_psd(acc["Bb"])
        return KFACTerms(
            QA=QA, a=a, QB=QB, b=b, QAW=QAW, aW=aW, QBW=QBW, bW=bW,
            Qb=Qb, bb=bb, mode_shape=acc["mode_shape"],
            n_samples=int(acc["n"]), n_points=int(acc["n_points"]),
        )

    def fit(self, xs, n_mc: int = 8, seed: int = 0, exact: bool = False) -> KFACTerms:
        """Факторы по списку сэмплов ``xs`` (один проход)."""
        device = self._w0.device
        gen = torch.Generator(device=device).manual_seed(seed)
        acc = self.new_accumulator()
        for x in tqdm(xs, desc="[KFAC] сэмплы", unit="smp", leave=False):
            x = x.to(device=device) if torch.is_tensor(x) else x
            self.accumulate(x, acc, n_mc=n_mc, generator=gen, exact=exact)
        return self.finalize(acc)

    # ------------------------------------------------------------- предиктив
    def mean_and_var(
        self,
        x: torch.Tensor,
        terms: KFACTerms,
        prior_precs,
        n_probe: int = 32,
        chunk: int = 8,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mean, var)``: ``mean`` формы ``(m,)``, ``var`` -- ``(len(prior_precs), m)``.

        Все кандидаты ``prior_prec`` используют один и тот же шум (common random
        numbers), поэтому цель калибровки по гриду гладкая. Стоимость --
        ``len(prior_precs) * n_probe`` JVP через хвост сети.
        """
        p = self.p
        if n_probe < 1:
            raise ValueError(f"n_probe должен быть >= 1, получено {n_probe}")
        precs = torch.as_tensor(prior_precs).reshape(-1).tolist()
        with torch.no_grad():
            bound = BoundAffine(p, x, self._model_fn, self._w0)
            z, s0, cskip = bound.z, bound.s0, bound.cskip
            rdtype, device = s0.dtype, s0.device
            zhat = self._zhat(z)
            if tuple(zhat.shape[2:]) != tuple(terms.mode_shape):
                raise ValueError(
                    f"KFAC: удержанные моды {tuple(zhat.shape[2:])} не совпали с "
                    f"модами факторов {tuple(terms.mode_shape)} -- факторы "
                    "посчитаны на другой сетке."
                )
            cdtype = zhat.dtype
            zh = zhat[0].flatten(1)  # (in, K)
            K, n_out = terms.num_modes, terms.QB.shape[1]
            N, X = terms.n_samples, terms.n_points

            # Не зависит ни от prior_prec, ни от пробы.
            zeta2 = torch.einsum("ik,kia->ka", zh, terms.QA.to(cdtype)).abs() ** 2
            ab = terms.a.unsqueeze(-1) * terms.b.unsqueeze(-2) / (2.0 * N)  # (K, in, out)
            zs = z[0].flatten(1)  # (in, X)
            zetaW = terms.QAW.to(rdtype).T @ zs  # (in, X)
            abW = terms.bW.unsqueeze(-1) * terms.aW.unsqueeze(-2) / float(N * X)  # (out, in)
            QB = terms.QB.to(cdtype)
            QBW = terms.QBW.to(rdtype)
            bias = p.conv.bias
            spatial = s0.shape[2:]

            sd = []  # sqrt(v_k[b]) на кандидата: (K, out)
            for lam in precs:
                sd.append(torch.sqrt((zeta2.unsqueeze(-1) / (lam + ab)).sum(dim=1)))

            acc = torch.zeros(len(precs), bound.fx.numel(), dtype=rdtype, device=device)
            for c0 in range(0, n_probe, max(1, chunk)):
                c = min(max(1, chunk), n_probe - c0)
                eta = torch.complex(
                    torch.randn(c, K, n_out, dtype=rdtype, device=device, generator=generator),
                    torch.randn(c, K, n_out, dtype=rdtype, device=device, generator=generator),
                ).to(cdtype)
                xiW = torch.randn(c, n_out, zs.shape[0], dtype=rdtype, device=device,
                                  generator=generator)
                xib = None
                if terms.Qb is not None:
                    xib = torch.randn(c, n_out, dtype=rdtype, device=device,
                                      generator=generator)
                zc = z.expand(c, *z.shape[1:])
                # clone: jvp делает primal dual-тензором, а у expand все копии
                # делят одну память.
                s0c = s0.expand(c, *s0.shape[1:]).clone()
                cskipc = cskip.expand(c, *cskip.shape[1:])
                for i, lam in enumerate(precs):
                    # R: delta s_hat[o, k] = sum_b Q_B[k, o, b] sqrt(v[k, b]) eta[k, b]
                    dsh = torch.einsum("kob,ckb->cok", QB, sd[i].to(cdtype) * eta)
                    dsh = dsh.reshape(c, n_out, *terms.mode_shape)
                    t = self._conv_with(zc, lambda xh, w, orig: dsh.to(xh.dtype))
                    if bias is not None:
                        t = t - bias
                    t = t.to(rdtype)
                    # W: delta W z, delta W = Q_BW (D^{-1/2} o xi) Q_AW^T
                    tW = QBW @ ((xiW / torch.sqrt(lam + abW)) @ zetaW)  # (c, out, X)
                    t = t + p.conv.transform(
                        tW.reshape(c, n_out, *z.shape[2:]), output_shape=None
                    )
                    if xib is not None:
                        db = (xib / torch.sqrt(lam + terms.bb.to(rdtype))) @ terms.Qb.to(rdtype).T
                        t = t + db.reshape(c, n_out, *([1] * len(spatial)))
                    jv = torch.func.jvp(
                        lambda s: self._tail(s, cskipc), (s0c,), (t,)
                    )[1]
                    acc[i].add_((jv.reshape(c, -1) ** 2).sum(dim=0))
            return bound.fx.reshape(-1), acc / n_probe
