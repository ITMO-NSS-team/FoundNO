"""Точный J*v для весов последнего FNO-блока: аффинный тангенс без AD через conv.

Модель *ровно* аффинна по настраиваемым весам последнего блока (R, b спектрального
conv и W точечного skip):

    z  = fno_blocks(x, 0 .. nl-1)      # вход последнего блока, от w не зависит
    a  = conv(z; R, b) + transform(skip(z; W))
    y  = channel_mlp(nl)(a) + channel_mlp_skip(nl)(z)      # norm=None, preactivation=False
    f  = projection(y)

`conv(z; R, b)` аффинна (точнее линейна) по (R, b), `skip(z; W)` линейна по W, всё
после суммы блока -- замороженные слои. Так как зависимость от w всюду линейна,
якобиан J не зависит от точки приложения: JV ниже -- точная производная в w0 (не
приближение и не односторонняя разность). При этом f(x, w0 + v) != f(x, w0) + J v
для конечного v -- нелинейный замороженный хвост даёт нелинейный член; точным
является именно касательное направление (limit_{eps->0} (f(w0+eps v) - f(w0))/eps = J v),

    J v = P'(a0) . C'(s0) . ( conv_tangent(z; v_R, v_b) + transform(skip_tangent(z; v_W)) )

где conv_tangent/skip_tangent -- обычные forward-проходы соответствующих модулей с
касательными весами (градиенты не строятся вовсе), а fwAD применяется только к
замороженному хвосту (ChannelMLP + projection: чистые Linear/gelu).

Смысл режима -- не «спасение» от fwAD (fwAD на реальном весе проверен и точен:
float32 CPU 1e-6..3e-5, float64 3e-16), а независимая реализация той же производной:
здесь спектральный conv вообще не дифференцируется AD, поэтому результат не зависит
от версии torch и устройства. Используется как ``--jv-mode affine``; по умолчанию
включён ``--jv-mode forward`` (luno_torch.jacobian, torch.func.jvp).
"""

from __future__ import annotations

import math

import torch
from torch.func import functional_call


class AffineLastBlockError(RuntimeError):
    """Аффинная декомпозиция последнего блока не применима к этой модели."""


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = max(a.abs().max().item(), b.abs().max().item(), 1e-30)
    return (a - b).abs().max().item() / denom


class AffineLastBlock:
    """Контекст аффинного J*v: знает модули последнего блока и способ их запустить.

    Параметры
    ----------
    core:
        FNO-ядро (то, что стоит между лифтингом и проекцией).
    projection:
        Внешний модуль проекции (применяется после ядра).
    keys:
        Результат :func:`luno_torch.adapter.discover_last_block_keys` для ``core``.
    dtype:
        Рабочая точность (float32 по умолчанию).
    """

    def __init__(
        self,
        core: torch.nn.Module,
        projection: torch.nn.Module,
        keys: dict,
        dtype: torch.dtype = torch.float32,
    ):
        self.core = core
        self.projection = projection
        self.nl = int(keys["nl"])
        self.dtype = dtype
        self.R_key, self.W_key, self.b_key = keys["R"], keys["W"], keys["b"]

        blocks = core.fno_blocks
        self.blocks = blocks
        self.conv = self._submodule(core, f"fno_blocks.convs.{self.nl}", "spectral conv")
        self.skip = self._submodule(core, f"fno_blocks.fno_skips.{self.nl}", "fno skip")
        self.cmlp = self._submodule(
            core, f"fno_blocks.channel_mlp.{self.nl}", "channel_mlp (use_channel_mlp?)"
        )
        self.cmlp_skip = self._submodule(
            core, f"fno_blocks.channel_mlp_skips.{self.nl}",
            "channel_mlp_skip (channel_mlp_skip?)",
        )

        # Ключи state_dict относительно соответствующего модуля.
        conv_path = f"fno_blocks.convs.{self.nl}."
        skip_path = f"fno_blocks.fno_skips.{self.nl}."
        self._R_key = self.R_key[len(conv_path):]
        self._b_key = self.b_key[len(conv_path):] if self.b_key is not None else None
        self._W_key = self.W_key[len(skip_path):] if self.W_key is not None else None

        # Замороженный хвост *внутри* core между выходом последнего блока и выходом
        # core. Для neuralop FNO это core.projection (плюс unpad доменного padding,
        # если он есть). Сам lifting не входит: он применяется до блоков и от w
        # не зависит. Если core -- блок-only (собственного projection нет), хвост
        # пустой.
        self.core_projection = getattr(core, "projection", None)
        self.domain_padding = getattr(core, "domain_padding", None)

        sd = core.state_dict()
        self.R_shape = self._shape_of(sd, self.R_key, "спектральный вес R")
        self.W_shape = self._shape_of(sd, self.W_key, "skip-вес W")
        self.b_shape = self._shape_of(sd, self.b_key, "bias b")
        self.n_r = math.prod(self.R_shape) if self.R_shape else 0
        self.n_w = math.prod(self.W_shape) if self.W_shape else 0
        self.n_b = math.prod(self.b_shape) if self.b_shape is not None else 0

        self._validate()

    @staticmethod
    def _shape_of(sd, key, what: str):
        """Форма параметра по state_dict-ключу; ``None`` -- если параметра нет."""
        if key is None:
            return None
        if key not in sd:
            raise AffineLastBlockError(
                f"Ключ '{key}' ({what}) отсутствует в state_dict ядра -- структура "
                "последнего блока не совпадает с ожидаемой (fno_skip/channel_mlp_skip)."
            )
        return tuple(sd[key].shape)

    @staticmethod
    def _submodule(core: torch.nn.Module, path: str, what: str):
        """Достаёт подмодуль или падает с понятным сообщением (fno_skip=None и т.п.)."""
        try:
            return core.get_submodule(path)
        except AttributeError as exc:
            raise AffineLastBlockError(
                f"Не найден модуль '{path}' ({what}). Проверьте конфигурацию блока: "
                "при fno_skip/channel_mlp_skip=None соответствующие ветки отсутствуют."
            ) from exc

    def _validate(self) -> None:
        """Проверяет, что структура блока совпадает с аффинной декомпозицией."""
        blocks = self.blocks
        if self.nl != blocks.n_layers - 1:
            raise AffineLastBlockError(
                f"Ожидался последний блок (nl={blocks.n_layers - 1}), получен nl={self.nl}."
            )
        if blocks.preactivation:
            raise AffineLastBlockError(
                "preactivation=True: структура блока отличается от postactivation."
            )
        if blocks.norm is not None:
            norm = blocks.norm
            name = type(norm[0]).__name__ if hasattr(norm, "__getitem__") else type(norm).__name__
            raise AffineLastBlockError(
                f"norm={name}: после суммы блока стоит нормализация, "
                "аффинная декомпозиция не применима."
            )
        if blocks.stabilizer is not None:
            raise AffineLastBlockError(
                f"stabilizer={blocks.stabilizer}: активация применяется до conv."
            )
        if blocks.fno_skips is None:
            raise AffineLastBlockError("fno_skip=None: в блоке нет skip-ветки.")
        if not blocks.use_channel_mlp:
            raise AffineLastBlockError(
                "use_channel_mlp=False: хвост блока отличается от channel_mlp."
            )
        if self.conv.resolution_scaling_factor is not None:
            raise AffineLastBlockError(
                "resolution_scaling_factor != None: transform(block) меняет форму, "
                "аффинная декомпозиция не проверена для этого случая."
            )
        if self._R_key not in ("weight", "weight.tensor"):
            raise AffineLastBlockError(
                f"Ожидался плотный спектральный вес 'weight' (или 'weight.tensor' для "
                f"tltorch-обёртки DenseTensor), получен '{self._R_key}'. "
                "Нетехнизированные (tucker/cp/tt) веса не аффинны по факторам и "
                "не поддерживаются."
            )
        weight = self.dense_weight()
        if tuple(weight.shape) != self.R_shape:
            raise AffineLastBlockError(
                f"Форма спектрального веса {tuple(weight.shape)} не совпадает с "
                f"формой из state_dict {self.R_shape}."
            )
        if not weight.is_complex():
            raise AffineLastBlockError(
                f"Спектральный вес должен быть комплексным, получен {weight.dtype}."
            )
        if getattr(self.conv, "separable", False):
            raise AffineLastBlockError(
                "separable=True: спектральный вес имеет другую структуру, "
                "аффинная декомпозиция не проверена для separable-conv."
            )
        skip_bias = getattr(getattr(self.skip, "conv", None), "bias", None)
        if skip_bias is not None:
            raise AffineLastBlockError(
                "У skip-слоя есть bias: касательная по W потребовала бы его обнуления."
            )

    def dense_weight(self) -> torch.Tensor:
        """Плотный спектральный вес последнего conv.

        neuralop хранит его двумя способами в зависимости от версии: как обычный
        ``Parameter`` ``conv.weight`` либо как tltorch-обёртку ``conv.weight.tensor``
        (FactorizedTensor с factorization="Dense", т.е. без разложения). Оба варианта
        -- один и тот же плотный комплексный тензор, поэтому оба аффинны по нему.
        Настоящие факторизации (tucker/cp/tt) дают набор отдельных факторов и сюда
        не попадают: они отвергаются в :meth:`_validate`.
        """
        obj = self.conv
        for part in self._R_key.split("."):
            obj = getattr(obj, part)
        if not isinstance(obj, torch.Tensor):
            raise AffineLastBlockError(
                f"'{self._R_key}' не является тензором (получено {type(obj).__name__}); "
                "спектральный вес факторизован -- аффинная декомпозиция не применима."
            )
        return obj

    def bind(self, x: torch.Tensor, model_fn, w0: torch.Tensor) -> "BoundAffine":
        """Готовит состояние (z, s0, a0) для одного входа ``x``; кэшируется вызывающим."""
        return BoundAffine(self, x, model_fn, w0)

    def core_tail(self, a: torch.Tensor) -> torch.Tensor:
        """Замороженный хвост от выхода последнего блока до выхода core.

        Для real neuralop FNO это ``projection`` (ChannelMLP 60->60->60), т.е.
        параметры, которых нет в ``w0``, но tangent обязан пройти через них.
        При ``domain_padding`` сначала снимается padding.
        """
        if self.domain_padding is not None:
            a = self.domain_padding.unpad(a)
        if self.core_projection is None:
            return a
        return self.core_projection(a)

    def tail(self, a: torch.Tensor) -> torch.Tensor:
        """Полный замороженный хвост после суммы блока: channel_mlp -> core -> внешняя проекция."""
        return self.projection(self.core_tail(self.cmlp(a)))


class BoundAffine:
    """Состояние аффинного якобиана, привязанное к одному входу ``x``."""

    def __init__(
        self,
        parent: AffineLastBlock,
        x: torch.Tensor,
        model_fn,
        w0: torch.Tensor,
    ):
        self.p = parent
        self.z = None      # вход последнего блока
        self.s0 = None     # сумма блока (вход channel_mlp последнего блока)
        self.blk0 = None   # выход последнего блока (вход core-проекции)
        self.a0 = None     # выход ядра core
        self.cskip = None  # channel_mlp_skip(z), не зависит от w
        self.fx = None     # f(x, w0)
        self._capture(x, model_fn, w0)

    def _capture(self, x: torch.Tensor, model_fn, w0: torch.Tensor) -> None:
        """Один reference-проход с хуками: z, s0, blk0, a0, cskip и f(x, w0).

        ``blk0`` ловится на ``fno_blocks`` (последний вызов), ``a0`` -- на выходе
        всего ``core``. Для real neuralop FNO между ними лежит собственная
        ``core.projection``, поэтому её нельзя пропускать.
        """
        p = self.p
        store = {}

        def _pre_conv(mod, args):
            store["z"] = args[0].detach()

        def _pre_cmlp(mod, args):
            store["s0"] = args[0].detach()

        def _post_cmlp_skip(mod, args, out):
            store["cskip"] = out.detach()

        def _post_blocks(mod, args, out):
            store["blk0"] = out.detach()

        def _post_core(mod, args, out):
            store["a0"] = out.detach()

        handles = [
            p.conv.register_forward_pre_hook(_pre_conv),
            p.cmlp.register_forward_pre_hook(_pre_cmlp),
            p.cmlp_skip.register_forward_hook(_post_cmlp_skip),
            p.blocks.register_forward_hook(_post_blocks),
            p.core.register_forward_hook(_post_core),
        ]
        try:
            with torch.no_grad():
                out = model_fn(x, w0)
        finally:
            for h in handles:
                h.remove()

        missing = [k for k in ("z", "s0", "cskip", "blk0", "a0") if k not in store]
        if missing:
            raise AffineLastBlockError(
                f"Не удалось захватить {missing}: хуки не сработали (структура ядра отличается "
                "от ожидаемой)."
            )

        self.z = store["z"]
        self.s0 = store["s0"]
        self.blk0 = store["blk0"]
        self.a0 = store["a0"]
        self.cskip = store["cskip"]
        self.fx = out.detach()

        # Гейт на каждом сегменте хвоста: блок обязан быть ровно
        # channel_mlp(conv + skip) + channel_mlp_skip, core -- ровно последним
        # блоком плюс собственный projection, и всё это -- внешняя проекция.
        with torch.no_grad():
            rel_block = _rel_err(p.cmlp(self.s0) + self.cskip, self.blk0)
            rel_core = _rel_err(p.core_tail(self.blk0), self.a0)
            rel_model = _rel_err(p.projection(self.a0), self.fx)
        if max(rel_block, rel_core, rel_model) > 1e-5:
            raise AffineLastBlockError(
                "Аффинная декомпозиция не воспроизводит модель "
                f"(блок rel={rel_block:.3e}, core rel={rel_core:.3e}, "
                f"модель rel={rel_model:.3e})."
            )

    def tangent_sum(self, v: torch.Tensor) -> torch.Tensor:
        """Касательная суммы блока a = conv(z;R,b) + transform(skip(z;W)) по направлению v.

        ВАЖНО про bias: ``b`` тоже входит в ``w0``, поэтому он подменяется на
        ``v_b``, и ``functional_call`` возвращает ровно ``L(z) @ R_t + b_t`` --
        вычитать frozen ``conv.bias`` НЕ нужно (и нельзя: это портит результат,
        rel ~0.5..0.8). Утечка bias возможна только если забыть подставить ``b_t``.
        Проверено на float64: совпадение с fwAD побитовое (rel = 0).
        """
        p = self.p
        v = v.to(dtype=self.a0.dtype)
        n_r, n_w = p.n_r, p.n_w
        R_t = torch.complex(
            v[:n_r].reshape(p.R_shape), v[n_r: 2 * n_r].reshape(p.R_shape)
        )
        W_t = v[2 * n_r: 2 * n_r + n_w].reshape(p.W_shape)
        params = {p._R_key: R_t}
        if p._b_key is not None:
            params[p._b_key] = v[2 * n_r + n_w:].reshape(p.b_shape)
        with torch.no_grad():
            t_conv = functional_call(p.conv, params, (self.z,))
            t_skip = functional_call(p.skip, {p._W_key: W_t}, (self.z,))
            return t_conv + p.conv.transform(t_skip, output_shape=None)

    def jv(self, v: torch.Tensor) -> torch.Tensor:
        """J v: касательная выхода модели по направлению v (вектор длины d).

        Касательная проходит весь замороженный хвост: channel_mlp -> core.projection
        -> внешняя проекция. Параметры ``core.projection`` не входят в ``w0``, но их
        касательные существенны.
        """
        p = self.p
        t_sum = self.tangent_sum(v)
        with torch.no_grad():
            t_blk = torch.func.jvp(p.cmlp, (self.s0,), (t_sum,))[1]
            t_a = torch.func.jvp(p.core_tail, (self.blk0,), (t_blk,))[1]
            t_out = torch.func.jvp(p.projection, (self.a0,), (t_a,))[1]
        return t_out.reshape(-1)

    def jmatmul(self, weights: torch.Tensor) -> torch.Tensor:
        """J v (вектор) или J @ W при W формы (d, k).

        Колонки здесь НЕ батчатся через ``vmap``, в отличие от прямого режима
        (``LastFNOBlockWeightJacobian._matmul_forward``): ``SpectralConv.forward``
        пишет в ``out_fft`` индексированным присваиванием
        ``out_fft[slices_x] = self._contract(...)``, а vmap запрещает in-place
        арифметику, когда один операнд батчится, а ``self`` -- нет
        ("inplace arithmetic ... other is vmapped over but self is not").
        Поэтому каждая колонка -- свой jvp, последовательно; лишний параметр
        батчинга не выставляется, чтобы не обещать то, чего здесь нет.

        Пик памяти при этом всё равно не зависит от rank: результат пишется в
        заранее выделенный буфер (m, k), а не собирается ``torch.stack`` из k
        готовых (m,) тензоров (это держало бы (m, k) дважды).
        """
        if weights.dim() == 1:
            return self.jv(weights)
        k = weights.shape[-1]
        m = self.fx.numel()
        if k == 0:
            return torch.zeros(m, 0, dtype=self.a0.dtype, device=self.a0.device)
        out = torch.empty(m, k, dtype=self.a0.dtype, device=self.a0.device)
        for j in range(k):
            out[:, j] = self.jv(weights[:, j])
        return out
