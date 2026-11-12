import torch

from muno.utils.metrics import check_same_shape, check_batched_channel_tensor


def _validate_inputs(pred, band, target):
    check_same_shape(pred, band)
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)


def covered_mask(pred, band, target, k=1.0):
    _validate_inputs(pred, band, target)
    error = torch.abs(pred - target)
    return error <= k * band


def coverage_rate(pred, band, target, k=1.0):
    mask = covered_mask(pred, band, target, k=k)
    return torch.mean(mask.to(pred.dtype))


def normalized_avg_bandwidth(pred, band, target, k=1.0, eps=1e-7):
    _validate_inputs(pred, band, target)

    field_scale = torch.sqrt(torch.mean(target ** 2)).clamp_min(eps)

    mask = covered_mask(pred, band, target, k=k)

    mean_bw_all = torch.mean(band)

    if torch.any(mask):
        mean_bw_covered = torch.mean(band[mask])
    else:
        mean_bw_covered = torch.full((), float("nan"), dtype=band.dtype, device=band.device)

    if torch.any(~mask):
        mean_bw_missed = torch.mean(band[~mask])
    else:
        mean_bw_missed = torch.full((), float("nan"), dtype=band.dtype, device=band.device)

    return {
        "n_avg_bw_all": float((mean_bw_all / field_scale).detach().cpu()),
        "n_avg_bw_covered": float((mean_bw_covered / field_scale).detach().cpu()),
        "n_avg_bw_missed": float((mean_bw_missed / field_scale).detach().cpu()),
    }


UQ_METRIC_REGISTRY = {
    "coverage_rate": coverage_rate,
    "normalized_avg_bandwidth": normalized_avg_bandwidth,
}


def compute_uq_metrics(pred, band, target, metric_configs=None):
    if metric_configs is None:
        metric_configs = [
            {"name": "coverage_rate"},
            {"name": "normalized_avg_bandwidth"},
        ]

    results = {}

    for metric_config in metric_configs:
        name = metric_config["name"]
        kwargs = {
            key: value
            for key, value in metric_config.items()
            if key != "name"
        }

        if name not in UQ_METRIC_REGISTRY:
            raise ValueError(
                f"Unknown physical metric '{name}'. "
                f"Available physical metrics: {list(UQ_METRIC_REGISTRY.keys())}"
            )

        value = UQ_METRIC_REGISTRY[name](pred, band, target, **kwargs)

        if isinstance(value, dict):
            results.update(value)
        else:
            results[name] = float(value.detach().cpu())

    return results

def normalized_ause(std: torch.Tensor, err2: torch.Tensor) -> torch.Tensor:
    """Нормированная AUSE (sparsification error) по строкам ``(G, n)``.

    Для каждой группы из ``n`` пикселей строятся две кривые «RMSE оставшихся
    пикселей после удаления доли ``f`` пикселей»:

    * по ``std`` -- удаляем самые неуверенные (по убыванию ``std``);
    * оракул -- удаляем пиксели с наибольшей реальной ошибкой.

    ``AUSE`` -- площадь между ними (``f`` от 0 до 1 с шагом ``1/n``). Нормировка
    на площадь между случайным удалением (RMSE не меняется -- горизонтальная
    линия на уровне полного RMSE) и оракулом делает метрику безразмерной::

        0  -- std ранжирует пиксели как сама ошибка (идеал);
        ~1 -- не лучше случайного порядка;
        >1 -- хуже случайного (std больше там, где ошибка меньше).

    Метрика зависит только от ПОРЯДКА ``std``: ни масштаб (prior_prec), ни
    околонулевые значения ``std`` на неё не влияют, фильтр не нужен. Группа с
    неконечным ``std`` или с нулевым знаменателем (все ошибки равны) даёт nan.

    ``std`` и ``err2`` (квадрат ошибки) -- одной формы ``(G, n)``; возвращает
    ``(G,)`` в float64. Стоимость -- две сортировки и два ``cumsum`` по строке.
    """
    if std.shape != err2.shape or std.dim() != 2:
        raise ValueError(
            f"normalized_ause: ожидались std и err2 формы (G, n), получено "
            f"{tuple(std.shape)} и {tuple(err2.shape)}"
        )
    std = std.double()
    err2 = err2.double()
    n = std.shape[1]
    counts = torch.arange(n, 0, -1, dtype=torch.float64, device=std.device)

    def remaining_rmse(order_key):
        # Сортировка по возрастанию ключа: удаление k худших оставляет первые
        # n - k элементов, их RMSE -- из префиксной суммы.
        idx = torch.argsort(order_key, dim=1)
        cs = torch.cumsum(torch.gather(err2, 1, idx), dim=1)  # (G, n)
        # r = n, n-1, ..., 1 оставшихся пикселей
        return torch.sqrt(cs.flip(1) / counts)

    by_std = remaining_rmse(std)
    oracle = remaining_rmse(err2)
    full = by_std[:, :1]
    ause = (by_std - oracle).mean(dim=1)
    denom = (full - oracle).mean(dim=1)
    tiny = torch.finfo(torch.float64).tiny
    ratio = torch.where(denom > tiny, ause / denom.clamp_min(tiny),
                        torch.full_like(ause, float("nan")))
    bad = ~torch.isfinite(std).all(dim=1) | ~torch.isfinite(err2).all(dim=1)
    return torch.where(bad, torch.full_like(ratio, float("nan")), ratio)


def reliability_bin_sums(
    std: torch.Tensor, err2: torch.Tensor, n_bins: int = 10
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Суммы ``std²`` и ``err²`` по бинам предсказанного std, для строк ``(G, n)``.

    Внутри каждой группы пиксели сортируются по ``std`` и делятся на ``n_bins``
    бинов с одинаковым числом точек (бин ``b`` -- ``b``-я доля самых уверенных).
    Бины строятся по ``std``, а не по ошибке: отбор по исходу смещает картину
    даже у идеально откалиброванной модели (регрессия к среднему).

    Возвращает ``(sum_std2, sum_err2, count)`` формы ``(G, n_bins)`` в float64.
    Суммы аддитивны, поэтому их можно копить по сэмплам и только потом брать
    ``RMV = sqrt(sum_std2 / count)``, ``RMSE = sqrt(sum_err2 / count)``. Группа
    с неконечным ``std`` или ``err2`` получает nan во всех бинах.
    """
    if std.shape != err2.shape or std.dim() != 2:
        raise ValueError(
            f"reliability_bin_sums: ожидались std и err2 формы (G, n), получено "
            f"{tuple(std.shape)} и {tuple(err2.shape)}"
        )
    g, n = std.shape
    if n < n_bins:
        raise ValueError(f"reliability_bin_sums: {n} пикселей меньше, чем {n_bins} бинов")
    std = std.double()
    err2 = err2.double()
    idx = torch.argsort(std, dim=1)
    s_std2 = torch.gather(std, 1, idx) ** 2
    s_err2 = torch.gather(err2, 1, idx)
    bin_of = (torch.arange(n, device=std.device) * n_bins) // n  # равные доли
    sum_std2 = torch.zeros(g, n_bins, dtype=torch.float64, device=std.device)
    sum_err2 = torch.zeros_like(sum_std2)
    sum_std2.index_add_(1, bin_of, s_std2)
    sum_err2.index_add_(1, bin_of, s_err2)
    count = torch.bincount(bin_of, minlength=n_bins).to(sum_std2).expand(g, -1).clone()
    bad = ~torch.isfinite(std).all(dim=1) | ~torch.isfinite(err2).all(dim=1)
    sum_std2[bad] = float("nan")
    sum_err2[bad] = float("nan")
    return sum_std2, sum_err2, count


def log_rmse_over_rmv(sum_err2: torch.Tensor, sum_std2: torch.Tensor) -> torch.Tensor:
    """``log(RMSE / RMV)`` по накопленным суммам одного и того же набора точек.

    Счётчики сокращаются: ``RMSE / RMV = sqrt(sum_err2 / sum_std2)``. 0 --
    идеально, ``> 0`` -- std занижен (самоуверенность), ``< 0`` -- завышен;
    ``±0.69`` -- ошибка в 2 раза. ``std = 0`` при ненулевой ошибке даёт
    ``+inf``, ``0/0`` -- nan.
    """
    with torch.no_grad():
        return 0.5 * torch.log(sum_err2 / sum_std2)
