"""AUSE эвристических оценок ошибки на тесте всех задач (без LUNO).

Зачем
-----
Проверка, предсказуемо ли расположение ошибки внутри поля вообще. Вместо std
LUNO ранжируем пиксели простыми функциями ПРЕДСКАЗАННОГО поля ``ŷ`` (target не
используется) и считаем ту же нормированную AUSE, что в
``run_luno_ggn_calibrate.evaluate_luno``:

* ``grad`` -- ``sqrt(sum_d (∂_d ŷ)²)`` по пространственным осям (центральные
  разности, ``torch.gradient``): фронты и резкие перепады;
* ``lap``  -- ``|sum_d ∂²_d ŷ|``, 5-точечный шаблон с повтором краевого значения:
  мелкие структуры, вихри;
* ``dt``   -- ``|ŷ_t - ŷ_{t-1}|`` (для ``t = 0`` -- ``|ŷ_1 - ŷ_0|``): где поле
  быстро меняется. Только для выходов с осью времени и ``T > 1``.

Если эвристика даёт AUSE заметно ниже, чем std LUNO, ошибка внутри поля
предсказуема, а LUNO эту информацию не ловит. Если и эвристики около 1 --
расположение ошибки почти шумовое.

Метрики (тест, все -- по каналу, ``{tt}`` -- момент времени; среднее по сэмплам):

* ``nrmse_{канал}`` и ``nrmse_{канал}_t{tt}`` -- ``RMSE / RMS(target)`` на
  срезе (как ``nrmse`` в muno.utils.metrics);
* ``ause_{эвристика}_{канал}_t{tt}`` и ``ause_{эвристика}_{канал}`` (среднее
  по времени) -- нормированная AUSE (0 -- идеал, ~1 -- случайный порядок);
* ``err_top10_{канал}_t{tt}`` и ``err_top10_{канал}`` -- доля суммы ``err²``
  в 10 % пикселей с наибольшей ошибкой. Показывает, есть ли что ранжировать.
  Ориентир -- НЕ 0.1: у гауссовой ошибки с одинаковой по полю дисперсией
  доля ~0.44 (хвост chi²₁), и ранжировать там нечего, AUSE любого σ ~1. При
  лог-нормальном разбросе истинной σ по полю в ~2 / 4 / 8 раз доля ~0.53 /
  0.72 / 0.88, а AUSE ИДЕАЛЬНОГО σ (истинной) ~0.52 / 0.24 / 0.10.

Аргументы данных и чекпойнтов -- те же, что у ``run_luno_ggn_calibrate.py``;
блоки ``luno:`` конфига игнорируются. Пример::

    python muno/uncertainty/run_heuristic_ause.py \\
        --config experiments/scripts/configs/luno_turbo_inf.yaml \\
        --core-checkpoint ... --lift-checkpoint-dir ... --proj-checkpoint-dir ... \\
        --in-normalizers ... --out-normalizers ...
"""

from __future__ import annotations

import argparse
import gc
import math
import pickle
import traceback
from datetime import datetime

import torch

import run_luno_ggn_calibrate as run
from metrics_uq import normalized_ause
from luno_torch.progress import set_enabled, tqdm

HEURISTICS = ("grad", "lap", "dt")
TOP_FRAC = 0.1


def parse_args():
    p = argparse.ArgumentParser(
        description="AUSE эвристик (grad, lap, dt) и NRMSE на тесте всех задач."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--tasks", default="all",
                   help="'all' или имена задач через запятую.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--core-checkpoint", required=True)
    p.add_argument("--lift-checkpoint-dir", required=True)
    p.add_argument("--proj-checkpoint-dir", required=True)
    p.add_argument("--in-normalizers", required=True)
    p.add_argument("--out-normalizers", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Сколько тестовых сэмплов на задачу; 0 -- все.")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Эвристики
# ----------------------------------------------------------------------------

def _layout(pred: torch.Tensor) -> tuple[int, int, tuple[int, ...]]:
    """``(C, T, spatial_dims)`` для ``(1, C, T, *space)`` или ``(1, C, *space)``.

    Раскладка та же, что у ``run.ause_by_channel_time``: ось времени есть, если
    ``ndim >= 5``.
    """
    if pred.shape[0] != 1:
        raise ValueError(f"ожидался батч из одного сэмпла, получено {pred.shape[0]}")
    has_time = pred.dim() >= 5
    n_t = pred.shape[2] if has_time else 1
    first = 3 if has_time else 2
    return pred.shape[1], n_t, tuple(range(first, pred.dim()))


def _second_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    """``x[i+1] - 2 x[i] + x[i-1]`` с повтором краевого значения."""
    n = x.shape[dim]
    if n < 3:
        return torch.zeros_like(x)
    xp = torch.cat([x.narrow(dim, 0, 1), x, x.narrow(dim, n - 1, 1)], dim=dim)
    return xp.narrow(dim, 2, n) - 2.0 * x + xp.narrow(dim, 0, n)


def heuristic_maps(pred: torch.Tensor) -> dict[str, torch.Tensor]:
    """Карты эвристик той же формы, что ``pred``; ``dt`` -- только при ``T > 1``."""
    _, n_t, spatial = _layout(pred)
    pred = pred.double()
    maps = {}
    grads = torch.gradient(pred, dim=spatial)
    maps["grad"] = torch.sqrt(sum(g ** 2 for g in grads))
    maps["lap"] = sum(_second_diff(pred, d) for d in spatial).abs()
    if n_t > 1:
        d = (pred[:, :, 1:] - pred[:, :, :-1]).abs()  # (1, C, T-1, ...)
        maps["dt"] = torch.cat([d[:, :, :1], d], dim=2)
    return maps


def top_share(err2: torch.Tensor, frac: float = TOP_FRAC) -> torch.Tensor:
    """Доля суммы ``err2`` в ``frac`` пикселях с наибольшей ошибкой, по строкам ``(G, n)``."""
    k = max(1, math.ceil(frac * err2.shape[1]))
    top = torch.topk(err2, k, dim=1).values.sum(dim=1)
    total = err2.sum(dim=1)
    return torch.where(total > 0, top / total, torch.full_like(total, float("nan")))


def sample_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Метрики одного сэмпла: тензоры ``(C, T)`` и ``nrmse_ch`` формы ``(C,)``."""
    n_ch, n_t, _ = _layout(pred)
    g = n_ch * n_t
    pred, target = pred.double(), target.double()
    err2 = ((pred - target) ** 2).reshape(g, -1)
    y2 = (target ** 2).reshape(g, -1)
    eps = 1e-7
    out = {
        "nrmse": (err2.mean(1).sqrt() / y2.mean(1).sqrt().clamp_min(eps)).reshape(n_ch, n_t),
        "nrmse_ch": (err2.reshape(n_ch, -1).mean(1).sqrt()
                     / y2.reshape(n_ch, -1).mean(1).sqrt().clamp_min(eps)),
        "err_top10": top_share(err2).reshape(n_ch, n_t),
    }
    for name, h in heuristic_maps(pred).items():
        out[f"ause_{name}"] = normalized_ause(h.reshape(g, -1), err2).reshape(n_ch, n_t)
    return out


def aggregate(per_sample: list[dict], keys: list[str]) -> dict:
    """Среднее по сэмплам (nan пропускаются) -> плоский словарь метрик."""
    metrics = {}
    first = per_sample[0]
    for name, value in first.items():
        stacked = torch.stack([s[name] for s in per_sample]).cpu()  # (S, C[, T])
        mean = stacked.nanmean(dim=0)
        if name == "nrmse_ch":
            for c, key in enumerate(keys):
                metrics[f"nrmse_{key}"] = float(mean[c])
            continue
        for c, key in enumerate(keys):
            per_t = mean[c].tolist()
            for t, v in enumerate(per_t):
                metrics[f"{name}_{key}_t{t:02d}"] = v
            if name != "nrmse":
                metrics[f"{name}_{key}"] = run.mean_finite(per_t)
    metrics["samples"] = len(per_sample)
    return metrics


# ----------------------------------------------------------------------------
# Прогон
# ----------------------------------------------------------------------------

def predict_raw(lifting, core, projection, x, out_normalizer, out_shape):
    """Предсказание в raw-пространстве: projection(core(lifting(x))) + денормализация."""
    y_norm = projection(core(lifting(x.to(run.DTYPE))))
    zeros = torch.zeros_like(y_norm.reshape(-1))
    mean_raw, _ = run.raw_from_normalized(y_norm.reshape(-1), zeros, out_normalizer, out_shape)
    return mean_raw.reshape(out_shape)


def evaluate_task(args, cfg, task_idx, model_config, core, asset_cache, name):
    # in_distr=False: val не режется на калибровку, нужен только test.
    pipeline = run.build_task_pipeline(args, cfg, task_idx, model_config, core,
                                       asset_cache, in_distr=False)
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    lifting = pipeline["lifting"].to(device=device).to(dtype=run.DTYPE).eval()
    projection = pipeline["projection"].to(device=device).to(dtype=run.DTYPE).eval()
    data_processor, _ = run.build_data_processor(args, pipeline, task_idx)
    out_normalizer = data_processor.out_normalizer

    loader = pipeline["test_loader"]
    n_total = len(loader.dataset)
    n = n_total if args.max_samples <= 0 else min(args.max_samples, n_total)
    per_sample = []
    n_ch = None
    with torch.no_grad():
        for sample in tqdm(run.iter_samples(loader, n), total=n,
                           desc=f"[heur:{name}] test", unit="smp", leave=False):
            sample = run.preprocess_sample(data_processor, sample)
            target = sample[0]["y"]
            pred = predict_raw(lifting, core, projection, sample[0]["x"],
                               out_normalizer, target.shape)
            per_sample.append(sample_metrics(pred, target))
            n_ch = target.shape[1]
    if not per_sample:
        raise SystemExit(f"[heur:{name}] в test нет ни одного сэмпла.")

    channel_names = (cfg.get("adapter") or {}).get("variable_names")
    keys = run.channel_key_names(channel_names, n_ch)
    metrics = aggregate(per_sample, keys)

    names = [h for h in HEURISTICS if f"ause_{h}_{keys[0]}" in metrics]
    print(f"[heur:{name}] {metrics['samples']}/{n_total} тестовых сэмплов; "
          f"AUSE -- среднее по времени (0 идеал, 1 случайно), top10 -- доля err² "
          f"в 10% худших пикселей (однородный гаусс: 0.44):")
    for key in keys:
        ause = " ".join(f"{h}={metrics[f'ause_{h}_{key}']:.3f}" for h in names)
        print(f"[heur:{name}]   {key}: nrmse={metrics[f'nrmse_{key}']:.4f} | "
              f"AUSE {ause} | top10={metrics[f'err_top10_{key}']:.3f}")
    return metrics, n_total


def main():
    args = parse_args()
    set_enabled(not args.no_progress)
    if args.dtype == "float64":
        run.DTYPE, run.CDTYPE = torch.float64, torch.complex128

    config_path = run.rmi.resolve_path(args.config)
    config = run.rmi.load_yaml_config(config_path)
    model_config = config.get("model", {})
    selected = run.select_tasks(config.get("tasks", []), args.tasks)
    print(f"[config] {config_path}")
    print(f"[tasks] {[run.task_name(cfg, i) for i, cfg in selected]}")

    run._install_pickle_compat()
    core = run.rmi.load_from_dir(args.core_checkpoint, {"map_location": "cpu"})[0]
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    core = core.to(device=device)
    # Как в run_luno_ggn_calibrate.main: не кастить core целиком (теряется
    # мнимая часть спектральных весов), а выставить точность поэлементно.
    for t in list(core.parameters()) + list(core.buffers()):
        t.data = t.data.to(dtype=run.CDTYPE if t.is_complex() else run.DTYPE)
    core.eval()

    output_root = (args.output_root if args.output_root is not None
                   else config.get("output", {}).get("root", "runs"))
    prefix = args.run_name if args.run_name is not None else config_path.stem
    output_dir = (run.rmi.resolve_path(output_root)
                  / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_heuristic")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir: {output_dir}")

    asset_cache = {}
    done, failed = [], []
    for position, cfg in selected:
        name = run.task_name(cfg, position)
        task_idx = run.resolve_task_idx(cfg, position)
        task_dir = output_dir / name
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 70}\n[task:{name}] task_idx={task_idx}, "
              f"{cfg.get('task_type')}\n{'=' * 70}")
        result = {"config": config_path.stem, "task_name": name,
                  "task_idx": task_idx, "task_type": cfg.get("task_type"),
                  "heuristics": HEURISTICS, "top_frac": TOP_FRAC}
        try:
            metrics, n_total = evaluate_task(args, cfg, task_idx, model_config,
                                             core, asset_cache, name)
            result.update(status="ok", test=metrics, test_samples_total=n_total)
            done.append(name)
        except Exception as exc:
            print(f"[task:{name}] ОШИБКА: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc())
            failed.append(name)
        finally:
            with open(task_dir / "heuristic_results.pkl", "wb") as f:
                pickle.dump(result, f)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nГотово: {len(done)} задач посчитано, упали {failed}. "
          f"Результаты: {output_dir}")


if __name__ == "__main__":
    main()
