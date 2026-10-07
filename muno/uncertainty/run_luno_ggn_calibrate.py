"""
Полный LUNO-пайплайн для модели Muno (однозадачная упрощённая версия).

Одна задача, один лифтинг и одна проекция (берутся из переданных файлов),
внутренние мультизадачные адаптеры в вычислениях НЕ участвуют. Тюнимые
параметры -- веса последнего FNO-блока общего core (R, W, b); лифтинг и
проекция фиксированы.

Пайплайн:
    1) загрузка модели (torch.load + dill через rmi.load_from_dir) и данных
       (rmi.loadData; val-сплит 0.1 -> данные для калибровки);
    2) сплит модели: last FNO block core -> w0 = [R.real, R.imag, W, b];
       печать слоёв, выбранных в качестве последних;
    3) low-rank GGN последнего фурье-слоя на train-лоадере; при заданном
       --ggn-checkpoint шаг пропускается и LowRankTerms читается из pickle
       (файла нет -- предупреждение и расчёт как обычно);
    4) калибровка scalar prior_prec на всех сэмплах calib-части val
       (полный log10-грид, пробы Хатчинсона переиспользуются по кандидатам);
    5) финальные метрики NLL (+RMSE): val -- весь остаток val, test --
       ограничен --max-eval-samples;
    6) сохранение GGN (LowRankTerms, если он считался, а не загружен) и метрик
       в pkl в output-папке эксперимента (структура папок как в
       run_multiphysics_inference.py).

Запуск (флаги -- как в run_multiphysics_inference.py):
    python muno/uncertainty/run_luno_ggn_calibrate.py \
        --config experiments/scripts/configs/pdebench_multiphysics.yaml \
        --core-checkpoint <path>/core.pt \
        --lift-checkpoint-dir <dir> \
        --proj-checkpoint-dir <dir> \
        --in-normalizers <dir> --out-normalizers <dir> \
        --output-root <root> --run-name <name>

По умолчанию (``--jv-mode forward``) Jv/J@W считаются forward-mode AD через
torch.func.jvp: одна касательная на всё направление, один прямой проход сети, память
как у одного графа; колонки J@W считаются батчем по --fwd-batch. Альтернатива
(``--jv-mode affine``, luno_torch.affine) разбирает последний блок на спектральный
conv + skip (касательные веса, обычный forward без AD) и замороженный хвост (jvp) --
тоже точно, но fwAD не дифференцирует спектральный conv вовсе. Оба режима не
материализуют строки J и заметно быстрее прежнего chunked reverse; обратные проходы
в обоих режимах остаются для J^T u и diag(JJ^T).

На старте Jv выбранного режима сверяется с reverse-строками на одном батче: при
расхождении скрипт падает с явной ошибкой. Допуск разный для float32 и float64
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_SCRIPTS = PROJECT_ROOT / "experiments" / "scripts"
UNCERTAINTY_DIR = Path(__file__).resolve().parent

for _p in (str(PROJECT_ROOT), str(EXPERIMENTS_SCRIPTS), str(UNCERTAINTY_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# expandable_segments снижает фрагментацию CUDA-памяти (крупным аллокациям вроде
# rows = vjp_chunk*d проще найти место). Окружение должно быть выставлено ДО
# импорта torch / инициализации CUDA-аллокатора.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


import torch
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset

from muno.data.benchmarks.datasets import MultiPhysicsDataset
from muno.data.benchmarks.inspections import (
    _inference_time_index,
    canonical_image,
    save_image,
)
from muno.data.data.transforms.data_processors import DefaultDataProcessor
from muno.data.data.transforms.normalizers import MultiphysicsUnitGaussianNormalizer

from luno_torch.adapter import (
    TorchFNOWrapper,
    discover_last_block_keys,
    ref_state_dict,
    split_wrapper,
)
from luno_torch.affine import AffineLastBlock
from luno_torch.calibrate import chi_squared, nll_gaussian
from luno_torch.factory import create_luno_cov
from luno_torch.ggn import GGNMatvec, LowRankTerms, low_rank_ggn
from luno_torch.jacobian import (
    LastFNOBlockWeightJacobian,
    check_jacobian,
    default_check_tol,
    sample_congruence_probes,
    var_from_congruence_probes,
    var_of_congruence,
)
from luno_torch.progress import set_enabled, tqdm

import run_multiphysics_inference as rmi

from check_split import _resolve_key_owner

DTYPE = torch.float32
CDTYPE = torch.complex64

# Percentile levels for the per-sample sqrt(chi2) metric (percent over samples).
SQRT_CHI2_PERCENTILES = (5, 25, 50, 75, 95)


# ----------------------------------------------------------------------------
# Аргументы
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LUNO pipeline for a single-adapter Muno.")
    p.add_argument("--config", default="experiments/configs/pdebench_multiphysics_pretrain.yaml")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--core-checkpoint", default=None)
    p.add_argument("--lift-checkpoint-dir", default=None)
    p.add_argument("--proj-checkpoint-dir", default=None)
    p.add_argument("--in-normalizers", default=None)
    p.add_argument("--out-normalizers", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-root", default=None)
    # --- LUNO-specific ---
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                   help="Точность всех вычислений (float32 вдвое экономит память).")
    p.add_argument("--jv-mode", choices=["forward", "affine"], default="forward",
                   help="Режим прямых произведений Jv/J@W: forward-mode AD "
                        "(torch.func.jvp, по умолчанию) или аффинная декомпозиция "
                        "последнего блока (conv/skip через прямой forward, хвост "
                        "через jvp). Оба точны; выбирается по памяти/скорости.")
    p.add_argument("--fwd-batch", type=int, default=4,
                   help="Сколько колонок J@W считать одним батчем через vmap "
                        "(только для --jv-mode forward). Больше -- быстрее, но "
                        "кратно больше памяти.")
    p.add_argument("--check-probes", type=int, default=3,
                   help="Сколько строк J использовать в стартовом self-check.")
    p.add_argument("--check-tol", type=float, default=None,
                   help="Допуск стартового self-check Jv против reverse-строк. "
                        "По умолчанию разный для float32/float64 "
                        "(см. luno_torch.jacobian.default_check_tol).")
    p.add_argument("--vjp-chunk", type=int, default=16,
                   help="Сколько выходных строк J материализовать за раз в обратном "
                        "пути (diag JJ^T, transpose). На Jv/J@W не влияет: они "
                        "считаются прямым режимом (--jv-mode), без строк J.")
    p.add_argument("--max-rank", type=int, default=10, help="Low-rank rank для GGN.")
    p.add_argument("--ggn-checkpoint", type=str, default=None,
                   help="Путь к pickle с LowRankTerms (артефакт *_low_rank_terms.pkl "
                        "из прошлого прогона). Если файл найден и валиден -- расчёт "
                        "GGN пропускается целиком (сбор сэмплов, GGNMatvec, скетч), "
                        "артефакт в новый ранг не пишется, а --max-rank/--ggn-method/"
                        "--ggn-oversample/--max-num-samples игнорируются: авторитетен "
                        "ранг из чекпойнта. Если файла нет -- предупреждение и GGN "
                        "считается как обычно (сохранение артефакта тоже). Битый или "
                        "несовместимый (другая d) файл -- ошибка.")
    p.add_argument("--ggn-method", type=str, default="randomized",
                   choices=["randomized", "skerch"],
                   help="Метод низкоранговой аппроксимации GGN. "
                        "'randomized' (по умолчанию) держит 2 полноразмерных "
                        "буфера и доходит до rank ~50 на GPU ~16 ГиБ; 'skerch' "
                        "(seigh) держит 3 буфера плюс блок шума, поэтому при "
                        "rank >= 50 не помещается ни при каком размере блока.")
    p.add_argument("--ggn-oversample", type=int, default=10,
                   help="Число лишних случайных столбцов скетча: q = rank + "
                        "oversample. Больше -- точнее и точнее сходимость, но "
                        "пик памяти линейно растёт вместе с q. Единственный "
                        "параметр, реально влияющий на память скетча: "
                        "--sketch-blocksize у skerch на ro_sketch не влияет.")
    p.add_argument("--max-num-samples", type=int, default=25,
                   help="Макс. число сэмплов для GGN (как max_num_of_samples в luno).")
    p.add_argument("--val-calib-frac", type=float, default=0.8,
                   help="Доля val, отдаваемая калибровке prior_prec. Остаток "
                        "(1 - доля) идёт на валидацию с картинками и метриками "
                        "(Хатчинсон, см. --diag-hutchinson). Требует >= 2 сэмплов "
                        "в val, иначе ошибка.")
    p.add_argument("--diag-hutchinson", type=int, default=200,
                   help="Число проб Хатчинсона для diag(J Sigma J^T) НА ВСЕХ "
                        "прогонах: калибровка prior_prec, val и test. Вместо "
                        "ceil(m/vjp_chunk) обратных проходов считается вся "
                        "конгруэнция прямым режимом (n_probe прямых проходов; при "
                        "n_probe=16 это ~256x быстрее). 0 -- точный обратный путь, "
                        "≈369 с/сэмпл (на калибровке или val это десятки часов, "
                        "спросит подтверждение).")
    p.add_argument("--calib-max-samples", type=int, default=0,
                   help="Сколько сэмплов calib-loader реально уходит в калибровку "
                        "prior_prec. 0 -- все (доля --val-calib-frac от val). "
                        "Меньше всех -- только ради смоук-тестов.")
    p.add_argument("--calib-grid-min", type=float, default=-3.0)
    p.add_argument("--calib-grid-max", type=float, default=3.0)
    p.add_argument("--calib-grid-size", type=int, default=50)
    p.add_argument("--calib-objective", choices=["chi2", "nll"], default="chi2",
                   help="Objective калибровки: 'chi2' -- |chi_squared - 1| "
                        "(дефолт, как в laplax), 'nll' -- NLL.")
    p.add_argument("--no-progress", action="store_true",
                   help="Отключить прогресс-бары (tqdm).")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Утилиты данных
# ----------------------------------------------------------------------------

def _cast_to_double(sample):
    """Кастит x/y всех подбатчей в рабочую точность (DTYPE)."""
    for sub in sample.values():
        if "x" in sub:
            sub["x"] = sub["x"].to(DTYPE)
        if "y" in sub:
            sub["y"] = sub["y"].to(DTYPE)
        if "mask" in sub:
            sub["mask"] = sub["mask"].to(DTYPE)
    return sample


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


def _get_channelwise_reduce_dims(batch_tensor):
    return [0] + list(range(2, batch_tensor.ndim))


def preprocess_sample(data_processor, sample):
    """Нормализует входы батча (dict {task: {x, y}}); y остаётся в raw-пространстве.

    Напоминает rmi.predict_batch: сначала переносим x/y на device, затем
    preprocess (в else-ветке DefaultDataProcessor не делает .to(device)).
    """
    sample = _cast_to_double(sample)
    device = data_processor.device
    for sub in sample.values():
        sub["x"] = sub["x"].to(device)
        if "y" in sub:
            sub["y"] = sub["y"].to(device)
        if "mask" in sub:
            sub["mask"] = sub["mask"].to(device)
    return data_processor.preprocess(sample, training=False, batched=True)


def iter_samples(loader, max_samples):
    """Итерация по отдельным сэмплам задача-0 из батчей multidict-лоадера.

    Сохраняет batch-ось размера 1: (a) conv-слои лифтинга/core требуют batch-dim,
    (b) mean/std нормализаторов имеют форму (1, C, ...), поэтому срез без
    batch-оси не заbroadcastится.
    """
    count = 0
    for batch in loader:
        for sub in batch.values():
            b = sub["x"].shape[0]
            for i in range(b):
                if count >= max_samples:
                    return
                yield {
                    0: {
                        "x": sub["x"][i : i + 1],
                        "y": sub["y"][i : i + 1],
                    }
                }
                count += 1


# ----------------------------------------------------------------------------
# model_fn: лифтинг -> core -> проекция
# ----------------------------------------------------------------------------

def make_model_fn(wrapper: TorchFNOWrapper, lifting, projection):
    def model_fn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        sd = wrapper.reconstruct(w)
        x = x.to(dtype=DTYPE, device=wrapper.device)
        z = lifting(x)
        out = functional_call(wrapper.model, sd, (z,))
        return projection(out)

    return model_fn


# ----------------------------------------------------------------------------
# Построение пайплайна (модель + данные + нормализаторы)
# ----------------------------------------------------------------------------

def split_val_loaders(args, val_set):
    """Режет val на две непересекающиеся части: калибровку и валидацию.

    Калибровочная часть (``--val-calib-frac``) идёт в ``calibrate_prior_prec``,
    остаток -- на метрики и качественные картинки. Обе части нужны, чтобы
    решение о prior_prec не принималось на тех же сэмплах, на которых потом
    считаются отчётные метрики.

    ВАЖНО: ``rmi.loadData`` возвращает СПИСОК датасетов по числу задач, а не сами
    сэмплы, поэтому ``len(val_set) == числу задач``. Настоящая длина сэмплов --
    это ``len(MultiPhysicsDataset(...))``; ровно его и режем.

    Shuffle в лоадерах выключен, поэтому head/tail-разрез детерминирован и не
    требует seed.
    """
    val_mp = MultiPhysicsDataset(list(val_set))
    n_val = len(val_mp)
    frac = args.val_calib_frac
    if not 0.0 < frac < 1.0:
        raise ValueError(f"--val-calib-frac должен быть в (0, 1), получено {frac}")
    if n_val < 2:
        raise ValueError(
            f"val содержит {n_val} сэмпл(ов), поэтому поделить его на калибровку "
            f"({frac}) и валидацию (1-{frac}) нельзя. Нужно >= 2 сэмплов в val: "
            "поднимите split.val в конфиге или max_samples_per_split.val."
        )
    n_calib = int(frac * n_val)
    if n_calib < 1 or (n_val - n_calib) < 1:
        raise ValueError(
            f"Сплит val={n_val} сэмплов на доли {frac}/1-{frac} дал "
            f"калибровку={n_calib}, валидацию={n_val - n_calib}: обе части должны "
            "быть непустыми."
        )
    print(f"[split] val={n_val} сэмплов -> калибровка={n_calib}, "
          f"валидация={n_val - n_calib} (доля {frac})")

    calib_loader = DataLoader(
        dataset=Subset(val_mp, list(range(n_calib))), batch_size=1, shuffle=False
    )
    val_eval_loader = DataLoader(
        dataset=Subset(val_mp, list(range(n_calib, n_val))), batch_size=1, shuffle=False
    )
    return calib_loader, val_eval_loader


def build_pipeline(args):
    config_path = rmi.resolve_path(args.config)
    config = rmi.load_yaml_config(config_path)
    model_config = config.get("model", {})
    task_configs = config.get("tasks", [])
    assert task_configs, "No tasks in the config."

    print("Загрузка чекпойнтов (torch.load + dill):")
    core = rmi.load_from_dir(args.core_checkpoint)[0]
    liftings = (
        rmi.load_from_dir(args.lift_checkpoint_dir)
        if args.lift_checkpoint_dir is not None
        else None
    )
    projections = (
        rmi.load_from_dir(args.proj_checkpoint_dir)
        if args.proj_checkpoint_dir is not None
        else None
    )

    # Однозадачная версия: один лифтинг и одна проекция.
    assert liftings is not None and projections is not None, \
        "For the single-task LUNO pipeline both --lift/--proj checkpoint dirs are required."
    lifting, projection = liftings[0], projections[0]
    print("Используем lifting[0] и projection[0] (одна задача).")

    # Данные: для одного адаптера используем только первую задачу из конфига.
    single_task_configs = task_configs[:1]
    train_set, val_set, test_set, _ = rmi.loadData(single_task_configs)

    train_loader = DataLoader(
        dataset=MultiPhysicsDataset(list(train_set)),
        batch_size=1,
        shuffle=False,
    )
    calib_loader, val_eval_loader = split_val_loaders(args, val_set)
    test_loader = DataLoader(dataset=MultiPhysicsDataset(list(test_set)), batch_size=1, shuffle=False)

    loader_channels = rmi.get_loaders_channels(train_loader)
    print("loader_channels:", loader_channels)

    model_blocks = rmi.build_model(
        loader_channels,
        model_config,
        core,
        [lifting],
        [projection],
    )
    assert isinstance(model_blocks, tuple), "Expected lifting-core-projection architecture."
    model_liftings, model_core, model_projections = model_blocks

    return {
        "config": config,
        "model_config": model_config,
        "core": model_core,
        "lifting": model_liftings[0],
        "projection": model_projections[0],
        "train_loader": train_loader,
        "calib_loader": calib_loader,
        "val_eval_loader": val_eval_loader,
        "test_loader": test_loader,
        "loader_channels": loader_channels,
    }


def build_data_processor(args, pipeline):
    first_batch = next(iter(pipeline["train_loader"]))

    dims_x = {i: _get_channelwise_reduce_dims(sub["x"]) for i, sub in first_batch.items()}
    dims_y = {i: _get_channelwise_reduce_dims(sub["y"]) for i, sub in first_batch.items()}

    in_normalizer = MultiphysicsUnitGaussianNormalizer(num=1, dim=dims_x, key="x")
    inp_norm_files = sorted(rmi.get_all_files(args.in_normalizers, ".pkl"))
    print("[norm] in-normalizer files (task-0 берёт первый):", inp_norm_files[:1])
    in_normalizer.from_file(inp_norm_files)

    out_normalizer = MultiphysicsUnitGaussianNormalizer(num=1, dim=dims_y, key="y")
    out_norm_files = sorted(rmi.get_all_files(args.out_normalizers, ".pkl"))
    print("[norm] out-normalizer files (task-0 берёт первый):", out_norm_files[:1])
    out_normalizer.from_file(out_norm_files)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    in_normalizer.to(device)
    out_normalizer.to(device)
    data_processor = DefaultDataProcessor(
        in_normalizer=in_normalizer,
        out_normalizer=out_normalizer,
        device=device,
    )
    return data_processor, device


def raw_from_normalized(mean_norm, std_norm, out_normalizer, out_shape):
    """Переводит model output (norm) в raw-пространство задачи 0.

    y = y_norm * (std + eps) + mean  =>  mean_raw = ...;  var_raw = var_norm * (std+eps)^2.
    mean_norm/std_norm -- плоские; out_shape должен повторять форму model output
    (B, C_out, ...) для поканального broadcasting нормализатора.
    """
    norm = out_normalizer.normalizers[0]
    mean = mean_norm.reshape(out_shape)
    std = std_norm.reshape(out_shape)
    scale = norm.std + norm.eps
    mean_raw = mean * scale + norm.mean
    std_raw = std * scale
    return mean_raw.reshape(-1), std_raw.reshape(-1)


# ----------------------------------------------------------------------------
# GGN / калибровка / метрики
# ----------------------------------------------------------------------------

def compute_low_rank_ggn(args, model_fn, w0, train_loader, data_processor, affine,
                         mode="forward", fwd_batch=4):
    xs = []
    for sample in tqdm(
        iter_samples(train_loader, args.max_num_samples),
        total=args.max_num_samples,
        desc="[GGN] загрузка сэмплов",
        unit="smp",
        leave=False,
    ):
        sample = preprocess_sample(data_processor, sample)
        xs.append(sample[0]["x"])
    print(f"[GGN] собрано {len(xs)} сэмплов из train loader.")
    assert xs, "No samples collected for GGN."

    ggn_mv = GGNMatvec(
        model_fn, w0, xs, affine=affine, mode=mode, fwd_batch=fwd_batch,
        loss_fn="mse", factor=1.0, vjp_chunk=args.vjp_chunk,
    )
    _print_mem("ggn_mv", w0.device)

    low_rank = low_rank_ggn(
        ggn_mv,
        rank=args.max_rank,
        oversample=args.ggn_oversample,
        method=args.ggn_method,
        device=str(w0.device),
        dtype=w0.dtype,
        fwd_batch=fwd_batch,
    )
    print(f"[GGN] low-rank terms: U={tuple(low_rank.U.shape)} S={tuple(low_rank.S.shape)}")
    return low_rank


def load_low_rank_ggn(path, w0):
    """Читает LowRankTerms из pickle (артефакт *_low_rank_terms.pkl).

    Возвращает ``None``, если файла нет -- тогда вызывающий код считает GGN
    как обычно (предупреждение печатается здесь). Файл с другими проблемами
    (битый pickle, не LowRankTerms, несовместимая d) даёт исключение: путь
    указан явно, молчаливый пересчёт спрячет опечатку.

    U/S лежат на CPU (это CPU-копия для pickle), переносятся на устройство и
    точность ``w0``.
    """
    p = rmi.resolve_path(path)
    if not p.is_file():
        print(f"[GGN] --ggn-checkpoint: файла нет ({p}); считаю GGN как обычно.")
        return None
    with open(p, "rb") as f:
        terms = pickle.load(f)
    if not isinstance(terms, LowRankTerms):
        raise TypeError(
            f"--ggn-checkpoint: ожидался LowRankTerms, получен "
            f"{type(terms).__name__} ({p})."
        )
    U, S = terms.U, terms.S
    if U.ndim != 2:
        raise ValueError(f"--ggn-checkpoint: U.ndim={U.ndim}, ожидалось 2 ({p}).")
    if S.ndim != 1 or S.numel() != U.shape[1]:
        raise ValueError(
            f"--ggn-checkpoint: S {tuple(S.shape)} не согласуется с U "
            f"{tuple(U.shape)} (нужен S длины U.shape[1]) ({p})."
        )
    if U.shape[0] != w0.numel():
        raise ValueError(
            f"--ggn-checkpoint: U.shape[0]={U.shape[0]} != w0.numel()="
            f"{w0.numel()} -- GGN от другой модели или другого сплита ({p})."
        )
    U = U.to(device=w0.device, dtype=w0.dtype)
    S = S.to(device=w0.device, dtype=w0.dtype)
    scalar = terms.scalar.to(device=w0.device) if isinstance(terms.scalar, torch.Tensor) else terms.scalar
    print(f"[GGN] загружен из {p}: U={tuple(U.shape)} S={tuple(S.shape)} "
          f"scalar={scalar}, dtype={U.dtype}, device={U.device}")
    return LowRankTerms(U=U, S=S, scalar=scalar)


def _batch_predictive_stats(model_fn, w0, weight_cov, x, out_normalizer,
                            num_output_channels, affine, mode="forward",
                            fwd_batch=4, vjp_chunk=64, diag_probes=0):
    """(mean_raw, std_raw) для одного входного тензора x в raw-пространстве."""
    with torch.no_grad():
        out_shape = model_fn(x, w0).shape
    jac = LastFNOBlockWeightJacobian(
        model_fn,
        x,
        w0,
        affine=affine,
        mode=mode,
        fwd_batch=fwd_batch,
        num_output_channels=num_output_channels,
        vjp_chunk=vjp_chunk,
        diag_probes=diag_probes,
    )
    # Граф последнего чанка diag_JJT не должен пережить выход из функции: иначе
    # он удерживается до конца эпохи, и на eval-цикле из 500 сэмплов память
    # накапливается до OOM. При diag_probes > 0 обратный граф вообще не строится.
    try:
        mean_norm = jac.fx.reshape(-1)
        var_norm = var_of_congruence(jac, weight_cov, diag_probes=diag_probes)
    finally:
        jac._release_vjp()
    std_norm = torch.sqrt(var_norm)
    return raw_from_normalized(mean_norm, std_norm, out_normalizer, out_shape)


def calibrate_prior_prec(args, model_fn, w0, low_rank, wrapper, data_processor,
                         out_normalizer, val_loader, affine, objective_name="chi2",
                         mode="forward", fwd_batch=4):
    """Калибровка scalar prior_prec на сэмплах калибровочной части val.

    ``val_loader`` здесь -- калибровочная часть val (см. ``--val-calib-frac``);
    сколько из неё реально взять, решает ``--calib-max-samples`` (0 -- все).
    Остаток val идёт на валидацию с метриками и картинками, поэтому решение о
    ``prior_prec`` не принимается на тех же сэмплах, на которых считаются
    отчётные метрики.

    Цикл инвертирован относительно ``grid_search``: сэмпл снаружи, грид внутри.
    От ``prior_prec`` зависит только ``scalar``/``coeff`` (O(k)), а ``J@Z``,
    ``J@U``, ``U^T Z`` и ``diag(JJ^T)`` -- нет, поэтому пробы считаются один раз
    на сэмпл и переиспользуются для всех кандидатов; иначе каждый из
    ``len(grid)`` кандидатов оплачивал бы свой прогон по сэмплам. Побочный
    эффект -- никакого early-stop: цель считается по всему гриду целиком и
    берётся точный ``argmin``.

    ``objective_name``: "chi2" -- |chi_squared - 1| (как дефолт в laplax),
    "nll" -- negative log-likelihood. Обе складываются как суммы по элементам и
    только потом делятся на общее число элементов, то есть усредняются по всем
    элементам всех сэмплов, а не по сэмплам.
    """
    n_total = len(val_loader.dataset)
    n_calib = n_total if args.calib_max_samples <= 0 else min(
        args.calib_max_samples, n_total
    )
    n_probe = args.diag_hutchinson

    grid = torch.logspace(
        args.calib_grid_min, args.calib_grid_max, args.calib_grid_size, base=10.0
    )
    # cov строится по кандидату, а не по сэмплу: linverse переиспользует тот же
    # low_rank.U, поэтому все объекты держат один и тот же буфер (0.55 ГиБ при
    # rank 5) и весят O(k) каждый.
    covs = [create_luno_cov(low_rank, {"prior_prec": p}) for p in grid]
    n_grid = len(covs)
    _print_mem("calib_covs", w0.device)
    chi2_acc = torch.zeros(n_grid, dtype=torch.float64, device=w0.device)
    nll_acc = torch.zeros(n_grid, dtype=torch.float64, device=w0.device)
    n_elements = 0
    num_output_channels = wrapper.num_output_channels
    how = f"Хатчинсон n_probe={n_probe}" if n_probe > 0 else "точный обратный путь"
    print(f"\n[calibrate] prior_prec: {n_calib}/{n_total} сэмплов calib-loader, "
          f"{n_grid} кандидатов грида, objective={objective_name}, {how}, "
          f"fwd_batch={fwd_batch}")

    bar = tqdm(total=n_calib, desc="[calibrate] сэмплы", unit="smp", leave=False)
    try:
        for sample in iter_samples(val_loader, n_calib):
            sample = preprocess_sample(data_processor, sample)
            x = sample[0]["x"]
            target_raw = sample[0]["y"]
            target = target_raw.reshape(-1)
            out_shape = None  # считается лениво, один forward на сэмпл
            jac = LastFNOBlockWeightJacobian(
                model_fn, x, w0, affine=affine, mode=mode, fwd_batch=fwd_batch,
                num_output_channels=num_output_channels,
                vjp_chunk=args.vjp_chunk,
            )
            A = B = W = None
            try:
                # fx == f(x, w0) уже ПЛОСКИЙ (см. _flat_fn), поэтому из него
                # нельзя брать out_shape: нормализатору нужна форма
                # (B, C_out, ...) для поканального broadcasting, иначе
                # (65536,) * (4,1,1,1) раздувается до (4,1,1,65536) и chi2
                # падает на несовпадении размеров с target. Форму берём отдельным
                # forward'ом -- один раз на сэмпл, как в _batch_predictive_stats.
                if out_shape is None:
                    with torch.no_grad():
                        out_shape = model_fn(x, w0).shape
                mean_norm = jac.fx.reshape(-1)
                if n_probe > 0:
                    A, B, W = sample_congruence_probes(jac, covs[0].U, n_probe)
                for i, cov in enumerate(covs):
                    if n_probe > 0:
                        var_norm = var_from_congruence_probes(A, B, W, cov)
                    else:
                        var_norm = var_of_congruence(jac, cov)
                    std_norm = torch.sqrt(var_norm)
                    mean_raw, std_raw = raw_from_normalized(
                        mean_norm, std_norm, out_normalizer, out_shape
                    )
                    if objective_name == "chi2":
                        chi2_acc[i] += chi_squared(
                            mean_raw, std_raw, target, averaged=False
                        )
                    else:
                        nll_acc[i] += nll_gaussian(
                            mean_raw, std_raw, target, scaled=False
                        )
                n_elements += target.numel()
            finally:
                # Граф forward-замыкания (model_fn, x) держится за self, а в
                # точном пути -- ещё и обратный граф. Освобождаем сразу: сам jac
                # подавится привязкой в начале следующей итерации.
                jac._release_vjp()
            bar.update(1)
    finally:
        bar.close()

    if not n_elements:
        raise SystemExit(
            f"Калибровка не увидела ни одного сэмпла: len(calib_loader.dataset)="
            f"{n_total}, --calib-max-samples={args.calib_max_samples}."
        )
    if objective_name == "chi2":
        objective = (chi2_acc / n_elements - 1.0).abs()
    else:
        objective = nll_acc / n_elements
    if not bool(torch.isfinite(objective).all()):
        raise RuntimeError(
            f"Objective калибровки не конечен ({objective.tolist()}, "
            f"n_calib={n_calib}, n_probe={n_probe}): var неотрицателен по "
            f"построению при n_probe>0, поэтому дело в target или в std."
        )
    best_idx = int(objective.argmin())
    best_prec = grid[best_idx]
    print(f"[calibrate] best prior_prec={best_prec.item():.6e}, "
          f"{objective_name}={float(objective[best_idx]):.6e} "
          f"(на {n_calib} сэмплах, {n_grid} кандидатов)")
    return {"prior_prec": best_prec, "objective": objective_name,
            "samples": n_calib, "diag_probes": n_probe}


def save_luno_images(pred, band, target, output_prefix, batch_idx):
    """Картинки LUNO: target / predict (mean_raw) / std_raw / sqrt(chi2).

    sqrt(chi2) = |target - mean_raw| / std_raw -- калиброванная стандартизованная
    ошибка по элементам. Имена файлов: <prefix>_<key>_<idx>_<suffix>_.png.
    """
    for key in pred.keys():
        time_index = _inference_time_index(pred[key])
        p, b, t = pred[key], band[key], target[key]
        eps = torch.finfo(p.dtype).eps
        sqrt_chi2 = (t - p).abs() / b.clamp_min(eps)

        save_image(canonical_image(t, channel_index=0, time_index=time_index),
                   output_prefix.with_name(output_prefix.name + f"{key}_{batch_idx}_target_.png"),
                   "luno target")
        save_image(canonical_image(p, channel_index=0, time_index=time_index),
                   output_prefix.with_name(output_prefix.name + f"{key}_{batch_idx}_predict_.png"),
                   "luno predict")
        save_image(canonical_image(b, channel_index=0, time_index=time_index),
                   output_prefix.with_name(output_prefix.name + f"{key}_{batch_idx}_std_raw_.png"),
                   "luno std_raw")
        save_image(canonical_image(sqrt_chi2, channel_index=0, time_index=time_index),
                   output_prefix.with_name(output_prefix.name + f"{key}_{batch_idx}_sqrt_chi2_.png"),
                   "luno sqrt(chi2)")


def evaluate_luno(args, loader, model_fn, w0, low_rank, prior_args, wrapper,
                  data_processor, out_normalizer, task_name, max_samples,
                  affine, metrics_config=None, output_dir=None,
                  mode="forward", fwd_batch=4,
                  save_images=True, diag_probes=0):
    """Метрики на eval-сэмплах.

    ``luno_chi2`` -- глобальное среднее chi2 по элементам всех сэмплов.
    ``luno_sqrt_chi2_*`` -- статистики приведённого sqrt(chi2) = sqrt(chi2 /
    n_elements) (RMS стандартизованного остатка, ~1 при идеальной калибровке),
    посчитанные по каждому сэмплу: перцентили p5/p25/p50/p75/p95 и среднее.

    ``save_images=False`` -- не писать PNG (диагностику смотрим только на
    валидации, на тесте это лишняя работа).
    ``diag_probes`` -- число проб Хатчинсона для ``diag(J Sigma J^T)`` (вся
    конгруэнция, не только ``diag(J J^T)``); 0 -- точный обратный режим.
    Калибровка, val и test берут одно значение ``--diag-hutchinson``: смещение
    оценщика зависит только от числа проб (``~2/n_probe`` на chi2), поэтому
    держать его разным ради "эталонности" не нужно.
    """
    cov = create_luno_cov(low_rank, prior_args)
    nll_sum = torch.tensor(0.0, dtype=DTYPE)
    rmse_sum = torch.tensor(0.0, dtype=DTYPE)
    chi2_sum = torch.tensor(0.0, dtype=DTYPE)
    sqrt_chi2_samples = []
    cfg_sums = {}
    n_elements = 0
    n_samples = 0

    output_prefix = None
    if output_dir is not None and save_images:
        output_prefix = Path(output_dir) / f"inspections_luno_{task_name}"
        output_prefix.mkdir(parents=True, exist_ok=True)
        output_prefix = Path.joinpath(output_prefix, "luno_")

    # Без no_grad аккумуляторы nll_sum/rmse_sum/chi2_sum -- не-leaf тензоры,
    # и каждый следующий плюс цепляется к предыдущему через grad_fn. Граф каждого
    # eval-сэмпла тогда живёт до конца цикла (до .item() ниже), и память растёт
    # линейно по числу сэмплов: на rank=20 это ~12 ГиБ при одном лишь --vjp-chunk,
    # от которого эффект нулевой. Графы здесь не нужны вообще.
    with torch.no_grad():
        for sample in iter_samples(loader, max_samples):
            sample = preprocess_sample(data_processor, sample)
            x = sample[0]["x"]
            target_raw = sample[0]["y"]
            target = target_raw.reshape(-1)
            mean_raw, std_raw = _batch_predictive_stats(
                model_fn, w0, cov, x, out_normalizer, wrapper.num_output_channels,
                affine=affine, mode=mode, fwd_batch=fwd_batch, vjp_chunk=args.vjp_chunk,
                diag_probes=diag_probes,
            )
            nll_sum = nll_sum + nll_gaussian(mean_raw, std_raw, target, scaled=False)
            rmse_sum = rmse_sum + torch.sqrt(torch.mean((mean_raw - target) ** 2))
            chi2_sample = chi_squared(mean_raw, std_raw, target, averaged=False)
            chi2_sum = chi2_sum + chi2_sample
            # Приведённый sqrt(chi2) одного сэмпла: sqrt(chi2 / n_elements).
            n_elem_sample = target.numel()
            if n_elem_sample:
                sqrt_chi2_samples.append(
                    float(torch.sqrt(chi2_sample.double() / n_elem_sample))
                )
            n_elements += n_elem_sample
            n_samples += 1

            out_shape = target_raw.shape
            pred = {0: mean_raw.reshape(out_shape)}
            band = {0: std_raw.reshape(out_shape)}
            target_dict = {0: target_raw}

            if output_prefix is not None:
                save_luno_images(pred, band, target_dict, output_prefix, n_samples - 1)

            if metrics_config:
                batch_metrics = rmi.compute_batch_metrics(
                    pred,
                    band,
                    target_dict,
                    metrics_config=metrics_config,
                    task_name=task_name,
                )
                for name, value in batch_metrics[0].items():
                    cfg_sums[name] = cfg_sums.get(name, 0.0) + float(value)

    nll = (nll_sum / n_elements).item() if n_elements else float("nan")
    rmse = (rmse_sum / n_samples).item() if n_samples else float("nan")
    chi2 = (chi2_sum / n_elements).item() if n_elements else float("nan")

    # Перцентили приведённого sqrt(chi2) по сэмплам.
    sqrt_chi2_pct = {}
    if sqrt_chi2_samples:
        vals = torch.tensor(sqrt_chi2_samples, dtype=torch.float64)
        levels = torch.tensor(
            [q / 100.0 for q in SQRT_CHI2_PERCENTILES], dtype=torch.float64
        )
        for q, v in zip(SQRT_CHI2_PERCENTILES, torch.quantile(vals, levels).tolist()):
            sqrt_chi2_pct[f"luno_sqrt_chi2_p{q}"] = v
        sqrt_chi2_pct["luno_sqrt_chi2_mean"] = vals.mean().item()

    cfg_metrics = {name: value / n_samples for name, value in cfg_sums.items()}
    metrics = {
        **cfg_metrics,
        **sqrt_chi2_pct,
        "luno_nll": nll,
        "luno_rmse": rmse,
        "luno_chi2": chi2,
        "samples": n_samples,
    }
    cfg_str = " ".join(f"{k}={v:.6e}" for k, v in cfg_metrics.items())
    pct_str = " ".join(f"{k}={v:.6e}" for k, v in sqrt_chi2_pct.items())
    print(f"[eval:{task_name}] luno_nll={nll:.6e} luno_rmse={rmse:.6e} "
          f"luno_chi2={chi2:.6e} {pct_str} {cfg_str} "
          f"(samples={n_samples}, elements={n_elements})")
    return metrics


# Измерено на r5_hutch_fix: 4096 чанков @ 11.23 chunk/s (m=65536, vjp_chunk=16)
# -- то есть один сэмпл точного diag(J Sigma J^T) стоит ~369 с.
EXACT_SEC_PER_SAMPLE = 369.0


def confirm_exact_diag(args, pipeline):
    """Просит подтверждение при --diag-hutchinson 0 (точный обратный путь).

    Точный путь одинаково дорог на всех трёх стадиях -- калибровке, val и test,
    -- поэтому оцениваем их разом и останавливаемся до GGN и до записи
    low_rank_terms.pkl. Неинтерактивный запуск (setsid/nohup, как у нас
    фоновые GPU-задачи) вопрос задать не может, поэтому там отказ сразу:
    молча прогнать 83-часовой прогон нельзя.
    """
    if args.diag_hutchinson > 0:
        return
    n_calib = len(pipeline["calib_loader"].dataset)
    n_val = len(pipeline["val_eval_loader"].dataset)
    n_test = len(pipeline["test_loader"].dataset)
    n_all = n_calib + n_val + n_test
    total_h = EXACT_SEC_PER_SAMPLE * n_all / 3600.0
    msg = (
        f"--diag-hutchinson 0: точный обратный путь на {n_all} сэмплах "
        f"≈ {total_h:.1f} ч (калибровка {n_calib} + val {n_val} + test {n_test}, "
        f"~{EXACT_SEC_PER_SAMPLE:.0f} с/сэмпл)."
    )
    print(f"[!] {msg}")
    if not sys.stdin.isatty():
        raise SystemExit(
            f"{msg} Неинтерактивный запуск -- отказ. Перезапустите с "
            f"--diag-hutchinson N (рекомендуется 200)."
        )
    try:
        answer = input(" Продолжить? [y/N]: ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes", "д", "да"):
        raise SystemExit(
            "Отменено. Перезапустите с --diag-hutchinson N (рекомендуется 200)."
        )


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()

    set_enabled(not args.no_progress)

    if args.dtype == "float64":
        global DTYPE, CDTYPE
        DTYPE = torch.float64
        CDTYPE = torch.complex128

    if args.core_checkpoint is None:
        raise SystemExit("--core-checkpoint обязателен.")

    pipeline = build_pipeline(args)
    confirm_exact_diag(args, pipeline)
    core = pipeline["core"]
    lifting = pipeline["lifting"]
    projection = pipeline["projection"]

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    core = core.to(device=device)
    lifting = lifting.to(device=device).to(dtype=DTYPE)
    projection = projection.to(device=device).to(dtype=DTYPE)

    # ВАЖНО: core НЕ кастатится `to(dtype=DTYPE)` целиком — `Tensor.to(torch.float64)`
    # отбрасывает мнимую часть комплексных спектральных весов SpectralConv
    # (warning: "Casting complex values to real discards the imaginary part"),
    # после чего forward падает в einsum ("expected ComplexDouble but found Double"),
    # а GGN строится по неправильным весам. Вместо этого переносим core на device
    # и выставляем точность поканально, сохраняя комплексность.
    # Сам `functional_call(core, sd, ...)` и так работает в нужной точности:
    # она приходит из reconstruct(w) / _base_state_dict().
    for p in core.parameters():
        p.data = p.data.to(
            dtype=CDTYPE if p.is_complex() else DTYPE, device=device
        )
    for b in core.buffers():
        b.data = b.data.to(
            dtype=CDTYPE if b.is_complex() else DTYPE, device=device
        )

    # --- output-папка эксперимента (как в run_multiphysics_inference.py) ---
    output_config = pipeline["config"].get("output", {})
    output_root = (
        args.output_root
        if args.output_root is not None
        else output_config.get("root", "runs")
    )
    config_path = rmi.resolve_path(args.config)
    run_prefix = args.run_name if args.run_name is not None else config_path.stem
    run_name = f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = rmi.resolve_path(output_root) / run_name
    for d in (output_dir / "checkpoints", output_dir / "logs", output_dir / "metrics"):
        d.mkdir(parents=True, exist_ok=True)
    print(f"output_dir: {output_dir}")

    # --- нормализаторы и data_processor ---
    data_processor, device = build_data_processor(args, pipeline)

    # --- сплит: last FNO block core -> w0 ---
    wrapper = TorchFNOWrapper(core, dtype=DTYPE)
    model_fn_wrap, w0 = split_wrapper(wrapper)
    print(f"[split] w0: d={w0.numel()}, dtype={w0.dtype}, device={w0.device}")

    # keys = discover_last_block_keys(core)
    # print("\n[последний фурье-слой core]:")
    # for label in ("R", "W", "b"):
    #     path, cls, attr = _resolve_key_owner(core, keys[label])
    #     print(f"    {label}: layer='{path}' ({cls}, attr='{attr}')")
    # R, W, b = ref_state_dict(core, keys)
    # print(f"    R: {tuple(R.shape)} (complex -> R.real | R.imag)")
    # print(f"    W: {tuple(W.shape)}")
    # print(f"    b: {tuple(b.shape) if b is not None else None}")
    # assert torch.is_complex(R), (
    #     "Спектральный вес R стал некомплексным (мнимая часть потеряна из-за "
    #     "core.to(dtype=...)). Не кастать core целиком в float64; см. комментарий"
    #     " к переносу core на device в main."
    # )

    # Верификация сплита (реконструкция весов core из w0).
    base = wrapper._base_state_dict()
    recon = wrapper.reconstruct(w0)
    max_diff = max(
        (recon[k] - base[k]).abs().max().item()
        for k in base
    )
    status = "SPLIT OK" if max_diff == 0 else f"SPLIT NOT OK (max diff={max_diff:.3e})"
    print(f"    [проверка] max |reconstruct(w0) - base| = {max_diff:.3e} -> {status}")
    _print_mem("after_split_w0", device)

    # model_fn: лифтинг -> core(с w) -> проекция (фиксированы лифтинг и проекция).
    model_fn = make_model_fn(wrapper, lifting, projection)

    # Стартовый гейт: Jv выбранного режима против reverse-строк на ОДНОМ батче.
    # Эталон -- обратный режим (AD не трогает спектральный conv "изнутри"); при
    # расхождении падаем явно, а не молча считаем кривую кривизну. Допуск зависит
    # от точности: float32 набирает ошибку суммирования по 28M элементам, float64 --
    # на три порядка точнее (см. default_check_tol).
    probe = preprocess_sample(
        data_processor, next(iter_samples(pipeline["train_loader"], 1))
    )
    affine = (
        AffineLastBlock(wrapper.model, projection, wrapper.keys, dtype=DTYPE)
        if args.jv_mode == "affine" else None
    )
    ok, rel = check_jacobian(
        model_fn, probe[0]["x"], w0,
        affine=affine, mode=args.jv_mode, fwd_batch=args.fwd_batch,
        num_probes=args.check_probes, tol=args.check_tol,
    )
    tol_used = args.check_tol if args.check_tol is not None else default_check_tol(w0.dtype)
    print(f"[{args.jv_mode}-JV] self-check vs reverse rows на 1 батче: "
          f"{'OK' if ok else 'FAILED'} (max rel err={rel:.3e}, tol={tol_used:.1e}, "
          f"dtype={w0.dtype}, torch={torch.__version__})")
    if not ok:
        raise RuntimeError(
            f"Jv в режиме '{args.jv_mode}' не совпал с reverse-строками "
            f"(max rel err={rel:.3e} > tol={tol_used:.1e}). Проверьте структуру "
            "последнего FNO-блока (conv/skip/norm/transform), точность весов и "
            "то, что complex-веса не потеряли мнимую часть при кастинге."
        )
    _print_mem("after_jv_check", device)

    # --- low-rank GGN на train (или загрузка готового из --ggn-checkpoint) ---
    print("\n[GGN] low-rank аппроксимация GGN последнего фурье-слоя:")
    low_rank = (
        load_low_rank_ggn(args.ggn_checkpoint, w0) if args.ggn_checkpoint else None
    )
    ggn_loaded = low_rank is not None
    if not ggn_loaded:
        low_rank = compute_low_rank_ggn(
            args, model_fn, w0, pipeline["train_loader"], data_processor, affine,
            mode=args.jv_mode, fwd_batch=args.fwd_batch,
        )

        # U имеет форму (d, rank) -- при d = 27.6M и rank = 50 это 5.15 ГиБ. Пишем
        # артефакт с CPU-копией: сам pickle всё равно такой размер, но GPU-копия
        # продолжает жить в low_rank для калибровки и метрик. При загрузке из
        # --ggn-checkpoint шаг пропускается: артефакт уже есть в чекпойнте.
        low_rank_cpu = LowRankTerms(
            U=low_rank.U.detach().to("cpu"), S=low_rank.S.detach().to("cpu"),
            scalar=low_rank.scalar,
        )
        terms_path = output_dir / f"{run_prefix}_low_rank_terms.pkl"
        with open(terms_path, "wb") as f:
            pickle.dump(low_rank_cpu, f)
        print(f"[save] low_rank_terms -> {terms_path} "
              f"({terms_path.stat().st_size / 1024 ** 3:.2f} GiB)")

    # --- калибровка на сэмплах калибровочной части val (доля --val-calib-frac) ---
    # Память здесь -- единственное, что отличает дешёвую калибровку от OOM: один
    # прогон прямого режима стоит ~fwd_batch * активации сети, и если GGN оставил
    # после себя буферы, места на него может не хватить.
    _print_mem("before_calibrate", device)
    prior_args = calibrate_prior_prec(
        args, model_fn, w0, low_rank, wrapper, data_processor,
        data_processor.out_normalizer, pipeline["calib_loader"], affine,
        objective_name=args.calib_objective, mode=args.jv_mode,
        fwd_batch=args.fwd_batch,
    )

    out_normalizer = data_processor.out_normalizer

    # --- метрики после калибровки: val и test делят одно значение --diag-hutchinson ---
    # val (остаток val): Хатчинсон + PNG; test: тот же Хатчинсон, без PNG.
    print("\n[metrics] финальные метрики после калибровки:")
    val_metrics = evaluate_luno(
        args, pipeline["val_eval_loader"], model_fn, w0, low_rank, prior_args,
        wrapper, data_processor, out_normalizer, "val", len(pipeline["val_eval_loader"].dataset),
        affine=affine,
        metrics_config=pipeline["config"].get("metrics", {}),
        output_dir=output_dir,
        mode=args.jv_mode, fwd_batch=args.fwd_batch,
        save_images=True, diag_probes=args.diag_hutchinson,
    )
    test_metrics = evaluate_luno(
        args, pipeline["test_loader"], model_fn, w0, low_rank, prior_args, wrapper,
        data_processor, out_normalizer, "test", len(pipeline["test_loader"].dataset),
        affine=affine,
        metrics_config=pipeline["config"].get("metrics", {}),
        output_dir=output_dir,
        mode=args.jv_mode, fwd_batch=args.fwd_batch,
        save_images=False, diag_probes=args.diag_hutchinson,
    )

    results = {
        "prior_prec": prior_args["prior_prec"].item(),
        "calib_objective": prior_args.get("objective", args.calib_objective),
        "calib_samples": prior_args["samples"],
        "calib_diag_probes": prior_args["diag_probes"],
        "calib_max_samples": args.calib_max_samples,
        "val": val_metrics,
        "test": test_metrics,
        "max_rank": args.max_rank,
        "ggn_method": args.ggn_method,
        "ggn_oversample": args.ggn_oversample,
        "max_num_samples_ggn": args.max_num_samples,
        # Откуда взят low-rank: "checkpoint" -- загружен из --ggn-checkpoint
        # (артефакт не переписывался), "computed" -- посчитан на train loader.
        # ggn_rank -- фактический ранг (из чекпойнта, если он авторитетен),
        # max_rank остаётся значением CLI, чтобы не ломать сравнение с ранами.
        "ggn_source": "checkpoint" if ggn_loaded else "computed",
        "ggn_checkpoint": (
            str(rmi.resolve_path(args.ggn_checkpoint)) if ggn_loaded else None
        ),
        "ggn_rank": int(low_rank.U.shape[1]),
        "diag_hutchinson": args.diag_hutchinson,
        "max_eval_samples": len(pipeline["val_eval_loader"].dataset),
        "max_test_samples": len(pipeline["test_loader"].dataset),
        "val_calib_frac": args.val_calib_frac,
    }
    with open(output_dir / "metrics" / "val_metrics.pkl", "wb") as f:
        pickle.dump({"val": val_metrics}, f)
    with open(output_dir / "metrics" / "test_metrics.pkl", "wb") as f:
        pickle.dump({"test": test_metrics}, f)
    with open(output_dir / "luno_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nГотово. Результаты: {output_dir}")


if __name__ == "__main__":
    main()