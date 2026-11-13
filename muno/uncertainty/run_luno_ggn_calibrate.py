from __future__ import annotations

import argparse
import gc
import math
import os
import pickle
import sys
import traceback
from dataclasses import dataclass
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

import dill

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
    TorchLiftingWrapper,
    discover_last_block_keys,
    ref_state_dict,
    split_wrapper,
)
from luno_torch.affine import AffineLastBlock, AffineLastBlockError, LastBlockSuffix
from luno_torch.calibrate import (
    STD_REL_FLOOR,
    chi_squared,
    nll_gaussian,
    standardized_residual,
    std_valid_mask,
)
from luno_torch.factory import create_luno_cov
from metrics_uq import log_rmse_over_rmv, normalized_ause, reliability_bin_sums
from luno_torch.ggn import (
    GGNMatvec,
    LiftingGGNMatvec,
    LowRankTerms,
    auto_ggn_mc,
    low_rank_ggn,
    sampled_full_ggn,
)
from luno_torch.kfac import KFACLastBlock, KFACTerms
from luno_torch.jacobian import (
    LastFNOBlockWeightJacobian,
    LiftingWeightJacobian,
    check_jacobian,
    default_check_tol,
    _var_of_congruence_hutchinson,
    sample_congruence_probes,
    var_from_congruence_probes,
    var_of_congruence,
)
from luno_torch.progress import set_enabled, tqdm

import run_multiphysics_inference as rmi

from check_split import _resolve_key_owner

DTYPE = torch.float32
CDTYPE = torch.complex64


def _install_pickle_compat():
    """Совместимость чекпойнтов, сохранённых dill вместе с исходным модулем.

    Адаптеры вроде ``PostLiftMambaLifting`` сериализованы dill целиком, со
    ссылками на модуль ``muno/models/mamba_fno.py``, а тот импортирует
    ``torch.utils.data.Dataset``. Старые torch держали в
    ``torch.utils.data.dataset`` типовую переменную ``T_co``; в новых её нет, и
    unpickle падает с ``Can't get attribute 'T_co'``. Подставляем эквивалент,
    только если атрибута нет: на старых torch это no-op.
    """
    import typing

    import torch.utils.data.dataset as _tud

    if not hasattr(_tud, "T_co"):
        _tud.T_co = typing.TypeVar("T_co", covariant=True)

# Квантили элементного sqrt(chi2) = |остаток|/std внутри канала (и внутри каждой
# пары (канал, t)), затем среднее по сэмплам.
PER_CHANNEL_PCT = (50, 75, 80, 90, 99)

# Число бинов предсказанного std для ENCE и кривой надёжности (равные доли
# пикселей внутри каждой пары (канал, t); на поле 128x128 -- ~1600 точек на бин).
ENCE_BINS = 10

# Те же квантили для |N(0, 1)| -- ориентиры идеально откалиброванного гаусса.
GAUSS_ABS_QUANTILES = {50: 0.674, 75: 1.150, 80: 1.282, 90: 1.645, 99: 2.576}


# ----------------------------------------------------------------------------
# Аргументы
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LUNO pipeline for a multi-adapter Muno.")
    p.add_argument("--config", default="experiments/configs/pdebench_multiphysics_pretrain.yaml")
    p.add_argument("--tasks", default="all",
                   help="Какие задачи из config.tasks запускать: 'all' (по "
                        "умолчанию) или имена через запятую (поле name задачи). "
                        "Порядок и task_idx берутся из описания задачи.")
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
                        "через jvp). Оба точны; выбирается по памяти/скорости. "
                        "Для --ggn-target lifting доступен только forward.")
    p.add_argument("--ggn-target", choices=["last-block", "lifting"],
                   default="last-block",
                   help="По каким параметрам считать GGN и калибровать prior_prec: "
                        "'last-block' -- веса последнего FNO-блока core (как "
                        "раньше), 'lifting' -- все веса и bias адаптера лифтинга "
                        "для каждой задачи.")
    p.add_argument("--fwd-batch", type=int, default=4,
                   help="Сколько колонок J@W считать одним батчем через vmap "
                        "(только для --jv-mode forward). Больше -- быстрее, но "
                        "кратно больше памяти.")
    p.add_argument("--jacobian-check", action="store_true",
                   help="Перед GGN сравнить J*v (forward-режим) с reverse-строками "
                        "на одном train-сэмпле и упасть при расхождении. Нужен при "
                        "смене ядра (трансформер, CNO, ...) или версии torch: "
                        "неподдерживаемая forward-AD операция проявится здесь, а "
                        "не молча кривой кривизной.")
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
    p.add_argument("--prior-prec-source", type=str, default=None,
                   help="Значение по умолчанию для luno.prior_prec_source "
                        "out_distr-задач: папка in_distr-задачи (или прогона) с "
                        "luno_results.pkl того же task_idx. Используется, только "
                        "если in_distr-задачи этого task_idx нет в прогоне.")
    p.add_argument("--ggn-checkpoint", type=str, default=None,
                   help="Значение по умолчанию для luno.ggn_checkpoint (файл или "
                        "папка с *_low_rank_terms.pkl / *_kfac_terms.pkl). "
                        "out_distr: кривизна для оценки, метод определяется "
                        "файлом (нет файла -- Iso). in_distr: "
                        "если файл найден и валиден -- расчёт GGN "
                        "пропускается целиком (сбор сэмплов, GGNMatvec, скетч), "
                        "артефакт в новый ранг не пишется, а --max-rank/--ggn-method/"
                        "--ggn-oversample/--max-num-samples игнорируются: авторитетен "
                        "ранг из чекпойнта. Если файла нет -- предупреждение и GGN "
                        "считается как обычно (сохранение артефакта тоже). Битый или "
                        "несовместимый (другая d) файл -- ошибка.")
    p.add_argument("--ggn-method", type=str, default="randomized",
                   choices=["randomized", "skerch", "kfac"],
                   help="Метод аппроксимации GGN. 'kfac' -- НЕ низкоранговая, а "
                        "кронекерова кривизна по Фурье-модам последнего блока "
                        "(luno_torch.kfac): один проход по сэмплам, без буферов "
                        "(d, rank); только --ggn-target last-block, числа с "
                        "low-rank не совпадают (другая аппроксимация). Остальные "
                        "-- низкоранговые: "
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
    p.add_argument("--ggn-power-iter", type=int, default=2,
                   help="Число степенных итераций randomized-скетча. Проходов по "
                        "train-сэмплам -- power_iter + 1 (финальная проекция "
                        "берётся из того же скетча, Nystrom). 2 (по умолчанию) "
                        "-- точность прежней схемы за 3 прохода вместо 4; 1 -- "
                        "2 прохода, ведущие собственные пары точны, в плоском "
                        "хвосте спектра ошибка до процентов.")
    p.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Для --ggn-target last-block: считать замороженный "
                        "префикс (лифтинг + блоки до последнего) один раз на "
                        "сэмпл и дифференцировать только хвост сети. Тот же "
                        "якобиан, меньше работы. --no-prefix-cache -- прежний "
                        "путь через всю сеть.")
    p.add_argument("--ggn-mc", type=int, default=0,
                   help="--ggn-target lifting: число случайных котангенсов "
                        "(обратных проходов) на train-сэмпл для полной "
                        "сэмплированной GGN. 0 -- авто: n_samples * n_mc >= "
                        "1.1 d, чтобы оценка была полного ранга.")
    p.add_argument("--kfac-mc", type=int, default=8,
                   help="Только --ggn-method kfac: число случайных котангенсов "
                        "(backward через хвост сети) на сэмпл для фактора "
                        "градиентов.")
    p.add_argument("--max-num-samples", type=int, default=25,
                   help="Макс. число сэмплов для GGN (как max_num_of_samples в luno).")
    p.add_argument("--ggn-sample-batch", type=int, default=4,
                   help="Сколько сэмплов держать на GPU одновременно при расчёте GGN.")
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
    p.add_argument("--calib-objective", choices=["chi2", "nll"], default="nll",
                   help="Objective калибровки: 'chi2' -- |chi_squared - 1| "
                        "(дефолт, как в laplax), 'nll' -- NLL.")
    p.add_argument("--no-progress", action="store_true",
                   help="Отключить прогресс-бары (tqdm).")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Мультизадачный запуск: выбор задач и пер-таск overrides
# ----------------------------------------------------------------------------

def _optional_str(value):
    """Необязательная строка: ``None`` остаётся ``None``, остальное -- ``str``."""
    return None if value is None else str(value)


# Ключи, которые можно переопределять для отдельной задачи блоком ``luno:``
# в yaml. Сюда НЕ входят: общие флаги (config/device/dtype/...), пути-веса
# (core/lift/proj/normalizers -- они берутся по task_idx) и аргументы
# стартового self-check (--check-probes/--check-tol), которые остаются общими.
# ggn_checkpoint/prior_prec_source -- явные per-task пути, а не пути по task_idx.
#
# Какие ключи допустимы, зависит от task_type (см. TASK_TYPE_LUNO_KEYS):
# GGN и калибровку настраивает только in_distr, out_distr лишь оценивается.
PER_TASK_LUNO_KEYS = {
    "ggn_target": str,
    "jv_mode": str,
    "fwd_batch": int,
    "vjp_chunk": int,
    "max_rank": int,
    "ggn_method": str,
    "ggn_oversample": int,
    "ggn_power_iter": int,
    "kfac_mc": int,
    "ggn_mc": int,
    "max_num_samples": int,
    "ggn_sample_batch": int,
    "ggn_checkpoint": _optional_str,
    "prior_prec_source": _optional_str,
    "val_calib_frac": float,
    "diag_hutchinson": int,
    "calib_max_samples": int,
    "calib_grid_min": float,
    "calib_grid_max": float,
    "calib_grid_size": int,
    "calib_objective": str,
}

TASK_TYPES = ("in_distr", "out_distr")

# Настройки оценки (прямые произведения и Хатчинсон) -- общие для обоих типов.
_EVAL_LUNO_KEYS = {"jv_mode", "fwd_batch", "vjp_chunk", "diag_hutchinson"}

TASK_TYPE_LUNO_KEYS = {
    # in_distr строит GGN (или берёт её из своего ggn_checkpoint) и сам
    # калибрует prior_prec, поэтому prior_prec_source ему не нужен.
    "in_distr": set(PER_TASK_LUNO_KEYS) - {"prior_prec_source"},
    # out_distr ничего не строит и не калибрует: метод (target, lifting/kfac/
    # iso/...) берётся у источника -- in_distr-задачи этого прогона либо
    # явных ggn_checkpoint + prior_prec_source.
    "out_distr": _EVAL_LUNO_KEYS | {"ggn_checkpoint", "prior_prec_source"},
}

_LUNO_KEY_CHOICES = {
    "ggn_target": ("last-block", "lifting"),
    "jv_mode": ("forward", "affine"),
    "ggn_method": ("randomized", "skerch", "kfac"),
    "calib_objective": ("chi2", "nll"),
}


def task_name(cfg, position):
    return cfg.get("name", f"task_{position}")


def select_tasks(task_configs, spec):
    """Выбирает задачи из ``task_configs`` по spec (``all`` или имена).

    Возвращает список ``(позиция_в_конфиге, cfg)`` в порядке конфига.
    Неизвестное имя -- ошибка со списком доступных.
    """
    if not task_configs:
        raise SystemExit("В конфиге нет задач (config.tasks пуст).")
    names = [task_name(cfg, i) for i, cfg in enumerate(task_configs)]
    spec = (spec or "all").strip()
    if spec == "all":
        return list(enumerate(task_configs))
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    if not wanted:
        raise SystemExit("--tasks пуст. Укажите 'all' или имена задач через запятую.")
    index = {n: i for i, n in enumerate(names)}
    missing = [w for w in wanted if w not in index]
    if missing:
        raise SystemExit(
            f"--tasks: неизвестные задачи {missing}. Доступные: {names}"
        )
    chosen = {index[w] for w in wanted}
    return [(i, cfg) for i, cfg in enumerate(task_configs) if i in chosen]


def task_type(cfg):
    """``task_type`` задачи: ``in_distr`` или ``out_distr`` (обязателен)."""
    value = cfg.get("task_type")
    if value not in TASK_TYPES:
        raise ValueError(
            f"задача '{task_name(cfg, '?')}': task_type={value!r}; укажите "
            f"один из {TASK_TYPES}."
        )
    return value


def apply_task_overrides(args, cfg):
    """Копия ``args`` c применённым блоком ``luno:`` задачи (валидация ключей).

    Допустимые ключи зависят от ``task_type`` (:data:`TASK_TYPE_LUNO_KEYS`):
    ключ GGN или калибровки у out_distr -- ошибка, а не молчаливый игнор.
    """
    kind = task_type(cfg)
    allowed = TASK_TYPE_LUNO_KEYS[kind]
    overrides = cfg.get("luno") or {}
    if not isinstance(overrides, dict):
        raise ValueError(
            f"задача '{task_name(cfg, '?')}': блок luno должен быть словарём, "
            f"получено {type(overrides).__name__}."
        )
    task_args = argparse.Namespace(**vars(args))
    for key, value in overrides.items():
        if key not in PER_TASK_LUNO_KEYS:
            raise ValueError(
                f"задача '{task_name(cfg, '?')}': неизвестный ключ luno.{key}; "
                f"допустимые для {kind}: {sorted(allowed)}"
            )
        if key not in allowed:
            hint = (
                "out_distr не строит GGN и не калибрует: метод и prior_prec "
                "берутся у in_distr-задачи прогона или из ggn_checkpoint + "
                "prior_prec_source"
                if kind == "out_distr" else
                "in_distr калибрует prior_prec сам"
            )
            raise ValueError(
                f"задача '{task_name(cfg, '?')}' ({kind}): ключ luno.{key} "
                f"здесь не применяется ({hint}); уберите его. Допустимые: "
                f"{sorted(allowed)}"
            )
        caster = PER_TASK_LUNO_KEYS[key]
        try:
            value = caster(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"задача '{task_name(cfg, '?')}': luno.{key}={value!r} не "
                f"приводится к {caster.__name__}: {exc}"
            )
        choices = _LUNO_KEY_CHOICES.get(key)
        if choices is not None and value not in choices:
            raise ValueError(
                f"задача '{task_name(cfg, '?')}': luno.{key}={value!r} вне "
                f"допустимых {choices}"
            )
        setattr(task_args, key, value)
    return task_args


def resolve_task_idx(cfg, position):
    """``task_idx`` задачи -- номер файла весов (тип физики).

    Если поле не задано, используется позиция задачи в конфиге (обратная
    совместимость с однозадачными конфигами вроде inference_local.yaml).
    """
    if cfg.get("task_idx") is not None:
        return int(cfg["task_idx"])
    print(
        f"[task:{task_name(cfg, position)}] task_idx не задан -> использую "
        f"позицию в конфиге ({position}); веса берутся по этому номеру."
    )
    return position


def is_in_distr(cfg):
    """Задача -- источник GGN и калибровки для своего ``task_idx``?

    Строит GGN и калибрует ``prior_prec`` только ``task_type: in_distr``.
    ``out_distr``-задачи только оцениваются с чужими GGN и prior_prec.
    """
    return task_type(cfg) == "in_distr"


class TaskSkipped(RuntimeError):
    """Источник GGN/калибровки для out_distr-задачи не найден или не годится.

    Бросается при планировании (до запуска задач), см. :func:`plan_tasks`.
    """


def load_calibration_results(directory, task_idx):
    """Читает результаты калибровки задачи ``task_idx`` (``luno_results.pkl``).

    ``directory`` -- папка in_distr-задачи, папка прогона
    (``<output-root>/<prefix>_<timestamp>``, ищется на один уровень вглубь)
    или сам файл ``luno_results.pkl``. Берётся запись с этим ``task_idx`` и
    ``prior_prec``; при нескольких -- in_distr-задача (сама калибровавшая).
    Возвращает ``(path, results)``; ``ggn_target`` старых результатов без поля
    считается last-block.
    """
    root = rmi.resolve_path(directory)
    if not root.exists():
        raise TaskSkipped(f"prior_prec_source: {root} не существует.")
    if root.is_file():
        files = [root]
    else:
        files = [root / "luno_results.pkl"] + sorted(
            d / "luno_results.pkl" for d in root.iterdir() if d.is_dir()
        )
    found = []
    for p in files:
        if not p.is_file():
            continue
        try:
            with open(p, "rb") as f:
                results = pickle.load(f)
        except Exception:
            continue
        if int(results.get("task_idx", -1)) != int(task_idx):
            continue
        if results.get("prior_prec") is None:
            continue
        found.append((p, results))
    if not found:
        raise TaskSkipped(
            f"prior_prec_source: в {root} нет luno_results.pkl с "
            f"task_idx={task_idx} и prior_prec. Укажите папку in_distr-задачи "
            f"этого task_idx."
        )
    found.sort(key=lambda pr: 0 if pr[1].get("in_distr") else 1)
    return found[0]


def find_indexed_file(directory, prefix, suffix, task_idx):
    """Файл ``{prefix}_{task_idx}{suffix}`` внутри directory (или None).

    Сначала точное имя, затем любой файл с суффиксом ``_{task_idx}``, затем
    (с предупреждением) позиция в отсортированном списке -- чтобы не падать
    на нестандартных именах вроде ``{run_prefix}_low_rank_terms.pkl``.
    """
    if directory is None:
        return None
    directory = rmi.resolve_path(directory)
    exact = directory / f"{prefix}_{task_idx}{suffix}"
    if exact.is_file():
        return exact
    files = sorted(Path(f) for f in rmi.get_all_files(str(directory), suffix))
    by_suffix = [f for f in files if f.stem.endswith(f"_{task_idx}")]
    if by_suffix:
        return by_suffix[0]
    if task_idx < len(files):
        print(
            f"[assets] точного файла {exact.name} нет; беру {files[task_idx].name} "
            f"(позиция {task_idx} в {directory})."
        )
        return files[task_idx]
    return None


def find_ggn_terms_file(directory, prefix, task_idx, suffix=".pkl"):
    """Строгий поиск low-rank GGN-артефакта по ``prefix``.

    В отличие от :func:`find_indexed_file`, не берёт позицию в общем списке:
    иначе в папке рядом с ``low_rank_terms_0.pkl`` можно случайно выбрать
    last-block-артефакт вместо ``lifting_low_rank_terms_0.pkl``. Допускаются:

    * точное ``{prefix}_{task_idx}.pkl``;
    * ``*{prefix}_{task_idx}.pkl`` (например ``{run_prefix}_...``);
    * единственный файл ``*{prefix}.pkl`` (папка одной задачи).
    """
    if directory is None:
        return None
    directory = rmi.resolve_path(directory)
    exact = directory / f"{prefix}_{task_idx}{suffix}"
    if exact.is_file():
        return exact
    files = sorted(Path(f) for f in rmi.get_all_files(str(directory), suffix))
    by_task = [
        f for f in files
        if prefix in f.stem and f.stem.endswith(f"_{task_idx}")
    ]
    if by_task:
        return by_task[0]
    same = [f for f in files if prefix in f.stem]
    if len(same) == 1:
        return same[0]
    return None


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


@dataclass
class GGNSetup:
    """Всё, что зависит от выбранного target GGN (last-block / lifting)."""

    target: str
    model_fn: callable
    w0: torch.Tensor
    wrapper: object
    jacobian_cls: type
    matvec_cls: type
    affine: object | None
    artifact_prefix: str
    # bind(x) -> (model_fn, state): пер-сэмпловая привязка, срезающая
    # замороженный префикс (см. luno_torch.affine.LastBlockSuffix). None --
    # якобиан идёт через полный model_fn.
    bind: callable | None = None
    # KFAC-контекст (--ggn-method kfac): вместо LowRankTerms по пайплайну идут
    # KFACTerms, а дисперсию считает kfac.mean_and_var.
    kfac: object | None = None
    # Лифтинг: полная GGN d x d сэмплированием выхода (luno_torch.ggn.
    # sampled_full_ggn) вместо низкорангового скетча; калибруется вместе с
    # температурой.
    full_sampled: bool = False


def build_ggn_setup(args, core, core_wrapper: TorchFNOWrapper | None,
                    core_w0: torch.Tensor | None, lifting, projection) -> GGNSetup:
    """Собирает model_fn/w0/классы якобиана для выбранного --ggn-target.

    lifting-режим использует :class:`TorchLiftingWrapper`: все параметры
    лифтинга становятся ``w0``, а core и projection заморожены. Аффинная
    декомпозиция для лифтинга пока не реализована, поэтому ``jv_mode`` там
    обязан быть ``forward``.
    """
    use_kfac = args.ggn_method == "kfac"
    if use_kfac and args.ggn_target != "last-block":
        raise ValueError(
            "--ggn-method kfac реализован только для --ggn-target last-block."
        )
    if args.ggn_target == "lifting":
        if args.jv_mode != "forward":
            raise ValueError(
                "--ggn-target lifting поддерживает только --jv-mode forward: "
                "аффинная декомпозиция для лифтинга не реализована."
            )
        try:
            for p in core.parameters():
                p.requires_grad_(False)
        except Exception:
            pass
        if projection is not None:
            try:
                for p in projection.parameters():
                    p.requires_grad_(False)
            except Exception:
                pass
        try:
            core.eval()
        except Exception:
            pass
        if projection is not None:
            try:
                projection.eval()
            except Exception:
                pass
        wrapper = TorchLiftingWrapper(
            lifting, core, projection, dtype=DTYPE
        )
        return GGNSetup(
            target="lifting",
            model_fn=wrapper.model_fn,
            w0=wrapper.w0,
            wrapper=wrapper,
            jacobian_cls=LiftingWeightJacobian,
            matvec_cls=LiftingGGNMatvec,
            affine=None,
            artifact_prefix="lifting_low_rank_terms",
            full_sampled=True,
        )

    if core_wrapper is None:
        raise ValueError(
            "--ggn-target last-block требует FNO-сплит ядра (TorchFNOWrapper), "
            "а он не построен."
        )
    model_fn = make_model_fn(core_wrapper, lifting, projection)
    affine = (
        AffineLastBlock(core_wrapper.model, projection, core_wrapper.keys, dtype=DTYPE)
        if args.jv_mode == "affine" else None
    )
    # В forward-режиме срезаем замороженный префикс: структуру блока проверяет
    # тот же AffineLastBlock. Если блок устроен иначе (norm, preactivation,
    # факторизованный вес...) -- честно идём через всю сеть.
    kfac = None
    if use_kfac:
        # Здесь отката нет: без аффинной структуры блока KFAC по модам не определён.
        kfac = KFACLastBlock(
            AffineLastBlock(
                core_wrapper.model, projection, core_wrapper.keys, dtype=DTYPE
            ),
            model_fn, core_w0,
        )
    bind = None
    if not use_kfac and args.jv_mode == "forward" and getattr(args, "prefix_cache", True):
        try:
            bind = LastBlockSuffix(
                AffineLastBlock(
                    core_wrapper.model, projection, core_wrapper.keys, dtype=DTYPE
                ),
                model_fn, core_w0,
            ).bind
            print("[prefix-cache] включён: префикс сети считается один раз на сэмпл.")
        except AffineLastBlockError as exc:
            print(f"[prefix-cache] недоступен, якобиан идёт через всю сеть: {exc}")
    return GGNSetup(
        target="last-block",
        model_fn=model_fn,
        w0=core_w0,
        wrapper=core_wrapper,
        jacobian_cls=LastFNOBlockWeightJacobian,
        matvec_cls=GGNMatvec,
        affine=affine,
        artifact_prefix="kfac_terms" if use_kfac else "low_rank_terms",
        bind=bind,
        kfac=kfac,
    )


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


def load_task_adapters(args, task_idx, asset_cache):
    """Загружает лифтинг и проекцию для типа задачи ``task_idx`` (с кэшем).

    Файлы ищутся в переданных --lift/--proj-checkpoint-dir по имени
    ``lift_{task_idx}.pt`` / ``proj_{task_idx}.pt`` (см. find_indexed_file).
    """
    key = ("adapters", task_idx)
    if key in asset_cache:
        return asset_cache[key]
    lift_path = find_indexed_file(args.lift_checkpoint_dir, "lift", ".pt", task_idx)
    proj_path = find_indexed_file(args.proj_checkpoint_dir, "proj", ".pt", task_idx)
    if lift_path is None:
        raise FileNotFoundError(
            f"task_idx={task_idx}: не найден lifting в {args.lift_checkpoint_dir} "
            f"(ожидался lift_{task_idx}.pt)."
        )
    if proj_path is None:
        raise FileNotFoundError(
            f"task_idx={task_idx}: не найдена проекция в {args.proj_checkpoint_dir} "
            f"(ожидался proj_{task_idx}.pt)."
        )
    print(f"[assets] task_idx={task_idx}: lifting={lift_path.name}, "
          f"projection={proj_path.name}")
    _install_pickle_compat()
    # map_location="cpu": чекпойнты сохранены с CUDA-тензорами; на нужное
    # устройство адаптеры переносятся в run_one_task.
    lifting = torch.load(lift_path, pickle_module=dill, map_location="cpu")
    projection = torch.load(proj_path, pickle_module=dill, map_location="cpu")
    asset_cache[key] = (lifting, projection)
    return lifting, projection


def build_task_pipeline(args, cfg, task_idx, model_config, core, asset_cache,
                        in_distr=True):
    """Данные + лоадеры + модель для ОДНОЙ задачи (адаптер по task_idx).

    ``rmi.loadData([cfg])`` возвращает датасеты одной задачи, поэтому ключ
    в батчах снова ``0`` и весь per-task код продолжает работать с ``sample[0]``.
    out_distr не калибруется, поэтому весь её val идёт на оценку
    (``calib_loader = None``).
    """
    lifting, projection = load_task_adapters(args, task_idx, asset_cache)

    train_set, val_set, test_set, _ = rmi.loadData([cfg])

    train_loader = DataLoader(
        dataset=MultiPhysicsDataset(list(train_set)),
        batch_size=1,
        shuffle=False,
    )
    if in_distr:
        calib_loader, val_eval_loader = split_val_loaders(args, val_set)
    else:
        calib_loader = None
        val_eval_loader = DataLoader(
            dataset=MultiPhysicsDataset(list(val_set)), batch_size=1, shuffle=False
        )
        print(f"[split] out_distr: весь val ({len(val_eval_loader.dataset)} "
              f"сэмплов) -> валидация, калибровки нет.")
    test_loader = DataLoader(
        dataset=MultiPhysicsDataset(list(test_set)), batch_size=1, shuffle=False
    )

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
        "core": model_core,
        "lifting": model_liftings[0],
        "projection": model_projections[0],
        "train_loader": train_loader,
        "calib_loader": calib_loader,
        "val_eval_loader": val_eval_loader,
        "test_loader": test_loader,
        "loader_channels": loader_channels,
    }


def build_data_processor(args, pipeline, task_idx):
    first_batch = next(iter(pipeline["train_loader"]))

    dims_x = {i: _get_channelwise_reduce_dims(sub["x"]) for i, sub in first_batch.items()}
    dims_y = {i: _get_channelwise_reduce_dims(sub["y"]) for i, sub in first_batch.items()}

    in_normalizer = MultiphysicsUnitGaussianNormalizer(num=1, dim=dims_x, key="x")
    in_path = find_indexed_file(args.in_normalizers, "input_normalizer", ".pkl", task_idx)
    if in_path is None:
        raise FileNotFoundError(
            f"task_idx={task_idx}: не найден input_normalizer_{task_idx}.pkl "
            f"в {args.in_normalizers}."
        )
    print(f"[norm] in-normalizer (task_idx={task_idx}): {in_path.name}")
    in_normalizer.from_file([str(in_path)])

    out_normalizer = MultiphysicsUnitGaussianNormalizer(num=1, dim=dims_y, key="y")
    out_path = find_indexed_file(args.out_normalizers, "output_normalizer", ".pkl", task_idx)
    if out_path is None:
        raise FileNotFoundError(
            f"task_idx={task_idx}: не найден output_normalizer_{task_idx}.pkl "
            f"в {args.out_normalizers}."
        )
    print(f"[norm] out-normalizer (task_idx={task_idx}): {out_path.name}")
    out_normalizer.from_file([str(out_path)])

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
                         mode="forward", fwd_batch=4, matvec_cls=GGNMatvec,
                         bind=None):
    xs = []
    for sample in tqdm(
        iter_samples(train_loader, args.max_num_samples),
        total=args.max_num_samples,
        desc="[GGN] загрузка сэмплов",
        unit="smp",
        leave=False,
    ):
        sample = preprocess_sample(data_processor, sample)
        xs.append(sample[0]["x"].detach().cpu())
    print(f"[GGN] собрано {len(xs)} сэмплов из train loader.")
    assert xs, "No samples collected for GGN."

    sample_batch_size = getattr(args, "ggn_sample_batch", 1)
    try:
        sample_batch_size = int(sample_batch_size)
    except Exception:
        sample_batch_size = 1
    ggn_mv = matvec_cls(
        model_fn, w0, xs, affine=affine, mode=mode, fwd_batch=fwd_batch,
        loss_fn="mse", factor=1.0, vjp_chunk=args.vjp_chunk,
        sample_batch_size=sample_batch_size, bind=bind,
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
        power_iter=getattr(args, "ggn_power_iter", 2),
    )
    print(f"[GGN] low-rank terms: U={tuple(low_rank.U.shape)} S={tuple(low_rank.S.shape)}")
    return low_rank


def compute_kfac_terms(args, kfac, train_loader, data_processor):
    """KFAC-факторы последнего блока по train-сэмплам (один проход)."""
    xs = []
    for sample in tqdm(
        iter_samples(train_loader, args.max_num_samples),
        total=args.max_num_samples,
        desc="[KFAC] загрузка сэмплов",
        unit="smp",
        leave=False,
    ):
        sample = preprocess_sample(data_processor, sample)
        xs.append(sample[0]["x"].detach().cpu())
    print(f"[KFAC] собрано {len(xs)} сэмплов из train loader.")
    assert xs, "No samples collected for KFAC."
    terms = kfac.fit(xs, n_mc=getattr(args, "kfac_mc", 8), seed=0)
    print(f"[KFAC] факторы: мод={terms.num_modes} {tuple(terms.mode_shape)}, "
          f"A={tuple(terms.QA.shape)} B={tuple(terms.QB.shape)}, "
          f"сэмплов={terms.n_samples}, n_mc={getattr(args, 'kfac_mc', 8)}")
    return terms


def compute_sampled_ggn(args, model_fn, w0, train_loader, data_processor,
                        jacobian_cls, fwd_batch=2, bind=None):
    """Полная GGN лифтинга сэмплированием выхода (один проход по train)."""
    xs = []
    for sample in tqdm(
        iter_samples(train_loader, args.max_num_samples),
        total=args.max_num_samples, desc="[GGN] загрузка сэмплов",
        unit="smp", leave=False,
    ):
        sample = preprocess_sample(data_processor, sample)
        xs.append(sample[0]["x"].detach().cpu())
    print(f"[GGN] собрано {len(xs)} сэмплов из train loader.")
    assert xs, "No samples collected for GGN."
    n_mc = getattr(args, "ggn_mc", 0)
    if n_mc <= 0:
        n_mc = auto_ggn_mc(w0.numel(), len(xs), fwd_batch)
        print(f"[GGN] ggn_mc=auto -> {n_mc} котангенсов на сэмпл")
    elif len(xs) * n_mc < w0.numel():
        print(f"[GGN] ВНИМАНИЕ: сэмплов x ggn_mc = {len(xs) * n_mc:,} < d = "
              f"{w0.numel():,}: оценка GGN неполного ранга, в остальных "
              f"направлениях работает только prior.")
    return sampled_full_ggn(
        model_fn, w0, xs, jacobian_cls, n_mc=n_mc, fwd_batch=fwd_batch,
        vjp_chunk=args.vjp_chunk, bind=bind,
    )


def load_kfac_terms(path, w0):
    """Читает KFACTerms из pickle; ``None``, если файла нет (как load_low_rank_ggn)."""
    p = rmi.resolve_path(path)
    if not p.is_file():
        print(f"[KFAC] checkpoint: файла нет ({p}); считаю факторы как обычно.")
        return None
    with open(p, "rb") as f:
        terms = pickle.load(f)
    if not isinstance(terms, KFACTerms):
        raise TypeError(
            f"KFAC checkpoint: ожидался KFACTerms, получен "
            f"{type(terms).__name__} ({p}). Для --ggn-method kfac нужен "
            f"*_kfac_terms.pkl, а не low-rank артефакт."
        )
    terms = terms.to(device=w0.device, dtype=w0.dtype)
    print(f"[KFAC] загружен из {p}: мод={terms.num_modes}, "
          f"сэмплов={terms.n_samples}, device={terms.QA.device}")
    return terms


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
        print(f"[GGN] checkpoint: файла нет ({p}); считаю GGN как обычно.")
        return None
    with open(p, "rb") as f:
        terms = pickle.load(f)
    return low_rank_to(terms, w0, p)


def low_rank_to(terms, w0, p):
    """Проверяет LowRankTerms против ``w0`` и переносит на его device/dtype."""
    if not isinstance(terms, LowRankTerms):
        raise TypeError(
            f"GGN checkpoint: ожидался LowRankTerms, получен "
            f"{type(terms).__name__} ({p})."
        )
    U, S = terms.U, terms.S
    if U.ndim != 2:
        raise ValueError(f"GGN checkpoint: U.ndim={U.ndim}, ожидалось 2 ({p}).")
    if S.ndim != 1 or S.numel() != U.shape[1]:
        raise ValueError(
            f"GGN checkpoint: S {tuple(S.shape)} не согласуется с U "
            f"{tuple(U.shape)} (нужен S длины U.shape[1]) ({p})."
        )
    if U.shape[0] != w0.numel():
        raise ValueError(
            f"GGN checkpoint: U.shape[0]={U.shape[0]} != w0.numel()="
            f"{w0.numel()} -- GGN от другого target (--ggn-target), модели "
            f"или сплита ({p})."
        )
    U = U.to(device=w0.device, dtype=w0.dtype)
    S = S.to(device=w0.device, dtype=w0.dtype)
    scalar = terms.scalar.to(device=w0.device) if isinstance(terms.scalar, torch.Tensor) else terms.scalar
    print(f"[GGN] загружен из {p}: U={tuple(U.shape)} S={tuple(S.shape)} "
          f"scalar={scalar}, dtype={U.dtype}, device={U.device}")
    return LowRankTerms(U=U, S=S, scalar=scalar)


def _batch_predictive_stats(model_fn, w0, weight_cov, x, out_normalizer,
                            num_output_channels, affine, mode="forward",
                            fwd_batch=4, vjp_chunk=64, diag_probes=0,
                            jacobian_cls=LastFNOBlockWeightJacobian,
                            bind=None, out_shape=None, kfac=None):
    """(mean_raw, std_raw) для одного входного тензора x в raw-пространстве.

    При ``kfac`` (KFACLastBlock) ``weight_cov`` -- пара ``(KFACTerms, prior_prec)``.

    ``out_shape`` -- форма выхода модели ``(B, C_out, ...)``; у регрессии она
    совпадает с формой target, поэтому вызывающий код передаёт её готовой и
    отдельный forward ради формы не нужен. ``bind`` -- см. ``GGNSetup.bind``.
    """
    if out_shape is None:
        with torch.no_grad():
            out_shape = model_fn(x, w0).shape
    if kfac is not None:
        terms, prior_prec = weight_cov
        mean_norm, var = kfac.mean_and_var(
            x, terms, [float(prior_prec)], n_probe=diag_probes, chunk=fwd_batch,
        )
        return raw_from_normalized(
            mean_norm, torch.sqrt(var[0]), out_normalizer, out_shape
        )
    fn, state = bind(x) if bind is not None else (model_fn, x)
    jac = jacobian_cls(
        fn,
        state,
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
                         mode="forward", fwd_batch=4,
                         jacobian_cls=LastFNOBlockWeightJacobian, bind=None,
                         kfac=None, temperature=False):
    """Калибровка scalar prior_prec на сэмплах калибровочной части val.

    ``temperature=True`` (полная GGN лифтинга): вместе с ``prior_prec``
    подбирается температура ``T``, ``Sigma = T (prec I + G)^{-1}``. При полной
    кривизне ``prior_prec`` влияет только на слабо определённые направления,
    и общий масштаб std ему не подвластен. Для NLL оптимальная ``T`` при
    фиксированном кандидате -- в замкнутом виде, ``T* = mean(r²/σ²)`` по
    валидным элементам, поэтому лишних проходов нет: кандидат оценивается по
    NLL при своей ``T*``. Только для ``objective="nll"`` (при ``chi2``
    температура делает chi2 = 1 у любого кандидата).

    Путь оценки дисперсии выбирается по стоимости: общие пробы ``(A, B, W)``
    стоят ``k + n_probe`` JVP на сэмпл (``k`` -- столбцов ``U``), пробы на
    кандидата -- ``n_grid * n_probe``. При полном базисе (``k = d``) второй
    дешевле; шум ``Z`` общий для всех кандидатов (common random numbers).

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
    "nll" -- negative log-likelihood. Обе складываются как суммы по ВАЛИДНЫМ
    элементам и делятся на их число (пер-кандидат: маска зависит от std, а std
    масштабируется sqrt(prior_prec)). Невалидные элементы (std==0/nan и
    near-zero, см. ``std_valid_mask``) отбрасываются, иначе objective определялся
    бы десятком дегенеративных пикселей. Итог усредняется по элементам всех
    сэмплов, а не по сэмплам.
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
    # KFAC: ковариация в пространстве весов не строится вовсе, дисперсию по
    # всему гриду разом отдаёт kfac.mean_and_var (общий шум на всех кандидатах).
    if kfac is not None:
        if n_probe <= 0:
            raise SystemExit("--ggn-method kfac требует --diag-hutchinson > 0.")
        covs = [None] * len(grid)
    else:
        covs = [create_luno_cov(low_rank, {"prior_prec": p}) for p in grid]
    n_grid = len(covs)
    if temperature and objective_name != "nll":
        raise SystemExit(
            "Калибровка с температурой (полная GGN лифтинга) поддерживает только "
            "--calib-objective nll: при chi2 температура делает chi2 = 1 у любого "
            "кандидата prior_prec."
        )
    k_cols = 0 if kfac is not None else int(low_rank.U.shape[1])
    per_candidate = (kfac is None and n_probe > 0
                     and k_cols + n_probe > n_grid * n_probe)
    _print_mem("calib_covs", w0.device)
    # Суммы по ВАЛИДНЫМ элементам на кандидата: r²/σ² (chi2), log σ и их число.
    # NLL собирается из них в конце (с температурой или без).
    chi2_acc = torch.zeros(n_grid, dtype=torch.float64, device=w0.device)
    logstd_acc = torch.zeros(n_grid, dtype=torch.float64, device=w0.device)
    # Число ВАЛИДНЫХ по std элементов на кандидата грида: std масштабируется
    # sqrt(prior_prec), поэтому маска (и знаменатель objective) у кандидатов
    # разные. Элементы с std==0/nan и near-zero std отбрасываются -- иначе
    # objective определяется десятком дегенеративных пикселей.
    n_valid_acc = torch.zeros(n_grid, dtype=torch.float64, device=w0.device)
    n_elem_total = 0
    num_output_channels = wrapper.num_output_channels
    how = f"Хатчинсон n_probe={n_probe}" if n_probe > 0 else "точный обратный путь"
    if per_candidate:
        how += f", пробы на кандидата ({n_grid * n_probe} JVP/сэмпл вместо {k_cols + n_probe})"
    if temperature:
        how += ", + температура"
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
            # Форма выхода модели совпадает с формой target (регрессия поле в
            # поле), поэтому отдельный forward ради неё не нужен.
            out_shape = target_raw.shape
            jac = var_grid = None
            if kfac is None:
                fn, state = bind(x) if bind is not None else (model_fn, x)
                jac = jacobian_cls(
                    fn, state, w0, affine=affine, mode=mode, fwd_batch=fwd_batch,
                    num_output_channels=num_output_channels,
                    vjp_chunk=args.vjp_chunk,
                )
            A = B = W = None
            try:
                # fx == f(x, w0) уже ПЛОСКИЙ (см. _flat_fn), а нормализатору
                # нужна форма (B, C_out, ...) для поканального broadcasting --
                # она берётся из target выше.
                if kfac is not None:
                    mean_norm, var_grid = kfac.mean_and_var(
                        x, low_rank, grid, n_probe=n_probe, chunk=fwd_batch,
                    )
                else:
                    mean_norm = jac.fx.reshape(-1)
                if mean_norm.numel() != target.numel():
                    raise RuntimeError(
                        f"Размер выхода модели ({mean_norm.numel()}) не совпал с "
                        f"target ({target.numel()}, форма {tuple(target_raw.shape)})."
                    )
                scale = torch.sqrt(torch.mean(target.double() ** 2))
                Z = None
                if per_candidate:
                    Z = torch.empty(w0.numel(), n_probe, dtype=w0.dtype,
                                    device=w0.device).bernoulli_(0.5).mul_(2).sub_(1)
                elif kfac is None and n_probe > 0:
                    A, B, W = sample_congruence_probes(jac, covs[0].U, n_probe)
                for i, cov in enumerate(covs):
                    if kfac is not None:
                        var_norm = var_grid[i]
                    elif per_candidate:
                        var_norm = _var_of_congruence_hutchinson(jac, cov, n_probe, Z=Z)
                    elif n_probe > 0:
                        var_norm = var_from_congruence_probes(A, B, W, cov)
                    else:
                        var_norm = var_of_congruence(jac, cov)
                    std_norm = torch.sqrt(var_norm)
                    mean_raw, std_raw = raw_from_normalized(
                        mean_norm, std_norm, out_normalizer, out_shape
                    )
                    valid = std_valid_mask(std_raw, scale)
                    sr = standardized_residual(mean_raw, std_raw, target)
                    valid = valid & torch.isfinite(sr)
                    n_valid = int(valid.sum())
                    if n_valid == 0:
                        continue
                    sr_v = sr[valid].double()
                    std_v = std_raw[valid].double()
                    chi2_acc[i] += (sr_v ** 2).sum()
                    logstd_acc[i] += std_v.log().sum()
                    n_valid_acc[i] += n_valid
                n_elem_total += target.numel()
            finally:
                # Граф forward-замыкания (model_fn, x) держится за self, а в
                # точном пути -- ещё и обратный граф. Освобождаем сразу: сам jac
                # подавится привязкой в начале следующей итерации.
                if jac is not None:
                    jac._release_vjp()
            bar.update(1)
    finally:
        bar.close()

    if not n_elem_total:
        raise SystemExit(
            f"Калибровка не увидела ни одного сэмпла: len(calib_loader.dataset)="
            f"{n_total}, --calib-max-samples={args.calib_max_samples}."
        )
    n_valid_safe = n_valid_acc.clamp_min(1.0)
    half_log_2pi = 0.5 * math.log(2.0 * math.pi)
    temps = None
    if objective_name == "chi2":
        objective = (chi2_acc / n_valid_safe - 1.0).abs()
    elif temperature:
        # NLL(T) = 0.5 chi2/T + 0.5 log T + mean log σ + const -> T* = mean chi2.
        temps = chi2_acc / n_valid_safe
        objective = 0.5 + 0.5 * temps.log() + logstd_acc / n_valid_safe + half_log_2pi
    else:
        objective = (0.5 * chi2_acc + logstd_acc) / n_valid_safe + half_log_2pi
    if not bool(torch.isfinite(objective).all()):
        raise RuntimeError(
            f"Objective калибровки не конечен ({objective.tolist()}, "
            f"n_calib={n_calib}, n_probe={n_probe}): var неотрицателен по "
            f"построению при n_probe>0, поэтому дело в target или в std."
        )
    best_idx = int(objective.argmin())
    best_prec = grid[best_idx]
    out = {"prior_prec": best_prec, "objective": objective_name,
           "samples": n_calib, "diag_probes": n_probe}
    t_str = ""
    if temps is not None:
        out["temperature"] = float(temps[best_idx])
        t_str = f", temperature={out['temperature']:.6e}"
    edge = (" -- ВНИМАНИЕ: оптимум на краю грида, расширьте "
            "calib_grid_min/calib_grid_max"
            if best_idx in (0, n_grid - 1) else "")
    print(f"[calibrate] best prior_prec={best_prec.item():.6e}{t_str}, "
          f"{objective_name}={float(objective[best_idx]):.6e} "
          f"(на {n_calib} сэмплах, {n_grid} кандидатов){edge}")
    return out


def mask_image_outliers(x, vmax=10.0):
    """Копия ``x``, где non-finite и значения > ``vmax`` заменены на nan.

    Только для визуализации sqrt(chi2): один inf/огромное значение растягивает
    автомасштаб ``imshow`` и делает остальную картинку нечитаемой. nan
    matplotlib рисует как пустой пиксель и не учитывает в vmin/vmax, поэтому
    шкала строится по валидным значениям.
    """
    x = x.clone()
    x[~torch.isfinite(x) | (x > vmax)] = float("nan")
    return x


def save_luno_images(pred, band, target, output_prefix, batch_idx):
    """Картинки LUNO: target / predict (mean_raw) / std_raw / sqrt(chi2).

    sqrt(chi2) = |target - mean_raw| / std_raw -- калиброванная стандартизованная
    ошибка по элементам. На этой панели значения >10 и non-finite заменяются на
    nan (``mask_image_outliers``), чтобы выбросы не растягивали цветовую шкалу;
    остальные панели пишутся как есть.
    Имена файлов: <prefix>_<key>_<idx>_<suffix>_.png.
    """
    for key in pred.keys():
        time_index = _inference_time_index(pred[key])
        p, b, t = pred[key], band[key], target[key]
        eps = torch.finfo(p.dtype).eps
        sqrt_chi2 = (t - p).abs() / b.clamp_min(eps)
        sqrt_chi2 = mask_image_outliers(sqrt_chi2)

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


def mean_finite(values) -> float:
    """Среднее по конечным значениям; nan, если конечных нет вообще.

    Нужно для метрик, которые считаются на сэмпл и могут дать nan на
    вырожденном сэмпле (например, AUSE с нулевым знаменателем): один плохой сэмпл не должен
    обнулять всю метрику.
    """
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def channel_key_names(channel_names, n_ch) -> list:
    """Имена каналов для ключей метрик: из конфига или ch0..ch{C-1}.

    ``channel_names`` -- adapter.variable_names (порядок = ось канала).
    Конфиг не обязан совпадать с моделью (другой бенчмарк, переименование),
    поэтому при расхождении длины молча уходим на индексы: лучше потерять
    подписи, чем подписать канал чужим именем.
    """
    if channel_names and len(channel_names) == n_ch:
        return [str(name) for name in channel_names]
    return [f"ch{i}" for i in range(n_ch)]


def per_channel_p99(pred, std, target, pct=PER_CHANNEL_PCT) -> list:
    """Перцентили ``pct`` элементного sqrt(chi2) = |pred - target| / std ПО КАЖДОМУ КАНАЛУ.

    Формы ``(B=1, C, ...)``: ось канала = 1 (см.
    ``_get_channelwise_reduce_dims`` и ``canonical_image``), режем как
    ``t[0, c]``. Перцентиль берётся ПО ЭЛЕМЕНТАМ внутри одного канала (все
    пиксели и моменты времени) -- это хвосты распределения стандартизованного
    остатка, посмотренные отдельно на каждом поле.

    ``pct`` -- одно число или последовательность уровней (по умолчанию
    ``PER_CHANNEL_PCT``). Возвращает список СПИСКОВ: по одному списку длины C
    на каждый уровень, в порядке ``pct`` (для одного числа -- список из одного
    списка).

    Фильтра по величине std нет: обработка ``std == 0`` -- см.
    :func:`_grouped_sqrt_chi2_quantiles`.
    """
    if pred.shape[0] != 1:
        raise ValueError(
            f"per_channel_p99: ожидался батч из одного сэмпла (B=1), "
            f"получено {pred.shape[0]}"
        )
    return _grouped_sqrt_chi2_quantiles(pred, std, target, pct, pred.shape[1])


def per_channel_time_sqrt_chi2(pred, std, target, pct=PER_CHANNEL_PCT) -> list:
    """Те же перцентили sqrt(chi2), что в :func:`per_channel_p99`, но внутри
    каждой пары (канал, момент времени).

    Формы ``(B=1, C, T, *space)`` (ось канала 1, ось времени 2); без оси времени
    (``ndim < 5``) ``T = 1``. Обработка ``std == 0`` -- та же, что у
    поканальной версии. Возвращает список по уровням ``pct``,
    в каждом -- ``C`` списков длины ``T``.
    """
    if pred.shape[0] != 1:
        raise ValueError(
            f"per_channel_time_sqrt_chi2: ожидался батч из одного сэмпла (B=1), "
            f"получено {pred.shape[0]}"
        )
    n_ch = pred.shape[1]
    n_t = pred.shape[2] if pred.dim() >= 5 else 1
    flat = _grouped_sqrt_chi2_quantiles(pred, std, target, pct, n_ch * n_t)
    return [[row[c * n_t:(c + 1) * n_t] for c in range(n_ch)] for row in flat]


def _grouped_sqrt_chi2_quantiles(pred, std, target, pct, n_groups) -> list:
    """Перцентили ``|pred - target| / std`` по группам ``reshape(n_groups, -1)``.

    Фильтра по величине std нет. Отношение при ``std == 0`` и ненулевом
    остатке -- ``+inf``: такой пиксель честно лежит в верхнем хвосте и меняет
    квантиль, только если таких пикселей больше ``1 - q``. Отбрасываются лишь
    неопределённые отношения: ``0/0`` (нулевые и остаток, и std) и NaN.

    ``nanquantile`` интерполирует между соседними порядковыми статистиками, и
    при ``inf`` на границе это даёт ``inf*0 = nan``. Поэтому ``inf`` на время
    подменяется большим конечным числом, а результат выше порога возвращается
    в ``inf``. Пустая группа даёт nan.

    Возвращает список по уровням ``pct``, в каждом -- ``n_groups`` значений.
    """
    levels = [pct] if isinstance(pct, (int, float)) else list(pct)
    if pred.numel() == 0:
        return [[float("nan")] * n_groups for _ in levels]
    ratio = ((pred - target).abs().double() / std.double()).reshape(n_groups, -1)
    big = torch.finfo(torch.float64).max / 4
    ratio = torch.where(torch.isposinf(ratio), torch.full_like(ratio, big), ratio)
    # Уровни -- на том же устройстве, что и ratio (иначе quantile падает на CUDA).
    q_levels = torch.tensor(
        [p / 100.0 for p in levels], dtype=ratio.dtype, device=ratio.device
    )
    q = torch.nanquantile(ratio, q_levels, dim=1)  # (n_levels, n_groups)
    q = torch.where(q >= big / 2, torch.full_like(q, float("inf")), q)
    return [[float(v) for v in row] for row in q]


def mean_non_nan(values) -> float:
    """Среднее, где пропускаются только nan, а ``inf`` сохраняется.

    Для квантилей sqrt(chi2): ``inf`` у сэмпла значит, что больше ``1 - q``
    пикселей имеют ``std == 0`` при ненулевой ошибке, и это должно быть видно в
    итоговом числе, а не молча выпасть из среднего.
    """
    kept = [v for v in values if not math.isnan(v)]
    return sum(kept) / len(kept) if kept else float("nan")


def ause_by_channel_time(pred, std, target) -> list:
    """Нормированная AUSE для каждой пары (канал, момент времени) одного сэмпла.

    Формы ``(B=1, C, T, *space)`` -- ось канала 1, ось времени 2 (как у
    временных адаптеров, ``data_order: TXY``). Для полей без оси времени
    (``ndim < 5``) вся пространственная часть канала -- одна группа, ``T = 1``.

    Пиксели ранжируются внутри одного поля (канал, t): при общей сортировке
    порядок определялся бы разницей масштабов каналов и ростом ошибки со
    временем, а не тем, где std внутри поля указывает на ошибку.

    Возвращает список ``C`` списков длины ``T`` (см. :func:`normalized_ause`).
    """
    if pred.shape[0] != 1:
        raise ValueError(
            f"ause_by_channel_time: ожидался батч из одного сэмпла (B=1), "
            f"получено {pred.shape[0]}"
        )
    n_ch = pred.shape[1]
    n_t = pred.shape[2] if pred.dim() >= 5 else 1
    err2 = (pred - target) ** 2
    ratio = normalized_ause(std.reshape(n_ch * n_t, -1), err2.reshape(n_ch * n_t, -1))
    return ratio.reshape(n_ch, n_t).tolist()


def reliability_by_channel_time(pred, std, target, n_bins=ENCE_BINS):
    """Суммы для кривой надёжности по бинам std внутри каждой пары (канал, t).

    Раскладка осей как у :func:`ause_by_channel_time`. Возвращает
    ``(sum_std2, sum_err2, count)`` формы ``(C, T, n_bins)`` в float64 на CPU --
    их складывают по сэмплам (см. :func:`metrics_uq.reliability_bin_sums`).
    """
    if pred.shape[0] != 1:
        raise ValueError(
            f"reliability_by_channel_time: ожидался батч из одного сэмпла (B=1), "
            f"получено {pred.shape[0]}"
        )
    n_ch = pred.shape[1]
    n_t = pred.shape[2] if pred.dim() >= 5 else 1
    sums = reliability_bin_sums(
        std.reshape(n_ch * n_t, -1), ((pred - target) ** 2).reshape(n_ch * n_t, -1),
        n_bins=n_bins,
    )
    return tuple(x.reshape(n_ch, n_t, n_bins).cpu() for x in sums)


def evaluate_luno(args, loader, model_fn, w0, low_rank, prior_args, wrapper,
                  data_processor, out_normalizer, task_name, max_samples,
                  affine, metrics_config=None, output_dir=None,
                  mode="forward", fwd_batch=4,
                  save_images=True, diag_probes=0, channel_names=None,
                  jacobian_cls=LastFNOBlockWeightJacobian, bind=None,
                  kfac=None):
    """Метрики неопределённости и точности на eval-сэмплах, ВСЕ -- по каналам.

    Ни одна метрика не усредняет по каналам: у каналов разный масштаб и разное
    поведение std, и общее число смешивало бы их. ``{канал}`` -- имя из
    ``adapter.variable_names`` (при расхождении с числом выходов -- ``ch0..``),
    ``{tt}`` -- момент времени, ``{bb}`` -- бин std.

    По каналу (среднее по сэмплам):

    * ``luno_nll_{канал}`` -- гауссов NLL на элемент. Единственная метрика с
      фильтром по std: элементы с ``std < STD_REL_FLOOR * RMS(target канала)``
      (и неконечным остатком/std) отбрасываются, иначе ``log std`` и
      ``остаток²/std²`` уходят в бесконечность. Сколько отброшено --
      ``luno_valid_percent_{канал}`` и ``luno_dropped_elements_{канал}``;
    * ``luno_rmse_{канал}`` -- RMSE среднего (raw), среднее по сэмплам;
    * ``luno_std_mean_{канал}`` -- средний предсказанный std (raw) --
      резкость; без него калибровку можно "купить" раздуванием std;
    * ``luno_sqrt_chi2_{канал}_p{q}`` -- квантили ``|остаток|/std`` по всем
      пикселям и моментам канала, ``q`` из ``PER_CHANNEL_PCT``; ориентиры
      гаусса -- ``GAUSS_ABS_QUANTILES``;
    * ``luno_ence_{канал}``, ``luno_ause_{канал}`` -- среднее по времени
      значений ниже;
    * метрики точности из ``metrics_config`` -- ``{имя}_{канал}``, посчитанные
      на срезе одного канала.

    По (канал, t): ``luno_sqrt_chi2_{канал}_t{tt}_p{q}``,
    ``luno_ence_{канал}_t{tt}``, ``luno_ause_{канал}_t{tt}`` и кривые
    надёжности ``luno_reliability_{канал}_t{tt}_b{bb}_{logratio|rmv|rmse}``.

    Кроме NLL фильтра по std нет: ``std == 0`` при ненулевой ошибке даёт
    ``+inf`` (квантили, ENCE), ``0/0`` отбрасывается; AUSE зависит только от
    порядка std. В лог идёт по строке на канал, таблицы по (канал, t) -- только
    в возвращаемый словарь (дальше -- в pkl).

    ``save_images=False`` -- не писать PNG. ``diag_probes`` -- число проб
    Хатчинсона для ``diag(J Sigma J^T)``; 0 -- точный обратный режим.
    """
    if kfac is not None:
        if diag_probes <= 0:
            raise SystemExit("--ggn-method kfac требует --diag-hutchinson > 0.")
        cov = (low_rank, prior_args["prior_prec"])
    else:
        cov = create_luno_cov(low_rank, prior_args)

    # Поканальные накопители создаются на первом сэмпле (число каналов
    # известно только по выходу). Суммы по сэмплам -- float64 на CPU.
    ch = None  # dict накопителей
    ch_p_samples = []  # по сэмплу: по уровням PER_CHANNEL_PCT список длины C
    ause_samples = []  # по сэмплу список (C x T) нормированной AUSE
    ch_t_pct_samples = []  # по сэмплу: по уровням список (C x T)
    # Накопленные по сэмплам суммы для ENCE: (C, T, ENCE_BINS) каждая.
    rel_std2 = rel_err2 = rel_count = None
    cfg_sums = {}  # {(имя, c): сумма по сэмплам}
    n_samples = 0

    output_prefix = None
    if output_dir is not None and save_images:
        output_prefix = Path(output_dir) / f"inspections_luno_{task_name}"
        output_prefix.mkdir(parents=True, exist_ok=True)
        output_prefix = Path.joinpath(output_prefix, "luno_")

    # Графы здесь не нужны: без no_grad накопители цеплялись бы через grad_fn,
    # и граф каждого eval-сэмпла жил бы до конца цикла.
    with torch.no_grad():
        for sample in iter_samples(loader, max_samples):
            sample = preprocess_sample(data_processor, sample)
            x = sample[0]["x"]
            target_raw = sample[0]["y"]
            out_shape = target_raw.shape
            mean_raw, std_raw = _batch_predictive_stats(
                model_fn, w0, cov, x, out_normalizer, wrapper.num_output_channels,
                affine=affine, mode=mode, fwd_batch=fwd_batch, vjp_chunk=args.vjp_chunk,
                diag_probes=diag_probes, jacobian_cls=jacobian_cls,
                bind=bind, out_shape=out_shape, kfac=kfac,
            )
            n_ch = out_shape[1]
            m_c = mean_raw.reshape(n_ch, -1).double()
            s_c = std_raw.reshape(n_ch, -1).double()
            y_c = target_raw.reshape(n_ch, -1).double()
            if ch is None:
                zeros = lambda: torch.zeros(n_ch, dtype=torch.float64)
                ch = {k: zeros() for k in
                      ("nll", "n_valid", "n_elem", "rmse", "std_mean")}

            # NLL по каналу -- с фильтром по std относительно RMS target ЭТОГО
            # канала (каналы разного масштаба, общий RMS резал бы мелкие).
            scale = torch.sqrt(torch.mean(y_c ** 2, dim=1, keepdim=True))
            sr = (m_c - y_c).abs() / s_c
            valid = std_valid_mask(s_c, scale) & torch.isfinite(sr)
            nll_el = 0.5 * sr ** 2 + torch.log(s_c) + 0.5 * math.log(2.0 * math.pi)
            ch["nll"] += torch.where(valid, nll_el, torch.zeros_like(nll_el)).sum(dim=1).cpu()
            ch["n_valid"] += valid.sum(dim=1).double().cpu()
            ch["n_elem"] += y_c.shape[1]
            ch["rmse"] += torch.sqrt(torch.mean((m_c - y_c) ** 2, dim=1)).cpu()
            ch["std_mean"] += torch.nanmean(s_c, dim=1).cpu()
            n_samples += 1

            pred = {0: mean_raw.reshape(out_shape)}
            band = {0: std_raw.reshape(out_shape)}
            target_dict = {0: target_raw}

            # Квантили sqrt(chi2) по каналу (по всем пикселям и моментам).
            ch_p_samples.append(per_channel_p99(pred[0], band[0], target_dict[0]))
            # Sparsification внутри каждого поля (канал, t).
            ause_samples.append(
                ause_by_channel_time(pred[0], band[0], target_dict[0])
            )
            # Хвосты sqrt(chi2) внутри каждого поля (канал, t).
            ch_t_pct_samples.append(
                per_channel_time_sqrt_chi2(pred[0], band[0], target_dict[0])
            )
            # ENCE: суммы std² / err² по бинам std внутри (канал, t). Копим
            # суммы, а не логарифмы на сэмпл: так меньше шума.
            s2, e2, cnt = reliability_by_channel_time(
                pred[0], band[0], target_dict[0]
            )
            # Поле с неконечным std/ошибкой у этого сэмпла в суммы не входит
            # (иначе один nan отравил бы пару (канал, t) для всех сэмплов).
            bad = torch.isnan(s2) | torch.isnan(e2)
            s2[bad], e2[bad], cnt[bad] = 0.0, 0.0, 0.0
            if rel_std2 is None:
                rel_std2, rel_err2, rel_count = s2, e2, cnt
            else:
                rel_std2 += s2
                rel_err2 += e2
                rel_count += cnt

            if output_prefix is not None:
                save_luno_images(pred, band, target_dict, output_prefix, n_samples - 1)

            if metrics_config:
                # Метрики точности из конфига сворачивают ось канала, поэтому
                # считаются на срезе каждого канала отдельно.
                for c in range(n_ch):
                    sl = slice(c, c + 1)
                    batch_metrics = rmi.compute_batch_metrics(
                        {0: pred[0][:, sl]},
                        {0: band[0][:, sl]},
                        {0: target_dict[0][:, sl]},
                        metrics_config=metrics_config,
                        task_name=task_name,
                    )
                    for name, value in batch_metrics[0].items():
                        cfg_sums[(name, c)] = cfg_sums.get((name, c), 0.0) + float(value)

    if ch is None:
        raise SystemExit(f"[eval:{task_name}] в loader нет ни одного сэмпла.")
    n_ch = ch["nll"].numel()
    keys = channel_key_names(channel_names, n_ch)

    base = {}
    for c, key in enumerate(keys):
        n_valid = float(ch["n_valid"][c])
        n_elem = float(ch["n_elem"][c])
        base[f"luno_nll_{key}"] = (float(ch["nll"][c]) / n_valid
                                   if n_valid else float("nan"))
        base[f"luno_rmse_{key}"] = float(ch["rmse"][c]) / n_samples
        base[f"luno_std_mean_{key}"] = float(ch["std_mean"][c]) / n_samples
        base[f"luno_valid_percent_{key}"] = 100.0 * n_valid / n_elem
        base[f"luno_dropped_elements_{key}"] = int(n_elem - n_valid)

    # Квантили sqrt(chi2) по каналу: среднее по сэмплам (inf сохраняется, nan
    # пропускается, см. mean_non_nan).
    ch_pct = {}
    for li, q in enumerate(PER_CHANNEL_PCT):
        for c, key in enumerate(keys):
            ch_pct[f"luno_sqrt_chi2_{key}_p{q}"] = mean_non_nan(
                smp[li][c] for smp in ch_p_samples
            )

    # Нормированная AUSE: по (канал, t) и среднее по времени на канал.
    ause = {}
    if ause_samples:
        n_t = len(ause_samples[0][0])
        for c, key in enumerate(keys):
            per_t = [
                mean_finite(smp[c][t] for smp in ause_samples) for t in range(n_t)
            ]
            for t, v in enumerate(per_t):
                ause[f"luno_ause_{key}_t{t:02d}"] = v
            ause[f"luno_ause_{key}"] = mean_finite(per_t)

    # Перцентили sqrt(chi2) по (канал, t): среднее по сэмплам для каждого уровня.
    ch_t_pct = {}
    if ch_t_pct_samples:
        n_t = len(ch_t_pct_samples[0][0][0])
        for li, q in enumerate(PER_CHANNEL_PCT):
            for c, key in enumerate(keys):
                for t in range(n_t):
                    ch_t_pct[f"luno_sqrt_chi2_{key}_t{t:02d}_p{q}"] = mean_non_nan(
                        smp[li][c][t] for smp in ch_t_pct_samples
                    )

    # ENCE = средний по бинам |log(RMSE/RMV)| для каждой пары (канал, t) и
    # среднее по времени на канал. Кривая надёжности -- для каждой пары
    # (канал, t): бин b -- b-я доля самых уверенных пикселей этого поля.
    ence = {}
    if rel_std2 is not None:
        _, n_t, n_bins = rel_std2.shape
        lr = log_rmse_over_rmv(rel_err2, rel_std2)  # (C, T, B)
        for c, key in enumerate(keys):
            per_t = [mean_non_nan(lr[c, t].abs().tolist()) for t in range(n_t)]
            for t, v in enumerate(per_t):
                ence[f"luno_ence_{key}_t{t:02d}"] = v
            ence[f"luno_ence_{key}"] = mean_non_nan(per_t)
            curves = lr[c].tolist()  # (T, B)
            rmv = torch.sqrt(rel_std2[c] / rel_count[c]).tolist()
            rmse_b = torch.sqrt(rel_err2[c] / rel_count[c]).tolist()
            for t in range(n_t):
                for b in range(n_bins):
                    pre = f"luno_reliability_{key}_t{t:02d}_b{b:02d}"
                    ence[f"{pre}_logratio"] = curves[t][b]
                    ence[f"{pre}_rmv"] = rmv[t][b]
                    ence[f"{pre}_rmse"] = rmse_b[t][b]

    cfg_metrics = {
        f"{name}_{keys[c]}": value / n_samples
        for (name, c), value in cfg_sums.items()
    }
    metrics = {
        **cfg_metrics,
        **base,
        **ch_pct,
        **ch_t_pct,
        **ause,
        **ence,
        "samples": n_samples,
    }

    # Лог: по строке на канал; таблицы по (канал, t), кривые надёжности и
    # метрики точности из конфига -- только в файл.
    refs = " ".join(f"p{q}={GAUSS_ABS_QUANTILES[q]}" for q in PER_CHANNEL_PCT
                    if q in GAUSS_ABS_QUANTILES)
    print(f"[eval:{task_name}] {n_samples} сэмплов; по каналам (sqrt_chi2 -- "
          f"квантили |остаток|/std, гаусс: {refs}; ENCE, AUSE -- среднее по времени):")
    for key in keys:
        pct = " ".join(f"p{q}={ch_pct[f'luno_sqrt_chi2_{key}_p{q}']:.3f}"
                       for q in PER_CHANNEL_PCT)
        extra = ""
        if ence:
            extra += f" | ENCE={ence[f'luno_ence_{key}']:.4f}"
        if ause:
            extra += f" AUSE={ause[f'luno_ause_{key}']:.4f}"
        print(f"[eval:{task_name}]   {key}: nll={base[f'luno_nll_{key}']:.4e} "
              f"rmse={base[f'luno_rmse_{key}']:.4e} "
              f"std_mean={base[f'luno_std_mean_{key}']:.4e} "
              f"valid={base[f'luno_valid_percent_{key}']:.2f}% | "
              f"sqrt_chi2 {pct}{extra}")
    print(f"[eval:{task_name}] только в файле: по (канал, t) -- "
          f"luno_sqrt_chi2_{{канал}}_t{{tt}}_p{{q}}, luno_ence_{{канал}}_t{{tt}}, "
          f"luno_ause_{{канал}}_t{{tt}}, "
          f"luno_reliability_{{канал}}_t{{tt}}_b{{bb}}_{{logratio|rmv|rmse}}"
          + ("; метрики точности из конфига -- {имя}_{канал}" if cfg_metrics else ""))
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
    calib = pipeline["calib_loader"]
    n_calib = len(calib.dataset) if calib is not None else 0
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

def uq_method(args) -> str:
    """Метка метода UQ по эффективным настройкам задачи.

    Лифтинг: ``lifting`` -- полная сэмплированная GGN, ``lifting_iso`` --
    ``max_rank == 0`` (только изотропный prior). Последний блок:
    ``lastblock_{kfac|iso|lowrank}`` (``--ggn-method kfac``, ``max_rank == 0``
    или низкоранговая GGN; её фактический ранг -- в ``ggn_rank`` результатов,
    он может прийти из ``ggn_checkpoint``).
    """
    if args.ggn_target == "lifting":
        return "lifting_iso" if args.max_rank == 0 else "lifting"
    target = "lastblock"
    if args.ggn_method == "kfac":
        method = "kfac"
    elif args.max_rank == 0:
        method = "iso"
    else:
        method = "lowrank"
    return f"{target}_{method}"


def run_method_label(plans) -> str:
    """Метка метода для имени папки прогона (см. :func:`plan_tasks`).

    Метод определяют in_distr-задачи; если их в прогоне нет -- источники
    out_distr-задач (ggn_checkpoint/prior_prec_source). Разные методы
    склеиваются через ``+``.
    """
    in_distr = [p.method for p in plans if p.in_distr]
    labels = in_distr or [p.method for p in plans]
    return "+".join(dict.fromkeys(labels))


def method_from_terms(target: str, terms) -> str:
    """Метка метода по реально использованной кривизне (см. :func:`uq_method`).

    Для задач, унаследовавших GGN (из кэша in_distr-соседа или из
    ``ggn_checkpoint``), настройки самой задачи ничего не говорят -- метод
    определяется содержимым: ``KFACTerms`` -> kfac, ``U`` с нулём столбцов ->
    iso, иначе lowrank.
    """
    t = "lifting" if target == "lifting" else "lastblock"
    if isinstance(terms, KFACTerms):
        return f"{t}_kfac"
    if terms.U.shape[1] == 0:
        return f"{t}_iso"
    if terms.U.shape[1] == terms.U.shape[0]:
        return "lifting" if t == "lifting" else f"{t}_full"
    return f"{t}_lowrank"


def iso_low_rank(w0: torch.Tensor) -> LowRankTerms:
    """Пустая кривизна для метода Iso: ``U`` формы ``(d, 0)``.

    Тогда ``create_luno_cov`` даёт ковариацию ``I / prior_prec`` -- чистый
    изотропный prior в пространстве весов, а предиктивная дисперсия --
    ``diag(J J^T) / prior_prec``.
    """
    return LowRankTerms(
        U=torch.zeros(w0.numel(), 0, dtype=w0.dtype, device=w0.device),
        S=torch.zeros(0, dtype=w0.dtype, device=w0.device),
        scalar=0.0,
    )


def resolve_low_rank(args, w0, model_fn, pipeline, data_processor, affine, task_idx,
                     output_dir, run_prefix, asset_cache,
                     matvec_cls=GGNMatvec, artifact_prefix="low_rank_terms",
                     bind=None, kfac=None, iso=False, full_sampled=False,
                     jacobian_cls=None, allow_compute=True):
    """Возвращает (low_rank, ggn_source, ggn_checkpoint) для одной задачи.

    ``iso=True`` (in_distr-задача с ``max_rank == 0``): GGN не строится и
    ``ggn_checkpoint`` не читается -- возвращается пустая кривизна
    (:func:`iso_low_rank`), ``ggn_source = "iso"``. Артефакт с ``U`` формы
    ``(d, 0)`` всё равно сохраняется, чтобы OOD-задачи могли унаследовать его
    через ``ggn_checkpoint``, как и низкоранговую GGN.

    ``full_sampled=True`` (лифтинг): GGN полная, ``d x d`` (см.
    :func:`compute_sampled_ggn`). Низкоранговый чекпойнт (``U`` меньше ``d``
    столбцов, например от прежнего метода) для неё не годится -- он
    игнорируется с предупреждением, и GGN считается заново.

    В ``args.ggn_checkpoint`` уже лежит эффективный путь: собственный
    ``luno.ggn_checkpoint`` in_distr-задачи, унаследованный от неё checkpoint
    для остальных задач или глобальный ``--ggn-checkpoint``. Файл используется
    всеми задачами; папка -- берётся файл этой задачи
    ``{artifact_prefix}_{task_idx}.pkl`` (с кэшем на task_idx). Если файла нет --
    GGN считается на train-лоадере и сохраняется в подпапке задачи как
    ``{run_prefix}_{artifact_prefix}.pkl``.

    ``artifact_prefix``/``matvec_cls`` выбираются по ``--ggn-target``:
    last-block использует старые имена, lifting -- ``lifting_low_rank_terms``.
    """
    if iso:
        if args.ggn_checkpoint:
            print(f"[GGN] max_rank=0 (метод Iso): ggn_checkpoint "
                  f"{args.ggn_checkpoint} игнорируется -- кривизна не нужна.")
        print("[GGN] метод Iso: GGN не строится, ковариация -- изотропный prior "
              "I / prior_prec.")
        low_rank = iso_low_rank(w0)
        terms_path = output_dir / f"{run_prefix}_{artifact_prefix}.pkl"
        with open(terms_path, "wb") as f:
            pickle.dump(LowRankTerms(U=low_rank.U.cpu(), S=low_rank.S.cpu(),
                                     scalar=0.0), f)
        print(f"[save] {artifact_prefix} (Iso, U=(d, 0)) -> {terms_path}")
        return low_rank, "iso", None

    ggn_path = None
    if args.ggn_checkpoint:
        raw = rmi.resolve_path(args.ggn_checkpoint)
        if raw.is_dir():
            if artifact_prefix == "low_rank_terms":
                ggn_path = find_indexed_file(
                    args.ggn_checkpoint, artifact_prefix, ".pkl", task_idx
                )
            else:
                # Строгий поиск: рядом могут лежать last-block-артефакты.
                ggn_path = find_ggn_terms_file(
                    raw, artifact_prefix, task_idx, ".pkl"
                )
            if ggn_path is None:
                print(f"[GGN] в {raw} нет {artifact_prefix}_{task_idx}.pkl; "
                      f"считаю GGN как обычно.")
        else:
            ggn_path = raw

    low_rank = None
    if ggn_path is not None:
        cache_key = ("ggn", task_idx, str(ggn_path))
        if cache_key in asset_cache:
            low_rank = asset_cache[cache_key]
            print(f"[GGN] из кэша (task_idx={task_idx}): {ggn_path}")
        else:
            low_rank = (load_kfac_terms(ggn_path, w0) if kfac is not None
                        else load_low_rank_ggn(ggn_path, w0))
            if (low_rank is not None and full_sampled
                    and low_rank.U.shape[1] < w0.numel()):
                print(f"[GGN] чекпойнт {ggn_path} низкоранговый "
                      f"(ранг {low_rank.U.shape[1]} < d={w0.numel()}): для "
                      f"лифтинга нужна полная GGN -- чекпойнт игнорируется, "
                      f"считаю заново.")
                low_rank = None
            if low_rank is not None:
                asset_cache[cache_key] = low_rank

    ggn_loaded = low_rank is not None
    if not ggn_loaded and not allow_compute:
        # Задача без in_distr-соседа: считать GGN на её собственных данных
        # нельзя (это была бы кривизна другой задачи) -- нужен чекпойнт.
        return None, "missing", None
    if not ggn_loaded and kfac is not None:
        low_rank = compute_kfac_terms(
            args, kfac, pipeline["train_loader"], data_processor
        )
        low_rank_cpu = low_rank.to(device="cpu")
    elif not ggn_loaded and full_sampled:
        low_rank = compute_sampled_ggn(
            args, model_fn, w0, pipeline["train_loader"], data_processor,
            jacobian_cls, fwd_batch=args.fwd_batch, bind=bind,
        )
        low_rank_cpu = LowRankTerms(
            U=low_rank.U.detach().to("cpu"), S=low_rank.S.detach().to("cpu"),
            scalar=low_rank.scalar,
        )
    elif not ggn_loaded:
        low_rank = compute_low_rank_ggn(
            args, model_fn, w0, pipeline["train_loader"], data_processor, affine,
            mode=args.jv_mode, fwd_batch=args.fwd_batch,
            matvec_cls=matvec_cls, bind=bind,
        )
        # U имеет форму (d, rank) -- при d = 27.6M и rank = 50 это 5.15 ГиБ. Пишем
        # артефакт с CPU-копией: сам pickle всё равно такой размер, но GPU-копия
        # продолжает жить в low_rank для калибровки и метрик. При загрузке из
        # --ggn-checkpoint шаг пропускается: артефакт уже есть в чекпойнте.
        low_rank_cpu = LowRankTerms(
            U=low_rank.U.detach().to("cpu"), S=low_rank.S.detach().to("cpu"),
            scalar=low_rank.scalar,
        )
    if not ggn_loaded:
        terms_path = output_dir / f"{run_prefix}_{artifact_prefix}.pkl"
        with open(terms_path, "wb") as f:
            pickle.dump(low_rank_cpu, f)
        print(f"[save] {artifact_prefix} -> {terms_path} "
              f"({terms_path.stat().st_size / 1024 ** 3:.2f} GiB)")

    ggn_source = "checkpoint" if ggn_loaded else "computed"
    ggn_checkpoint = str(ggn_path) if ggn_loaded else None
    return low_rank, ggn_source, ggn_checkpoint


@dataclass
class TaskPlan:
    """Задача прогона с уже разрешёнными настройками и источником GGN.

    ``source`` (только out_distr): ``"run"`` -- GGN и prior_prec берутся у
    in_distr-задачи ``peer`` этого прогона; ``"files"`` -- из
    ``ggn_checkpoint`` (``terms``, CPU; ``None`` -> метод Iso) и
    ``prior_prec_source`` (``calib`` -- его ``luno_results.pkl``).
    """
    position: int
    cfg: dict
    name: str
    task_idx: int
    in_distr: bool
    args: argparse.Namespace
    method: str
    source: str | None = None
    peer: str | None = None
    terms: object = None
    ggn_path: str | None = None
    calib: dict | None = None
    calib_path: str | None = None


def _ggn_artifact_target(stem):
    """Target по имени GGN-артефакта (``None`` -- не GGN-артефакт)."""
    if "lifting_low_rank_terms" in stem:
        return "lifting"
    if "kfac_terms" in stem or "low_rank_terms" in stem:
        return "last-block"
    return None


def find_ggn_artifact(path, target, task_idx):
    """Файл GGN для out_distr: сам ``path`` или артефакт ``target`` в папке."""
    raw = rmi.resolve_path(path)
    if raw.is_file():
        return raw
    if not raw.is_dir():
        raise TaskSkipped(f"ggn_checkpoint: {raw} не существует.")
    files = sorted(Path(f) for f in rmi.get_all_files(str(raw), ".pkl"))
    cands = [f for f in files if _ggn_artifact_target(f.stem) == target]
    by_idx = [f for f in cands if f.stem.endswith(f"_{task_idx}")]
    if by_idx:
        return by_idx[0]
    if len(cands) == 1:
        return cands[0]
    raise TaskSkipped(
        f"ggn_checkpoint: в {raw} нет однозначного GGN-артефакта "
        f"(target={target}, task_idx={task_idx}); найдено: "
        f"{[f.name for f in cands]}. Укажите файл явно."
    )


def resolve_out_distr_files(task_args, task_idx, terms_cache):
    """Источник out_distr из явных путей: ``(method, terms, ggn_path, calib, calib_path)``.

    ``prior_prec_source`` обязателен: без калибровки оценивать нечего.
    ``ggn_checkpoint`` задаёт кривизну, и метод (lifting/lastblock,
    kfac/lowrank/full) определяется её файлом; target -- по результатам
    калибровки. Без ``ggn_checkpoint`` это метод Iso, и калибровка тоже должна
    быть сделана для Iso -- иначе prior_prec не соответствует ковариации.
    """
    if not task_args.prior_prec_source:
        raise TaskSkipped(
            "задан ggn_checkpoint, но нет prior_prec_source: без калибровки "
            "(prior_prec) оценивать нечего."
        )
    calib_path, calib = load_calibration_results(
        task_args.prior_prec_source, task_idx
    )
    target = calib.get("ggn_target", "last-block")
    saved_method = calib.get("uq_method")
    saved_rank = calib.get("ggn_rank")
    if not task_args.ggn_checkpoint:
        is_iso = (saved_method.endswith("_iso") if saved_method
                  else saved_rank == 0)
        if not is_iso:
            raise TaskSkipped(
                f"ggn_checkpoint не задан (значит, метод Iso), но калибровка "
                f"{calib_path} сделана с GGN (метод "
                f"{saved_method or f'ранг {saved_rank}'}). Укажите "
                f"ggn_checkpoint той же in_distr-задачи."
            )
        method = "lifting_iso" if target == "lifting" else "lastblock_iso"
        return method, None, None, calib, str(calib_path)

    ggn_path = find_ggn_artifact(task_args.ggn_checkpoint, target, task_idx)
    key = str(ggn_path)
    if key not in terms_cache:
        with open(ggn_path, "rb") as f:
            terms_cache[key] = pickle.load(f)
    terms = terms_cache[key]
    if not isinstance(terms, (LowRankTerms, KFACTerms)):
        raise TaskSkipped(
            f"ggn_checkpoint {ggn_path}: ожидались LowRankTerms или KFACTerms, "
            f"получен {type(terms).__name__}."
        )
    if isinstance(terms, KFACTerms) and target != "last-block":
        raise TaskSkipped(
            f"ggn_checkpoint {ggn_path} -- KFAC (только last-block), а "
            f"калибровка {calib_path} сделана для target={target}."
        )
    method = method_from_terms(target, terms)
    if saved_method and saved_method != method:
        raise TaskSkipped(
            f"ggn_checkpoint {ggn_path} (метод {method}) не соответствует "
            f"калибровке {calib_path} (метод {saved_method}): возьмите оба "
            f"файла из одной in_distr-задачи."
        )
    if (isinstance(terms, LowRankTerms) and saved_rank is not None
            and int(saved_rank) != int(terms.U.shape[1])):
        raise TaskSkipped(
            f"ggn_checkpoint {ggn_path} (ранг {terms.U.shape[1]}) не "
            f"соответствует калибровке {calib_path} (ранг {saved_rank}): "
            f"возьмите оба файла из одной in_distr-задачи."
        )
    return method, terms, str(ggn_path), calib, str(calib_path)


def plan_tasks(args, selected):
    """Проверяет конфиг и разрешает источники GGN/prior_prec ДО запуска задач.

    * ``task_type`` обязателен: ``in_distr`` или ``out_distr``; ключи
      ``luno:`` проверяются по типу (:func:`apply_task_overrides`).
    * in_distr строит GGN (или берёт свой ``ggn_checkpoint``) и калибрует;
      на один ``task_idx`` в прогоне -- не больше одной in_distr-задачи.
    * out_distr: берёт GGN и калибровку у in_distr-задачи того же
      ``task_idx`` этого прогона (приоритетно); если её нет -- из
      ``ggn_checkpoint``/``prior_prec_source`` (в задаче или через CLI);
      иначе -- ошибка.

    Все ошибки собираются и выводятся разом (``SystemExit``): опечатка в
    конфиге не должна всплыть после многочасовой in_distr-задачи.
    """
    errors = []
    plans = []
    for position, cfg in selected:
        name = task_name(cfg, position)
        try:
            task_args = apply_task_overrides(args, cfg)
            in_distr = is_in_distr(cfg)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        task_idx = resolve_task_idx(cfg, position)
        plans.append(TaskPlan(
            position=position, cfg=cfg, name=name, task_idx=task_idx,
            in_distr=in_distr, args=task_args,
            method=uq_method(task_args) if in_distr else "",
        ))

    peers = {}
    for plan in plans:
        if not plan.in_distr:
            continue
        if plan.task_idx in peers:
            errors.append(
                f"task_idx={plan.task_idx}: несколько in_distr-задач в прогоне "
                f"('{peers[plan.task_idx].name}', '{plan.name}'); out_distr-"
                f"задачам непонятно, чью калибровку брать. Оставьте одну."
            )
            continue
        peers[plan.task_idx] = plan

    terms_cache = {}
    for plan in plans:
        if plan.in_distr:
            continue
        a = plan.args
        peer = peers.get(plan.task_idx)
        try:
            if peer is not None:
                plan.source, plan.peer, plan.method = "run", peer.name, peer.method
                a.ggn_target = peer.args.ggn_target
                a.ggn_method = peer.args.ggn_method
                a.max_rank = peer.args.max_rank
                if a.ggn_checkpoint or a.prior_prec_source:
                    print(f"[plan] '{plan.name}': in_distr-задача "
                          f"'{peer.name}' есть в прогоне -- её GGN и "
                          f"калибровка приоритетнее; ggn_checkpoint/"
                          f"prior_prec_source игнорируются.")
            elif a.ggn_checkpoint or a.prior_prec_source:
                (plan.method, plan.terms, plan.ggn_path, plan.calib,
                 plan.calib_path) = resolve_out_distr_files(
                    a, plan.task_idx, terms_cache)
                plan.source = "files"
                a.ggn_target = plan.calib.get("ggn_target", "last-block")
                a.ggn_method = ("kfac" if isinstance(plan.terms, KFACTerms)
                                else "randomized")
                a.max_rank = (int(plan.terms.U.shape[1])
                              if isinstance(plan.terms, LowRankTerms)
                              else 0 if plan.terms is None else a.max_rank)
            else:
                raise TaskSkipped(
                    f"в прогоне нет in_distr-задачи task_idx={plan.task_idx}, "
                    f"а ggn_checkpoint/prior_prec_source не заданы. Добавьте "
                    f"in_distr-задачу в --tasks или пропишите в luno: "
                    f"prior_prec_source (папка in_distr-задачи) и "
                    f"ggn_checkpoint (без него -- метод Iso)."
                )
        except TaskSkipped as exc:
            errors.append(f"задача '{plan.name}' (out_distr): {exc}")
            continue
        if a.ggn_target == "lifting" and a.jv_mode != "forward":
            errors.append(
                f"задача '{plan.name}': источник -- лифтинг, а он "
                f"поддерживает только jv_mode: forward."
            )

    if errors:
        raise SystemExit(
            "Конфиг задач некорректен:\n  - " + "\n  - ".join(errors)
        )
    for plan in plans:
        if plan.in_distr:
            origin = ("GGN из " + plan.args.ggn_checkpoint
                      if plan.args.ggn_checkpoint else "GGN считается")
            origin += ", калибровка"
        elif plan.source == "run":
            origin = f"GGN и калибровка in_distr-задачи '{plan.peer}'"
        else:
            origin = (f"GGN {plan.ggn_path or '-- нет (Iso)'}, "
                      f"калибровка {plan.calib_path}")
        print(f"[plan] {plan.name} ({'in_distr' if plan.in_distr else 'out_distr'},"
              f" task_idx={plan.task_idx}, {plan.method}): {origin}")
    return plans


def _out_distr_prior_args(plan, w0):
    """prior_args из результатов калибровки (out_distr с source == "files")."""
    calib = plan.calib
    prior_args = {
        "prior_prec": torch.as_tensor(
            float(calib["prior_prec"]), dtype=DTYPE, device=w0.device
        ),
        "objective": calib.get("calib_objective"),
        "samples": 0,
        "diag_probes": plan.args.diag_hutchinson,
    }
    if calib.get("temperature") is not None:
        prior_args["temperature"] = float(calib["temperature"])
    print(f"[prior_prec] из {plan.calib_path}: "
          f"prior_prec={float(calib['prior_prec']):.6e}"
          + (f", temperature={prior_args['temperature']:.6e}"
             if "temperature" in prior_args else ""))
    return prior_args


def run_one_task(plan, model_config, core, wrapper, w0,
                 output_dir, run_prefix, config_name, metrics_config, asset_cache,
                 shared=None):
    """Полный LUNO-пайплайн одной задачи; результаты пишутся в output_dir.

    in_distr строит GGN и калибрует prior_prec сам, кладя результат в
    ``shared[task_idx]``. out_distr берёт их из ``shared`` (``source="run"``)
    или из файлов, найденных :func:`plan_tasks` (``source="files"``).
    """
    args, cfg, task_idx = plan.args, plan.cfg, plan.task_idx
    in_distr = plan.in_distr
    name = plan.name
    if shared is None:
        shared = {}

    pipeline = build_task_pipeline(args, cfg, task_idx, model_config, core,
                                   asset_cache, in_distr=in_distr)
    confirm_exact_diag(args, pipeline)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    lifting = pipeline["lifting"].to(device=device).to(dtype=DTYPE)
    projection = pipeline["projection"].to(device=device).to(dtype=DTYPE)

    # --- нормализаторы и data_processor ---
    data_processor, device = build_data_processor(args, pipeline, task_idx)

    # model_fn/w0/якобиан зависят от --ggn-target. Для last-block это прежний
    # путь (w -- веса последнего FNO-блока), для lifting -- все параметры
    # адаптера лифтинга, а core и проекция заморожены.
    setup = build_ggn_setup(args, core, wrapper, w0, lifting, projection)
    model_fn = setup.model_fn
    w0 = setup.w0
    wrapper = setup.wrapper
    affine = setup.affine

    # Стартовый гейт (--jacobian-check): Jv выбранного режима против
    # reverse-строк на ОДНОМ сэмпле. Эталон -- обратный режим; при расхождении
    # падаем явно, а не молча считаем кривую кривизну. Для нового ядра это
    # главная проверка, что forward-mode AD через него вообще работает. Допуск
    # зависит от точности (см. default_check_tol).
    if getattr(args, "jacobian_check", False):
        probe = preprocess_sample(
            data_processor, next(iter_samples(pipeline["train_loader"], 1))
        )
        ok, rel = check_jacobian(
            model_fn, probe[0]["x"], w0,
            affine=affine, mode=args.jv_mode, fwd_batch=args.fwd_batch,
            num_probes=args.check_probes, tol=args.check_tol,
            jacobian_cls=setup.jacobian_cls,
        )
        tol_used = (args.check_tol if args.check_tol is not None
                    else default_check_tol(w0.dtype))
        print(f"[jacobian-check] target={setup.target}, mode={args.jv_mode}: "
              f"{'OK' if ok else 'FAILED'} (max rel err={rel:.3e}, "
              f"tol={tol_used:.1e}, dtype={w0.dtype}, torch={torch.__version__})")
        if not ok:
            raise RuntimeError(
                f"Jv (target={setup.target}, mode={args.jv_mode}) не совпал с "
                f"reverse-строками (max rel err={rel:.3e} > tol={tol_used:.1e}). "
                "Для нового ядра: проверьте, что все его операции поддерживают "
                "forward-mode AD (attention-бэкенды, кастомные CUDA-ядра, "
                "checkpointing)."
            )
        _print_mem("after_jacobian_check", device)

    # --- GGN + prior_prec: in_distr считает сам; out_distr берёт у in_distr- ---
    # --- задачи прогона или из файлов (источник выбран в plan_tasks).       ---
    prior_prec_source = "calibrated"
    ggn_reused = False
    shared_from = None
    ggn_label = ("лифтинга" if setup.target == "lifting"
                 else "последнего фурье-слоя")
    if in_distr:
        print(f"\n[GGN] метод {uq_method(args)}: GGN {ggn_label} "
              f"(target={setup.target}):")
        low_rank, ggn_source, ggn_checkpoint = resolve_low_rank(
            args, w0, model_fn, pipeline, data_processor, affine, task_idx,
            output_dir, run_prefix, asset_cache,
            matvec_cls=setup.matvec_cls, artifact_prefix=setup.artifact_prefix,
            bind=setup.bind, kfac=setup.kfac,
            iso=setup.kfac is None and args.max_rank == 0,
            full_sampled=setup.full_sampled, jacobian_cls=setup.jacobian_cls,
        )

        # --- калибровка на сэмплах калибровочной части val (доля --val-calib-frac) ---
        # Память здесь -- единственное, что отличает дешёвую калибровку от OOM:
        # один прогон прямого режима стоит ~fwd_batch * активации сети, и если
        # GGN оставил после себя буферы, места на него может не хватить.
        _print_mem("before_calibrate", device)
        prior_args = calibrate_prior_prec(
            args, model_fn, w0, low_rank, wrapper, data_processor,
            data_processor.out_normalizer, pipeline["calib_loader"], affine,
            objective_name=args.calib_objective, mode=args.jv_mode,
            fwd_batch=args.fwd_batch, jacobian_cls=setup.jacobian_cls,
            bind=setup.bind, kfac=setup.kfac,
            temperature=setup.full_sampled and args.max_rank != 0,
        )
        shared[task_idx] = (low_rank, prior_args)
    elif plan.source == "run":
        # GGN и prior_prec in_distr-задачи ЭТОГО прогона (она шла раньше).
        if task_idx not in shared:
            raise RuntimeError(
                f"in_distr-задача '{plan.peer}' (task_idx={task_idx}) не дала "
                f"GGN и prior_prec (упала?) -- оценивать '{name}' не с чем."
            )
        low_rank, prior_args = shared[task_idx]
        shared_from = task_idx
        prior_prec_source = "shared"
        ggn_reused = True
        ggn_source, ggn_checkpoint = "shared-in_distr", None
        print(f"[shared] метод {plan.method}: GGN и prior_prec взяты у "
              f"in_distr-задачи '{plan.peer}' этого прогона "
              f"(prior_prec={prior_args['prior_prec'].item():.6e})")
    else:
        # Явные файлы: ggn_checkpoint (нет -- Iso) и prior_prec_source.
        print(f"\n[GGN] метод {plan.method}: из файлов (in_distr-задача "
              f"не в этом прогоне или пути заданы явно).")
        if plan.terms is None:
            low_rank, ggn_source, ggn_checkpoint = iso_low_rank(w0), "iso", None
            print("[GGN] ggn_checkpoint не задан -- метод Iso: ковариация "
                  "I / prior_prec.")
        else:
            cache_key = ("ggn", plan.ggn_path)
            if cache_key not in asset_cache:
                asset_cache[cache_key] = (
                    plan.terms.to(device=w0.device, dtype=w0.dtype)
                    if isinstance(plan.terms, KFACTerms)
                    else low_rank_to(plan.terms, w0, plan.ggn_path)
                )
            low_rank = asset_cache[cache_key]
            ggn_source, ggn_checkpoint = "checkpoint", plan.ggn_path
        prior_args = _out_distr_prior_args(plan, w0)
        prior_prec_source = "results.pkl"
        ggn_reused = True

    out_normalizer = data_processor.out_normalizer

    # --- метрики после калибровки: val и test делят одно значение --diag-hutchinson ---
    # val (остаток val): Хатчинсон + PNG; test: тот же Хатчинсон, без PNG.
    # Имена каналов для per-channel метрик: adapter.variable_names этой задачи
    # (порядок совпадает с осью канала). Конфиг может разойтись с моделью --
    # внутри evaluate_luno это проверяется по длине и уходит на ch0..
    channel_names = (cfg.get("adapter") or {}).get("variable_names")

    print("\n[metrics] финальные метрики после калибровки:")
    val_metrics = evaluate_luno(
        args, pipeline["val_eval_loader"], model_fn, w0, low_rank, prior_args,
        wrapper, data_processor, out_normalizer, "val", len(pipeline["val_eval_loader"].dataset),
        affine=affine,
        metrics_config=metrics_config,
        output_dir=output_dir,
        mode=args.jv_mode, fwd_batch=args.fwd_batch,
        save_images=True, diag_probes=args.diag_hutchinson,
        channel_names=channel_names, jacobian_cls=setup.jacobian_cls,
        bind=setup.bind, kfac=setup.kfac,
    )
    test_metrics = evaluate_luno(
        args, pipeline["test_loader"], model_fn, w0, low_rank, prior_args, wrapper,
        data_processor, out_normalizer, "test", len(pipeline["test_loader"].dataset),
        affine=affine,
        metrics_config=metrics_config,
        output_dir=output_dir,
        mode=args.jv_mode, fwd_batch=args.fwd_batch,
        save_images=False, diag_probes=args.diag_hutchinson,
        channel_names=channel_names, jacobian_cls=setup.jacobian_cls,
        bind=setup.bind, kfac=setup.kfac,
    )

    results = {
        "config": config_name,
        "task_name": cfg.get("name"),
        "task_idx": task_idx,
        "task_type": cfg.get("task_type"),
        "in_distr": in_distr,
        "ggn_target": setup.target,
        # Метод UQ: у in_distr -- по своим настройкам, у остальных -- метод
        # in_distr-задачи, от которой унаследованы GGN и prior_prec.
        "uq_method": method_from_terms(setup.target, low_rank),
        "w0_dim": int(w0.numel()),
        "prior_prec": prior_args["prior_prec"].item(),
        # Температура (только полная GGN лифтинга): Sigma = T (prec I + G)^{-1}.
        "temperature": (float(prior_args["temperature"])
                        if prior_args.get("temperature") is not None else None),
        "prior_prec_source": prior_prec_source,
        # Откуда out_distr взяла GGN и prior_prec: имя in_distr-задачи прогона
        # или путь к её luno_results.pkl; у in_distr -- None.
        "calib_from": plan.peer if plan.source == "run" else plan.calib_path,
        "ggn_reused": ggn_reused,
        "shared_from_task_idx": shared_from,
        "calib_objective": prior_args.get("objective", args.calib_objective),
        "calib_samples": prior_args["samples"],
        "calib_diag_probes": prior_args["diag_probes"],
        "calib_max_samples": args.calib_max_samples if in_distr else None,
        "val": val_metrics,
        "test": test_metrics,
        "max_rank": args.max_rank,
        "ggn_method": args.ggn_method,
        "ggn_oversample": args.ggn_oversample,
        "ggn_sample_batch": getattr(args, "ggn_sample_batch", 1),
        "max_num_samples_ggn": args.max_num_samples,
        # Откуда взят low-rank: "checkpoint" -- загружен из --ggn-checkpoint
        # (артефакт не переписывался), "computed" -- посчитан на train loader.
        # ggn_rank -- фактический ранг (из чекпойнта, если он авторитетен),
        # max_rank остаётся значением CLI, чтобы не ломать сравнение с ранами.
        "ggn_source": ggn_source,
        "ggn_checkpoint": ggn_checkpoint,
        "ggn_rank": (int(low_rank.U.shape[1])
                     if isinstance(low_rank, LowRankTerms) else None),
        "diag_hutchinson": args.diag_hutchinson,
        "max_eval_samples": len(pipeline["val_eval_loader"].dataset),
        "max_test_samples": len(pipeline["test_loader"].dataset),
        "val_calib_frac": args.val_calib_frac if in_distr else None,
    }
    with open(output_dir / "metrics" / "val_metrics.pkl", "wb") as f:
        pickle.dump({"val": val_metrics}, f)
    with open(output_dir / "metrics" / "test_metrics.pkl", "wb") as f:
        pickle.dump({"test": test_metrics}, f)
    with open(output_dir / "luno_results.pkl", "wb") as f:
        pickle.dump(results, f)
    return results


def main():
    args = parse_args()

    set_enabled(not args.no_progress)

    if args.dtype == "float64":
        global DTYPE, CDTYPE
        DTYPE = torch.float64
        CDTYPE = torch.complex128

    if args.core_checkpoint is None:
        raise SystemExit("--core-checkpoint обязателен.")

    if args.ggn_target == "lifting" and args.jv_mode != "forward":
        raise SystemExit(
            "--ggn-target lifting поддерживает только --jv-mode forward: "
            "аффинная декомпозиция для лифтинга не реализована."
        )
    print(f"[ggn] target={args.ggn_target}, jv_mode={args.jv_mode}")

    config_path = rmi.resolve_path(args.config)
    config = rmi.load_yaml_config(config_path)
    model_config = config.get("model", {})
    task_configs = config.get("tasks", [])
    selected = select_tasks(task_configs, args.tasks)
    print(f"[config] {config_path}")
    print(f"[tasks] выбрано {len(selected)} из {len(task_configs)}: "
          f"{[task_name(cfg, i) for i, cfg in selected]}")

    # Проверка конфига и выбор источников GGN/prior_prec для out_distr --
    # до загрузки core: ошибка конфига видна сразу, а не через часы.
    plans = plan_tasks(args, selected)

    # --- core общий для всех задач: загрузка и перенос на device ОДИН раз ---
    _install_pickle_compat()
    print("Загрузка core (torch.load + dill):")
    # Сначала на CPU (чекпойнт может быть сохранён с CUDA-тензорами), на
    # нужное устройство ядро переносится сразу ниже.
    core = rmi.load_from_dir(args.core_checkpoint, {"map_location": "cpu"})[0]

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    core = core.to(device=device)

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

    # --- корневая output-папка прогона (подпапки -- на каждую задачу) ---
    output_config = config.get("output", {})
    output_root = (
        args.output_root
        if args.output_root is not None
        else output_config.get("root", "runs")
    )
    run_prefix = args.run_name if args.run_name is not None else config_path.stem
    # Метка метода UQ в имени папки: {prefix}_{время}_{target}_{iso|lowrank|kfac}.
    method_label = run_method_label(plans)
    run_name = (f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_{method_label}")
    print(f"[method] {method_label}")
    output_dir = rmi.resolve_path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir: {output_dir}")

    # --- сплит ядра: нужен только для --ggn-target last-block ---
    # Для лифтинга ядро -- чёрный ящик (TorchLiftingWrapper вызывает core(z)),
    # поэтому никаких предположений о FNO здесь нет: подходит любое ядро с
    # поддержкой forward-mode AD.
    n_core = sum(p.numel() for p in core.parameters())
    print(f"[core] {type(core).__module__}.{type(core).__qualname__}, "
          f"параметров={n_core:,}")
    # FNO-сплит нужен, если хотя бы одна выбранная задача (in_distr -- по
    # luno.ggn_target, out_distr -- по источнику) работает с последним блоком.
    wrapper, w0 = None, None
    targets = {plan.args.ggn_target for plan in plans}
    print(f"[ggn] target по задачам: {sorted(targets)}")
    if "last-block" in targets:
        wrapper = TorchFNOWrapper(core, dtype=DTYPE)
        _, w0 = split_wrapper(wrapper)
        print(f"[split] w0: d={w0.numel()}, dtype={w0.dtype}, device={w0.device}")

        keys = discover_last_block_keys(core)
        print("\n[последний фурье-слой core]:")
        for label in ("R", "W", "b"):
            path, cls, attr = _resolve_key_owner(core, keys[label])
            print(f"    {label}: layer='{path}' ({cls}, attr='{attr}')")
        R, W, b = ref_state_dict(core, keys)
        print(f"    R: {tuple(R.shape)} (complex -> R.real | R.imag)")
        print(f"    W: {tuple(W.shape)}")
        print(f"    b: {tuple(b.shape) if b is not None else None}")
        assert torch.is_complex(R), (
            "Спектральный вес R стал некомплексным (мнимая часть потеряна из-за "
            "core.to(dtype=...)). Не кастать core целиком в float64; см. комментарий"
            " к переносу core на device в main."
        )

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

    asset_cache = {}
    # Общий кэш значений in_distr-задач: {task_idx: (low_rank, prior_args)}.
    # in_distr-задачи обрабатываются первыми (см. сортировку ниже), поэтому к
    # моменту запуска out_distr-задачи того же task_idx значения уже готовы.
    shared = {}
    # Сортировка стабильная -- прочий порядок конфига сохраняется.
    plans = sorted(plans, key=lambda plan: 0 if plan.in_distr else 1)
    computed = []
    failed = []
    for plan in plans:
        name, task_idx, in_distr = plan.name, plan.task_idx, plan.in_distr
        kind = "in_distr" if in_distr else "out_distr"

        task_dir = output_dir / name
        for d in (task_dir / "checkpoints", task_dir / "logs", task_dir / "metrics"):
            d.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}\n[task:{name}] старт ({kind}, "
              f"task_idx={task_idx}, метод {plan.method}, "
              f"ggn_target={plan.args.ggn_target})\n{'=' * 70}")
        try:
            run_one_task(
                plan, model_config, core, wrapper, w0,
                task_dir, run_prefix, config_path.stem,
                config.get("metrics", {}), asset_cache, shared,
            )
            computed.append(name)
            print(f"[task:{name}] ГОТОВО -> {task_dir}")
        except Exception as exc:
            failed.append(name)
            print(f"[task:{name}] ОШИБКА, задача пропущена: "
                  f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            with open(task_dir / "luno_results.pkl", "wb") as f:
                pickle.dump(
                    {
                        "config": config_path.stem,
                        "task_name": name,
                        "task_idx": task_idx,
                        "task_type": kind,
                        "in_distr": in_distr,
                        "ggn_target": plan.args.ggn_target,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                    f,
                )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nГотово. Посчитаны задачи ({len(computed)}): {computed}. "
          f"Упали ({len(failed)}): {failed}. "
          f"Результаты: {output_dir}")


if __name__ == "__main__":
    main()
