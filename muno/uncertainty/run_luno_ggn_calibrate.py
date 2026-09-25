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
    3) low-rank GGN последнего фурье-слоя на train-лоадере;
    4) калибровка scalar prior_prec на первом val-батче (log10 grid, patience);
    5) финальные метрики NLL (+RMSE) на val/test (limited max-eval-batches);
    6) сохранение GGN (LowRankTerms) и метрик в pkl в output-папке эксперимента
       (структура папок как в run_multiphysics_inference.py).

Запуск (флаги -- как в run_multiphysics_inference.py):
    python muno/uncertainty/run_luno_ggn_calibrate.py \
        --config experiments/scripts/configs/pdebench_multiphysics.yaml \
        --core-checkpoint <path>/core.pt \
        --lift-checkpoint-dir <dir> \
        --proj-checkpoint-dir <dir> \
        --in-normalizers <dir> --out-normalizers <dir> \
        --output-root <root> --run-name <name>

При OOM из-за фрагментации CUDA-памяти можно включить сегменты расширяемого
размера (переменная окружения должна быть выставлена ДО импорта torch):
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    #   python muno/uncertainty/run_luno_ggn_calibrate.py ...
"""

from __future__ import annotations

import argparse
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

import torch
from torch.func import functional_call
from torch.utils.data import DataLoader

from muno.data.benchmarks.datasets import MultiPhysicsDataset
from muno.data.data.transforms.data_processors import DefaultDataProcessor
from muno.data.data.transforms.normalizers import MultiphysicsUnitGaussianNormalizer

from luno_torch.adapter import (
    TorchFNOWrapper,
    discover_last_block_keys,
    ref_state_dict,
    split_wrapper,
)
from luno_torch.calibrate import grid_search, nll_gaussian
from luno_torch.factory import create_luno_cov
from luno_torch.ggn import GGNMatvec, skerch_low_rank
from luno_torch.jacobian import LastFNOBlockWeightJacobian, var_of_congruence

import run_multiphysics_inference as rmi

from check_split import _resolve_key_owner

DTYPE = torch.float32
CDTYPE = torch.complex64


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
    p.add_argument("--vjp-chunk", type=int, default=16,
                   help="Сколько выходных байтов J материализовать за раз; меньший "
                        "чанк = меньше пиковая память в GGN-математике. "
                        "Пик на чанк ~ vjp_chunk * d * itemsize (при d=27.6M, float32, "
                        "chunk=16 это ~1.8 GiB).")
    p.add_argument("--max-rank", type=int, default=10, help="Low-rank rank for GGN.")
    p.add_argument("--max-num-samples", type=int, default=25,
                   help="Макс. число сэмплов для GGN (как max_num_of_samples в luno).")
    p.add_argument("--max-eval-samples", type=int, default=2,
                   help="Макс. число сэмплов для финальных val/test метрик.")
    p.add_argument("--calib-grid-min", type=float, default=-3.0)
    p.add_argument("--calib-grid-max", type=float, default=3.0)
    p.add_argument("--calib-grid-size", type=int, default=50)
    p.add_argument("--calib-patience", type=int, default=5)
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
    val_loader = DataLoader(dataset=MultiPhysicsDataset(list(val_set)), batch_size=1, shuffle=False)
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
        "val_loader": val_loader,
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

def compute_low_rank_ggn(args, model_fn, w0, train_loader, data_processor):
    xs = []
    for sample in iter_samples(train_loader, args.max_num_samples):
        sample = preprocess_sample(data_processor, sample)
        xs.append(sample[0]["x"])
    print(f"[GGN] собрано {len(xs)} сэмплов из train loader.")
    assert xs, "No samples collected for GGN."

    ggn_mv = GGNMatvec(
        model_fn, w0, xs, loss_fn="mse", factor=1.0,
        vjp_chunk=args.vjp_chunk,
    )
    _print_mem("ggn_mv", w0.device)

    low_rank = skerch_low_rank(
        ggn_mv,
        rank=args.max_rank,
        device=str(w0.device),
        dtype=w0.dtype,
    )
    print(f"[GGN] low-rank terms: U={tuple(low_rank.U.shape)} S={tuple(low_rank.S.shape)}")
    return low_rank


def _batch_predictive_stats(model_fn, w0, weight_cov, x, out_normalizer,
                            num_output_channels, vjp_chunk=64):
    """(mean_raw, std_raw) для одного входного тензора x в raw-пространстве."""
    with torch.no_grad():
        out_shape = model_fn(x, w0).shape
    jac = LastFNOBlockWeightJacobian(
        model_fn,
        x,
        w0,
        num_output_channels=num_output_channels,
        vjp_chunk=vjp_chunk,
    )
    mean_norm = jac.fx.reshape(-1)
    var_norm = var_of_congruence(jac, weight_cov)
    std_norm = torch.sqrt(var_norm)
    return raw_from_normalized(mean_norm, std_norm, out_normalizer, out_shape)


def calibrate_prior_prec(args, model_fn, w0, low_rank, wrapper, data_processor,
                         out_normalizer, val_loader):
    """Калибровка scalar prior_prec на первом val-батче (как luno_experiments).

    Jacobian на входе кэшируется один раз; по гриду меняется только cov.
    """
    calib_sample = next(iter(val_loader))
    calib_sample = preprocess_sample(data_processor, calib_sample)
    x = calib_sample[0]["x"]
    target = calib_sample[0]["y"].reshape(-1)

    with torch.no_grad():
        out_shape = model_fn(x, w0).shape
    num_output_channels = wrapper.num_output_channels
    jac = LastFNOBlockWeightJacobian(
        model_fn, x, w0, num_output_channels=num_output_channels,
        vjp_chunk=args.vjp_chunk,
    )
    mean_norm = jac.fx.reshape(-1)

    grid = torch.logspace(
        args.calib_grid_min, args.calib_grid_max, args.calib_grid_size, base=10.0
    )

    def objective(prec):
        cov = create_luno_cov(low_rank, {"prior_prec": prec})
        var_norm = var_of_congruence(jac, cov)
        std_norm = torch.sqrt(var_norm)
        mean_raw, std_raw = raw_from_normalized(
            mean_norm, std_norm, out_normalizer, out_shape
        )
        return nll_gaussian(mean_raw, std_raw, target, scaled=True)

    best_value, best_idx = grid_search(grid, objective, patience=args.calib_patience)
    best_prec = grid[best_idx]
    print(f"[calibrate] best prior_prec={best_prec.item():.6e}, nll={best_value:.6e}")
    return {"prior_prec": best_prec}


def evaluate_luno(args, loader, model_fn, w0, low_rank, prior_args, wrapper,
                  data_processor, out_normalizer, task_name, max_samples):
    cov = create_luno_cov(low_rank, prior_args)
    nll_sum = torch.tensor(0.0, dtype=DTYPE)
    rmse_sum = torch.tensor(0.0, dtype=DTYPE)
    n_elements = 0
    n_samples = 0

    for sample in iter_samples(loader, max_samples):
        sample = preprocess_sample(data_processor, sample)
        x = sample[0]["x"]
        target = sample[0]["y"].reshape(-1)
        mean_raw, std_raw = _batch_predictive_stats(
            model_fn, w0, cov, x, out_normalizer, wrapper.num_output_channels,
            vjp_chunk=args.vjp_chunk,
        )
        nll_sum = nll_sum + nll_gaussian(mean_raw, std_raw, target, scaled=False)
        rmse_sum = rmse_sum + torch.sqrt(torch.mean((mean_raw - target) ** 2))
        n_elements += target.numel()
        n_samples += 1

    nll = (nll_sum / n_elements).item() if n_elements else float("nan")
    rmse = (rmse_sum / n_samples).item() if n_samples else float("nan")
    print(f"[eval:{task_name}] nll={nll:.6e} rmse={rmse:.6e} "
          f"(samples={n_samples}, elements={n_elements})")
    return {"nll": nll, "rmse": rmse, "samples": n_samples}


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.dtype == "float64":
        global DTYPE, CDTYPE
        DTYPE = torch.float64
        CDTYPE = torch.complex128

    if args.core_checkpoint is None:
        raise SystemExit("--core-checkpoint обязателен.")

    pipeline = build_pipeline(args)
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

    # --- low-rank GGN на train ---
    print("\n[GGN] low-rank аппроксимация GGN последнего фурье-слоя (train loader):")
    low_rank = compute_low_rank_ggn(
        args, model_fn, w0, pipeline["train_loader"], data_processor
    )

    with open(output_dir / f"{run_prefix}_low_rank_terms.pkl", "wb") as f:
        pickle.dump(low_rank, f)
    print(f"[save] low_rank_terms -> {output_dir / f'{run_prefix}_low_rank_terms.pkl'}")

    # --- калибровка на первом val-батче ---
    print("\n[calibrate] оптимизация prior_prec на первом val-батче (val 0.1):")
    prior_args = calibrate_prior_prec(
        args, model_fn, w0, low_rank, wrapper, data_processor,
        data_processor.out_normalizer, pipeline["val_loader"],
    )

    out_normalizer = data_processor.out_normalizer

    # --- финальные метрики после калибровки ---
    print("\n[metrics] финальные метрики после калибровки:")
    val_metrics = evaluate_luno(
        args, pipeline["val_loader"], model_fn, w0, low_rank, prior_args, wrapper,
        data_processor, out_normalizer, "val", args.max_eval_samples,
    )
    test_metrics = evaluate_luno(
        args, pipeline["test_loader"], model_fn, w0, low_rank, prior_args, wrapper,
        data_processor, out_normalizer, "test", args.max_eval_samples,
    )

    results = {
        "prior_prec": prior_args["prior_prec"].item(),
        "val": val_metrics,
        "test": test_metrics,
        "max_rank": args.max_rank,
        "max_num_samples_ggn": args.max_num_samples,
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