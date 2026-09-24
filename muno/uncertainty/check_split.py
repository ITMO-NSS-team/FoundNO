"""
Проверка первого шага LUNO-пайплайна на реальной модели Muno.

Переиспользует функции загрузки модели из run_multiphysics_inference.py
(load_yaml_config, resolve_path, load_from_dir, build_model, Muno),
но НЕ загружает данные (PDEBench датасеты / /workspace): loader_channels
выводятся напрямую из загруженных модулей (liftings/projections).

Скрипт печатает:
  1) инфо об исходной модели Muno (части: liftings / core / projections);
  2) первый шаг пайплайна — сплит модели на части:
       - head  — фиксированные параметры core (всё, кроме последнего FNO-блока);
       - веса последнего FNO-блока (R, W, b) -> плоский w0;
       - projection — фиксированные модули Muno (projections).
     с проверкой, что split_wrapper(w0) корректно реконструирует веса core.

Запуск (флаги те же, что в run_multiphysics_inference.py):
    python muno/uncertainty/check_split.py \
        --config experiments/scripts/configs/pdebench_multiphysics.yaml \
        --core-checkpoint <path>/core.pt \
        --lift-checkpoint-dir <dir> \
        --proj-checkpoint-dir <dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_SCRIPTS = PROJECT_ROOT / "experiments" / "scripts"
UNCERTAINTY_DIR = Path(__file__).resolve().parent

for _p in (str(PROJECT_ROOT), str(EXPERIMENTS_SCRIPTS), str(UNCERTAINTY_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from muno.utils.model_factory import _infer_in_channels, _infer_out_channels

from luno_torch.adapter import (
    TorchFNOWrapper,
    discover_last_block_keys,
    ref_state_dict,
    split_wrapper,
)

import run_multiphysics_inference as rmi


def _channels(mod):
    """(in_channels, out_channels) модуля: из conv/linear-слоёв, иначе из атрибутов."""
    in_c = _infer_in_channels(mod)
    if in_c is None:
        in_c = getattr(mod, "in_channels", None)
    out_c = _infer_out_channels(mod)
    if out_c is None:
        out_c = getattr(mod, "out_channels", None)
    return in_c, out_c


def _resolve_key_owner(root, key):
    """Для state_dict-ключа возвращает (module_path, class_name, attr_path).

    Находит самый длинный префикс ключа, совпадающий с путём подмодуля
    (учитывает вложенные атрибуты вида ``weight.tensor``).
    """
    if key is None:
        return None, None, None
    best_path, best_mod = "", None
    for path, mod in root.named_modules():
        prefix = path + "." if path else ""
        if key.startswith(prefix) and len(prefix) >= len(best_path):
            best_path, best_mod = prefix, mod
    if best_mod is None:
        return None, None, None
    attr_path = key[len(best_path):]
    mod_name = type(best_mod).__qualname__
    return best_path.rstrip("."), mod_name, attr_path


def infer_loader_channels(liftings, projections):
    loader_channels = []
    for lift, proj in zip(liftings, projections):
        in_c, _ = _channels(lift)
        _, out_c = _channels(proj)
        loader_channels.append([in_c, out_c])
    return loader_channels


def print_param_table(label, mod, indent="    "):
    total = 0
    print(f"{indent}-- {label} :: {type(mod).__module__}.{type(mod).__qualname__}")
    for name, p in sorted(mod.named_parameters()):
        total += p.numel()
        print(
            f"{indent}   {name:<50} "
            f"shape={tuple(p.shape)} dtype={p.dtype} grad={p.requires_grad} "
            f"numel={p.numel()}"
        )
    if total == 0:
        print(f"{indent}   (параметров нет)")
    print(f"{indent}   total params: {total}")
    return total


def print_original_model(model):
    print("\n================ ИСХОДНАЯ МОДЕЛЬ (Muno) ================")
    grand = 0
    for i, lift in enumerate(model._liftings):
        grand += print_param_table(f"lifting[{i}]", lift)
        in_c, out_c = _channels(lift)
        print(f"          in_channels={in_c}, out_channels={out_c}")
    grand += print_param_table("core", model._core)
    in_c, out_c = _channels(model._core)
    print(f"          in_channels={in_c}, out_channels={out_c}")
    for i, proj in enumerate(model._projections):
        grand += print_param_table(f"projection[{i}]", proj)
        in_c, out_c = _channels(proj)
        print(f"          in_channels={in_c}, out_channels={out_c}")
    print(f"---- TOTAL params: {grand}")


def print_split_check(core, projections):
    print("\n======================== СПЛИТ (первый шаг пайплайна) ========================")
    print("Цель: linearized-параметры -- веса последнего FNO-блока ядра (R, W, b).")

    keys = discover_last_block_keys(core)
    print("\n[discover_last_block_keys] nl={}  R={}  W={}  b={}".format(
        keys['nl'], keys['R'], keys['W'], keys['b']))

    R, W, b = ref_state_dict(core, keys)
    wrapper = TorchFNOWrapper(core)
    n_r, n_w, n_b = wrapper.weight_splits()
    d = 2 * n_r + n_w + (n_b or 0)

    owners = {label: _resolve_key_owner(core, keys[label]) for label in ("R", "W", "b")}

    print("\n[head] -- фиксированные параметры core (всё, кроме последнего FNO-блока):")
    sd = {k: v for k, v in core.state_dict().items() if isinstance(v, torch.Tensor)}
    block_keys = {keys["R"], keys["W"]}
    if b is not None:
        block_keys.add(keys["b"])
    head_keys = sorted(k for k in sd if k not in block_keys)
    head_numel = 0
    for k in head_keys:
        head_numel += sd[k].numel()
        print(f"    {k:<55} shape={tuple(sd[k].shape)} dtype={sd[k].dtype} numel={sd[k].numel()}")
    print(f"    head total: {len(head_keys)} параметров, {head_numel} элементов")

    print("\n[last FNO block] -- linearized-параметры (R, W, b):")
    for label in ("R", "W", "b"):
        if owners[label][0] is None:
            print(f"    {label}: не найден слой для ключа '{keys[label]}'")
            continue
        path, cls, attr = owners[label]
        print(f"    {label}: layer='{path}' ({cls}, attr='{attr}')")
    print(f"    R: shape={tuple(R.shape)} dtype={R.dtype} numel={n_r}   (complex -> R.real | R.imag)")
    print(f"    W: shape={tuple(W.shape)} dtype={W.dtype} numel={n_w}")
    if b is not None:
        print(f"    b: shape={tuple(b.shape)} dtype={b.dtype} numel={n_b}")
    else:
        print("    b: None")

    print(f"\n[w0] -- плоский вектор параметров, d={d}:")
    off_r2 = 2 * n_r
    off_w = off_r2 + n_w
    print(f"    R.real [0:{n_r}) | R.imag [{n_r}:{off_r2}) | W [{off_r2}:{off_w}) |"
          f" b [{off_w}:{d})")

    print("\n[projection] -- фиксированные модули Muno (проекции):")
    proj_numel = 0
    for i, proj in enumerate(projections):
        n = sum(p.numel() for p in proj.parameters())
        proj_numel += n
        print(
            f"    projection[{i}] :: {type(proj).__module__}.{type(proj).__qualname__} "
            f"params={n}  in/out={_channels(proj)}"
        )

    print("\n[проверка] реконструкция весов core из w0:")
    base = wrapper._base_state_dict()
    model_fn, w0 = split_wrapper(wrapper)
    recon = wrapper.reconstruct(w0)
    max_diff = 0.0
    for k in base:
        if k not in recon:
            print(f"    ключ {k} пропущен в reconstruct!")
            max_diff = float("inf")
            continue
        diff = (recon[k] - base[k]).abs().max().item()
        max_diff = max(max_diff, diff)
    status = "SPLIT OK" if max_diff == 0 else f"SPLIT NOT OK (max diff={max_diff:.3e})"
    print(f"    max |reconstruct(w0) - base| = {max_diff:.3e}  ->  {status}")
    print(f"    model_fn: {model_fn}")


def main():
    args = rmi.parse_args()

    if args.core_checkpoint is None:
        raise SystemExit("--core-checkpoint обязателен для проверки сплита модели.")

    config_path = rmi.resolve_path(args.config)
    config = rmi.load_yaml_config(config_path)
    model_config = config.get("model", {})

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

    loader_channels = infer_loader_channels(liftings, projections)
    print("\nloader_channels (inferred из модулей, без загрузки данных):", loader_channels)

    model_blocks = rmi.build_model(
        loader_channels, model_config,
        core, liftings, projections,
    )
    if not isinstance(model_blocks, tuple):
        raise RuntimeError(
            "Ожидалась lifting-core-projection архитектура, "
            f"вместо этого получена модель {type(model_blocks)}."
        )

    model = rmi.Muno(
        liftings=model_blocks[0],
        core=model_blocks[1],
        projections=model_blocks[2],
    )

    print_original_model(model)
    print_split_check(model._core, model._projections)

    print("\nГотово. Дальнейшие шаги пайплайна (Jacobian -> GGN -> GP) не выполнялись.")


if __name__ == "__main__":
    main()