import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from muno.utils.seed import set_global_seed
from muno.data.benchmarks.config_io import load_yaml_config
from muno.data.benchmarks.multiphysics_loaders import get_loaders_channels
from muno.data.benchmarks.datasets import MultiPhysicsDataset
from muno.utils.model_factory import build_model, load_from_dir, get_all_files
from muno.models.muno import Muno
from muno.models.mamba_fno import LiftingFeatureDropout, LiftingGaussianPerturbation
from muno.data.data.transforms.normalizers import MultiphysicsUnitGaussianNormalizer
from muno.data.data.transforms.data_processors import DefaultDataProcessor

from run_multiphysics_inference import (
    resolve_path,
    parse_args,
    loadData,
    evaluate_loader,
)


def reset_positional_embedding_caches(model):
    parts = []
    if isinstance(model, Muno):
        if model._core is not None:
            parts.append(model._core)
        parts.extend(model._liftings or [])
        parts.extend(model._projections or [])
    else:
        parts.append(model)

    for part in parts:
        for module in part.modules():
            if hasattr(module, "_grid") and hasattr(module, "_res"):
                module._grid = None
                module._res = None


def setup_experiment(config, args):
    device = f'cuda:{args.device}'

    task_configs = config["tasks"]
    train_set, val_set, test_set, _ = loadData(task_configs)
    train_set = MultiPhysicsDataset(train_set)
    val_set   = MultiPhysicsDataset(val_set)
    test_set  = MultiPhysicsDataset(test_set)

    train_loader = DataLoader(dataset=train_set, shuffle=False)
    val_loader   = DataLoader(dataset=val_set,   shuffle=False)
    test_loader  = DataLoader(dataset=test_set,  shuffle=False)

    loader_channels = get_loaders_channels(train_loader)
    print("loader_channels:", loader_channels)

    CORE_IDX = 0
    core_checkpoint = load_from_dir(args.core_checkpoint)[CORE_IDX] if args.core_checkpoint is not None else None
    liftings = load_from_dir(args.lift_checkpoint_dir) if args.lift_checkpoint_dir is not None else None
    projections = load_from_dir(args.proj_checkpoint_dir) if args.proj_checkpoint_dir is not None else None

    model_blocks = build_model(
        loader_channels,
        config["model"],
        core_checkpoint,
        liftings,
        projections,
    )

    if not isinstance(model_blocks, tuple):
        print(f'Currently, we are aimed only on lifting-core-projections architectures, instead got a single model {type(model_blocks)}.')

    if isinstance(model_blocks, tuple):
        model = Muno(liftings=model_blocks[0], core=model_blocks[1], projections=model_blocks[2])
    else:
        raise RuntimeError("Expected Muno blocked model, instead got a single torch.nn.Module.")

    mc_config = config["inference"].get("mc_drop_out", {})
    if mc_config:
        mc_method = mc_config.get("method", "bernoulli")
        for i, lift in enumerate(model._liftings):
            if not isinstance(lift, (LiftingFeatureDropout, LiftingGaussianPerturbation)):
                if mc_method == "bernoulli":
                    model._liftings[i] = LiftingFeatureDropout(lift, p=0.1)
                elif mc_method == "gaussian":
                    model._liftings[i] = LiftingGaussianPerturbation(lift, p=0.1)
                else:
                    raise ValueError(
                        f"Unknown MC method '{mc_method}'. "
                        f"Expected one of: 'bernoulli', 'gaussian'."
                    )

    model.to(device)
    reset_positional_embedding_caches(model)

    def get_channelwise_reduce_dims(batch_tensor):
        if not isinstance(batch_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(batch_tensor)}")
        if batch_tensor.ndim < 3:
            raise ValueError(f"Expected batched tensor [B, C, ...], got shape {tuple(batch_tensor.shape)}")

        return [0] + list(range(2, batch_tensor.ndim))

    first_batch = next(iter(train_loader))

    dims_x = {i: get_channelwise_reduce_dims(subbatch['x']) for i, subbatch in first_batch.items()}
    dims_y = {i: get_channelwise_reduce_dims(subbatch['y']) for i, subbatch in first_batch.items()}

    in_normalizer = MultiphysicsUnitGaussianNormalizer(num=len(model_blocks[0]), dim=dims_x, key='x')
    inp_norm_files = get_all_files(args.in_normalizers, '.pkl')
    in_normalizer.from_file(inp_norm_files)

    out_normalizer = MultiphysicsUnitGaussianNormalizer(num=len(model_blocks[0]), dim=dims_y, key='y')
    out_norm_files = get_all_files(args.out_normalizers, '.pkl')
    out_normalizer.from_file(out_norm_files)

    in_normalizer.to(device)
    out_normalizer.to(device)
    data_processors = DefaultDataProcessor(
        in_normalizer=in_normalizer,
        out_normalizer=out_normalizer,
        device=device,
    )

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "optimizer": None,
        "normalizer": data_processors,
    }


def main():
    args = parse_args()

    config_path = resolve_path(args.config)
    config = load_yaml_config(config_path)

    seed_config = config.get("seed", {})
    if seed_config.get("value") is not None:
        seed = set_global_seed(
            seed_config["value"],
            deterministic=seed_config.get("deterministic", False),
        )
        print(f"seed: {seed}")

    output_root = (
        args.output_root
        if args.output_root is not None
        else config.get("output", {}).get("root", "runs")
    )
    run_prefix = args.run_name if args.run_name is not None else config_path.stem

    metrics_config = config.get("metrics", {})
    assert metrics_config, 'No metrics were passed for evaluation.'

    exp_config = config["inference"]["experiment"]
    T_list = exp_config["T"]
    p_list = exp_config["p"]
    assert len(T_list) == len(p_list), 'T and p arrays must have the same length.'
    assert len(p_list) > 0, 'No experiment points provided.'

    setup = setup_experiment(config, args)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = resolve_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for idx, (p_i, T_i) in enumerate(zip(p_list, T_list)):
        config["inference"]["mc_drop_out"]["p"] = p_i
        config["inference"]["mc_drop_out"]["T"] = T_i

        run_name = f"{run_prefix}_exp{idx:02d}_p{p_i}_T{T_i}_{timestamp}"
        output_dir = output_root / run_name
        eval_metrics_dir = output_dir / "metrics"
        eval_metrics_dir.mkdir(parents=True, exist_ok=True)

        val_metrics = evaluate_loader(
            model=setup["model"],
            loader=setup["val_loader"],
            data_processor=setup["normalizer"],
            metrics_config=metrics_config,
            task_name="val",
            output_dir=output_dir,
            inference_config=config["inference"],
            k=None,
        )
        print(f'[{idx}] p={p_i} T={T_i} val_metrics: {val_metrics}')
        with open(os.path.join(eval_metrics_dir, "val_metrics.pkl"), "wb") as file:
            pickle.dump(val_metrics, file)

        k = {
            task: task_metrics["k"]
            for task, task_metrics in val_metrics.items()
            if "k" in task_metrics
        }
        k = k or None

        test_metrics = evaluate_loader(
            model=setup["model"],
            loader=setup["test_loader"],
            data_processor=setup["normalizer"],
            metrics_config=metrics_config,
            task_name="test",
            output_dir=output_dir,
            inference_config=config["inference"],
            k=k,
        )
        print(f'[{idx}] p={p_i} T={T_i} test_metrics: {test_metrics}')
        with open(os.path.join(eval_metrics_dir, "test_metrics.pkl"), "wb") as file:
            pickle.dump(test_metrics, file)

        print(f'[{idx}] output_dir: {output_dir}')

    print("done")


if __name__ == "__main__":
    main()