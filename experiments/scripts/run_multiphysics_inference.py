import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# import socket

# def findFreePort():
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind(('', 0))
#         s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         return s.getsockname()[1]

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = str(findFreePort())
# print(f'USING MASTER PORT {os.environ["MASTER_PORT"]}')

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from muno.utils.seed import set_global_seed
from muno.data.benchmarks.config_io import load_yaml_config
from muno.data.benchmarks.multiphysics_loaders import (
    build_multitask_loaders,
    get_loaders_channels,
    build_multitask_datasets
)

from muno.utils.metrics import compute_metrics
from muno.data.benchmarks.evaluation import filter_physical_metric_configs, compute_physical_metrics

from muno.data.benchmarks.datasets import MultiPhysicsDataset
from muno.data.benchmarks.normalization import build_data_processors
from muno.data.benchmarks.inspections import inspect_tasks
from muno.utils.custom_trainer import Trainer
from muno.utils.training_utils import BalancedRelL2Loss
from muno.utils.model_factory import build_model, load_from_dir, get_all_files
from muno.data.benchmarks.evaluation import (
    evaluate_multitask_loaders,
    save_metrics,
    compute_batch_metrics
)

from muno.models.muno import Muno

def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/configs/pdebench_multiphysics_pretrain.yaml",
        help="Path to YAML experiment config.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train the model.")
    parser.add_argument("--device", type=int, default=0, help="Index of the used GPU device.")

    parser.add_argument("--core-checkpoint", default=None, help="Path to pre-trained core checkpoints, a single file.")
    parser.add_argument("--lift-checkpoint-dir", default=None, help="Path to pre-trained core checkpoints, a directory.")
    parser.add_argument("--proj-checkpoint-dir", default=None, help="Path to pre-trained core checkpoints, a directory.")

    parser.add_argument("--in-normalizers", default=None, help="Path to pre-trained input normalizers, a directory.")
    parser.add_argument("--out-normalizers", default=None, help="Path to pre-trained output normalizers, a directory.")

    parser.add_argument("--run-name", default=None, help="Name, with which the training results will be stored.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--sample-idx", type=int, default=-1, help="Index of sample, to be used for prediction. \
                                                                    -1 denotes an entire dataset.")
    return parser.parse_args()

def loadData(task_configs): # , seed = None
    # TODO: add processor for download = False
    train_set, val_set, test_set, task_metadata = build_multitask_datasets(task_configs) # , seed = seed
    return train_set, val_set, test_set, task_metadata

def predict_batch(model, sample, data_processor=None, device='cuda:0'):
    assert isinstance(sample, dict), 'Sample has to be passed as dict.'
    test_key = list(sample.keys())[0]
    assert isinstance(sample[test_key], dict), \
        'A sample, obtained for a single-physics dataset has to be a dict.'

    for key in sample.keys():    
        sample[key]["x"] = sample[key]["x"].to(device)
        sample[key]["y"] = sample[key]["y"].to(device)

        if "mask" in sample[key].keys():
            sample[key]["mask"] = sample[key]["mask"].to(device)

    if data_processor is not None:
        sample = data_processor.preprocess(sample, training=False)

    out = model({key: sample[key]["x"] for key in sample})

    if data_processor is not None:
        out, sample = data_processor.postprocess(out, sample, training=False)

    assert isinstance(out, dict), 'Multisample model prediction is expected to be a dict.'
    return out, {sample[key]["y"] for key in sample.keys()}

def compute_batch_metrics(pred: dict, target: dict, metrics_config, task_name):
    results = {}

    metric_names = metrics_config.get("names", [])
    if metric_names:
        results.update({key: compute_metrics(pred[key], target[key], metric_names=metric_names)
                        for key in pred.keys()})

    physical_configs = filter_physical_metric_configs(
        metrics_config.get("physical", []),
        task_name,
    )
    if physical_configs:
        results.update({key: compute_physical_metrics(pred, target, metric_configs=physical_configs)
                        for key in pred.keys()})

    return results

def evaluate_loader(
    model,
    loader,
    data_processor,
    metrics_config,
    task_name,
):
    metric_sums = {}
    n_samples = 0

    with torch.no_grad():
        for sample in loader:
            pred, target = predict_batch(
                model,
                sample,
                data_processor=data_processor,
            )

            batch_metrics = compute_batch_metrics(pred, target, metrics_config=metrics_config,
                                                  task_name=task_name)

    return batch_metrics


def main():
    args = parse_args()

    config_path = resolve_path(args.config)
    config = load_yaml_config(config_path)
    seed = None
    seed_config = config.get("seed", {})

    if seed_config.get("value") is not None:
        seed = set_global_seed(
            seed_config["value"],
            deterministic=seed_config.get("deterministic", False),
        )
        print(f"seed: {seed}")

    training_config = config.get("training", {})
    model_config = config.get("model", {})

    epochs = args.epochs if args.epochs is not None else training_config.get("epochs", 1)
    # device = args.device if args.device is not None else training_config.get("device", "cuda")

    device = f'cuda:{args.device}' #[i for i in range(torch.cuda.device_count())] #[int(arg) for arg in devices]

    output_config = config.get("output", {})
    output_root = (
        args.output_root
        if args.output_root is not None
        else output_config.get("root", "runs")
    )

    run_prefix = args.run_name if args.run_name is not None else config_path.stem
    run_name = f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = resolve_path(output_root) / run_name
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    task_configs = config["tasks"]
    inspections_config = config.get("inspections", {})
    if inspections_config.get("enabled", False):
        inspect_tasks(
            task_configs,
            output_dir / "inspections"
        )

    train_set, val_set, test_set, metadata = loadData(task_configs)
    train_set = MultiPhysicsDataset(train_set)
    val_set   = MultiPhysicsDataset(val_set)
    test_set  = MultiPhysicsDataset(test_set)

    train_loader = DataLoader(dataset = train_set, shuffle = False)
    val_loader   = DataLoader(dataset = val_set,   shuffle = False)
    test_loader  = DataLoader(dataset = test_set,  shuffle = False)    

    loader_channels = get_loaders_channels(train_loader)
    print("loader_channels:", loader_channels)

    write_run_metadata(output_dir, config_path, config, metadata, loader_channels)

    CORE_IDX = 0
    core_checkpoint = load_from_dir(args.core_checkpoint)[CORE_IDX] if args.core_checkpoint is not None else None
    liftings = load_from_dir(args.lift_checkpoint_dir) if args.lift_checkpoint_dir is not None else None
    projections = load_from_dir(args.lift_checkpoint_dir) if args.proj_checkpoint_dir is not None else None

    model_blocks = build_model(loader_channels, model_config, 
                               core_checkpoint, liftings, projections)

    if not isinstance(model_blocks, tuple):
        print(f'Currently, we are aimed only on lifting-core-projections architectures, instead got a single model {type(model_blocks)}.')


    if isinstance(model_blocks, tuple):
        model = Muno(liftings = model_blocks[0], core = model_blocks[1], projections = model_blocks[2])
    else:
        raise RuntimeError("Expected Muno blocked model, instead got a single torch.nn.Module.")
        model = Muno(single_model = model_blocks)


    from muno.data import UnitGaussianNormalizer, MultiphysicsUnitGaussianNormalizer
    from muno.data.data.transforms.data_processors import DefaultDataProcessor

    def get_channelwise_reduce_dims(batch_tensor):
        if not isinstance(batch_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(batch_tensor)}")
        if batch_tensor.ndim < 3:
            raise ValueError(f"Expected batched tensor [B, C, ...], got shape {tuple(batch_tensor.shape)}")

        return [0] + list(range(2, batch_tensor.ndim))

    first_batch = next(iter(train_loader))

    dims = {i: get_channelwise_reduce_dims(subbatch[key]) for i, subbatch in first_batch.items()}

    in_normalizer = MultiphysicsUnitGaussianNormalizer(num=len(model_blocks[0]), dim = dims, key = 'x')
    inp_norm_files = get_all_files(args.in_normalizers, '.pickle')
    in_normalizer.from_file(inp_norm_files)

    out_normalizer = MultiphysicsUnitGaussianNormalizer(num=len(model_blocks[0]), dim = dims, key = 'y')
    out_norm_files = get_all_files(args.out_normalizers, '.pickle')
    out_normalizer.from_file(out_norm_files)

    data_processors = DefaultDataProcessor(in_normalizer=in_normalizer,
                                           out_normalizer=out_normalizer,
                                           device=device)

    metrics_config = config.get("metrics", {})
    assert metrics_config, 'No metrics were passed for evaluation.'

    import pickle
    eval_metrics_dir = output_dir / "metrics"
    eval_metrics_dir.mkdir(parents=True, exist_ok=True)

    train_metrics = evaluate_loader(
        model=model,
        loaders=train_loader,
        data_processors=data_processors,
        task_metadata=metadata,
        metrics_config=metrics_config,
        split_name="train"
    )
    print(f'train_metrics: {train_metrics}')
    with open(os.path.join(eval_metrics_dir, "train_metrics.pkl"), "wb") as file:
        pickle.dump(train_metrics, file)

    val_metrics = evaluate_loader(
        model=model,
        loaders=val_loader,
        data_processors=data_processors,
        task_metadata=metadata,
        metrics_config=metrics_config,
        split_name="val"
    )
    print(f'train_metrics: {val_metrics}')
    with open(os.path.join(eval_metrics_dir, "val_metrics.pkl"), "wb") as file:
        pickle.dump(val_metrics, file)

    test_metrics = evaluate_loader(
        model=model,
        loaders=test_loader,
        data_processors=data_processors,
        task_metadata=metadata,
        metrics_config=metrics_config,
        split_name="test"
    )
    print(f'train_metrics: {test_metrics}')
    with open(os.path.join(eval_metrics_dir, "test_metrics.pkl"), "wb") as file:
        pickle.dump(test_metrics, file)

    print("done")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
