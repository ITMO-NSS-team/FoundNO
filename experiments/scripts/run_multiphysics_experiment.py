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

from muno.data.benchmarks.datasets import MultiPhysicsDataset
from muno.data.benchmarks.normalization import build_data_processors
from muno.data.benchmarks.inspections import inspect_tasks
from muno.utils.custom_trainer import Trainer
from muno.utils.training_utils import BalancedRelL2Loss
from muno.utils.model_factory import build_model
from muno.data.benchmarks.evaluation import (
    evaluate_multitask_loaders,
    save_metrics,
)

from muno.models.muno import Muno

def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_run_metadata(output_dir, config_path, config, task_metadata, loader_channels):
    metadata = {
        "config_path": str(config_path),
        "config": config,
        "task_metadata": task_metadata,
        "loader_channels": loader_channels,
    }

    path = output_dir / "run_metadata.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(f"metadata saved to: {path}")


def save_data_processors(data_processors, output_dir):
    normalizer_dir = output_dir / "normalizers"
    normalizer_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(data_processors, list):
        for idx, processor in enumerate(data_processors):
            if processor is None:
                continue

            if processor.in_normalizer is not None:
                in_normalizer = copy.deepcopy(processor.in_normalizer).cpu()
                in_normalizer.to_file(str(normalizer_dir / f"input_normalizer_{idx}.pkl"))

            if processor.out_normalizer is not None:
                out_normalizer = copy.deepcopy(processor.out_normalizer).cpu()
                out_normalizer.to_file(str(normalizer_dir / f"output_normalizer_{idx}.pkl"))
    else:
        print(f'data_processors: {data_processors}, in: {data_processors.in_normalizer}, out {data_processors.out_normalizer}')
        in_normalizer = copy.deepcopy(data_processors.in_normalizer)
        in_normalizer.to('cpu')
        out_normalizer = copy.deepcopy(data_processors.out_normalizer)
        out_normalizer.to('cpu')

        in_names = [str(normalizer_dir / f"input_normalizer_{idx}.pkl") for idx in range(len(in_normalizer))]
        out_names = [str(normalizer_dir / f"output_normalizer_{idx}.pkl") for idx in range(len(out_normalizer))]
        in_normalizer.to_file(in_names)
        out_normalizer.to_file(out_names)

    print(f"normalizers saved to: {normalizer_dir}")


def build_optimizer_config(training_config):
    return training_config.get(
        "optimizer",
        {
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-5,
        },
    )


def build_scheduler_config(training_config, epochs):
    scheduler_config = training_config.get(
        "scheduler",
        {
            "scheduler": "reducelr",
            "patience": 2,
            "factor": 0.5,
            "min_lr": 1e-6,
        },
    )

    if scheduler_config.get("scheduler") == "cosine":
        scheduler_config["max_cosine_lr_epochs"] = epochs

    return scheduler_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/configs/pdebench_multiphysics_pretrain.yaml",
        help="Path to YAML experiment config.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train the model.")
    parser.add_argument("--device", type=int, default=0, help="Index of the used GPU device.")

    parser.add_argument("--core-checkpoint", default=None, help="Path to pre-trained core checkpoints.")
    parser.add_argument("--lift-checkpoint-dir", default=None, help="Path to pre-trained core checkpoints.")
    parser.add_argument("--proj-checkpoint-dir", default=None, help="Path to pre-trained core checkpoints.")

    parser.add_argument("--run-name", default=None, help="Name, with which the training results will be stored.")
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()

def loadData(task_configs): # , seed = None
    # TODO: add processor for download = False
    train_set, val_set, test_set, task_metadata = build_multitask_datasets(task_configs) # , seed = seed
    return train_set, val_set, test_set, task_metadata

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
        model = Muno(single_model = model_blocks)

    trainer = Trainer(backup_loc=str(checkpoint_dir)) # , devices=devices

    trainer.gradient_accumulation_steps = int(training_config["gradient_accumulation_steps"])
    print(f"gradient_accumulation_steps: {trainer.gradient_accumulation_steps}")

    trainer.setLogger(filename=str(log_dir / "train.log"))
    if seed is not None:
        trainer._logger.write(
            f"seed {seed} | deterministic {seed_config.get('deterministic', False)}"
        )

    trainer.train_main_fno = training_config.get("train_main_fno", False)
    print(f"train_main_fno: {trainer.train_main_fno}")

    trainer.buildModel(model)
    trainer.to(device)

    # trainer.save_paths = (
    #     [str(checkpoint_dir / f"lift_{idx}.pt") for idx in range(len(train_loader))],
    #     str(checkpoint_dir / "core.pt"),
    #     [str(checkpoint_dir / f"proj_{idx}.pt") for idx in range(len(train_loader))]
    # )

    trainer.save_paths = (
        [str(checkpoint_dir / f"lift_{idx}.pt") for idx in range(len(model_blocks[0]))],
        str(checkpoint_dir / "core.pt"),
        [str(checkpoint_dir / f"proj_{idx}.pt") for idx in range(len(model_blocks[0]))]
    )

    trainer.buildOptimizer(
        n_dim=model_config.get("n_dim", 2),
        params_scheduler=build_scheduler_config(training_config, epochs),
        params_opt=build_optimizer_config(training_config),
        trainer_loss=BalancedRelL2Loss()
    )

    data_processors = build_data_processors(
        train_loader,
        config=config.get("normalization"),
        device=device
    )
    save_data_processors(data_processors, output_dir)

    print('-' * 30)
    print(f'Training model on devices {device}.')
    print('-' * 30)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        train_epochs=epochs,
        data_processor=data_processors
    )

    metrics_config = config.get("metrics", {})
    evaluation_config = config.get("evaluation", {})
    evaluate_after_training = evaluation_config.get("after_training", True)

    if evaluate_after_training and metrics_config:
        val_metrics = evaluate_multitask_loaders(
            trainer=trainer,
            loaders=val_loader,
            data_processors=data_processors,
            task_metadata=metadata,
            metrics_config=metrics_config,
            split_name="val"
        )
        save_metrics(
            val_metrics,
            output_dir=output_dir,
            filename_stem="val_metrics"
        )

        test_metrics = evaluate_multitask_loaders(
            trainer=trainer,
            loaders=test_loader,
            data_processors=data_processors,
            task_metadata=metadata,
            metrics_config=metrics_config,
            split_name="test"
        )
        save_metrics(
            test_metrics,
            output_dir=output_dir,
            filename_stem="test_metrics"
        )

    print("done")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
