import json
from datetime import datetime
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.models import FNO

from muno.data.benchmarks.config_io import load_yaml_config
from muno.data.benchmarks.normalization import build_data_processors
from muno.data.benchmarks.multiphysics_loaders import (
    build_multitask_loaders,
    get_loaders_channels,
)
from muno.utils.custom_trainer import Trainer
from muno.utils.training_utils import BalancedRelL2Loss


def build_debug_model(loader_channels):
    hidden_channels = 16

    liftings = []
    projections = []

    for in_channels, out_channels in loader_channels:
        liftings.append(
            ChannelMLP(
                in_channels=in_channels,
                out_channels=hidden_channels,
                hidden_channels=hidden_channels,
                n_layers=2,
                n_dim=2,
            )
        )

        projections.append(
            ChannelMLP(
                in_channels=hidden_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                n_layers=2,
                n_dim=2,
            )
        )

    core = FNO(
        in_channels=hidden_channels,
        out_channels=hidden_channels,
        hidden_channels=hidden_channels,
        n_modes=(16, 16),
        n_layers=2,
    )

    return liftings, core, projections


def write_run_metadata(output_dir, task_metadata, loader_channels, config_path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) 

    metadata = {
        "config_path": str(config_path),
        "task_metadata": task_metadata,
        "loader_channels": loader_channels,
    }

    path = output_dir / "run_metadata.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(f"metadata saved to: {path}")


def main():
    config_path = PROJECT_ROOT / "experiments" / "configs" / "poseidon_debug.yaml"
    config = load_yaml_config(config_path)
    task_configs = config["tasks"]

    train_loaders, val_loaders, test_loaders, task_metadata = build_multitask_loaders(task_configs)

    loader_channels = get_loaders_channels(train_loaders)
    print("loader_channels:", loader_channels)

    model = build_debug_model(loader_channels)

    run_name = datetime.now().strftime("debug_multitask_training_%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "runs" / run_name
    backup_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    backup_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    write_run_metadata(output_dir, task_metadata, loader_channels, config_path)

    trainer = Trainer(backup_loc=str(backup_dir))

    log_path = log_dir / "train.log"
    trainer.setLogger(filename=str(log_path))
    trainer.buildModel(model)

    trainer.save_paths = (
        [
            str(backup_dir / f"lift_{idx}.pt")
            for idx in range(len(train_loaders))
        ],
        str(backup_dir / "core.pt"),
        [
            str(backup_dir / f"proj_{idx}.pt")
            for idx in range(len(train_loaders))
        ],
    )

    trainer.buildOptimizer(
        n_dim=2,
        params_scheduler={
            "scheduler": "reducelr",
            "patience": 2,
            "factor": 0.5,
            "min_lr": 1e-6,
        },
        params_opt={
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-5,
        },
        trainer_loss=BalancedRelL2Loss(),
    )

    trainer.to("cuda")

    data_processors = build_data_processors(
        train_loaders,
        config=config.get("normalization"),
        device="cuda",
    )

    trainer.train(
        train_loader=train_loaders,
        val_loader=val_loaders,
        train_epochs=1,
        data_processor=data_processors,
    )

    print("done")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
