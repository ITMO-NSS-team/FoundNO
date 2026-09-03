import csv
import json
from pathlib import Path

import torch

from muno.utils.metrics import compute_metrics
from muno.utils.metrics_physical import compute_physical_metrics


def prefix_metrics(metrics, prefix):
    return {
        f"{prefix}/{name}": value
        for name, value in metrics.items()
    }


def filter_physical_metric_configs(metric_configs, task_name):
    selected_configs = []

    for config in metric_configs or []:
        tasks = config.get("tasks", "all")

        if tasks == "all" or task_name in tasks:
            clean_config = {
                key: value
                for key, value in config.items()
                if key != "tasks"
            }
            selected_configs.append(clean_config)

    return selected_configs


def compute_batch_metrics(pred, target, metrics_config, task_name):
    results = {}

    metric_names = metrics_config.get("names", [])
    if metric_names:
        results.update(
            compute_metrics(
                pred,
                target,
                metric_names=metric_names,
            )
        )

    physical_configs = filter_physical_metric_configs(
        metrics_config.get("physical", []),
        task_name,
    )
    if physical_configs:
        results.update(
            compute_physical_metrics(
                pred,
                target,
                metric_configs=physical_configs,
            )
        )

    return results


def set_trainer_eval_mode(trainer):
    if trainer._single_model:
        trainer.model.eval()
    else:
        trainer.main_fno.eval()

        for input_adapter, output_adapter in zip(
            trainer.input_adapters,
            trainer.output_adapters,
        ):
            input_adapter.eval()
            output_adapter.eval()


def predict_batch(trainer, sample, data_processor=None):
    sample = dict(sample)

    sample["x"] = sample["x"].to(trainer.device)
    sample["y"] = sample["y"].to(trainer.device)

    if "mask" in sample:
        sample["mask"] = sample["mask"].to(trainer.device)

    if data_processor is not None:
        sample = data_processor.preprocess(sample, training=False)

    if trainer._single_model:
        pred = trainer.model(sample["x"])
    else:
        eq_idx = int(sample["eq_idx"][0].item())
        pred = trainer.input_adapters[eq_idx](sample["x"])
        pred = trainer.main_fno(pred)
        pred = trainer.output_adapters[eq_idx](pred)

    if data_processor is not None:
        pred, sample = data_processor.postprocess(
            pred,
            sample,
            training=False,
        )

    return pred, sample["y"]


def evaluate_loader(
    trainer,
    loader,
    data_processor,
    metrics_config,
    task_name,
):
    metric_sums = {}
    n_samples = 0

    set_trainer_eval_mode(trainer)

    with torch.no_grad():
        for sample in loader:
            pred, target = predict_batch(
                trainer,
                sample,
                data_processor=data_processor,
            )

            batch_metrics = compute_batch_metrics(
                pred,
                target,
                metrics_config=metrics_config,
                task_name=task_name,
            )

            batch_size = int(target.shape[0])
            n_samples += batch_size

            for name, value in batch_metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value) * batch_size

    if n_samples == 0:
        return {}

    return {
        name: value / n_samples
        for name, value in metric_sums.items()
    }


def evaluate_multitask_loaders(
    trainer,
    loaders,
    data_processors,
    task_metadata,
    metrics_config,
    split_name,
):
    results = {}

    for task_idx, loader in enumerate(loaders):
        task_name = task_metadata[task_idx].get("name", f"task_{task_idx}")
        data_processor = data_processors[task_idx]

        task_metrics = evaluate_loader(
            trainer=trainer,
            loader=loader,
            data_processor=data_processor,
            metrics_config=metrics_config,
            task_name=task_name,
        )

        results.update(
            prefix_metrics(
                task_metrics,
                f"{split_name}/{task_name}",
            )
        )

    return results


def save_metrics(metrics, output_dir, filename_stem):
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    json_path = metrics_dir / f"{filename_stem}.json"
    csv_path = metrics_dir / f"{filename_stem}.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])

        for name, value in metrics.items():
            writer.writerow([name, value])

    print(f"metrics saved to: {json_path}")
    print(f"metrics saved to: {csv_path}")
