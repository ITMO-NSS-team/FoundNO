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
from muno.utils.metrics_physical import compute_physical_metrics
from muno.uncertainty.metrics_uq import compute_uq_metrics
from muno.data.benchmarks.evaluation import filter_physical_metric_configs

from muno.data.benchmarks.datasets import MultiPhysicsDataset
from muno.data.benchmarks.normalization import build_data_processors
from muno.data.benchmarks.inspections import inspect_tasks, result_img_save, result_uq_img_save
from muno.utils.custom_trainer import Trainer
from muno.utils.training_utils import BalancedRelL2Loss
from muno.utils.model_factory import build_model, load_from_dir, get_all_files
from muno.data.benchmarks.evaluation import (
    evaluate_multitask_loaders,
    save_metrics
)

from muno.models.muno import Muno
from muno.models.mamba_fno import LiftingFeatureDropout, LiftingGaussianPerturbation

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
    return out, {key: sample[key]["y"] for key in sample.keys()}

def compute_batch_metrics(pred: dict, band: dict, target: dict, metrics_config, task_name, k=None):
    results = {}
    metric_names = metrics_config.get("names", [])

    physical_configs = filter_physical_metric_configs(
        metrics_config.get("physical", []),
        task_name,
    )
    uq_configs = filter_physical_metric_configs(
        metrics_config.get("uncertainty", []),
        task_name,
    )

    for key in pred.keys():
        entry = {}
        if metric_names:
            entry.update(compute_metrics(pred[key], target[key], metric_names=metric_names))
        if physical_configs:
            entry.update(compute_physical_metrics(pred[key], target[key], metric_configs=physical_configs))
        if uq_configs:
            k_key = k.get(key, 1.0) if isinstance(k, dict) else k
            k_cfgs = [{**config, "k": k_key} for config in uq_configs] if k is not None else uq_configs
            entry.update(compute_uq_metrics(pred[key], band[key], target[key], metric_configs=k_cfgs))
        results[key] = entry

    return results


def compute_batch_metrics_per_time(pred: dict, band: dict, target: dict, metrics_config, task_name, k=None):
    results = {}
    metric_names = metrics_config.get("names", [])

    physical_configs = filter_physical_metric_configs(
        metrics_config.get("physical", []),
        task_name,
    )
    uq_configs = filter_physical_metric_configs(
        metrics_config.get("uncertainty", []),
        task_name,
    )

    for key in pred.keys():
        pred_t, band_t, target_t = pred[key], band[key], target[key]
        n_time = pred_t.shape[2] if pred_t.ndim >= 4 else 1

        per_time = {}
        for t in range(n_time):
            if n_time > 1:
                p, b, y = pred_t[:, :, t], band_t[:, :, t], target_t[:, :, t]
            else:
                p, b, y = pred_t, band_t, target_t

            entry = {}
            if metric_names:
                entry.update(compute_metrics(p, y, metric_names=metric_names))
            if physical_configs:
                entry.update(compute_physical_metrics(p, y, metric_configs=physical_configs))
            if uq_configs:
                k_key = k.get(key, 1.0) if isinstance(k, dict) else k
                k_cfgs = [{**config, "k": k_key} for config in uq_configs] if k is not None else uq_configs
                entry.update(compute_uq_metrics(p, b, y, metric_configs=k_cfgs))
            per_time[t] = entry

        entry = {}
        for name in per_time[0]:
            values = torch.tensor([per_time[t][name] for t in range(n_time)], dtype=torch.float32)
            entry[name] = values
        if k is not None:
            k_key = k.get(key, 1.0) if isinstance(k, dict) else k
            entry["k"] = float(k_key)
        results[key] = entry

    return results

    
def set_mc_lifting(model: 'Muno', on: bool, adapter_idx: int = None, last_layer_drop: bool = False, p: float = 0.1):
    """Turn structure-aware sampling on/off for one adapter, or all if adapter_idx is None."""
    #model.eval()  # core + projections always deterministic
    indices = range(len(model._liftings)) if adapter_idx is None else [adapter_idx]
    for i in indices:
        lift = model._liftings[i]
        if not isinstance(lift, (LiftingFeatureDropout, LiftingGaussianPerturbation)):
            raise TypeError(
                f"Lifting adapter {i} is {type(lift).__name__}, not wrapped with "
                f"LiftingFeatureDropout/LiftingGaussianPerturbation. "
                f"Call the wrapping step first."
            )
        lift.sample_noise = on
        lift.last_layer_drop = last_layer_drop
        lift.p = 0.0 if not on else p

@torch.no_grad()
def mc_lifting_predict(model, sample, data_processor=None, on_dropout= False, last_layer_drop = True, adapter_idx=0, T=20, p=0.1, device='cuda:0', **kwargs):
    for key in sample.keys():
        sample[key]["x"] = sample[key]["x"].to(device)
        sample[key]["y"] = sample[key]["y"].to(device)
        if "mask" in sample[key].keys():
            sample[key]["mask"] = sample[key]["mask"].to(device)

    if data_processor is not None:
        sample = data_processor.preprocess(sample, training=False)

    x = {key: sample[key]["x"] for key in sample}
    target = {key: sample[key]["y"] for key in sample.keys()}

    set_mc_lifting(model, on=on_dropout, adapter_idx=adapter_idx, last_layer_drop = last_layer_drop, p=p)
    raw_samples = [model(x, adapter_idx=adapter_idx, **kwargs) for _ in range(T)]

    if isinstance(raw_samples[0], dict):
        keys = raw_samples[0].keys()
        mean, band = {}, {}
        for key in keys:
            stacked = torch.stack([s[key] for s in raw_samples], dim=0)
            mean[key] = stacked.mean(0)
            band[key] = stacked.pow(2).mean(0).sub(mean[key].pow(2)).clamp_min(0).sqrt()
    else:
        stacked = torch.stack(raw_samples, dim=0)
        mean = stacked.mean(0)
        band = stacked.pow(2).mean(0).sub(mean.pow(2)).clamp_min(0).sqrt()

    if data_processor is not None:
        mean, sample = data_processor.postprocess(mean, sample, training=False)

    return mean, band, target


def calibrate_k(band_raw, target, pred, target_coverage=0.9, k_min=0.1, k_max=5.0, eps=1e-12):
    # band_raw, residuals — на КАЛИБРОВОЧНОМ сплите (не train, не test)
    residuals = (pred - target)
    residuals = residuals.abs().flatten()
    band = band_raw.flatten()

    ratio = torch.full_like(residuals, float("inf"))
    nonzero_band = band > eps
    ratio[nonzero_band] = residuals[nonzero_band] / band[nonzero_band].clamp_min(eps)
    ratio[~nonzero_band] = torch.where(
        residuals[~nonzero_band] > eps,
        torch.tensor(float("inf"), dtype=ratio.dtype, device=ratio.device),
        torch.tensor(0.0, dtype=ratio.dtype, device=ratio.device),
    )

    k = torch.clamp(
    torch.quantile(ratio, target_coverage, interpolation="higher"),
    k_min, k_max,
    ).item()

    coverage = (residuals <= k * band + eps).float().mean().item()
    if coverage >= target_coverage:
        return k
    return k_max

def evaluate_loader(
    model,
    loader,
    data_processor,
    metrics_config,
    task_name,
    output_dir,
    inference_config=None,
    k=None,
):
    metric_sums = {}
    n_batches = 0

    mc_config = (inference_config or {}).get("mc_drop_out", {})
    use_mc = bool(mc_config)
    T = mc_config.get("T", 10)
    on_dropout = mc_config.get("on_dropout", True)
    lift_last_layer_drop = mc_config.get("lift_last_layer_drop", True)
    p = mc_config.get("p", 0.1)
    target_coverage =  mc_config.get("target_coverage", 0.9)
    k_sums = {}
    k_counts = {}

    model.eval()
    output_prefix = Path(output_dir / f"inspections_inf_{task_name}")
    output_prefix.mkdir(parents=True, exist_ok=True)
    output_prefix = Path.joinpath(output_prefix, "inf_")
    with torch.no_grad():
        for sample in loader:
            if use_mc:
                pred, band, target = mc_lifting_predict(
                    model, sample, data_processor=data_processor, T=T, 
                    on_dropout=on_dropout, last_layer_drop = lift_last_layer_drop, p = p,
                    device = data_processor.device
                )
                if task_name == "val":
                    k = {
                        key: calibrate_k(band[key], target[key], pred[key], target_coverage=target_coverage)
                        for key in pred
                    }
                    for key, k_value in k.items():
                        k_sums[key] = k_sums.get(key, 0.0) + k_value
                        k_counts[key] = k_counts.get(key, 0) + 1
                if k is not None:
                    result_uq_img_save(pred, band, target, output_prefix, n_batches, k=k)
            else:
                pred, target = predict_batch(
                    model,
                    sample,
                    data_processor=data_processor,
                    device = data_processor.device
                )
                band = pred

            result_img_save(pred, target, output_prefix, n_batches)

            batch_metrics = compute_batch_metrics(
                pred,
                band,
                target,
                metrics_config=metrics_config,
                task_name=task_name,
                k=k,
            )
            # batch_metrics = compute_batch_metrics_per_time(
            #     pred,
            #     band,
            #     target,
            #     metrics_config=metrics_config,
            #     task_name=task_name,
            #     k=k,
            # )
            
            for task_name, task_metrics in batch_metrics.items():
                if task_name not in metric_sums:
                    metric_sums[task_name] = {}

                for name, value in task_metrics.items():
                    metric_sums[task_name][name] = (
                        metric_sums[task_name].get(name, 0.0) + value
                    )

            n_batches += 1

    metrics = {
        task_name: {
            name: value / n_batches
            for name, value in task_metrics.items()
        }
        for task_name, task_metrics in metric_sums.items()
    }

    final_k = {}
    if k_sums:
        final_k = {
            key: k_sums[key] / k_counts[key]
            for key in k_sums
        }
    elif k is not None:
        final_k = k

    if final_k:
        for task_name in metrics:
            metrics[task_name]["k"] = final_k[task_name] if task_name in final_k else float("nan")

    return metrics


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
    inference_config = config.get("inference", {})
    mc_config = inference_config.get("mc_drop_out", {})
    use_inference_mc = bool(mc_config)
    mc_method = mc_config.get("method", "bernoulli")
    target_coverage = mc_config.get("target_coverage", 0.9)

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

    #write_run_metadata(output_dir, config_path, config, metadata, loader_channels)

    CORE_IDX = 0
    core_checkpoint = load_from_dir(args.core_checkpoint)[CORE_IDX] if args.core_checkpoint is not None else None
    liftings = load_from_dir(args.lift_checkpoint_dir) if args.lift_checkpoint_dir is not None else None
    projections = load_from_dir(args.proj_checkpoint_dir) if args.proj_checkpoint_dir is not None else None

    model_blocks = build_model(loader_channels, model_config, 
                               core_checkpoint, liftings, projections)

    if not isinstance(model_blocks, tuple):
        print(f'Currently, we are aimed only on lifting-core-projections architectures, instead got a single model {type(model_blocks)}.')


    if isinstance(model_blocks, tuple):
        model = Muno(liftings = model_blocks[0], core = model_blocks[1], projections = model_blocks[2])
    else:
        raise RuntimeError("Expected Muno blocked model, instead got a single torch.nn.Module.")
        model = Muno(single_model = model_blocks)


    if use_inference_mc:
        for i, lift in enumerate(model._liftings):
            if not isinstance(lift, (LiftingFeatureDropout, LiftingGaussianPerturbation)):
                if mc_method == "bernoulli":
                    model._liftings[i] = LiftingFeatureDropout(lift, p=0.1)
                elif  mc_method == "gaussian":
                    model._liftings[i] = LiftingGaussianPerturbation(lift, p=0.1)
                else:
                    raise ValueError(
                        f"Unknown MC method '{mc_method}'. "
                        f"Expected one of: 'bernoulli', 'gaussian'."
                    )

    model.to(device)

    from muno.data.data.transforms.normalizers import UnitGaussianNormalizer, MultiphysicsUnitGaussianNormalizer
    from muno.data.data.transforms.data_processors import DefaultDataProcessor

    def get_channelwise_reduce_dims(batch_tensor):
        if not isinstance(batch_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(batch_tensor)}")
        if batch_tensor.ndim < 3:
            raise ValueError(f"Expected batched tensor [B, C, ...], got shape {tuple(batch_tensor.shape)}")

        return [0] + list(range(2, batch_tensor.ndim))

    first_batch = next(iter(train_loader))

    dims_x = {i: get_channelwise_reduce_dims(subbatch['x']) for i, subbatch in first_batch.items()}
    dims_y = {i: get_channelwise_reduce_dims(subbatch['y']) for i, subbatch in first_batch.items()}

    in_normalizer = MultiphysicsUnitGaussianNormalizer(num=len(model_blocks[0]), dim = dims_x, key = 'x')
    print(args.in_normalizers)
    inp_norm_files = get_all_files(args.in_normalizers, '.pkl')
    in_normalizer.from_file(inp_norm_files)

    out_normalizer = MultiphysicsUnitGaussianNormalizer(num=len(model_blocks[0]), dim = dims_y, key = 'y')
    out_norm_files = get_all_files(args.out_normalizers, '.pkl')
    out_normalizer.from_file(out_norm_files)

    # for batch in train_loader:
    #     print(batch[0]['x'].shape)
    #     in_normalizer.partial_fit(batch)
    #     out_normalizer.partial_fit(batch)
    # for batch in test_loader:
    #     in_normalizer.partial_fit(batch)
    #     out_normalizer.partial_fit(batch)
    # for batch in val_loader:
    #     in_normalizer.partial_fit(batch)
    #     out_normalizer.partial_fit(batch)

    
    # print(out_normalizer.normalizers[0].mean.flatten())
    # print(out_normalizer.normalizers[0].std.flatten())

    in_normalizer.to(device)
    out_normalizer.to(device)
    data_processors = DefaultDataProcessor(in_normalizer=in_normalizer,
                                           out_normalizer=out_normalizer,
                                           device=device)

    metrics_config = config.get("metrics", {})
    assert metrics_config, 'No metrics were passed for evaluation.'

    import pickle
    eval_metrics_dir = output_dir / "metrics"
    eval_metrics_dir.mkdir(parents=True, exist_ok=True)

    # train_metrics = evaluate_loader(
    #     model=model,
    #     loader=train_loader,
    #     data_processor=data_processors,
    #     #task_metadata=metadata,
    #     metrics_config=metrics_config,
    #     task_name="train",
    #     output_dir = output_dir
    # )
    # print(f'train_metrics: {train_metrics}')
    # with open(os.path.join(eval_metrics_dir, "train_metrics.pkl"), "wb") as file:
    #     pickle.dump(train_metrics, file)

    val_metrics = evaluate_loader(
        model=model,
        loader=val_loader,
        data_processor=data_processors,
        #task_metadata=metadata,
        metrics_config=metrics_config,
        task_name="val",
        output_dir = output_dir,
        inference_config=inference_config,
        k=None,
    )
    print(f'val_metrics: {val_metrics}')
    with open(os.path.join(eval_metrics_dir, "val_metrics.pkl"), "wb") as file:
        pickle.dump(val_metrics, file)

    k = {
        task: task_metrics["k"]
        for task, task_metrics in val_metrics.items()
        if "k" in task_metrics
    }
    k = k or None

    test_metrics = evaluate_loader(
        model=model,
        loader=test_loader,
        data_processor=data_processors,
        #task_metadata=metadata,
        metrics_config=metrics_config,
        task_name="test",
        output_dir = output_dir,
        inference_config=inference_config,
        k=k,
    )
    print(f'test_metrics: {test_metrics}')
    with open(os.path.join(eval_metrics_dir, "test_metrics.pkl"), "wb") as file:
        pickle.dump(test_metrics, file)


    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
