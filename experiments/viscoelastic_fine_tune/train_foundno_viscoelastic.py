"""Full pretrained FoundNO fine-tuning on the unchanged official Well splits."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from viscoelastic_data import (FIELD_NAMES, ViscoelasticDataset, inventory,
                               fit_normalization, prepare_batch)
from foundno_viscoelastic import build_model
from memory_efficient_adamw import ChunkedAdamW


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = True


def optimizer_for(model, config):
    training = config["training"]
    groups = [{"name": name, "params": list(module.parameters()),
               "lr": training["core_lr"] if name == "core" else training["adapter_lr"]}
              for name, module in (("core", model.core), ("lifting", model.lifting),
                                   ("projection", model.projection))]
    return ChunkedAdamW(groups, chunk_elements=4194304, weight_decay=training["weight_decay"],
                        betas=(0.9, 0.999), eps=1e-8)


@torch.no_grad()
def evaluate(model, dataset, stats, device, output=None):
    model.eval()
    metrics, persistence, losses = [], [], []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        x, target, true, mean, std = prepare_batch(batch, stats, device)
        prediction = model(x)
        assert torch.isfinite(prediction).all()
        losses.append((prediction - target).square().mean().item())
        physical = prediction * std + mean
        variance = true.var(dim=(-2, -1), unbiased=True) + 1e-7
        scores = ((physical - true).square().mean(dim=(-2, -1)) / variance).sqrt()
        base = ((true[:, :, :1] - true).square().mean(dim=(-2, -1)) / variance).sqrt()
        metrics.append(scores[0].cpu().numpy())
        persistence.append(base[0].cpu().numpy())
    metrics, persistence = np.stack(metrics), np.stack(persistence)
    future = metrics[:, :, 1:]
    summary = {"normalized_mse_all20": float(np.mean(losses)),
               "vrmse_future19": float(future.mean()),
               "median_sample_vrmse_future19": float(np.median(future.mean(axis=(1, 2)))),
               "persistence_vrmse_future19": float(persistence[:, :, 1:].mean()),
               "field_vrmse_future19": {name: float(future[:, i].mean()) for i, name in enumerate(FIELD_NAMES)},
               "vrmse_per_time": metrics.mean(axis=(0, 1)).tolist(),
               "per_regime": {}, "trajectories": len(dataset), "time_indices": list(range(20))}
    for regime in sorted({r["regime"] for r in dataset.records}):
        ids = [i for i, r in enumerate(dataset.records) if r["regime"] == regime]
        summary["per_regime"][regime] = {"trajectories": len(ids),
                                        "normalized_mse_all20": float(np.mean(np.asarray(losses)[ids])),
                                        "vrmse_future19": float(future[ids].mean()),
                                        "persistence_vrmse_future19": float(persistence[ids, :, 1:].mean())}
    if output:
        output = Path(output)
        write_json(output / "summary.json", summary)
        np.savez_compressed(output / "vrmse.npz", vrmse=metrics, persistence_vrmse=persistence)
    return summary


def save_checkpoint(path, model, optimizer, scheduler, epoch, config, stats, metadata, best):
    payload = {"epoch": epoch, "model_state_dict": {k: v for k, v in model.state_dict().items() if torch.is_tensor(v)},
               "config": config, "normalization": stats, "model_metadata": metadata, "best": best}
    if optimizer is not None:
        payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_states(model, payload):
    result = model.load_state_dict(payload["model_state_dict"], strict=False)
    assert not result.unexpected_keys, result
    assert all(k.endswith("_metadata") for k in result.missing_keys), result


def plot_history(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [r["train_normalized_mse"] for r in rows], label="train")
    axes[0].plot(epochs, [r["valid_normalized_mse"] for r in rows], label="valid")
    axes[0].set_title("MSE: same train normalization, all 20 frames")
    axes[0].legend()
    axes[1].plot(epochs, [r["valid_vrmse_future19"] for r in rows])
    axes[1].set_title("Validation VRMSE: future 19 frames")
    axes[2].semilogy(epochs, [r["core_lr"] for r in rows], label="core")
    axes[2].semilogy(epochs, [r["adapter_lr"] for r in rows], label="adapters")
    axes[2].set_title("Learning rate")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(run / "training_curves.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    assert config["data"]["time_indices"] == list(range(20))
    assert config["training"]["finetune_mode"] == "full"
    assert config["training"]["batch_size"] == 1
    assert not config["model"]["freeze_core"]
    torch.set_num_threads(4)
    seed(config["seed"])
    root = Path(config["data"]["root"])
    run = Path(config["output"]["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    try:
        records = inventory(root)
        datasets = {split: ViscoelasticDataset(rs, config["data"]["resolution"],
                                              Path(config["data"]["cache_dir"]) / split)
                    for split, rs in records.items()}
        for split, ds in datasets.items():
            write_json(run / "splits" / f"{split}.json", records[split])
            for i in range(len(ds)):
                ds.materialize(i)
                if (i + 1) % 25 == 0 or i + 1 == len(ds):
                    print(f"CACHE {split} {i+1}/{len(ds)}", flush=True)
        stats_path = run / "normalization_stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            assert stats["spatial_resolution"] == config["data"]["resolution"]
        else:
            stats = fit_normalization(datasets["train"])
            write_json(stats_path, stats)
        write_json(run / "config.json", config)
        write_json(run / "data_setup.json", {"split_counts": {k: len(v) for k, v in records.items()},
                   "time_indices": list(range(20)), "resolution": config["data"]["resolution"],
                   "field_order": list(FIELD_NAMES), "normalization_fitted_on": "train only",
                   "original_files_unchanged": True, "resize": "area averaging when resolution <512",
                   "time_conditioning": "actual elapsed time; transition dt~4 versus EIT/CAR/AH dt~1",
                   "constants": "Re, Wi, beta, epsilon, Lmax are identical across trajectories; omitted from inputs"})
        if args.prepare_only:
            return
        device = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(config["training"]["gpu_memory_fraction"])
        model, metadata = build_model(config)
        write_json(run / "model_metadata.json", metadata)
        model.to(device)
        optimizer = optimizer_for(model, config)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=.5, patience=config["training"]["scheduler_patience"],
            threshold=1e-4, min_lr=[1e-7, 1e-6, 1e-6])
        if args.smoke_only:
            model.train()
            results = []
            before = {name: p.detach().clone() for name, p in model.core.named_parameters() if p.numel() < 1000}
            for step in range(3):
                batch = next(iter(DataLoader(datasets["train"], batch_size=1)))
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                x, target, *_ = prepare_batch(batch, stats, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                loss = (prediction - target).square().mean()
                assert torch.isfinite(loss)
                loss.backward()
                missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
                core_gradients = sum(p.grad is not None for p in model.core.parameters())
                assert core_gradients > 0
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                assert torch.isfinite(norm)
                optimizer.step()
                torch.cuda.synchronize()
                record = {"step": step + 1, "seconds": time.perf_counter() - start,
                          "loss": loss.item(), "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
                          "core_parameters_with_gradient": core_gradients,
                          "parameters_without_gradient": missing}
                print(json.dumps(record), flush=True)
                results.append(record)
            changed = [n for n, p in model.core.named_parameters() if n in before and not torch.equal(before[n], p)]
            assert changed, "Core did not update"
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            model.eval()
            with torch.no_grad():
                prediction = model(x)
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - started
            write_json(run / "smoke_test.json", {"status": "passed", "steps": results,
                       "all_parameters_unfrozen": all(p.requires_grad for p in model.parameters()),
                       "changed_core_parameter_examples": changed[:20], "evaluation_seconds": forward_seconds,
                       "estimated_epoch_seconds": np.mean([r["seconds"] for r in results[1:]]) * len(datasets["train"]) + forward_seconds * len(datasets["valid"]),
                       "smoke_weights_discarded": True})
            return
        epoch_start = 1
        best = {"mse": float("inf"), "vrmse": float("inf"), "mse_epoch": 0, "vrmse_epoch": 0}
        if args.resume:
            payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            assert payload["normalization"] == stats
            load_states(model, payload)
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            epoch_start = payload["epoch"] + 1
            best = payload["best"]
            del payload
        elif (run / "history.jsonl").exists():
            raise RuntimeError("Existing training history: use --resume or a new run directory")
        print("TRAIN_START " + json.dumps(metadata), flush=True)
        write_json(run / "process.json", {"pid": os.getpid(), "started_epoch": epoch_start})
        if epoch_start == 1:
            initial = evaluate(model, datasets["valid"], stats, device, run / "validation" / "epoch_0000")
            print("INITIAL_VALID " + json.dumps(initial), flush=True)
        for epoch in range(epoch_start, config["training"]["epochs"] + 1):
            model.train()
            assert all(p.requires_grad for p in model.parameters()) and model.core.training
            generator = torch.Generator().manual_seed(config["seed"] + epoch)
            loader = DataLoader(datasets["train"], batch_size=1, shuffle=True,
                                generator=generator, num_workers=0)
            start = time.perf_counter()
            total = 0.
            for step, batch in enumerate(loader, 1):
                x, target, *_ = prepare_batch(batch, stats, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                loss = (prediction - target).square().mean()
                assert torch.isfinite(loss), (epoch, step)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                assert torch.isfinite(norm), (epoch, step)
                optimizer.step()
                total += loss.item()
                if step == 1 or step % 50 == 0 or step == len(loader):
                    print(json.dumps({"epoch": epoch, "step": step, "steps": len(loader),
                                      "train_mse_so_far": total / step}), flush=True)
            train_seconds = time.perf_counter() - start
            optimizer.zero_grad(set_to_none=True)
            validation_start = time.perf_counter()
            valid = evaluate(model, datasets["valid"], stats, device,
                             run / "validation" / f"epoch_{epoch:04d}")
            validation_seconds = time.perf_counter() - validation_start
            scheduler.step(valid["normalized_mse_all20"])
            row = {"epoch": epoch, "train_normalized_mse": total / len(loader),
                   "valid_normalized_mse": valid["normalized_mse_all20"],
                   "valid_vrmse_future19": valid["vrmse_future19"],
                   "persistence_vrmse_future19": valid["persistence_vrmse_future19"],
                   "core_lr": optimizer.param_groups[0]["lr"], "adapter_lr": optimizer.param_groups[1]["lr"],
                   "train_seconds": train_seconds, "validation_seconds": validation_seconds}
            print("EPOCH " + json.dumps(row), flush=True)
            with (run / "history.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            csv_path = run / "metrics.csv"
            first = not csv_path.exists()
            with csv_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if first:
                    writer.writeheader()
                writer.writerow(row)
            mse_best = valid["normalized_mse_all20"] < best["mse"]
            vrmse_best = valid["vrmse_future19"] < best["vrmse"]
            if mse_best:
                best.update(mse=valid["normalized_mse_all20"], mse_epoch=epoch)
            if vrmse_best:
                best.update(vrmse=valid["vrmse_future19"], vrmse_epoch=epoch)
            if mse_best:
                save_checkpoint(run / "best.pt", model, None, scheduler, epoch, config, stats, metadata, best)
            if vrmse_best:
                save_checkpoint(run / "best_vrmse.pt", model, None, scheduler, epoch, config, stats, metadata, best)
            save_checkpoint(run / "latest.pt", model, optimizer, scheduler, epoch, config, stats, metadata, best)
            plot_history(run)
            write_json(run / "status.json", {"status": "training", "epoch": epoch, "epochs": config["training"]["epochs"],
                       "best": best, "last_metrics": row,
                       "estimated_remaining_hours": (train_seconds + validation_seconds) * (config["training"]["epochs"] - epoch) / 3600})
        # Test is held out for the whole training; checkpoints are selected only on validation.
        del optimizer, scheduler
        torch.cuda.empty_cache()
        for name in ("best", "best_vrmse"):
            payload = torch.load(run / (name + ".pt"), map_location="cpu", weights_only=False)
            load_states(model, payload)
            summary = evaluate(model, datasets["test"], stats, device, run / "final_test" / name)
            summary["checkpoint_epoch"] = payload["epoch"]
            write_json(run / "final_test" / name / "summary.json", summary)
            del payload
        write_json(run / "completed.json", {"epochs": config["training"]["epochs"], "best": best})
    except BaseException as exc:
        write_json(run / ("smoke_failed.json" if args.smoke_only else "failed.json"),
                   {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
