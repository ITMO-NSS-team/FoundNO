import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))

import finetune_combustion_adapters as train_mod


def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="none")


def kinetic_energy(x):
    u_prime = ((x[..., 0] - x[..., 0].mean(dim=1, keepdim=True)) ** 2).mean(1)
    v_prime = ((x[..., 1] - x[..., 1].mean(dim=1, keepdim=True)) ** 2).mean(1)
    return 0.5 * (u_prime + v_prime)


def eval_metrics(pred, target, channels):
    pred_all = pred[..., :channels]
    target_all = target[..., :channels]
    b, t, h, w, c = target_all.size()
    device = target.device

    se = mse_loss(pred_all, target_all)
    rmse = torch.sqrt(torch.mean(se))
    mae = torch.mean(torch.abs(pred_all - target_all))

    err_l2 = torch.norm(pred_all.reshape(b, -1) - target_all.reshape(b, -1), dim=1)
    norm = torch.norm(target_all.reshape(b, -1), dim=1)
    rel_l2_error = torch.mean(err_l2 / torch.clamp(norm, min=1e-12))

    r2_denom = torch.sum((target_all - target_all.mean(0, keepdim=True)) ** 2)
    r2 = 1 - torch.sum((pred_all - target_all) ** 2) / torch.clamp(r2_denom, min=1e-12)

    if c < 2:
        ke_error = torch.tensor(0.0, device=device)
    else:
        ke_error = (kinetic_energy(pred_all) - kinetic_energy(target_all)).abs().mean()

    pred_f = torch.fft.fftn(pred_all, dim=[1, 2, 3])
    target_f = torch.fft.fftn(target_all, dim=[1, 2, 3])
    err_f_raw = torch.abs(pred_f - target_f) ** 2
    norm_f_raw = torch.abs(target_f) ** 2

    radius_max = min(t // 2, h // 2, w // 2)
    err_f = torch.zeros([b, radius_max, c], device=device)
    norm_f = torch.zeros([b, radius_max, c], device=device)
    for i in range(t // 2):
        for j in range(h // 2):
            for k in range(w // 2):
                radius = math.floor(math.sqrt(i**2 + j**2 + k**2))
                if radius >= radius_max:
                    continue
                err_f[:, radius] += err_f_raw[:, i, j, k]
                norm_f[:, radius] += norm_f_raw[:, i, j, k]

    err_f = torch.sqrt(torch.mean(err_f, axis=0)) / (t * h * w)
    norm_f = torch.sqrt(torch.mean(norm_f, axis=0)) / (t * h * w)

    i_low = int(round(radius_max / 3))
    i_high = int(round(radius_max * 2 / 3))
    low_f_error = err_f[:i_low].mean()
    mid_f_error = err_f[i_low:i_high].mean()
    high_f_error = err_f[i_high:].mean()
    f_error = err_f.mean()

    rel_f = err_f / torch.clamp(norm_f, min=1e-12)
    rel_low_f_error = rel_f[:i_low].mean()
    rel_mid_f_error = rel_f[i_low:i_high].mean()
    rel_high_f_error = rel_f[i_high:].mean()

    sum_pred = torch.sum(pred_all, dim=[2, 3, 4])
    sum_target = torch.sum(target_all, dim=[2, 3, 4])
    sum_pred_f = torch.fft.fftn(sum_pred, dim=1)
    sum_target_f = torch.fft.fftn(sum_target, dim=1)
    freq_error = torch.mean(torch.abs(sum_pred_f - sum_target_f))

    return {
        "rmse": rmse,
        "mae": mae,
        "rel_l2_error": rel_l2_error,
        "r2": r2,
        "ke_error": ke_error,
        "f_error": f_error,
        "low_f_error": low_f_error,
        "mid_f_error": mid_f_error,
        "high_f_error": high_f_error,
        "rel_low_f_error": rel_low_f_error,
        "rel_mid_f_error": rel_mid_f_error,
        "rel_high_f_error": rel_high_f_error,
        "freq_error": freq_error,
    }


def stats_from_metadata(metadata, key_prefix):
    mean = torch.tensor(metadata["normalization"][f"{key_prefix}_mean"], dtype=torch.float32).view(-1, 1, 1, 1)
    std = torch.tensor(metadata["normalization"][f"{key_prefix}_std"], dtype=torch.float32).view(-1, 1, 1, 1)
    return mean, std


def build_eval_dataset(metadata, hdf5_root_override, split, max_samples_override):
    hdf5_root = hdf5_root_override or metadata.get("hdf5_root", "")
    cache_dir = metadata.get("cache_dir", "")
    if max_samples_override is not None:
        max_samples = max_samples_override
    elif split == "val":
        max_samples = int(metadata.get("val_samples", 0) or 0)
    else:
        max_samples = None

    x_stats = stats_from_metadata(metadata, "x")
    y_stats = stats_from_metadata(metadata, "y")

    if hdf5_root:
        val_source = metadata["val_source"]
        sub_s = metadata["sub_s_real"] if val_source == "real" else metadata["sub_s_numerical"]
        return train_mod.CombustionRealPDEHDF5Windows(
            hdf5_root,
            val_source,
            split,
            in_step=metadata["in_step"],
            out_step=metadata["out_step"],
            n_autoregressive=metadata["n_autoregressive"],
            sub_s=sub_s,
            mask_prob=1.0 if val_source == "real" else metadata.get("mask_prob", 0.0),
            max_samples=max_samples,
            x_stats=x_stats,
            y_stats=y_stats,
            seed=1,
            verbose=False,
            hdf5_rdcc_nbytes=int(metadata.get("hdf5_rdcc_nbytes") or 1048576),
            hdf5_read_retries=int(metadata.get("hdf5_read_retries") or 3),
        )

    if cache_dir:
        if split != "val":
            raise ValueError("Memmap cache evaluation currently supports only split='val'.")
        return train_mod.MemmapWindowDataset(cache_dir, "val", x_stats=x_stats, y_stats=y_stats)

    raise ValueError("Checkpoint metadata does not contain hdf5_root or cache_dir.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned combustion adapters with RealPDEBench metrics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hdf5-root", default="", help="Override HDF5 root from checkpoint metadata.")
    parser.add_argument("--main-ckpt", default="", help="Override frozen core checkpoint from metadata.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Evaluation split.")
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional generic cap for --split val/test.")
    parser.add_argument("--eval-channels", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    checkpoint = train_mod.torch_load(checkpoint_path, map_location="cpu")
    metadata = checkpoint["metadata"]
    main_ckpt = args.main_ckpt or metadata["main_ckpt"]
    max_samples = args.max_samples if args.max_samples > 0 else None
    if max_samples is None and args.max_val_samples > 0:
        max_samples = args.max_val_samples

    val_ds = build_eval_dataset(metadata, args.hdf5_root, args.split, max_samples)
    if hasattr(val_ds, "entries"):
        val_ds.entries = sorted(
            val_ds.entries,
            key=lambda e: (str(e.get("sim_id", "")), int(e.get("time_id", 0))),
        )
    input_channels = int(metadata["input_channels"])
    output_channels = int(metadata["output_channels"])
    model = train_mod.build_model(main_ckpt, input_channels, output_channels, device)
    model.input_adapter.load_state_dict(checkpoint["input_adapter_state_dict"])
    model.output_adapter.load_state_dict(checkpoint["output_adapter_state_dict"])
    model.eval()

    y_mean, y_std = stats_from_metadata(metadata, "y")
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)

    eval_channels = args.eval_channels or int(metadata.get("val_loss_channels") or 0)
    if eval_channels <= 0:
        eval_channels = output_channels

    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"Evaluating {checkpoint_path} on {device}; split={args.split}, samples={len(val_ds)}, "
        f"batch_size={args.batch_size}, eval_channels={eval_channels}",
        flush=True,
    )

    normalized_se_sum = 0.0
    normalized_count = 0
    se_sum = 0.0
    mae_sum = 0.0
    elem_count = 0
    sample_count = 0
    rel_l2_sum = 0.0
    target_sum = None
    target_sq_sum = None
    err_f_sum = None
    norm_f_sum = None
    radius_index = None
    radius_valid = None
    freq_error_sum = 0.0
    freq_error_count = 0
    radius_max = None
    start = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch_start = time.time()
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            pred = model(x)

            normalized_se = mse_loss(pred[:, :eval_channels], y[:, :eval_channels])
            normalized_se_sum += normalized_se.sum().item()
            normalized_count += normalized_se.numel()

            pred_denorm = pred * y_std + y_mean
            y_denorm = y * y_std + y_mean
            pred_metric = pred_denorm[:, :eval_channels].permute(0, 2, 3, 4, 1)
            target_metric = y_denorm[:, :eval_channels].permute(0, 2, 3, 4, 1)

            diff = pred_metric - target_metric
            se = diff.square()
            se_sum += se.sum().item()
            mae_sum += diff.abs().sum().item()
            elem_count += se.numel()
            batch_size = pred_metric.shape[0]
            sample_count += batch_size

            rel_l2_sum += (
                torch.norm(diff.reshape(batch_size, -1), dim=1)
                / torch.clamp(torch.norm(target_metric.reshape(batch_size, -1), dim=1), min=1e-12)
            ).sum().item()

            cur_target_sum = target_metric.sum(dim=0)
            cur_target_sq_sum = target_metric.square().sum(dim=0)
            target_sum = cur_target_sum if target_sum is None else target_sum + cur_target_sum
            target_sq_sum = cur_target_sq_sum if target_sq_sum is None else target_sq_sum + cur_target_sq_sum

            b, t, h, w, c = target_metric.size()
            if radius_max is None:
                radius_max = min(t // 2, h // 2, w // 2)
                err_f_sum = torch.zeros([radius_max, c], device=device)
                norm_f_sum = torch.zeros([radius_max, c], device=device)
                ii, jj, kk = torch.meshgrid(
                    torch.arange(t // 2, device=device),
                    torch.arange(h // 2, device=device),
                    torch.arange(w // 2, device=device),
                    indexing="ij",
                )
                radius_index = torch.floor(torch.sqrt(ii.square() + jj.square() + kk.square())).long().reshape(-1)
                radius_valid = radius_index < radius_max
                radius_index = radius_index[radius_valid]

            pred_f = torch.fft.fftn(pred_metric, dim=[1, 2, 3])
            target_f = torch.fft.fftn(target_metric, dim=[1, 2, 3])
            err_f_raw = torch.abs(pred_f - target_f) ** 2
            norm_f_raw = torch.abs(target_f) ** 2
            err_flat = err_f_raw[:, : t // 2, : h // 2, : w // 2].reshape(b, -1, c).sum(dim=0)[radius_valid]
            norm_flat = norm_f_raw[:, : t // 2, : h // 2, : w // 2].reshape(b, -1, c).sum(dim=0)[radius_valid]
            scatter_index = radius_index[:, None].expand(-1, c)
            err_f_sum.scatter_add_(0, scatter_index, err_flat)
            norm_f_sum.scatter_add_(0, scatter_index, norm_flat)

            sum_pred = torch.sum(pred_metric, dim=[2, 3, 4])
            sum_target = torch.sum(target_metric, dim=[2, 3, 4])
            sum_pred_f = torch.fft.fftn(sum_pred, dim=1)
            sum_target_f = torch.fft.fftn(sum_target, dim=1)
            freq_error = torch.abs(sum_pred_f - sum_target_f)
            freq_error_sum += freq_error.sum().item()
            freq_error_count += freq_error.numel()

            elapsed = time.time() - start
            per_batch = elapsed / max(1, batch_idx)
            eta = per_batch * max(0, len(loader) - batch_idx)
            print(
                f"[eval] {batch_idx}/{len(loader)} batch={train_mod.format_seconds(time.time() - batch_start)} "
                f"elapsed={train_mod.format_seconds(elapsed)} eta={train_mod.format_seconds(eta)}",
                flush=True,
            )

    err_f = torch.sqrt(err_f_sum / max(1, sample_count)) / (t * h * w)
    norm_f = torch.sqrt(norm_f_sum / max(1, sample_count)) / (t * h * w)
    i_low = int(round(radius_max / 3))
    i_high = int(round(radius_max * 2 / 3))
    rel_f = err_f / torch.clamp(norm_f, min=1e-12)
    r2_denom = (target_sq_sum - target_sum.square() / max(1, sample_count)).sum().item()
    metrics = {
        "rmse": math.sqrt(se_sum / max(1, elem_count)),
        "mae": mae_sum / max(1, elem_count),
        "rel_l2_error": rel_l2_sum / max(1, sample_count),
        "r2": 1.0 - se_sum / max(r2_denom, 1e-12),
        "ke_error": 0.0,
        "f_error": float(err_f.mean().item()),
        "low_f_error": float(err_f[:i_low].mean().item()),
        "mid_f_error": float(err_f[i_low:i_high].mean().item()),
        "high_f_error": float(err_f[i_high:].mean().item()),
        "rel_low_f_error": float(rel_f[:i_low].mean().item()),
        "rel_mid_f_error": float(rel_f[i_low:i_high].mean().item()),
        "rel_high_f_error": float(rel_f[i_high:].mean().item()),
        "freq_error": freq_error_sum / max(1, freq_error_count),
    }
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_val_loss": float(checkpoint.get("val_loss", float("nan"))),
        "split": args.split,
        "samples": int(sample_count),
        "eval_channels": int(eval_channels),
        "normalized_mse": float(normalized_se_sum / max(1, normalized_count)),
    }
    result.update({key: float(value) for key, value in metrics.items()})

    print(json.dumps(result, indent=2), flush=True)

    out_json = Path(args.out_json) if args.out_json else checkpoint_path.with_name("best_adapters_eval_metrics.json")
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved metrics to {out_json}", flush=True)


if __name__ == "__main__":
    main()
