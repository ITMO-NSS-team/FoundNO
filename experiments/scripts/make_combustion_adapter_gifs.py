import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parent))

import finetune_combustion_adapters as train_mod


def stats_from_metadata(metadata, prefix):
    mean = torch.tensor(metadata["normalization"][f"{prefix}_mean"], dtype=torch.float32).view(-1, 1, 1, 1)
    std = torch.tensor(metadata["normalization"][f"{prefix}_std"], dtype=torch.float32).view(-1, 1, 1, 1)
    return mean, std


def build_split_dataset(metadata, hdf5_root, split_kind, n_samples, seed):
    x_stats = stats_from_metadata(metadata, "x")
    y_stats = stats_from_metadata(metadata, "y")

    if split_kind == "train_numerical":
        source = metadata["source"]
        split = "train"
        sub_s = metadata["sub_s_real"] if source == "real" else metadata["sub_s_numerical"]
        mask_prob = metadata.get("mask_prob", 0.0)
        max_samples = metadata.get("train_samples", n_samples)
        ds_seed = seed
    elif split_kind == "val_real":
        source = metadata["val_source"]
        split = "val"
        sub_s = metadata["sub_s_real"] if source == "real" else metadata["sub_s_numerical"]
        mask_prob = 1.0 if source == "real" else metadata.get("mask_prob", 0.0)
        max_samples = metadata.get("val_samples", n_samples)
        ds_seed = seed + 1
    elif split_kind == "test_real":
        source = "real"
        split = "test"
        sub_s = metadata["sub_s_real"]
        mask_prob = 1.0
        max_samples = n_samples
        ds_seed = seed + 2
    else:
        raise ValueError(f"Unsupported split_kind: {split_kind}")

    return train_mod.CombustionRealPDEHDF5Windows(
        hdf5_root,
        source,
        split,
        in_step=metadata["in_step"],
        out_step=metadata["out_step"],
        n_autoregressive=metadata["n_autoregressive"],
        sub_s=sub_s,
        mask_prob=mask_prob,
        max_samples=max_samples,
        x_stats=x_stats,
        y_stats=y_stats,
        seed=ds_seed,
        verbose=False,
        hdf5_rdcc_nbytes=int(metadata.get("hdf5_rdcc_nbytes") or 1048576),
        hdf5_read_retries=int(metadata.get("hdf5_read_retries") or 3),
    )


def denorm_y(tensor, metadata, device):
    mean, std = stats_from_metadata(metadata, "y")
    return tensor * std.to(device) + mean.to(device)


def render_frame(true_rows, pred_rows, err_rows, labels, frame_idx, value_range, error_range):
    n_rows = len(true_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11.5, 3.15 * n_rows), dpi=120)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    vmin, vmax = value_range
    emin, emax = error_range

    for row_idx in range(n_rows):
        panels = [
            (true_rows[row_idx][frame_idx], "True", "viridis", vmin, vmax),
            (pred_rows[row_idx][frame_idx], "Predict", "viridis", vmin, vmax),
            (err_rows[row_idx][frame_idx], "Absolute error", "magma", emin, emax),
        ]
        for col_idx, (arr, title, cmap, cur_min, cur_max) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(arr, cmap=cmap, vmin=cur_min, vmax=cur_max, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=13, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(labels[row_idx], fontsize=9)
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"Target window frame {frame_idx + 1}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(image)


def save_gif(path, frames, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000 / max(1e-6, fps)))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def collect_predictions(model, dataset, metadata, device, n_samples):
    true_rows = []
    pred_rows = []
    err_rows = []
    labels = []
    random.seed(0)
    with torch.no_grad():
        for sample_idx in range(n_samples):
            sample = dataset[sample_idx]
            x = sample["x"].unsqueeze(0).to(device)
            y = sample["y"].unsqueeze(0).to(device)
            pred = model(x)

            y_denorm = denorm_y(y, metadata, device)[0, 0].detach().cpu().numpy()
            pred_denorm = denorm_y(pred, metadata, device)[0, 0].detach().cpu().numpy()
            abs_error = np.abs(pred_denorm - y_denorm)

            true_rows.append(y_denorm)
            pred_rows.append(pred_denorm)
            err_rows.append(abs_error)

            sim_id = sample.get("sim_id", "")
            time_id = int(sample.get("time_id", 0))
            start_t = time_id + int(metadata["in_step"])
            end_t = start_t + y_denorm.shape[0] - 1
            labels.append(f"{sample_idx + 1}: {sim_id}\\nt={start_t}..{end_t}")

    return true_rows, pred_rows, err_rows, labels


def make_split_gif(model, dataset, metadata, device, out_path, n_samples, fps):
    true_rows, pred_rows, err_rows, labels = collect_predictions(model, dataset, metadata, device, n_samples)
    true_pred_values = np.concatenate([np.ravel(x) for x in true_rows + pred_rows])
    error_values = np.concatenate([np.ravel(x) for x in err_rows])

    value_range = (
        float(np.percentile(true_pred_values, 1)),
        float(np.percentile(true_pred_values, 99)),
    )
    error_range = (
        0.0,
        float(max(np.percentile(error_values, 99), 1e-12)),
    )

    n_frames = min(row.shape[0] for row in true_rows)
    frames = [
        render_frame(true_rows, pred_rows, err_rows, labels, frame_idx, value_range, error_range)
        for frame_idx in range(n_frames)
    ]
    save_gif(out_path, frames, fps)
    return {
        "path": str(out_path),
        "frames": n_frames,
        "samples": n_samples,
        "value_range": value_range,
        "absolute_error_range": error_range,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Make True/Predict/Absolute-error GIFs for combustion adapters.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hdf5-root", default="")
    parser.add_argument("--main-ckpt", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = train_mod.torch_load(checkpoint_path, map_location="cpu")
    metadata = checkpoint["metadata"]
    hdf5_root = args.hdf5_root or metadata["hdf5_root"]
    main_ckpt = args.main_ckpt or metadata["main_ckpt"]
    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parent / "gifs"
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = train_mod.build_model(
        main_ckpt,
        input_channels=int(metadata["input_channels"]),
        output_channels=int(metadata["output_channels"]),
        device=device,
    )
    model.input_adapter.load_state_dict(checkpoint["input_adapter_state_dict"])
    model.output_adapter.load_state_dict(checkpoint["output_adapter_state_dict"])
    model.eval()

    results = {}
    for split_kind, file_name in [
        ("train_numerical", "train_numerical_true_predict_absolute_error.gif"),
        ("test_real", "test_real_true_predict_absolute_error.gif"),
    ]:
        dataset = build_split_dataset(metadata, hdf5_root, split_kind, args.n_samples, args.seed)
        result = make_split_gif(
            model=model,
            dataset=dataset,
            metadata=metadata,
            device=device,
            out_path=out_dir / file_name,
            n_samples=args.n_samples,
            fps=args.fps,
        )
        results[split_kind] = result
        print(json.dumps({split_kind: result}, indent=2), flush=True)

    (out_dir / "gif_metadata.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved GIFs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
