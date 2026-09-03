import argparse
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
from datasets import load_from_disk


def format_seconds(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(rem):02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m"


def h5_name(sim_id):
    name = str(sim_id)
    return name if name.endswith(".h5") else f"{name}.h5"


def decode_observed(row):
    shape = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
    return np.frombuffer(row["observed"], dtype=np.float32).reshape(shape)


def decode_numerical(row):
    shape = (
        int(row["shape_t"]),
        int(row["shape_h"]),
        int(row["shape_w"]),
        int(row["numerical_channels"]),
    )
    return np.frombuffer(row["numerical"], dtype=np.float32).reshape(shape)


def dataset_kwargs(args, shape):
    if len(shape) == 3:
        chunks = (
            min(args.time_chunk, shape[0]),
            min(args.spatial_chunk, shape[1]),
            min(args.spatial_chunk, shape[2]),
        )
    else:
        chunks = (
            min(args.time_chunk, shape[0]),
            min(args.spatial_chunk, shape[1]),
            min(args.spatial_chunk, shape[2]),
            shape[3],
        )

    kwargs = {"dtype": "float32", "chunks": chunks}
    if args.compression != "none":
        kwargs["compression"] = args.compression
        if args.compression == "gzip":
            kwargs["compression_opts"] = args.gzip_level
    return kwargs


def should_skip(path, dataset_name, expected_shape, overwrite):
    if overwrite or not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            if dataset_name in f and tuple(f[dataset_name].shape) == tuple(expected_shape):
                return True
    except OSError:
        return False
    return False


def write_h5(path, dataset_name, array, args, attrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(tmp_path, "w") as f:
        ds = f.create_dataset(
            dataset_name,
            data=array,
            **dataset_kwargs(args, array.shape),
        )
        for key, value in attrs.items():
            ds.attrs[key] = value
            f.attrs[key] = value
    tmp_path.replace(path)


def copy_metadata(data_root, out_root):
    copied = []
    channels_path = data_root / "channels.json"
    if channels_path.exists():
        shutil.copy2(channels_path, out_root / "channels.json")
        copied.append("channels.json")

    hf_root = data_root / "hf_dataset"
    index_out = out_root / "hf_dataset"
    index_out.mkdir(parents=True, exist_ok=True)
    for index_path in hf_root.glob("*_index_*.json"):
        shutil.copy2(index_path, index_out / index_path.name)
        copied.append(f"hf_dataset/{index_path.name}")
    return copied


def convert_real(data_root, out_root, args):
    ds = load_from_disk(str(data_root / "hf_dataset" / "real"))
    n = len(ds) if args.max_rows <= 0 else min(len(ds), args.max_rows)
    start = time.time()
    print(f"[real] start rows={n}", flush=True)

    for i in range(n):
        row_start = time.time()
        row = ds[i]
        sim_id = row["sim_id"]
        observed_shape = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
        out_path = out_root / "real" / h5_name(sim_id)
        if should_skip(out_path, "trajectory", observed_shape, args.overwrite):
            print(f"[real] {i + 1}/{n} {sim_id} skip existing", flush=True)
            continue

        observed = decode_observed(row)
        write_h5(
            out_path,
            "trajectory",
            observed,
            args,
            {"sim_id": str(sim_id), "source": "real", "dataset": "trajectory"},
        )
        elapsed = time.time() - start
        eta = elapsed / max(1, i + 1) * max(0, n - i - 1)
        print(
            f"[real] {i + 1}/{n} {sim_id} wrote {out_path.name} "
            f"shape={observed.shape} row={format_seconds(time.time() - row_start)} "
            f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
            flush=True,
        )


def convert_numerical(data_root, out_root, args):
    ds = load_from_disk(str(data_root / "hf_dataset" / "numerical"))
    n = len(ds) if args.max_rows <= 0 else min(len(ds), args.max_rows)
    start = time.time()
    print(f"[numerical] start rows={n}", flush=True)

    for i in range(n):
        row_start = time.time()
        row = ds[i]
        sim_id = row["sim_id"]
        observed_shape = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
        numerical_shape = (
            int(row["shape_t"]),
            int(row["shape_h"]),
            int(row["shape_w"]),
            int(row["numerical_channels"]),
        )

        surrogate_path = out_root / "surrogate" / h5_name(sim_id)
        numerical_path = out_root / "numerical" / h5_name(sim_id)
        surrogate_done = should_skip(surrogate_path, "measured_data", observed_shape, args.overwrite)
        numerical_done = should_skip(numerical_path, "measured_data", numerical_shape, args.overwrite)
        if surrogate_done and numerical_done:
            print(f"[numerical] {i + 1}/{n} {sim_id} skip existing", flush=True)
            continue

        if not surrogate_done:
            observed = decode_observed(row)
            write_h5(
                surrogate_path,
                "measured_data",
                observed,
                args,
                {"sim_id": str(sim_id), "source": "numerical_observed", "dataset": "measured_data"},
            )

        if not numerical_done:
            numerical_start = time.time()
            print(f"[numerical] {i + 1}/{n} {sim_id} decoding numerical blob...", flush=True)
            numerical = decode_numerical(row)
            print(
                f"[numerical] {i + 1}/{n} {sim_id} decoded in "
                f"{format_seconds(time.time() - numerical_start)}; writing h5...",
                flush=True,
            )
            write_h5(
                numerical_path,
                "measured_data",
                numerical,
                args,
                {"sim_id": str(sim_id), "source": "numerical", "dataset": "measured_data"},
            )

        elapsed = time.time() - start
        eta = elapsed / max(1, i + 1) * max(0, n - i - 1)
        print(
            f"[numerical] {i + 1}/{n} {sim_id} wrote "
            f"surrogate={not surrogate_done} numerical={not numerical_done} "
            f"row={format_seconds(time.time() - row_start)} elapsed={format_seconds(elapsed)} "
            f"eta={format_seconds(eta)}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert RealPDEBench combustion HF/Arrow trajectories to per-simulation HDF5 files."
    )
    parser.add_argument(
        "--data-root",
        default=r"C:\Users\Nikita\Documents\RealPDE\data\RealPDEBench\combustion",
        help="Path to RealPDEBench combustion directory with hf_dataset/.",
    )
    parser.add_argument(
        "--out-root",
        default=r"D:\Dataset_Combustion",
        help="Output directory. Writes real/, numerical/, surrogate/ inside it.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["real", "numerical"],
        choices=["real", "numerical"],
        help="Which HF sources to convert. numerical also writes surrogate/.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows per source for testing. 0 means all.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing valid HDF5 files.")
    parser.add_argument("--time-chunk", type=int, default=20)
    parser.add_argument("--spatial-chunk", type=int, default=64)
    parser.add_argument("--compression", default="none", choices=["none", "lzf", "gzip"])
    parser.add_argument("--gzip-level", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    copied = copy_metadata(data_root, out_root)
    metadata = {
        "format": "combustion_hdf5_per_trajectory_v1",
        "data_root": str(data_root),
        "out_root": str(out_root),
        "sources": args.sources,
        "time_chunk": args.time_chunk,
        "spatial_chunk": args.spatial_chunk,
        "compression": args.compression,
        "gzip_level": args.gzip_level if args.compression == "gzip" else None,
        "copied_metadata": copied,
    }
    (out_root / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if "real" in args.sources:
        convert_real(data_root, out_root, args)
    if "numerical" in args.sources:
        convert_numerical(data_root, out_root, args)

    print(f"Done. HDF5 dataset saved to {out_root}", flush=True)


if __name__ == "__main__":
    main()
