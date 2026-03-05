import h5py
import torch
from torch.utils.data import Dataset


class H5pyDataset(Dataset):
    """PDE h5py dataset"""

    def __init__(
        self,
        data_path,
        resolution=128,
        transform_x=None,
        transform_y=None,
        n_samples=None,
    ):
        resolution_to_step = {128: 8, 256: 4, 512: 2, 1024: 1}
        try:
            subsample_step = resolution_to_step[resolution]
        except KeyError:
            raise ValueError(
                f"Got resolution={resolution}, "
                f"expected one of {resolution_to_step.keys()}"
            )

        self.subsample_step = subsample_step
        self.data_path = data_path
        self._data = None
        self.transform_x = transform_x
        self.transform_y = transform_y

        if n_samples is not None:
            self.n_samples = n_samples
        else:
            with h5py.File(str(self.data_path), "r") as f:
                self.n_samples = f["x"].shape[0]

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(str(self.data_path), "r")
        return self._data

    def _attribute(self, variable, name):
        return self.data[variable].attrs[name]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            assert (
                idx < self.n_samples
            ), f"Trying to access sample {idx} of dataset with {self.n_samples} samples"
        else:
            for i in idx:
                assert (
                    i < self.n_samples
                ), f"Trying to access sample {i} " \
                   f"of dataset with {self.n_samples} samples"

        x = self.data["x"][idx, :: self.subsample_step, :: self.subsample_step]
        y = self.data["y"][idx, :: self.subsample_step, :: self.subsample_step]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.transform_x:
            x = self.transform_x(x)

        if self.transform_y:
            y = self.transform_y(y)

        return {"x": x, "y": y}


class HDF5Adapter:
    def __init__(self, file_path, spec):
        self.file = h5py.File(file_path, "r")
        self.spec = spec
        self.run_keys = list(self.file.keys())
        self.hdf5_spec = {
            "inputs": {
                "path": "inputs",
                "variables": [
                    "initial condition",
                    # ["b", "d", "epsilon_1", ...] для Gray-Scott
                ]
            },
            "outputs": {
                "path": "outputs",
                "variables": ["U"]
            }
        }

    def load(self, n_samples):
        xs, ys = [], []

        for k in self.run_keys[:n_samples]:
            run = self.file[k]

            x = self._read_group(run, self.spec["inputs"])
            y = self._read_group(run, self.spec["outputs"])

            xs.append(x)
            ys.append(y)

        return {
            "x": torch.stack(xs, dim=0),
            "y": torch.stack(ys, dim=0)
        }

    def _read_group(self, run, group_spec):
        data = []
        group = run[group_spec["path"]]
        for var in group_spec["variables"]:
            data.append(torch.tensor(group[var][:], dtype=torch.float32))
        return torch.stack(data, dim=0)
