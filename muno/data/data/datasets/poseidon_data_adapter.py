import xarray as xr
import torch


class PoseidonDataset:
    def __init__(
        self,
        file_path,
        split,
        solution_name,
        n_train,
        n_val,
        n_tests,
        n_samples,
    ):
        self.ds = xr.open_dataset(file_path)
        self.solution = self.ds[solution_name]

        if n_train + n_val + n_tests[0] != int(self.solution.shape[0]):
            n_train *= 0.75
            n_val *= 0.1
            n_tests[0] *= 0.15

        if split == "train":
            start = 0
            end = int(n_train)
        elif split == "val":
            start = n_train
            end = int(n_train + n_val)
        elif split == "test":
            start = n_train + n_val
            end = int(n_train + n_val + n_tests[0])
        else:
            raise ValueError(f"Unknown split {split}")

        assert n_samples <= (end - start)

        self.indices = slice(start, start + n_samples)

        self.X_res = self.solution.shape[-2]
        self.Y_res = self.solution.shape[-1]
        self.x_coordinate = torch.linspace(0, 1, self.X_res)
        self.y_coordinate = torch.linspace(0, 1, self.Y_res)

    def load(self):
        u = torch.from_numpy(
            self.solution.isel(sample=self.indices).values
        ).float()

        return {
            "x": u[:600, 0:1, ...],
            "y": u[:600, ...],
            "x-coordinate": self.x_coordinate,
            "y-coordinate": self.y_coordinate,
        }
