import torch
from torch.utils.data import Dataset, DataLoader
from neuraloperator.neuralop.models import FNO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
import os


class PDEBench1DDataset(Dataset):
    def __init__(self, tensor_data, x_grid, t_grid):
        """
        Parameters:
            tensor_data: [N, T, X] - full spatiotemporal solution
            x_grid: [X] - spatial grid
            t_grid: [T] - temporal grid
        """
        self.tensor = tensor_data
        self.x = x_grid
        self.t = t_grid

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        sol = self.tensor[idx]  # shape [T, X]
        u0 = sol[0:1]  # [1, X] initial condition

        # Create input with spatial and time coords (T, X, 2)
        x_coords = self.x[None, :].repeat(len(self.t), 1)  # [T, X]
        t_coords = self.t[:, None].repeat(1, len(self.x))  # [T, X]
        coords = torch.stack([x_coords, t_coords], dim=-1)  # [T, X, 2]

        inp = torch.cat([u0.repeat(len(self.t), 1).unsqueeze(-1), coords], dim=-1)  # [T, X, 3]
        out = sol.unsqueeze(-1)  # [T, X, 1]

        return {
            'x': inp.permute(2, 0, 1),  # [3, T, X]
            'y': out.permute(2, 0, 1)   # [1, T, X]
        }


class TemporalFNO(FNO):
    def __init__(self, modes=12, width=32):
        super().__init__(
            n_modes=(modes, modes),
            hidden_channels=width,
            in_channels=3,  # u0, x, t
            out_channels=1,  # u(x,t)
            padding=8,
            use_mlp=True,
            mlp_dropout=0.1,
            n_layers=4
        )

    def forward(self, x):
        # Input x: [B, 3, T, X] (u0, x, t)
        return super().forward(x)
# Основной скрипт
if __name__ == "__main__":
    MODEL_PATH = "best_fno_pdebench_4_based_finetune.pt"
    DATA_PATH = r"C:\\Users\\EVM\\Downloads\\pde_bench_operators\\ReacDiff_Nu0.5_Rho10.0.hdf5"
    SAMPLE_IDX = 10000-1
    OUTPUT_FILE = "fno_comparison_pcolor.png"
    CMAP = "viridis"  # Colormap: viridis, plasma, inferno, magma, cividis
    ASPECT = "auto"  # Axes ration: 'auto', 'equal'
    # ==================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Загрузка данных
    print("Loading data...")
    with h5py.File(DATA_PATH, "r") as f:
        data = torch.tensor(f['tensor'][:]).float()
        x_grid = torch.tensor(f['x-coordinate'][:]).float()
        t_grid = torch.tensor(f['t-coordinate'][:][:-1]).float()

    dataset = PDEBench1DDataset(data, x_grid, t_grid)
    sample = dataset[SAMPLE_IDX]
    print(sample['y'].shape)
    print("Loading model...")
    model = TemporalFNO(modes=12, width=32).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    model.eval()

    print("Predicting...")
    with torch.no_grad():
        x_input = sample['x'].unsqueeze(0).to(device)
        prediction = model(x_input).squeeze().cpu().numpy()

    print("Visualizing...")
    original = sample['y'].squeeze().numpy()
    x = x_grid.numpy()
    t = t_grid.numpy()

    X, T = np.meshgrid(x, t)

    vmin = min(original.min(), prediction.min())
    vmax = max(original.max(), prediction.max())

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 0.05], width_ratios=[1, 1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(X, T, original, shading='auto', cmap=CMAP, vmin=vmin, vmax=vmax)
    ax1.set_title("Original solution")
    ax1.set_xlabel("Space (x)")
    ax1.set_ylabel("Time (t)")
    ax1.set_aspect(ASPECT)

    # 2. FNO pred.
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(X, T, prediction, shading='auto', cmap=CMAP, vmin=vmin, vmax=vmax)
    ax2.set_title("Prediction FNO")
    ax2.set_xlabel("Space (x)")
    ax2.set_ylabel("Time (t)")
    ax2.set_aspect(ASPECT)

    # 3. Difference between solutions
    ax3 = fig.add_subplot(gs[0, 2])
    diff = np.abs(original - prediction)
    im3 = ax3.pcolormesh(X, T, diff, shading='auto', cmap='hot')
    ax3.set_title("Difference, abs.")
    ax3.set_xlabel("Space (x)")
    ax3.set_ylabel("Time (t)")
    ax3.set_aspect(ASPECT)

    cbar_ax1 = fig.add_subplot(gs[1, 0])
    cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label("Values of u(x,t)")

    cbar_ax2 = fig.add_subplot(gs[1, 1])
    cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label("Values of u(x,t)")

    cbar_ax3 = fig.add_subplot(gs[1, 2])
    cbar3 = fig.colorbar(im3, cax=cbar_ax3, orientation='horizontal')
    cbar3.set_label("Absolute error")

    # Saving results
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Результат сохранен как: {OUTPUT_FILE}")
    plt.show()