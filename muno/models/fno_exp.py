import torch
from torch.utils.data import Dataset, DataLoader
from neuralop.models import FNO
import torch.nn.functional as F
import time

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
            n_layers = 4
        )

    def forward(self, x):
        # Input x: [B, 3, T, X] (u0, x, t)
        return super().forward(x)


def train_fno_pdebench(train_data, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = TemporalFNO(modes=64, width=32).to(device)
    # model.load_state_dict(torch.load('best_fno_pdebench_downsample_64chan_32modes_4layers_batchsize_16_based_400_epochs.pt', weights_only=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)


    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_absperc = torch.tensor(0.0, device=device)
        n_samples = 0
        timee = time.time()
        for batch in train_loader:
            x = batch['x'].cuda()
            y = batch['y'].cuda()
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            with torch.no_grad():
                ape = torch.abs((y - pred)) / (y.max()-y.min())#.clamp(min=-1e-3, max = 1e-3)))  # [B, C, x, t]
                total_absperc += ape.sum()
                n_samples += ape.numel()

            optimizer.step()

            train_loss += loss.item()
        mape_epoch = (total_absperc / n_samples) * 100
        avg_train = train_loss / len(train_loader)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train:.3e}, Validation metric: {mape_epoch:.3f}% | Time: ' {time.time()-timee} s.' | LR: {lr:.2e}")
        torch.save(model.state_dict(), "best_fno_pdebench_downsample_64chan_32modes_4layers_batchsize_16_based_400_epochs_colab.pt")

        scheduler.step()
    return model


# Example usage
if __name__ == "__main__":
    import h5py

    with h5py.File(r"/content/drive/MyDrive/PDE_Bench_experiments/ReacDiff_Nu0.5_Rho10.0.hdf5", "r") as f:
        data = torch.tensor(f['tensor'][:]).float()  
        x_grid = torch.tensor(f['x-coordinate'][:]).float() 
        print(x_grid.shape)
        t_grid = torch.tensor(f['t-coordinate'][:][:-1]).float()
        print(t_grid.shape)# [201]


    train_size = int(1.0 * len(data))
    train_dataset = PDEBench1DDataset(data[:train_size], x_grid, t_grid)

    model = train_fno_pdebench(train_dataset)