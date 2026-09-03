import torch
from torch.utils.data import Dataset, DataLoader
from neuraloperator.neuralop.models import FNO
import torch.nn.functional as F
import time
from torch.cuda.amp import GradScaler, autocast

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


def train_fno_pdebench(train_data, epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = TemporalFNO(modes=24, width=36).to(device)
    # model.load_state_dict(torch.load('best_fno_pdebench_4_1.pt', weights_only=False))
    # model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
    # scaler = torch.amp.GradScaler()
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=16)

    # best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_absperc = torch.tensor(0.0, device=device)
        n_samples = 0
        timee = time.time()
        for batch in train_loader:
            x = batch['x'].cuda()
            y = batch['y'].cuda()
            # print(f'{x.shape} - original shape')
            # x_interp = F.interpolate(x, size=(201, 1024), mode="bilinear", align_corners=False)
            # print(f'{x_interp.shape} - interpolated shape')
            optimizer.zero_grad()
            # pred = model(x_interp)
            pred = model(x)
            # with torch.amp.autocast(device_type='cuda'):
            #     # 2. Forward pass on interpolated input
            # pred_interp = model(x_interp)  # assumes model output is [B, C, 256, 1024]
            #
            #     # 3. Interpolate prediction back to original size
            # pred = F.interpolate(pred_interp, size=(101, 1024), mode="bilinear", align_corners=False)
            #     print(pred)
            #     # 4. Compute loss on original resolution
            #     loss = F.mse_loss(pred, y)
            loss = F.mse_loss(pred, y)
            # print(f'Loss - {loss}')
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            with torch.no_grad():
                # y[torch.abs(y)<1e-3] = 1e-3*torch.sign(y[torch.abs(y)<1e-3])
                ape = torch.abs((y - pred)) / (y.max()-y.min())#.clamp(min=-1e-3, max = 1e-3)))  # [B, C, x, t]
                # print(ape.max())
                total_absperc += ape.sum()
                n_samples += ape.numel()

        # After loop:

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
        mape_epoch = (total_absperc / n_samples) * 100
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for batch in val_loader:
        #         x = batch['x'].cuda()
        #         y = batch['y'].cuda()
        #         pred = model(x)
        #         val_loss += F.mse_loss(pred, y).item()
        avg_train = train_loss / len(train_loader)
        # avg_val = val_loss / len(val_loader)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train:.3e}, Validation metric: {mape_epoch:.3f}% | Time: ' {time.time()-timee} s.' | LR: {lr:.2e}")

        # if avg_val < best_val_loss:
        #     best_val_loss = avg_val
        torch.save(model.state_dict(), "best_fno_pdebench_downsample_36chan_24modes_4layers_based.pt")

        scheduler.step()

    return model


# Example usage
if __name__ == "__main__":
    import h5py

    with h5py.File(r"merged_training_128x128_based.hdf5", "r") as f:
        data = torch.tensor(f['tensor'][:]).float()  # [10000, 201, 1024]
        x_grid = torch.tensor(f['x-coordinate'][:]).float() # [1024]
        print(x_grid.shape)
        t_grid = torch.tensor(f['t-coordinate'][:][:]).float()
        print(t_grid.shape)# [201]

    # data = torch.tensor(data).float()
    # x_grid = torch.tensor(x_grid).float()
    # t_grid = torch.tensor(t_grid).float()

    train_size = int(1.0 * len(data))
    train_dataset = PDEBench1DDataset(data[:train_size], x_grid, t_grid)
    # val_dataset = PDEBench1DDataset(data[train_size:], x_grid, t_grid)

    model = train_fno_pdebench(train_dataset)