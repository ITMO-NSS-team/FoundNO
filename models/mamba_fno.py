import time
from typing import Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Adjust this import to your neuralop package layout:
from neuralop.layers.embeddings import GridEmbeddingND, GridEmbedding2D
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.complex import ComplexValued
from neuralop.models import FNO        # user had this import in latest snippet
from neuralop.layers.padding import DomainPadding
from neuralop.models.base_model import BaseModel

# -------------------------
# PDEBench dataset (unchanged)
# -------------------------
class PDEBench1DDataset(Dataset):
    def __init__(self, tensor_data, x_grid, t_grid):
        self.tensor = tensor_data  # [N, T, X]
        self.x = x_grid            # [X]
        self.t = t_grid            # [T]

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        sol = self.tensor[idx]         # [T, X]
        u0 = sol[0:1]                  # [1, X]
        T, X = sol.shape

        # coordinate grids
        x_coords = self.x[None, :].repeat(T, 1)
        t_coords = self.t[:, None].repeat(1, X)
        coords = torch.stack([x_coords, t_coords], dim=-1)  # [T, X, 2]

        inp = torch.cat([u0.repeat(T, 1).unsqueeze(-1), coords], dim=-1)  # [T, X, 3]
        out = sol.unsqueeze(-1)  # [T, X, 1]

        return {'x': inp.permute(2,0,1),  # [3, T, X]
                'y': out.permute(2,0,1)}   # [1, T, X]


# -------------------------
# Try to import Mamba (multiple possible package names)
# -------------------------
MAMBA_AVAILABLE = False
MambaClass = None
try:
    # common package name
    import mamba_ssm as mamba_pkg
    if hasattr(mamba_pkg, "Mamba"):
        MambaClass = mamba_pkg.Mamba
        MAMBA_AVAILABLE = True
except Exception:
    try:
        # alternate package / repo layout
        from mamba import Mamba as MambaClass
        MAMBA_AVAILABLE = True
    except Exception:
        MAMBA_AVAILABLE = False

# -------------------------
# Pure-PyTorch fallback SSM (cheap, portable)
# -------------------------
class SimpleSSM(nn.Module):
    """
    Small pure-PyTorch SSM-ish block used as a fallback if Mamba isn't installed.
    It's a depthwise conv (per-channel linear kernel along sequence) + simple gating.
    Input: [B, L, D]  -> Output: [B, L, D]
    """
    def __init__(self, dim, kernel_size=9):
        super().__init__()
        pad = (kernel_size - 1) // 2
        # depthwise conv simulates a linear sequence kernel
        self.dw = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim, bias=True)
        self.gate = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

        # init small
        nn.init.xavier_uniform_(self.dw.weight)
        nn.init.zeros_(self.dw.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        # x: [B, L, D] -> conv expects [B, D, L]
        x_t = x.permute(0, 2, 1)
        y = self.dw(x_t)                # [B, D, L]
        y = torch.tanh(y)
        g = torch.sigmoid(self.gate(x_t))
        out = (1 - g) * x_t + g * y
        out = out.permute(0, 2, 1)      # [B, L, D]
        return self.norm(out)


# -------------------------
# Post-lift Mamba wrapper / processor
# -------------------------
class PostLiftMambaProcessor(nn.Module):
    """
    Apply Mamba (SSM) across the spatial axis after FNO.lifting.
    Input to this processor: x_lift of shape [B, C, T, X]
    It processes each time-slice independently: for each time t we run an SSM over the spatial sequence length X.
    - If Mamba is installed, we attempt to use it (fast CUDA kernels).
    - Otherwise we use a pure-PyTorch SimpleSSM fallback.
    """
    def __init__(self, hidden_channels, mamba_kwargs=None, fallback_kernel=9):
        super().__init__()
        self.C = hidden_channels
        self.mamba_kwargs = mamba_kwargs or {}
        self._use_mamba = False

        if MAMBA_AVAILABLE and (MambaClass is not None):
            # try to instantiate Mamba with common constructor signatures
            try:
                # typical constructor: Mamba(d_model=.., d_state=.., d_conv=..)
                self.ssm = MambaClass(d_model=self.C, d_state=self.C, d_conv=2, **self.mamba_kwargs)
                self._use_mamba = True
            except Exception:
                try:
                    # sometimes Mamba takes single argument width
                    self.ssm = MambaClass(self.C)
                    self._use_mamba = True
                except Exception:
                    # fallback to simple SSM if instantiation fails
                    self.ssm = SimpleSSM(dim=self.C, kernel_size=fallback_kernel)
                    self._use_mamba = False
        else:
            self.ssm = SimpleSSM(dim=self.C, kernel_size=fallback_kernel)
            self._use_mamba = False

        # small layernorm on last dim after processing
        self.norm = nn.LayerNorm(self.C)

    def forward(self, x_lift):
        """
        x_lift: [B, C, T, X]
        returns processed x: [B, C, T, X]
        """
        B, C, T, X = x_lift.shape
        assert C == self.C, f"expected channels {self.C}, got {C}"

        # Flatten time into batch dimension: tokens shape [B*T, X, C]
        tokens = x_lift.permute(0, 2, 3, 1).contiguous().view(B * T, X, C)  # [B*T, X, C]
        print
        # Process: Mamba typically expects [B, L, D] -> [B, L, D].
        if self._use_mamba:
            try:
                # Try call assuming signature: out = ssm(tokens)
                out = self.ssm(tokens)
            except Exception:
                try:
                    # Try permuted API: [B, D, L] -> [B, D, L]
                    out_tmp = self.ssm(tokens.permute(0, 2, 1))            # [B*T, C, X]
                    out = out_tmp.permute(0, 2, 1)                         # [B*T, X, C]
                except Exception as e:
                    # if Mamba fails at runtime, fallback to SimpleSSM path
                    out = self.ssm(tokens) if not isinstance(self.ssm, SimpleSSM) else self.ssm(tokens)
        else:
            out = self.ssm(tokens)  # fallback SimpleSSM

        # ensure shape and dtype
        out = out.contiguous()
        out = self.norm(out)  # layernorm over last dim
        # reshape back to [B, C, T, X]
        out = out.view(B, T, X, C).permute(0, 3, 1, 2).contiguous()
        return out


class PostLiftMambaProcessor3D(nn.Module):
    """
    Apply SSM across spatial dimensions (H and W) for each time slice.
    Input: [B, C, H, W, T] → process each (H, W) plane independently per time.
    """
    def __init__(self, hidden_channels, mamba_kwargs=None, fallback_kernel=9):
        super().__init__()
        self.C = hidden_channels
        self.mamba_kwargs = mamba_kwargs or {}
        self._use_mamba = False

        if MAMBA_AVAILABLE and MambaClass is not None:
            # print('HEREEEEE')
            try:
                self.ssm = MambaClass(d_model=self.C, d_state=self.C, d_conv=2, **self.mamba_kwargs)
                self._use_mamba = True
            except Exception:
                try:
                    self.ssm = MambaClass(self.C)
                    self._use_mamba = True
                except Exception:
                    self.ssm = SimpleSSM(dim=self.C, kernel_size=fallback_kernel)
                    self._use_mamba = False
        else:
            self.ssm = SimpleSSM(dim=self.C, kernel_size=fallback_kernel)
            self._use_mamba = False

        self.norm = nn.LayerNorm(self.C)

    def forward(self, x_lift):
        """
        x_lift: [B, C, H, W, T]
        Process each (H, W) spatial plane per time step.
        """
        B, C, H, W, T = x_lift.shape
        assert C == self.C

        # Flatten spatial dims: [B*T, H*W, C]
        tokens = x_lift.permute(0, 4, 2, 3, 1).contiguous().view(B * T, H * W, C)

        if self._use_mamba:
            try:
                out = self.ssm(tokens)
            except Exception:
                try:
                    out_tmp = self.ssm(tokens.permute(0, 2, 1))
                    out = out_tmp.permute(0, 2, 1)
                except Exception:
                    out = self.ssm(tokens)
        else:
            out = self.ssm(tokens)

        out = self.norm(out)
        out = out.view(B, T, H, W, C).permute(0, 4, 2, 3, 1).contiguous()
        return out


# -------------------------
# FNO subclass: apply Mamba after lifting (post-lift)
# -------------------------
class PostLiftMambaFNO(FNO):
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 modes=(64, 64),
                 width=64,
                 n_layers=4,
                 use_mamba_kwargs=None,
                 mamba_fallback_kernel=9,
                 padding=8,
                 use_mlp=False):
        """
        A compact FNO that runs the standard lifting, then a Mamba/SSM processor across spatial axis,
        then continues with FNO spectral blocks and projection.
        """
        super().__init__(n_modes=modes,
                         hidden_channels=width,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         padding=padding,
                         use_mlp=use_mlp,
                         n_layers=n_layers)

        # post-lift processor (Mamba or fallback)
        self.post_lift_ssm = PostLiftMambaProcessor(hidden_channels=width,
                                                    mamba_kwargs=use_mamba_kwargs,
                                                    fallback_kernel=mamba_fallback_kernel)

    def forward(self, x, output_shape=None, **kwargs):
        """
        FNO forward with Post-lift SSM:
         - positional embedding (optional)
         - lifting (inherited)
         - domain padding (optional)
         - post-lift Mamba / SSM across spatial axis per time-slice
         - FNO blocks, unpad, projection
        """
        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        # 1) positional embedding (if configured)
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # 2) lifting -> [B, C, T, X]
        x = self.lifting(x)

        # 3) domain padding (optional)
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        # 4) post-lift SSM processing (Mamba or fallback)
        x = self.post_lift_ssm(x)  # [B, C, T, X]

        # 5) apply FNO blocks
        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        # 6) unpad (if was padded)
        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        # 7) projection to output channels
        x = self.projection(x)
        return x

class PostLiftMambaFNO3D(FNO):
    def __init__(self, in_channels=5, out_channels=2, modes=(32, 32, 16), width=32, n_layers=4,
                 use_mamba_kwargs=None, mamba_fallback_kernel=9):
        super().__init__(
            n_modes=modes,
            hidden_channels=width,
            in_channels=in_channels,      # u0(2) + f(1) + x(1) + y(1)
            out_channels=out_channels,
            factorization='tucker',
            rank=0.05,
            implementation='factorized',
            n_layers=n_layers,
            use_channel_mlp = True,
            channel_mlp_dropout = 0.1,
            positional_embedding=None
        )
        self.post_lift_ssm = PostLiftMambaProcessor3D(
            hidden_channels=width,
            mamba_kwargs=use_mamba_kwargs,
            fallback_kernel=mamba_fallback_kernel
        )

    def forward(self, x):
        # Standard FNO pipeline
        if self.positional_embedding is not None:
            print(x.shape)
            x = self.positional_embedding(x)
        x = self.lifting(x)
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        # Apply Mamba/SSM after lifting
        x = self.post_lift_ssm(x)  # [B, C, H, W, T]

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)
        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)
        x = self.projection(x)
        return x

class PostLiftMambaLifting(BaseModel):
    def __init__(self,
                 in_channels=3,
                 out_channels=64,
                 modes=None,
                 width=None,
                 n_dim = 3,
                 padding=8,
                 resolution_scaling_factor = None,
                 use_mamba_kwargs=None,
                 mamba_fallback_kernel=9,
                 positional_embedding: Union[str, nn.Module] = "grid",
                 non_linearity: nn.Module = F.gelu):
        """
        A compact FNO that runs the standard lifting, then a Mamba/SSM processor across spatial axis,
        then continues with FNO spectral blocks and projection.
        """
        self.complex_data = False
        super().__init__()

        if width is None:
            width = out_channels
        else:
            assert out_channels == width, 'out_channels != width'
        self.n_dim = n_dim
        self.in_channels = in_channels

        ## Positional embedding
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=self.n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(f"Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}")
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding is None:
            self.positional_embedding = None
        else:
            raise ValueError(f"Error: tried to instantiate FNO positional embedding with {positional_embedding},\
                              expected one of 'grid', GridEmbeddingND")

        ## Domain padding
        if padding is not None and (
            (isinstance(padding, list) and sum(padding) > 0)
            or (isinstance(padding, (float, int)) and padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=padding,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        ## Resolution scaling factor
        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        ## Lifting layer
        # if adding a positional embedding, add those channels to lifting
        lifting_in_channels = in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += n_dim
        self.lifting = ChannelMLP(
            in_channels=lifting_in_channels,
            out_channels=out_channels,
            hidden_channels=out_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)

        self.post_lift_ssm = PostLiftMambaProcessor3D(hidden_channels=width,
                                                      mamba_kwargs=use_mamba_kwargs,
                                                      fallback_kernel=mamba_fallback_kernel)


    def forward(self, x, output_shape=None, **kwargs):
        """
        FNO forward with Post-lift SSM:
         - positional embedding (optional)
         - lifting (inherited)
         - domain padding (optional)
         - post-lift Mamba / SSM across spatial axis per time-slice
         - FNO blocks, unpad, projection
        """
        # 1) positional embedding (if configured)
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # 2) lifting -> [B, C, T, X, Y]
        x = self.lifting(x)

        # 3) domain padding (optional)
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        # 4) post-lift SSM processing (Mamba or fallback)
        x = self.post_lift_ssm(x)  # [B, C, T, X, Y]
        return x


# -------------------------
# Training loop (keeps your original loop style)
# -------------------------
def train(dataset, epochs=10, bs=2, lr=1e-3, device=None, compile_model=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    name_in = '/content/postlift_mamba_fno'
    loss_hist = []
    val_hist = []

    model = PostLiftMambaFNO(modes=(64,64), width=32, n_layers=4,
                             use_mamba_kwargs=None,
                             mamba_fallback_kernel=9).to(device)
    model.load_state_dict(torch.load(f'{name_in}.pth',  weights_only=False))
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print("torch.compile failed, continuing without it:", e)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        epoch_time = time.time()
        total_absperc = torch.tensor(0.0, device=device)
        n_samples = 0

        for b in loader:
            # batch_time = time.time()
            xi = b['x'].to(device, non_blocking=True)
            yi = b['y'].to(device, non_blocking=True)

            opt.zero_grad()
            pred = model(xi)

            loss = F.mse_loss(pred, yi)
            loss.backward()
            opt.step()

            with torch.no_grad():
                # normalized absolute percentage error used as metric
                denom = (yi.max() - yi.min()).clamp(min=1e-6)
                ape = torch.abs((yi - pred) / denom)
                total_absperc += ape.sum()
                n_samples += int(ape.numel())

            total += float(loss.item())
            # print(time.time()-batch_time)-
        sched.step()
        mape_epoch = (total_absperc.item() / n_samples) * 100.0
        avg_loss = total / len(loader)
        print(f"Epoch {ep}/{epochs}: MSE={avg_loss:.5e} Time={time.time()-epoch_time:.2f}s Val(MAPE)={mape_epoch:.5f}%")
        loss_hist.append(avg_loss)
        val_hist.append(mape_epoch)
        np.save(f'{name_in}_loss.npy', loss_hist)
        np.save(f'{name_in}_val.npy', val_hist)
        torch.save(model.state_dict(), f'{name_in}.pth')

    return model


if __name__ == '__main__':
    import h5py
    # adjust path for your environment (Colab or local)
    hdf5_path = "/content/ReacDiff_Nu0.5_Rho10.0.hdf5"
    #"/content/drive/MyDrive/PDE_Bench_experiments/ReacDiff_Nu0.5_Rho10.0.hdf5"
    # '/content/drive/MyDrive/PDE_Bench_experiments/merged_samples_fullres.hdf5'
    with h5py.File(hdf5_path, 'r') as f:
        data = torch.tensor(f['tensor'][:]).float()
        xg = torch.tensor(f['x-coordinate'][:]).float()
        tg = torch.tensor(f['t-coordinate'][:][:-1]).float()

    ds = PDEBench1DDataset(data, xg, tg)
    model = train(ds, epochs=10, bs=1, lr=1e-3, compile_model=False)
