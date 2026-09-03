import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
from torch.nn import MultiheadAttention
from torch.nn import LayerNorm, GELU
# from xformers.ops import memory_efficient_attention

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- PDEBench Dataset ---
class PDEBench1DDataset(Dataset):
    def __init__(self, tensor_data, x_grid, t_grid):
        self.tensor = tensor_data  # [N, T, X]
        self.x = x_grid           # [X]
        self.t = t_grid           # [T]

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


import math

# Try import xFormers memory-efficient attention
try:
    from xformers.ops import memory_efficient_attention as x_mem_eff_attn
    HAVE_XFORMERS = True
except Exception:
    HAVE_XFORMERS = False

# Helper: torch SDPA wrapper
def torch_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    # q,k,v: shapes [BT*H, L, Dh] (flattened batch+heads) or [B, L, H, Dh] will be handled below
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

class LocalWindowMultiHeadAttentionFast(nn.Module):
    """
    Fast sliding-window multi-head attention using xFormers (preferred) or PyTorch SDPA fallback.

    Input expected: x: [L, B, D] (same as your original)
    Parameters:
      - d_model, nhead: same as before
      - window_size: odd integer, local attention half-window = window//2
      - use_xformers: try to use xformers if available (default True)
    Returns: out [L, B, D]
    """
    def __init__(self, d_model, nhead, window_size, dropout=0.0, use_xformers=True):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert window_size % 2 == 1, "window_size should be odd (symmetric)"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.window = window_size
        self.pad = self.window // 2
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.use_xformers = use_xformers and HAVE_XFORMERS

        # cache for local band mask to avoid realloc each forward
        self.register_buffer("_band_mask", torch.tensor([]), persistent=False)

    def _build_band_mask(self, L, device, dtype):
        """
        Build mask shape [L, L] with 0 for allowed, -inf for disallowed (works with torch SDPA).
        We'll reuse by caching per size L.
        """
        if (self._band_mask is not None) and (self._band_mask.numel() != 0) and (self._band_mask.shape[0] == L):
            # ensure device/dtype match at call-time when used
            return self._band_mask.to(device=device, dtype=dtype)

        # build boolean allowed region: True where allowed
        idx = torch.arange(L, device=device)
        # distance matrix |i - j|
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        allowed = dist <= self.pad  # shape [L, L], True inside window
        # For torch.scaled_dot_product_attention, attn_mask expects float with -inf where masked
        mask = torch.full((L, L), float("-inf"), device=device, dtype=dtype)
        mask[allowed] = 0.0
        # store a CPU float32 version to avoid GPU persistent memory growth
        self._band_mask = mask.detach().cpu()
        return self._band_mask.to(device=device, dtype=dtype)

    def forward(self, x):
        # x: [L, B, D]
        L, B, D = x.shape
        assert D == self.d_model

        # 1) compute qkv [L, B, 3D]
        qkv = self.qkv(x)  # same device/dtype as x
        q, k, v = qkv.chunk(3, dim=-1)  # each [L, B, D]

        # 2) move to [B, L, D] and split heads -> [B, L, H, Dh]
        q = q.permute(1, 0, 2).contiguous()
        k = k.permute(1, 0, 2).contiguous()
        v = v.permute(1, 0, 2).contiguous()

        B2, L2, _ = q.shape  # B2==B, L2==L
        q = q.view(B2, L2, self.nhead, self.d_head)
        k = k.view(B2, L2, self.nhead, self.d_head)
        v = v.view(B2, L2, self.nhead, self.d_head)

        # xFormers path (preferred) - note: xformers expects [B, seq, heads, head_dim]
        if self.use_xformers:
            try:
                q_scaled = q / math.sqrt(self.d_head)
                out = x_mem_eff_attn(q_scaled, k, v, attn_bias=None, p=self.dropout)  # [B, L, H, Dh]
                out = out.reshape(B2, L2, D).permute(1, 0, 2).contiguous()  # [L, B, D]
                out = self.out_proj(out)
                return out
            except Exception as e:
                pass

        # -------------------------
        # Fallback: PyTorch scaled_dot_product_attention with band mask (local window)
        # We'll flatten batch+heads and call SDPA: [B*H, L, Dh]
        # -------------------------
        # reshape for SDPA
        # q,k,v currently [B, L, H, Dh]; permute to [B, H, L, Dh] -> reshape [B*H, L, Dh]
        q_flat = q.permute(0, 2, 1, 3).reshape(B2 * self.nhead, L2, self.d_head)
        k_flat = k.permute(0, 2, 1, 3).reshape(B2 * self.nhead, L2, self.d_head)
        v_flat = v.permute(0, 2, 1, 3).reshape(B2 * self.nhead, L2, self.d_head)

        # create attn_mask shaped for flattened batch: SDPA accepts (L, L) or (B*, L, L)
        mask = self._build_band_mask(L2, device=x.device, dtype=x.dtype)  # [L,L] with 0/-inf


        out_flat = torch_sdpa(q_flat, k_flat, v_flat, attn_mask=mask, dropout_p=self.dropout, is_causal=False)
        # out_flat: [B*H, L, Dh] -> reshape back to [B, L, H, Dh]
        out = out_flat.view(B2, self.nhead, L2, self.d_head).permute(2, 0, 1, 3).contiguous().view(L2, B2, D)
        # final linear projection
        out = self.out_proj(out)
        return out

# -------------------------
# Small Transformer layer using LocalWindow attention
# -------------------------
class LocalTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, window_size, dropout=0.1):
        super().__init__()
        self.attn = LocalWindowMultiHeadAttentionFast(d_model, nhead, window_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [seq, batch, d_model]
        y = self.attn(x)
        x = self.norm1(x + y)
        x2 = self.ffn(x)
        return self.norm2(x + x2)


# -------------------------
# FNO subclass with per-time-slice local attention after lifting
# -------------------------
class LocalAttnFNO(FNO):
    def __init__(self,
                 in_channels=3,
                 out_channels=1, 
                 modes=(64,64), 
                 width=64,
                 n_local_layers=1, 
                 n_heads=4, 
                 window_size=33, 
                 dropout=0.1):
        super().__init__(n_modes=modes,
                         hidden_channels=width,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         padding=8,
                         use_mlp=False)
        # stack of local transformer layers applied per-time-slice on spatial axis
        self.local_layers = nn.ModuleList([
            LocalTransformerLayer(d_model=width, nhead=n_heads, window_size=window_size, dropout=dropout)
            for _ in range(n_local_layers)
        ])

    def forward(self, x, output_shape=None, **kwargs):
        # keep signature compatible
        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        # optional pos emb
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # lifting -> [B, C=width, T, X]
        x = self.lifting(x)

        # optional padding
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        # APPLY local attention PER TIME-SLICE (efficient)
        B, C, T, X = x.shape
        # reshape to process all time-slices as independent batches:
        # target layout for layers: [seq=X, batch=B*T, d_model=C]
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(B * T, X, C)  # [B*T, X, C]
        x_seq = x_seq.permute(1, 0, 2).contiguous()                  # [X, B*T, C]

        for layer in self.local_layers:
            x_seq = layer(x_seq)  # [X, B*T, C]

        # back to [B, C, T, X]
        x_seq = x_seq.permute(1, 0, 2).contiguous().view(B, T, X, C)
        x = x_seq.permute(0, 3, 1, 2).contiguous()

        # FNO spectral blocks
        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        # unpad / project
        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)
        return x


# --- Training Loop ---
def train(postlift_dataset, epochs=10, bs=1, lr=1e-3):
    name_in = 'local_attention_fno_64_width_2_layers_4_heads_127_window_xformers'
    loss_hist = []
    val_hist = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = PostLiftTransformerFNO().to(device)
    # model = PerceiverFNO().to(device)
    model = LocalAttnFNO(width=64, n_local_layers=2, n_heads=4, window_size=127).to(device)
    # model = LocalPlusCoarseFNO().to(device)
    # model.load_state_dict(torch.load(f'{name_in}.pth',  weights_only=False))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loader = DataLoader(postlift_dataset, batch_size=bs, shuffle=True,)
    print(len(loader))
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        epoch_time = time.time()
        total_absperc = torch.tensor(0.0, device=device)
        n_samples = 0
        for b in loader:
            batch_time = time.time()
            xi = b['x'].to(device, non_blocking=True); yi = b['y'].to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(xi)

            loss = F.mse_loss(pred, yi)
            loss.backward()

            opt.step()

            with torch.no_grad():
                # y[torch.abs(y)<1e-3] = 1e-3*torch.sign(y[torch.abs(y)<1e-3])
                ape = torch.abs((yi - pred) / (yi.max() - yi.min()))  # .clamp(min=-1e-3, max = 1e-3)))  # [B, C, x, t]
                # print(ape.max())
                total_absperc += ape.sum()
                n_samples += ape.numel()

            # print(total)

            total += loss.item()
            # print(time.time() - batch_time)
        sched.step()
        mape_epoch = (total_absperc.item() / n_samples) * 100
        print(f"Epoch {ep}/{epochs}: MSE={total/len(loader):.5e} Time={time.time()-epoch_time} Val= {mape_epoch:.5f}%")
        loss_hist.append(total/len(loader))
        np.save(f'{name_in}_RD_loss_1e-3.npy', loss_hist)
        val_hist.append(mape_epoch)
        np.save(f'{name_in}_RD_val_1e-3.npy', val_hist)
        torch.save(model.state_dict(), f'{name_in}_finetune.pth')
    return model

if __name__ == '__main__':
    import h5py
    with h5py.File(r'/content/drive/MyDrive/PDE_Bench_experiments/ReacDiff_Nu0.5_Rho10.0.hdf5','r') as f:
        # r'C:\Users\EVM\Downloads\pde_bench_operators\ReacDiff_Nu0.5_Rho10.0.hdf5'
        # '/content/drive/MyDrive/PDE_Bench_experiments/merged_samples_fullres.hdf5'
        data = torch.tensor(f['tensor'][:]).float()
        xg = torch.tensor(f['x-coordinate'][:]).float()
        tg = torch.tensor(f['t-coordinate'][:][:-1]).float()
    name_in = 'local_attention_fno_64_width_2_layers_4_heads_127_window_xformers'
    ds = PDEBench1DDataset(data, xg, tg)
    m = train(ds)