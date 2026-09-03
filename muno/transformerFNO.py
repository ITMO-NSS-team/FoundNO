import torch
import torch.nn as nn
import torch.nn.functional as F
from neuraloperator.neuralop.models import FNO
from torch.utils.data import Dataset, DataLoader
import time
from torch.nn import MultiheadAttention
from torch.nn import LayerNorm, GELU
# from xformers.ops import memory_efficient_attention

import torch
import torch.nn as nn
import torch.nn.functional as F

# class FastAttnLayerFlash(nn.Module):
#     def __init__(self, d_model, nhead, dropout=0.0, causal=False):
#         super().__init__()
#         assert d_model % nhead == 0, "d_model must be divisible by nhead"
#         self.d_model = d_model
#         self.nhead = nhead
#         self.d_head = d_model // nhead
#         self.causal = causal
#         self.dropout = dropout
#
#         # combine QKV for efficiency
#         self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
#         self.out = nn.Linear(d_model, d_model)
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.ff = nn.Sequential(
#             nn.Linear(d_model, d_model * 2),
#             nn.GELU(),
#             nn.Linear(d_model * 2, d_model),
#             nn.Dropout(dropout),
#         )
#
#     def forward(self, x):
#         # x: [seq_len, batch, d_model]
#         seq_len, B, _ = x.shape
#
#         # 1) project and split Q/K/V
#         qkv = self.qkv(x)                               # [L, B, 3*d_model]
#         q, k, v = qkv.chunk(3, dim=-1)                  # each [L, B, d_model]
#         # reshape to [L, B*nhead, d_head]
#         def reshape(t):
#             L, B, D = t.shape
#             return (t
#                 .view(L, B, self.nhead, self.d_head)
#                 .permute(1,2,0,3)
#                 .reshape(B*self.nhead, L, self.d_head)
#             )
#         q, k, v = reshape(q), reshape(k), reshape(v)
#
#         # 2) build attn_bias mask if needed
#         attn_bias = None
#         if self.causal:
#             # lower-triangular mask: [1, L, L]
#             mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
#             attn_bias = mask.unsqueeze(0)
#
#         # 3) flash attention
#         #    PyTorch >=2.1: scaled_dot_product_attention
#         y = F.scaled_dot_product_attention(
#             q, k, v,
#             attn_mask=attn_bias,
#             dropout_p=self.dropout,
#             is_causal=self.causal,
#             # backend="flash" is automatic if available
#         )  # [B*nhead, L, d_head]
#
#         # 4) reshape back to [L, B, d_model]
#         y = (y
#              .view(B, self.nhead, seq_len, self.d_head)
#              .permute(2,0,1,3)
#              .reshape(seq_len, B, self.d_model)
#         )
#         # output projection + residual + norm
#         x2 = self.norm1(x + self.out(y))
#
#         # feed-forward
#         x3 = self.ff(x2)
#         return self.norm2(x2 + x3)

# class TransformerWithFastAttn(nn.Module):
#     def __init__(self, width, n_heads, dropout):
#         super().__init__()
#         self.attn = FastAttnLayer(
#             d_model=width,
#             nhead=n_heads,
#             dropout=dropout,
#             causal=False  # set True if you need autoregressive masking
#         )
#         self.norm1 = LayerNorm(width)
#         self.ff    = nn.Sequential(
#             nn.Linear(width, width*2),
#             GELU(),
#             nn.Linear(width*2, width),
#             nn.Dropout(dropout)
#         )
#         self.norm2 = LayerNorm(width)
#
#     def forward(self, x):
#         # x: [seq_len, batch, width]
#         y = self.attn(x)
#         x = self.norm1(x + y)
#         y = self.ff(x)
#         return self.norm2(x + y)
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

# --- FNO with Post-Lift Transformer inserted in forward ---

class PostLiftTransformerFNO(FNO):
    def __init__(self, modes=(64,64), width=16,
                 n_layers=2, n_heads=4, dropout=0.1):
        super().__init__(
            n_modes=modes,
            hidden_channels=width,
            in_channels=3,
            out_channels=1,
            padding=8,
            use_mlp=False,
        )
        # Transformer after lifting
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=n_heads,
            dim_feedforward=width*2, dropout=dropout,
            activation='gelu'
        )
#         encoder_layers = [
#     FastAttnLayerFlash(d_model=width, nhead=n_heads, dropout=dropout, causal=False)
#     for _ in range(n_layers)
# ]
#         self.post_lift_trans = nn.Sequential(*encoder_layers)
        self.post_lift_trans = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x, output_shape=None, **kwargs):
        # 1) Positional embedding if exists
        if self.positional_embedding:
            x = self.positional_embedding(x)

        # 2) Lift to latent space
        x = self.lifting(x)  # [B, width, T, X]

        # 3) Optional padding
        if self.domain_padding:
            x = self.domain_padding.pad(x)

        # 4) Apply post-lift Transformer over spatial axis per time-slice
        B, C, T, X = x.shape
        # reshape to [B*T, X, C]
        x_t = x.permute(0,2,3,1).reshape(B*T, X, C)
        # permute to [X, B*T, C]
        x_t = x_t.permute(1,0,2).contiguous()
        # Transformer
        x_t = self.post_lift_trans(x_t)
        # back to [B, width, T, X], ensure contiguous
        x = x_t.permute(1,2,0).reshape(B, T, C, X).permute(0,2,1,3).contiguous()

        # 5) FNO spectral blocks
        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx,
                               output_shape=(None if output_shape is None else output_shape[layer_idx]) if output_shape else None)

        # 6) Optional unpadding
        if self.domain_padding:
            x = self.domain_padding.unpad(x)

        # 7) Projection to output
        x = self.projection(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuraloperator.neuralop.models import FNO
from torch.utils.data import Dataset, DataLoader

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

# --- FNO with Perceiver-Style Cross Attention Enhancement ---
class PerceiverFNO(FNO):
    def __init__(self, modes=(12,12), width=64,
                 n_latents=16, n_heads=4, dropout=0.1):
        super().__init__(
            n_modes=modes,
            hidden_channels=width,
            in_channels=3,
            out_channels=1,
            padding=8,
            use_mlp=False,
        )
        # learnable latent queries: [n_latents, width]
        self.latent_queries = nn.Parameter(torch.randn(n_latents, width))
        # cross-attention to pool from tokens into latents
        self.cross_attn = nn.MultiheadAttention(embed_dim=width,
                                                num_heads=n_heads,
                                                dropout=dropout,
                                                batch_first=True)
        # small latent transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=n_heads,
            dim_feedforward=width*2, dropout=dropout,
            activation='gelu'
        )
        self.latent_trans = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=1)

    def forward(self, x, output_shape=None, **kwargs):
        # 1) optional pos embedding
        if self.positional_embedding:
            x = self.positional_embedding(x)
        # 2) lift
        x = self.lifting(x)  # [B, C=width, T, X]
        # 3) optional pad
        if self.domain_padding:
            x = self.domain_padding.pad(x)
        B, C, T, X = x.shape
        # 4) flatten tokens: [B, T*X, C]
        tokens = x.permute(0,2,3,1).reshape(B, T*X, C)
        # 5) prepare queries: expand to batch
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        # 6) cross-attention: queries <- tokens
        lat, _ = self.cross_attn(queries, tokens, tokens)
        # 7) latent transformer: [B, n_latents, C]
        lat = self.latent_trans(lat.permute(1,0,2)).permute(1,0,2)
        # 8) summary vector: mean over latents: [B, C]
        summary = lat.mean(dim=1)
        # 9) broadcast back to spatial: [B, C, T, X]
        summary = summary.view(B, C, 1, 1).expand(-1, -1, T, X)
        # 10) add global context
        x = x + summary
        # 11) FNO spectral blocks
        for li in range(self.n_layers):
            x = self.fno_blocks(x, li,
                               output_shape=(None if output_shape is None else output_shape[li]) if output_shape else None)
        # 12) unpad & project
        if self.domain_padding:
            x = self.domain_padding.unpad(x)
        return self.projection(x)

# Example train loop similar to before, replace PostLiftTransformerFNO with PerceiverFNO.

# --- Training Loop ---
def train(postlift_dataset, epochs=100, bs=4, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = PostLiftTransformerFNO().to(device)
    model = PerceiverFNO().to(device)
    # model.load_state_dict(torch.load('postlift_transformer_fno_100_epochs.pth',  weights_only=False))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loader = DataLoader(postlift_dataset, batch_size=bs, shuffle=True)
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        epoch_time = time.time()
        for b in loader:
            batch_time = time.time()
            xi = b['x'].to(device); yi = b['y'].to(device)
            opt.zero_grad()
            pred = model(xi)
            loss = F.mse_loss(pred, yi)
            loss.backward()
            opt.step()
            total += loss.item()
            # print(time.time()-batch_time)
        sched.step()
        print(f"Epoch {ep}/{epochs}: MSE={total/len(loader):.4e} Time={time.time()-epoch_time}")
    return model

# --- Example ---
if __name__ == '__main__':
    import h5py
    with h5py.File(r'merged_training_128x128_based.hdf5','r') as f:
        data = torch.tensor(f['tensor'][:]).float()
        xg = torch.tensor(f['x-coordinate'][:]).float()
        tg = torch.tensor(f['t-coordinate'][:]).float()

    ds = PDEBench1DDataset(data, xg, tg)
    m = train(ds)
    torch.save(m.state_dict(),'postlift_perciever_fno_100_epochs_finetune.pth')
