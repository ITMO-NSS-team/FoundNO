"""
Grid embeddings — append normalized coordinate grids to feature tensors.
"""

import torch
import torch.nn as nn

from typing import Literal

class GridEmbeddingND(nn.Module):
    """
    Append normalized grid coordinates (ξ, η) ∈ [0,1]^{n} to the input tensor.

    The grid is cached as a buffer after the first forward pass
    so it can be reused by DNO's geometry injection layers.
    """

    def __init__(self, spatial_dim = 2, grid_boundaries = [0, 1]): # , dim_ord_mode: Literal['old', 'new'] = 'old'
        super().__init__()
        self.dim = spatial_dim
        
        if len(grid_boundaries) == 2 and not isinstance(grid_boundaries[0], (tuple, list)):
            grid_boundaries = [[grid_boundaries[0], grid_boundaries[1]]]

        if len(grid_boundaries) != spatial_dim or not isinstance(grid_boundaries[0], (tuple, list)):
            if len(grid_boundaries) == 2 and not isinstance(grid_boundaries[0], (tuple, list)):
                grid_boundaries = [[grid_boundaries[0], grid_boundaries[1]],] * spatial_dim
            elif len(grid_boundaries) == 1 and isinstance(grid_boundaries[0], (tuple, list)):
                grid_boundaries = list(grid_boundaries) * spatial_dim
            else:
                raise RuntimeError("Incorrect mode of the grid boundaries!")
            
        self.grid_boundaries = grid_boundaries

        self.register_buffer('grid', torch.empty(*[1, spatial_dim, 1] + [0,] * spatial_dim))

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor [B, C, T, H, W, ...]

        Returns
        -------
            torch.Tensor [B, C+dim, T, H, W, ...]
        """
        batch_size, res = x.shape[0], tuple(reversed([x.shape[-(i+1)] for i in range(self.dim)]))

        # Recompute grid only if resolution changed
        if any([self.grid.shape[i+1] != res[i] for i in range(len(res))]):
            grids = [torch.linspace(self.grid_boundaries[i][0], self.grid_boundaries[i][1], res[i], device=x.device)
                     for i in range(self.dim)]

            grids = torch.meshgrid(*grids, indexing='ij')
            grids = torch.stack(grids, dim=0).unsqueeze(0).unsqueeze(2)

            self.grid = grids.to(x.device)

        grid = self.grid.expand(batch_size, -1, x.shape[2], *[-1,] * (x.ndim-3))
        return torch.cat((x, grid), dim = 1)