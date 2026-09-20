"""
Grid embeddings — append normalized coordinate grids to feature tensors.
"""

import torch
import torch.nn as nn


class GridEmbeddingND(nn.Module):
    """
    Append normalized grid coordinates (ξ, η) ∈ [0,1]² to the input tensor.

    The grid is cached as a buffer after the first forward pass
    so it can be reused by DNO's geometry injection layers.
    """

    def __init__(self, dim=2, grid_boundaries=[[0, 1], [0, 1]]):
        super().__init__()
        self.dim = dim
        self.grid_boundaries = grid_boundaries
        self.register_buffer('grid', torch.empty(1, 0, 0, dim))

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor [B, H, W, C]

        Returns
        -------
        torch.Tensor [B, H, W, C+dim]
        """
        batch_size, res_x, res_y = x.shape[0], x.shape[1], x.shape[2]

        # Recompute grid only if resolution changed
        if self.grid.shape[1] != res_x or self.grid.shape[2] != res_y:
            grid_x = torch.linspace(
                self.grid_boundaries[0][0], self.grid_boundaries[0][1],
                res_x, device=x.device)
            grid_y = torch.linspace(
                self.grid_boundaries[1][0], self.grid_boundaries[1][1],
                res_y, device=x.device)
            grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
            grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
            self.grid = grid.to(x.device)

        grid = self.grid.expand(batch_size, -1, -1, -1)
        return torch.cat((x, grid), dim=-1)
