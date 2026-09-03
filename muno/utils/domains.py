from typing import Dict, List
from functools import singledispatch

import torch
import numpy as np


class Domain():
    """ 
    Creates a uniform grid of N-dimensional spatial points
    """

    def __init__(self, params: dict, device = 'cuda'):
        self.ndim = len(params)
        self._params = params
        # print(f'Initializing domain with params {params}')

        self._dxs = {key : val['L'] / (val['n'] - 1) for key, val in params.items()}
        # self._unique_coords = {key : torch.from_numpy(np.arange(0, val['L'] + self._dxs[key], self._dxs[key])).to(device) 
        #                        for key, val in params.items()}
        self._unique_coords = {key : torch.linspace(0, val['L'], val['n'], device=device) 
                               for key, val in params.items()}        #.to(device) 
        # print([(key, value.shape, self._unique_coords) for key, value in self._unique_coords.items()])
        self.dimensions = [grid.shape[0] for key, grid in self._unique_coords.items()]

        coords = torch.meshgrid(*self._unique_coords.values(), indexing = 'ij') # .to(device)
        # print('coords[0].shape ', coords[0].shape)
        self._grid = torch.stack([var_coord for var_coord in coords]).to(device).to(dtype=torch.float32)

    def get_step(self, dim: str = 't'):
        assert self._unique_coords[dim].size()[0] > 1, f'Single element {dim} dimension can not yield step'
        return (self._unique_coords[dim][1] - self._unique_coords[dim][0])

    def get_grid(self, incl_t: bool = False):
        # print(f'self._grid in Domain: {self._grid.shape}')
        if incl_t:
            return self._grid
        else:
            return self._grid[1:, ...]


def assertGridUniformity(grids: List[torch.Tensor]):
    for grid in grids:
        delta = grid[1] - grid[0]
        for idx, point in enumerate(grid[:-1]):
            if (not np.isclose(grid[idx+1], point + delta)):
                raise ValueError('Not uniform grids are not supported for FFT-based neural operators.')

def formGridDict(grid: torch.Tensor):
    assert isinstance(grid, torch.Tensor) and (grid.ndim == 1), 'grids have to a single dimensional torch.Tensors'

    return {'n': grid.size(0), 'L': grid[1] - grid[0]}

@singledispatch
def createDomain(arg):
    raise NotImplementedError('Trying to call generic createDomain')

@createDomain.register
def _(arg: dict): # Dict[str, dict]
    for elem in arg.values():
        assert ('n' in elem.keys() and 'L' in elem.keys())
    return Domain(arg)

@createDomain.register
def _(arg: list): # List[torch.Tensor]
    labels = ['t',] + ['x' + str(elem) for elem in range(len(arg))]
    assertGridUniformity(arg)
    arg_parsed = {labels[idx] : formGridDict(arg) for idx, arg in enumerate(arg)}
    return Domain(arg_parsed)