import pytest
import numpy as np

from typing import Callable

import torch
import torch.nn as nn
import time

import os
# import sys
from pathlib import Path
import sys

# Find the folder of the current script and add it
script_dir = Path(__file__).resolve().parent.parent.parent.parent
print(script_dir)
sys.path.append(str(script_dir))

from os.path import join as pjoin

from muno.layers.channel_wise_conv import weightsOuterProduct, FactorizedDimensionSpectralConv
from neuralop.layers.spectral_convolution import SpectralConv

basename = pjoin(os.path.dirname(__file__), "..")

def prepareDummies(n_modes_dict: dict = None, channels: int = 3, rank: int = 4):
    device = 'cpu'
    if n_modes_dict is None:
        n_modes_dict = {"t": 16, "x": 32}

    max_spatial_dim = 3

    weights_t = [torch.empty((channels, n_modes_dict["t"]), device=device),] * rank
    for tensor_idx in range(rank):
        torch.nn.init.constant_(weights_t[tensor_idx], 1.)
        weights_t[tensor_idx] = torch.nn.Parameter(weights_t[tensor_idx], requires_grad=True)

    # Create/init spectral weight tensors: for spatial dimensions
    weights_x = [[torch.empty((channels, n_modes_dict["x"]), device=device)]*rank 
                        for _ in range(max_spatial_dim)]

    for dim_idx in range(max_spatial_dim):
        for tensor_idx in range(rank):
            torch.nn.init.constant_(weights_x[dim_idx][tensor_idx], 1.)
            weights_x[dim_idx][tensor_idx] = torch.nn.Parameter(weights_x[dim_idx][tensor_idx],
                                                                        requires_grad=True)
    return weights_t, weights_x, n_modes_dict

def measureTime(func: Callable, args: list, n_runs: int) -> int:
    run_durations = np.zeros(n_runs)

    for i in range(n_runs):
        t1 = time.time()
        last_func_op = func(*args)
        t2 = time.time()
        run_durations[i] = t2 - t1

    return last_func_op, run_durations

if __name__ == "__main__":
    N_RUNS = 100
    channels = 3

    x = torch.rand((10, channels, 21, 35, 35))

    rank = 200
    max_spatial_dim = 3
    print(f'RANK = {rank}, MAX_SPATIAL_DIM = {max_spatial_dim}')
    wt, wxs, n_modes = prepareDummies(channels=channels, rank=rank)
    mixer = torch.nn.Linear(rank, 1, bias = False)

    weights, run_durations = measureTime(weightsOuterProduct, [x, wxs, wt, mixer], N_RUNS)

    assert weights.ndim == x.ndim - 1, f'weightsOuterProduct botched dimensions of the weights: {weights.ndim} vs {x.ndim - 1}'
    assert weights.shape[0] == channels, 'weightsOuterProduct botched shapes: 0-th dim (number of channels)'
    print(f'weights type is {type(weights)}, shape: {weights.shape}, mean time: {np.mean(run_durations)}')

    conv_factorized = FactorizedDimensionSpectralConv(in_channels=channels, out_channels=channels, n_modes=n_modes,
                                                      complex_data=False, rank = rank, max_spatial_dim = max_spatial_dim)

    conv_default = SpectralConv(in_channels=channels*2, out_channels=channels*2, 
                                n_modes=(n_modes['t'], n_modes['x'], n_modes['x']),
                                complex_data=False, bias = False)
    
    conv_default_hd = SpectralConv(in_channels=channels*6, out_channels=channels*6, 
                                   n_modes=(n_modes['t'], n_modes['x'], n_modes['x'], n_modes['x']),
                                   complex_data=False, bias = False)

    x = x.repeat(1, 2, 1, 1, 1)

    x_out_f, fact_run_durations = measureTime(conv_factorized, [x,], 100)
    x_out_d, def_run_durations  = measureTime(conv_default, [x,], 100)

    print(f'Factorized: x_out type is {type(x_out_f)}, shape: {x_out_f.shape}, mean time: {np.mean(fact_run_durations)}')
    print(f'Default: x_out type is {type(x_out_d)}, shape: {x_out_d.shape}, mean time: {np.mean(def_run_durations)}')

    x_high_dim = torch.rand((10, channels*6, 21, 35, 35, 35))
    x_low_dim  = torch.rand((10, channels, 21, 35))

    x_out_hd, hd_run_durations        = measureTime(conv_factorized, [x_high_dim,], 2)
    x_out_hd_d, hd_def_run_durations  = measureTime(conv_default_hd, [x_high_dim,], 2)
    x_out_ld, ld_run_durations        = measureTime(conv_factorized, [x_low_dim,], 2)

    print(f'Default: x_out_hd type is {type(x_out_hd)}, shape: {x_out_hd.shape}, mean time: {np.mean(hd_run_durations)}')
    print(f'Default: x_out_hd_d type is {type(x_out_hd_d)}, shape: {x_out_hd_d.shape}, mean time: {np.mean(hd_def_run_durations)}')

    print(f'Default: x_out type is {type(x_out_ld)}, shape: {x_out_ld.shape}, mean time: {np.mean(ld_run_durations)}')

    print('Number of entries in registered named_parameters in factorized FNO:' + \
          f'{len(list(conv_factorized.named_parameters()))} vs {rank * (max_spatial_dim + 1) + 2}')

# [param_key for param_key, _ in conv_factorized.named_parameters()]
    # x_out = conv_factorized(x)

    # print('x_out for FactorizedDimensionSpectralConv.forward is ', x_out.shape)