import pytest
import numpy as np

from typing import Any, Tuple
import warnings

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

from tensordict import TensorDict, make_tensordict

from muno.data.benchmarks.multiphysics_loaders import filterChannelsBySkips, getSkipsChannel
from muno.utils.model_factory import generateFiLMSkips, generateDNOSkips, build_model
from muno.models.muno import Muno

def splitChannels(indexes: Any, to_slice: torch.Tensor, axis: int = 1) -> Tuple[torch.Tensor, TensorDict]:
    try:
        cutout_idxs = set(indexes)
    except:
        warnings.warn(f"Got incorrect indexes in splitChannels, defaulting to spliting nothing from the argument")
        return to_slice, TensorDict({})
    
    remaining_idxs = torch.tensor([idx for idx in list(range(to_slice.shape[axis])) 
                                    if idx not in cutout_idxs]).to(torch.int32)

    def prepareSkipTensor(tensor: torch.Tensor, axis: int, key: int):
        tensor_slice = torch.select(tensor, axis, key).unsqueeze(axis)
        assert tensor_slice.shape[1] == 1, \
            f'Tensors must represent single channels of the input, instead got {tensor_slice.shape[1]} chan. at once.'

        if all([len(torch.unique(tensor_slice[traj_idx:traj_idx+1, 0:1, ...])) == 1 for traj_idx in range(tensor_slice.shape[0])]):
            tensor_slice = torch.unique(tensor_slice[:, 0:1, ...]).unsqueeze(axis)

        return tensor_slice

    with torch.no_grad(): # torch.select(to_slice, axis, key).unsqueeze(axis)
        skip_args = make_tensordict({str(key): prepareSkipTensor(to_slice, axis, key) for key in cutout_idxs}, # 
                                    batch_size=[to_slice.shape[0],],
                                    device = to_slice.device)
        to_slice = to_slice.index_select(axis, remaining_idxs)
    
    return to_slice, skip_args


if __name__ == "__main__":
    x = torch.randn((15, 6, 16, 32, 32))
    for i in range(x.shape[0]):
        x[i, 2, ...] = torch.full((1, 1, 16, 32, 32), i + 0.3)
        x[i, 3, ...] = torch.full((1, 1, 16, 32, 32), i ** 3)

    y = torch.randn((15, 2, 16, 32, 32))

    batch_dummy = {'x': x, 'y': y}
    channels = (filterChannelsBySkips(x), y.shape[1])
    print(channels)

    # {'kind': 'single', 'name': 'fno'}
    model_blocks: torch.nn.Module = build_model([(6, 6),], {'kind': 'adapter_core_adapter',
                                                            'name': 'adapted_cfno',
                                                            "params": [{}, {"n_modes": {"t": 1, "x": 32},
                                                                            "domain_padding": 0.,
                                                                            "residual_u0": False}, {}]}) # "residual_u0": False # _no_mamba
                                                            # })
    try:
        print(type(model_blocks), model_blocks.in_channels, model_blocks.out_channels)
    except:
        print(type(model_blocks[1]), model_blocks[0][0].in_channels, model_blocks[1].n_modes, model_blocks[2][0].out_channels)
        
    if isinstance(model_blocks, tuple):
        model = Muno(liftings = model_blocks[0], core = model_blocks[1],
                     projections = model_blocks[2], rollover_mode = 'autoreg', autoreg_training_mode='full') # autoreg vs rollout 
    else:
        model = Muno(single_model = model_blocks, rollover_mode = 'autoreg', autoreg_training_mode='full') # rolling_origin vs full


    dno_handlers  = generateDNOSkips(model, grid_channels = (4, 5)) # , core_is_factorized=False
    film_handlers = generateFiLMSkips(model, film_gen_kwargs = {'input_channels': (2, 3),
                                                                'num_layers': 3,
                                                                'layers_widths': 5})

    # print(f'len(dno_handlers): {len(dno_handlers)}')
    # print(f'len(film_handlers): {len(film_handlers)}')
     
    # for idx, handler in enumerate(dno_handlers):
    #     print(f'handler {idx} of {len(dno_handlers)}', type(handler), handler.info(), hash(handler))

    # for idx, handler in enumerate(film_handlers):
    #     print(f'handler {idx} of {len(film_handlers)}', type(handler), handler.info(), hash(handler))

    skip_handlers = dno_handlers + film_handlers
    skip_handlers = [{hash(handler): handler for handler in skip_handlers},]
    assert len(skip_handlers[0]) == len(dno_handlers) + len(film_handlers), \
        f'Somehow, multiple skips had the same ID: {len(skip_handlers)} vs {len(dno_handlers)} and {len(film_handlers)}'

    # model.setSkips(skip_handlers)
    model.to('cuda')

    # print(model._skip_handlers)
    # raise NotImplementedError('!')
    skips_channels = getSkipsChannel(skip_handlers[0], lambda x: x.origin == -2)

    # print(f'skips_channels are: {skips_channels}')
    # x, skip_tensors = splitChannels((2, 3, 4), batch_dummy['x'])
    # print(type(x), x.shape, type(skip_tensors), [t.shape for t in skip_tensors.values()])

    RERUNS = 100
    times = []
    for i in range(RERUNS):
        # print(i)
        t1 = time.time()
        pred = model(batch_dummy["x"].to('cuda'), adapter_idx = 0)
        t2 = time.time()
        times.append(t2-t1)

    print(f'Avg. time is {np.mean(times)}')
    print(f'pred.shape is {pred.shape}')

