from typing import Tuple, List, Union, Literal, Dict, Any
from types import BuiltinFunctionType

import numpy as np
import dill
import inspect
import warnings

# from abc import ABC, abstractmethod

from functools import singledispatchmethod

from collections.abc import Callable, Iterator, Mapping

import torch
from tensordict import TensorDict, make_tensordict
import torch.nn as nn
from torch.nn.parameter import Parameter

from muno.layers.skips import SkipLike
from muno.data.benchmarks.multiphysics_loaders import getSkipsChannel
        
        # sig = inspect.signature(combinator)
        # assert 

# Presets indicate, which tensors shall be passed into the corresponding hidden layers:
# x_0 -> L -> x_1 -> 

# TODO: refactor as a class?

# def generateSkips() -> List[Dict[Tuple[str, int, int], torch.nn.Module]]:
#     return {}

# PRESET_SKIPS = {'dno': ('g', -1, StandardSkip),
#                 'film': ('p', -1, FiLM),
#                 'unet': ('x', , )}


class Muno(nn.Module):
    _single_model: bool = False
    _empty: bool = True

    def __init__(self, liftings: List[torch.nn.Module] = None, core: torch.nn.Module = None,
                 projections: List[torch.nn.Module] = None, single_model: torch.nn.Module = None) -> None:
        assert single_model is None or (liftings is None and core is None and projections is None), \
            'incorrect setting of the Muno model: either single_model or liftings, core and projections have to be None'

        super().__init__()
        if single_model is None and core is not None:
            self._single_model = False
            if liftings is not None:
                assert isinstance(liftings, list), 'adapeters have to be passed as a LIST of torch.nn.Modules.'
                assert all([isinstance(lift, torch.nn.Module) for lift in liftings]), \
                    'adapeters have to be passed as a list of TORCH.NN.MODULES.'

                assert isinstance(projections, list), 'adapeters have to be passed as a LIST of torch.nn.Modules.'
                assert all([isinstance(projection, torch.nn.Module) for projection in projections]), \
                    'adapeters have to be passed as a list of TORCH.NN.MODULES.'
                assert len(projections) == len(liftings), \
                    f'numbers of projections and liftings have to match, got {len(liftings)} liftings and {len(projections)} projs.'

                self._liftings = torch.nn.ModuleList(liftings)
                self._projections = torch.nn.ModuleList(projections)
                self._adapters_set = True

                self._skip_handlers: List[Dict[int, SkipLike]] = []
            else:
                assert projections is None, 'If liftings arg is None, projections arg has to be None as well.'
                self._adapters_set = False
                self._liftings, self._projections = [], []

            assert isinstance(core, torch.nn.Module), 'core have to be passed as a torch.nn.Module.'
            self._core = core

            self._empty = False
        elif core is None and single_model is not None:
            self._single_model = True
            self._liftings, self._projections = None, None            
            self._core = single_model
            self._empty = False

    # def setSkip(self, skip: torch.nn.Module, mode: str, skip_from: int, skip_to: int):
    #     self._horizontal_skips_map[]
    # TODO: implement correct mapping method

    def setAdapter(self, lifting: torch.nn.Module, projection: torch.nn.Module, skip: Dict[int, SkipLike] = None):
        self._liftings.append(lifting)
        self._projections.append(projection)
        
        if skip is None:
            assert isinstance(skip, dict), \
                'Skip must be passed as a DICT of format int: SkipLike-object.'
            assert all([isinstance(key, int) for key in skip.keys()]), \
                'Skip must be passed as a dict of format INT: SkipLike-object.'
            assert all([isinstance(value, SkipLike) for value in skip.values()]), \
                'Skip must be passed as a dict of format int: SKIPLIKE-object.'
            
            self._skip_handlers.append(skip)

    def to(self, device):
        if not self._single_model:
            for idx, _ in enumerate(self._liftings):
                self._liftings[idx].to(device=device)
                self._projections[idx].to(device=device)

        self._core.to(device=device)

    def parameters(self, recurse = True) -> Iterator[Parameter]:
        for lift in self._liftings:
            yield from lift.parameters(recurse=recurse)

        yield from self._core.parameters(recurse=recurse)

        for proj in self._projections:
            yield from proj.parameters(recurse=recurse)

        yield from ()

    def named_parameters(self, prefix = '', recurse = True, remove_duplicate = True):
        for lift in self._liftings:
            yield from lift.named_parameters(prefix = prefix, recurse = recurse, remove_duplicate = remove_duplicate)

        yield from self._core.named_parameters(prefix = prefix, recurse = recurse, remove_duplicate = remove_duplicate)

        for proj in self._projections:
            yield from proj.named_parameters(prefix = prefix, recurse = recurse, remove_duplicate = remove_duplicate)

        yield from ()

    def setMode(self, mode: Literal['pretrain', 'finetune', 'eval'] = 'pretrain') -> None:
        assert mode in {'pretrain', 'finetune', 'eval'}, \
            f"Got incorrect mode {mode}, expected 'pretrain', 'finetune', or 'eval'."
        assert not self._empty, 'Trying to set mode for an empty model.'
        self._mode = mode
        if mode == 'finetune' or mode == 'eval':
            for param in self._core.parameters():
                param.requires_grad = False

        if mode == 'eval':
            for adapter_idx, _ in enumerate(self._liftings):
                for param in self._liftings[adapter_idx].parameters():
                    param.requires_grad = False

                for param in self._projections[adapter_idx].parameters():
                    param.requires_grad = False

    @staticmethod
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

            if all([torch.unique(tensor[traj_idx:traj_idx+1, 0:1, ...]).ndim == 1 for traj_idx in range(tensor_slice.shape[0])]):
                tensor_slice = torch.unique(tensor_slice[:, 0:1, ...]).unsqueeze(axis)
            return tensor_slice

        with torch.no_grad(): # torch.select(to_slice, axis, key).unsqueeze(axis)
            skip_args = make_tensordict({str(key): prepareSkipTensor(to_slice, axis, key) for key in cutout_idxs}, 
                                        batch_size=[to_slice.shape[0],],
                                        device = to_slice.device)
            to_slice = to_slice.index_select(axis, remaining_idxs)
        
        return to_slice, skip_args

    @singledispatchmethod
    def forward(self, x, adapter_idx: int = 0, output_shape = None, **kwargs):
        raise NotImplementedError('Default generic singledispatch method is not available.')

    @forward.register
    def _(self, x: torch.Tensor, adapter_idx: int = 0, output_shape = None, **kwargs) -> torch.Tensor:
        if any([-2 == skip_hash[1] for skip_hash in self._skip_handlers[adapter_idx]]):
            skips_channels = getSkipsChannel(self._skip_handlers[adapter_idx], lambda x: x.origin == -2)

            x, skip_from_init = self.splitChannels(skips_channels, x, axis = 1)
            skip_tensors = {-2: skip_from_init,}
        else:
            skip_tensors = {}

        if output_shape is not None:
            raise NotImplementedError('Unexpected behavior, output shape has to be None')
        if self._empty or not self._adapters_set:
            raise RuntimeError('Trying to call an unprepared model')

        if not self._single_model:
            x = self._liftings[adapter_idx](x) # add **kwargs processor 

        if any([-1 == skip_hash[1] for skip_hash in self._skip_handlers]):
            skip_tensors[-1] = torch.clone(x)

        x = self._core(x, skip_tensors, self._skip_handlers[adapter_idx])

        if not self._single_model:
            x = self._projections[adapter_idx](x) # add **kwargs processor

        return x

    @forward.register
    def _(self, x: tuple, adapter_idx: int = 0, output_shape = None, **kwargs) -> torch.Tensor:
        # Argument x is expected to have forms of Tuple[torch.Tensor]
        if output_shape is not None:
            raise NotImplementedError('Unexpected behavior, output shape has to be None')
        if self._empty or not self._adapters_set:
            raise RuntimeError('Trying to call an unprepared model')

        if not self._single_model:
            x[0] = self._liftings[adapter_idx](x[0]) # add **kwargs processor 

        x[0] = self._core(x[0])

        if not self._single_model:
            x[0] = self._projections[adapter_idx](x[0]) # add **kwargs processor 

        return x[0]

    @forward.register
    def _(self, x: dict, adapter_idx: int = None, output_shape = None, **kwargs) -> Dict[int, torch.Tensor]:
        # Argument x is expected to have forms of Dict[int, torch.Tensor]
        assert len(x) == len(self._liftings), 'Mismatching adapters and problems in forward inputs.'
        if adapter_idx is not None:
            warnings.warn(f"Calling dict-mapped forward desipte having explicitly passed adapter index: {adapter_idx}.")

        return {adapter_idx: self.forward(inp_tensor, adapter_idx = adapter_idx) for adapter_idx, inp_tensor in x.items()}

    @classmethod
    def load(cls, model_path: Union[str, Tuple[Union[None, str, Tuple[str]]]], _SAVE_LOAD_PARAMS: dict = {}):
        if isinstance(model_path, str):
            core = torch.load(f = model_path, pickle_module = dill, **_SAVE_LOAD_PARAMS)
            return cls(single_model = core)
        else:
            assert isinstance(model_path, tuple) and len(model_path) == 3, \
                'Saving lifting-main part-projection model requires tuple of str arg with len 3.'
            assert isinstance(model_path[1], str), 'Main core path has to be a str.'
            main_fno = torch.load(f = model_path[1], pickle_module = dill, **_SAVE_LOAD_PARAMS)

            if model_path[0] is None:
                assert (model_path[0] is None), 'Can not load projections without liftings.'
                input_adapters, output_adapters = None, None

            elif isinstance(model_path[0], str):
                assert isinstance(model_path[2], str), 'If lifting is passed as a str, proj. has to be a str too.'
                input_adapters = torch.load(f = model_path[0], pickle_module = dill, **_SAVE_LOAD_PARAMS)
                output_adapters = torch.load(f = model_path[2], pickle_module = dill, **_SAVE_LOAD_PARAMS)

            else:
                assert (isinstance(model_path[0], (list, tuple))), \
                    'Liftings have to be passed as list or tuple, if multiple adapters are expected.'
                assert (len(model_path[0]) == len(model_path[2])), \
                    f'If liftings are passed as {len(model_path[0])} elems, proj. has to be a {len(model_path[2])} elems.'
                input_adapters, output_adapters = [], []
                for adapter_idx in range(len(model_path[0])):
                    input_adapters.append(torch.load(f = model_path[0][adapter_idx], 
                                                     pickle_module = dill, **_SAVE_LOAD_PARAMS))
                    output_adapters.append(
                        torch.load(f = model_path[2][adapter_idx],
                                   pickle_module = dill, **_SAVE_LOAD_PARAMS))

            return cls(liftings = input_adapters, core = main_fno, projections = output_adapters)

    def save(self, model_path: Union[str, Tuple[str, List[str]]], _SAVE_LOAD_PARAMS: dict = {}):
        if self._single_model:
            assert isinstance(model_path, str), 'Saving of a single model requires a single path str argument'
            torch.save(obj=self._core, f=model_path, pickle_module=dill, **_SAVE_LOAD_PARAMS)
        else:
            assert isinstance(model_path, tuple) and len(model_path) == 3, \
                'Saving lifting-main part-projection model requires tuple of str arg with len 3'
            torch.save(obj=self._core, f=model_path[1], pickle_module=dill, **_SAVE_LOAD_PARAMS)

            if isinstance(model_path[0], str):
                assert isinstance(model_path[2], str), \
                    'If a string is a path for lifting model, a string has to be a path for proj. too.'
                warnings.warn("Saving a single lifting and projection.")
                torch.save(obj=self._liftings[0], pickle_module=dill, f=model_path[0], **_SAVE_LOAD_PARAMS)
                torch.save(obj=self._projections[0], pickle_module=dill, f=model_path[2], **_SAVE_LOAD_PARAMS)

            elif isinstance(model_path[0], (list, tuple)):
                assert (isinstance(model_path[2], (list, tuple)) and len(model_path[0]) == len(model_path[2])), \
                    'If a list/tuple is a path for lifting model, a list/tuple has to be a path for proj. too.'
                assert len(self._liftings) == len(model_path[2]), 'Mismatching numbers of filenames and submodels.'
                for idx in range(len(model_path[0])):
                    torch.save(obj=self._liftings[idx], pickle_module=dill, f=model_path[0][idx],
                               **_SAVE_LOAD_PARAMS)
                    torch.save(obj=self._projections[idx], pickle_module=dill, f=model_path[2][idx],
                               **_SAVE_LOAD_PARAMS)

    def toDataParallel(self, devices: Union[List[int], int] = [], dim: int = 0) -> torch.nn.DataParallel:
        if isinstance(devices, int):
            devices = [devices,]

        self.to(devices[0])
        parallelized = torch.nn.DataParallel(self, device_ids = devices, dim = dim)
        parallelized.to(devices[0])
        return parallelized