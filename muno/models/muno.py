from typing import Tuple, List, Union, Literal, Dict
import dill

from functools import singledispatchmethod

from collections.abc import Callable, Iterator, Mapping

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

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

                self._liftings = liftings
                self._projections = projections
                self._adapters_set = True
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

    def set_adapter(self, lifting, projection):
        self._liftings.append(lifting)
        self._projections.append(projection)

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

    @singledispatchmethod
    def forward(self, x, adapter_idx: int = 0, output_shape = None, **kwargs):
        raise NotImplementedError('Default generic singledispatch method is not available.')

    @forward.register
    def _(self, x: torch.Tensor, adapter_idx: int = 0, output_shape = None, **kwargs) -> torch.Tensor:    
        if output_shape is not None:
            raise NotImplementedError('Unexpected behavior, output shape has to be None')
        if self._empty or not self._adapters_set:
            raise RuntimeError('Trying to call an unprepared model')

        if not self._single_model:
            x = self._liftings[adapter_idx](x) # add **kwargs processor 

        x = self._core(x)

        if not self._single_model:
            x = self._projections[adapter_idx](x) # add **kwargs processor 

        return x

    @forward.register
    def _(self, x: dict, adapter_idx: int = 0, output_shape = None, **kwargs) -> Dict[int, torch.Tensor]: # x: Dict[int, torch.Tensor]
        assert len(x) == len(self._liftings), 'Mismatching adapters and problems in forward inputs.'

        return {adapter_idx: self.forward(inp_tensor, adapter_idx = adapter_idx) for adapter_idx, inp_tensor in x.items()}

    @classmethod
    def load(cls, model_path: Union[str, Tuple[None, str, Tuple[str]]], _SAVE_LOAD_PARAMS: dict = {}):
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