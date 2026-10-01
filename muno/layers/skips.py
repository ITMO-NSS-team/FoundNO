import warnings
import inspect

from typing import Tuple, Any, Callable, Literal, Final, List, Union
from types import BuiltinFunctionType
from functools import singledispatch

import numpy as np

import torch
from tensordict import TensorDict

def checkSkipRequirements(core: torch.nn.Module) -> bool:
    try:
        sgn = inspect.signature(core.__class__.forward)
    except AttributeError:
        warnings.warn(f'Alert of unexpected behavior! Core of type {type(core)} for some reason misses .forward method!')
        return False

SKIP_MAX = 1e3
ALL_IDS = list(np.arange(SKIP_MAX))
USED_IDS = []

class SkipLike(torch.nn.Module):
    def __init__(self, skip_from: int, skip_to: int, mode: Literal['i', 'a'], channels: Tuple[int]): # Tuple[int, int]
        super().__init__()

        assert mode in ['i', 'a', 'o'], f'Incompatible mode: expected "i", or "a", instead got {mode}!'
        self._from = skip_from  # -1 denotes, that inputs are takes from the original input data, 0 - from liftings, etc.
        self._to   = skip_to    # (layer, skip_loc)
        self._mode = mode
        self._channels = channels

        cur_availible = np.setdiff1d(ALL_IDS, USED_IDS)
        self._ID      = int(np.random.choice(cur_availible))

    @property
    def origin(self) -> int:
        return self._from

    @property
    def target(self) -> Tuple[int, str]:
        return (self._to, self._mode)

    def info(self):
        return (self._ID, self._from, self._to, self._mode) # super().__hash__()

    def __hash__(self):
        return hash(self.info())

    def forward(self, F: torch.Tensor, x: Union[torch.Tensor, TensorDict], output_shape: Tuple[int] = None) -> torch.Tensor:
        return F

def generateDefaultFiLMMapping(input_channels: Union[Tuple[int], List[int]],
                               output_channels: int, 
                               num_layers: int = 1,
                               layers_widths: Union[int, List[int]] = 10,
                               activation = torch.nn.GELU):
    if isinstance(layers_widths, int): # 'layers_widths must be passed as a list'
        layers_widths = [layers_widths,] * num_layers 
    else:
        assert len(layers_widths) == num_layers, 'length of layers_widths must be equal to the num_layers value'

    layers_widths.insert(0, len(input_channels))
    layers = []
    for i in range(num_layers):
        layers.append(torch.nn.Linear(layers_widths[i], layers_widths[i+1]))
        layers.append(activation())

    layers.append(torch.nn.Linear(layers_widths[-1], output_channels))
    return torch.nn.Sequential(*layers)

# DEFAULT_FILM = nn.Sequential(torch.nn.Linear(d_model, d_model * 2),
#                              nn.GELU(),
#                              nn.Linear(d_model * 2, d_model),

class FiLM(SkipLike):
    def __init__(self, mappings: Tuple[torch.nn.Module], skip_from: int,
                 skip_to: Tuple[int, int], mode: Literal['i', 'a'], channels: Tuple[int]):
        super().__init__(skip_from, skip_to, mode, channels)

        assert len(mappings) == 2, 'FiLM has to contain 2 modules: map., that produce weights and biases.'
        # TODO: implement assertions to check correctness of the mappings

        # FiLM(F_{i, c} | \gamma_{i, c}, \beta_{i, c}) = \gamma_{i, c} F_{i, c} + \beta_{i, c}
        self.gamma = mappings[0] 
        self.beta  = mappings[1]

    def forward(self, F: torch.Tensor, x: Union[torch.Tensor, TensorDict], output_shape: Tuple[int] = None):
        if isinstance(x, TensorDict):
            x = torch.concat([value for key, value in x.items() if int(key) in self._channels], dim = 1)
            assert x.shape[1] == len(self._channels), 'Mismatch in tensordict, passed in SkipLike.' # len(x.keys())

        gamma_val, beta_val = self.gamma(x), self.beta(x)

        # gamma_val & beta_val shapes: [B, C_{hidden}] + [1,] * (spatial_dim + time) as films are constant across the domain. 
        ax_diff = F.dim() - gamma_val.dim()
        for _ in range(ax_diff):
            gamma_val = gamma_val.unsqueeze(-1)
            beta_val  = beta_val.unsqueeze(-1)

        gamma_val = gamma_val.expand(*[-1, -1] + [F.shape[-idx] for idx in range(ax_diff, 0, -1)])
        beta_val  = beta_val.expand(*[-1, -1] + [F.shape[-idx] for idx in range(ax_diff, 0, -1)])

        F = gamma_val * F + beta_val
        if output_shape is not None:
            F = F.reshape(output_shape)

        return F


class StandardSkip(SkipLike):
    def __init__(self, mapping: Union[torch.nn.Module, List[torch.nn.Module]],
                 skip_from: int, skip_to: Tuple[int, int], mode: Literal['i', 'a'],
                 channels: Tuple[int], combinator: Callable = torch.add, combinator_args: tuple = None,
                 combinator_kwargs: dict = None) -> None:
        super().__init__(skip_from, skip_to, mode, channels)
        # TODO: add signature inspection for combinator function and a mapping  

        if combinator_args is None:
            combinator_args = ()

        if combinator_kwargs is None:
            combinator_kwargs = {}

        self._combinator_args = combinator_args
        self._combinator_kwargs = combinator_kwargs
        self.validateCombinator(combinator = combinator,
                                combinator_args = self._combinator_args,
                                combinator_kwargs = self._combinator_kwargs)

        if isinstance(mapping, torch.nn.Module):
            self._mapping = mapping
        else:
            self._mapping = torch.nn.Sequential(*mapping)

        self._combinator = combinator

    def forward(self, F: torch.Tensor, x: Union[torch.Tensor, TensorDict], output_shape: Tuple[int] = None) -> torch.Tensor:
        if isinstance(x, TensorDict):
            x = torch.concat([value for key, value in x.items() if int(key) in self._channels], dim = 1)
            assert x.shape[1] == len(self._channels), 'Mismatch in tensordict, passed in SkipLike' # len(x.keys())

        time_axis_dropped = False
        if isinstance(self._mapping, torch.nn.Conv2d) and x.ndim == 5:
            t_steps = x.shape[2]
            x = x[:, :, 0, :, :]
            time_axis_dropped = True
        elif isinstance(self._mapping, torch.nn.Conv3d) and x.ndim == 6:
            t_steps = x.shape[2]
            x = x[:, :, 0, :, :, :]
            time_axis_dropped = True
        else:
            print(f'No need to alter dims.: {self._mapping}: {isinstance(self._mapping, torch.nn.Conv2d)}, {x.ndim}')

        x = self._mapping(x)
        if time_axis_dropped:
            x = x.unsqueeze(2).repeat([1, 1, t_steps,] + [1]*(F.ndim - 3))

        ax_diff = F.dim() - x.dim()
        for _ in range(ax_diff):
            x = x.unsqueeze(-1)

        F = self._combinator(F, x, *self._combinator_args, **self._combinator_kwargs)
        if output_shape is not None:
           F = F.reshape(output_shape)

        return F

    @staticmethod
    def validateCombinator(combinator: Callable, combinator_args: tuple, combinator_kwargs: dict) -> None:
        assert inspect.isfunction(combinator) or isinstance(combinator, BuiltinFunctionType), \
            f'Combinator in skip is {type(combinator)}, while expected a function or builtin_function_or_method.'



# class SkipGenerator(object):
#     def __init__(self, pattern: dict): # Union[dict, SKIPS_TYPES]
#         if isinstance(pattern, str):
#             assert pattern in []

#     @classmethod
#     def fromPreset(self, pattern: SKIPS_TYPES):
