import warnings
import inspect

from typing import Tuple, Any, Callable, Literal, Final, List
from types import BuiltinFunctionType
from functools import singledispatch

import numpy as np

import torch

def checkSkipRequirements(core: torch.nn.Module) -> bool:
    try:
        sgn = inspect.signature(core.__class__.forward)
        # sgn.parameters.
    except AttributeError:
        warnings.warn(f'Alert of unexpected behavior! Core of type {type(core)} for some reason misses .forward method!')
        return False

# class GenericSkip(torch.nn.Module):
#     def __init__(self, layer_from: int, layer_to: int):
#         super().__init__()

SKIP_MAX = 1e3
ALL_IDS = list(np.arange(SKIP_MAX))
USED_IDS = []

class SkipLike(torch.nn.Module):
    def __init__(self, skip_from: int, skip_to: Tuple[int, int], mode: Literal['i', 'a']):
        super().__init__()

        assert mode in ['i', 'a', 'o'], f'Incompatible mode: expected "i", or "a", instead got {mode}!'
        self._from = skip_from  # -1 denotes, that inputs are takes from the original input data, 0 - from liftings, etc.
        self._to   = skip_to    # (layer, skip_loc)
        self._mode = mode

        cur_availible = np.setdiff1d(ALL_IDS, USED_IDS)
        self._ID      = np.random.choice(cur_availible)

    def __hash__(self):
        return (self._ID, self._from, self._to) # super().__hash__()

def generateDefaultFiLMMapping(scalars_num: int,
                               output_channels: int, 
                               num_layers: int = 1,
                               layers_widths: List[int] = [10,],
                               activation = torch.nn.GELU,):
    assert isinstance(layers_widths, list), 'layers_widths must be passed as a list'
    assert len(layers_widths) == num_layers, 'length of layers_widths must be equal to the num_layers value'

    layers_widths.insert(0, scalars_num)
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
                 skip_to: Tuple[int, int], mode: Literal['i', 'a']):
        super().__init__(skip_from, skip_to, mode)

        assert len(mappings) == 2, 'FiLM has to contain 2 modules: map., that produce weights and biases.'
        # TODO: implement assertions to check correctness of the mappings

        # FiLM(F_{i, c} | \gamma_{i, c}, \beta_{i, c}) = \gamma_{i, c} F_{i, c} + \beta_{i, c}
        self.gamma = mappings[0] 
        self.beta  = mappings[1]

    def forward(self, F: torch.Tensor, x: torch.Tensor, output_shape: Tuple[int] = None):
        gamma_val, beta_val = self.gamma(x), self.beta(x)

        # gamma_val & beta_val shapes: [B, C_{hidden}] + [1,] * (spatial_dim + time) as films are constant across the domain. 
        ax_diff = F.dim() - gamma_val.dim()
        for _ in range(ax_diff):
            gamma_val = gamma_val.unsqueeze(-1)
            beta_val  = beta_val.unsqueeze(-1)

        gamma_val = gamma_val.view_as(F)
        beta_val  = beta_val.view_as(F)

        F = gamma_val * F + beta_val
        if output_shape is not None:
            F = F.reshape(output_shape)

        return F


class StandardSkip(SkipLike):
    def __init__(self, mapping: torch.nn.Module, skip_from: int, skip_to: Tuple[int, int], mode: Literal['i', 'a'],
                 channels: Tuple[int], combinator: Callable = torch.add, combinator_args: tuple = None,
                 combinator_kwargs: dict = None) -> None:
        super().__init__(skip_from, skip_to, mode)
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

        self._mapping = mapping
        self._combinator = combinator
        self._channels = channels

    def forward(self, F: torch.Tensor, x: torch.Tensor, output_shape: Tuple[int] = None) -> torch.Tensor:
        x = self._mapping(x)
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


NAMED_SKIP_MAPS: Final = ('skip', 'dno') # TODO: unet, cno
# SKIPS_TYPES = Literal[NAMED_SKIP_MAPS]

def generateDNOSkips(model: torch.nn.Module, **kwargs) -> List[SkipLike]:
    from muno.layers.embeddings import GridEmbeddingND
    assert 'grid_channels' in kwargs.keys(), 'generateDNOskip requires explicitly set geometry channels'

    skips = []
    try:
        model.n_layers
    except AttributeError:
        raise RuntimeError(f'Incorrect model loaded into DNO skip generator: expected something like FNO, instead got {type(model)}')

    for i in range(model.n_layers):
        skips.append(StandardSkip(torch.nn.Identity, -2, i, mode = 'i', channels = kwargs['grid_channels'], ))

    common_embedding = GridEmbeddingND()
    for i in range(model.n_layers):
        skips.append(StandardSkip(common_embedding, -2, i, mode = 'i', channels = kwargs['grid_channels'], ))

    return skips

def generateFiLMSkips(model: torch.nn.Module, **kwargs):
    film_scalar_inputs = kwargs.get('scalar_inputs', ())

    assert isinstance(film_scalar_inputs, tuple) and all([isinstance(inp_idx, int) for inp_idx in film_scalar_inputs]), \
        f'Argument film_scalar_inputs have to be passed as a tuple of integers, instead got {type(film_scalar_inputs)}'
    
    if len(film_scalar_inputs) == 0:
        warnings.warn('FiLM inputs have not been initialized due to absence of arguments.')
        return []

    try:
        model.n_layers
    except AttributeError:
        raise RuntimeError(f'Incorrect model loaded into DNO skip generator: expected something like FNO, instead got {type(model)}')

    skips = []
    film_gen_func   = kwargs.get('film_gen_func', generateDefaultFiLMMapping)
    film_gen_kwargs = kwargs.get('film_gen_kwargs', generateDefaultFiLMMapping)

    REQUIRED_FILM_ARGS = ['scalars_num', 'num_layers', 'layers_widths']  # 'output_channels', 
    # Do not mistake num_layers and layers_widths of the FiLM mapping with neural operators'
    # 'output_channels' are expected to be parsed from the model's layer  
    
    assert isinstance(film_gen_kwargs, dict), \
        f'Mandatory film_gen_kwargs argument is not a dict, as it must be, but {type(film_gen_kwargs)}.'
    assert all([arg in film_gen_kwargs.keys() for arg in REQUIRED_FILM_ARGS]), \
        f'Required FiLM args are missing: expected {REQUIRED_FILM_ARGS}, instead got {film_gen_kwargs.keys()} keys.'

    def inspectGenFuncArgs(func: Any):
        sig: inspect.Signature = inspect.signature(func)
        return all([arg in sig.parameters for arg in REQUIRED_FILM_ARGS])

    assert inspect.isfunction(film_gen_func) and inspectGenFuncArgs(film_gen_func)
    hidden_channels = model.hidden_channels

    for i in range(model.n_layers):
        skips.append(FiLM((film_gen_func(scalars_num     = film_gen_kwargs["scalars_num"],
                                         output_channels = hidden_channels,               # film_gen_kwargs["output_channels"],
                                         num_layers      = film_gen_kwargs["num_layers"],
                                         layers_widths   = film_gen_kwargs["layers_widths"]),
                           film_gen_func(scalars_num     = film_gen_kwargs["scalars_num"],
                                         output_channels = hidden_channels,               # film_gen_kwargs["output_channels"],
                                         num_layers      = film_gen_kwargs["num_layers"],
                                         layers_widths   = film_gen_kwargs["layers_widths"])),
                          skip_from=-2, skip_to=i))
    

@singledispatch
def skipGeneration(pattern, model: torch.nn.Module) -> List[SkipLike]: #  Union[dict, SKIPS_TYPES]
    raise NotImplementedError(f'Unsupported type of skip patterns: expected dict or str, got {type(pattern)}: {pattern}.')

@singledispatch
def skipGeneration(pattern: str, model: torch.nn.Module) -> List[SkipLike]:
    assert pattern in NAMED_SKIP_MAPS, \
        f'Incorrect type string, expected something from {NAMED_SKIP_MAPS}, instead got {pattern}.'
    match pattern:
        case 'skip':
            return "Success"
        case _:
            warnings.warn(f"Incorrect string of pattern: {pattern}")
            return "Unknown Status"


# class SkipGenerator(object):
#     def __init__(self, pattern: dict): # Union[dict, SKIPS_TYPES]
#         if isinstance(pattern, str):
#             assert pattern in []

#     @classmethod
#     def fromPreset(self, pattern: SKIPS_TYPES):
