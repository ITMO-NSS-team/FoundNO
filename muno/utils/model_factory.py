import importlib.util
import inspect

import glob
import dill
from functools import singledispatch

from pathlib import Path
import warnings

from typing import Union, List, Final, Any

import torch

from neuralop.layers.channel_mlp import ChannelMLP
# from neuralop.layers.spectral_convolution import SpectralConv
from muno.layers.spectral_convolution import SpectralConv
from neuralop.models import UNO #, FNO

from muno.utils.training_utils import validateOperator

from muno.layers.skips import SkipLike, StandardSkip, FiLM, generateDefaultFiLMMapping
from muno.layers.channel_wise_conv import FactorizedDimensionSpectralConv
from muno.layers.embeddings import GridEmbeddingND
from muno.models.muno import Muno
from muno.models.fno import FNO
from muno.models.cno import CFNO

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_class_from_file(relative_path, class_name):
    module_path = PROJECT_ROOT / relative_path
    module_name = f"_foundno_dynamic_{module_path.stem}_{class_name.lower()}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _post_lift_mamba_fno3d():
    return _load_class_from_file("muno/models/mamba_fno.py", "PostLiftMambaFNO3D")


def _post_lift_mamba_lifting():
    return _load_class_from_file("muno/models/mamba_fno.py", "PostLiftMambaLifting")


def _local_attn_fno():
    return _load_class_from_file("muno/models/localattn_exp.py", "LocalAttnFNO")


def _pecoda_no():
    return _load_class_from_file("muno/models/pecoda.py", "PeCODANO")


MODEL_REGISTRY = {
    "fno": {
        "kind": "single",
        "model": UNO,
        "params": {
            "hidden_channels": 16,
            "n_layers": 5,
            "uno_n_modes": [[20, 40, 40]] * 5,
            "uno_out_channels": [16, 32, 32, 32, 16],
            "uno_scalings": [
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
            "non_linearity": torch.nn.functional.gelu,
            "horizontal_skips_map": {4: 0, 3: 1},
            "channel_mlp_skip": "linear",
        },
    },
    "mambafno": {
        "kind": "single",
        "model": _post_lift_mamba_fno3d,
        "params": {
            "modes": (20, 40, 40),
            "width": 65,
            "n_layers": 4,
            "use_mamba_kwargs": None,
            "mamba_fallback_kernel": 9,
        },
    },
    "localattnfno": {
        "kind": "single",
        "model": _local_attn_fno,
        "params": {
            "width": 64,
            "n_local_layers": 2,
            "n_heads": 4,
            "window_size": 127,
        },
    },
    "pecoda": {
        "kind": "single",
        "model": _pecoda_no,
        "params": {
            "hidden_variable_codimension": 16,
            "n_layers": 2,
            "n_layers_fno": 2,
            "n_modes": [[64, 64], [64, 64], 64, 64],
        },
    },
    "adapted_fno": {
        "kind": "adapter_core_adapter",
        "model": [_post_lift_mamba_lifting, FNO, ChannelMLP],
        "params": [
            {
                "width": 32,
                "use_mamba_kwargs": None,
                "mamba_fallback_kernel": 9,
                "padding": 0,
                "n_dim": 3,
                "non_linearity": torch.nn.functional.gelu,
            },
            {
                "hidden_channels": 32,
                "n_layers": 4,
                "n_modes": {"t": 1, "x": 32}, # [10, 40, 40],
                "disable_lifting_and_projection": True,
                "conv_module": SpectralConv # FactorizedDimensionSpectralConv # SpectralConv # 
            }, 
            {
                "hidden_channels": 32,
                "n_layers": 2,
                "n_dim": 3,
                "non_linearity": torch.nn.functional.gelu,
            },
        ],
    },
    "adapted_cfno": {
        "kind": "adapter_core_adapter",
        "model": [_post_lift_mamba_lifting, CFNO, ChannelMLP],
        "params": [
            {
                "width": 80,
                "use_mamba_kwargs": None,
                "mamba_fallback_kernel": 9,
                "padding": 0,
                "n_dim": 3,
                "non_linearity": torch.nn.functional.gelu,
            },
            {
                "hidden_channels": 80,
                "n_layers": 4,
                "n_modes": {"t": 10, "x": 32}, # [10, 40, 40],
                "disable_lifting_and_projection": True,
                "local_branch": "parallel",
                "diff_kernels": "parallel",
                "arch": "temporal",
                "conv_module": SpectralConv
            }, 
            {
                "hidden_channels": 80,
                "n_layers": 2,
                "n_dim": 3,
                "non_linearity": torch.nn.functional.gelu,
            },
        ],
    },    
    "adapted_fno_no_mamba": {
        "kind": "adapter_core_adapter",
        "model": [ChannelMLP, FNO, ChannelMLP],
        "params": [
            {
                "hidden_channels": 32,
                "n_layers": 2,
                "n_dim": 3,
                "non_linearity": torch.nn.functional.gelu,
            },
            {
                "hidden_channels": 32,
                "n_layers": 4,
                "n_modes": {"t": 10, "x": 32}, # [20, 42, 42],
                "disable_lifting_and_projection": True,
                "conv_module": SpectralConv # FactorizedDimensionSpectralConv # SpectralConv # 
            },
            {
                "hidden_channels": 32,
                "n_layers": 2,
                "n_dim": 3,
                "non_linearity": torch.nn.functional.gelu,
            },
        ],
    },
    "dno": {
        "kind": "adapter_core_adapter",
        "model": [ChannelMLP, FNO, ChannelMLP],
        "params": [
            {
                "hidden_channels": 32,
                "n_layers": 2,
                "n_dim": 3,
                "non_linearity": torch.nn.functional.gelu,
            },
            {
                "hidden_channels": 32,
                "n_layers": 4,
                "n_modes": {"t": 10, "x": 32}, # [20, 42, 42],
                "disable_lifting_and_projection": True,
                "conv_module": SpectralConv # FactorizedDimensionSpectralConv
            },
            {
                "hidden_channels": 32,
                "n_layers": 2,
                "n_dim": 3,
                "non_linearity": torch.nn.functional.gelu,
            },
        ],
    },    
}


def available_models():
    return sorted(MODEL_REGISTRY)


def _merge_params(default_params, override_params):
    params = dict(default_params)
    if override_params:
        params.update(override_params)
    return params


def _filter_init_params(model_cls, params):
    signature = inspect.signature(model_cls.__init__)
    parameters = signature.parameters.values()

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return dict(params)

    allowed = set(signature.parameters) - {"self"}
    return {
        name: value
        for name, value in params.items()
        if name in allowed
    }


def _resolve_model_config(model_config):
    model_name = model_config.get("type", model_config.get("name", "adapted_fno_no_mamba"))

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models()}"
        )

    spec = MODEL_REGISTRY[model_name]

    resolved = {
        "name": model_name,
        "kind": spec["kind"],
        "model": spec["model"],
    }

    if spec["kind"] == "adapter_core_adapter":
        param_overrides = model_config.get("params")
        if param_overrides is None:
            param_overrides = [{}, {}, {}]

        if isinstance(param_overrides, dict):
            param_overrides = [
                param_overrides.get("lifting", {}),
                param_overrides.get("core", {}),
                param_overrides.get("projection", {}),
            ]

        if len(param_overrides) != 3:
            raise ValueError(
                "Adapter-core-adapter model params must contain exactly 3 blocks: "
                "lifting, core, projection"
            )

        resolved["params"] = [
            _merge_params(default, override)
            for default, override in zip(spec["params"], param_overrides)
        ]

        hidden_channels = model_config.get("hidden_channels")
        if hidden_channels is not None:
            if "hidden_channels" in resolved["params"][0]:
                resolved["params"][0]["hidden_channels"] = hidden_channels
            if "width" in resolved["params"][0]:
                resolved["params"][0]["width"] = hidden_channels
            resolved["params"][1]["hidden_channels"] = hidden_channels
            resolved["params"][2]["hidden_channels"] = hidden_channels

        n_dim = model_config.get("n_dim")
        if n_dim is not None:
            resolved["params"][0]["n_dim"] = n_dim
            resolved["params"][2]["n_dim"] = n_dim

        n_modes = model_config.get("n_modes")
        if n_modes is not None:
            resolved["params"][1]["n_modes"] = n_modes

        if "core_layers" in model_config:
            resolved["params"][1]["n_layers"] = model_config["core_layers"]
        if "lifting_layers" in model_config:
            resolved["params"][0]["n_layers"] = model_config["lifting_layers"]
        if "projection_layers" in model_config:
            resolved["params"][2]["n_layers"] = model_config["projection_layers"]

    else:
        resolved["params"] = _merge_params(spec["params"], model_config.get("params"))

    return resolved


def get_all_files(dir: str, file_type: str = '.pt'):
    return sorted(glob.glob(dir + "/*" + file_type))


def load_from_dir(dir: str, SAVE_LOAD_ARGS = None):
    if SAVE_LOAD_ARGS is None:
        SAVE_LOAD_ARGS = {}

    files = get_all_files(dir)
    print('loading from {}'.format(files))
    return [torch.load(file, pickle_module=dill, **SAVE_LOAD_ARGS) for file in files]
   

def build_model(loader_channels, model_config, 
                pretr_core: torch.nn.Module = None,
                pretr_liftings: List[torch.nn.Module] = None,
                pretr_projections: List[torch.nn.Module] = None):
    resolved = _resolve_model_config(model_config or {})

    if resolved["kind"] == "single":
        if len(loader_channels) != 1:
            raise ValueError(
                f"Model '{resolved['name']}' is a single-model architecture and can be used only "
                f"with one task/loader. For multiphysics training use 'adapted_fno' or "
                f"'adapted_fno_no_mamba'. Got {len(loader_channels)} loaders."
            )

        model_cls = resolved["model"]
        if not isinstance(model_cls, type):
            model_cls = model_cls()
        params = _filter_init_params(model_cls, resolved["params"])
        validateOperator(model_cls, ["in_channels", "out_channels"] + list(params.keys()))

        if isinstance(loader_channels[0], int):
            in_channels, out_channels = loader_channels[0]
        else:    
            in_channels, out_channels = loader_channels[0][0], loader_channels[0][1]

        return model_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            **params,
        )

    if resolved["kind"] == "adapter_core_adapter":
        model_classes = resolved["model"]
        params = resolved["params"]

        lifting_cls, core_cls, projection_cls = model_classes
        if isinstance(lifting_cls, list):
            if not isinstance(lifting_cls[0], type):
                lifting_cls[0] = lifting_cls[0]()
            if not isinstance(lifting_cls[1], type):
                lifting_cls[1] = lifting_cls[1]()
        else:
            if not isinstance(lifting_cls, type):
                lifting_cls = lifting_cls()

        if not isinstance(core_cls, type):
            core_cls = core_cls()
        if not isinstance(projection_cls, type):
            projection_cls = projection_cls()

        lifting_params, core_params, projection_params = params

        if core_params["conv_module"] == FactorizedDimensionSpectralConv:
            hidden_channels = 2 * core_params["hidden_channels"]
        else:
            hidden_channels = core_params["hidden_channels"]

        liftings = []
        projections = []

        for in_channels, out_channels in loader_channels:
            current_lifting_params = dict(lifting_params)
            if not isinstance(lifting_cls, list) and lifting_cls.__name__ == "PostLiftMambaLifting":
                current_lifting_params.pop("hidden_channels", None)
            current_lifting_params = _filter_init_params(lifting_cls, current_lifting_params)

            if isinstance(lifting_cls, list):
                assert lifting_cls[0] == GridEmbeddingND, \
                    'Multiple models in projections are allowed only in a scenario, where the 1st model is GridEmbeddingND'
                if "grid_boundaries" in current_lifting_params.keys():
                    bnds = current_lifting_params.pop("grid_boundaries", None)
                else:
                    bnds = [0, 1]
                NDIM = 2

                lft = torch.nn.Sequential([lifting_cls[0](NDIM, bnds), 
                                           lifting_cls[1](in_channels  =in_channels, 
                                                          out_channels =hidden_channels,
                                                          **current_lifting_params)])
            else:
                lft = lifting_cls(in_channels=in_channels, out_channels=hidden_channels, **current_lifting_params)

            liftings.append(lft)

            current_projection_params = _filter_init_params(projection_cls, projection_params)
            projections.append(
                projection_cls(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    **current_projection_params,
                )
            )

        if pretr_liftings is not None or pretr_projections is not None:
            assert pretr_liftings is not None and pretr_projections is not None, \
                'If build_model gets pretrained adapters, both liftings and projections have to be passed!'
            assert len(pretr_liftings) == len(pretr_projections), 'Incosistent lengths of liftings and projections.'
            assert len(pretr_liftings) == len(liftings), 'Number of passed liftings does not match the problem.'

            for ad_idx, _ in enumerate(liftings):
                if liftings[ad_idx].state_dict().keys() != pretr_liftings[ad_idx].state_dict().keys():
                    warnings.warn(f'Parameter dict of pretr. lifting {ad_idx} does not match the one, set in config. \
                                    Defaulting to the passed one.')
                    liftings[ad_idx] = pretr_liftings[ad_idx]
                else:
                    try:
                        liftings[ad_idx].load_state_dict(pretr_liftings[ad_idx].state_dict())
                    except:
                        warnings.warn(f'Parameter dict of pretr. lifting {ad_idx} does not match the one, set in config. \
                                        Defaulting to the passed one, despite matching state_dict keys.')
                        liftings[ad_idx] = pretr_liftings[ad_idx]

                if projections[ad_idx].state_dict().keys() != pretr_projections[ad_idx].state_dict().keys():
                    warnings.warn(f'Parameter dict of pretr. proj. {ad_idx} does not match the one, set in config. \
                                    Defaulting to the passed one.')
                    projections[ad_idx] = pretr_projections[ad_idx]
                else:
                    try:
                        projections[ad_idx].load_state_dict(pretr_projections[ad_idx].state_dict())
                    except:
                        warnings.warn(f'Parameter dict of pretr. proj. {ad_idx} does not match the one, set in config. \
                                        Defaulting to the passed one, despite matching state_dict keys.')
                        projections[ad_idx] = pretr_projections[ad_idx]

                

        current_core_params = _filter_init_params(core_cls, core_params)
        core = core_cls(in_channels=core_params["hidden_channels"],
                        out_channels=core_params["hidden_channels"],
                        **current_core_params,
                        )

        if pretr_core is not None:
                if core.state_dict().keys() != pretr_core.state_dict().keys():
                    warnings.warn(f'Parameter dict of the passed pretrained core does not match the one, set in config. \
                                    Defaulting to the passed one.')
                    core = pretr_core
                else:
                    try:
                        core.load_state_dict(pretr_core.state_dict())
                    except:
                        warnings.warn(f'Parameter dict of the passed pretrained core does not match the one, set in config. \
                                        Defaulting to the passed one, despite matching state_dict keys.')
                        core = pretr_core


        return liftings, core, projections

    raise ValueError(f"Unsupported model kind: {resolved['kind']}")


NAMED_SKIP_MAPS: Final = ('skip', 'dno') # TODO: unet, cno
# SKIPS_TYPES = Literal[NAMED_SKIP_MAPS]

def generateDNOSkips(model: torch.nn.Module, **kwargs) -> List[SkipLike]: # core_is_factorized: bool = True, 
    # from muno.layers.embeddings import GridEmbeddingND
    assert 'grid_channels' in kwargs.keys(), 'generateDNOskip requires explicitly set geometry channels'

    skips = []
    try:
        if isinstance(model, Muno):
            n_layers        = model._core.n_layers
            hidden_channels = model._core.hidden_channels
            if (model._core.fno_blocks.convSignature[0] == FactorizedDimensionSpectralConv and 
                model._core.fno_blocks.convSignature[1]):
                hidden_channels *= 2 # TODO: add variable hc multiplier, with respect to proj. dim
        else:
            n_layers        = model.n_layers
            hidden_channels = model.hidden_channels
            if (model.fno_blocks.convSignature[0] == FactorizedDimensionSpectralConv and 
                model.fno_blocks.convSignature[1]):
                hidden_channels *= 2 # TODO: add variable hc multiplier, with respect to proj. dim

    except AttributeError:
        raise RuntimeError(f'Incorrect model loaded into DNO skip generator: expected something like FNO, instead got {type(model)}')

    for i in range(n_layers):
        skips.append(StandardSkip(torch.nn.Conv2d(len(kwargs['grid_channels']), hidden_channels, 1), -2, i, # torch.nn.Identity()
                                  mode = 'i', channels = kwargs['grid_channels'], ))

    for i in range(n_layers):
        skips.append(StandardSkip(torch.nn.Conv2d(len(kwargs['grid_channels']), hidden_channels, 1), # [common_embedding, ],
                                  -2, i, mode = 'i', channels = kwargs['grid_channels'], ))

    return skips

def generateFiLMSkips(model: torch.nn.Module, **kwargs):
    film_scalar_inputs = kwargs.get('scalar_inputs', ())
    # Tuple of channel-like entries of the film layers from the skips tensordict (keys) or tensor (idxs along axis №1)

    assert isinstance(film_scalar_inputs, tuple) and all([isinstance(inp_idx, int) for inp_idx in film_scalar_inputs]), \
        f'Argument film_scalar_inputs have to be passed as a tuple of integers, instead got {type(film_scalar_inputs)}'
    
    # if len(input_channels) == 0:
    #     warnings.warn('FiLM inputs have not been initialized due to absence of arguments.')
    #     return []

    try:
        if isinstance(model, Muno):
            n_layers = model._core.n_layers
        else:
            n_layers = model.n_layers
    except AttributeError:
        raise RuntimeError(f'Incorrect model loaded into DNO skip generator: expected something like FNO, instead got {type(model)}')

    skips = []
    film_gen_func   = kwargs.get('film_gen_func', generateDefaultFiLMMapping)
    film_gen_kwargs = kwargs.get('film_gen_kwargs', {}) # "input_channels": 0

    REQUIRED_FILM_ARGS = ['input_channels', 'num_layers', 'layers_widths']  # 'output_channels', 
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

    if isinstance(model, Muno):
        hidden_channels = model._core.hidden_channels
        if (model._core.fno_blocks.convSignature[0] == FactorizedDimensionSpectralConv and 
            model._core.fno_blocks.convSignature[1]):
            hidden_channels *= 2        
    else:
        hidden_channels = model.hidden_channels
        if (model.fno_blocks.convSignature[0] == FactorizedDimensionSpectralConv and 
            model.fno_blocks.convSignature[1]):
            hidden_channels *= 2

    for i in range(n_layers):
        skips.append(FiLM((film_gen_func(input_channels  = film_gen_kwargs["input_channels"],
                                         output_channels = hidden_channels,               # film_gen_kwargs["output_channels"],
                                         num_layers      = film_gen_kwargs["num_layers"],
                                         layers_widths   = film_gen_kwargs["layers_widths"]),
                           film_gen_func(input_channels  = film_gen_kwargs["input_channels"],
                                         output_channels = hidden_channels,               # film_gen_kwargs["output_channels"],
                                         num_layers      = film_gen_kwargs["num_layers"],
                                         layers_widths   = film_gen_kwargs["layers_widths"])),
                          skip_from=-2, skip_to=i, mode = 'a', 
                          channels = film_gen_kwargs["input_channels"]))
    return skips
    

@singledispatch
def skipGeneration(pattern, model: torch.nn.Module, **kwargs) -> List[SkipLike]: #  Union[dict, SKIPS_TYPES]
    raise NotImplementedError(f'Unsupported type of skip patterns: expected dict or str, got {type(pattern)}: {pattern}.')

@skipGeneration.register
def _(pattern: str, model: torch.nn.Module, **kwargs) -> List[SkipLike]:
    assert pattern in NAMED_SKIP_MAPS, \
        f'Incorrect type string, expected something from {NAMED_SKIP_MAPS}, instead got {pattern}.'
    
    match pattern:
        case 'film':
            for argname in ['scalars_num', 'num_layers', 'layers_widths']:
                assert argname in kwargs.keys(), \
                    f'Missing {argname} argument from kwargs for skip generation.'

            return generateFiLMSkips(model, **kwargs)
        case 'skips':
            for argname in ['grid_channels',]:
                assert argname in kwargs.keys(), \
                    f'Missing {argname} argument from kwargs for skip generation.'

            return generateDNOSkips(model, **kwargs)
        case _:
            warnings.warn(f"Incorrect string of pattern: {pattern}")
            return "Unknown Status"


def passModelToDevice(model: Union[torch.nn.Module, torch.nn.DataParallel, tuple], device: str = 'cuda') \
    -> Union[torch.nn.Module, torch.nn.DataParallel, tuple]:
    if isinstance(model, (torch.nn.Module, torch.nn.DataParallel)):
        model.to(device)
    else:
        assert len(model) == 3, \
            f'Model must have structure of a list/tuple of 3 elems: liftings-core-projections, instead got {len(model)}.'
        if model[0] is None or model[1] is None or model[2] is None:
            raise AttributeError('Hidden Fourier NO layers and projection or liftings are not yet declared.')

        model[1].to(device)
        for idx, _ in enumerate(model[0]):
            model[0][idx].to(device)
            model[2][idx].to(device)

    return model


def addSkips(model: torch.nn.Module, skips_pattern):
    skips = skips_pattern.create_skips(model)
    model.addSkips(skips)