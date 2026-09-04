import importlib.util
import inspect

import glob
import dill

from pathlib import Path
import warnings

from typing import Union, List

import torch

from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.models import UNO, FNO

from muno.utils.training_utils import validateOperator


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
                "width": 20,
                "use_mamba_kwargs": None,
                "mamba_fallback_kernel": 9,
                "padding": 0,
                "n_dim": 3,
                "non_linearity": torch.nn.functional.gelu,
            },
            {
                "hidden_channels": 20,
                "n_layers": 4,
                "n_modes": [10, 40, 40],
                "disable_lifting_and_projection": True,
            },
            {
                "hidden_channels": 20,
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
                "n_modes": [20, 42, 42],
                "disable_lifting_and_projection": True,
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
    return glob.glob(dir + "/*" + file_type)


def load_from_dir(dir: str, SAVE_LOAD_ARGS = {}):
    files = get_all_files(dir) # glob.glob(dir + "/*.pt")
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

        in_channels, out_channels = loader_channels[0]
        return model_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            **params,
        )

    if resolved["kind"] == "adapter_core_adapter":
        model_classes = resolved["model"]
        params = resolved["params"]

        lifting_cls, core_cls, projection_cls = model_classes
        if not isinstance(lifting_cls, type):
            lifting_cls = lifting_cls()
        if not isinstance(core_cls, type):
            core_cls = core_cls()
        if not isinstance(projection_cls, type):
            projection_cls = projection_cls()

        lifting_params, core_params, projection_params = params

        hidden_channels = core_params["hidden_channels"]

        liftings = []
        projections = []

        for in_channels, out_channels in loader_channels:
            current_lifting_params = dict(lifting_params)
            if lifting_cls.__name__ == "PostLiftMambaLifting":
                current_lifting_params.pop("hidden_channels", None)
            current_lifting_params = _filter_init_params(lifting_cls, current_lifting_params)

            liftings.append(
                lifting_cls(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    **current_lifting_params,
                )
            )

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
                    liftings[ad_idx].load_state_dict(pretr_liftings[ad_idx].state_dict())

                if projections[ad_idx].state_dict().keys() != pretr_projections[ad_idx].state_dict().keys():
                    warnings.warn(f'Parameter dict of pretr. proj. {ad_idx} does not match the one, set in config. \
                                    Defaulting to the passed one.')
                    projections[ad_idx] = pretr_projections[ad_idx]
                else:
                    projections[ad_idx].load_state_dict(pretr_projections[ad_idx].state_dict())
                

        current_core_params = _filter_init_params(core_cls, core_params)
        core = core_cls(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            **current_core_params,
        )

        if pretr_core is not None:
                if core.state_dict().keys() != pretr_core.state_dict().keys():
                    warnings.warn(f'Parameter dict of the passed pretrained core does not match the one, set in config. \
                                    Defaulting to the passed one.')
                    core = pretr_core
                else:
                    core.load_state_dict(pretr_core.state_dict())

        return liftings, core, projections

    raise ValueError(f"Unsupported model kind: {resolved['kind']}")


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


# def modelToDefaultParallel(model: torch.nn.Module, devices: Union[int, List[int]] = []) -> torch.nn.Module:
#     if isinstance(devices, (list, tuple)) and len(devices) == 0:
#         if torch.cuda.is_available():
#             passModelToDevice(model, 'cuda')
#             return model
#         else:
#             raise RuntimeError('For some reasons the module is not able to access pytorch cuda.')

#     if isinstance(model, torch.nn.Module):
#         model = torch.nn.DataParallel(model, device_ids=devices)
#     elif isinstance(model, tuple):
#         if not (isinstance(model[0], list) and isinstance(model[1], torch.nn.Module) and isinstance(model[2], list)):
#             raise RuntimeError('Adapted model does not respect requred format of liftings-core-projections.')

#         adapters, projections = [], []
#         core = torch.nn.DataParallel(model[1], device_ids=devices)
#         for idx, _ in enumerate(model[0]):
#             model[0][idx].to(device)
#             model[2][idx].to(device)

#         model = (adapters, core, projections)
         
#     passModelToDevice(model, device=devices) if isinstance(devices, int) else passModelToDevice(model, device=devices[0])
    
#     return model