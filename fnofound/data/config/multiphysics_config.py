from typing import Dict, Any, List, Optional, Union

from zencfg import ConfigBase
from fnofound.data.config.distributed import DistributedConfig
from fnofound.data.config.models import ModelConfig, FNO_Small2d
from fnofound.data.config.opt import OptimizationConfig, PatchingConfig
from fnofound.data.config.wandb import WandbConfig


class MultiphysicsOptConfig(OptimizationConfig):
    n_epochs: int = 300
    learning_rate: float = 5e-3
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 60
    gamma: float = 0.5


# class AdvectionDatasetConfig(ConfigBase):
#     file_name: str = 'advection_1d_8_9_9.hdf5'
#     train_resolution: int = 32
#     test_resolutions: List[int] = [32]
#     test_batch_sizes: List[int] = [16]
#     n_tests: List[int] = [400]
#     spatial_subsample: Optional[int] = None
#     temporal_subsample: Optional[int] = None
#     interpolate_mode: Optional[str] = None  # 'nearest'
#     encode_input: bool = True
#     encode_output: bool = True
#     include_endpoint: List[bool] = [True, False]
#     download_params: Dict[str, Any] = {
#         'source': '',
#         'dataset_id': '',
#         'download': False
#     }


class BurgersDatasetConfig(ConfigBase):
    file_train: str = 'burgers_train_16.pt'
    file_test: str = 'burgers_test_16.pt'
    channel_dim: int = 1
    custom_in_channels: dict = {'x': [0, 1]}
    # train_resolution: Optional[int] = None  # int = 16
    train_resolution: Union[int, dict] = {
        "output": (16, 32),
        "input": 32
    }
    # test_resolutions: List[int] = [32]
    test_resolutions: Union[int, dict] = {
        "output": (16, 32),
        "input": 32
    }
    test_batch_sizes: List[int] = [16]
    n_tests: List[int] = [400]
    spatial_subsample: Optional[int] = None
    temporal_subsample: Optional[int] = None
    interpolate_mode: Optional[str] = None  # 'nearest'
    encode_input: bool = True
    encode_output: bool = True
    include_endpoint: List[bool] = [True, False]
    download_params: Dict[str, Any] = {
        'source': 'pdebench',
        'dataset_id': '268190',
        'download': False
    }


class DarcyDatasetConfig(ConfigBase):
    file_train: str = 'darcy_train_16.pt'
    file_test: str = 'darcy_test_16.pt'
    channel_dim: int = 1
    custom_in_channels: dict = {'x': [0, 1], 'y': [-1, 1]}
    # train_resolution: Optional[int] = None  # int = 16
    train_resolution: Union[int, dict] = {
        "output": (16, 32),
        "input": (16, 32),
    }
    # test_resolutions: List[int] = [32]
    test_resolutions: Union[int, dict] = {
        "output": (16, 32),
        "input": (16, 32)
    }
    test_batch_sizes: List[int] = [16]
    n_tests: List[int] = [100]
    spatial_subsample: Optional[int] = None
    temporal_subsample: Optional[int] = None
    interpolate_mode: Optional[str] = None  # 'nearest'
    encode_input: bool = True
    encode_output: bool = True
    download_params: Dict[str, Any] = {
        'source': 'zenodo',
        'dataset_id': '12784353',
        'download': False
    }


class ReactionDiffusionDatasetConfig(ConfigBase):
    file_train: str = 'ReacDiff_Nu0.5_Rho1.0.hdf5'
    file_test: str = 'ReacDiff_react_Nu5.0_Rho5.0.hdf5'
    channel_dim: int = 1
    custom_in_channels: dict = {'x': 'data'}
    # train_resolution: Optional[int] = None  # int = 16
    train_resolution: Union[int, dict] = {
        "output": (102, 1024), # "output": (102, 512), from (102, 1024)
        "input": (102, 1024),  # "input": (1, 128),  from (1, 1024)
    }
    # test_resolutions: List[int] = [32]
    test_resolutions: Union[int, dict] = {
        "output": (102, 1024),  # "output": (102, 512), from (102, 1024)
        "input": (102, 1024)  # "input": (1, 128),  from (1, 1024)
    }
    test_batch_sizes: List[int] = [16]
    n_tests: List[int] = [400]
    spatial_subsample: Optional[int] = None
    temporal_subsample: Optional[int] = None
    interpolate_mode: Optional[str] = None  # 'nearest'
    encode_input: bool = True
    encode_output: bool = True
    include_endpoint: List[bool] = [True, False]
    download_params: Dict[str, Any] = {
        'source': 'pdebench',
        'dataset_id': '268190',
        'download': False
    }


class MultiphysicsDatasetConfig(ConfigBase):
    folder: str = 'neuralop/data/datasets/data/'
    batch_size: int = 8
    n_train: int = 1000
    datasets: Dict[str, Any] = {
        # 'advection': AdvectionDatasetConfig(),
        # 'burgers': BurgersHDF5DatasetConfig(),
        'burgers': BurgersDatasetConfig(),
        'darcy': DarcyDatasetConfig(),
        'reaction-diffusion': ReactionDiffusionDatasetConfig()
    }


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNO_Small2d()
    opt: OptimizationConfig = MultiphysicsOptConfig()
    data: MultiphysicsDatasetConfig = MultiphysicsDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
