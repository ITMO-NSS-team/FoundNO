from typing import Dict, Any, List, Optional

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


class BurgersDatasetConfig(ConfigBase):
    train_resolution: int = 32
    test_resolutions: List[int] = [32]
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
    train_resolution: int = 32
    test_resolutions: List[int] = [32]
    test_batch_sizes: List[int] = [16]
    n_tests: List[int] = [100]
    spatial_subsample: Optional[int] = None
    temporal_subsample: Optional[int] = None
    interpolate_mode: Optional[str] = None  # 'bilinear'
    encode_input: bool = True
    encode_output: bool = True
    download_params: Dict[str, Any] = {
        'source': 'zenodo',
        'dataset_id': '12784353',
        'download': False
    }


class MultiphysicsDatasetConfig(ConfigBase):
    folder: str = 'neuralop/data/datasets/data/'
    batch_size: int = 8
    n_train: int = 1000
    datasets: Dict[str, Any] = {
        'burgers': BurgersDatasetConfig(),
        'darcy': DarcyDatasetConfig()
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
