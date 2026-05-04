"""
DNO configuration — model, data, and optimization settings.

Used by training scripts:
    from fnofound.data.config.dno_config import DnoDefault

    config = DnoDefault()
    config.model.n_modes       # [16, 16]
    config.data.case_type      # "darcy"
    config.opt.learning_rate   # 0.001

Supports three cases (data.case_type):
  - "darcy":    4 in → 1 out, CSV with NaN masking
  - "fluid":    3 in → 3 out, CSV with Z-score normalization
  - "reservoir": 39 in → 72 out, lazy HDF5 loading
"""

from typing import List, Optional, Any
from zencfg import ConfigBase

from fnofound.data.config.distributed import DistributedConfig
from fnofound.data.config.wandb import WandbConfig
from fnofound.data.config.opt import OptimizationConfig, PatchingConfig
from fnofound.data.config.models import DnoModelConfig


class DnoOptConfig(OptimizationConfig):
    """Optimization settings for DNO training.
    """
    n_epochs: int = 150
    learning_rate: float = 1e-3
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 50
    gamma: float = 0.5


class DnoDataConfig(ConfigBase):
    """Data loading and preprocessing settings.

    case_type determines the expected data format.

    Per-case:
    # ── darcy ──
    #   folder:   ./data/dno_tasks/diff_heptagon/
    #   files:    train_C.csv, train_x_data.csv, train_y_data.csv, train_U.csv
    #   u_scale:  10.0        # output scaling (not Z-score)
    #   mask_loss: True       # MaskedLpLoss for holes
    #
    # ── fluid (obstacle) ──
    #   folder:   ./data/obstacle/
    #   files:    x_data.csv, y_data.csv, u_data.csv, v_data.csv, p_data.csv, re_data.csv
    #   u_scale:  N/A         # uses Z-score normalization instead
    #   mask_loss: False      # fluid domain has no holes
    #
    # ── reservoir ──
    #   folder:   /path/to/data.hdf5
    #   files:    HDF5 with keys: dataset_128, x_map, y_map
    #   u_scale:  N/A         # uses p_scale=100, s_scale=1000 from training script
    #   mask_loss: False
    """
    folder: str = "./data/dno_tasks/"
    case_type: str = "darcy"
    batch_size: int = 16
    train_resolution: int = 128
    n_train: int = 1000
    u_scale: float = 10.0
    mask_loss: bool = True
    val_split: float = 0.1


class DnoDefault(ConfigBase):
    """Complete DNO configuration.

    Aggregates model, data, and optimizer settings.
    Follows the standard FoundNO config pattern.

    CLI usage:
        python train.py \\
            --model.in_channels 39 --model.out_channels 72 \\
            --data.case_type reservoir \\
            --opt.learning_rate 1e-3 --opt.n_epochs 200
    """
    verbose: bool = True
    arch: str = "dno"

    distributed: DistributedConfig = DistributedConfig()
    wandb: WandbConfig = WandbConfig()
    patching: PatchingConfig = PatchingConfig()

    model: DnoModelConfig = DnoModelConfig()
    data: DnoDataConfig = DnoDataConfig()
    opt: DnoOptConfig = DnoOptConfig()
