"""
airfoil_config.py - configuration for the AirfRANS experiments
(FNO / RNO / DNO / Geo-FNO).

Follows the FoundNO config pattern (see dno_config.py): an aggregator
AirfoilDefault with model / data / opt subsections, overridable from the
command line via dotted keys (--model.n_modes, --data.batch_size, ...).
"""

from typing import List, Literal

from zencfg import ConfigBase

from fnofound.data.config.opt import OptimizationConfig

CaseType = Literal['fno', 'rno', 'dno', 'geofno']


class AirfoilModelConfig(ConfigBase):
    """Shared model hyperparameters.

    Per-case channel layout (overridden by the registry in airfoil_train.py):
      fno/rno: in=3 (vx_in, vy_in, mask) + grid
      dno:     in=3 (vx_in, vy_in, mask) + grid + grid_mesh
      geofno:  in=2 (vx_in, vy_in) + grid, padding>0 (C-grid)
    """
    n_modes: List[int] = [16, 16]
    hidden_channels: int = 32
    n_layers: int = 4
    in_channels: int = 3
    out_channels: int = 4
    use_grid: bool = True
    padding: int = 0


class AirfoilDataConfig(ConfigBase):
    """Data paths and loading settings."""
    case_type: CaseType = 'fno'
    # Root with the per-case subfolders:
    #   fno/rno:   {root_dir}/fno_dataset
    #   dno:       {root_dir}/DNO_data/dno_small
    #   geofno:    {root_dir}/Geo-FNO_data
    root_dir: str = '/media/expiwt/f2d8c27a-bdac-4400-b1be-cdb75b6b5a40/airfrans'
    batch_size: int = 8
    num_workers: int = 4
    cache: bool = False          # preload Geo-FNO data into RAM


class AirfoilOptConfig(OptimizationConfig):
    """Optimization settings."""
    n_epochs: int = 150
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = 'StepLR'
    step_size: int = 50
    gamma: float = 0.5


class AirfoilDefault(ConfigBase):
    """Complete AirfRANS experiment configuration."""
    verbose: bool = True
    arch: str = 'airfoil'

    model: AirfoilModelConfig = AirfoilModelConfig()
    data: AirfoilDataConfig = AirfoilDataConfig()
    opt: AirfoilOptConfig = AirfoilOptConfig()
