from muno.data.data.datasets.darcy import DarcyDataset, load_darcy_flow_small
from muno.data.data.datasets.navier_stokes import NavierStokesDataset, load_navier_stokes_pt
from muno.data.data.datasets.pt_dataset import PTDataset
from muno.data.data.datasets.burgers import Burgers1dTimeDataset, load_mini_burgers_1dtime
from muno.data.data.datasets.dict_dataset import DictDataset
from muno.data.data.datasets.mesh_datamodule import MeshDataModule
from muno.data.data.datasets.car_cfd_dataset import CarCFDDataset, load_mini_car

# only import SphericalSWEDataset if torch_harmonics is built locally
try:
    from .spherical_swe import load_spherical_swe
except ModuleNotFoundError:
    pass

# only import TheWell if the_well is built
try:
    from .the_well_dataset import (TheWellDataset,
                           ActiveMatterDataset,
                           MHD64Dataset)
except ModuleNotFoundError:
    pass