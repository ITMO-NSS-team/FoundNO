import logging
from pathlib import Path
from typing import Union, List

from torch.utils.data import DataLoader

from fnofound.data.data.datasets.multiphysicis_dataset import MultiphysicsDataset
from neuralop.data.datasets.web_utils import download_from_zenodo_record

logger = logging.Logger(logging.root.level)


class MultitaskDataset(MultiphysicsDataset):
    """
    MultitaskDataset stores data generated according to set of tasks.
    Input is a coefficient function and outputs describe flow.

    Attributes
    ----------
    train_db: torch.utils.data.Dataset of training examples
    test_db:  ""                       of test examples
    data_processor: neuralop.data.transforms.DataProcessor to process data examples
        optional, default is None
    """

    def __init__(self,
                 root_dir: Union[Path, str],
                 n_train: int,
                 n_tests: List[int],
                 batch_size: int,
                 test_batch_sizes: List[int],
                 train_resolution: int,
                 test_resolutions: List[int] = [16, 32],
                 temporal_subsample: int = 1,
                 spatial_subsample: int = 1,
                 interpolate_mode: str = None,
                 encode_input: bool = False,
                 encode_output: bool = True,
                 encoding="channel-wise",
                 channel_dim=1,
                 subsampling_rate=None,
                 download_params: dict = None,
                 physics_name: str = None):

        """MultitaskDataset

        Parameters
        ----------
        root_dir : Union[Path, str]
            root at which to download data files
        dataset_name : str
            prefix of pt data files to store/access
        n_train : int
            number of train instances
        n_tests : List[int]
            number of test instances per test dataset
        batch_size : int
            batch size of training set
        test_batch_sizes : List[int]
            batch size of test sets
        train_resolution : int
            resolution of data for training set
        test_resolutions : List[int], optional
            resolution of data for testing sets, by default [16,32]
        encode_input : bool, optional
            whether to normalize inputs in provided DataProcessor,
            by default False
        encode_output : bool, optional
            whether to normalize outputs in provided DataProcessor,
            by default True
        encoding : str, optional
            parameter for input/output normalization. Whether
            to normalize by channel ("channel-wise") or
            by pixel ("pixel-wise"), default "channel-wise"
        input_subsampling_rate : int or List[int], optional
            rate at which to subsample each input dimension, by default None
        output_subsampling_rate : int or List[int], optional
            rate at which to subsample each output dimension, by default None
        channel_dim : int, optional
            dimension of saved tensors to index data channels, by default 1
        """

        # convert root dir to Path
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        # List of resolutions needed for dataset object
        resolutions = set(test_resolutions + [train_resolution])

        # Download data from source (zenodo, pdebench) archive if passed
        if download_params['download']:
            files_to_download = []
            already_downloaded_files = [x.name for x in root_dir.iterdir()]
            for res in resolutions:
                if f"f{physics_name}_train_{res}.pt" not in already_downloaded_files or \
                        f"f{physics_name}_test_{res}.pt" not in already_downloaded_files:
                    files_to_download.append(f"f{physics_name}_{res}.tgz")
                download_params['files_to_download'] = files_to_download
            download_from_zenodo_record(**download_params)

        # Once downloaded/if files already exist, init MultiphysicsDataset
        super().__init__(
            root_dir=root_dir,
            physics_name=physics_name,
            n_train=n_train,
            n_tests=n_tests,
            batch_size=batch_size,
            test_batch_sizes=test_batch_sizes,
            train_resolution=train_resolution,
            test_resolutions=test_resolutions,
            interpolate_mode=interpolate_mode,
            encode_input=encode_input,
            encode_output=encode_output,
            encoding=encoding,
            channel_dim=channel_dim,
            input_subsampling_rate=subsampling_rate,
            output_subsampling_rate=[temporal_subsample, spatial_subsample],
        )
        super().data_preprocessing_pipline()


def load_data(n_train,
              n_tests,
              batch_size,
              test_batch_sizes,
              data_root,
              train_resolution=16,
              temporal_subsample: int = 1,
              spatial_subsample: int = 1,
              test_resolutions=[16, 32],
              interpolate_mode=None,
              encode_input=False,
              encode_output=True,
              encoding="channel-wise",
              channel_dim=1,
              physics_name=None,
              download_params=None):

    dataset = MultitaskDataset(
        root_dir=data_root,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        train_resolution=train_resolution,
        test_resolutions=test_resolutions,
        temporal_subsample=temporal_subsample,
        spatial_subsample=spatial_subsample,
        interpolate_mode=interpolate_mode,
        encode_input=encode_input,
        encode_output=encode_output,
        channel_dim=channel_dim,
        encoding=encoding,
        download_params=download_params,
        physics_name=physics_name,
    )

    # return dataloaders for backwards compat
    train_loader = DataLoader(
        dataset.train_db,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False
    )

    test_loaders = {}
    for res, test_bsize in zip(test_resolutions, test_batch_sizes):
        test_loaders[res] = DataLoader(
            dataset.test_dbs[res],
            batch_size=test_bsize,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            persistent_workers=False
        )

    return train_loader, test_loaders, dataset.data_processor
