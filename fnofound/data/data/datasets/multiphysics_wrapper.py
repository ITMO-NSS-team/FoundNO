import logging
from pathlib import Path
from typing import Union, List

import torch
from torch.utils.data import DataLoader

from fnofound.data.data.transforms.data_processors import DefaultDataProcessor
from fnofound.data.data.datasets.multiphysicis_dataset import MultiphysicsDataset
from neuralop.data.datasets.web_utils import download_from_zenodo_record

logger = logging.Logger(logging.root.level)


class MultiphysicsDataProcessor(DefaultDataProcessor):
    def __init__(self,
                 spatial_inputs,
                 channel_dim,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_inputs = spatial_inputs
        self.channel_dim = channel_dim

    def preprocess(self, data_dict, batched=True):
        data_dict["x"] = data_dict["x"].to('cuda')
        data_dict["y"] = data_dict["y"].to('cuda')
        return super().preprocess(data_dict, batched)


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
                 train_resolution: Union[int, dict] = None,
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
                 physics_train_name: str = None,
                 physics_test_name: str = None,
                 custom_in_channels: dict = None,
                 device: str = 'cuda'):

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

        # Once downloaded/if files already exist, init MultiphysicsDataset
        super().__init__(
            root_dir=root_dir,
            physics_train_name=physics_train_name,
            physics_test_name=physics_test_name,
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
            custom_in_channels=custom_in_channels,
            device=device
        )

        super().data_preprocessing_pipline()
        self._data_processor = MultiphysicsDataProcessor(
            custom_in_channels,
            channel_dim,
            self._data_processor.in_normalizer,
            self._data_processor.out_normalizer
        )


def load_data(n_train,
              n_tests,
              batch_size,
              test_batch_sizes,
              data_root,
              train_resolution=None,
              temporal_subsample: int = 1,
              spatial_subsample: int = 1,
              test_resolutions=[16, 32],
              interpolate_mode=None,
              encode_input=False,
              encode_output=True,
              encoding="channel-wise",
              channel_dim=1,
              physics_train_name: str = None,
              physics_test_name: str = None,
              download_params=None,
              custom_in_channels=None,
              device='cuda'):

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
        physics_train_name=physics_train_name,
        physics_test_name=physics_test_name,
        custom_in_channels=custom_in_channels,
        device=device
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
