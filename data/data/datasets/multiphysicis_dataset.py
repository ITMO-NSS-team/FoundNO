import torch
import torch.nn.functional as F
import numpy as np
import h5py

from pathlib import Path
from typing import List, Union
from scipy.interpolate import RegularGridInterpolator

from fnofound.data.data.datasets.tensor_dataset import TensorDataset
from fnofound.data.data.transforms.data_processors import DefaultDataProcessor
from fnofound.data.data.transforms.normalizers import UnitGaussianNormalizer


def resize_to_common_grid_batch(data, target_grid_shape):
    _, *dim_shapes = data.shape
    x, y = [np.linspace(0, 1, dim_i) for dim_i in dim_shapes]
    interpolator_new = RegularGridInterpolator(
        (x, y), data
    )
    grid_new = np.stack(
        list(np.meshgrid(*(np.linspace(0, 1, target_grid_shape) for _ in range(len(dim_shapes))), indexing='ij')),
        axis=-1
    )
    return interpolator_new(grid_new)


def get_mode(data):
    if data.dtype is torch.bool:
        return 'nearest'

    if len(data.shape) != 5:
        return 'bilinear'
    else:
        return 'trilinear'


def resize_to_common_grid(data, target_resolution, interpolate_mode):
    original_shape = data.shape
    original_ndim = data.ndim

    if torch.all(torch.tensor(original_shape[1:]) == torch.tensor(target_resolution)):
        return data

    # (B, C, H, W)
    if original_ndim == 2:
        # (B, X) -> (B, 1, 1, X)
        data = data.unsqueeze(1).unsqueeze(2)
        target_resolution = (1, target_resolution)
    elif original_ndim == 3:
        # (B, X, Y) -> (B, 1, X, Y)
        data = data.unsqueeze(1)

    if interpolate_mode is None:
        interpolate_mode = get_mode(data)

    data_resized = F.interpolate(
        data.float(),
        size=target_resolution,
        mode=interpolate_mode
    ).squeeze()

    return data_resized


def subsample(data, subsampling_rate, n_train, channel_dim):
    data_dims = data.ndim - 2

    if not subsampling_rate:
        subsampling_rate = 1
    if not isinstance(subsampling_rate, list):
        subsampling_rate = [subsampling_rate] * data_dims
    assert len(subsampling_rate) == data_dims, \
        f"Error: length mismatch between input_subsampling_rate and dimensions of data.\
                    input_subsampling_rate must be one int shared across all dims, or an iterable of\
                        length {len(data_dims)}, got {subsampling_rate}"

    train_indices = [slice(0, n_train, None)] + [slice(None, None, rate) for rate in subsampling_rate]
    train_indices.insert(channel_dim, slice(None))
    return data[train_indices]


class MultiphysicsDataset:
    """MultiphysicsDataset is a base Dataset class for our library.
            MultiphysicsDataset contain input-output pairs a(x), u(x) and may also
            contain additional information, e.g. function parameters,
            input geometry or output query points.

            datasets may implement a download flag at init, which provides
            access to a number of premade datasets for sample problems provided
            in our Zenodo archive.

        All datasets are required to expose the following attributes after init:

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
                 test_resolutions: List[int],
                 encode_input: bool = False,
                 encode_output: bool = True,
                 encoding="channel-wise",
                 interpolate_mode: str = None,
                 input_subsampling_rate=None,
                 output_subsampling_rate=None,
                 channel_dim=1,
                 channels_squeezed=True,
                 physics_name: str = None
                 ):
        """MultiphysicsDataset

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
        channels_squeezed : bool, optional
            If the channels dim is 1, whether that is explicitly kept in the saved tensor.
            If not, we need to unsqueeze it to explicitly have a channel dim.
            Only applies when there is only one data channel, as in our example problems
            Defaults to True
        """
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_resolution = train_resolution
        self.test_resolutions = test_resolutions
        self.test_batch_sizes = test_batch_sizes
        self.physics_name = physics_name
        self.interpolate_mode = interpolate_mode
        self.channel_dim = channel_dim
        self.channels_squeezed = channels_squeezed
        self.input_subsampling_rate = input_subsampling_rate
        self.output_subsampling_rate = output_subsampling_rate
        self.n_train = n_train
        self.n_tests = n_tests
        self.encode_input = encode_input
        self.encode_output = encode_output
        self.encoding = encoding

    def load_data(self, process_type, file_format=".pt"):
        file_path = Path(self.root_dir).joinpath(f"{self.physics_name}_{process_type}_16{file_format}")

        # print(f'loading file {file_path}')
        if file_format == ".pt":
            data = torch.load(file_path.as_posix())
            return data
        elif file_format == ".h5":
            with h5py.File(file_path, "r") as f:
                keys = list(f.keys())
                print(f"File {file_path} includes: {keys}")
                data = {key: f[key][:] for key in keys}
            return data
        else:
            raise ValueError(f"Unknown file format: {file_format}")

    def preprocess_raw_data(self, data):
        data["x"] = resize_to_common_grid(data["x"], self.train_resolution, self.interpolate_mode)
        data["y"] = resize_to_common_grid(data["y"], self.train_resolution, self.interpolate_mode)

        x = data["x"].type(torch.float32).clone()
        y = data["y"].type(torch.float32).clone()

        if self.channels_squeezed:
            x = x.unsqueeze(self.channel_dim)
            y = y.unsqueeze(self.channel_dim)

        x = subsample(x, self.input_subsampling_rate, self.n_train, self.channel_dim)
        y = subsample(y, self.output_subsampling_rate, self.n_train, self.channel_dim)
        return x, y

    def normalize(self, data):
        if not self.encode_input:
            return None
        else:
            if self.encoding == "channel-wise":
                reduce_dims = list(range(data.ndim))
                reduce_dims.pop(self.channel_dim)
            elif self.encoding == "pixel-wise":
                reduce_dims = [0]

            encoder = UnitGaussianNormalizer(dim=reduce_dims)
            encoder.fit(data)
            return encoder

    def data_preprocessing_pipline(self):
        # Load train data
        train_data = self.load_data('train', file_format='.pt')
        # print([(key, elem.shape) for key, elem in train_data.items()])
        # raise NotImplementedError()
        x_train, y_train = self.preprocess_raw_data(train_data)
        del train_data

        input_encoder = self.normalize(x_train)
        output_encoder = self.normalize(y_train)

        # Save train dataset
        self._train_db = TensorDataset(
            x_train,
            y_train,
        )

        self._data_processor = DefaultDataProcessor(in_normalizer=input_encoder,
                                                    out_normalizer=output_encoder)

        self._test_dbs = {}
        for (res, n_test) in zip(self.test_resolutions, self.n_tests):
            print(f"Loading test db for resolution {res} with {n_test} samples ")

            # Load test data
            test_data = self.load_data('test', file_format='.pt')
            x_test, y_test = self.preprocess_raw_data(test_data)
            del test_data

            # Save test dataset
            test_db = TensorDataset(
                x_test,
                y_test,
            )
            self._test_dbs[res] = test_db

    @property
    def data_processor(self):
        return self._data_processor

    @property
    def train_db(self):
        return self._train_db

    @property
    def test_dbs(self):
        return self._test_dbs
