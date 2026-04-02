from typing import Dict
from math import prod
import pickle

# from fnofound.data.data.utils import count_tensor_params
from fnofound.data.data.transforms.base_transforms import Transform, DictTransform
import torch


def masked_std(arg: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor = None, n_elements=None):
    # Assume, the input has shape of [B, C, T, X, ...]
    # The result is expected to follow the shapes of [B, C, 1, ... 1]

    dim = [0, ] + list(range(2, arg.ndim))
    if n_elements is None:
        n_elements = torch.count_nonzero(mask)

    if mean is None:
        mean = (torch.sum(arg * mask, dim=dim, keepdim=True) / n_elements)
    return torch.sqrt(torch.sum(mask * torch.square(arg - mean), dim=dim, keepdim=True) / n_elements)


def setPrecision(arg, precision):
    if isinstance(arg, torch.Tensor):
        arg = arg.to(precision)
    return arg


def iterativeMean(batch: torch.Tensor, mask: torch.Tensor, prev_mean=0, n_prev_elem: int = 0, n_batch_elem: int = None,
                  dim=None):
    PRECISION = torch.float64
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(PRECISION)

    if mask is None:
        mask = torch.ones_like(batch[
                                   0, 0, ...])  # Introduced to match batch-channel-time dimensions, implement better option, working with dim

    setPrecision(batch, PRECISION), setPrecision(prev_mean, PRECISION), setPrecision(mask, PRECISION)

    if n_batch_elem is None:
        n_batch_elem = torch.count_nonzero(mask)

    n_updated_elem = n_batch_elem + n_prev_elem
    mean = (prev_mean * n_prev_elem + torch.sum(batch * mask, dim=dim, keepdim=True)) / n_updated_elem

    torch.set_default_dtype(default_dtype)
    return n_updated_elem, mean


def iterativeSTD(batch: torch.Tensor, mask: torch.Tensor = None,
                 prev_mean=0, updated_mean=None,
                 prev_std=1., n_prev_elem: int = 0,
                 n_batch_elem: int = None, dim=None):
    PRECISION = torch.float64

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(PRECISION)
    setPrecision(batch, PRECISION), setPrecision(prev_mean, PRECISION), setPrecision(prev_std, PRECISION)

    if mask is None:
        mask = torch.ones_like(batch[
                                   0, 0, ...])  # Introduced to match batch-channel-time dimensions, implement better option, working with dim
    setPrecision(mask, PRECISION)

    if n_batch_elem is None:
        n_batch_elem = torch.count_nonzero(mask)

    if updated_mean is None:
        n_updated_elem, updated_mean = iterativeMean(batch, mask, prev_mean, n_prev_elem, n_batch_elem, dim)
    else:
        n_updated_elem = n_batch_elem + n_prev_elem

    delta_mean = updated_mean - prev_mean

    std = torch.sqrt((torch.sum(mask * torch.square(batch - updated_mean), dim=dim, keepdim=True) +
                      torch.square(prev_std) * (n_prev_elem) + n_prev_elem * torch.square(delta_mean)) /
                     (n_updated_elem))  # (n_prev_elem - 1)  (n_updated_elem - 1)

    torch.set_default_dtype(default_dtype)
    return n_updated_elem, std


def count_tensor_params(tensor: torch.Tensor, dims=None):
    """Returns the number of parameters (elements) in a single tensor, optionally, along certain dimensions only

    Parameters
    ----------
    tensor : torch.tensor
    dims : int list or None, default is None
        if not None, the dimensions to consider when counting the number of parameters (elements)

    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    if dims is None:
        dims = list(tensor.shape)
    else:
        dims = [tensor.shape[d] for d in dims]
    n_params = prod(dims)
    if tensor.is_complex():
        return 2 * n_params
    return n_params


class Normalizer(Transform):
    def __init__(self, mean, std, eps=1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def transform(self, data):
        return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data):
        return (data * (self.std + self.eps)) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class UnitGaussianNormalizer(Transform):
    """
    UnitGaussianNormalizer normalizes data to be zero mean and unit std.
    """

    def __init__(self, mean=None, std=None, eps=1e-7, dim=None, mask=None):
        """
        mean : torch.tensor or None
            has to include batch-size as a dim of 1
            e.g. for tensors of shape ``(batch_size, channels, height, width)``,
            the mean over height and width should have shape ``(1, channels, 1, 1)``
        std : torch.tensor or None
        eps : float, default is 0
            for safe division by the std
        dim : int list, default is None
            if not None, dimensions of the data to reduce over to compute the mean and std.

            .. important::

                Has to include the batch-size (typically 0).
                For instance, to normalize data of shape ``(batch_size, channels, height, width)``
                along batch-size, height and width, pass ``dim=[0, 2, 3]``

        mask : torch.Tensor or None, default is None
            If not None, a tensor with the same size as a sample,
            with value 0 where the data should be ignored and 1 everywhere else

        Notes
        -----
        The resulting mean will have the same size as the input MINUS the specified dims.
        If you do not specify any dims, the mean and std will both be scalars.

        Returns
        -------
        UnitGaussianNormalizer instance
        """
        super().__init__()

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("mask", mask)

        self.eps = eps
        if mean is not None:
            self.ndim = mean.ndim
        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim
        self.n_elements = 0

    def fit(self, data_batch):
        self.update_mean_std(data_batch)

    def partial_fit(self, data_batch, batch_size=1):
        if 0 in list(data_batch.shape):
            return
        count = 0

        if batch_size == 1:
            data_batch = data_batch.unsqueeze(0)
        n_samples = len(data_batch)
        while count < n_samples:
            samples = data_batch[count: count + batch_size]

            if self.n_elements:
                self.incremental_update_mean_std(samples)
            else:
                self.update_mean_std(samples)
            count += batch_size

    def update_mean_std(self, data_batch):
        # dim = [i + 2 for i in self.dim if i]
        stat_shape = [1, data_batch.shape[1], ] + [1, ] * (data_batch.ndim - 2)
        self.mean = torch.zeros(size=stat_shape).to(data_batch.device)
        self.std = torch.ones(size=stat_shape).to(data_batch.device)

        # print(f'In update_mean_std procedure: dims are {self.dim}')

        self.ndim = data_batch.ndim  # Note this includes batch-size

        prev_mean = self.mean;
        prev_numel = 0
        self.n_elements, self.mean = iterativeMean(data_batch, self.mask, self.mean, 0, dim=self.dim)
        # print(f'after iterativeMean: tensors of means {self.mean.shape} & stds {self.std.shape}')

        _, self.std = iterativeSTD(data_batch, self.mask, prev_mean, self.mean, self.std,
                                   n_prev_elem=prev_numel, dim=self.dim)

        # if self.mask is None:
        #     self.n_elements = count_tensor_params(data_batch, self.dim)
        #     self.mean = torch.mean(data_batch, dim=self.dim, keepdim=True)
        #     self.squared_mean = torch.mean(data_batch ** 2, dim=self.dim, keepdim=True)
        #     self.std = torch.std(data_batch, dim=self.dim, keepdim=True)
        #     print(f'Got mean {self.mean} & std {self.std} from databatch of size {data_batch.shape}')
        # else:
        #     batch_size, channels_num = data_batch.shape[0], data_batch.shape[1]
        #     dim = [i - 2 for i in self.dim if i]
        #     shape = [s for i, s in enumerate(self.mask.shape) if i not in dim]
        #     self.n_elements = torch.count_nonzero(self.mask, dim=dim) * batch_size * channels_num
        #     self.mean = torch.zeros(shape)
        #     self.std = torch.zeros(shape)
        #     self.squared_mean = torch.zeros(shape)
        #     # data_batch[:, :, self.mask == 1] = 0 # [self.mask == 1]
        #     self.mean = (torch.sum(data_batch * self.mask , dim=dim, keepdim=True) / self.n_elements)

        #     self.squared_mean = (torch.sum(data_batch ** 2, dim=dim, keepdim=True) / self.n_elements)
        #     self.std = masked_std(data_batch, self.mask, self.mean)
        #     # self.std = torch.std(data_batch, dim=self.dim, keepdim=True) # Is it correct?
        #     print(f'Got mean {self.mean} & std {self.std} from databatch of size {data_batch.shape}')

    def incremental_update_mean_std(self, data_batch):
        # dim = [i - 2 for i in self.dim if i]
        # print(f'In incremental_update_mean_std procedure: dims are {dim}')

        prev_mean = self.mean
        prev_numel = self.n_elements
        self.n_elements, self.mean = iterativeMean(data_batch, self.mask, self.mean, prev_numel, dim=self.dim)
        _, self.std = iterativeSTD(data_batch, self.mask, prev_mean, self.mean, self.std,
                                   n_prev_elem=prev_numel, dim=self.dim)
        # print(
        #     f'after iterativeMean (in incremental_update_mean_std): tensors of means {self.mean.shape} & stds {self.std.shape}')

        # if self.mask is None:
        #     n_elements = count_tensor_params(data_batch, self.dim)
        #     dim = self.dim
        # else:
        #     dim = [i - 1 for i in self.dim if i]
        #     n_elements = torch.count_nonzero(self.mask, dim=dim) * data_batch.shape[0] * data_batch.shape[1]
        #     data_batch[:, :, self.mask == 1] = 0

        # self.mean = (1.0 / (self.n_elements + n_elements)) * (
        #         self.n_elements * self.mean + torch.sum(data_batch, dim=dim, keepdim=True)
        # )
        # self.squared_mean = (1.0 / (self.n_elements + n_elements)) * (
        #         self.n_elements * self.squared_mean
        #         + torch.sum(data_batch ** 2, dim=dim, keepdim=True)
        # )
        # self.n_elements += n_elements

        # # 1/(n_i + n_j) * (n_i * sum(x_i^2)/n_i + sum(x_j^2) - (n_i*sum(x_i)/n_i + sum(x_j))^2)
        # # = 1/(n_i + n_j)  * (sum(x_i^2) + sum(x_j^2) - sum(x_i)^2 - 2sum(x_i)sum(x_j) - sum(x_j)^2))
        # # multiply by (n_i + n_j) / (n_i + n_j + 1) for unbiased estimator
        # # if self.mask is None:
        # self.std = torch.sqrt(self.squared_mean - self.mean ** 2) * self.n_elements / (self.n_elements - 1)
        # # else:
        # #     self.std = masked_std(data_batch, self.mask, self.mean)

        # print(f'Got mean changed to {self.mean} & std {self.std} from databatch of size {data_batch.shape}')

    def transform(self, x):
        # print(x.shape, self.mean.shape, self.std.shape, self.eps)
        # print('in transform:', x.shape, self.std.shape, self.mean.shape)
        # print(self.std, self.mean)
        # print('means:', torch.mean(x), torch.mean((x - self.mean) / (self.std + self.eps)))
        return (x - self.mean) / (self.std + self.eps)  # * self.mask)

    def inverse_transform(self, x):
        # print('in inverse_transform:', x.shape, self.std.shape, self.mean.shape)
        # print(self.std, self.mean)
        # print('means:', torch.mean(x), torch.mean(x * (self.std + self.eps) + self.mean))

        return x * (self.std + self.eps) + self.mean  # * self.mask)

    def forward(self, x):
        return self.transform(x)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        self.mask = self.mask.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

    def to(self, device):
        self.mask.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None):
        """Return a dictionary of normalizer instances, fitted on the given dataset

        Parameters
        ----------
        dataset : pytorch dataset
            each element must be a dict {key: sample}
            e.g. {'x': input_samples, 'y': target_labels}
        dim : int list, default is None
            * If None, reduce over all dims (scalar mean and std)
            * Otherwise, must include batch-dimensions and all over dims to reduce over
        keys : str list or None
            if not None, a normalizer is instanciated only for the given keys
        """
        for i, data_dict in enumerate(dataset):
            if not i:
                if not keys:
                    keys = data_dict.keys()
        instances = {key: cls(dim=dim, mask=mask) for key in keys}
        for i, data_dict in enumerate(dataset):
            for key, sample in data_dict.items():
                if key in keys:
                    instances[key].partial_fit(sample.unsqueeze(0))
        return instances

    def to_file(self, filename):
        assert 'pkl' in filename, 'Incorrect filename'

        saved_dict = {'mask': self.mask, 'mean': self.mean, 'std': self.std}
        with open(filename, 'wb') as f:
            pickle.dump(saved_dict, f)

    def from_file(self, filename):
        assert 'pkl' in filename, 'Incorrect filename'
        # savedict = {'mask': self.mask, 'mean': self.mean, 'std': self.std}
        with open(filename, 'rb') as f:
            loaded_dict = pickle.load(f)

        assert ('mean' in loaded_dict.keys()) and ('std' in loaded_dict.keys()), 'Missing mean and std from loaded norm. dict'
        self.mean = loaded_dict['mean']
        self.std  = loaded_dict['std']

class DictUnitGaussianNormalizer(DictTransform):
    """DictUnitGaussianNormalizer composes
    DictTransform and UnitGaussianNormalizer to normalize different
    fields of a model output tensor to Gaussian distributions w/
    mean 0 and unit variance.

        Parameters
        ----------
        normalizer_dict : Dict[str, UnitGaussianNormalizer]
            dictionary of normalizers, keyed to fields
        input_mappings : Dict[slice]
            slices of input tensor to grab per field, must share keys with above
        return_mappings : Dict[slice]
            _description_
        """

    def __init__(self,
                 normalizer_dict: Dict[str, UnitGaussianNormalizer],
                 input_mappings: Dict[str, slice],
                 return_mappings: Dict[str, slice]):
        assert set(normalizer_dict.keys()) == set(input_mappings.keys()), \
            "Error: normalizers and model input fields must be keyed identically"
        assert set(normalizer_dict.keys()) == set(return_mappings.keys()), \
            "Error: normalizers and model output fields must be keyed identically"

        super().__init__(transform_dict=normalizer_dict,
                         input_mappings=input_mappings,
                         return_mappings=return_mappings)

    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None):
        """Return a dictionary of normalizer instances, fitted on the given dataset

        Parameters
        ----------
        dataset : pytorch dataset
            each element must be a dict {key: sample}
            e.g. {'x': input_samples, 'y': target_labels}
        dim : int list, default is None
            * If None, reduce over all dims (scalar mean and std)
            * Otherwise, must include batch-dimensions and all over dims to reduce over
        keys : str list or None
            if not None, a normalizer is instanciated only for the given keys
        """
        for i, data_dict in enumerate(dataset):
            if not i:
                if not keys:
                    keys = data_dict.keys()
        instances = {key: cls(dim=dim, mask=mask) for key in keys}
        for i, data_dict in enumerate(dataset):
            for key, sample in data_dict.items():
                if key in keys:
                    instances[key].partial_fit(sample.unsqueeze(0))
        return instances


class MultiphysicsUnitGaussianNormalizer(Transform):
    """Multiphysics version of UnitGaussianNormalizer"""

    def __init__(self):
        super().__init__()
        self.normalizers = {}
        self.current_physics = None

    def add_physics(self, physics_name: str, dim=None, mask=None):
        self.normalizers[physics_name] = UnitGaussianNormalizer(dim=dim, mask=mask)

    def set_physics(self, physics_name: str):
        if physics_name not in self.normalizers:
            raise ValueError(f"Normalizer for physics '{physics_name}' not found!")
        self.current_physics = physics_name

    def fit(self, data_batch):
        if self.current_physics is None:
            raise ValueError("Current physics is not installed!")
        self.normalizers[self.current_physics].fit(data_batch)

    def transform(self, x):
        if self.current_physics is None:
            raise ValueError("Current physics is not installed!")
        return self.normalizers[self.current_physics].transform(x)

    def inverse_transform(self, x):
        if self.current_physics is None:
            raise ValueError("Current physics is not installed!")
        return self.normalizers[self.current_physics].inverse_transform(x)

    def to(self, device):
        for normalizer in self.normalizers.values():
            normalizer.to(device)
        return self

    def cuda(self):
        for normalizer in self.normalizers.values():
            normalizer.cuda()
        return self

    def cpu(self):
        for normalizer in self.normalizers.values():
            normalizer.cpu()
        return self
