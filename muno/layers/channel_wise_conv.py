from typing import Union, Optional, List, Tuple, Callable
from functools import reduce
from itertools import permutations

import torch
from torch import nn

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor

from neuralop.utils import validate_scaling_factor
from neuralop.layers.einsum_utils import einsum_complexhalf
from neuralop.layers.base_spectral_conv import BaseSpectralConv
from neuralop.layers.resample import resample

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

'''
General idea of the convolution implementation and several folllowing methods are adapted from 
the neuralop/layers/spectral_convolution.py file.
'''

def _contract_dense(x, weight, separable=False):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    if x.dtype == torch.complex32:
        # if x is half precision, run a specialized einsum
        return einsum_complexhalf(eq, x, weight)
    else:
        return tl.einsum(eq, x, weight)


def _contract_dense_separable(x, weight, separable):
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    return x * weight


def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out
    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...
    eq = f'{x_syms},{rank_sym},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, cp_weight.weights, *cp_weight.factors)
    else:
        return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        # x, y, ...
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]

    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        # x, y, ...
        factor_syms += [xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]

    eq = f'{x_syms},{core_syms},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, tucker_weight.core, *tucker_weight.factors)
    else:
        return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, *tt_weight.factors)
    else:
        return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)
    separable: bool
        if True, performs contraction with individual tensor factors.
        if False,
    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        if separable:
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith("dense"):
                return _contract_dense
            elif weight.name.lower().endswith("tucker"):
                return _contract_tucker
            elif weight.name.lower().endswith("tt"):
                return _contract_tt
            elif weight.name.lower().endswith("cp"):
                return _contract_cp
            else:
                raise ValueError(f"Got unexpected factorized weight type {weight.name}")
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f'Got implementation={implementation}, expected "reconstructed" or "factorized"'
        )

def cwc_process_dim(x: torch.Tensor,
                    weights: torch.Tensor,
                    dim: int,
                    fft_norm: str,
                    contraction: Callable = _contract_dense,
                    separable: bool = False):
    x = torch.fft.rfft(x, dim = dim, norm = fft_norm)
    x = contraction(x, weights, separable)
    x = torch.fft.irfft(x, dim = dim, norm = fft_norm)
    return x

Number = Union[int, float]

class SplitDimensionSpectralConv(BaseSpectralConv):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 n_modes,
                 complex_data,
                 max_n_modes,
                 separable,
                 time_axis, # Along time axis a separate set of parameters will be applied
                 bias=True,
                 resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
                 fno_block_precision="full",
                 rank=1.0,
                 factorization=None,
                 implementation="reconstructed",
                 enforce_hermitian_symmetry=True,
                 fixed_rank_modes=False,
                 decomposition_kwargs: Optional[dict] = None,
                 init_std="auto",
                 fft_norm="forward",                 
                 device=None):
        super().__init__(device)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.complex_data = complex_data

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)
        self._time_axis = time_axis

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation
        self.enforce_hermitian_symmetry = enforce_hermitian_symmetry
        
        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.order)

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None
        self.fft_norm = fft_norm

        if factorization is None:
            factorization = "Dense"  # No factorization

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} and "
                    f"out_channels={out_channels}",
                )
            weight_shape = (in_channels, *max_n_modes)
        else:
            weight_shape = (in_channels, out_channels, *max_n_modes)
        self.separable = separable

        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}

        # Create/init spectral weight tensor

        self.weights_t = FactorizedTensor.new(weight_shape,
                                            rank=self.rank,
                                            factorization=factorization,
                                            fixed_rank_modes=fixed_rank_modes,
                                            **tensor_kwargs,
                                            dtype=torch.cfloat)

        self.weights_spat = FactorizedTensor.new(weight_shape,
                                                 rank=self.rank,
                                                 factorization=factorization,
                                                 fixed_rank_modes=fixed_rank_modes,
                                                 **tensor_kwargs,
                                                 dtype=torch.cfloat)

        self.mixer = nn.Linear(len(self.n_modes), 1)

        self.weight.normal_(0, init_std)

        self._contract = get_contract_fun(self.weight, implementation=implementation, separable=separable)

        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([self.out_channels]) + (1,) * self.order))
            )
        else:
            self.bias = None

    def transform(self, x, output_shape=None):
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):  # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # the real FFT is skew-symmetric, so the last mode has a redundacy if our data is real in space
        # As a design choice we do the operation here to avoid users dealing with the +1
        # if we use the full FFT we cannot cut off informtion from the last mode
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """Method is similar to the approach, proposed byh

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient in real spatial data
        fft_dims = list(range(-self.order, 0))

        if self.complex_data:
            raise NotImplementedError("Complex data processing has not been implemented yet.")
        else:
            channels_projs = []
            for dim in range(len(mode_sizes)):
                if dim == self._time_axis:
                    channels_projs.append(cwc_process_dim(x = x,
                                                          weights = self.weights_t,
                                                          dim = dim,
                                                          fft_norm = self.fft_norm,
                                                          contraction = self._contract))
                else:
                    channels_projs.append(cwc_process_dim(x = x,
                                                          weights = self.weights_t,
                                                          dim = dim,
                                                          fft_norm = self.fft_norm,
                                                          contraction = self._contract))

            x = torch.stack(channels_projs, dim = 1)

            x = x.moveaxis(1, -1)
            x = self.mixer(x)
            x = x.moveaxis(1, -1)

            return x


def weightsOuterProduct(x: torch.Tensor,
                        weights_s: List[List[torch.nn.Parameter]],
                        weights_t: List[torch.nn.Parameter],
                        mixer: torch.nn.Module): # Presumably, Linear is enough, a check for rank -> 1 inputs needed. 
                        # complex_data: bool = False):
    spatial_dim = x.ndim - 3

    # if complex_data:
    #     trunc_idx = [-1 for _ in weights_s]
    # else:
    #     trunc_idx = [(weights_s[0][idx].size(dim=-1)//2+1) for idx in range(len(weights_s[0]))] # assuming, all modes are the same 

    einsum_symbols_local = 'abdefghijk'
    left_symb, right_symb = [], 'c'

    for dim_idx in range(spatial_dim + 1):
        left_symb.append('c' + einsum_symbols_local[dim_idx])
        right_symb += einsum_symbols_local[dim_idx]

    equation = ','.join(left_symb) + '->' + right_symb

    # einsum_operands = [[weights_t[rank_idx],] + [weights_s[dim_idx][rank_idx] for dim_idx in range(spatial_dim-1)] + 
    #                    [weights_s[spatial_dim][rank_idx][:, :trunc_idx[rank_idx]],] for rank_idx in range(len(weights_t))]

    einsum_operands = [[weights_t[rank_idx],] + [weights_s[dim_idx][rank_idx] for dim_idx in range(spatial_dim)]
                       for rank_idx in range(len(weights_t))]

    full_dim_comps = [tl.einsum(equation, *einsum_operand) for einsum_operand in einsum_operands]

    weights = torch.stack(full_dim_comps, dim = -1)
    weights = torch.squeeze(mixer(weights), dim = -1)
    return weights


class FactorizedDimensionSpectralConv(BaseSpectralConv):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 n_modes: dict,
                 complex_data: bool = False,
                #  bias: bool = True,
                #  resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
                 fno_block_precision: str = "full",
                 max_spatial_dim: int = 3,
                 rank: int = 1,
                 enforce_hermitian_symmetry: bool = True,
                 init_std: str = "auto",
                 fft_norm: str = "forward",                 
                 device: Union[str, torch.device] = None, *args, **kwargs):
        super(FactorizedDimensionSpectralConv, self).__init__(device)

        err_txt: str = 'n_modes have to be a dict, similar to {"t": 16, "x": 32}'
        assert isinstance(n_modes, dict), \
            err_txt + f' instead got {n_modes}.'
        assert "x" in n_modes.keys() and "t" in n_modes.keys(), \
            err_txt + f' instead got {n_modes}.'

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.complex_data = complex_data

        # n_modes is the total number of modes kept along each dimension

        self._n_modes_dict = n_modes

        self.fno_block_precision = fno_block_precision
        self.rank = int(rank)
        self.enforce_hermitian_symmetry = enforce_hermitian_symmetry
        
        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = None # validate_scaling_factor(resolution_scaling_factor, self.order)

        if init_std == "auto": # Analogue of Xavier init
            init_std = (2 / (in_channels + out_channels)) ** 0.5

        self.fft_norm = fft_norm

        if in_channels != out_channels:
            chan_shape = [in_channels, out_channels]
            self.separable = False
        else:
            chan_shape = [in_channels,]
            self.separable = True

        # torch.nn.ModuleList
        print()
        self.weights_t = [torch.empty(chan_shape + [self._n_modes_dict["t"],], device = device),] * self.rank
        for tensor_idx in range(self.rank):
            torch.nn.init.xavier_normal_(self.weights_t[tensor_idx])
            self.weights_t[tensor_idx] = torch.nn.Parameter(self.weights_t[tensor_idx], requires_grad=True)
        self.weights_t = nn.ParameterList(self.weights_t)

        # Create/init spectral weight 2-nd rank tensors: for time and all spatial dimensions
        self.weights_x = [[torch.empty(chan_shape + [self._n_modes_dict["x"],], device = device)]*self.rank 
                          for _ in range(max_spatial_dim)]

        for dim_idx in range(max_spatial_dim):
            for tensor_idx in range(self.rank):
                torch.nn.init.xavier_normal_(self.weights_x[dim_idx][tensor_idx])
                self.weights_x[dim_idx][tensor_idx] = torch.nn.Parameter(self.weights_x[dim_idx][tensor_idx],
                                                                         requires_grad=True)
            self.weights_x[dim_idx] = nn.ParameterList(self.weights_x[dim_idx])

        self.weights_x = nn.ParameterList(self.weights_x)
        # print(f'self.weights_t type is {type(self.weights_t)}, and self.weights_x type - {type(self.weights_x)}')


        self.weigths_mixer = torch.nn.Linear(self.rank, 1)

        self._contract = _contract_dense # as the tensor factorization is already considered in the block

        # if bias:
        #     self.bias = nn.Parameter(init_std * torch.randn(*(tuple([self.out_channels]) + (1,) * self.order)))
        # else:
        self.bias = None

    def transform(self, x, output_shape=None):
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):  # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # the real FFT is skew-symmetric, so the last mode has a redundacy if our data is real in space
        # As a design choice we do the operation here to avoid users dealing with the +1
        # if we use the full FFT we cannot cut off informtion from the last mode
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """Method is similar to the approach, proposed by ...

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, x_channels, *mode_sizes = x.shape

        weights_single = weightsOuterProduct(x, self.weights_x, self.weights_t, self.weigths_mixer) # , self.complex_data
        weights_repeated = []
        permut_axes = list(permutations(range(2, len(mode_sizes) + 1), len(mode_sizes) - 1))
        # print(f'permut_axes: {[pm for pm in list(permut_axes)]}, len: {len(list(permut_axes))}')
        # raise NotImplementedError('Boop!')

        assert x_channels == len(permut_axes) * self.in_channels, \
            f'Mismatching number of channels: expected {len(permut_axes) * self.in_channels},' + \
              f' instead got {x_channels} channels.'

        for orders in permut_axes:
            orders = [0, 1] + list(orders)
            # print(f'permutating axes {orders} for tensor with shapes {weights_single.shape}')
            weights_repeated.append(weights_single.permute(*orders))

        # print(f'len(weights_repeated) is {len(weights_repeated)}')
        # raise NotImplementedError('...')

        if self.complex_data:
            raise NotImplementedError("Complex data processing has not been implemented yet.")

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient in real spatial data
        fft_dims = list(range(-len(mode_sizes), 0))

        if self.fno_block_precision == "half":
            x = x.half()

        if self.complex_data:
            x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
            # When x is real in spatial domain, the last half of the last dim is redundant.
            # See :ref:`fft_shift_explanation` for discussion of the FFT shift.
            dims_to_fft_shift = fft_dims[:-1]

        if len(fft_dims) > 1:
            x = torch.fft.fftshift(x, dim=dims_to_fft_shift)

        if self.fno_block_precision == "mixed":
            # if 'mixed', the above fft runs in full precision, but the
            # following operations run at half precision
            x = x.chalf()

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat
        out_fft = torch.zeros([batchsize, self.out_channels * len(weights_repeated), *fft_size],
                              device=x.device, dtype=out_dtype)
        
        # print(f'out_fft shape is {out_fft.shape}')

        # TODO: consider reworking this piece of code for factorized weights.

        # if current modes are less than max, start indexing modes closer to the center of the weight tensor
        # starts = [
        #     (max_modes - min(size, n_mode))
        #     for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)
        # ]

        # if contraction is separable, weights have shape (channels, modes_x, ...)
        # otherwise they have shape (in_channels, out_channels, modes_x, ...)
        # if self.separable:
        #     slices_w = [slice(None)]  # channels
        # else:
        #     slices_w = [slice(None), slice(None)]  # in_channels, out_channels
        # if self.complex_data:
        #     slices_w += [
        #         slice(start // 2, -start // 2) if start else slice(start, None)
        #         for start in starts
        #     ]
        # else:
        #     # The last mode already has redundant half removed in real FFT
        #     slices_w += [
        #         slice(start // 2, -start // 2) if start else slice(start, None)
        #         for start in starts[:-1]
        #     ]
        #     slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        # slices_w = tuple(slices_w)
        # weight = self.weight[slices_w]

        ### Pick the first n_modes modes of FFT signal along each dim

        # if separable conv, weight tensor only has one channel dim
        if self.separable:
            weight_start_idx = 1
        # otherwise drop first two dims (in_channels, out_channels)
        else:
            weight_start_idx = 2

        slices_x = [slice(None), None]  # Batch_size, channels

        for all_modes, kept_modes in zip(fft_size, list(weights_single.shape[weight_start_idx:])):
            # After fft-shift, the 0th frequency is located at n // 2 in each direction
            # We select n_modes modes around the 0th frequency (kept at index n//2) by grabbing indices
            # n//2 - n_modes//2  to  n//2 + n_modes//2       if n_modes is even
            # n//2 - n_modes//2  to  n//2 + n_modes//2 + 1   if n_modes is odd
            center = all_modes // 2
            negative_freqs = kept_modes // 2
            positive_freqs = kept_modes // 2 + kept_modes % 2

            # this slice represents the desired indices along each dim
            slices_x += [slice(center - negative_freqs, center + positive_freqs)]

        if weights_single.shape[-1] < fft_size[-1]:
            slices_x[-1] = slice(None, weights_single.shape[-1])
        else:
            slices_x[-1] = slice(None)

        # if self.complex_data:
        #     trunc_idx = [-1 for _ in weights_s]
        # else:
        #     trunc_idx = [(weights_s[0][idx].size(dim=-1)//2+1) for idx in range(len(weights_s[0]))] # assuming, all modes are the same 

        for c_idx, weights in enumerate(weights_repeated):
            slices_x[1] = slice(c_idx * weights_single.shape[0], (c_idx + 1) * weights_single.shape[0])

            slices_tupled = tuple(slices_x)

            if not self.complex_data:
                weights = weights[..., : weights.shape[-1] // 2 + 2]

            # print(f'slices_tupled is: {slices_tupled}')
            # print(f'In contraction: {out_fft[slices_tupled].shape} vs {x[slices_tupled].shape} vs {weights.shape}')
            # # raise NotImplementedError('...')
            out_fft[slices_tupled] = self._contract(x[slices_tupled], weights, separable=self.separable)

        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)])

        if output_shape is not None:
            mode_sizes = output_shape


        if len(fft_dims) > 1:
            out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])
        

        # Inverse FFT 
        if self.complex_data:
            # For complex data, we can use ifftn.
            x = torch.fft.ifftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        
        else:
            # For real data, we need to enforce Hermitian symmetry conditions for irfft.
            # On certain GPUs and for certain input sizes, this is not handled within irfftn in cuFFT, 
            # and as a result causes line artifacts.  
            # To fix this, we split the ifftn into a ifftn in (n-1) dimensions and a irfft in the last dimension,
            # although it incurs a small additional computational cost.
            
            if self.enforce_hermitian_symmetry:
                out_fft = torch.fft.ifftn(out_fft, s=mode_sizes[:-1], dim=fft_dims[:-1], norm=self.fft_norm)
                
                # Enforce Hermitian symmetry conditions for irfft
                # 0th frequency must be real
                out_fft[..., 0].imag.zero_()
                
                # Nyquist frequency must be real if the spatial size is even
                if mode_sizes[-1] % 2 == 0:
                    out_fft[..., -1].imag.zero_()
                
                # Now that the Hermitian symmetry conditions are enforced, we can use irfft on the last dimension.
                x = torch.fft.irfft(out_fft, n=mode_sizes[-1], dim=fft_dims[-1], norm=self.fft_norm)
            
            else:
                
                # If Hemrmitian symmetry is not a concern, we can use irfftn on all dimensions.
                x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)

        # print('convo module output.shape: ', x.shape)

        # if self.bias is not None:
        #     x = x + self.bias
          
        return x