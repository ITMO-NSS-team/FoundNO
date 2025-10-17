import os
import json
import warnings
import h5py

import inspect

from typing import List, Tuple

import numpy as np
import math
import torch

from neuralop.models import FNO

def setEnviron(world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def merge_dicts(*args: Tuple[dict]) -> dict:
    assert len(args) > 0 and all([isinstance(arg, dict) for arg in args]), 'Arguments of merge_dicts must be dicts!'
    output = args[0]
    for op in args[1:]:
        for key, value in op.items():
            if key in output.keys():
                raise RuntimeError('Duplicating keys in dicts, undesired behavior.')
            output[key] = value
    
    return output

class ParamContainerMeta(type):
    _param_container_instances = {}
    
    def __call__(cls, *args, **kwargs): 
        if cls not in cls._param_container_instances:
            instance = super().__call__(*args, **kwargs)
            cls._param_container_instances[cls] = instance
            
        return cls._param_container_instances[cls]
    
    def reset(self):
        self._param_container_instances = {}

class ParameterLoading(metaclass = ParamContainerMeta):
    '''
    Loading of default parameters. 
    Inspired by https://github.com/aimclub/FEDOT/blob/master/fedot/core/repository/default_params_repository.py
    '''
    
    def __init__(self, parameter_file : str = None) -> None:
        if parameter_file is None:
            parameter_file = 'utils/params/FNO_params.json'
        
        repo_folder = str(os.path.dirname(__file__))
        file = os.path.join('parameters', parameter_file)
        self._repo_path = os.path.join(repo_folder, file)
        self._repo = self._initialise_repo()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback) -> None:
        self._repo_path = None

    def _initialise_repo(self) -> dict:
        with open(self._repo_path) as repository_json_file:
            repository_json = json.load(repository_json_file)

        return repository_json

    def get_default_params_for_operator(self, param_name : str) -> dict:
        if param_name in self._repo:
            return self._repo[param_name]
        else:
            raise Exception(f'Parameter {param_name} is missing from the repo with params')
    
    def change_param_value(self, parameter_name : str, new_value) -> None:
        if type(new_value) != type(self._repo[parameter_name]):
            old_type = type(self._repo[parameter_name])
            new_type = type(new_value)
            warnings.warn(f'Possibly incorrect parameter change: from {old_type} to {new_type}.')
        self._repo[parameter_name] = new_value

class LpLoss(object):
    """
    LpLoss provides the L-p norm between two 
    discretized d-dimensional functions. Note that 
    LpLoss always averages over the spatial dimensions.

    .. note :: 
        In function space, the Lp norm is an integral over the
        entire domain. To ensure the norm converges to the integral,
        we scale the matrix norm by quadrature weights along each spatial dimension.

        If no quadrature is passed at a call to LpLoss, we assume a regular 
        discretization and take ``1 / measure`` as the quadrature weights. 

    Parameters
    ----------
    d : int, optional
        dimension of data on which to compute, by default 1
    p : int, optional
        order of L-norm, by default 2
        L-p norm: [\\sum_{i=0}^n (x_i - y_i)**p] ** (1/p)
    measure : float or list, optional
        measure of the domain, by default 1.0
        either single scalar for each dim, or one per dim

        .. note::

        To perform quadrature, ``LpLoss`` scales ``measure`` by the size
        of each spatial dimension of ``x``, and multiplies them with 
        ||x-y||, such that the final norm is a scaled average over the spatial
        dimensions of ``x``. 
    reduction : str, optional
        whether to reduce across the batch and channel dimensions
        by summing ('sum') or averaging ('mean')

        .. warning:: 

            ``LpLoss`` always reduces over the spatial dimensions according to ``self.measure``.
            `reduction` only applies to the batch and channel dimensions.
    eps : float, optional
        small number added to the denominator for numerical stability when using the relative loss

    Examples
    --------

    ```
    """

    def __init__(self, d=1, p=2, measure=1., reduction='sum', eps=1e-8):
        super().__init__()

        self.d = d
        self.p = p
        self.eps = eps
        
        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure
    
    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"
    
    def uniform_quadrature(self, x):
        """
        uniform_quadrature creates quadrature weights
        scaled by the spatial size of ``x`` to ensure that 
        ``LpLoss`` computes the average over spatial dims. 

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        quadrature : list
            list of quadrature weights per-dim
        """
        quadrature = [0.0]*self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j]/x.size(-j)
        
        return quadrature

    def reduce_all(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Params
        ------
        x: torch.Tensor
            inputs
        """
        if self.reduction == 'sum':
            x = torch.sum(x)
        else:
            x = torch.mean(x)
        
        return x

    def abs(self, x, y, quadrature=None):
        """absolute Lp-norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            quadrature weights for integral
            either single scalar or one per dimension
        """
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        const = math.prod(quadrature)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def rel(self, x, y):
        """
        rel: relative LpLoss
        computes ||x-y||/(||y|| + eps)

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        """

        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/(ynorm + self.eps)

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)
    
def standartize(matrix: np.ndarray):
    if np.isclose(torch.std(matrix).item(), 0.):
        return matrix
    else:
        return (matrix - torch.mean(matrix))/torch.std(matrix)

def load_files_hdf5(filepaths: List[str], finetune: bool = False, finetune_label: str = None, 
                    to_standardize: bool = True, original_shape: Tuple[int] = None):
    assert original_shape is not None and finetune_label is not None, 'Incorrect arguments!'
    
    data_samples = []
    for path in filepaths:
        print(path)
        with h5py.File(path, 'r') as f:
            for idx, subset in enumerate(f):
                
                has_conv = f[subset]['info'][finetune_label][()]
                if has_conv == finetune:
                    inp_sets = [torch.from_numpy(f[subset]['inputs'][channel][()]).view(original_shape) for channel in f[subset]['inputs']]
                    out_sets = [torch.from_numpy(f[subset]['outputs'][channel][()]).view(original_shape) for channel in f[subset]['outputs']]

                    if any([torch.isclose(torch.min(oset), torch.max(oset)).item() for oset in out_sets]):
                        print(f'Dropping dataset {idx} from file {path}')
                        continue
                    else:
                        print(f'Accepted dataset {idx} from file {path}')

                    if to_standardize:
                        for idx in range(len(out_sets)):
                            out_sets[idx] = standartize(out_sets[idx])

                    data_samples.append((torch.stack(inp_sets, dim = 0), torch.stack(out_sets, dim = 0)))

                    print(len(inp_sets), len(out_sets))
                    print('Shape:', [inp.shape for inp in inp_sets])
    
    return data_samples

def validate_operator(operator_class: type, args: List[str]):
    if torch.nn.Module not in operator_class.__mro__:
        raise TypeError('Fourier operator has to be a subclass of torch.nn.Module.')

    if FNO not in operator_class.__mro__:
        warnings.warn('It is recommended to use operators, inhereted from neuralop.models.FNO.')

    to_break = False
    operator_init_signature = inspect.signature(operator_class.__init__).parameters
    for argument in args:
        if argument not in operator_init_signature:
            to_break = True
            warnings.warn(f'Argument {argument} is missing from the operator class __init__ signature')

    if to_break:
        raise TypeError('Operator class, passed in the model, is missing vital properties. Further execution is terminated.')