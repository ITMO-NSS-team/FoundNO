"""
Adaptation of the approach from https://github.com/ShashankSubramanian/neuraloperators-TL-scaling, 
changed to work on N-dimensional data (not only on 2D, as in the original). 
"""

from typing import Union, List, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.models import FNO, TFNO
from neuralop.models.base_model import BaseModel
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.embeddings import GridEmbeddingND, GridEmbedding2D
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.padding import DomainPadding

from utils.YParams import YParams


def getAct(activation):
    if activation == 'tanh':
        func = F.tanh
    elif activation == 'gelu':
        func = F.gelu
    elif activation == 'relu':
        func = F.relu_
    elif activation == 'elu':
        func = F.elu_
    elif activation == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{activation} is not supported')
    return func

class AdaptedFNO(FNO):
    """
    Basic FNO architecture from https://github.com/neuraloperator/neuraloperator/, taken to better specify parameters & adapt
    to the trainer's inputs, if neccessary. 
    """
    def __init__(self,
                 n_modes: Tuple[int],
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 n_layers: int=4,
                 lifting_channel_ratio: int=2,
                 projection_channel_ratio: int=2,
                 positional_embedding: Union[str, nn.Module]="grid",
                 non_linearity: nn.Module=F.gelu,
                 norm: Literal ["ada_in", "group_norm", "instance_norm"]=None,
                 complex_data: bool=False,
                 use_channel_mlp: bool=True,
                 channel_mlp_dropout: float=0,
                 channel_mlp_expansion: float=0.5,
                 channel_mlp_skip: Literal['linear', 'identity', 'soft-gating']="soft-gating",
                 fno_skip: Literal['linear', 'identity', 'soft-gating']="soft-gating",
                 resolution_scaling_factor: Union[float, List[float]]=None,
                 domain_padding: Union[float, List[float]]=None,
                 domain_padding_mode: Literal['symmetric', 'one-sided']="symmetric",
                 fno_block_precision: str="full",
                 stabilizer: str=None,
                 max_n_modes: Tuple[int]=None,
                 factorization: str=None,
                 rank: float=1.0,
                 fixed_rank_modes: bool=False,
                 implementation: str="factorized",
                 decomposition_kwargs: dict=dict(),
                 separable: bool=False,
                 preactivation: bool=False,
                 conv_module: nn.Module=SpectralConv,
                 **kwargs):
        super().__init__(n_modes = n_modes,
                         hidden_channels = hidden_channels,
                         in_channels = in_channels,
                         out_channels = out_channels,
                         lifting_channels = lifting_channel_ratio * self.hidden_channels,
                         projection_channels = projection_channel_ratio * self.hidden_channels,
                         positional_embedding = positional_embedding,
                         n_layers = n_layers,
                         resolution_scaling_factor = resolution_scaling_factor,
                         non_linearity = non_linearity,
                         stabilizer = stabilizer,
                         complex_data = complex_data,
                         fno_block_precision = fno_block_precision,
                         channel_mlp_dropout = channel_mlp_dropout,
                         channel_mlp_expansion = channel_mlp_expansion,
                         max_n_modes = max_n_modes,
                         norm = norm,
                         skip = fno_skip,
                         separable = separable,
                         preactivation = preactivation,
                         factorization = factorization,
                         rank = rank,
                         fixed_rank_modes = fixed_rank_modes,
                         implementation = implementation,
                         decomposition_kwargs = decomposition_kwargs,
                         domain_padding = domain_padding,
                         domain_padding_mode = domain_padding_mode,
                         conv_module = conv_module, **kwargs)

def fno(params_container: YParams):
    """
    params_container : YParams
        standartized parameters container, presumably loaded from a yaml-file,
        containing all the neccessary arguments of the constructor method for the FNO class.
    """

    param_dict = dict()

    param_dict['in_channels'] = params_container.in_dim
    param_dict['out_channels'] = params_container.out_dim
    param_dict['hidden_channels'] = params_container.fc_dim

    param_dict['non_linearity'] = 'gelu'


    if params_container.embed_cut > 0:
        params_container.layers = [params_container.embed_cut]*len(params_container.layers)
    param_dict['n_layers'] = params_container.layers


    if params_container.in_dim == 1:
        if params_container.mode_cut > 0:
            params_container.modes1 = [params_container.mode_cut]*len(params_container.modes1)
        param_dict['n_modes'] = (params_container.modes1,)

    elif params_container.in_dim == 2:
        if params_container.mode_cut > 0:
            params_container.modes1 = [params_container.mode_cut]*len(params_container.modes1)
            params_container.modes2 = [params_container.mode_cut]*len(params_container.modes2)
        param_dict['n_modes'] = (params_container.modes1, params_container.modes2)

    elif params_container.in_dim == 3:
        if params_container.mode_cut > 0:
            params_container.modes1 = [params_container.mode_cut]*len(params_container.modes1)
            params_container.modes2 = [params_container.mode_cut]*len(params_container.modes2)
            params_container.modes3 = [params_container.mode_cut]*len(params_container.modes3)

        param_dict['n_modes'] = (params_container.modes1, params_container.modes2,
                                 params_container.modes3)

    elif params_container.in_dim == 4:
        if params_container.mode_cut > 0:
            params_container.modes1 = [params_container.mode_cut]*len(params_container.modes1)
            params_container.modes2 = [params_container.mode_cut]*len(params_container.modes2)
            params_container.modes3 = [params_container.mode_cut]*len(params_container.modes3)
            params_container.modes4 = [params_container.mode_cut]*len(params_container.modes4)

        param_dict['n_modes'] = (params_container.modes1, params_container.modes2,
                                 params_container.modes3, params_container.modes4)
    
    else:
        raise NotImplementedError('4-dimensional (i.e. time + 3D) data is the highest dimensional data supported.')

    return AdaptedFNO(**param_dict)

DEFAULT_PARAMS = {
                    'n_modes': (0,), # Tuple[int],
                    'out_channels' : 0, # int,
                    'hidden_channels': 10, # int,
                    'n_layers': 4, # int=4,
                    'positional_embedding': "grid", # Union[str, nn.Module]=          
                    'non_linearity': F.gelu, # nn.Module=
                    'norm': None, # Literal ["ada_in", "group_norm", "instance_norm"]=
                    'complex_data': False, # bool=
                    'use_channel_mlp': True, #  bool=
                    'channel_mlp_dropout': 0, # float=
                    'channel_mlp_expansion': 0.5, # float=
                    'channel_mlp_skip': "soft-gating", # Literal['linear', 'identity', 'soft-gating']=
                    'fno_skip': "linear", # Literal['linear', 'identity', 'soft-gating']=
                    'resolution_scaling_factor': None, # : Union[Number, List[Number]]
                    'domain_padding': None, # Union[Number, List[Number]]=
                    'domain_padding_mode': "symmetric", # Literal['symmetric', 'one-sided']=
                    'fno_block_precision': "full", # str=
                    'stabilizer': None, # str=
                    'max_n_modes': None, # Tuple[int]=
                    'factorization': None, # str=
                    'rank': 1.0, # float=
                    'fixed_rank_modes': False, # bool=
                    'implementation': "factorized", # str=
                    'decomposition_kwargs': dict(), # dict=
                    'separable': False, # bool=
                    'preactivation': False, # bool=
                    'conv_module': SpectralConv, # nn.Module
}

# class CustomCallableWithLifting(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x: torch.Tensor, lifting: ChannelMLP, output_shape= None, **kwargs):
#         ctx.save_for_backward(input, weight, bias)
#         output = input @ weight.t() + bias
#         return output
    

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, weight, bias = ctx.saved_tensors
#         grad_input = grad_weight = grad_bias = None

#         # Compute gradients w.r.t inputs
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output @ weight  # Customizable
#         if ctx.needs_input_grad[1]:
#             grad_weight = grad_output.t() @ input  # Standard
#         if ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(0)  # Standard
        
#         return grad_input, grad_weight, grad_bias

class CompositeFNO(BaseModel):
    '''
    Basically, copy of neuraloperator FNO class, but without lifting in-built lifting layer. 
    '''
    def __init__(self,
                 n_modes: Tuple[int],
                 out_channels: int,
                 hidden_channels: int,
                 n_layers: int = 4,
                 projection_channel_ratio: float = 2.,
                 positional_embedding: Union[str, nn.Module]="grid",
                 non_linearity: nn.Module=F.gelu,
                 norm: Literal ["ada_in", "group_norm", "instance_norm"]=None,
                #  complex_data: bool=False,
                 use_channel_mlp: bool=True,
                 channel_mlp_dropout: float=0,
                 channel_mlp_expansion: float=0.5,
                 channel_mlp_skip: Literal['linear', 'identity', 'soft-gating']="soft-gating",
                 fno_skip: Literal['linear', 'identity', 'soft-gating']="linear",
                 resolution_scaling_factor: Union[float, List[float]]=None,
                 domain_padding: Union[float, List[float]]=None,
                 domain_padding_mode: Literal['symmetric', 'one-sided']="symmetric",
                 fno_block_precision: str="full",
                 stabilizer: str=None,
                 max_n_modes: Tuple[int]=None,
                 factorization: str=None,
                 rank: float=1.0,
                 fixed_rank_modes: bool=False,
                 implementation: str="factorized",
                 decomposition_kwargs: dict=dict(),
                 separable: bool=False,
                 preactivation: bool=False,
                 conv_module: nn.Module=SpectralConv,
                 **kwargs):
        super().__init__()
        self.n_dim = len(n_modes)
        
        # n_modes is a special property - see the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._n_modes = n_modes

        self.hidden_channels = hidden_channels
        # self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        # init lifting and projection channels using ratios w.r.t hidden channels
        # self.lifting_channel_ratio = lifting_channel_ratio
        # self.lifting_channels = int(lifting_channel_ratio * self.hidden_channels)

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = int(projection_channel_ratio * self.hidden_channels)

        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.fno_block_precision = fno_block_precision
        
        # if positional_embedding == "grid":
        #     spatial_grid_boundaries = [[0., 1.]] * self.n_dim
        #     self.positional_embedding = GridEmbeddingND(in_channels=self.in_channels,
        #                                                 dim=self.n_dim, 
        #                                                 grid_boundaries=spatial_grid_boundaries)
        # elif isinstance(positional_embedding, GridEmbedding2D):
        #     if self.n_dim == 2:
        #         self.positional_embedding = positional_embedding
        #     else:
        #         raise ValueError(f'Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}')
        # elif isinstance(positional_embedding, GridEmbeddingND):
        #     self.positional_embedding = positional_embedding
        # elif positional_embedding == None:
        #     self.positional_embedding = None
        # else:
        #     raise ValueError(f"Error: tried to instantiate FNO positional embedding with {positional_embedding},\
        #                       expected one of \'grid\', GridEmbeddingND")
        
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode
        # self.complex_data = self.complex_data

        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            resolution_scaling_factor=resolution_scaling_factor,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=conv_module,
            n_layers=n_layers,
            **kwargs
        )
        
        # if adding a positional embedding, add those channels to lifting
        # lifting_in_channels = self.in_channels
        # if self.positional_embedding is not None:
        #     lifting_in_channels += self.n_dim
        # if lifting_channels is passed, make lifting a Channel-Mixing MLP
        # with a hidden layer of size lifting_channels
        # if self.lifting_channels:
        #     self.lifting = ChannelMLP(
        #         in_channels=lifting_in_channels,
        #         out_channels=self.hidden_channels,
        #         hidden_channels=self.lifting_channels,
        #         n_layers=2,
        #         n_dim=self.n_dim,
        #         non_linearity=non_linearity
        #     )
        # otherwise, make it a linear layer
        # else:
        #     self.lifting = ChannelMLP(
        #         in_channels=lifting_in_channels,
        #         hidden_channels=self.hidden_channels,
        #         out_channels=self.hidden_channels,
        #         n_layers=1,
        #         n_dim=self.n_dim,
        #         non_linearity=non_linearity
        #     )
        # Convert lifting to a complex ChannelMLP if self.complex_data==True
        # if self.complex_data:
        #     self.lifting = ComplexValued(self.lifting)

        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        # if self.complex_data:
        #     self.projection = ComplexValued(self.projection)
    
    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes

    # def forward(self, x: torch.Tensor, lifting: ChannelMLP, output_shape= None, **kwargs):
    #     """FNO's forward pass
        
    #     1. Applies optional positional encoding

    #     2. Sends inputs through a lifting layer to a high-dimensional latent space

    #     3. Applies optional domain padding to high-dimensional intermediate function representation

    #     4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity) 

    #     5. If domain padding was applied, domain padding is removed

    #     6. Projection of intermediate function representation to the output channels

    #     Parameters
    #     ----------
    #     x : tensor
    #         input tensor
        
    #     lifting : ChannelMLP
    #         lifting element of the fourier neural operator, matching the requirements, posed by the training/inference equation

    #     output_shape : {tuple, tuple list, None}, default is None
    #         Gives the option of specifying the exact output shape for odd shaped inputs.
            
    #         * If None, don't specify an output shape

    #         * If tuple, specifies the output-shape of the **last** FNO Block

    #         * If tuple list, specifies the exact output-shape of each FNO Block
    #     """

    #     if output_shape is None:
    #         output_shape = [None]*self.n_layers
    #     elif isinstance(output_shape, tuple):
    #         output_shape = [None]*(self.n_layers - 1) + [output_shape]

    #     # append spatial pos embedding if set
    #     if self.positional_embedding is not None:
    #         x = self.positional_embedding(x)
        
    #     x = lifting(x)

    #     if self.domain_padding is not None:
    #         x = self.domain_padding.pad(x)

    #     for layer_idx in range(self.n_layers):
    #         x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

    #     if self.domain_padding is not None:
    #         x = self.domain_padding.unpad(x)

    #     x = self.projection(x)

    #     return x
    
    def backward():
        pass

def fno_without_lifting(n_modes: Tuple[int], params: dict = None) -> CompositeFNO:
    _params = DEFAULT_PARAMS
    if params is not None and isinstance(params, dict):
        for key, param in params:
            _params[key] = param
    _params['n_modes'] = n_modes
    print('Parameters:')
    print(_params)
    return CompositeFNO(**_params)

def lifting_multiple_eq(input_channels: List[int], n_dim: int, 
                        hidden_channels: int, non_linearity: nn.Module = F.gelu) -> List[ChannelMLP]:
    # assert len(input_channels) == len(n_dim), 'Mismatching numbers of input channels and dimensionalities.'
    return [ChannelMLP(in_channels=input_channels[idx],
                       hidden_channels=hidden_channels,
                       out_channels=hidden_channels,
                       n_layers=1,
                       n_dim=n_dim,
                       non_linearity=non_linearity)
            for idx in range(len(input_channels))]