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

from ..utils.YParams import YParams

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

class PercieverProjector(BaseModel):
    def __init__(self, modes=(12,12), width=64,
                 n_latents=16, n_heads=4, dropout=0.1):
        super().__init__()
        

        
        self.latent_queries = nn.Parameter(torch.randn(n_latents, width))
        # cross-attention to pool from tokens into latents
        self.cross_attn = nn.MultiheadAttention(embed_dim=width,
                                                num_heads=n_heads,
                                                dropout=dropout,
                                                batch_first=True)
        # small latent transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=n_heads,
            dim_feedforward=width*2, dropout=dropout,
            activation='gelu'
        )
        self.latent_trans = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=1)
        
        
    def forward(self, x, output_shape=None, **kwargs):
        if self.positional_embedding:
            x = self.positional_embedding(x)
        # 2) lift
        x = self.lifting(x)  # [B, C=width, T, X]
        # 3) optional pad
        if self.domain_padding:
            x = self.domain_padding.pad(x)
        B, C, T, X = x.shape
        # 4) flatten tokens: [B, T*X, C]
        tokens = x.permute(0,2,3,1).reshape(B, T*X, C)
        # 5) prepare queries: expand to batch
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        # 6) cross-attention: queries <- tokens
        lat, _ = self.cross_attn(queries, tokens, tokens)
        # 7) latent transformer: [B, n_latents, C]
        lat = self.latent_trans(lat.permute(1,0,2)).permute(1,0,2)
        # 8) summary vector: mean over latents: [B, C]
        summary = lat.mean(dim=1)
        # 9) broadcast back to spatial: [B, C, T, X]
        summary = summary.view(B, C, 1, 1).expand(-1, -1, T, X)
        # 10) add global context
        return x + summary
    
class CompositePFNO(BaseModel):
    def __init__(self,
                 n_modes: Tuple[int],
                 in_channel: int,
                 width_p: int,
                 latents_p: int,
                 out_channels: int,
                 hidden_channels: int,
                 n_layers: int = 4,
                 projection_channel_ratio: float = 2.,
                 positional_embedding: Union[str, nn.Module]="grid",
                 non_linearity: nn.Module=F.gelu,
                 norm: Literal ["ada_in", "group_norm", "instance_norm"]=None,
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

        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes

    def forward(self, x: torch.Tensor, output_shape= None, **kwargs):
        """FNO's forward pass
        
        1. Applies optional positional encoding

        2. Sends inputs through a lifting layer to a high-dimensional latent space

        3. Applies optional domain padding to high-dimensional intermediate function representation

        4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity) 

        5. If domain padding was applied, domain padding is removed

        6. Projection of intermediate function representation to the output channels

        Parameters
        ----------
        x : tensor
            input tensor


        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            
            * If None, don't specify an output shape

            * If tuple, specifies the output-shape of the **last** FNO Block

            * If tuple list, specifies the exact output-shape of each FNO Block
        """

        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        # # append spatial pos embedding if set
        # if self.positional_embedding is not None:
        #     x = self.positional_embedding(x)
        
        # x = lifting(x)

        # if self.domain_padding is not None:
        #     x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x
    
    def backward():
        pass

def fno_without_lifting(n_modes: Tuple[int], params: dict = None) -> CompositeFNO:
    _params = DEFAULT_PARAMS
    if params is not None and isinstance(params, dict):
        for key, param in params.items():
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