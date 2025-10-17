from typing import List, Tuple, Union, Dict, Callable
from functools import partial
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, einsum

from functools import reduce

from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.coda_blocks import CODABlocks
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.skip_connections import skip_connection
from neuralop.layers.padding import DomainPadding

from neuralop.layers.resample import resample
from neuralop.layers.embeddings import GridEmbedding2D, GridEmbeddingND

from utils.training_utils import merge_dicts

# einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

SPATIAL_AXES = ["h", "w", "d", "ex1", "ex2"]

MERGE_SYMB = lambda args: reduce(lambda x, y: x + ' ' + y, args)

def instanceNorm(n_features: int, dim: int) -> torch.nn.Module: # torch.Tensor
    match dim:
        case 3:
            return torch.nn.InstanceNorm1d(n_features)
        case 4:
            return torch.nn.InstanceNorm2d(n_features)
        case 5:
            return torch.nn.InstanceNorm3d(n_features)
        case _:
            raise NotImplementedError(f'Can not construct InstanceNorm for {len(dim)}-dimensional tensor')


def is_index_in_slice(index: int, slice_obj: Union[slice, int], sequence_length: int):
    """
    Checks if a given index is "contained" within a slice object's definition
    for a sequence of a specific length.

    Args:
        index (int): The index to check.
        slice_obj (slice): The slice object.
        sequence_length (int): The length of the sequence the slice would apply to.

    Returns:
        bool: True if the index is within the slice's range, False otherwise.
    """
    if isinstance(slice_obj, int):
        return index == slice_obj
    else:
        start, stop, step = slice_obj.indices(sequence_length)

        if step > 0:
            if not (start <= index < stop):
                return False
            if (index - start) % step != 0:
                return False
        elif step < 0:
            if not (stop < index <= start):
                return False
            if (start - index) % abs(step) != 0:
                return False
        else:
            return False
        return True

def unify_indexing_op(arg: list, idx: Union[int, slice]):
    return [arg[idx],] if isinstance(idx, int) else arg[idx]

def get_einsymbols(tensor: Union[int, np.array, torch.Tensor], initial_axes: List[str] = []) -> str:
    if isinstance(tensor, torch.Tensor) or isinstance(tensor, np.ndarray):
        tensor = tensor.ndim 
    if tensor < len(initial_axes):
        raise ValueError('Too few axes in data.')
    
    spatial_axes_len = tensor - len(initial_axes)
    einstr = initial_axes + SPATIAL_AXES[:spatial_axes_len]
    return MERGE_SYMB(einstr), einstr

def group_einsymbols(einlist: List[str], group_idxs: Tuple[Union[int, slice]], grouped_axes_pos: int = 0):
    einstr_merging = reduce(lambda x, y: x+y, [unify_indexing_op(einlist, c_idx) for c_idx in group_idxs])
    
    einstr_merged = '(' + MERGE_SYMB(einstr_merging) + ')'
    einstr_remaining = [einlist[c_idx] for c_idx in range(len(einlist)) 
                        if all([not is_index_in_slice(c_idx, elem, len(einlist))
                               for elem in group_idxs])] # )

    einstr_remaining.insert(grouped_axes_pos, einstr_merged)

    return MERGE_SYMB(einstr_remaining), einstr_merging

class PrepareMultihead(nn.Module):
    def __init__(self, heads_dim: int = 1, n_heads: int = 1):
        super().__init__()
        self._n_heads  = n_heads
        self._heads_dim = heads_dim

    def forward(self, x: torch.Tensor):
        shape = [1,] * x.ndim
        shape[self._heads_dim] = self._n_heads
        return x.repeat(*shape)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), #  * 2
                                 torch.nn.ReLU(),
                                 nn.Linear(dim * mult, dim))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.net(x)
        x = x.transpose(1, -1)
        return x

def apply_operator_or_ffn(arg: torch.Tensor, model: torch.nn.Module, dims: Dict[str, Union[int, List[int]]]):
    # out = einsum(attn, v, 'b n j, b j d -> b n d')
    if arg.ndim > 3:
        arg = rearrange(arg, 'bh n ... -> bh n (...)')
    if isinstance(model, nn.Linear):
        arg = rearrange(arg, '(b h) n d -> b d (h n)', h = dims['h'])
    elif isinstance(model, FNOBlocks):
        arg = arg.view(dims['b'], dims['c']*dims['h'], *dims['spatials'])
    elif isinstance(model, PrepareMultihead):
        pass
    else:
        print(f'Other to_out model of type: {type(model)}')

    arg = model(arg)
    if isinstance(model, nn.Linear):
        arg = rearrange(arg, 'b d n -> b n d')
    return arg

def attention_vanilla(Q: torch.Tensor,
                      K: torch.Tensor,
                      V: torch.Tensor,
                      mask: torch.Tensor = None,
                      h: int = 1, scale: float = 1.):
    Q, K, V = map(lambda t: rearrange(t, 'b (h d) n -> (b h) d n', h = h), (Q, K, V))

    sim = einsum(Q, K, 'b i n, b j n -> b i j') * scale

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h = h)
        sim.masked_fill_(~mask, max_neg_value)

    sim = sim.softmax(dim = -1) # calculate attention
    return einsum(sim, V, 'b n j, b j d -> b n d')
     

def attention_fourier(Q: torch.Tensor,
                      K: torch.Tensor,
                      V: torch.Tensor,
                      mask: torch.Tensor = None,
                      h: int = 1, scale: float = 1.):
    q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> (b h) d n', h = h), (q, k, v))

    sim = einsum(q, k, 'b i n, b j n -> b i j') * scale

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h = h)
        sim.masked_fill_(~mask, max_neg_value)

    return sim.softmax(dim = -1)


class AttentionBlock(nn.Module):
    '''
    Implemented, as in https://github.com/lucidrains/perceiver-pytorch/
    '''
    def __init__(self, 
                 query_dim: int,
                 context_dim: int = None,
                 heads: int = 8,
                 dim_head: int = 64,
                 to_q_model: torch.nn.Module = nn.Linear,
                 to_q_params: dict = None,
                 to_kv_model: torch.nn.Module = nn.Linear,
                 to_kv_params: dict = None,                 
                 to_out_model: torch.nn.Module = nn.Linear,
                 to_out_params: dict = None,
                 channel_wise: bool = True,
                 attention: Callable = attention_vanilla):
        if not isinstance(to_q_model, nn.Linear) and to_q_params is None:
            raise ValueError('Custom nn.Module to get Queries should can not use default params.')
        elif isinstance(to_q_model, nn.Linear) and to_q_params is None:
            to_q_params = {'in_features': query_dim, 
                           'out_features': inner_dim,
                           'bias': False}

        if not isinstance(to_kv_model, nn.Linear) and to_kv_params is None:
            raise ValueError('Custom nn.Module to get Keys & Values should can not use default params.')
        elif isinstance(to_q_model, nn.Linear) and to_q_params is None:
            to_kv_params = {'in_features': context_dim, 
                            'out_features': inner_dim * 2,
                            'bias': False}

        if not isinstance(to_out_model, nn.Linear) and to_out_params is None:
            raise ValueError('Custom nn.Module to map outputs should can not use default params.')
        elif isinstance(to_q_model, nn.Linear) and to_q_params is None:
            to_q_params = {'in_features': inner_dim, 
                           'out_features': query_dim,
                           'bias': True}

        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = to_q_model(**to_q_params)    # nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = to_kv_model(**to_kv_params)  # nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = to_out_model(**to_out_params) # nn.Linear(inner_dim, query_dim)
        self._channel_wise = channel_wise
        self._attention = attention

    def forward(self, x, context = None, mask = None, batch_size: int = -1):
        b = x.shape[0]
        h = self.heads
        c = x.shape[1]
        spatial_dims = x.shape[2:]

        q = apply_operator_or_ffn(x, self.to_q, {'b': b, 'h': h, 'c': c, 'spatials': spatial_dims})

        if context is None:
            context = x if context is None else context

        if isinstance(self.to_kv, FNOBlocks) and self._channel_wise:
            context = rearrange(context, 'b n ... -> (b n) ...')
            context = torch.unsqueeze(context, 1)
        temp = apply_operator_or_ffn(context, self.to_kv, {'b': b, 'h': h, 'c': c, 'spatials': spatial_dims})
        temp = rearrange(temp, 'b n ... -> b n (...)')
        q    = rearrange(q, 'b n ... -> b n (...)')
        k, v = temp.chunk(2, dim = 1)

        out = self._attention(q, k, v, mask, h, self.scale)

        out = apply_operator_or_ffn(out, self.to_out, {'b': b, 'h': h, 'c': c, 'spatials': spatial_dims})

        out = out.view(b, c, *spatial_dims)
        return out


class PreNorm(nn.Module):
    def __init__(self, input_info: Tuple[int], fn: torch.nn.Module, context_info: Tuple[int] = None):
        super().__init__()
        self.fn = fn
        self.norm = instanceNorm(n_features = input_info[0],
                                 dim = input_info[1]) # nn.LayerNorm(norm_dim)
        self.norm_context = None if context_info is None else instanceNorm(n_features = context_info[0],
                                                                           dim = context_info[1])
        self._dim_labels = ['b', 'c']

        self._ein_init = None; self._ein_norm = None
        self._ctx_ein_init = None; self._ctx_ein_norm = None


    def forward(self, x, **kwargs):
        original_shape = x.shape
        if self._ein_init is None:
            self._ein_init = get_einsymbols(x, self._dim_labels)
            self._ein_norm = group_einsymbols(self._ein_init[1], (0, 1), 0) # TODO: validate normalization approach slice(2, None, None)
        
        # x = rearrange(x, self._ein_init[0] + ' -> ' + self._ein_norm[0]) # 'b c ... -> (b ...) c')
        # print(f'Before normalization x shape is {x.shape}')
        # raise NotImplementedError()
        x = self.norm(x)
        # x = rearrange(x, self._ein_norm[0] + ' -> ' + self._ein_init[0], b = original_shape[0], 
        #               **{value: original_shape[idx + len(self._dim_labels)]
        #                  for idx, value in enumerate(self._ein_norm[1][1:])})
        
        if self.norm_context is not None:
            context = kwargs['context']
            original_shape = context.shape

            if self._ctx_ein_init is None:
                self._ctx_ein_init = get_einsymbols(context, self._dim_labels)
                self._ctx_ein_norm = group_einsymbols(self._ctx_ein_init[1], (0, slice(2, None, None)), 0)

            # context = rearrange(context, self._ctx_ein_init[0] + ' -> ' + self._ctx_ein_norm[0])
            context = self.norm_context(context)
            # context = rearrange(context, self._ctx_ein_norm[0] + ' -> ' + self._ctx_ein_init[0], 
            #                     b = original_shape[0], 
            #                     **{value: original_shape[idx + len(self._dim_labels)]
            #                         for idx, value in enumerate(self._ctx_ein_norm[1][1:])})
            kwargs.update(context = context)

        output = self.fn(x, **kwargs)
        return output

class PeCODALayer(nn.Module):
    """Co-domain Attention Blocks (PeCODALayer) implement the transformer
    architecture in the operator learning framework, as described in [1]_.

    Parameters
    ----------
    n_modes : list
        Number of modes for each dimension used in K, Q, V operator.
    n_heads : int
        Number of heads for the attention mechanism.
    token_codimension : int
        Co-dimension of each variable, i.e. number of
        output channels associated with each variable.
    head_codimension : int
        Co-dimension of each output token for each head.
    codimension_size : int
        Size of the codimension for the whole function. Only used for permutation_eq = False.
    per_channel_attention : bool, optional
        Whether to use per-channel attention. Default is True (overwrites token_codimension to 1).
    permutation_eq : bool, optional
        Whether to use permutation equivariant mixer layer after the attention mechanism.
    norm : literal `{'instance_norm'}` or None
        Normalization module to be used. If 'instance_norm', instance normalization
        is applied to the token outputs of the attention module.
        Defaults to `'instance_norm'`
    nonlinear_attention : bool, optional
        Whether to use non-linear activation for K, Q, V operator.
    scale : int
        Scale for downsampling Q, K functions before calculating the attention matrix.
        Higher scale will downsample more.
    resolution_scaling_factor : float
        Scaling factor for the output.
    
    Other Parameters
    ----------------
    incremental_n_modes : list
        Incremental number of modes for each dimension (for incremental training).
    use_channel_mlp : bool, optional
        Whether to use MLP layers to parameterize skip connections. Default is True.
    channel_mlp_expansion : float, optional
        Expansion parameter for self.channel_mlp. Default is 0.5.
    non_linearity : callable
        Non-linearity function to be used.
    preactivation : bool, optional
        Whether to use preactivation. Default is False.
    fno_skip : str, optional
        Type of skip connection to be used. Default is 'linear'.
    channel_mlp_skip : str, optional
        Module to use for ChannelMLP skip connections. Default is "linear".
    separable : bool, optional
        Whether to use separable convolutions. Default is False.
    factorization : str, optional
        Type of factorization to be used. Default is 'tucker'.
    rank : float, optional
        Rank of the factorization. Default is 1.0.
    conv_module : callable
        Spectral convolution module to be used.
    joint_factorization : bool, optional
        Whether to factorize all spectralConv weights as one tensor. Default is False.
    
    References
    ----------
    .. [1]: M. Rahman, R. George, M. Elleithy, D. Leibovici, Z. Li, B. Bonev, 
        C. White, J. Berner, R. Yeh, J. Kossaifi, K. Azizzadenesheli, A. Anandkumar (2024).
        "Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs."
        arxiv:2403.12553
    """
    def __init__(
        self,
        in_channels: int,
        n_modes:List[int],
        n_heads=1,
        n_sublayers=6,
        head_codimension=None,
        codimension_size=None,
        per_channel_attention=True,
        permutation_eq=True,
        norm="instance_norm",
        nonlinear_attention=False,
        scale=None,
        resolution_scaling_factor=None,
        incremental_n_modes=None,
        non_linearity=F.gelu,
        use_channel_mlp=False,
        channel_mlp_expansion=1.0,
        fno_skip='linear',
        channel_mlp_skip='linear',
        preactivation=False,
        separable=False,
        factorization='tucker',
        rank=1.0,
        joint_factorization=False,
        conv_module=SpectralConv,
        fixed_rank_modes=False,
        implementation='factorized',
        decomposition_kwargs=None,
        neural_op_processing = False,
        # dropout_perc = 0.1,
        **_kwargs,
    ):
        super().__init__()

        self._n_sublayers = n_sublayers

        n_lat_perc = in_channels

        # Co-dimension of each variable/token. The token embedding space is
        # identical to the variable space, so their dimensionalities are equal.
        if per_channel_attention:
            # for per channel attention, forcing the values of token dims
            in_channels = 1
            head_codimension = 1

        # codim of attention from each head
        self.head_codimension = (head_codimension
                                 if head_codimension is not None
                                 else in_channels)

        self.n_heads = n_heads  # number of heads
        self.resolution_scaling_factor = resolution_scaling_factor
        self.n_dim = len(n_modes)

        print(f'Initializing PeCoDA with n_heads {self.n_heads} & n_dim {self.n_dim}')

        if norm is None:
            norm_module = torch.nn.Identity
        elif norm == "instance_norm":
            norm_module = partial(
                nn.InstanceNorm2d,
                affine=True) if self.n_dim == 2 else partial(
                nn.InstanceNorm3d,
                affine=True)
        else:
            raise ValueError(f"Unknown normalization type {norm}")

        # K,Q,V operator with or without non_liniarity
        if nonlinear_attention:
            kqv_activation = non_linearity
        else:
            kqv_activation = torch.nn.Identity()

        self.permutation_eq = permutation_eq

        self.codimension_size = codimension_size
        self.mixer_token_codimension = in_channels

        mixer_modes = [int(i*scale) for i in n_modes]

        if decomposition_kwargs is None:
            decomposition_kwargs = {}

        shared_fno_configs = dict(
            use_channel_mlp=use_channel_mlp,
            preactivation=preactivation,
            channel_mlp_skip=channel_mlp_skip,
            mlp_dropout=0,
            incremental_n_modes=incremental_n_modes,
            rank=rank,
            channel_mlp_expansion=channel_mlp_expansion,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
        )

        inp_arr_proc_args = dict(in_channels = in_channels,
                                 out_channels = 2*self.n_heads * self.head_codimension, #  2 - for k-v split
                                 n_modes=mixer_modes,
                                 non_linearity=kqv_activation,
                                 fno_skip='linear',
                                 norm=None,
                                 n_layers=1)

        lat_proc_args_q = dict(in_channels=n_lat_perc,
                               out_channels=self.n_heads * self.head_codimension,
                               n_modes=mixer_modes,
                               non_linearity=kqv_activation,
                               fno_skip='linear',
                               norm=None,
                               n_layers=1)
        lat_proc_args_kv = deepcopy(lat_proc_args_q)
        lat_proc_args_kv['out_channels'] = 2 * self.n_heads * self.head_codimension

        kv_dec_arg = dict(in_channels=self.head_codimension,
                          out_channels=2*self.n_heads * self.head_codimension,
                          n_modes=mixer_modes,
                          non_linearity=kqv_activation,
                          fno_skip='linear',
                          norm=None,
                          n_layers=1,
                            )

        q_dec_args = dict(in_channels=in_channels,
                          out_channels=self.n_heads * self.head_codimension,
                          n_modes=mixer_modes,
                          non_linearity=kqv_activation,
                          fno_skip='linear',
                          norm=None,
                          n_layers=1,
                         )

        self._latent_queries = None # nn.Parameter(torch.randn(n_lat_perc, *n_modes).reshape((n_lat_perc, -1))) # [N x 1 x D]
        self._latent_queries_modes = None
        self._latent_queries_features = n_lat_perc

        encoder = AttentionBlock(query_dim     = self._latent_queries_features,
                                 context_dim   = in_channels,
                                 heads         = self.n_heads, 
                                 dim_head      = self.head_codimension,
                                 to_q_model    = PrepareMultihead,
                                 to_q_params   = {'heads_dim': 1, 'n_heads': self.n_heads},
                                 to_kv_model   = FNOBlocks,
                                 to_kv_params  = merge_dicts({'resolution_scaling_factor': 1,
                                                             'conv_module': conv_module},
                                                             inp_arr_proc_args,
                                                             shared_fno_configs),
                                 to_out_model  = nn.Linear, #nn.Identity,
                                 to_out_params = {'in_features': int(inp_arr_proc_args['out_channels']/2.),
                                                 'out_features': self._latent_queries_features},
                                 channel_wise  = False)

        self._cross_attend_blocks = nn.ModuleList([PreNorm((self._latent_queries_features, 2 + len(n_modes)),
                                                           encoder, 
                                                           context_info = (in_channels, 2 + len(n_modes))),# token_codimension),
                                                   PreNorm((self._latent_queries_features, 2 + len(n_modes)), #  * self.n_heads
                                                           FeedForward(dim = self._latent_queries_features))]) # * self.n_heads
        
        if neural_op_processing:
            proc_models = FNOBlocks
            proc_models_q_params = merge_dicts({'resolution_scaling_factor': 1,
                                                'conv_module': conv_module},
                                               lat_proc_args_q,
                                               shared_fno_configs)
            proc_models_kv_params = merge_dicts({'resolution_scaling_factor': 1,
                                                 'conv_module': conv_module},
                                                lat_proc_args_kv,
                                                shared_fno_configs)
        else:
            proc_models = nn.Linear #(in_features=, out_features=)
            proc_models_q_params = {'in_features' : self._latent_queries_features,
                                    'out_features': self.n_heads * self.head_codimension}
            proc_models_kv_params = {'in_features' : self._latent_queries_features,
                                     'out_features': 2 * self.n_heads * self.head_codimension}

        get_proc_layer = lambda: AttentionBlock(query_dim     = self.head_codimension,
                                                context_dim   = self.head_codimension,
                                                heads         = self.n_heads, 
                                                dim_head      = self.head_codimension,
                                                to_q_model    = proc_models,
                                                to_q_params   = proc_models_q_params,
                                                to_kv_model   = proc_models,
                                                to_kv_params  = proc_models_kv_params,
                                                to_out_model  = nn.Linear,
                                                to_out_params = {'in_features': self.n_heads * self.head_codimension,
                                                                 'out_features': self._latent_queries_features},
                                                channel_wise  = False)

        self._sublayers = nn.ModuleList([])
        for idx in range(self._n_sublayers):
            self._sublayers.append(nn.ModuleList([PreNorm((self._latent_queries_features, 2 + len(n_modes)),
                                                          get_proc_layer()),
                                                  PreNorm((self._latent_queries_features, 2 + len(n_modes)),
                                                          FeedForward(dim = self._latent_queries_features))]))

        # Output model in decoder represents multi-head projection
        output_model = FNOBlocks #if self.n_heads * self.head_codimension != in_channels else nn.Identity
        decoder = AttentionBlock(query_dim     = in_channels,
                                 context_dim   = self.head_codimension,
                                 heads         = self.n_heads, 
                                 dim_head      = self.head_codimension,
                                 to_q_model    = FNOBlocks,
                                 to_q_params   = merge_dicts({'resolution_scaling_factor': 1,
                                                             'conv_module': conv_module},
                                                             q_dec_args,
                                                             shared_fno_configs),
                                 to_kv_model   = FNOBlocks,
                                 to_kv_params  = merge_dicts({'resolution_scaling_factor': 1,
                                                             'conv_module': conv_module},
                                                             kv_dec_arg,
                                                             shared_fno_configs),
                                 to_out_model  = output_model,
                                 to_out_params = merge_dicts({'in_channels': n_heads * self.head_codimension, # self.n_heads * 
                                                             'out_channels': in_channels,
                                                             'n_modes': n_modes,
                                                             'resolution_scaling_factor': 1,
                                                             # args below are shared with KQV blocks
                                                             'non_linearity': non_linearity, # torch.nn.Identity(),
                                                             'fno_skip': 'linear',
                                                             'norm': None,
                                                             'conv_module': conv_module,
                                                             'n_layers': 1},
                                                             shared_fno_configs),
                                 channel_wise  = False)

        self.decoder_cross_attn = PreNorm((in_channels, 2 + len(n_modes)), 
                                          decoder,
                                          context_info = (self.head_codimension, 2 + len(n_modes)))

        self.attention_normalizer = norm_module(in_channels)

        mixer_args = dict(n_modes=n_modes,
                          resolution_scaling_factor=1,
                          non_linearity=non_linearity,
                          norm='instance_norm',
                          fno_skip=fno_skip,
                          conv_module=conv_module)
        

        # We have an option to make the last operator (MLP in regular
        # Transformer block) permutation equivariant. i.e., applying the
        # operator per variable or applying the operator on the whole channel
        # (like regular FNO).

        if permutation_eq:
            self.mixer = FNOBlocks(
                in_channels=self.mixer_token_codimension,
                out_channels=self.mixer_token_codimension,
                n_layers=2,
                **mixer_args,
                **shared_fno_configs,
            )

            self.norm1 = norm_module(in_channels)
            self.mixer_in_normalizer = norm_module(self.mixer_token_codimension)
            self.mixer_out_normalizer = norm_module(
                self.mixer_token_codimension)

        else:
            self.mixer = FNOBlocks(
                in_channels=codimension_size,
                out_channels=codimension_size,
                n_layers=2,
                **mixer_args,
                **shared_fno_configs,
            )
            self.norm1 = norm_module(codimension_size)
            self.mixer_in_normalizer = norm_module(codimension_size)
            self.mixer_out_normalizer = norm_module(codimension_size)

    def set_latents(self, domain_shape: Tuple[int], n_latents: int = None, device: str = 'cuda'):
        '''
        Latent arrays has to match the dimensionality of the domian, otherwise the transformer architecture fails.
        '''
        if self._latent_queries is None:
            if n_latents is None:
                n_latents = self._latent_queries_features 

            assert len(domain_shape) == self.n_dim, 'Numbers of points does not match dimensionality'
            print(f'Setting latents of shape {self.n_dim} with {domain_shape}')
            self._latent_queries_modes = domain_shape # .reshape((n_latents, -1))
            self._latent_queries = nn.Parameter(torch.randn(n_latents, *domain_shape).to(device)) #.reshape((n_latents, -1)))

    def forward(self, x: torch.Tensor, output_shape=None):
        """
        CoDANO's forward pass. 

        * If ``self.permutation_eq == True``, computes the permutation-equivariant forward pass,\
            where the mixer FNO block is applied to each token separately, making\
            the final result equivariant to any permutation of tokens.

        * If ``self.permutation_eq == True``, the mixer is applied to the whole function together,\
            and tokens are treated as channels within the same function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (b, t * d, h, w, ...), where
            b is the batch size, t is the number of tokens, and d is the token codimension.
        """
        # print(f'x.shape in forward:', x.shape)
        if self.resolution_scaling_factor is not None and output_shape is None:
            output_shape = [int(i * j) for (i,j) in zip(x.shape[-self.n_dim:], self.resolution_scaling_factor)]

        self.set_latents(x.shape[-self.n_dim:], device=x.device)

        if self.permutation_eq:
            return self._forward_equivariant(x, output_shape=output_shape)
        else:
            return self._forward_non_equivariant(x, output_shape=output_shape)

    def _forward_equivariant(self, x, output_shape=None, mask = None, queries = None):
        """
        Forward pass with a permutation equivariant mixer layer after the
        attention mechanism. Shares the same mixer layer for all tokens, meaning
        that outputs are equivariant to permutations of the tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (b, t * d, h, w, ...), where
            b is the batch size, t is the number of tokens, and d is the token codimension.
        """
        batch_size = x.shape[0]
        input_shape = x.shape[-self.n_dim:]

        t = x.size(1) #// self.token_codimension

        tokens = x #.view(x.shape[0] * x.shape[1],
                        # 1, # Assume token codim as 1
                        # *x.shape[-self.n_dim:])
        
        # normalization and attention mechanism
        # tokens_norm = self.norm1(tokens)

        latents = repeat(self._latent_queries, 'n ... -> b n ...', b = batch_size)

        attn, ffn = self._cross_attend_blocks
        latents = attn(latents, context = tokens) + latents
        latents = ffn(latents) + latents
        # print(f'ffn(attn(latents, context = tokens)) {ffn(attn(latents, context = tokens)).shape}, latents {latents.shape}')
        # latents = ffn(attn(latents, context = tokens)) + latents

        for sublayer in self._sublayers:
            attn, ffn = sublayer
            latents = attn(latents) + latents
            latents = ffn(latents)  + latents

        # How to best implement ouput queries: context = tokens 
        # context = tokens

        # tokens = torch.layer_norm(tokens)

        attn = self.decoder_cross_attn
        output = attn(latents, context = tokens) #  + latents # + tokens # + tokens # context
        
        for i in range(self.mixer.n_layers):
            output = self.mixer(output, index=i, output_shape=input_shape)
        output = self.mixer_out_normalizer(output) # + attention

        # reshape from shape (b*t) d h w... to b (t d) h w ...
        t = output.size(0) // batch_size
        output = output.view(
            batch_size,
            t * output.size(1),
            *output.shape[-self.n_dim:])
        
        if output_shape is not None:
            output = resample(output,
                              res_scale=[j/i for (i, j) in zip(output.shape[-self.n_dim:], output_shape)],
                              axis=list(range(-self.n_dim, 0)),
                              output_shape=output_shape)

        return output

    def _forward_non_equivariant_old(self, x, output_shape=None):
        """
        Forward pass with a non-permuatation equivariant mixer layer and normalizations.
        After attention, the tokens are stacked along the channel dimension before mixing,
        meaning that the outputs are not equivariant to the ordering of the tokens.

        Parameters
        ----------
        x: torch.tensor. 
            Has shape (b, t*d, h, w, ...) 
            where, t = number of tokens, d = token codimension
        """
        raise NotImplementedError('Method has not been implemented yet')

        batch_size = x.shape[0]
        input_shape = x.shape[-self.n_dim:]

        # assert x.shape[1] % self.token_codimension == 0,\
        #       "Number of channels in x should be divisible by token_codimension"

        # reshape from shape b (t*d) h w ... to (b*t) d h w ...
        t = x.size(1) #// self.token_codimension

        tokens = x.view(
            x.size(0) * t,
            1, # Assume token codim as 1
            *x.shape[-self.n_dim:])
        
        # normalization and attention mechanism
        # tokens_norm = self.norm1(tokens)

        latents = repeat(self._latent_queries, 'n d -> b n d', b = batch_size)
        print(f'Context shape before all operations is {context.shape}')

        x = self._cross_attend_blocks(tokens, latents)

        for layer in self._layers:
            x = layer(x)

        # How to best implement ouput queries: context = tokens 
        context = tokens
        output = self.decoder_cross_attn(x, context)

        output = output.view(batch_size, t * output.size(2), *output.shape[-self.n_dim:])

        output = self.mixer_in_normalizer(output)
        for i in range(self.mixer.n_layers):
            output = self.mixer(output, index=i, output_shape=input_shape)

        output = self.mixer_out_normalizer(output) #   + attention

        # output = self.compute_decoder_attention(tokens_norm, attention, batch_size)

        # for i in range(self.mixer.n_layers):
        #     output = self.mixer(output, index=i, output_shape=input_shape)
        # output = self.mixer_out_normalizer(output) + attention        

        # t = output.size(0) // batch_size
        # output = output.view(
        #     batch_size,
        #     t * output.size(1),
        #     *output.shape[-self.n_dim:])
        
        # if output_shape is not None:
        #     output = resample(output,
        #                       res_scale=[j/i for (i, j) in zip(output.shape[-self.n_dim:], output_shape)],
        #                       axis=list(range(-self.n_dim, 0)),
        #                       output_shape=output_shape)


        if output_shape is not None:
            output = resample(output,
                              res_scale=[j/i for (i, j) in zip(output.shape[-self.n_dim:], output_shape)],
                              axis=list(range(-self.n_dim, 0)),
                              output_shape=output_shape)

        return output