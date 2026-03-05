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
# from neuralop.layers.coda_blocks import CODABlocks
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.skip_connections import skip_connection
from neuralop.layers.padding import DomainPadding

from neuralop.layers.resample import resample
from neuralop.layers.embeddings import GridEmbedding2D, GridEmbeddingND

from fnofound.utils.training_utils import merge_dicts
from fnofound.layers.transformer import AttentionBlock, PrepareMultihead, PreNorm, FeedForward

# einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


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
        n_sublayers=4,
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
        neural_op_processing=True,
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

        # print(f'Initializing PeCoDA with n_heads {self.n_heads} & n_dim {self.n_dim}')

        if norm is None:
            norm_module = torch.nn.Identity
        elif norm == "instance_norm":
            norm_module = partial(
                nn.InstanceNorm2d,
                affine=False) if self.n_dim == 2 else partial(
                nn.InstanceNorm3d,
                affine=False)
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
            channel_mlp_dropout=0,
            # incremental_n_modes=incremental_n_modes,
            rank=rank,
            channel_mlp_expansion=channel_mlp_expansion,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            # joint_factorization=joint_factorization,
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

        # mixer_args = dict(n_modes=n_modes,
        #                   resolution_scaling_factor=1,
        #                   non_linearity=non_linearity,
        #                   norm='instance_norm',
        #                   fno_skip=fno_skip,
        #                   conv_module=conv_module)
        

        # We have an option to make the last operator (MLP in regular
        # Transformer block) permutation equivariant. i.e., applying the
        # operator per variable or applying the operator on the whole channel
        # (like regular FNO).

        if permutation_eq:
            # self.mixer = FNOBlocks(
            #     in_channels=self.mixer_token_codimension,
            #     out_channels=self.mixer_token_codimension,
            #     n_layers=1,
            #     **mixer_args,
            #     **shared_fno_configs,
            # )

            # self.mixer = torch.nn.Sequential(torch.nn.Linear(2 * self.mixer_token_codimension, 64),
            #                                  torch.nn.Tanh(),
            #                                  torch.nn.Linear(64, 64),
            #                                  torch.nn.Tanh(),
            #                                  torch.nn.Linear(64, 64),
            #                                  torch.nn.Tanh(),                                                          
            #                                  torch.nn.Linear(64, self.mixer_token_codimension))
            self.mixer = torch.nn.Linear(2 * self.mixer_token_codimension, self.mixer_token_codimension) 
            # Reminding: self.mixer_token_codimension = in_channels


            self.norm1 = norm_module(in_channels)
            self.mixer_in_normalizer = norm_module(self.mixer_token_codimension)
            self.mixer_out_normalizer = norm_module(in_channels)

        else:
            raise NotImplementedError('Non-permutation equivariant methods are unneccessary!')
            # self.mixer = FNOBlocks(
            #     in_channels=codimension_size,
            #     out_channels=codimension_size,
            #     n_layers=1,
            #     **mixer_args,
            #     **shared_fno_configs,
            # )
            # self.norm1 = norm_module(codimension_size)
            # self.mixer_in_normalizer = norm_module(codimension_size)
            # self.mixer_out_normalizer = norm_module(codimension_size)

    def set_latents(self, domain_shape: Tuple[int], n_latents: int = None, device: str = 'cuda'):
        '''
        Latent arrays has to match the dimensionality of the domian, otherwise the transformer architecture fails.
        '''
        if self._latent_queries is None:
            if n_latents is None:
                n_latents = self._latent_queries_features 

            assert len(domain_shape) == self.n_dim, 'Numbers of points does not match dimensionality'
            # print(f'Setting latents of shape {self.n_dim} with {domain_shape}')
            self._latent_queries_modes = domain_shape # .reshape((n_latents, -1))
            self._latent_queries = nn.Parameter(torch.randn(n_latents, *domain_shape).to(device)) #.reshape((n_latents, -1)))

            print(f'self._latent_queries after initialization {self._latent_queries.shape}')
            # print('Set!')

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

        # tokens = x #.view(x.shape[0] * x.shape[1],
                     # 1, # Assume token codim as 1
                     # *x.shape[-self.n_dim:])
        
        # normalization and attention mechanism
        tokens = self.norm1(x) # maybe, TODO: insert separate tokens not to normalize args for decoder. 

        latents = repeat(self._latent_queries, 'n ... -> b n ...', b = batch_size)

        attn, ffn = self._cross_attend_blocks
        latents = attn(latents, context = tokens) + latents
        latents = ffn(latents) + latents

        for sublayer in self._sublayers:
            attn, ffn = sublayer
            latents = attn(latents) + latents
            latents = ffn(latents)  + latents

        # How to best implement ouput queries: context = tokens 
        # context = tokens

        # tokens = torch.layer_norm(tokens)

        # print(f'Reached decoder in pecoda')

        # attn = self.decoder_cross_attn
        # # output = attn(latents, context = x) #  + latents # + tokens # + tokens # context
        # tokens = attn(tokens, context = latents)
        # print('x:', torch.mean(x), torch.std(x), 'atten:', torch.mean(tokens), torch.std(tokens) )

        output = torch.concat((tokens, latents), dim = 1)  # + x
        # print('x.shape', x.shape, 'attn.shape', attn(x, context = latents).shape)
        output = torch.moveaxis(output, 1, -1)
        # output = self.mixer_out_normalizer(output)

        # print(f'output shape before norm: {output.shape}')
        output = nn.functional.normalize(output, dim = (0, 1, 2))
        # print(f'output shape after norm: {output.shape}')

        output = self.mixer(output)
        output = torch.moveaxis(output, -1, 1)

        # output = self.mixer_out_normalizer(output)#  + tokens # + attention

        # print(f'Executed  decoder and output normalizer in pecoda')

        # reshape from shape (b*t) d h w... to b (t d) h w ...
        t = output.size(0) // batch_size
        output = output.view(
            batch_size,
            t * output.size(1),
            *output.shape[-self.n_dim:])
        # print([shp == output.shape[-len(output_shape)+idx]
        #                                      for idx, shp in enumerate(output_shape)])
        if output_shape is not None and any([shp != output.shape[-len(output_shape)+idx]
                                             for idx, shp in enumerate(output_shape)]):
            print(f'output_shape: from {output_shape, output.shape} - resampled')
            output = resample(output,
                              res_scale=[j/i for (i, j) in zip(output.shape[-self.n_dim:], output_shape)],
                              axis=list(range(-self.n_dim, 0)),
                              output_shape=output_shape)
            print(f'to {output.shape}')


        # print(f'Ready to exit forward in pecoda')
        del x, latents
        torch.cuda.empty_cache()


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