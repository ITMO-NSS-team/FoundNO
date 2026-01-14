from typing import List, Tuple, Dict, Union, Callable
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat, einsum

from neuralop.layers.fno_block import FNOBlocks

SPATIAL_AXES = ["h", "w", "d", "ex1", "ex2"]

MERGE_SYMB = lambda args: reduce(lambda x, y: x + ' ' + y, args)

def instanceNorm(n_features: int, dim: int) -> torch.nn.Module: # torch.Tensor
    match dim:
        case 3:
            return nn.InstanceNorm1d(n_features, affine=True)
        case 4:
            return nn.InstanceNorm2d(n_features, affine=True)
        case 5:
            return nn.InstanceNorm3d(n_features, affine=True)
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

    Q = einsum(Q, K, 'b i n, b j n -> b i j') * scale # sim -> Q

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(Q.dtype).max # sim -> Q
        mask = repeat(mask, 'b j -> (b h) () j', h = h)
        Q.masked_fill_(~mask, max_neg_value) # sim -> Q

    Q = Q.softmax(dim = -1) # calculate attention sim -> Q
    return einsum(Q, V, 'b n j, b j d -> b n d')
     

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
        x = apply_operator_or_ffn(context, self.to_kv, {'b': b, 'h': h, 'c': c, 'spatials': spatial_dims})
        x = rearrange(x, 'b n ... -> b n (...)')
        q    = rearrange(q, 'b n ... -> b n (...)')
        k, v = x.chunk(2, dim = 1)

        x = self._attention(q, k, v, mask, h, self.scale)

        x = apply_operator_or_ffn(x, self.to_out, {'b': b, 'h': h, 'c': c, 'spatials': spatial_dims})

        x = x.view(b, c, *spatial_dims)
        return x


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
        # original_shape = x.shape
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
            # original_shape = context.shape

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
