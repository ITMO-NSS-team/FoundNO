"""Standard FP32 AdamW applied to storage-sharing chunks of large parameters.

Complex Fourier weights are exposed as real/imaginary views, as in PyTorch AdamW.
Chunking limits temporary optimizer allocations without quantizing any state.
"""
from __future__ import annotations

import torch
from torch import nn


class ChunkedAdamW(torch.optim.AdamW):
    def __init__(self, param_groups, *, chunk_elements: int, **kwargs):
        if chunk_elements < 1:
            raise ValueError("chunk_elements must be positive")
        self.original_parameters = []
        self.chunk_views = []
        groups = []
        for source_group in param_groups:
            group = dict(source_group)
            chunks = []
            for parameter in source_group["params"]:
                if not parameter.is_contiguous():
                    raise ValueError("ChunkedAdamW requires contiguous parameters")
                self.original_parameters.append(parameter)
                real = torch.view_as_real(parameter) if parameter.is_complex() else parameter
                flat = real.detach().view(-1)
                for start in range(0, flat.numel(), chunk_elements):
                    stop = min(start + chunk_elements, flat.numel())
                    chunk = nn.Parameter(flat[start:stop], requires_grad=True)
                    chunks.append(chunk)
                    self.chunk_views.append((parameter, chunk, start, stop))
            group["params"] = chunks
            groups.append(group)
        kwargs["foreach"] = False
        super().__init__(groups, **kwargs)

    def zero_grad(self, set_to_none=True):
        for parameter in self.original_parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()
        super().zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for parameter, chunk, start, stop in self.chunk_views:
            if parameter.grad is None:
                chunk.grad = None
                continue
            grad = parameter.grad
            if grad.is_sparse or not grad.is_contiguous():
                raise ValueError("ChunkedAdamW requires dense contiguous gradients")
            real_grad = torch.view_as_real(grad) if grad.is_complex() else grad
            chunk.grad = real_grad.view(-1)[start:stop]
        super().step()
        return loss
