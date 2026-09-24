from __future__ import annotations

import torch

from .adapter import TorchFNOWrapper
from .ggn import LowRankTerms
from .gp import FNOGPLastLayer
from .lino_ops import (
    CircularlySymmetricDiagonal,
    IsotropicScalingPlusSymmetricLowRank,
    PositiveDiagonalPlusSymmetricLowRank,
    SymmetricLowRank,
    linverse,
)


def create_luno_cov(curv_est: LowRankTerms, prior_args: dict, params=None):
    """Covariance of the last-block weight posterior.

    Mirrors ``luno_experiments.uncertainty.methods._luno.create_luno_cov``: scalar
    prior gives ``(prec I + U S U^T)^{-1}`` via the closed-form linverse; a dict prior
    gives the block-diagonal-plus-low-rank case.
    """
    U = curv_est.U
    S = curv_est.S
    prec = prior_args.get("prior_prec", 1.0)

    if not isinstance(prec, dict):
        if not isinstance(prec, torch.Tensor):
            prec = torch.tensor(prec, dtype=U.dtype, device=U.device)
        return linverse(
            IsotropicScalingPlusSymmetricLowRank(prec.to(U), U, S.to(U.dtype))
        )

    if params is None:
        raise ValueError("Params required for block-diagonal prior")
    R, W, b = params
    diag = CircularlySymmetricDiagonal(
        R_real=_to_device(prior_args["R"], U),
        W=_to_device(prior_args["W"], U),
        b=None if b is None else _to_device(prior_args["b"], U),
    )
    low_rank = SymmetricLowRank(U=U, S=S.to(U.dtype))
    return linverse(PositiveDiagonalPlusSymmetricLowRank(diagonal=diag, low_rank=low_rank))


def _to_device(x, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=ref.dtype, device=ref.device)
    return torch.tensor(x, dtype=ref.dtype, device=ref.device)


def create_luno_posterior(curv_est: LowRankTerms, wrapper: TorchFNOWrapper):
    R, W, b = wrapper.params

    def posterior(prior_args: dict):
        cov = create_luno_cov(curv_est, prior_args, params=(R, W, b))
        return FNOGPLastLayer(
            model_fn=wrapper.model_fn,
            w0=wrapper.w0,
            R=R,
            W=W,
            b=b,
            weight_cov=cov,
            num_output_channels=wrapper.num_output_channels,
        )

    return posterior


def _unpad(x, ref, pad=2):
    return x.reshape(ref.shape)


def create_grid(inp_shape, padding=2, dtype=None):
    return None


def luno_mean_std(results, aux, **kwargs):
    gp, grid = aux["gp"], aux["grid"]
    pred = results["map"]
    m, s = gp.mean_and_std()
    results.update(
        {
            "pred": pred,
            "pred_mean": _unpad(m.reshape(pred.shape), pred),
            "pred_std": _unpad(s.reshape(pred.shape), pred),
        }
    )
    return results, aux


def luno_samples(results, aux, **kwargs):
    gp, grid, key = aux["gp"], aux["grid"], aux["key"]
    pred = results["map"]
    num_samples = kwargs.get("num_samples", 10)
    samples = gp.sample(key=key, grid=grid, size=(num_samples,))
    results.update(
        {
            "pred": pred,
            "samples": samples.reshape(num_samples, *pred.shape),
        }
    )
    return results, aux


def set_luno_predictive(
    model_fn,
    mean_params,
    prior_arguments,
    posterior_fn,
    pushforward_fns,
    key,
    **kwargs,
):
    gp_op = posterior_fn(prior_arguments)

    def prob_predictive(input):
        pred = model_fn(input, params=mean_params)
        gp = gp_op(input)
        grid = create_grid(input.shape)
        aux = {"gp": gp, "grid": grid, "key": key}
        res = {"map": pred}
        for fn in pushforward_fns:
            res, aux = fn(res, aux, **kwargs)
        return res

    return prob_predictive