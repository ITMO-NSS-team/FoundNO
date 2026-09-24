from __future__ import annotations

import torch

from .jacobian import LastFNOBlockWeightJacobian, var_of_congruence
from .lino_ops import (
    LinearOperator,
    ProductLinearOperator,
    congruence_transform,
    lsqrt,
)


class ParametricGaussianProcess:
    """Weight-space Gaussian process with linearized forward features.

    Assumes a zero-mean parameter prior with covariance ``weight_cov`` over the last
    FNO block weights; the function value at ``x`` is
    ``mean(x) + features(x) * w,  w ~ N(0, weight_cov)``.
    """

    def __init__(self, weight_cov: LinearOperator, mean_and_features=None):
        self._weight_cov = weight_cov
        self._mean_and_features = mean_and_features

    @property
    def weight_cov(self) -> LinearOperator:
        return self._weight_cov

    def mean_and_features(self, x: torch.Tensor):
        if self._mean_and_features is None:
            raise NotImplementedError(
                "mean_and_features must be provided at construction"
            )
        return self._mean_and_features(x)

    def sample(self, key: int = 0, x: torch.Tensor | None = None, size=()):
        weight_cov_lsqrt = lsqrt(self._weight_cov)
        dim = weight_cov_lsqrt.shape()[0]
        gen = torch.Generator(device=weight_cov_lsqrt.device).manual_seed(key)
        z = torch.randn(
            dim,
            *size,
            dtype=weight_cov_lsqrt.dtype,
            device=weight_cov_lsqrt.device,
            generator=gen,
        )
        weight_sample = weight_cov_lsqrt._matmul(z)

        def sample_fn(xx):
            mean_x, features_x = self.mean_and_features(xx)
            shifted = features_x._matmul(weight_sample)
            return (mean_x.unsqueeze(-1) + shifted).T

        if x is None:
            return sample_fn
        return sample_fn(x)

    def mean_and_cov(self, x: torch.Tensor):
        mean_x, features_x = self.mean_and_features(x)
        return mean_x, congruence_transform(features_x, self._weight_cov)

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_and_features(x)[0]

    def cov(self, x0: torch.Tensor, x1: torch.Tensor | None = None) -> LinearOperator:
        if x1 is None:
            return self.mean_and_cov(x0)[1]
        _, f0 = self.mean_and_features(x0)
        _, f1 = self.mean_and_features(x1)
        return ProductLinearOperator([f0, self._weight_cov, f1.T])

    def mean_and_var(self, x: torch.Tensor):
        mean_x, features_x = self.mean_and_features(x)
        var = var_of_congruence(features_x, self._weight_cov)
        return mean_x, var

    def var(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_and_var(x)[1]

    def mean_and_std(self, x: torch.Tensor):
        mean_x, var_x = self.mean_and_var(x)
        return mean_x, torch.sqrt(var_x)

    def std(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_and_std(x)[1]


class FNOGPLastLayer:
    """Torch port of ``luno._fno_last_layer.FNOGPLastLayer``.

    The model is split at the last Fourier block: ``model_fn`` is the wrapped full
    forward whose tunable params are the flat last-block vector, and ``w0`` is the MAP
    weight vector. Calling the layer with an input fixes the input and returns a GP
    over the last-block weights.
    """

    def __init__(
        self,
        model_fn: callable,
        w0: torch.Tensor,
        R: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor | None,
        weight_cov: LinearOperator,
        num_output_channels: int,
        vjp_chunk: int = 64,
    ):
        self._model_fn = model_fn
        self._w0 = w0
        self._R = R
        self._W = W
        self._b = b
        self._weight_cov = weight_cov
        self._num_output_channels = num_output_channels
        self._vjp_chunk = vjp_chunk

    def __call__(self, a: torch.Tensor) -> "FNOGPLastLayer.FixedInputGaussianProcess":
        a = a.detach().to(dtype=self._w0.dtype, device=self._w0.device)
        return FNOGPLastLayer.FixedInputGaussianProcess(
            model_fn=self._model_fn,
            x=a,
            w0=self._w0,
            weight_cov=self._weight_cov,
            num_output_channels=self._num_output_channels,
            vjp_chunk=self._vjp_chunk,
        )

    class FixedInputGaussianProcess(ParametricGaussianProcess):
        def __init__(
            self,
            model_fn: callable,
            x: torch.Tensor,
            w0: torch.Tensor,
            weight_cov: LinearOperator,
            num_output_channels: int,
            vjp_chunk: int = 64,
        ):
            self._model_fn = model_fn
            self._x = x
            self._w0 = w0
            self._num_output_channels = num_output_channels
            self._vjp_chunk = vjp_chunk
            self._jacobian = LastFNOBlockWeightJacobian(
                model_fn=model_fn,
                x=x,
                w0=w0,
                num_output_channels=num_output_channels,
                vjp_chunk=vjp_chunk,
            )
            super().__init__(weight_cov=weight_cov)

        def mean_and_features(self, x=None):
            """Return (u, J) with u = MAP prediction (flat) and J the Jacobian."""
            if x is not None:
                return self._model_fn(x, self._w0).reshape(-1), self._jacobian
            return self._jacobian.fx, self._jacobian

        def sample(self, key: int = 0, grid=None, size=(1,)) -> torch.Tensor:
            """Draw sample(s) from the fixed-input GP."""
            mean_x, features_x = self.mean_and_features()
            weight_cov_lsqrt = lsqrt(self._weight_cov)
            dim = weight_cov_lsqrt.shape()[0]
            gen = torch.Generator(device=weight_cov_lsqrt.device).manual_seed(key)
            z = torch.randn(
                dim,
                *size,
                dtype=weight_cov_lsqrt.dtype,
                device=weight_cov_lsqrt.device,
                generator=gen,
            )
            weight_sample = weight_cov_lsqrt._matmul(z)
            shifted = features_x._matmul(weight_sample)
            return (mean_x.unsqueeze(-1) + shifted).T

        def mean(self, x=None) -> torch.Tensor:
            return self.mean_and_features()[0]

        def mean_and_cov(self, x=None):
            mean_x, features_x = self.mean_and_features()
            return mean_x, congruence_transform(features_x, self._weight_cov)

        def mean_and_var(self, x=None):
            mean_x, features_x = self.mean_and_features()
            return mean_x, var_of_congruence(features_x, self._weight_cov)

        def mean_and_std(self, x=None):
            mean_x, var_x = self.mean_and_var()
            return mean_x, torch.sqrt(var_x)

        def var(self, x=None) -> torch.Tensor:
            return self.mean_and_var()[1]

        def std(self, x=None) -> torch.Tensor:
            return self.mean_and_std()[1]