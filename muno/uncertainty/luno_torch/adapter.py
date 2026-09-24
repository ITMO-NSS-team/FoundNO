from __future__ import annotations

import re

import torch


def discover_last_block_keys(
    model: torch.nn.Module,
) -> dict[str, str | None]:
    """Find state_dict keys of the last Fourier block tunable parameters.

    Mirrors `FNOWrapper.params` of luno_experiments: the complex spectral-convolution
    weight R, the linear skip kernel W, and the bias b of the last FNO block are the
    linearized parameters. Everything else (lifting / padding / previous blocks /
    unpadding / projection) is treated as the fixed head+projection.

    Supports both neuralop <=1.x (``fno_blocks.<i>.conv_weights``) and neuralop 2.x
    (``fno_blocks.convs.<i>.weight.tensor``, ``fno_blocks.fno_skips.<i>.conv.weight``)
    key layouts.
    """
    sd = {
        k: v
        for k, v in model.state_dict().items()
        if isinstance(v, torch.Tensor)
    }

    conv_weights: dict[int, str] = {}
    for k in sd:
        m = re.search(r"fno_blocks\.convs\.(\d+)\.weight(?:\.tensor)?$", k)
        if m is not None:
            conv_weights[int(m.group(1))] = k
            continue
        m = re.search(r"fno_blocks\.(\d+)\.conv_weights$", k)
        if m is not None:
            conv_weights[int(m.group(1))] = k

    if not conv_weights:
        raise ValueError(
            "Could not find last-FNO-block conv weights in state_dict. Keys: "
            + ", ".join(sorted(sd)[:12])
        )

    nl = max(conv_weights)
    R_key = conv_weights[nl]

    W_key = None
    for k in sd:
        if re.search(rf"fno_blocks\.fno_skips\.{nl}\.conv\.weight$", k):
            W_key = k
            break
        if re.search(rf"fno_blocks\.{nl}\.fc\.weight$", k):
            W_key = k
            break

    b_key = None
    for k in sd:
        if re.search(rf"fno_blocks\.convs\.{nl}\.bias$", k):
            b_key = k
            break
        if re.search(rf"fno_blocks\.{nl}\.conv_bias$", k):
            b_key = k
            break
        if re.search(rf"fno_blocks\.fno_skips\.{nl}\.conv\.bias$", k):
            b_key = k
            break

    if W_key is None:
        raise ValueError(
            "Could not find last-FNO-block skip weight in state_dict. Keys: "
            + ", ".join(sorted(sd)[:12])
        )

    return {"R": R_key, "W": W_key, "b": b_key, "nl": nl}


def ref_state_dict(
    model: torch.nn.Module, keys: dict[str, str | None]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    sd = {
        k: v
        for k, v in model.state_dict().items()
        if isinstance(v, torch.Tensor)
    }
    R = sd[keys["R"]]
    W = sd[keys["W"]]
    b = sd[keys["b"]] if keys["b"] is not None else None
    return R, W, b


class TorchFNOWrapper:
    """Split a neuralop FNO into (fixed head) / (last Fourier block weights) / (projection).

    The last block weights (R, W, b) are the linearized GP parameters, exactly like
    :class:`luno_experiments.nn.wrapper.FNOWrapper`.

    Parameters
    ----------
    model:
        A neuralop FNO (any subclass whose ``forward`` calls ``functional_call``-able
        state dict).
    dtype:
        dtype used for the GP computations (defaults to float64, like luno).
    keys:
        optional explicit state_dict keys; auto-discovered otherwise.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype = torch.float64,
        keys: dict[str, str | None] | None = None,
    ):
        self.model = model
        self.dtype = dtype
        self.keys = keys if keys is not None else discover_last_block_keys(model)
        self._base = None
        self.R, self.W, self.b = ref_state_dict(model, self.keys)
        self._probe_out_channels()
        self.device = next(model.parameters()).device

    def _probe_out_channels(self) -> None:
        out_channels = getattr(self.model, "out_channels", None)
        if out_channels is None:
            raise ValueError(
                "model has no .out_channels; pass num_output_channels explicitly"
            )
        self.num_output_channels = out_channels

    @property
    def params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self.R, self.W, self.b

    @property
    def w0(self) -> torch.Tensor:
        """Flat [R.real, R.imag, W, b] parameter vector of the last block."""
        cdtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
        R = self.R.to(dtype=cdtype, device=self.device)
        W = self.W.to(dtype=self.dtype, device=self.device)
        b = self.b.to(dtype=self.dtype, device=self.device) if self.b is not None else None
        parts = [R.real.reshape(-1), R.imag.reshape(-1), W.reshape(-1)]
        if b is not None:
            parts.append(b.reshape(-1))
        return torch.cat(parts)

    def weight_splits(self) -> tuple[int, int, int, int | None]:
        R = self.R
        W = self.W
        b = self.b
        n_r = R.numel()
        n_w = W.numel()
        n_b = b.numel() if b is not None else None
        return n_r, n_w, n_b

    def _base_state_dict(self) -> dict[str, torch.Tensor]:
        if self._base is None:
            cdtype = (
                torch.complex128
                if self.dtype == torch.float64
                else torch.complex64
            )
            self._base = {}
            for k, v in self.model.state_dict().items():
                if not isinstance(v, torch.Tensor):
                    continue
                v = v.detach().to(device=self.device)
                if torch.is_complex(v):
                    v = v.to(dtype=cdtype)
                else:
                    v = v.to(dtype=self.dtype)
                self._base[k] = v
        return dict(self._base)

    def reconstruct(
        self, w: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Build a full state_dict from a flat last-block weight vector."""
        w = w.to(dtype=self.dtype, device=self.device)
        sd = self._base_state_dict()
        n_r, n_w, _ = self.weight_splits()
        R = torch.complex(
            w[:n_r].reshape(self.R.shape),
            w[n_r : 2 * n_r].reshape(self.R.shape),
        )
        sd[self.keys["R"]] = R
        sd[self.keys["W"]] = w[2 * n_r : 2 * n_r + n_w].reshape(self.W.shape)
        if self.b is not None:
            sd[self.keys["b"]] = w[2 * n_r + n_w :].reshape(self.b.shape)
        return sd

    def model_fn(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Forward of the full model with last-block params replaced by `params`."""
        sd = self.reconstruct(params)
        x = x.to(dtype=self.dtype, device=self.device)
        return torch.func.functional_call(self.model, sd, (x,))


def split_wrapper(
    wrapper: TorchFNOWrapper,
) -> tuple[callable, torch.Tensor]:
    """Mirror of luno_experiments ``split_wrapper``.

    Returns ``(model_fn, relevant_params)`` where ``relevant_params`` is the flat
    (R, W, b) last-block weight vector used as the GP parameter.
    """
    return wrapper.model_fn, wrapper.w0