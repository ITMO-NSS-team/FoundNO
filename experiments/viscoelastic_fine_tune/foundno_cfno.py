"""Euler adaptation of the intact cfno_9_11 pretrained CFNO core."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).parent / "cfno_upstream"))
from muno.models.cno import CFNO
from muno.layers.cno_block import ConvBlock

_conv_forward = ConvBlock.forward


def _checkpoint_conv(self, x):
    if torch.is_grad_enabled():
        return checkpoint(lambda z: _conv_forward(self, z), x, use_reentrant=False)
    return _conv_forward(self, x)


ConvBlock.forward = _checkpoint_conv


class CheckpointedCFNO(CFNO):
    """Same operations and parameters as upstream; recompute stage activations."""
    def train(self, mode=True):
        # Frozen weights still transmit gradients to lifting. Keep dropout off,
        # including when the outer model switches into training mode.
        return super().train(False if getattr(self, "frozen_core", False) else mode)

    def forward(self, x, skips=None, **kwargs):
        assert self._disable_lifting_and_projection and not self.residual_u0
        assert self.gt_pre is None and self.gt_post is None and self.aux_head is None

        def call(fn, value):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                return checkpoint(fn, value, use_reentrant=False)
            return fn(value)

        h = x
        if self.spatial is not None:
            h = call(self.spatial, h)
        if self.temporal is not None:
            h = call(self.temporal, h)
        h_local = self.cno(h) if self.cno is not None else None
        if self.local_branch == "only":
            return h_local
        gw = 1.0 / max(h.shape[-2:])
        for i in range(self.n_layers):
            def stage(value, index=i):
                result, _ = self.fno_blocks(value, index)
                if self.diff is not None:
                    result = result + self.diff_alpha[index] * self.diff[index](value, gw)
                return result
            h = call(stage, h)
        if h_local is not None:
            h = h + self.cno_alpha * h_local
        self._last_hidden = None
        return h


def build_cfno(config, metadata=None):
    import dill
    from neuralop.layers.channel_mlp import ChannelMLP
    from foundno_lifting import PostLiftMambaLifting
    from foundno_euler_common import FoundNOEulerModel, FIELD_NAMES

    source = config["model"]["pretrained_core_path"]
    core = torch.load(source, map_location="cpu", pickle_module=dill,
                      weights_only=False, mmap=True)
    assert isinstance(core, CFNO) and core.spatial.kind == core.temporal.kind == "mamba"
    core.__class__ = CheckpointedCFNO
    core.gradient_checkpointing = config["model"].get("gradient_checkpointing", True)
    mode = config.get("training", {}).get("finetune_mode", "full")
    if mode not in ("full", "adapter_only"):
        raise ValueError(f"Unsupported CFNO finetune_mode: {mode}")
    frozen = mode == "adapter_only"
    core.frozen_core = frozen
    for parameter in core.parameters():
        parameter.requires_grad_(not frozen)
    if frozen:
        core.eval()
    width = core.in_channels
    n_in = len(FIELD_NAMES) + int(config["conditioning"]["gamma"])
    lifting = PostLiftMambaLifting(
        in_channels=n_in, out_channels=width, width=width, n_dim=3,
        padding=0, positional_embedding="grid",
        non_linearity=torch.nn.functional.gelu,
    )
    projection = ChannelMLP(in_channels=core.out_channels, out_channels=4,
                            hidden_channels=width, n_layers=2, n_dim=3,
                            non_linearity=torch.nn.functional.gelu)
    model = FoundNOEulerModel(lifting, core, projection)
    model.checkpoint_adapters = core.gradient_checkpointing
    return model, {
        "model_kind": "cfno_9_11", "architecture": "Euler lifting -> CFNO (spatial/temporal Mamba + parallel CNO + FNO/differential) -> Euler projection",
        "initialization": "intact pretrained core; new Euler adapters, as in U-NO comparison",
        "source_checkpoint_files": {"core": source}, "field_order": list(FIELD_NAMES),
        "input_channels": n_in, "output_channels": 4, "hidden_channels": width,
        "core_layers": core.n_layers, "frozen_core": frozen,
        "gradient_checkpointing": core.gradient_checkpointing,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "spectral_modes": [list(c.n_modes) for c in core.fno_blocks.convs],
    }
