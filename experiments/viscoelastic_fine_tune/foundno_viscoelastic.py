"""Full fine-tuning of the original pretrained CFNO on viscoelastic fields."""
from pathlib import Path
import hashlib

import dill
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.embeddings import GridEmbeddingND
from mamba_ssm import Mamba

from foundno_cfno import CFNO, CheckpointedCFNO
from viscoelastic_data import FIELD_NAMES


class MambaLifting(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.positional_embedding = GridEmbeddingND(in_channels=9, dim=3,
                                                     grid_boundaries=[[0., 1.]] * 3)
        self.lifting = ChannelMLP(in_channels=12, out_channels=width, hidden_channels=width,
                                   n_layers=2, n_dim=3, non_linearity=torch.nn.functional.gelu)
        # Matches original cfno_9_11/lift_0.pt, rather than the Euler SimpleSSM.
        self.post_lift_ssm = nn.Module()
        self.post_lift_ssm.ssm = Mamba(d_model=width, d_state=width, d_conv=2, expand=2)
        self.post_lift_ssm.norm = nn.LayerNorm(width)

    def forward(self, x):
        x = self.lifting(self.positional_embedding(x))
        b, c, t, h, w = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).contiguous().view(b, t * h * w, c)
        y = self.post_lift_ssm.norm(self.post_lift_ssm.ssm(tokens))
        return y.view(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()


class ViscoelasticFoundNO(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.lifting = MambaLifting(core.in_channels)
        self.projection = ChannelMLP(in_channels=core.out_channels, out_channels=8,
                                     hidden_channels=core.in_channels, n_layers=2, n_dim=3,
                                     non_linearity=torch.nn.functional.gelu)

    def forward(self, x):
        if torch.is_grad_enabled():
            z = checkpoint(self.lifting, x, use_reentrant=False)
            return checkpoint(self.projection, self.core(z), use_reentrant=False)
        return self.projection(self.core(self.lifting(x)))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def build_model(config):
    source = config["model"]["pretrained_core_path"]
    core = torch.load(source, map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)
    assert isinstance(core, CFNO)
    assert core.spatial.kind == core.temporal.kind == "mamba"
    core.__class__ = CheckpointedCFNO
    core.gradient_checkpointing = True
    core.frozen_core = False
    model = ViscoelasticFoundNO(core)
    for p in model.parameters():
        p.requires_grad_(True)
    metadata = {"model_kind": "cfno_9_11", "source_core": source,
                "source_core_sha256": file_sha256(source), "source_euler_checkpoint": None,
                "initialization": "original pretrained core; task-specific input/output dimensions",
                "lifting_backend": "Mamba", "lifting_d_state": core.in_channels,
                "lifting_d_conv": 2, "field_order": list(FIELD_NAMES),
                "time_conditioning": "additional channel: actual elapsed time / train maximum horizon",
                "input_channels": 9, "output_channels": 8, "frozen_core": False,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "spectral_modes": [list(c.n_modes) for c in core.fno_blocks.convs]}
    # Optional tensor-only extraction from ORIGINAL lifting; first input layer
    # and final output layer remain new because the physical fields differ.
    tensors_path = config["model"].get("original_lifting_tensors")
    if tensors_path:
        saved = torch.load(tensors_path, map_location="cpu", weights_only=True)
        target = model.lifting.state_dict()
        transfer = {k: v for k, v in saved.items()
                    if k in target and target[k].shape == v.shape and "grid" not in k}
        assert {k for k in target if k.startswith("post_lift_ssm.")} <= set(transfer)
        model.lifting.load_state_dict(transfer, strict=False)
        with torch.no_grad():
            # Only the appended grid coordinates have known shared semantics.
            model.lifting.lifting.fcs[0].weight[:, -3:].copy_(saved["lifting.fcs.0.weight"][:, -3:])
        metadata["pretrained_lifting_tensor_keys"] = sorted(transfer)
        metadata["pretrained_lifting_tensors_source"] = str(tensors_path)
        metadata["pretrained_lifting_grid_columns_reused"] = True
    projection_path = config["model"].get("pretrained_projection_path")
    if projection_path:
        original = torch.load(projection_path, map_location="cpu", pickle_module=dill,
                              weights_only=False, mmap=True)
        target = model.projection.state_dict()
        transfer = {k: v for k, v in original.state_dict().items()
                    if torch.is_tensor(v) and k in target and target[k].shape == v.shape}
        assert {"fcs.0.weight", "fcs.0.bias"} <= set(transfer)
        model.projection.load_state_dict(transfer, strict=False)
        metadata["pretrained_projection_tensor_keys"] = sorted(transfer)
        metadata["pretrained_projection_source"] = str(projection_path)
    assert metadata["trainable_parameters"] == metadata["parameters"]
    return model, metadata
