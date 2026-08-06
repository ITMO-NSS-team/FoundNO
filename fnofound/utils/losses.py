"""
losses.py - training and reporting losses for the AirfRANS experiments.

Training loss (used for all models):
    FieldLpLoss - sum of per-field relative L2 over vel_x, vel_y, pressure
                  (nu_t is excluded: it is noisy and would dominate the
                  joint norm). Each field is normalized by its own norm, so
                  all fields contribute equally regardless of scale.
                  Optional mask: when given, x and y are masked before the
                  computation (holes are ignored). Some cases have a mask
                  (FNO/RNO/DNO), Geo-FNO does not.

Reporting metric:
    PerFieldLoss - per-field relative L2 as a dict (all 4 fields including
                   nu_t) - written into summary.json / leaderboard.
"""

import torch
import torch.nn as nn


class FieldLpLoss(nn.Module):
    """
    Sum of relative L2 over the selected fields (no joint norm).

    loss = mean_batch( sum_f ||df||_2 / (||f||_2 + eps) )

    Each field is normalized by its own norm, so field contributions to the
    gradient do not depend on their scale (unlike a joint LpLoss, where small
    fields - e.g. nu_t - barely influence the total loss).

    Parameters
    ----------
    field_indices : tuple[int]
        Channels that participate (default (0, 1, 2): vel_x, vel_y, pressure;
        nu_t is excluded).
    weights : tuple[float] or None
        Optional per-field weights (default all 1.0).
    """

    def __init__(self, field_indices=(0, 1, 2), weights=None):
        super().__init__()
        self.field_indices = list(field_indices)
        self.weights = weights

    def forward(self, x, y, mask=None):
        """
        x, y : [B, H, W, C] or [B, N, C]
        mask : [B, H, W, 1] or None - multiply x and y by the mask first
               (masked-out points are ignored); None = no masking.
        returns: scalar
        """
        if mask is not None:
            x = x * mask
            y = y * mask

        batch_size = x.shape[0]
        total = None
        for f in self.field_indices:
            xf = x[..., f].reshape(batch_size, -1)
            yf = y[..., f].reshape(batch_size, -1)
            diff = torch.norm(xf - yf, p=2, dim=1)
            norm = torch.norm(yf, p=2, dim=1)
            w = 1.0 if self.weights is None else self.weights[f]
            term = w * diff / (norm + 1e-8)
            total = term if total is None else total + term

        return total.mean()


class PerFieldLoss(nn.Module):
    """
    Relative L2 loss for each field separately - reporting metric.

    Returns a dict field_name -> rel-L2 (all fields, including nu_t),
    used for logs / summary.json / leaderboard.
    """

    def __init__(self, field_names=('vel_x', 'vel_y', 'pressure', 'nu_t')):
        super().__init__()
        self.field_names = field_names

    def forward(self, pred, target, mask=None):
        """
        pred, target : [B, H, W, C] or [B, N, C]
        mask : [B, H, W, 1] or None
        returns: dict {field_name: float}
        """
        losses = {}
        for i, name in enumerate(self.field_names):
            p = pred[..., i:i + 1]
            t = target[..., i:i + 1]
            if mask is not None:
                p = p * mask
                t = t * mask

            batch_size = p.shape[0]
            p_flat = p.reshape(batch_size, -1)
            t_flat = t.reshape(batch_size, -1)

            diff = torch.norm(p_flat - t_flat, p=2, dim=1)
            norm = torch.norm(t_flat, p=2, dim=1)
            losses[name] = (diff / (norm + 1e-8)).mean().item()

        return losses
