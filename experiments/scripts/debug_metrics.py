import torch

from muno.utils.metrics import (
    # check_same_shape,
    mse,
    mae,
    rmse,
    nrmse,
    vrmse,
    flatten_per_sample,
    relative_l2,
    balanced_relative_l2,
    nmae_range,
    nmae_mean,
    nmae_std,
    compute_metrics,
    r2_score,
    mean_linf_error,
    max_linf_error
)

from muno.utils.metrics_physical import (
    kinetic_energy_2d,
    kinetic_energy_error,
    relative_kinetic_energy_error,
    spectral_mse,
    band_spectral_mse_2d,
    spectral_nrmse,
    band_spectral_nrmse_2d,
    compute_physical_metrics
)

# # check_same_shape
# pred = torch.zeros(2, 3, 4)
# target_check_same_shape = torch.ones(2, 3, 4)
#
# check_same_shape(pred, target_check_same_shape)
# print("check_same_shape: same shape ok")
#
# target_check_same_shape_bad = torch.ones(2, 3, 5)
# check_same_shape(pred, target_check_same_shape_bad)

# mse, mae, rmse, rel_l2, balanced_rel_l2
target = torch.ones(2, 3, 4)
pred = target + 0.5
print("mse:", mse(pred, target).item())
print("mae:", mae(pred, target).item())
print("rmse:", rmse(pred, target).item())

x = torch.zeros(2, 3, 4)
flat = flatten_per_sample(x)
print("original:", x.shape)
print("flat:", flat.shape)
print("relative_l2:", relative_l2(pred, target).item())
print("balanced_relative_l2:", balanced_relative_l2(pred, target).item())

target = torch.zeros(1, 2, 4)
target[:, 0] = 100.0
target[:, 1] = 1.0

pred = target.clone()
pred[:, 0] = 110.0
pred[:, 1] = 2.0

print("relative_l2 different scales:", relative_l2(pred, target).item())
print("balanced_relative_l2 different scales:", balanced_relative_l2(pred, target).item())

# nmae_range
target = torch.tensor([
    [[0., 1., 2., 3.]],
    [[0., 2., 4., 6.]],
])
pred = target + 1.0
print("nmae_range:", nmae_range(pred, target).item())

# nmae_mean
target = torch.tensor([
    [[1., 1., 1., 1.]],
    [[2., 2., 2., 2.]],
])
pred = target + 1.0
print("nmae_mean:", nmae_mean(pred, target).item())

# nmae_std
target = torch.tensor([
    [[0., 0., 2., 2.]],
    [[0., 0., 4., 4.]],
])
pred = target + 1.0
print("nmae_std:", nmae_std(pred, target).item())

# rmse
target = torch.zeros(2, 2, 4)
pred = torch.zeros(2, 2, 4)
pred[:, 0] = 1.0
pred[:, 1] = 3.0
print("vrmse:", vrmse(pred, target).item())
target = torch.ones(2, 3, 4)
pred = target + 0.5
print("nrmse:", nrmse(pred, target).item())

# r2_score
target = torch.tensor([
    [[0., 1., 2., 3.]],
    [[0., 2., 4., 6.]],
])
pred = target.clone()
print("r2_score perfect:", r2_score(pred, target).item())
pred = torch.mean(target, dim=-1, keepdim=True).expand_as(target)
print("r2_score mean:", r2_score(pred, target).item())

# linf_error
target = torch.zeros(2, 1, 4)
pred = torch.tensor([
    [[0., 1., 2., 3.]],
    [[0., 2., 4., 6.]],
])
print("mean_linf_error:", mean_linf_error(pred, target).item())
print("max_linf_error:", max_linf_error(pred, target).item())

# KE_error
target = torch.zeros(2, 2, 4)
target[:, 0] = 2.0  # vx
target[:, 1] = 0.0  # vy
pred = torch.zeros(2, 2, 4)
pred[:, 0] = 4.0
pred[:, 1] = 0.0
print("target_ke:", kinetic_energy_2d(target).tolist())
print("pred_ke:", kinetic_energy_2d(pred).tolist())
print("ke_error:", kinetic_energy_error(pred, target).item())
print("relative_ke_error:", relative_kinetic_energy_error(pred, target).item())

target = torch.zeros(2, 3, 4)
target[:, 0] = 2.0  # density
target[:, 1] = 2.0  # vx
target[:, 2] = 0.0  # vy

pred = torch.zeros(2, 3, 4)
pred[:, 0] = 2.0
pred[:, 1] = 4.0
pred[:, 2] = 0.0

print(
    "target_ke_with_density:",
    kinetic_energy_2d(
        target,
        velocity_channels=(1, 2),
        density_channel=0,
    ).tolist(),
)

print(
    "pred_ke_with_density:",
    kinetic_energy_2d(
        pred,
        velocity_channels=(1, 2),
        density_channel=0,
    ).tolist(),
)

target = torch.zeros(1, 2, 4)
target[:, 0] = torch.tensor([1., 2., 3., 4.])  # vx
target[:, 1] = 0.0  # vy

pred = target.clone()

print(
    "target_ke_full:",
    kinetic_energy_2d(
        target,
        velocity_channels=(0, 1),
        subtract_mean=False,
    ).tolist(),
)

print(
    "target_ke_fluctuation:",
    kinetic_energy_2d(
        target,
        velocity_channels=(0, 1),
        subtract_mean=True,
    ).tolist(),
)

target = torch.zeros(1, 3, 4)
target[:, 0] = 2.0  # density
target[:, 1] = 2.0  # vx
target[:, 2] = 0.0  # vy

pred = torch.zeros(1, 3, 4)
pred[:, 0] = 3.0  # density
pred[:, 1] = 2.0  # vx
pred[:, 2] = 0.0  # vy

print(
    "target_ke_density_2:",
    kinetic_energy_2d(
        target,
        velocity_channels=(1, 2),
        density_channel=0,
    ).tolist(),
)

print(
    "pred_ke_density_3:",
    kinetic_energy_2d(
        pred,
        velocity_channels=(1, 2),
        density_channel=0,
    ).tolist(),
)

target = torch.zeros(1, 3, 4)
target[:, 0] = 2.0  # density
target[:, 1] = torch.tensor([1., 2., 3., 4.])  # vx
target[:, 2] = 0.0  # vy

print(
    "target_ke_density_fluctuation:",
    kinetic_energy_2d(
        target,
        velocity_channels=(1, 2),
        density_channel=0,
        subtract_mean=True,
    ).tolist(),
)

# spectral_mse
target = torch.ones(2, 1, 8, 8)
pred = target.clone()
print("spectral_mse perfect:", spectral_mse(pred, target).item())
pred = target * 2.0
print("spectral_mse scaled:", spectral_mse(pred, target).item())

# spectral_rmse
target = torch.ones(2, 1, 8, 8)
pred = target.clone()
print("spectral_nrmse perfect:", spectral_nrmse(pred, target).item())
pred = target * 2.0
print("spectral_nrmse scaled:", spectral_nrmse(pred, target).item())

# band_spectral_mse_2d
target = torch.ones(2, 1, 8, 8)
pred = target.clone()
print("band_spectral_mse_2d perfect:", band_spectral_mse_2d(pred, target))
pred = target * 2.0
print("band_spectral_mse_2d scaled:", band_spectral_mse_2d(pred, target))

# band_spectral_nrmse_2d
target = torch.ones(2, 1, 8, 8)
pred = target.clone()
print("band_spectral_nrmse_2d perfect:", band_spectral_nrmse_2d(pred, target))
pred = target * 2.0
print("band_spectral_nrmse_2d scaled:", band_spectral_nrmse_2d(pred, target))

# SINE
height = 32
width = 32
y = torch.arange(height).float()
x = torch.arange(width).float()
yy, xx = torch.meshgrid(y, x, indexing="ij")
target = torch.sin(2 * torch.pi * 2 * xx / width)
target = target.unsqueeze(0).unsqueeze(0)

# band_spectral_mse_2d SINE
pred = target.clone()
print("band_spectral_mse_2d sine perfect:", band_spectral_mse_2d(pred, target))
pred = torch.zeros_like(target)
print("band_spectral_mse_2d sine zero:", band_spectral_mse_2d(pred, target))

# band_spectral_nrmse_2d SINE
pred = target.clone()
print("band_spectral_nrmse_2d sine perfect:", band_spectral_nrmse_2d(pred, target))
pred = torch.zeros_like(target)
print("band_spectral_nrmse_2d sine zero:", band_spectral_nrmse_2d(pred, target))

# compute_metrics
target = torch.ones(2, 3, 4)
pred = target + 0.5

metrics = compute_metrics(
    pred,
    target,
    metric_names=["mse", "mae", "rmse", "relative_l2"],
)
print("compute_metrics:", metrics)

# compute_physical_metrics
target = torch.zeros(2, 2, 8, 8)
target[:, 0] = 2.0
target[:, 1] = 0.0

pred = torch.zeros(2, 2, 8, 8)
pred[:, 0] = 4.0
pred[:, 1] = 0.0

physical_metrics = compute_physical_metrics(
    pred,
    target,
    metric_configs=[
        {
            "name": "kinetic_energy_error",
            "velocity_channels": (0, 1),
        },
        {
            "name": "relative_kinetic_energy_error",
            "velocity_channels": (0, 1),
        },
        {
            "name": "spectral_nrmse",
        },
        {
            "name": "band_spectral_nrmse_2d",
        },
    ],
)

print("compute_physical_metrics:", physical_metrics)
