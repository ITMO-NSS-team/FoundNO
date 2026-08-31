"""
Вычисление метрик по всем датасетам (horizontal)

Адаптация вашего скрипта с добавлением:
- Загрузки трёх horizontal датасетов
- Фильтрации выбросов
- Сохранения результатов
"""

import torch
import torch.nn.functional as F

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
plt.rcParams['savefig.bbox'] = 'tight'


def get_well_locations(forcings, threshold=1e-8):
    T = forcings.shape[0]
    wells_with_timing = []
    forcing_sum = torch.abs(forcings).sum(dim=(0, 1))
    well_mask = forcing_sum > threshold
    well_indices = torch.nonzero(well_mask, as_tuple=False)

    for idx in well_indices:
        i, j = int(idx[0]), int(idx[1])
        forcing_at_well = torch.abs(forcings[:, :, i, j]).sum(dim=1)
        active_times = torch.where(forcing_at_well > threshold)[0]
        if len(active_times) > 0:
            t_start = int(active_times[0])
            wells_with_timing.append((i, j, t_start))

    return wells_with_timing


def compute_depression(pressure, i, j):
    H, W = pressure.shape[-2:]
    i_up = min(i + 1, H - 1)
    i_down = max(i - 1, 0)
    j_right = min(j + 1, W - 1)
    j_left = max(j - 1, 0)

    if pressure.dim() == 3:
        depression = 0.25 * (
            pressure[:, i, j_right] + pressure[:, i, j_left] +
            pressure[:, i_up, j] + pressure[:, i_down, j]
        )
    else:
        depression = 0.25 * (
            pressure[i, j_right] + pressure[i, j_left] +
            pressure[i_up, j] + pressure[i_down, j]
        )
    return depression


def compute_field_metrics(pred, target, mask, pressure_component=0, valid_timesteps=21):
    pred_p = pred[0, pressure_component, :valid_timesteps].cpu().detach()
    target_p = target[:valid_timesteps, pressure_component].cpu().detach()

    if isinstance(mask, np.ndarray):
        mask_tensor = torch.from_numpy(mask)
    else:
        mask_tensor = mask

    mask_3d = mask_tensor[None, :, :].expand_as(pred_p).permute(1, 2, 0)
    abs_error = torch.abs(pred_p - target_p)
    relative_error = abs_error.permute(1, 2, 0) / (torch.abs(target_p.mean(dim=(1, 2))) + 1e-8)
    relative_error_masked = relative_error[mask_3d].cpu().numpy()

    return {
        'mean': np.mean(relative_error_masked),
        'std': np.std(relative_error_masked)
    }


def compute_well_metrics(pred, target, forcings, mask, pressure_component=0, valid_timesteps=21):
    pred_p = pred[0, pressure_component, :valid_timesteps].cpu().detach()
    target_p = target[:valid_timesteps, pressure_component].cpu().detach()
    wells = get_well_locations(forcings)

    if len(wells) == 0:
        return {
            'mape': 0.0, 'mape_std': 0.0,
            'l1_rel_depression_norm': 0.0, 'l1_rel_depression_norm_std': 0.0,
            'mape_depression_norm': 0.0, 'mape_depression_norm_std': 0.0,
            'n_wells': 0, 'wells_info': []
        }

    mape_values = []
    l1_rel_depression_values = []
    mape_depression_values = []
    wells_info = []

    for (i, j, t_start) in wells:
        if t_start >= valid_timesteps:
            continue

        pred_well = pred_p[t_start:, i, j]
        target_well = target_p[t_start:, i, j]

        if len(pred_well) == 0:
            continue

        abs_error = torch.abs(pred_well - target_well)
        # mape = 100 * abs_error / (torch.abs(target_well) + 1e-4)
        mape = 200 * abs_error / (torch.abs(target_well) + torch.abs(pred_well) + 1e-4)
        mape_mean = mape.mean().item()
        mape_values.append(mape_mean)

        depression = compute_depression(target_p[t_start:], i, j)
        l1_rel_depression = abs_error / (depression + torch.abs(pred_well) + 1e-4)
        l1_rel_mean = l1_rel_depression.mean().item()
        l1_rel_depression_values.append(l1_rel_mean)

        avg_depression = depression.mean()
        mape_norm = 100 * abs_error / (avg_depression + 1e-4)
        mape_norm_mean = mape_norm.mean().item()
        mape_depression_values.append(mape_norm_mean)

        wells_info.append({
            'location': (i, j),
            't_start': t_start,
            'n_timesteps_evaluated': len(pred_well),
            'mape': mape_mean,
            'l1_rel_depression': l1_rel_mean,
            'mape_depression_norm': mape_norm_mean
        })

    if len(mape_values) == 0:
        return {
            'mape': 0.0, 'mape_std': 0.0,
            'l1_rel_depression_norm': 0.0, 'l1_rel_depression_norm_std': 0.0,
            'mape_depression_norm': 0.0, 'mape_depression_norm_std': 0.0,
            'n_wells': 0, 'wells_info': wells_info
        }

    return {
        'mape': np.mean(mape_values),
        'mape_std': np.std(mape_values),
        'l1_rel_depression_norm': np.mean(l1_rel_depression_values),
        'l1_rel_depression_norm_std': np.std(l1_rel_depression_values),
        'mape_depression_norm': np.mean(mape_depression_values),
        'mape_depression_norm_std': np.std(mape_depression_values),
        'n_wells': len(mape_values),
        'wells_info': wells_info
    }
def compute_neighborhood_error(pred_patch, target_patch, mask_patch,
                             weighting='uniform', normalize_by='target'):
    """
    Вычисляет относительную ошибку по окрестности скважины.

    Args:
        pred_patch, target_patch: [T, Kh, Kw] — патчи вокруг скважины
        mask_patch: [Kh, Kw] — маска активной области
        weighting: 'uniform' | 'gaussian' | 'distance'
        normalize_by: 'target' | 'depression' | 'mixed'

    Returns:
        dict с усреднёнными метриками по времени
    """
    T, Kh, Kw = pred_patch.shape
    abs_error = torch.abs(pred_patch - target_patch)  # [T, Kh, Kw]

    # === ВЕСА по пространству ===
    if weighting == 'uniform':
        weights = torch.ones((Kh, Kw), device=pred_patch.device)
    elif weighting == 'gaussian':
        # Гауссово ядро с σ ≈ 1 пиксель
        cy, cx = (Kh - 1) / 2, (Kw - 1) / 2
        y, x = torch.meshgrid(torch.arange(Kh), torch.arange(Kw), indexing='ij')
        dist2 = (x - cx)**2 + (y - cy)**2
        weights = torch.exp(-dist2 / (2 * 1.0**2))
    elif weighting == 'distance':
        # Обратное расстояние (центр = 1.0, края ~0.3-0.5)
        cy, cx = (Kh - 1) / 2, (Kw - 1) / 2
        y, x = torch.meshgrid(torch.arange(Kh), torch.arange(Kw), indexing='ij')
        dist = torch.sqrt((x - cx)**2 + (y - cy)**2) + 1e-6
        weights = 1.0 / dist
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    # Применяем маску и нормируем веса
    weights = weights * mask_patch.float()
    if weights.sum() < 1e-6:
        return {'mape': 0.0, 'l1_rel': 0.0, 'valid_pixels': 0}
    weights = weights / weights.sum()  # [Kh, Kw], сумма = 1

    # === МЕТРИКИ ПО ВРЕМЕНИ ===
    results = {'mape': [], 'l1_rel': [], 'valid_pixels': int((mask_patch > 0).sum().item())}

    for t in range(T):
        err_t = abs_error[t]  # [Kh, Kw]
        tgt_t = target_patch[t]

        # Знаменатель: адаптивный
        if normalize_by == 'target':
            denom = torch.abs(tgt_t) + 1e-4
        elif normalize_by == 'depression':
            # Депрессия как среднее по 4-соседям внутри патча
            dep = compute_depression(tgt_t, Kh//2, Kw//2)  # скаляр или [Kh, Kw]?
            # Упрощённо: используем среднее по патчу как референс
            denom = torch.abs(tgt_t).mean() + 1e-4
        else:  # mixed: max(|target|, mean_depression)
            dep_mean = torch.abs(tgt_t).mean()
            denom = torch.maximum(torch.abs(tgt_t), torch.tensor(dep_mean, device=tgt_t.device)) + 1e-4

        rel_err_t = err_t / denom  # [Kh, Kw]

        # Взвешенное усреднение
        mape_t = (100 * rel_err_t * weights).sum().item()
        l1_rel_t = (rel_err_t * weights).sum().item()

        results['mape'].append(mape_t)
        results['l1_rel'].append(l1_rel_t)

    # Усредняем по времени
    return {
        'mape': np.mean(results['mape']),
        'l1_rel': np.mean(results['l1_rel']),
        'valid_pixels': results['valid_pixels']
    }


def extract_patch(tensor_3d, i, j, patch_size, T):
    """
    Извлекает пространственно-временной патч [T, Kh, Kw] вокруг (i, j).
    tensor_3d: [T, H, W]
    """
    T, H, W = tensor_3d.shape
    half = patch_size // 2
    i0, i1 = max(i - half, 0), min(i + half + 1, H)
    j0, j1 = max(j - half, 0), min(j + half + 1, W)

    patch = tensor_3d[:, i0:i1, j0:j1]  # [T, kh, kw]

    # Если патч меньше desired из-за границы — дополняем нулями (или edge padding)
    if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
        pad_h = patch_size - patch.shape[1]
        pad_w = patch_size - patch.shape[2]
        patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='replicate')

    return patch


def compute_well_metrics_neighborhood(pred, target, forcings, mask,
                                     pressure_component=0,
                                     valid_timesteps=21,
                                     patch_size=5,
                                     weighting='gaussian',    # 'uniform', 'gaussian', 'distance'
                                     normalize_by='mixed',    # 'target', 'depression', 'mixed'
                                     mape_cap=500.0,
                                     dep_norm_cap=100.0,
                                     min_valid_ratio=0.5      # мин. доля валидных пикселей в патче
                                     ):
    """
    Окрестные метрики по скважинам: осреднение ошибки по локальному патчу.
    """
    pred_p = pred[0, pressure_component, :valid_timesteps].cpu().detach()  # [T, H, W]
    target_p = target[:valid_timesteps, pressure_component].cpu().detach()  # [T, H, W]
    wells = get_well_locations(forcings)

    if isinstance(mask, np.ndarray):
        mask_tensor = torch.from_numpy(mask)
    else:
        mask_tensor = mask

    if len(wells) == 0:
        return {
            'mape': 0.0, 'mape_std': 0.0,
            'l1_rel_depression_norm': 0.0, 'l1_rel_depression_norm_std': 0.0,
            'mape_depression_norm': 0.0, 'mape_depression_norm_std': 0.0,
            'n_wells': 0, 'wells_info': []
        }

    mape_values = []
    l1_rel_depression_values = []
    mape_depression_values = []
    wells_info = []

    for (i, j, t_start) in wells:
        if t_start >= valid_timesteps:
            continue

        # Извлекаем патчи
        pred_patch = extract_patch(pred_p[t_start:], i, j, patch_size, valid_timesteps - t_start)
        target_patch = extract_patch(target_p[t_start:], i, j, patch_size, valid_timesteps - t_start)
        mask_patch = extract_patch(mask_tensor[None].expand(valid_timesteps, -1, -1)[:, :, :],
                                  i, j, patch_size, valid_timesteps - t_start)[0]  # [Kh, Kw]

        # Проверяем, что в патче достаточно валидных пикселей
        valid_ratio = (mask_patch > 0).sum().item() / (patch_size * patch_size)
        if valid_ratio < min_valid_ratio:
            continue  # пропускаем скважины у границы с малым покрытием

        # === Основная метрика: окрестная относительная ошибка ===
        metrics = compute_neighborhood_error(
            pred_patch, target_patch, mask_patch,
            weighting=weighting, normalize_by=normalize_by
        )

        if metrics['valid_pixels'] == 0:
            continue

        mape_val = min(metrics['mape'], mape_cap)
        l1_rel_val = min(metrics['l1_rel'], dep_norm_cap)

        # === Depression-normalized: депрессия тоже считается по окрестности ===
        # Берём среднюю депрессию по патчу как референс
        target_mean = torch.abs(target_patch).mean(dim=(1,2))  # [T]
        dep_ref = target_mean.mean() + 1e-4  # скаляр

        abs_err_patch = torch.abs(pred_patch - target_patch)  # [T, Kh, Kw]
        weighted_err = (abs_err_patch * mask_patch.float()[None]).sum(dim=(1,2)) / (mask_patch.sum() + 1e-6)  # [T]
        mape_dep_norm = 100 * weighted_err / dep_ref
        mape_dep_val = min(mape_dep_norm.mean().item(), mape_cap)

        mape_values.append(mape_val)
        l1_rel_depression_values.append(l1_rel_val)
        mape_depression_values.append(mape_dep_val)

        wells_info.append({
            'location': (i, j),
            't_start': t_start,
            'patch_size': patch_size,
            'weighting': weighting,
            'valid_pixels': metrics['valid_pixels'],
            'mape': mape_val,
            'l1_rel_depression': l1_rel_val,
            'mape_depression_norm': mape_dep_val
        })

    if len(mape_values) == 0:
        return {
            'mape': 0.0, 'mape_std': 0.0,
            'l1_rel_depression_norm': 0.0, 'l1_rel_depression_norm_std': 0.0,
            'mape_depression_norm': 0.0, 'mape_depression_norm_std': 0.0,
            'n_wells': 0, 'wells_info': wells_info
        }

    return {
        'mape': np.mean(mape_values),
        'mape_std': np.std(mape_values),
        'l1_rel_depression_norm': np.mean(l1_rel_depression_values),
        'l1_rel_depression_norm_std': np.std(l1_rel_depression_values),
        'mape_depression_norm': np.mean(mape_depression_values),
        'mape_depression_norm_std': np.std(mape_depression_values),
        'n_wells': len(mape_values),
        'wells_info': wells_info
    }

# ============================================================================
# ФИЛЬТРАЦИЯ ВЫБРОСОВ
# ============================================================================

def filter_outliers(values, method='percentile', threshold=99):
    """Фильтрует выбросы."""
    if len(values) == 0:
        return [], 0

    values_arr = np.array(values)

    if method == 'percentile':
        cutoff = np.percentile(values_arr, threshold)
        mask = values_arr <= cutoff
    elif method == 'iqr':
        q1 = np.percentile(values_arr, 25)
        q3 = np.percentile(values_arr, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (values_arr >= lower) & (values_arr <= upper)
    elif method == 'cap':
        mask = values_arr <= threshold
    else:
        raise ValueError(f"Unknown method: {method}")

    filtered = values_arr[mask].tolist()
    n_outliers = len(values) - len(filtered)

    return filtered, n_outliers


def aggregate_metrics_across_samples(all_field_metrics, all_well_metrics,
                                     outlier_method='percentile', outlier_threshold=95):
    """
    Агрегирует метрики с фильтрацией выбросов НА УРОВНЕ СКВАЖИН.

    собираем ВСЕ индивидуальные значения по всем скважинам,
    а не средние по сэмплам!
    """
    # Field metrics
    field_l1_means = [m['mean'] for m in all_field_metrics]

    # ========== НОВЫЙ ПОДХОД: ФИЛЬТРАЦИЯ НА УРОВНЕ СКВАЖИН ==========
    # Собираем ВСЕ индивидуальные значения по скважинам из всех сэмплов
    individual_well_mapes = []
    individual_well_l1_depression = []
    individual_well_mape_depression = []

    total_wells = 0

    for well_metrics in all_well_metrics:
        if well_metrics['n_wells'] > 0 and 'wells_info' in well_metrics:
            # Берём ИНДИВИДУАЛЬНЫЕ значения каждой скважины
            for well_info in well_metrics['wells_info']:
                individual_well_mapes.append(well_info['mape'])
                individual_well_l1_depression.append(well_info['l1_rel_depression'])
                individual_well_mape_depression.append(well_info['mape_depression_norm'])
                total_wells += 1

    print(f"\n  Collected {total_wells} individual well measurements")

    # Фильтруем выбросы на уровне ОТДЕЛЬНЫХ СКВАЖИН
    well_mapes_filtered, mape_outliers = filter_outliers(
        individual_well_mapes, outlier_method, outlier_threshold
    )
    well_l1_filtered, l1_outliers = filter_outliers(
        individual_well_l1_depression, outlier_method, outlier_threshold
    )
    well_mape_dep_filtered, mape_dep_outliers = filter_outliers(
        individual_well_mape_depression, outlier_method, outlier_threshold
    )

    # Считаем количество сэмплов с скважинами
    n_samples_with_wells = sum(1 for m in all_well_metrics if m['n_wells'] > 0)

    aggregated = {
        'field': {
            'l1_relative': {
                'mean': np.mean(field_l1_means),
                'std': np.std(field_l1_means),
                'min': np.min(field_l1_means),
                'max': np.max(field_l1_means)
            },
        },
        'wells': {
            'n_samples_with_wells': n_samples_with_wells,
            'total_wells': total_wells,  # Общее количество скважин
            'total_wells_after_filtering': len(well_mapes_filtered),  # После фильтрации
            'mape': {
                'mean': np.mean(well_mapes_filtered) if well_mapes_filtered else 0.0,
                'std': np.std(well_mapes_filtered) if well_mapes_filtered else 0.0,
                'median': np.median(well_mapes_filtered) if well_mapes_filtered else 0.0,
                'min': np.min(well_mapes_filtered) if well_mapes_filtered else 0.0,
                'max': np.max(well_mapes_filtered) if well_mapes_filtered else 0.0,
                'n_outliers': mape_outliers
            },
            'l1_depression_norm': {
                'mean': np.mean(well_l1_filtered) if well_l1_filtered else 0.0,
                'std': np.std(well_l1_filtered) if well_l1_filtered else 0.0,
                'median': np.median(well_l1_filtered) if well_l1_filtered else 0.0,
                'min': np.min(well_l1_filtered) if well_l1_filtered else 0.0,
                'max': np.max(well_l1_filtered) if well_l1_filtered else 0.0,
                'n_outliers': l1_outliers
            },
            'mape_depression_norm': {
                'mean': np.mean(well_mape_dep_filtered) if well_mape_dep_filtered else 0.0,
                'std': np.std(well_mape_dep_filtered) if well_mape_dep_filtered else 0.0,
                'median': np.median(well_mape_dep_filtered) if well_mape_dep_filtered else 0.0,
                'min': np.min(well_mape_dep_filtered) if well_mape_dep_filtered else 0.0,
                'max': np.max(well_mape_dep_filtered) if well_mape_dep_filtered else 0.0,
                'n_outliers': mape_dep_outliers
            },
        }
    }

    return aggregated


def print_aggregated_metrics(agg_oil, agg_water, n_samples, outlier_method, outlier_threshold):
    """Вывод агрегированных метрик."""
    print("\n" + "="*80)
    print(f"AGGREGATED METRICS ACROSS {n_samples} SAMPLES")
    print(f"Outlier filtering: {outlier_method}, threshold={outlier_threshold}")
    print("="*80)

    for component, agg, icon in [('OIL', agg_oil, '🛢️'), ('WATER', agg_water, '💧')]:
        print(f"\n{icon}  {component} PRESSURE")
        print("-" * 80)
        print("  FIELD-WIDE METRICS (averaged over samples):")
        print(f"    L1 Relative Error:")
        print(f"      Mean: {agg['field']['l1_relative']['mean']:.6f} ± {agg['field']['l1_relative']['std']:.6f}")
        print(f"      Range: [{agg['field']['l1_relative']['min']:.6f}, {agg['field']['l1_relative']['max']:.6f}]")

        print("\n  WELL METRICS (per-well statistics):")
        print(f"    Total wells analyzed: {agg['wells']['total_wells']}")
        print(f"    Wells after filtering: {agg['wells']['total_wells_after_filtering']}")
        print(f"    Samples with wells: {agg['wells']['n_samples_with_wells']}/{n_samples}")

        print(f"\n    MAPE at wells:")
        print(f"      Mean: {agg['wells']['mape']['mean']:.4f}% ± {agg['wells']['mape']['std']:.4f}%")
        print(f"      Median: {agg['wells']['mape']['median']:.4f}%")
        print(f"      Range: [{agg['wells']['mape']['min']:.4f}%, {agg['wells']['mape']['max']:.4f}%]")
        if agg['wells']['mape']['n_outliers'] > 0:
            pct = 100 * agg['wells']['mape']['n_outliers'] / agg['wells']['total_wells']
            print(f"      Outliers filtered: {agg['wells']['mape']['n_outliers']}/{agg['wells']['total_wells']} ({pct:.1f}%)")

        print(f"\n    L1 (depression-normalized):")
        print(f"      Mean: {agg['wells']['l1_depression_norm']['mean']:.6f} ± {agg['wells']['l1_depression_norm']['std']:.6f}")
        print(f"      Median: {agg['wells']['l1_depression_norm']['median']:.6f}")
        print(f"      Range: [{agg['wells']['l1_depression_norm']['min']:.6f}, {agg['wells']['l1_depression_norm']['max']:.6f}]")
        if agg['wells']['l1_depression_norm']['n_outliers'] > 0:
            pct = 100 * agg['wells']['l1_depression_norm']['n_outliers'] / agg['wells']['total_wells']
            print(f"      Outliers filtered: {agg['wells']['l1_depression_norm']['n_outliers']}/{agg['wells']['total_wells']} ({pct:.1f}%)")

        print(f"\n    MAPE (depression-normalized):")
        print(f"      Mean: {agg['wells']['mape_depression_norm']['mean']:.4f}% ± {agg['wells']['mape_depression_norm']['std']:.4f}%")
        print(f"      Median: {agg['wells']['mape_depression_norm']['median']:.4f}%")
        print(f"      Range: [{agg['wells']['mape_depression_norm']['min']:.4f}%, {agg['wells']['mape_depression_norm']['max']:.4f}%]")
        if agg['wells']['mape_depression_norm']['n_outliers'] > 0:
            pct = 100 * agg['wells']['mape_depression_norm']['n_outliers'] / agg['wells']['total_wells']
            print(f"     Outliers filtered: {agg['wells']['mape_depression_norm']['n_outliers']}/{agg['wells']['total_wells']} ({pct:.1f}%)")

    print("="*80 + "\n")


# ============================================================================
# ЗАГРУЗКА ДАННЫХ (для horizontal датасетов)
# ============================================================================

def load_and_merge_datasets_with_nan_filtering(file_paths):
    """Загружает и фильтрует NaN."""
    all_clean_solutions = []
    all_clean_forcings = []

    for filepath in file_paths:
        print(f"  Loading: {filepath}")

        with h5py.File(filepath, "r") as f:
            raw_sol = torch.nan_to_num(torch.tensor(f['dataset'][:, :], dtype=torch.float32), nan=0.0)
            raw_frc = torch.nan_to_num(torch.tensor(f['source'][:, :], dtype=torch.float32), nan=0.0)

            T, N = raw_sol.shape[0], raw_sol.shape[1]

            clean_sol = []
            clean_frc = []

            for i in tqdm(range(N), desc=f"  Filtering", leave=False):
                sol = raw_sol[:, i, :, :, :]
                frc = raw_frc[i, :, :, :, :]
                frc = frc.permute(1, 0, 2, 3)

                if not (torch.isnan(sol).any() or torch.isnan(frc).any()):
                    clean_sol.append(sol)
                    clean_frc.append(frc)

            if len(clean_sol) > 0:
                all_clean_solutions.append(torch.stack(clean_sol, dim=0))
                all_clean_forcings.append(torch.stack(clean_frc, dim=0))

            print(f"    Clean: {len(clean_sol)}/{N}")

    solutions = torch.cat(all_clean_solutions, dim=0)
    forcings = torch.cat(all_clean_forcings, dim=0)

    return solutions, forcings


# ============================================================================
# MAIN
# ============================================================================

# def mainApplication():
#     # ========== НАСТРОЙКИ ==========
#     zero_thrs = 1e-6
#     valid_timesteps = 21  # Для horizontal датасета

#     # Пути к датасетам
#     filepaths = [
#         "/content/dataset_horizontal_1.hdf5",
#         "/content/dataset_horizontal_2.hdf5",
#         "/content/dataset_horizontal_3.hdf5"
#     ]

#     # Checkpoint модели
#     checkpoint_path = '/content/mamba_fno_masked_sampled_optimized_horizontal.pt'

#     # Фильтрация выбросов
#     OUTLIER_METHOD = 'percentile'  # 'percentile', 'iqr', 'cap'
#     OUTLIER_THRESHOLD = 99.9

#     # Output
#     output_dir = 'horizontal_metrics_results'
#     os.makedirs(output_dir, exist_ok=True)

#     # Device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # ========== ЗАГРУЗКА МОДЕЛИ ==========
#     print("Loading model...")
#     state_dict = torch.load(checkpoint_path, weights_only=False)
#     if '_metadata' in state_dict:
#         state_dict = {k: v for k, v in state_dict.items() if k != '_metadata'}

#     model = PostLiftMambaFNO3D(modes=(16, 48, 48), width=32, n_layers=4).to(device)
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()
#     print("✓ Model loaded successfully\n")

#     # ========== ЗАГРУЗКА ДАННЫХ ==========
#     print("Loading datasets...")
#     solutions, forcings = load_and_merge_datasets_with_nan_filtering(filepaths)
#     n_samples = len(solutions)
#     print(f"Total samples: {n_samples}")

#     # ========== ОБРАБОТКА СЭМПЛОВ ==========
#     all_field_metrics_oil = []
#     all_well_metrics_oil = []
#     all_field_metrics_water = []
#     all_well_metrics_water = []

#     print("Processing samples...")
#     for sample_idx in tqdm(range(n_samples), desc="Samples"):
#         sol = solutions[sample_idx]      # [T, C, H, W]
#         frc = forcings[sample_idx]       # [T, C, H, W]

#         # Create mask
#         mask = (torch.abs(sol[0, 0, ...]) > zero_thrs).detach().numpy()

#         # Process sample
#         field_oil, well_oil, field_water, well_water = process_single_sample(
#             model, sol, frc, mask, device, valid_timesteps=valid_timesteps
#         )

#         # Store metrics
#         all_field_metrics_oil.append(field_oil)
#         all_well_metrics_oil.append(well_oil)
#         all_field_metrics_water.append(field_water)
#         all_well_metrics_water.append(well_water)

#     print("All samples processed")

#     # ========== АГРЕГАЦИЯ С ФИЛЬТРАЦИЕЙ ==========
#     print("Aggregating metrics with outlier filtering...")
#     agg_oil = aggregate_metrics_across_samples(
#         all_field_metrics_oil, all_well_metrics_oil,
#         outlier_method=OUTLIER_METHOD, outlier_threshold=OUTLIER_THRESHOLD
#     )
#     agg_water = aggregate_metrics_across_samples(
#         all_field_metrics_water, all_well_metrics_water,
#         outlier_method=OUTLIER_METHOD, outlier_threshold=OUTLIER_THRESHOLD
#     )

#     # ========== ВЫВОД РЕЗУЛЬТАТОВ ==========
#     print_aggregated_metrics(agg_oil, agg_water, n_samples, OUTLIER_METHOD, OUTLIER_THRESHOLD)


#     print("Done")
