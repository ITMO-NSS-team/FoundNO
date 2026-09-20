# -*- coding: utf-8 -*-
"""
tnav_data.py — чтение датасета tNavigator HDF5 v6
(схема tnav_operator_hdf5_v6_compact_pvt_relperm_geometry_conn_switch_diagnostics).

Контракт (раздел 1 описания): в модель подаются ТОЛЬКО inputs/* и
targets/time_days. targets/* — это Y. diagnostics/* не используются как вход
(хук для physics-loss оставлен в комментарии в лоссе train_tnav.py).

Входные каналы модели (IN_CH = 21), порядок фиксирован:
   0  p0_n        начальное давление (норм. статистиками таргета; residual-база)
   1  sw0_n       начальная водонасыщенность (норм.)
   2  mask        активная маска слоя (0/1)
   3  poro_n      пористость (z-score по train)
   4  lpermx_n    log10(permx) (z-score)
   5  lpermy_n    log10(permy) (z-score)
   6  thick_n     cell_thickness (z-score)
   7  depth_n     cell_depth (z-score)
   8  pv_n        pore_volume (z-score)
   9..16          scheduled_grid_tensor: producer_open, injector_open,
                  producer_bhp/s, producer_orat/s, injector_rate/s,
                  injector_bhp_limit/s (s = RMS по ненулевым на train),
                  producer_control_code/3, injector_control_code/3
  17  X           координата [0,1]
  18  Y           координата [0,1]
  19  t_norm      time_days / time_days[-1]  (календарь может быть неравномерным)
  20  dist        EDT до ближайшей когда-либо открытой completion, /max(H,W)

Выход модели: 2 канала — pressure, swat (норм.), residual: out = out + x[:, :2].

PVT/ОФП/rock в текущей выборке фиксированы между кейсами (§19.3) и во вход
не подаются; при появлении вариативности добавить сюда скалярные каналы.
"""

import os
import re
import json
import hashlib

import numpy as np
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

C_OUT = 2                      # pressure, swat
N_STATIC = 6                   # poro, lpermx, lpermy, thickness, depth, pore_volume
N_CTRL = 8
IN_CH = 2 + 1 + N_STATIC + N_CTRL + 4   # = 21 (без material balance)
N_MB = 4                       # признаки материального баланса (см. ниже)


# --- дополнительные каналы датасета orthogonal_2512_bhp_rate_v1 -------------
# Инспекция показала: разломы согласованы с метаданными (barrier в 99 кейсах
# из 2512), физические скаляры реально варьируются (rock_compr +-43%,
# water_compr +-41%, oil_visc вдвое-втрое) и прямо задают амплитуду давления
# через Δp = ΔV/(c_t*V_p). Маски ГРП НЕ согласованы с метаданными
# (ненулевые при hfrac_mode='none', bbox почти на весь пласт, скважины
# накрывают лишь частично) — семантика неизвестна, поэтому только по флагу.
N_FAULT = 3      # fault_mask, log10 multx, log10 multy
N_HFRAC = 1      # hfrac_core_mask (три маски побитово совпадают)
N_PHYS = 10      # oil_visc, water_compr, water_visc, rock_compr,
                 # swl, swcr, sowcr, krwr, now, nw  (broadcast по полю)

# (путь в h5, индекс в плоском массиве) для физических скаляров
PHYS_SPEC = [
    ('inputs/pvt/pvcdo', 3),        # oil_viscosity
    ('inputs/pvt/pvtw', 2),         # water_compressibility
    ('inputs/pvt/pvtw', 3),         # water_viscosity
    ('inputs/rock/rock', 1),        # rock_compressibility
    ('inputs/relperm/coreywo', 0),  # swl
    ('inputs/relperm/coreywo', 2),  # swcr
    ('inputs/relperm/coreywo', 3),  # sowcr
    ('inputs/relperm/coreywo', 6),  # krwr
    ('inputs/relperm/coreywo', 9),  # now  (oil exponent)
    ('inputs/relperm/coreywo', 10), # nw   (water exponent)
]


def extra_channels(fault=False, hfrac=False, physics=False):
    return (N_FAULT if fault else 0) + (N_HFRAC if hfrac else 0) \
        + (N_PHYS if physics else 0)


def in_channels(material_balance=False, fault=False, hfrac=False,
                physics=False):
    return (IN_CH + (N_MB if material_balance else 0)
            + extra_channels(fault, hfrac, physics))


def signed_log1p(x):
    return torch.sign(x) * torch.log1p(x.abs())


def compute_mb_features(ctrl, tdays, static, p0, mask):
    """Признаки материального баланса — ТОЛЬКО из inputs/* и targets/time_days
    (никаких diagnostics: фактические дебиты — это результат симуляции, их
    использование было бы утечкой).

    Каналы [T, 4, H, W]:
      0  локальный накопленный чистый отбор / поровый объём ячейки
      1  глобальный накопленный чистый отбор / суммарный поровый объём
         (скаляр на шаг, размножен по сетке) — по материальному балансу
         почти линейно связан со средним падением давления
      2  локальная накопленная движущая сила Дарси k*h*(p0-p_bhp)*open*dt
         / поровый объём ячейки — работает там, где скважина на BHP-режиме
         и канал дебита в расписании не заполнен
      3  её глобальный аналог

    Всё накопленное считается по шагам <= t (cumsum), поэтому будущее в
    признак не попадает. Интегрирование с реальным dt из time_days —
    календарь неравномерный.
    """
    T = ctrl.shape[0]
    m = mask.float()
    permx = torch.pow(10.0, static[1]) * m          # static[1] = log10(permx)
    thick = static[3]
    pv = static[5]
    pv_tot = (pv * m).sum().clamp_min(1e-8)

    dt = torch.zeros_like(tdays)
    if T > 1:
        dt[1:] = tdays[1:] - tdays[:-1]
        dt[0] = dt[1]
    else:
        dt[0] = 1.0
    dt = dt.clamp_min(0.0).view(T, 1, 1)

    # --- накопленный чистый отбор (producer_orat - injector_rate) ---
    net = (ctrl[:, 3] - ctrl[:, 4]) * dt * m                     # [T,H,W]
    cum_l_rate = torch.cumsum(net, dim=0) / (pv + 1e-8) * m
    cum_g_rate = (torch.cumsum(net.sum(dim=(1, 2)), dim=0) / pv_tot
                  ).view(T, 1, 1).expand(T, *mask.shape) * m

    # --- накопленная движущая сила по Дарси (BHP-режим) ---
    drive = (ctrl[:, 0] * (p0.unsqueeze(0) - ctrl[:, 2]).clamp_min(0.0)
             - ctrl[:, 1] * (ctrl[:, 5] - p0.unsqueeze(0)).clamp_min(0.0))
    drive = drive * (permx * thick).unsqueeze(0) * dt * m
    cum_l_drive = torch.cumsum(drive, dim=0) / (pv + 1e-8) * m
    cum_g_drive = (torch.cumsum(drive.sum(dim=(1, 2)), dim=0) / pv_tot
                   ).view(T, 1, 1).expand(T, *mask.shape) * m

    out = torch.stack([cum_l_rate, cum_g_rate, cum_l_drive, cum_g_drive], dim=1)
    return signed_log1p(out)                                     # [T,4,H,W]
CTRL_VALUE_CH = (2, 3, 4, 5)   # каналы-значения (bhp/rate), масштабируются
CTRL_CODE_CH = (6, 7)          # коды типов контроля, делятся на 3

TARGET_NAMES = ['pressure', 'swat']


# ============================================================================
# СПИСОК КЕЙСОВ И СПЛИТЫ
# ============================================================================

_CASE_NUM = re.compile(r'(\d+)')


def list_cases(h5_path):
    """[(case_id:int, group_name:str)], отсортировано по case_id."""
    out = []
    with h5py.File(h5_path, 'r') as f:
        for name in f['cases'].keys():
            m = _CASE_NUM.search(name)
            if m is None:
                print(f"WARN: не удалось извлечь номер из группы '{name}', пропуск")
                continue
            out.append((int(m.group(1)), name))
    out.sort(key=lambda x: x[0])
    return out


def parse_ranges(spec):
    """'512-639,1920-2047' -> set(int). Пустая строка -> пустое множество."""
    ids = set()
    if not spec:
        return ids
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            ids.update(range(int(a), int(b) + 1))
        else:
            ids.add(int(part))
    return ids


def make_splits(cases, val_sets, limit_cases=0):
    """
    cases: [(id, name)]; val_sets: dict имя_сплита -> строка диапазонов.
    Возвращает (train_names, {split: names}).
    """
    val_ids = {k: parse_ranges(v) for k, v in val_sets.items() if v}
    all_val = set().union(*val_ids.values()) if val_ids else set()

    train_names = [n for i, n in cases if i not in all_val]
    val_names = {k: [n for i, n in cases if i in ids] for k, ids in val_ids.items()}

    if limit_cases:  # быстрые смоук-запуски
        train_names = train_names[:limit_cases]
        val_names = {k: v[:max(2, limit_cases // 4)] for k, v in val_names.items()}

    return train_names, val_names


def cases_with_faults(h5_path, names=None):
    """Кейсы, где есть барьерный разлом (fault_mask ненулевая).

    Единственная структурная ось, которая в датасете
    orthogonal_2512_bhp_rate_v1 действительно выражена и согласована с
    метаданными: геология вырождена (86% кейсов на одном field_case_index),
    ГРП по метаданным выключен везде, а разломы честно присутствуют
    примерно в 4% кейсов. Отложив их целиком, получаем осмысленный OOD:
    "модель не видела разломов при обучении".
    """
    out = []
    with h5py.File(h5_path, 'r') as f:
        todo = names if names is not None else [n for _, n in list_cases(h5_path)]
        for n in todo:
            fm = np.asarray(f[f'cases/{n}/inputs/static/fault_mask'][:])
            if fm.any():
                out.append(n)
    return out


def split_random(names, frac=0.1, seed=0):
    """Случайный holdout с фиксированным сидом — честная, но слабая проверка:
    геология в датасете практически одна, так что это оценка in-distribution."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(names))
    k = max(1, int(round(len(names) * frac)))
    hold = {names[i] for i in idx[:k]}
    return [n for n in names if n not in hold], [n for n in names if n in hold]


def inspect_dataset(h5_path, n_show=1):
    """Печатает структуру первых кейсов — для быстрой проверки в ноутбуке."""
    cases = list_cases(h5_path)
    print(f"Cases: {len(cases)} (id {cases[0][0]}..{cases[-1][0]})")
    with h5py.File(h5_path, 'r') as f:
        for _, name in cases[:n_show]:
            print(f"\n[{name}]")
            def walk(g, prefix=''):
                for k, v in g.items():
                    if isinstance(v, h5py.Dataset):
                        print(f"  {prefix}{k:40s} {v.shape} {v.dtype}")
                    else:
                        walk(v, prefix + k + '/')
            walk(f[f'cases/{name}'])
    return cases


# ============================================================================
# ЧТЕНИЕ ОДНОГО КЕЙСА
# ============================================================================

def _arr(g, key):
    return np.asarray(g[key][:], dtype=np.float32)


def read_static_raw(f, name, fault=False, hfrac=False, physics=False):
    """mask [H,W] bool, static_stack [6+K,H,W] float (сырой), p0/sw0 [H,W].

    K зависит от флагов: разломы (+3), маска ГРП (+1), физические скаляры
    (+10, broadcast константами по полю). Дополнительные каналы кладутся в
    тот же стек, поэтому нормализация, маскирование и сборка входа работают
    для них без изменений.
    Неактивные ячейки содержат константы-заглушки экспортёра (для depth —
    порядка -1e9), поэтому все поля зануляются по маске ещё ДО нормализации."""
    gi = f[f'cases/{name}/inputs']
    mask = np.asarray(gi['static/mask'][:]) > 0.5
    mf = mask.astype(np.float32)
    poro = _arr(gi, 'static/poro') * mf
    lpx = np.log10(np.clip(_arr(gi, 'static/permx') * mf, 1e-3, None))
    lpy = np.log10(np.clip(_arr(gi, 'static/permy') * mf, 1e-3, None))
    thick = _arr(gi, 'geometry/cell_thickness') * mf
    depth = _arr(gi, 'geometry/cell_depth') * mf
    pv = _arr(gi, 'geometry/pore_volume') * mf
    chans = [poro, lpx, lpy, thick, depth, pv]

    if fault:
        fm = np.asarray(gi['static/fault_mask'][:], np.float32) * mf
        lmx = np.log10(np.clip(_arr(gi, 'static/multx'), 1e-6, None)) * mf
        lmy = np.log10(np.clip(_arr(gi, 'static/multy'), 1e-6, None)) * mf
        chans += [fm, lmx, lmy]

    if hfrac:
        hm = np.asarray(gi['static/hfrac_core_mask'][:], np.float32) * mf
        chans.append(hm)

    if physics:
        g = f[f'cases/{name}']
        ones = np.ones_like(mf)
        for path, idx in PHYS_SPEC:
            v = float(np.asarray(g[path][:], np.float64).ravel()[idx])
            chans.append(ones * v * mf)

    static = np.stack(chans, axis=0)
    p0 = _arr(gi, 'initial/pressure') * mf
    sw0 = _arr(gi, 'initial/swat') * mf
    return mask, static, p0, sw0


AUX_KEYS = ('q_o_surface', 'q_w_surface')   # diagnostics/actual_connection_source_terms


def read_aux_rates(f, name):
    """Фактические поточечные дебиты (нефть, вода) в completion-ячейках,
    [T, 2, H, W] или None, если группы нет в файле.

    Это diagnostics/*, то есть ВЫХОД симулятора — по контракту схемы такие
    поля нельзя подавать на вход, но можно использовать как auxiliary target.
    Дебит у скважины — это градиент давления умножить на проводимость, то
    есть ровно та прискважинная физика, которая даёт хвост ошибки; голова,
    обязанная его предсказывать, заставляет модель разрешать окрестность
    скважины честно (multi-task).
    """
    g = f[f'cases/{name}']
    base = 'diagnostics/actual_connection_source_terms'
    if base not in g:
        return None
    grp = g[base]
    arrs = []
    for k in AUX_KEYS:
        if k in grp:
            arrs.append(np.asarray(grp[k][:], np.float32))
    if not arrs:      # запасной вариант: любые (T,H,W) массивы в группе
        for k in grp.keys():
            a = np.asarray(grp[k][:], np.float32)
            if a.ndim == 3:
                arrs.append(a)
            if len(arrs) == 2:
                break
    if not arrs:
        return None
    if len(arrs) == 1:
        arrs.append(np.zeros_like(arrs[0]))
    return np.stack(arrs[:2], axis=1)              # [T,2,H,W]


def read_dynamic_raw(f, name):
    """ctrl [T,8,H,W], targets [T,2,H,W], time_days [T]."""
    g = f[f'cases/{name}']
    ctrl = np.asarray(g['inputs/controls/scheduled_grid_tensor'][:], dtype=np.float32)
    press = _arr(g, 'targets/pressure')
    swat = _arr(g, 'targets/swat')
    y = np.stack([press, swat], axis=1)                 # [T,2,H,W]
    tdays = _arr(g, 'targets/time_days')
    return ctrl, y, tdays


# ============================================================================
# НОРМАЛИЗАЦИЯ
# ============================================================================

class NormalizerV6(nn.Module):
    def __init__(self, tgt_mean, tgt_std, stat_mean, stat_std, ctrl_scale,
                 mb_mean=None, mb_std=None, use_mb=False,
                 extra_flags=None, aux_scale=None):
        super().__init__()
        self.register_buffer('tgt_mean', torch.as_tensor(tgt_mean).float())
        self.register_buffer('tgt_std', torch.as_tensor(tgt_std).float())
        self.register_buffer('stat_mean', torch.as_tensor(stat_mean).float())
        self.register_buffer('stat_std', torch.as_tensor(stat_std).float())
        self.register_buffer('ctrl_scale', torch.as_tensor(ctrl_scale).float())
        if mb_mean is None:
            mb_mean, mb_std = torch.zeros(N_MB), torch.ones(N_MB)
        self.register_buffer('mb_mean', torch.as_tensor(mb_mean).float())
        self.register_buffer('mb_std', torch.as_tensor(mb_std).float())
        self.register_buffer('use_mb', torch.tensor(bool(use_mb)))
        # какие группы доп. каналов включены: [fault, hfrac, physics].
        # Хранится в нормализаторе, чтобы predict_case и eval-скрипты
        # восстанавливали ту же раскладку входа без лишних аргументов.
        # aux-дебиты: разреженные (ненулевые только у скважин) и с широким
        # диапазоном -> signed log1p(|q|/scale), scale = медиана ненулевых.
        self.register_buffer('use_aux', torch.tensor(aux_scale is not None))
        self.register_buffer('aux_scale', torch.as_tensor(
            [1.0, 1.0] if aux_scale is None else list(aux_scale)).float())
        if extra_flags is None:
            extra_flags = [False, False, False]
        self.register_buffer('extra_flags',
                             torch.tensor([bool(v) for v in extra_flags]))

    def norm_aux(self, q):                                  # [T,2,H,W]
        s = self.aux_scale.view(1, -1, 1, 1)
        return torch.sign(q) * torch.log1p(q.abs() / s)

    def denorm_aux(self, z):
        s = self.aux_scale.view(1, -1, 1, 1)
        return torch.sign(z) * (torch.expm1(z.abs()) * s)

    @property
    def extras(self):
        f = self.extra_flags.tolist()
        return dict(fault=bool(f[0]), hfrac=bool(f[1]), physics=bool(f[2]))

    def _sh(self, x, ch_dim):
        s = [1] * x.ndim
        s[ch_dim] = -1
        return s

    def norm_targets(self, y, ch_dim=1):
        s = self._sh(y, ch_dim)
        return (y - self.tgt_mean.view(s)) / self.tgt_std.view(s)

    def denorm_targets(self, y, ch_dim=1):
        s = self._sh(y, ch_dim)
        return y * self.tgt_std.view(s) + self.tgt_mean.view(s)

    def norm_static(self, stack):                        # [6,H,W]
        return (stack - self.stat_mean.view(-1, 1, 1)) / self.stat_std.view(-1, 1, 1)

    def norm_mb(self, mb, ch_dim=1):
        s = self._sh(mb, ch_dim)
        return (mb - self.mb_mean.view(s)) / self.mb_std.view(s).clamp_min(1e-8)

    def norm_controls(self, ctrl):                       # [T,8,H,W]
        out = ctrl.clone()
        for i, c in enumerate(CTRL_VALUE_CH):
            out[:, c] = ctrl[:, c] / self.ctrl_scale[i]
        for c in CTRL_CODE_CH:
            out[:, c] = ctrl[:, c] / 3.0
        return out

    def state(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    @classmethod
    def from_state(cls, d):
        # старые чекпойнты не содержат mb-полей — деградируем в use_mb=False
        return cls(d['tgt_mean'], d['tgt_std'], d['stat_mean'],
                   d['stat_std'], d['ctrl_scale'],
                   mb_mean=d.get('mb_mean'), mb_std=d.get('mb_std'),
                   use_mb=bool(d.get('use_mb', torch.tensor(False))),
                   extra_flags=d.get('extra_flags'),
                   aux_scale=(d['aux_scale'].tolist()
                              if bool(d.get('use_aux', torch.tensor(False)))
                              else None))


def fit_stats(h5_path, train_names, cache_path=None, force=False,
              material_balance=False, fault=False, hfrac=False,
              physics=False, aux_rates=False):
    """
    Один потоковый проход по train-кейсам. Статистики:
      targets  — mean/std по активным ячейкам, все T;
      static   — mean/std по активным ячейкам;
      controls — RMS по ненулевым значениям каналов-значений.
    Кэшируется в cache_path (ключ — hash списка train-кейсов).
    """
    key = hashlib.md5(('|'.join(train_names)
                       + f'|mb={material_balance}|f={fault}|h={hfrac}'
                       + f'|p={physics}|aux={aux_rates}').encode()).hexdigest()[:12]
    if cache_path and os.path.exists(cache_path) and not force:
        d = torch.load(cache_path, map_location='cpu', weights_only=False)
        if d.get('key') == key:
            print(f"Stats: загружены из кэша {cache_path}")
            return NormalizerV6.from_state(d['state'])
        print("Stats: кэш от другого сплита, пересчитываю")

    tsum = np.zeros(C_OUT, np.float64)
    tsq = np.zeros(C_OUT, np.float64)
    tcnt = 0.0
    aux_vals = [[], []]
    n_stat = N_STATIC + extra_channels(fault, hfrac, physics)
    ssum = np.zeros(n_stat, np.float64)
    ssq = np.zeros(n_stat, np.float64)
    scnt = 0.0
    csq = np.zeros(len(CTRL_VALUE_CH), np.float64)
    ccnt = np.zeros(len(CTRL_VALUE_CH), np.float64)
    mbsum = torch.zeros(N_MB, dtype=torch.float64)
    mbsq = torch.zeros(N_MB, dtype=torch.float64)
    mbcnt = 0.0

    with h5py.File(h5_path, 'r') as f:
        for name in tqdm(train_names, desc='Fitting stats', ncols=90):
            mask, static, p0, _ = read_static_raw(f, name, fault, hfrac, physics)
            ctrl, y, tdays = read_dynamic_raw(f, name)
            m = mask.astype(np.float64)
            nm = m.sum()
            T = y.shape[0]

            tsum += (y.astype(np.float64) * m).sum(axis=(0, 2, 3))
            tsq += ((y.astype(np.float64) ** 2) * m).sum(axis=(0, 2, 3))
            tcnt += nm * T

            ssum += (static.astype(np.float64) * m).sum(axis=(1, 2))
            ssq += ((static.astype(np.float64) ** 2) * m).sum(axis=(1, 2))
            scnt += nm

            for i, c in enumerate(CTRL_VALUE_CH):
                v = ctrl[:, c]
                nz = np.abs(v) > 0
                csq[i] += (v[nz].astype(np.float64) ** 2).sum()
                ccnt[i] += nz.sum()

            if aux_rates:
                q = read_aux_rates(f, name)
                if q is not None:
                    for i in range(2):
                        nz = np.abs(q[:, i]) > 0
                        v = np.abs(q[:, i][nz])
                        if v.size > 20000:
                            v = v[np.random.default_rng(0).choice(v.size, 20000, replace=False)]
                        aux_vals[i].append(v)

            if material_balance:
                mb = compute_mb_features(
                    torch.from_numpy(ctrl), torch.from_numpy(tdays),
                    torch.from_numpy(static), torch.from_numpy(p0),
                    torch.from_numpy(mask)).double()          # [T,4,H,W]
                mm = torch.from_numpy(mask).double()[None, None]
                mbsum += (mb * mm).sum(dim=(0, 2, 3))
                mbsq += ((mb ** 2) * mm).sum(dim=(0, 2, 3))
                mbcnt += float(mm.sum()) * mb.shape[0]

    tmean = tsum / max(tcnt, 1.0)
    tstd = np.sqrt(np.maximum(tsq / max(tcnt, 1.0) - tmean ** 2, 1e-12))
    smean = ssum / max(scnt, 1.0)
    sstd = np.sqrt(np.maximum(ssq / max(scnt, 1.0) - smean ** 2, 1e-12))
    cscale = np.sqrt(csq / np.maximum(ccnt, 1.0))
    cscale = np.where(cscale > 1e-8, cscale, 1.0)

    print("Target stats:", {TARGET_NAMES[i]: (round(float(tmean[i]), 4),
                                              round(float(tstd[i]), 4))
                            for i in range(C_OUT)})
    print("Ctrl value RMS:", [round(float(x), 4) for x in cscale])

    if material_balance:
        mb_mean = mbsum / max(mbcnt, 1.0)
        mb_std = (mbsq / max(mbcnt, 1.0) - mb_mean ** 2).clamp_min(1e-12).sqrt()
        print("Material-balance stats:", [f"{v:.4g}" for v in mb_mean.tolist()],
              "std", [f"{v:.4g}" for v in mb_std.tolist()])
    else:
        mb_mean = mb_std = None

    aux_scale = None
    if aux_rates:
        aux_scale = []
        for i in range(2):
            v = np.concatenate(aux_vals[i]) if aux_vals[i] else np.array([1.0])
            aux_scale.append(float(max(np.median(v), 1e-8)))
        print(f"Aux-дебиты (diagnostics): scale (медиана |q| ненулевых) = "
              f"{[round(s, 4) for s in aux_scale]}")
    norm = NormalizerV6(tmean, tstd, smean, sstd, cscale,
                        mb_mean=mb_mean, mb_std=mb_std,
                        use_mb=material_balance,
                        extra_flags=[fault, hfrac, physics],
                        aux_scale=aux_scale)
    n_ex = extra_channels(fault, hfrac, physics)
    if n_ex:
        print(f"Доп. каналы: fault={fault} hfrac={hfrac} physics={physics} "
              f"(+{n_ex} каналов, static-стек {n_stat})")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        torch.save({'key': key, 'state': norm.state()}, cache_path)
        print(f"Stats: сохранены в {cache_path}")
    return norm


# ============================================================================
# DATASET / COLLATE
# ============================================================================

class TnavDataset(Dataset):
    """Ленивое чтение из одного HDF5. Файл открывается в каждом worker'е
    отдельно (после fork), поэтому num_workers > 0 безопасен."""

    def __init__(self, h5_path, case_names, normalizer):
        self.h5_path = h5_path
        self.names = list(case_names)
        self.norm = normalizer
        self._f = None
        # Кэш того, что не меняется между эпохами: статические каналы,
        # маска и distance transform. EDT по сетке 134x113 на каждом
        # обращении — заметная доля времени даталоадера, а результат
        # для кейса всегда один и тот же.
        self._cache = {}
        self.cache_static = True
        with h5py.File(h5_path, 'r') as f:   # только размеры, handle не храним
            ctrl, y, _ = read_dynamic_raw(f, self.names[0])
        self.T, _, self.H, self.W = y.shape

    def _file(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, 'r')
        return self._f

    def __len__(self):
        return len(self.names)

    def _static_cached(self, f, name):
        """Статика, маска, EDT и p0 — одни и те же на всех эпохах."""
        hit = self._cache.get(name)
        if hit is not None:
            return hit
        ex = self.norm.extras
        mask, static, p0, _ = read_static_raw(
            f, name, ex['fault'], ex['hfrac'], ex['physics'])
        ctrl0 = np.asarray(
            f[f'cases/{name}/inputs/controls/scheduled_grid_tensor'][:, :2],
            dtype=np.float32)
        ever = ((ctrl0[:, 0] > 0.5) | (ctrl0[:, 1] > 0.5)).any(axis=0)
        if ever.any():
            d = distance_transform_edt(~ever)
        else:
            d = np.full(mask.shape, float(max(mask.shape)))
        out = (torch.from_numpy(mask),
               self.norm.norm_static(torch.from_numpy(static)),
               torch.from_numpy(d).float() / float(max(mask.shape)),
               torch.from_numpy(p0),
               torch.from_numpy(static))
        if self.cache_static:
            self._cache[name] = out
        return out

    def __getitem__(self, idx):
        f = self._file()
        name = self.names[idx]
        mask_t, static_n, dist, p0_t, static_raw = self._static_cached(f, name)
        ctrl, y, tdays = read_dynamic_raw(f, name)

        ctrl_t = torch.from_numpy(ctrl)
        ctrl_n = self.norm.norm_controls(ctrl_t)
        y_n = self.norm.norm_targets(torch.from_numpy(y), ch_dim=1)
        t_norm = torch.from_numpy(tdays / max(float(tdays[-1]), 1e-8))
        open_mask = (ctrl_t[:, 0] > 0.5) | (ctrl_t[:, 1] > 0.5)   # [T,H,W]

        item = {
            'y': y_n, 'ctrl': ctrl_n, 'static': static_n, 'mask': mask_t,
            't': t_norm.float(), 'well': open_mask, 'dist': dist, 'case': name,
            'well_inj': ctrl_t[:, 1] > 0.5,                    # для веса нагнетательных
        }
        if bool(self.norm.use_aux):
            q = read_aux_rates(f, name)
            if q is None:
                q = np.zeros((ctrl.shape[0], 2) + mask_t.shape, np.float32)
            item['aux'] = self.norm.norm_aux(torch.from_numpy(q))   # [T,2,H,W]
        if bool(self.norm.use_mb):
            mb = compute_mb_features(ctrl_t, torch.from_numpy(tdays),
                                     static_raw, p0_t, mask_t)
            item['mb'] = self.norm.norm_mb(mb, ch_dim=1)      # [T,N_MB,H,W]
        return item


class TnavCollate:
    """Общий t0 на батч => батчинг при переменной длине окна.
    train: t0 ~ U[0, T-min_window]; val: t0 = 0."""

    def __init__(self, min_window, train):
        self.min_window = min_window
        self.train = train
        self._grids = {}

    def _grid(self, H, W):
        if (H, W) not in self._grids:
            x = torch.linspace(0, 1, W)
            y = torch.linspace(0, 1, H)
            Y, X = torch.meshgrid(y, x, indexing='ij')
            self._grids[(H, W)] = (X, Y)
        return self._grids[(H, W)]

    def __call__(self, items):
        T_full = items[0]['y'].shape[0]
        mw = min(self.min_window, T_full)
        t0 = np.random.randint(0, T_full - mw + 1) if self.train else 0

        xs, ys, rms, wms, wis, auxs = [], [], [], [], [], []
        for it in items:
            y = it['y'][t0:]                       # [Tw,2,H,W]
            ctrl = it['ctrl'][t0:]                 # [Tw,8,H,W]
            Tw, _, H, W = y.shape
            X, Y = self._grid(H, W)

            m2 = it['mask'].float()                    # [H,W]
            # Вне активной маски статика/u0 после z-score могут быть
            # экстремальными (depth=0 при малом std и т.п.) — зануляем:
            # 0 в норм. пространстве = "среднее", а FNO как глобальный
            # оператор иначе разносит этот мусор по всему полю.
            u0 = (y[0] * m2).unsqueeze(0).expand(Tw, -1, -1, -1)
            mch = m2.view(1, 1, H, W).expand(Tw, 1, H, W)
            sch = (it['static'] * m2).unsqueeze(0).expand(Tw, -1, -1, -1)
            Xc = X.view(1, 1, H, W).expand(Tw, 1, H, W)
            Yc = Y.view(1, 1, H, W).expand(Tw, 1, H, W)
            tc = it['t'][t0:].view(Tw, 1, 1, 1).expand(Tw, 1, H, W)
            dc = it['dist'].view(1, 1, H, W).expand(Tw, 1, H, W)

            parts = [u0, mch, sch, ctrl, Xc, Yc, tc, dc]
            if 'mb' in it:
                parts.append(it['mb'][t0:])                   # каналы 21..24
            x = torch.cat(parts, dim=1)
            xs.append(x.permute(1, 0, 2, 3))       # [21,Tw,H,W]
            ys.append(y.permute(1, 0, 2, 3))       # [2,Tw,H,W]
            rms.append(it['mask'])
            wms.append(it['well'][t0:])
            wis.append(it['well_inj'][t0:])
            if 'aux' in it:
                auxs.append(it['aux'][t0:].permute(1, 0, 2, 3))    # [2,Tw,H,W]

        out = {
            'x': torch.stack(xs), 'y': torch.stack(ys),
            'res': torch.stack(rms), 'well': torch.stack(wms),
            'well_inj': torch.stack(wis),
        }
        if auxs:
            out['aux'] = torch.stack(auxs)                     # [B,2,Tw,H,W]
        return out


# ============================================================================
# ИНФЕРЕНС ОДНОГО КЕЙСА (t0=0, u0 = inputs/initial — чистый контракт X->Y)
# ============================================================================

@torch.no_grad()
def predict_case(model, normalizer, h5_path, case_name, device,
                 clamp_swat=True):
    ex = normalizer.extras
    with h5py.File(h5_path, 'r') as f:
        mask, static, p0, sw0 = read_static_raw(
            f, case_name, ex['fault'], ex['hfrac'], ex['physics'])
        ctrl, y, tdays = read_dynamic_raw(f, case_name)
    static_raw_np, p0_np = static, p0

    H, W = mask.shape
    T = y.shape[0]
    norm_cpu = NormalizerV6.from_state(normalizer.state())

    mask_t = torch.from_numpy(mask)
    mfl = mask_t.float()
    static_n = norm_cpu.norm_static(torch.from_numpy(static)) * mfl
    ctrl_t = torch.from_numpy(ctrl)
    ctrl_n = norm_cpu.norm_controls(ctrl_t)
    u0_phys = torch.from_numpy(np.stack([p0, sw0], axis=0))         # [2,H,W]
    u0_n = norm_cpu.norm_targets(u0_phys, ch_dim=0) * mfl
    t_norm = torch.from_numpy(tdays / max(float(tdays[-1]), 1e-8)).float()

    open_mask = (ctrl_t[:, 0] > 0.5) | (ctrl_t[:, 1] > 0.5)
    ever = open_mask.any(dim=0)
    d = distance_transform_edt((~ever).numpy()) if bool(ever.any()) \
        else np.full((H, W), float(max(H, W)))
    dist = torch.from_numpy(d).float() / float(max(H, W))

    x_lin = torch.linspace(0, 1, W)
    y_lin = torch.linspace(0, 1, H)
    Yg, Xg = torch.meshgrid(y_lin, x_lin, indexing='ij')

    u0b = u0_n.unsqueeze(0).expand(T, -1, -1, -1)
    x = torch.cat([
        u0b,
        mask_t.float().view(1, 1, H, W).expand(T, 1, H, W),
        static_n.unsqueeze(0).expand(T, -1, -1, -1),
        ctrl_n,
        Xg.view(1, 1, H, W).expand(T, 1, H, W),
        Yg.view(1, 1, H, W).expand(T, 1, H, W),
        t_norm.view(T, 1, 1, 1).expand(T, 1, H, W),
        dist.view(1, 1, H, W).expand(T, 1, H, W),
    ], dim=1)
    if bool(normalizer.use_mb):
        mb = compute_mb_features(ctrl_t, torch.from_numpy(tdays),
                                 torch.from_numpy(static_raw_np),
                                 torch.from_numpy(p0_np), mask_t)
        mb = NormalizerV6.from_state(normalizer.state()).norm_mb(mb, ch_dim=1)
        x = torch.cat([x, mb], dim=1)
    x = x.permute(1, 0, 2, 3).unsqueeze(0)                          # [1,C,T,H,W]

    model.eval()
    pred_n = model(x.to(device))
    pred = normalizer.denorm_targets(pred_n, ch_dim=1)[0].cpu()      # [2,T,H,W]
    if clamp_swat:
        pred[1].clamp_(0.0, 1.0)

    return {
        'pred': pred.permute(1, 0, 2, 3),        # [T,2,H,W] физ. единицы
        'target': torch.from_numpy(y),           # [T,2,H,W]
        'mask': mask_t,
        'well': open_mask,
        'time_days': torch.from_numpy(tdays),
        'case': case_name,
    }
