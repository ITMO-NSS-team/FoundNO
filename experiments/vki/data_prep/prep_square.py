#!/usr/bin/env python3
"""
prep_square.py - VKI-LS59 data preparation: mesh -> unit square (DNO_dataset).

Ported from the VKI-LS59 workspace (dno_data_prep.py). Pipeline per sample:
  1. Harmonic map "mesh -> square" (fnofound.utils.square_dev: boundary_loop
     -> split_sides -> parametrize(by='index') -> harmonic_map with clamped
     cotangents). Square [0,1]x[0,1]: left=Inflow (u=0), right=Outflow (u=1),
     top=Intrado (v=1), bottom=Extrado (v=0); the v axis is periodic.
  2. Quadratic interpolation of the fields from mesh nodes onto the regular
     Nu x Nv grid (tensor biquadratic P2 on the structured (i,j) layout,
     Newton inversion of the (u,v) -> (i,j) map).
  3. Physical clamps: sdf >= 0, nut >= 0.
  4. M_iso 1D profiles along the blade (Base_1_2), resampled to Nu points.
  5. Gather map (u,v) of every mesh node (for back-to-mesh evaluation).
  6. Saving to the DNO_dataset dir (raw values, no normalization):
       npy (channel-last): train_inputs/train_outputs/train_scalars/
         train_out_scalars/train_miso/train_node_uv, test_inputs/test_scalars/
         test_node_uv, ij_to_node, wall_node_ids, meta.json
       csv (original DNO repo format): train_x/y_data, train_C, train_U_*,
         test_x/y_data, test_C, b.csv

Usage:
    python -m experiments.vki.data_prep.prep_square \\
        --plaid_dir /media/.../VKI-LS59/plaid_dataset \\
        --out /media/.../VKI-LS59/DNO_dataset \\
        --grid 128 128 --jobs 1
    python -m experiments.vki.data_prep.prep_square --ids 0 1 2   # debug, no save
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from fnofound.utils import square_dev as sd

# ═══════════════ конфигурация ═══════════════

DEFAULT_PLAID_DIR = "/media/expiwt/f2d8c27a-bdac-4400-b1be-cdb75b6b5a40/hugging_face/VKI-LS59/plaid_dataset"
DEFAULT_OUT = "/media/expiwt/f2d8c27a-bdac-4400-b1be-cdb75b6b5a40/3sem/hugging_face/VKI-LS59/DNO_dataset"

BASE = "Base_2_2"
BASE_WALL = "Base_1_2"                                   # 1D-сетка поверхности лопатки
FIELD_NAMES = ["mach", "nut", "ro", "roe", "rou", "rov"]  # 2D-выходы
IN_CHANNELS = ["x", "y", "sdf"]                           # 2D-входы
SCALAR_IN = ["angle_in", "mach_out"]
SCALAR_OUT = ["Q", "power", "Pr", "Tr", "eth_is", "angle_out"]
MISO_CHANNELS = ["M_iso_koryto_v0", "M_iso_spinka_v1"]    # корыто (v=0), спинка (v=1)

# ═══════════════ (i,j)-структура (константна для всех сэмплов) ═══════════════

_IJ_CACHE = {}   # "topo" -> (ij_to_node, ij, quads_ref)


def get_ij(quads, nodes, loop, sides):
    """(i,j)-прямоугольник структурированной сетки + таблица узел↔(i,j).
    Топология константна → кэшируем по первому сэмплу (с контролем связности)."""
    key = "topo"
    if key in _IJ_CACHE:
        ij_to_node, ij, quads_ref = _IJ_CACHE[key]
        assert quads.shape == quads_ref.shape and (quads == quads_ref).all(), \
            "топология изменилась между сэмплами — кэш (i,j) недействителен"
        assert ij.shape[0] == len(nodes)
        return ij_to_node, ij
    ij, _ = sd.structured_ij(nodes, quads, sides, loop)
    n = len(nodes)
    ni = int(ij[:, 0].max()) + 1
    nj = int(ij[:, 1].max()) + 1
    assert ij.shape[0] == n and (ij >= 0).all(), "не все узлы получили (i,j)"
    assert ni * nj == n, f"ni*nj={ni*nj} != n={n} — (i,j) не биекция"
    ij_to_node = np.full((ni, nj), -1, dtype=np.int64)
    for k in range(n):
        i, j = ij[k]
        assert ij_to_node[i, j] == -1, f"(i,j)=({i},{j}) занят дважды"
        ij_to_node[i, j] = k
    _IJ_CACHE[key] = (ij_to_node, ij, quads.copy())
    return _IJ_CACHE[key][:2]


# ═══════════════ биквадратичная интерполяция в (i,j) ═══════════════

def _lagrange_basis(t):
    """1D квадратичный Лагранж на узлах {-1,0,+1}: (l_{-1}, l_0, l_{+1})."""
    return np.stack([0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)], axis=-1)


def _lagrange_deriv(t):
    """Производные базиса по t."""
    return np.stack([t - 0.5, -2.0 * t, t + 0.5], axis=-1)


def _biquad_eval(stencil, t, s):
    """Тензорная биквадратичная интерполяция по 3×3-шаблону.
    stencil: [..., 3, 3] (последние две оси — (i,j)-шаблон);
    t, s:    локальные координаты точки в ячейке.
    value = Σ_ij L_i(t) L_j(s) * stencil[..., i, j]
    """
    Li = _lagrange_basis(t)     # [..., 3]
    Lj = _lagrange_basis(s)     # [..., 3]
    return np.einsum('...ij,...i,...j->...', stencil, Li, Lj)


def _biquad_map_jac(stencil_uv, t, s):
    """Значение карты (u,v) и её якобиан d(u,v)/d(t,s).
    stencil_uv: [..., 3, 3, 2]; возвращает X, dXdt, dXds размера [..., 2]."""
    Li = _lagrange_basis(t)
    Lj = _lagrange_basis(s)
    dLi = _lagrange_deriv(t)
    dLj = _lagrange_deriv(s)
    X = np.einsum('...ijc,...i,...j->...c', stencil_uv, Li, Lj)
    dXdt = np.einsum('...ijc,...i,...j->...c', stencil_uv, dLi, Lj)
    dXds = np.einsum('...ijc,...i,...j->...c', stencil_uv, Li, dLj)
    return X, dXdt, dXds


def biquad_resample(uv, fields, ij_to_node, U, V, newton_iters=5):
    """Биквадратичная интерполяция полей с узлов меша на регулярную сетку.

    uv:     (n, 2) — гармоническая карта узлов
    fields: список (n,) — значения полей в узлах
    U, V:   meshgrid (indexing='ij') регулярной сетки

    Возвращает [len(fields), Nu, Nv] float64.

    Границы: для ячейки i0=0 (или j0=0) шаблон сдвигается на {0,1,2} вместо
    {-1,0,1} (якорь max(i0,1), локальная координата t-1). Дублирование узла
    занижало бы производную вдвое и ломало ньютоновскую инверсию; виртуальный
    узел линейным продолжением неточен для квадратичных полей.
    """
    ni, nj = ij_to_node.shape
    node_uv = uv[ij_to_node]                    # (ni, nj, 2) — карта в (i,j)
    fields_grid = [f[ij_to_node] for f in fields]   # (ni, nj)

    q = np.column_stack([U.ravel(), V.ravel()])
    nq = len(q)

    def stencil_coords(iq, jq):
        """(i0, j0, t, s, rows, cols) для ньютоновской итерации/оценки."""
        i0 = np.floor(iq).astype(np.int64)
        j0 = np.floor(jq).astype(np.int64)
        np.clip(i0, 0, ni - 2, out=i0)
        np.clip(j0, 0, nj - 2, out=j0)
        t = iq - i0
        s = jq - j0
        ic = np.maximum(i0, 1)                  # якорь шаблона (граница: {0,1,2})
        jc = np.maximum(j0, 1)
        te = np.where(i0 == 0, t - 1.0, t)      # локальная координата от якоря
        se = np.where(j0 == 0, s - 1.0, s)
        rows = np.stack([ic - 1, ic, ic + 1], axis=-1)
        cols = np.stack([jc - 1, jc, jc + 1], axis=-1)
        return i0, j0, te, se, rows, cols

    # ── ньютоновская инверсия (u,v) → (i,j) ──
    iq = q[:, 0] * (ni - 1.0)
    jq = q[:, 1] * (nj - 1.0)

    for _ in range(newton_iters):
        i0, j0, te, se, rows, cols = stencil_coords(iq, jq)
        st = node_uv[rows[:, :, None], cols[:, None, :]]            # (nq,3,3,2)

        X, dXdt, dXds = _biquad_map_jac(st, te, se)
        det = dXdt[..., 0] * dXds[..., 1] - dXdt[..., 1] * dXds[..., 0]
        ok = np.abs(det) > 1e-10
        di = np.where(ok, (dXds[..., 1] * (q[:, 0] - X[..., 0]) - dXds[..., 0] * (q[:, 1] - X[..., 1])) / np.where(ok, det, 1.0), 0.0)
        dj = np.where(ok, (-dXdt[..., 1] * (q[:, 0] - X[..., 0]) + dXdt[..., 0] * (q[:, 1] - X[..., 1])) / np.where(ok, det, 1.0), 0.0)
        iq = iq + di
        jq = jq + dj

    iq = np.clip(iq, 0.0, ni - 1.0)
    jq = np.clip(jq, 0.0, nj - 1.0)

    # ── финальная оценка полей ──
    i0, j0, te, se, rows, cols = stencil_coords(iq, jq)
    out = np.empty((len(fields), nq))
    for fi, fg in enumerate(fields_grid):
        st = fg[rows[:, :, None], cols[:, None, :]]     # (nq, 3, 3)
        out[fi] = _biquad_eval(st, te, se)
    return out.reshape(len(fields), U.shape[0], U.shape[1])


# ═══════════════ сэмпл → квадрат ═══════════════

def sample_to_square(s):
    """Гармоническая карта сэмпла: (uv, tris, quads, nodes, loop, sides, tag_of)."""
    nodes = s.get_nodes(base_name=BASE)
    els = s.get_elements(base_name=BASE)
    quads = els[list(els.keys())[0]]
    tags = s.get_nodal_tags(base_name=BASE)
    tag_of = {}
    for tname, ids in tags.items():
        for i in ids:
            tag_of.setdefault(int(i), []).append(tname)

    n = len(nodes)
    loop = sd.boundary_loop(quads, n)
    sides = sd.split_sides(loop, tag_of)
    bnd_uv = sd.parametrize(nodes, loop, sides, by="index")
    bnd_idx = np.array(sorted(bnd_uv), dtype=np.int64)
    bnd_mat = np.array([bnd_uv[i] for i in bnd_idx])
    tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    uv = sd.harmonic_map(nodes, tris, bnd_idx, bnd_mat)
    return uv, tris, quads, nodes, loop, sides, tag_of


def extract_miso(s, node_uv):
    """M_iso (Base_1_2, 244 узла) → профили по u: [корыто(v0), спинка(v1)].
    Узлы Base_1_2 совпадают со стенкой Base_2_2 (проверено: max dist 0.0);
    u-координата берётся из гармонической карты, профиль ресемплится на Nu точек."""
    s.set_default_base(BASE_WALL)
    n1 = np.asarray(s.get_nodes())
    miso = np.asarray(s.get_field("M_iso")).astype(np.float64)

    s.set_default_base(BASE)
    n2 = np.asarray(s.get_nodes())
    # стенка Base_2_2: узлы с тегами Intrado (спинка) / Extrado (корыто)
    tags = s.get_nodal_tags(base_name=BASE)
    wall_ids = np.concatenate([np.sort(np.asarray(tags[t], dtype=np.int64)) for t in ("Intrado", "Extrado")])
    is_intrado = np.concatenate([
        np.ones(len(tags["Intrado"]), dtype=bool), np.zeros(len(tags["Extrado"]), dtype=bool)])

    tree = cKDTree(n2[wall_ids])
    d, j = tree.query(n1)
    assert d.max() < 1e-9, f"узлы Base_1_2 не совпадают со стенкой (max d={d.max()})"

    u_wall = node_uv[wall_ids[j], 0]
    koryto = ~is_intrado[j]   # Extrado = корыто (v=0)
    spinka = is_intrado[j]    # Intrado = спинка (v=1)
    return miso, u_wall, koryto, spinka


def miso_profiles(miso, u_wall, koryto, spinka, Nu):
    """Профили M_iso по u ∈ [0,1]: [корыто, спинка] → [2, Nu]."""
    u_grid = np.linspace(0.0, 1.0, Nu)
    out = np.empty((2, Nu))
    for ch, mask in enumerate([koryto, spinka]):
        if mask.sum() == 0:
            out[ch] = np.nan
            continue
        u = u_wall[mask]
        val = miso[mask]
        order = np.argsort(u)
        out[ch] = np.interp(u_grid, u[order], val[order])
    return out


# ═══════════════ воркеры ═══════════════

_WORKER_DS = None


def _init_worker(plaid_dir):
    global _WORKER_DS
    from plaid.containers.dataset import Dataset
    _WORKER_DS = Dataset.load_from_dir(plaid_dir, verbose=False)


def _process_worker(args):
    idx, Nu, Nv, want_outputs = args
    return process_one(_WORKER_DS, idx, Nu, Nv, want_outputs)


def process_one(ds, idx, Nu, Nv, want_outputs):
    """Возвращает dict с массивами квадрата для одного сэмпла."""
    t0 = time.time()
    s = ds[idx]
    uv, tris, quads, nodes, loop, sides, tag_of = sample_to_square(s)

    # поля для интерполяции: координаты + sdf (+ 6 выходов если есть)
    field_list = [nodes[:, 0], nodes[:, 1], s.get_field("sdf", base_name=BASE)]
    if want_outputs:
        for fn in FIELD_NAMES:
            field_list.append(s.get_field(fn, base_name=BASE))

    ij_to_node, _ = get_ij(quads, nodes, loop, sides)

    U, V = np.meshgrid(np.linspace(0, 1, Nu), np.linspace(0, 1, Nv), indexing="ij")
    res = biquad_resample(uv, field_list, ij_to_node, U, V)

    # физические зажимы: sdf >= 0, nut >= 0
    res[2] = np.maximum(res[2], 0.0)              # sdf
    if want_outputs:
        res[3 + FIELD_NAMES.index("nut")] = np.maximum(res[3 + FIELD_NAMES.index("nut")], 0.0)  # nut

    scal_in = np.array([s.get_scalar("angle_in"), s.get_scalar("mach_out")], dtype=np.float32)
    scal_out = (np.array([s.get_scalar(n) for n in SCALAR_OUT], dtype=np.float32)
                if want_outputs else None)

    # M_iso: 1D-профили вдоль рёбер квадрата (только если есть ответы)
    miso = None
    if want_outputs:
        miso_raw, u_wall, koryto, spinka = extract_miso(s, uv)
        miso = miso_profiles(miso_raw, u_wall, koryto, spinka, Nu).astype(np.float32)

    dt = time.time() - t0
    print(f"  sample {idx}: {dt:.1f}s  (mach max={res[3].max() if want_outputs else float('nan'):.3f})")
    return dict(
        idx=idx,
        node_uv=uv.astype(np.float32),            # [n_nodes, 2] gather-карта
        inputs=res[:3].astype(np.float32),        # [3, Nu, Nv]
        outputs=res[3:].astype(np.float32) if want_outputs else None,  # [6, Nu, Nv]
        miso=miso,                                # [2, Nu] или None
        scal_in=scal_in,
        scal_out=scal_out,
    )


# ═══════════════ main ═══════════════

def main():
    ap = argparse.ArgumentParser(description="VKI-LS59 → квадрат (биквадратика P2) → DNO_dataset")
    ap.add_argument("--plaid_dir", type=Path, default=Path(DEFAULT_PLAID_DIR))
    ap.add_argument("--grid", nargs=2, type=int, default=[128, 128], metavar=("Nu", "Nv"))
    ap.add_argument("--jobs", type=int, default=1,
                    help="воркеры; по умолчанию 1 (последовательно) — так не падает по памяти")
    ap.add_argument("--ids", nargs="+", type=int, default=None,
                    help="только указанные id (отладка, без сохранения)")
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    args = ap.parse_args()
    Nu, Nv = args.grid

    from plaid.containers.dataset import Dataset
    ds = Dataset.load_from_dir(args.plaid_dir, verbose=False)
    split = json.load(open(args.plaid_dir / "split.json"))
    train_ids = list(split["train"])
    test_ids = list(split["test"])
    train_500_ids = list(split["train_500"])
    val_ids = [i for i in train_ids if i not in set(train_500_ids)]

    if args.ids is not None:
        ids = args.ids
        train_ids = [i for i in ids if i in set(train_ids)]
        test_ids = [i for i in ids if i in set(test_ids)]
        if not train_ids and not test_ids:
            raise SystemExit("--ids не пересекаются со сплитами")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"plaid: {args.plaid_dir}")
    print(f"сетка {Nu}x{Nv}, train={len(train_ids)}, test={len(test_ids)}, jobs={args.jobs}")
    print(f"выход: {args.out}")

    def run(ids_, want_outputs):
        if not ids_:
            return []
        if args.jobs > 1 and len(ids_) > 1:
            # Dataset грузится ОДИН раз на воркер (initializer), а не пиклится на задачу
            with ProcessPoolExecutor(max_workers=args.jobs,
                                     initializer=_init_worker, initargs=(str(args.plaid_dir),)) as ex:
                return list(ex.map(_process_worker,
                                   [(i, Nu, Nv, want_outputs) for i in ids_]))
        return [process_one(ds, i, Nu, Nv, want_outputs) for i in ids_]

    t_start = time.time()
    tr = run(train_ids, True)
    te = run(test_ids, False)

    if args.ids is not None:
        print("\n=== отладка: сохранение отключено ===")
        return

    save_dataset(tr, te, ds, train_ids, test_ids, train_500_ids, val_ids, Nu, Nv, args.out)
    print(f"\nготово за {time.time() - t_start:.0f}s → {args.out}")


# ═══════════════ сборка и сохранение ═══════════════

def save_dataset(tr, te, ds, train_ids, test_ids, train_500_ids, val_ids, Nu, Nv, out):
    """Собирает результаты process_one в DNO_dataset (npy + csv + meta)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def stack(arrs, key):
        return np.stack([a[key] for a in arrs]) if arrs and arrs[0][key] is not None else None

    train_inputs = stack(tr, "inputs")            # [Ntr, 3, Nu, Nv]
    train_outputs = stack(tr, "outputs")
    train_scalars = stack(tr, "scal_in")
    train_out_scalars = stack(tr, "scal_out")
    train_miso = stack(tr, "miso")
    train_node_uv = stack(tr, "node_uv")
    test_inputs = stack(te, "inputs")
    test_scalars = stack(te, "scal_in")
    test_node_uv = stack(te, "node_uv")

    # [N, C, Nu, Nv] → [N, Nu, Nv, C] (channel-last для моделей)
    def to_channel_last(a):
        return np.transpose(a, (0, 2, 3, 1)) if a is not None else None

    train_inputs_cl = to_channel_last(train_inputs)
    train_outputs_cl = to_channel_last(train_outputs)
    test_inputs_cl = to_channel_last(test_inputs)

    np.save(out / "train_inputs.npy", train_inputs_cl)
    np.save(out / "train_outputs.npy", train_outputs_cl)
    np.save(out / "train_scalars.npy", train_scalars)
    np.save(out / "train_out_scalars.npy", train_out_scalars)
    np.save(out / "train_miso.npy", train_miso)
    np.save(out / "train_node_uv.npy", train_node_uv)
    np.save(out / "test_inputs.npy", test_inputs_cl)
    np.save(out / "test_scalars.npy", test_scalars)
    np.save(out / "test_node_uv.npy", test_node_uv)

    # константы (топология, одинаковая для всех сэмплов)
    s0 = ds[train_ids[0]]
    _, _, quads0, nodes0, loop0, sides0, _ = sample_to_square(s0)
    ij_to_node, _ = get_ij(quads0, nodes0, loop0, sides0)
    np.save(out / "ij_to_node.npy", ij_to_node)
    tags0 = s0.get_nodal_tags(base_name=BASE)
    wall_ids = np.concatenate([np.sort(np.asarray(tags0[t], dtype=np.int64))
                               for t in ("Intrado", "Extrado")])
    np.save(out / "wall_node_ids.npy", wall_ids)

    # ── CSV-экспорт (формат оригинального DNO-репо: строка = сэмпл, Nu*Nv значений) ──
    npts = Nu * Nv
    flat = lambda a: a.reshape(len(a), npts) if a is not None else None   # C-order: v быстрее

    def save_csv(name, a):
        if a is not None:
            np.savetxt(out / name, flat(a), delimiter=",")

    save_csv("train_x_data.csv", train_inputs_cl[..., 0])
    save_csv("train_y_data.csv", train_inputs_cl[..., 1])
    save_csv("train_C.csv", train_inputs_cl[..., 2])                  # sdf
    save_csv("test_x_data.csv", test_inputs_cl[..., 0])
    save_csv("test_y_data.csv", test_inputs_cl[..., 1])
    save_csv("test_C.csv", test_inputs_cl[..., 2])
    for ch, name in enumerate(FIELD_NAMES):
        save_csv(f"train_U_{name}.csv", train_outputs_cl[..., ch])
    # маска границы: рамка 1, внутри 0 (как в train.py оригинального DNO)
    b = np.zeros((Nu - 2, Nv - 2), dtype=np.float64)
    b = np.pad(b, pad_width=1, mode="constant", constant_values=1)
    np.savetxt(out / "b.csv", b.reshape(1, npts), delimiter=",")

    # ── статистики train (для будущей нормировки) ──
    def stats(a):
        return dict(min=float(a.min()), max=float(a.max()),
                    mean=float(a.mean()), std=float(a.std()))

    meta = dict(
        grid=[Nu, Nv],
        input_channels=IN_CHANNELS,
        output_channels=FIELD_NAMES,
        miso_channels=MISO_CHANNELS,
        input_scalars=SCALAR_IN,
        output_scalars=SCALAR_OUT,
        interp="biquadratic (i,j) 3x3 Lagrange, Newton inversion (P2)",
        clamps=["sdf >= 0", "nut >= 0"],
        n_mesh_nodes=int(ij_to_node.shape[0] * ij_to_node.shape[1]),
        train_ids=train_ids,
        test_ids=test_ids,
        train_500_ids=train_500_ids,
        val_ids=val_ids,
        csv_export=dict(
            format="original DNO repo: one row per sample, Nu*Nv values (C-order, v fastest)",
            x_data="physical x of grid points", y_data="physical y of grid points",
            C="sdf (input field)", U_per_channel=[f"train_U_{n}.csv" for n in FIELD_NAMES],
            b="boundary mask (border=1, interior=0), same for all samples",
            note="test has no outputs (U absent), as in the HF dataset",
        ),
        node_uv=dict(files=["train_node_uv.npy", "test_node_uv.npy"],
                     shape="[N, n_mesh_nodes, 2] (u, v) — gather-карта на меш"),
        train_inputs_stats={k: stats(train_inputs_cl[..., i]) for i, k in enumerate(IN_CHANNELS)},
        train_outputs_stats={k: stats(train_outputs_cl[..., i]) for i, k in enumerate(FIELD_NAMES)},
        train_miso_stats={k: stats(train_miso[:, i]) for i, k in enumerate(MISO_CHANNELS)},
        train_scalars_stats={k: stats(train_scalars[:, i]) for i, k in enumerate(SCALAR_IN)},
        train_out_scalars_stats={k: stats(train_out_scalars[:, i]) for i, k in enumerate(SCALAR_OUT)},
    )
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"сохранено → {out}")
    for name, a in [("train_inputs", train_inputs), ("train_outputs", train_outputs),
                    ("train_scalars", train_scalars), ("train_out_scalars", train_out_scalars),
                    ("train_miso", train_miso), ("train_node_uv", train_node_uv),
                    ("test_inputs", test_inputs), ("test_scalars", test_scalars),
                    ("test_node_uv", test_node_uv)]:
        if a is not None:
            print(f"  {name}: {a.shape}  {a.dtype}")


if __name__ == "__main__":
    main()
