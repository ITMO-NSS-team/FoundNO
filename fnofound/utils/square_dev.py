"""
square_dev.py - structured C-grid mesh -> unit square mapping (VKI-LS59).

Ported from the VKI-LS59 workspace (square_dev.py), the algorithm behind the
DNO-style preprocessing:

    1) boundary loop (quad edges)                      -> boundary_loop
    2) loop split into 4 sides by nodal tags           -> split_sides
       (left=Inflow, right=Outflow, top=Intrado, bottom=Extrado)
    3) side parametrization                            -> parametrize
    4) quad triangulation + cotangent Laplacian        -> build_cotangent_laplacian
    5) Laplace solve (harmonic map)                    -> harmonic_map
    6) validation (no flipped triangles)               -> check_flips
    7) structured (i,j) layout of the mesh             -> structured_ij

All functions are pure numpy/scipy - the `plaid` package is needed only by
the CALLERS that read the raw dataset (fnofound.data.data.datasets.vki_datasets
RawGridDataset, experiments.vki.data_prep.prep_square).
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from collections import defaultdict


# ═══════════════ utils ═══════════════

def build_cotangent_laplacian(X, faces, clamp_negative=True):
    n_nodes = X.shape[0]
    v0 = X[faces[:, 0]]; v1 = X[faces[:, 1]]; v2 = X[faces[:, 2]]
    def cotan(u, v):
        dot = np.sum(u * v, axis=1)
        cross = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        cross = np.maximum(cross, 1e-12)
        return dot / cross
    cot0 = cotan(v1 - v0, v2 - v0)
    cot1 = cotan(v2 - v1, v0 - v1)
    cot2 = cotan(v0 - v2, v1 - v2)
    if clamp_negative:
        cot0 = np.maximum(cot0, 0); cot1 = np.maximum(cot1, 0); cot2 = np.maximum(cot2, 0)
    rows, cols, data = [], [], []
    for k, cot in enumerate([cot0, cot1, cot2]):
        i = faces[:, (k + 1) % 3]; j = faces[:, (k + 2) % 3]
        w = cot * 0.5
        rows.extend(i); cols.extend(j); data.extend(w)
        rows.extend(j); cols.extend(i); data.extend(w)
    W = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    degrees = np.array(W.sum(axis=1)).flatten()
    L = (diags(degrees) - W).tocsr()
    return L


def boundary_loop(quads, n_nodes):
    """Граничные рёбра -> упорядоченная петля (list узлов)."""
    ecnt = defaultdict(int)
    for q in quads:
        for a, b in [(q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])]:
            e = (min(a, b), max(a, b))
            ecnt[e] += 1
    bnd = [e for e, c in ecnt.items() if c == 1]
    adj = defaultdict(list)
    for a, b in bnd:
        adj[a].append(b); adj[b].append(a)
    start = bnd[0][0]
    loop = [start]; prev = -1; cur = start
    while True:
        nxts = [x for x in adj[cur] if x != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == start:
            break
        loop.append(nxt); prev, cur = cur, nxt
    return np.array(loop, dtype=np.int64)


def split_sides(loop, tag_of):
    """Петля -> 4 стороны в порядке квадрата: left, top, right, bottom.
    Возвращает списки индексов узлов loop для каждой стороны (в порядке обхода стороны)
    и словарь side -> ('u'|'v', направление), чтобы параметризовать.
    """
    def cls(i):
        t = tag_of.get(int(i), [])
        if 'Inflow' in t: return 'left'
        if 'Outflow' in t: return 'right'
        if 'Intrado' in t: return 'intrado'
        if 'Extrado' in t: return 'extrado'
        if 'Periodic_1' in t: return 'p1'
        if 'Periodic_2' in t: return 'p2'
        return 'none'

    # границы классов вдоль петли
    classes = [cls(i) for i in loop]
    runs = []  # (start, end_exclusive, class)
    s = 0
    for k in range(1, len(loop) + 1):
        if k == len(loop) or classes[k] != classes[s]:
            runs.append((s, k, classes[s]))
            s = k
    L = len(loop)
    def idx_wrap(k):
        return k % L
    # найти left/right/intrado run
    def find_run(c):
        for r in runs:
            if r[2] == c:
                return r
        raise ValueError(f"run {c} not found")
    lr = find_run('left'); rr = find_run('right')
    ls, le = lr[0], lr[1] - 1      # start/end индексы (end включительно)
    rs, re = rr[0], rr[1] - 1

    # set1: runs между left-run и right-run (вперёд), set2: между right и left
    def runs_between(a_end, b_start):
        out = []
        k = (a_end + 1) % L
        while k != b_start:
            for r in runs:
                if r[0] == k:
                    out.append(r)
                    k = (r[1]) % L
                    break
        return out
    set1 = runs_between(le, rs)  # после left до right
    set2 = runs_between(re, ls)  # после right до left (wrap)
    set1c = [r[2] for r in set1]; set2c = [r[2] for r in set2]
    assert 'intrado' in set1c or 'intrado' in set2c
    if 'intrado' in set1c:
        top, bottom, top_set1 = set1, set2, True
    else:
        top, bottom, top_set1 = set2, set1, False

    def loop_slice(a, b):
        """узлы loop[a..b] включительно, с wrap, в порядке возрастания индекса"""
        if b >= a:
            return list(range(a, b + 1))
        return list(range(a, L)) + list(range(0, b + 1))

    # Стороны в порядке квадрата, каждая от своего «нулевого» угла:
    #   left   : от нижнего угла (v=0) к верхнему (v=1),         val = s
    #   top    : от Inflow-угла (u=0) к Outflow-углу (u=1),      val = s
    #   right  : от верхнего угла (v=1) к нижнему (v=0),         val = 1 - s
    #   bottom : от Inflow-угла (u=0) к Outflow-углу (u=1),      val = s
    if top_set1:   # top = set1: идёт le -> rs; bottom = set2: re -> ls (wrap)
        left_idx   = loop_slice(ls, le)                        # ls=низ, le=верх
        top_idx    = loop_slice(le, rs)                        # le=Inflow, rs=Outflow
        right_idx  = loop_slice(rs, re)                        # rs=верх, re=низ
        bottom_idx = loop_slice(re, ls)[::-1]                  # ls=Inflow, re=Outflow
    else:          # top = set2: re -> ls (wrap); bottom = set1: le -> rs
        left_idx   = loop_slice(ls, le)[::-1]                  # le=низ, ls=верх
        top_idx    = loop_slice(re, ls)[::-1]                  # ls=Inflow, re=Outflow
        right_idx  = loop_slice(rs, re)[::-1]                  # re=верх, rs=низ
        bottom_idx = loop_slice(le, rs)                        # le=Inflow, rs=Outflow

    sides = {
        'left':   (left_idx,  'v',  1),
        'top':    (top_idx,   'u',  1),
        'right':  (right_idx, 'v', -1),
        'bottom': (bottom_idx,'u',  1),
    }
    return sides


def parametrize(nodes, loop, sides, by='index'):
    """(u,v) для всех граничных узлов.
    by='index'  : равномерно по номеру узла (классический Tutte; для структурированной
                  сетки совпадает с (i,j)-индексами на границе)
    by='arclen' : по длине дуги
    """
    bnd_uv = {}
    for name, (idx, axis, sgn) in sides.items():
        ids = loop[idx]
        if by == 'arclen':
            pts = nodes[ids]
            d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            s = d / d[-1]
        else:
            s = np.linspace(0.0, 1.0, len(ids))
        if sgn == -1:
            s = 1 - s
        uv = np.zeros((len(ids), 2))
        uv[:, 0 if axis == 'u' else 1] = s
        if axis == 'u':
            uv[:, 1] = 1.0 if name == 'top' else 0.0
        else:
            uv[:, 0] = 1.0 if name == 'right' else 0.0
        for nd, uvv in zip(ids, uv):
            bnd_uv[int(nd)] = uvv
    return bnd_uv


def harmonic_map(nodes, tris, bnd_idx, bnd_uv, clamp=True):
    n = len(nodes)
    L = build_cotangent_laplacian(nodes, tris, clamp_negative=clamp)
    interior = np.setdiff1d(np.arange(n), bnd_idx)
    Lii = L[interior][:, interior].tocsr()
    Lib = L[interior][:, bnd_idx].tocsr()
    uv = np.zeros((n, 2))
    uv[bnd_idx] = bnd_uv
    for c in range(2):
        rhs = -Lib @ bnd_uv[:, c]
        uv[interior, c] = spsolve(Lii, rhs)
    return uv


def check_flips(nodes_xy, nodes_uv, tris):
    """Сравниваем знаки ориентированной площади в (x,y) и (u,v):
    расхождение знака = перевёрнутый треугольник.
    Возвращает area2_uv (для статистики) и маску флипов."""
    def sarea(P):
        v0 = P[tris[:, 0]]; v1 = P[tris[:, 1]]; v2 = P[tris[:, 2]]
        return (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    a_xy = sarea(nodes_xy)
    a_uv = sarea(nodes_uv)
    flips = np.sign(a_xy) != np.sign(a_uv)
    return a_uv, flips


def structured_ij(nodes, quads, sides, loop):
    """(i,j) структурированной сетки: классификация рёбер на I/J-семейства + BFS."""
    n = len(nodes)
    # рёбра квадов
    edge_id = {}
    edge_list = []
    for q in quads:
        for a, b in [(q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])]:
            e = (min(a, b), max(a, b))
            if e not in edge_id:
                edge_id[e] = len(edge_list)
                edge_list.append(e)
    E = len(edge_list)
    fam = np.zeros(E, dtype=int) - 1  # 0=I, 1=J
    # граничные рёбра: top/bottom -> I, left/right -> J
    bnd_set = set()
    for name, (idx, axis, sgn) in sides.items():
        ids = loop[idx]
        for k in range(len(ids) - 1):
            a, b = ids[k], ids[k + 1]
            bnd_set.add((min(a, b), max(a, b)))
    for e in bnd_set:
        fam[edge_id[e]] = 1 if e in {tuple(sorted([loop[sides['left'][0][0]], loop[sides['left'][0][0] + 1]]))} else 0
    # проще: помечаем по сторонам
    for name, (idx, axis, sgn) in sides.items():
        f = 0 if axis == 'u' else 1
        ids = loop[idx]
        for k in range(len(ids) - 1):
            e = (min(ids[k], ids[k + 1]), max(ids[k], ids[k + 1]))
            fam[edge_id[e]] = f
    # распространение через квады: противоположные рёбра одного семейства
    changed = True
    while changed:
        changed = False
        for q in quads:
            es = []
            for a, b in [(q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])]:
                es.append(edge_id[(min(a, b), max(a, b))])
            pairs = [(es[0], es[2]), (es[1], es[3])]
            for x, y in pairs:
                if fam[x] >= 0 and fam[y] < 0:
                    fam[y] = fam[x]; changed = True
                elif fam[y] >= 0 and fam[x] < 0:
                    fam[x] = fam[y]; changed = True
    if (fam < 0).any():
        print("  !! не все рёбра классифицированы:", (fam < 0).sum())
    # графы семейств
    adjI = defaultdict(list); adjJ = defaultdict(list)
    for e, f in zip(edge_list, fam):
        if f == 0:
            adjI[e[0]].append(e[1]); adjI[e[1]].append(e[0])
        elif f == 1:
            adjJ[e[0]].append(e[1]); adjJ[e[1]].append(e[0])
    # i: BFS по I-рёбрам от левой колонки; j: BFS по J-рёбрам от нижней стороны
    left_ids = [int(x) for x in loop[sides['left'][0]]]
    bottom_ids = [int(x) for x in loop[sides['bottom'][0]]]
    i_of = {x: 0 for x in left_ids}
    stack = list(left_ids)
    while stack:
        u = stack.pop()
        for w in adjI[u]:
            if w not in i_of:
                i_of[w] = i_of[u] + 1
                stack.append(w)
    j_of = {x: 0 for x in bottom_ids}
    stack = list(bottom_ids)
    while stack:
        u = stack.pop()
        for w in adjJ[u]:
            if w not in j_of:
                j_of[w] = j_of[u] + 1
                stack.append(w)
    ij = np.full((n, 2), -1, dtype=int)
    for k in range(n):
        if k in i_of:
            ij[k, 0] = i_of[k]
        if k in j_of:
            ij[k, 1] = j_of[k]
    return ij, fam
