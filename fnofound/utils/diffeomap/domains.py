"""
Domains — standard boundary condition definitions for common geometries.
Used by HarmonicMapper to build the Dirichlet BC for harmonic mapping.
"""

import numpy as np


def normalize_angle_order(X, node_indices, center=None):
    """
    Sort nodes by polar angle around a center point.

    Parameters
    ----------
    X : ndarray (N, 2)
        All node coordinates.
    node_indices : list of int
        Node indices to sort.
    center : ndarray (2,) or None
        Rotation center (default: mean of coordinates).

    Returns
    -------
    sorted_indices : list of int
    angles : ndarray
    """
    coords = X[node_indices]
    if center is None:
        center = np.mean(coords, axis=0)
    angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
    order = np.argsort(angles)
    return [node_indices[i] for i in order], angles[order]


def step_boundary_map(boundaries: dict):
    """
    Build boundary_map for a step geometry (obstacle).

    Xi:  Inlet(0) ↔ Outlet(1)
    Eta: Bottom(0) ↔ Top(1)

    Parameters
    ----------
    boundaries : dict
        { 'inlet': [...], 'outlet': [...], 'bottom': [...], 'top': [...] }

    Returns
    -------
    boundary_map : dict for HarmonicMapper.fit()
    """
    return {
        'xi': {
            'nodes': boundaries['inlet'] + boundaries['outlet'],
            'values': [0.0] * len(boundaries['inlet'])
                      + [1.0] * len(boundaries['outlet']),
        },
        'eta': {
            'nodes': boundaries['bottom'] + boundaries['top'],
            'values': [0.0] * len(boundaries['bottom'])
                      + [1.0] * len(boundaries['top']),
        },
    }


def square_hole_boundary_map(X, boundaries, hole_radius_target=0.1,
                              hole_center_target=np.array([0.5, 0.5])):
    """
    Build boundary_map for a square with a circular hole.

    Outer boundary → square [0,1]×[0,1]
    Inner boundary → circle of radius hole_radius_target

    Parameters
    ----------
    X : ndarray (N, 2)
    boundaries : dict
        { 'outer': [...], 'inner': [...] }
    hole_radius_target : float
    hole_center_target : ndarray (2,)

    Returns
    -------
    boundary_map : dict
    """
    outer = boundaries['outer']
    inner = boundaries['inner']

    outer_sorted, outer_angles = normalize_angle_order(X, outer)
    inner_sorted, inner_angles = normalize_angle_order(X, inner)

    n_outer = len(outer_sorted)
    xi_outer = np.interp(np.linspace(0, n_outer, n_outer),
                         [0, n_outer // 4, n_outer // 2,
                          3 * n_outer // 4, n_outer],
                         [0, 1, 1, 0, 0])
    eta_outer = np.interp(np.linspace(0, n_outer, n_outer),
                          [0, n_outer // 4, n_outer // 2,
                           3 * n_outer // 4, n_outer],
                          [0, 0, 1, 1, 0])

    hc = hole_center_target
    hr = hole_radius_target
    xi_inner = hc[0] + hr * np.cos(inner_angles)
    eta_inner = hc[1] + hr * np.sin(inner_angles)

    return {
        'xi': {
            'nodes': outer_sorted + inner_sorted,
            'values': np.concatenate([xi_outer, xi_inner]).tolist(),
        },
        'eta': {
            'nodes': outer_sorted + inner_sorted,
            'values': np.concatenate([eta_outer, eta_inner]).tolist(),
        },
    }


def polygon_boundary_map(X, boundary_nodes):
    """
    Build boundary_map for a simple polygon (no holes).
    The entire boundary maps to [0,1]×[0,1] by angle.

    Parameters
    ----------
    X : ndarray (N, 2)
    boundary_nodes : list of int
        Ordered boundary node indices.

    Returns
    -------
    boundary_map : dict
    """
    sorted_nodes, angles = normalize_angle_order(X, boundary_nodes)
    n = len(sorted_nodes)
    xi_vals = np.interp(np.linspace(0, n, n),
                        [0, n // 4, n // 2, 3 * n // 4, n],
                        [0, 1, 1, 0, 0])
    eta_vals = np.interp(np.linspace(0, n, n),
                         [0, n // 4, n // 2, 3 * n // 4, n],
                         [0, 0, 1, 1, 0])

    return {
        'xi':  {'nodes': sorted_nodes, 'values': xi_vals.tolist()},
        'eta': {'nodes': sorted_nodes, 'values': eta_vals.tolist()},
    }
