"""
Harmonic mapping — map a physical domain to the unit square [0,1]×[0,1]
by solving the Laplace equation with Dirichlet boundary conditions.

Uses cotangent weights from laplacian.py for discretization.
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from .laplacian import build_cotangent_laplacian


def solve_dirichlet(L, fixed_nodes, fixed_values, penalty=1e15):
    """
    Solve the Dirichlet problem L·u = 0 with fixed nodes
    using the penalty method.

    Parameters
    ----------
    L : csr_matrix (N, N)
        Laplacian matrix.
    fixed_nodes : list of int
        Indices of fixed (boundary) nodes.
    fixed_values : list of float
        Values at fixed nodes.
    penalty : float
        Penalty coefficient.

    Returns
    -------
    u : ndarray (N,)
        Solution vector.
    """
    idx = np.asarray(fixed_nodes, dtype=int)
    val = np.asarray(fixed_values, dtype=float)

    A = L.copy()
    diag = A.diagonal().copy()
    diag[idx] += penalty
    A.setdiag(diag)

    b = np.zeros(L.shape[0])
    b[idx] = val * penalty

    return spsolve(A, b)


class HarmonicMapper:
    """
    Build a harmonic map from a physical domain to [0,1]×[0,1].

    Boundary conditions are specified via a boundary_map dict:
        boundary_map = {
            'xi':  {'nodes': [...], 'values': [...]},   # Xi (horizontal)
            'eta': {'nodes': [...], 'values': [...]},   # Eta (vertical)
        }

    Example for a step geometry (obstacle):
        xi_nodes  = inlet + outlet   (0 = inlet, 1 = outlet)
        eta_nodes = bottom + top     (0 = bottom, 1 = top)
    """

    def __init__(self, X: np.ndarray, F: np.ndarray):
        """
        Parameters
        ----------
        X : ndarray (N, 2)
            Node coordinates.
        F : ndarray (M, 3)
            Triangle connectivity.
        """
        self.X = X
        self.F = F
        self.n_nodes = X.shape[0]
        self.L = None
        self.Y = None  # (N, 2) logical coords (xi, eta)

    def fit(self, boundary_map: dict):
        """
        Compute the harmonic map.

        Parameters
        ----------
        boundary_map : dict
            {
                'xi':  {'nodes': [...], 'values': [...]},
                'eta': {'nodes': [...], 'values': [...]},
            }
        """
        self.L = build_cotangent_laplacian(self.X, self.F)

        xi_nodes = boundary_map['xi']['nodes']
        xi_vals = boundary_map['xi']['values']
        xi = solve_dirichlet(self.L, xi_nodes, xi_vals)

        eta_nodes = boundary_map['eta']['nodes']
        eta_vals = boundary_map['eta']['values']
        eta = solve_dirichlet(self.L, eta_nodes, eta_vals)

        self.Y = np.column_stack([xi, eta])
        return self.Y

    def build_step_mapping(self, boundaries: dict):
        """
        Convenience method for step (obstacle) geometries:
          Xi:  Inlet(0) ↔ Outlet(1)
          Eta: Bottom(0) ↔ Top(1)

        Parameters
        ----------
        boundaries : dict
            { 'inlet': [...], 'outlet': [...], 'bottom': [...], 'top': [...] }
        """
        bmap = {
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
        return self.fit(bmap)

    def build_square_with_hole_mapping(self, boundaries: dict):
        """
        Convenience method for square-with-hole geometries.

        Parameters
        ----------
        boundaries : dict
            { 'outer': [...], 'inner': [...] }
        """
        bmap = self._build_square_hole_bcs(boundaries)
        return self.fit(bmap)

    def _build_square_hole_bcs(self, boundaries: dict):
        """Build boundary_map for square with a circular hole."""
        outer_nodes = boundaries.get('outer', [])
        inner_nodes = boundaries.get('inner', [])

        outer_coords = self.X[outer_nodes]
        inner_coords = self.X[inner_nodes]

        center = np.mean(inner_coords, axis=0)
        angles_outer = np.arctan2(outer_coords[:, 1] - center[1],
                                  outer_coords[:, 0] - center[0])
        angles_inner = np.arctan2(inner_coords[:, 1] - center[1],
                                  inner_coords[:, 0] - center[0])

        outer_order = np.argsort(angles_outer)
        inner_order = np.argsort(angles_inner)

        n_outer = len(outer_nodes)

        # Map outer boundary to square corners via angular parameterization
        xi_outer = 0.5 + 0.5 * np.cos(np.pi - angles_outer[outer_order]) * \
                   np.maximum(np.abs(np.cos(angles_outer[outer_order])),
                              np.abs(np.sin(angles_outer[outer_order])))
        eta_outer = 0.5 + 0.5 * np.sin(np.pi - angles_outer[outer_order]) * \
                    np.maximum(np.abs(np.cos(angles_outer[outer_order])),
                               np.abs(np.sin(angles_outer[outer_order])))

        # Map inner boundary to a small circle around [0.5, 0.5]
        r_target = 0.1
        xi_inner = 0.5 + r_target * np.cos(angles_inner[inner_order])
        eta_inner = 0.5 + r_target * np.sin(angles_inner[inner_order])

        bmap = {
            'xi': {
                'nodes': [outer_nodes[i] for i in outer_order]
                         + [inner_nodes[i] for i in inner_order],
                'values': np.concatenate([xi_outer, xi_inner]).tolist(),
            },
            'eta': {
                'nodes': [outer_nodes[i] for i in outer_order]
                         + [inner_nodes[i] for i in inner_order],
                'values': np.concatenate([eta_outer, eta_inner]).tolist(),
            },
        }
        return bmap
