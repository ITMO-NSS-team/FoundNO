"""
Cotangent Laplacian — discrete Laplace-Beltrami operator on a triangular mesh
using cotangent weights.

Used for harmonic mapping of a physical domain into a logical (universal) square.
"""

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csr import csr_matrix


def _triangle_areas(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Compute triangle areas via cross product."""
    v0 = X[F[:, 0]]
    v1 = X[F[:, 1]]
    v2 = X[F[:, 2]]
    return 0.5 * np.abs((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
                        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))


def build_cotangent_laplacian(X: np.ndarray, F: np.ndarray) -> csr_matrix:
    """
    Build the cotangent-weight Laplacian matrix for a 2D triangular mesh.

    L_{ij} = -0.5 * (cot α_{ij} + cot β_{ij})  for i ≠ j
    L_{ii} = -∑_{j≠i} L_{ij}

    Parameters
    ----------
    X : ndarray (N, 2)
        Node coordinates.
    F : ndarray (M, 3)
        Triangle connectivity.

    Returns
    -------
    L : csr_matrix (N, N)
        Laplacian matrix.
    """
    n_nodes = X.shape[0]
    n_tri = F.shape[0]

    v0 = X[F[:, 0]]
    v1 = X[F[:, 1]]
    v2 = X[F[:, 2]]

    # Edge vectors
    u01 = v1 - v0
    u12 = v2 - v1
    u20 = v0 - v2

    def cotan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """cot(θ) = (u·v) / |u×v| for a pair of vectors sharing a vertex."""
        dot = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]
        cross = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        cross = np.maximum(cross, 1e-12)
        return dot / cross

    # Cotangents at vertices 0, 1, 2
    cot0 = cotan(-u20, u01)   # angle at v0
    cot1 = cotan(-u01, u12)   # angle at v1
    cot2 = cotan(-u12, u20)   # angle at v2

    rows, cols, data = [], [], []

    # Opposite edges:
    #   edge (1,2) opposite vertex 0 → cot0
    #   edge (2,0) opposite vertex 1 → cot1
    #   edge (0,1) opposite vertex 2 → cot2
    for k, cot in enumerate([cot0, cot1, cot2]):
        i = F[:, (k + 1) % 3]
        j = F[:, (k + 2) % 3]
        w = cot * 0.5
        rows.extend(i)
        cols.extend(j)
        data.extend(w)
        rows.extend(j)
        cols.extend(i)
        data.extend(w)

    W = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    degrees = np.array(W.sum(axis=1)).flatten()
    D = diags(degrees)
    L = (D - W).tocsr()

    return L


def find_boundary_edges(F: np.ndarray) -> list:
    """
    Find edges that belong to exactly one triangle (boundary edges).

    Parameters
    ----------
    F : ndarray (M, 3)
        Triangle connectivity.

    Returns
    -------
    boundary_edges : list of (int, int)
        Sorted boundary edge pairs.
    """
    edges = {}
    for face in F:
        for i in range(3):
            edge = tuple(sorted([int(face[i]), int(face[(i + 1) % 3])]))
            edges[edge] = edges.get(edge, 0) + 1
    return [edge for edge, count in edges.items() if count == 1]


def build_boundary_loop(F: np.ndarray, start_node: int = 0) -> np.ndarray:
    """
    Reconstruct an ordered boundary loop from triangulation.

    Parameters
    ----------
    F : ndarray (M, 3)
        Triangle connectivity.
    start_node : int
        Starting node index.

    Returns
    -------
    boundary : ndarray
        Node indices in boundary traversal order.
    """
    be = find_boundary_edges(F)

    graph = {}
    for u, v in be:
        graph.setdefault(u, []).append(v)
        graph.setdefault(v, []).append(u)

    if not graph:
        return np.array([], dtype=np.int64)

    boundary = []
    current = start_node
    visited = set()
    while current not in visited:
        visited.add(current)
        boundary.append(current)
        neighbors = [n for n in graph[current] if n not in visited]
        if not neighbors:
            break
        current = neighbors[0]

    return np.array(boundary, dtype=np.int64)
