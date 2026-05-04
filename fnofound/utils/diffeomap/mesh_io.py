"""
Mesh I/O — load meshes from various formats (GMSH .msh, Wavefront .obj).

Supported formats:
  - GMSH 4.1 (.msh) via gmsh API
  - Wavefront OBJ (.obj) — triangular meshes only
"""

import warnings
import numpy as np


def read_msh(filename: str):
    """
    Read a .msh file (format 4.1) via the gmsh API.

    Parameters
    ----------
    filename : str
        Path to .msh file.

    Returns
    -------
    X : ndarray (N, 2)
        Node coordinates (x, y).
    F : ndarray (M, 3)
        Triangle connectivity (vertex indices).
    boundaries : dict
        Dictionary {name: [node_indices]}. Standard names:
        'inlet', 'outlet', 'bottom', 'top', 'outer', 'inner'
        mapped from physical group tags (1..6).
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.open(filename)

        # --- 1. Nodes ---
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        X = np.array(node_coords).reshape(-1, 3)[:, :2]
        tag2idx = {tag: i for i, tag in enumerate(node_tags)}

        # --- 2. Triangles (element type 2) ---
        elem_tags, elem_nodes = gmsh.model.mesh.getElementsByType(2)
        if len(elem_nodes) > 0:
            F_tags = np.array(elem_nodes).reshape(-1, 3)
            vec_map = np.vectorize(tag2idx.get)
            F = vec_map(F_tags).astype(np.int64)
        else:
            F = np.empty((0, 3), dtype=np.int64)
            warnings.warn(f"No triangles found in {filename}")

        # --- 3. Boundaries (Physical Groups, dim=1) ---
        boundaries = {}
        # Canonical group tag mapping. Override via known_groups if needed.
        known_groups = {
            1: 'inlet',
            2: 'outlet',
            3: 'bottom',
            4: 'top',
            5: 'outer',
            6: 'inner',
        }

        for tag_id, name in known_groups.items():
            try:
                bnodes_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, tag_id)
                if len(bnodes_tags) > 0:
                    boundaries[name] = [tag2idx[t] for t in bnodes_tags]
                else:
                    boundaries[name] = []
            except Exception:
                boundaries[name] = []

    except Exception as e:
        raise RuntimeError(f"Failed to read MSH file {filename}: {e}")
    finally:
        gmsh.finalize()

    return X, F, boundaries


def read_obj(filename: str):
    """
    Read a Wavefront .obj file (triangular mesh).

    Parameters
    ----------
    filename : str
        Path to .obj file.

    Returns
    -------
    X : ndarray (N, 3)
        Vertex coordinates (x, y, z).
    F : ndarray (M, 3)
        Triangle vertex indices (0-based).

    Note
    ----
    OBJ has no explicit boundary markup.
    Boundaries must be computed separately (see domains.py).
    """
    vertices = []
    faces = []

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                coords = [float(v) for v in parts[1:4]]
                vertices.append(coords)
            elif parts[0] == 'f':
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                # Triangulate polygons via fan triangulation
                if len(face) >= 3:
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])

    X = np.array(vertices, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)

    if F.shape[0] == 0:
        warnings.warn(f"No triangular faces found in {filename}")

    return X, F, {}


def read_mesh(filename: str):
    """
    Universal mesh loader — dispatches to read_msh or read_obj by extension.

    Parameters
    ----------
    filename : str
        Path to .msh or .obj file.

    Returns
    -------
    X, F, boundaries — see read_msh / read_obj.
    """
    if filename.endswith('.msh'):
        return read_msh(filename)
    elif filename.endswith('.obj'):
        return read_obj(filename)
    else:
        raise ValueError(f"Unsupported mesh format: {filename}. Use .msh or .obj.")
