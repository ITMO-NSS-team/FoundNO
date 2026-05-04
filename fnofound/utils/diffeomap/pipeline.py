"""
Diffeomorphism Pipeline — complete end-to-end pipeline from mesh file
to diffeomorphism maps on a regular grid (X, Y → x_data.csv, y_data.csv).

Covers all three DNO cases:
  - Darcy (polygons/holes): mesh → harmonic map → CSV
  - Fluid (steps): mesh → harmonic map → CSV 
  - Reservoir (wells): mesh → harmonic map → CSV

Usage:
    from fnofound.utils.diffeomap.pipeline import run_pipeline

    # Step geometry (obstacle)
    run_pipeline("step.msh", output_dir="./maps", case="step")

    # Square with hole
    run_pipeline("square_hole.msh", output_dir="./maps", case="square_hole")
"""

import os
import json
import numpy as np

from .mesh_io import read_mesh
from .harmonic import HarmonicMapper
from .interpolator import GridInterpolator


def run_pipeline(mesh_file: str, output_dir: str = ".", resolution: int = 128,
                 case: str = "step"):
    """
    Full pipeline: mesh → Cotangent Laplacian → Harmonic Map → Grid Interpolation → CSV.

    Parameters
    ----------
    mesh_file : str
        Path to .msh or .obj file.
    output_dir : str
        Output directory for results.
    resolution : int
        Grid resolution (resolution × resolution).
    case : str
        Geometry type: "step", "square_hole", or "polygon".
        Determines boundary conditions for the harmonic map.

    Returns
    -------
    dict
        { 'grid_x': ndarray, 'grid_y': ndarray, 'mapper': HarmonicMapper }
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load mesh
    X, F, boundaries = read_mesh(mesh_file)
    print(f"[Pipeline] Mesh loaded: {X.shape[0]} nodes, {F.shape[0]} triangles")

    # 2. Harmonic map
    mapper = HarmonicMapper(X, F)

    if case == "step":
        if not all(k in boundaries for k in ('inlet', 'outlet', 'bottom', 'top')):
            raise ValueError(
                "MSH for 'step' must have physical groups: "
                "1=inlet, 2=outlet, 3=bottom, 4=top"
            )
            #inlet → ξ=0
            #outlet → ξ=1
            #bottom → η=0
            #top → η=1
        mapper.build_step_mapping(boundaries)

    elif case == "square_hole":
        if 'outer' not in boundaries or 'inner' not in boundaries:
            # Auto-detect boundaries
            from .laplacian import build_boundary_loop
            outer = build_boundary_loop(F)
            from .laplacian import find_boundary_edges
            be = find_boundary_edges(F)
            all_boundary = set()
            for u, v in be:
                all_boundary.add(u)
                all_boundary.add(v)
            outer_set = set(outer)
            inner = list(all_boundary - outer_set)
            boundaries = {'outer': outer.tolist(), 'inner': inner}
            print(f"[Pipeline] Auto-detected: outer={len(outer)}, inner={len(inner)}")

        mapper.build_square_with_hole_mapping(boundaries)

    elif case == "polygon":
        # Polygon without holes — map entire boundary to [0,1]×[0,1]
        from .laplacian import build_boundary_loop
        outer = build_boundary_loop(F)
        outer_coords = X[outer]
        center = outer_coords.mean(axis=0)
        angles = np.arctan2(outer_coords[:, 1] - center[1],
                            outer_coords[:, 0] - center[0])
        order = np.argsort(angles)
        nodes_sorted = [outer[i] for i in order]
        n = len(nodes_sorted)
        xi_vals = np.interp(np.linspace(0, n, n),
                            [0, n // 4, n // 2, 3 * n // 4, n],
                            [0, 1, 1, 0, 0])
        eta_vals = np.interp(np.linspace(0, n, n),
                             [0, n // 4, n // 2, 3 * n // 4, n],
                             [0, 0, 1, 1, 0])

        mapper.fit({
            'xi':  {'nodes': nodes_sorted, 'values': xi_vals.tolist()},
            'eta': {'nodes': nodes_sorted, 'values': eta_vals.tolist()},
        })

    else:
        raise ValueError(f"Unknown case: {case}. "
                         f"Use 'step', 'square_hole', or 'polygon'.")

    print(f"[Pipeline] Harmonic mapping done. "
          f"xi=[{mapper.Y[:, 0].min():.4f}, {mapper.Y[:, 0].max():.4f}], "
          f"eta=[{mapper.Y[:, 1].min():.4f}, {mapper.Y[:, 1].max():.4f}]")

    # 3. Interpolate onto regular grid
    interpolator = GridInterpolator(X, mapper.Y, F)
    grid_x, grid_y = interpolator.save_maps(output_dir, resolution)
    print(f"[Pipeline] Maps saved to {output_dir}/")

    # 4. Save metadata
    meta = {
        'mesh_file': os.path.abspath(mesh_file),
        'case': case,
        'resolution': resolution,
        'n_nodes': int(X.shape[0]),
        'n_triangles': int(F.shape[0]),
        'grid_x_shape': list(grid_x.shape),
        'grid_y_shape': list(grid_y.shape),
    }
    with open(os.path.join(output_dir, 'pipeline_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return {'grid_x': grid_x, 'grid_y': grid_y, 'mapper': mapper}
