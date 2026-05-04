"""
fnofound.utils.diffeomap — diffeomorphism mapping utilities.

Full pipeline:
    Mesh (GMSH/OBJ) → Cotangent Laplacian → Harmonic Mapping → Grid Interpolation → CSV

Supported geometries:
  - Step (step) — inlet/outlet/bottom/top BC
  - Square with hole (square_hole) — outer/inner BC
  - Simple polygon (polygon) — full boundary mapped to [0,1]²
"""

from .mesh_io import read_msh, read_obj, read_mesh
from .laplacian import (
    build_cotangent_laplacian,
    find_boundary_edges,
    build_boundary_loop,
)
from .harmonic import HarmonicMapper, solve_dirichlet
from .interpolator import GridInterpolator
from .pipeline import run_pipeline
from .domains import (
    step_boundary_map,
    square_hole_boundary_map,
    polygon_boundary_map,
    normalize_angle_order,
)

__all__ = [
    'read_msh', 'read_obj', 'read_mesh',
    'build_cotangent_laplacian', 'find_boundary_edges', 'build_boundary_loop',
    'HarmonicMapper', 'solve_dirichlet',
    'GridInterpolator',
    'run_pipeline',
    'step_boundary_map', 'square_hole_boundary_map', 'polygon_boundary_map',
    'normalize_angle_order',
]
