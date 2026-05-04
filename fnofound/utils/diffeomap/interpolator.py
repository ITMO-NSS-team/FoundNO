"""
Grid Interpolator — rasterize a triangular mesh onto a regular N×N grid
in the logical [0,1]×[0,1] space.

Uses barycentric interpolation to transfer physical field values
from mesh vertices to regular grid pixels.
"""

import os
import csv
import numpy as np


class GridInterpolator:
    """
    Interpolate physical coordinates/fields from an irregular triangular mesh
    onto a regular N×N grid in the universal space.

    Parameters
    ----------
    phys_coords : ndarray (N, 2)
        Physical node coordinates (X, Y).
    logical_coords : ndarray (N, 2)
        Logical node coordinates (xi, eta) from HarmonicMapper.
    faces : ndarray (M, 3)
        Triangle connectivity.
    """

    def __init__(self, phys_coords, logical_coords, faces):
        self.phys_coords = phys_coords
        self.logical_coords = logical_coords
        self.faces = faces

    def interpolate_field(self, field_values, resolution=128):
        """
        Interpolate a scalar field from mesh vertices to a regular grid.

        Parameters
        ----------
        field_values : ndarray (N,)
            Node values.
        resolution : int
            Grid resolution (resolution × resolution).

        Returns
        -------
        grid : ndarray (resolution, resolution)
            Interpolated field. NaN outside the domain.
        """
        return self._compute_dual_interp(
            self.faces.T if self.faces.shape[0] == 3 else self.faces,
            self.logical_coords.T if self.logical_coords.shape[0] == 2
            else self.logical_coords,
            field_values,
            np.zeros_like(field_values),
            resolution,
            single=True,
        )

    def interpolate_coords(self, resolution=128):
        """
        Interpolate X and Y coordinates onto a regular grid.

        Parameters
        ----------
        resolution : int

        Returns
        -------
        grid_x, grid_y : ndarray (resolution, resolution)
        """
        faces = self.faces.T if self.faces.shape[0] != 3 and \
            self.faces.shape[1] == 3 else self.faces
        if faces.shape[0] != 3:
            faces = faces.T

        uv = self.logical_coords.T if self.logical_coords.shape[0] != 2 \
            else self.logical_coords
        phys_x = self.phys_coords[:, 0]
        phys_y = self.phys_coords[:, 1]

        Mx, My = self._compute_dual_interp(faces, uv, phys_x, phys_y,
                                           resolution, single=False)
        return Mx, My

    def _compute_dual_interp(self, face, vertex, v_x, v_y, n, single=False):
        """
        Rasterize triangles onto an n×n grid using barycentric interpolation.

        Parameters
        ----------
        face : ndarray (3, M)
        vertex : ndarray (2, N)
        v_x, v_y : ndarray (N,)
        n : int
        single : bool — if True, only v_x is used

        Returns
        -------
        grid_x (n, n) or (grid_x, grid_y)
        """
        if vertex.shape[0] != 2:
            vertex = vertex.T
        if face.shape[0] != 3:
            face = face.T

        vertex = np.clip(vertex, 0.0, 1.0)
        nface = face.shape[1]
        scale = n - 1

        Mx = np.zeros(n * n)
        if not single:
            My = np.zeros(n * n)
        Mnb = np.zeros(n * n)

        for i in range(nface):
            T = face[:, i]
            P = vertex[:, T]
            Vx = v_x[T]

            # Bounding box in pixel indices
            min_u, max_u = P[0].min(), P[0].max()
            min_v, max_v = P[1].min(), P[1].max()

            i0 = max(0, int(np.floor(min_u * scale)))
            i1 = min(n - 1, int(np.ceil(max_u * scale)))
            j0 = max(0, int(np.floor(min_v * scale)))
            j1 = min(n - 1, int(np.ceil(max_v * scale)))

            if i0 > i1 or j0 > j1:
                continue

            ix = np.arange(i0, i1 + 1)
            iy = np.arange(j0, j1 + 1)
            Iy_grid, Ix_grid = np.meshgrid(iy, ix, indexing='ij')

            pos_u = Ix_grid.flatten() / scale
            pos_v = Iy_grid.flatten() / scale
            pos = np.vstack([pos_u, pos_v])
            p_count = pos.shape[1]

            # Barycentric coordinates
            a = np.vstack([[1, 1, 1], P])
            try:
                inva = np.linalg.pinv(a)
            except np.linalg.LinAlgError:
                continue

            b = np.vstack([np.ones(p_count), pos])
            c = inva @ b

            eps = -1e-9
            inside = np.where((c[0] >= eps) & (c[1] >= eps) & (c[2] >= eps))[0]
            if len(inside) == 0:
                continue

            c_final = c[:, inside]
            final_ix = Ix_grid.flatten()[inside]
            final_iy = Iy_grid.flatten()[inside]
            J = final_iy * n + final_ix  # row-major

            vals_x = Vx[0] * c_final[0] + Vx[1] * c_final[1] + Vx[2] * c_final[2]
            np.add.at(Mx, J, vals_x)
            np.add.at(Mnb, J, 1)

            if not single:
                Vy = v_y[T]
                vals_y = Vy[0] * c_final[0] + Vy[1] * c_final[1] + Vy[2] * c_final[2]
                np.add.at(My, J, vals_y)

        Mx = Mx.reshape(n, n).astype(np.float64)
        Mnb = Mnb.reshape(n, n)
        mask = Mnb > 0
        Mx[mask] /= Mnb[mask]
        Mx[~mask] = np.nan

        if single:
            return Mx

        My = My.reshape(n, n).astype(np.float64)
        My[mask] /= Mnb[mask]
        My[~mask] = np.nan
        return Mx, My

    def interpolate_multi(self, fields: dict, resolution=128, output_dir="."):
        """
        Interpolate multiple fields and save as CSV.

        Parameters
        ----------
        fields : dict {name: ndarray (N,)}
        resolution : int
        output_dir : str

        Returns
        -------
        dict {name: ndarray (resolution, resolution)}
        """
        os.makedirs(output_dir, exist_ok=True)
        result = {}
        for name, vals in fields.items():
            grid = self.interpolate_field(vals, resolution)
            path = os.path.join(output_dir, f"{name}.csv")
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(grid.flatten())
            result[name] = grid
        return result

    def save_maps(self, output_dir, resolution=128):
        """
        Interpolate X, Y maps and save as x_data.csv, y_data.csv.
        """
        os.makedirs(output_dir, exist_ok=True)
        grid_x, grid_y = self.interpolate_coords(resolution)

        path_x = os.path.join(output_dir, "x_data.csv")
        path_y = os.path.join(output_dir, "y_data.csv")
        with open(path_x, 'w', newline='') as f:
            csv.writer(f).writerow(grid_x.flatten())
        with open(path_y, 'w', newline='') as f:
            csv.writer(f).writerow(grid_y.flatten())

        return grid_x, grid_y
