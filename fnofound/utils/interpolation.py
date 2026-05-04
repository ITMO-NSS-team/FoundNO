from scipy.interpolate import griddata
import numpy as np


def interpolate_to_logical(points, values, resolution=128):
    """
    Interpolate data from physical domain onto a regular logical grid.

    Parameters
    ----------
    points : ndarray (N, 2)
        Coordinates (x, y) in physical space.
    values : ndarray (N, C)
        Field values at each point.
    resolution : int
        Grid resolution (H=W=resolution).

    Returns
    -------
    grid_values : ndarray (resolution, resolution)
        Interpolated field on the regular [0,1]^2 grid.
    """
    grid_x, grid_y = np.mgrid[0:1:complex(resolution), 0:1:complex(resolution)]
    # Linear or nearest-neighbor interpolation via scipy Delaunay triangulation
    grid_values = griddata(points, values, (grid_x, grid_y),
                           method='linear', fill_value=0.0)
    return grid_values
