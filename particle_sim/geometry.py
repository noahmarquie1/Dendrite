import shapely
import numpy as np
from scipy.ndimage import distance_transform_edt
from jax.scipy.ndimage import map_coordinates
from shapely import Geometry

def generate_sdf(geometry: Geometry, res=256):
    x_min, y_min, x_max, y_max = geometry.bounds
    min_p = min(x_min, y_min) - 2
    max_p = max(x_max, y_max) + 2


    x = np.linspace(min_p, max_p, res)
    y = np.linspace(min_p, max_p, res)
    xv, yv = np.meshgrid(x, y)

    points = shapely.points(xv, yv)
    mask = shapely.contains(geometry, points)
    total_span = max_p - min_p
    dx = total_span / (res - 1)

    dist_outside = distance_transform_edt(~mask, sampling=dx)
    dist_inside = distance_transform_edt(mask, sampling=dx)
    sdf_grid = dist_outside - dist_inside
    
    grad_y, grad_x = np.gradient(sdf_grid, dx)
    
    return sdf_grid, grad_x, grad_y, min_p, max_p


def sample_sdf(grid, this, min_p, max_p):
    res = grid.shape[0]

    iy = ((this[1] - min_p) / (max_p - min_p)) * (res - 1)
    ix = ((this[0] - min_p) / (max_p - min_p)) * (res - 1)

    return map_coordinates(input=grid, coordinates=(iy, ix), order=1, mode='nearest')