import numpy as np
from scipy.spatial import KDTree, Delaunay
from scipy.ndimage import distance_transform_edt
from jax.scipy.ndimage import map_coordinates
import shapely
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


def in_area(p, s):
    tri = Delaunay(s)
    return tri.find_simplex(p) >= 0


def remove_in_area_points(points, s):
    inside = np.array([in_area(p, s) for p in points])
    return points[~inside]


def make_line(p1, p2, step_size):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    m = p2 - p1
    total_dist = np.linalg.norm(m)
    if total_dist < 1e-8:
        unit_step = np.zeros(shape=(0, len(p1)))
    else:
        unit_step = m / total_dist  # unit vector in direction of line

    total_steps = int(total_dist / step_size)
    if total_steps == 0:
        return np.zeros(shape=(0, len(p1)))
    actual_step_size = total_dist / total_steps
    local_step = unit_step * actual_step_size

    points = np.array([p1])
    for i in range(1, total_steps):
        points = np.append(points, [points[i - 1] + local_step], axis=0)
    return points


def make_square_edges(s, step_size):
    total_points = np.array([]).reshape(0, 2)
    s = np.append(s, [s[0]], axis=0)
    for i in range(s.shape[0] - 1):
        total_points = np.append(total_points, make_line(s[i], s[i+1], step_size), axis=0)
    return total_points


def fill_in_square(s, step_size):
    # travel from edge 1 to edge 3, using known length and direction of edge 4
    start_points = make_line(s[0], s[1], step_size)
    in_area_points = np.array([]).reshape(0, 2)
    for point in start_points:
        in_area_points = np.append(in_area_points, make_line(point, point + (s[3] - s[0]), step_size), axis=0)
    return in_area_points


def fetch_neighbors(p, mesh_points, n):
    tree = KDTree(mesh_points)
    neighbor_distances, neighbor_indices = tree.query(p, k=n)
    mask = np.where(neighbor_distances > 1e-8)[0]
    return np.array([mesh_points[i] for i in neighbor_indices[mask]])