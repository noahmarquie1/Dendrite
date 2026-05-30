import numpy as np
from scipy.spatial import KDTree, Delaunay
from scipy.ndimage import distance_transform_edt
from jax.scipy.ndimage import map_coordinates
import shapely
from shapely import Polygon
from shapely import Geometry
import matplotlib.pyplot as plt


# 3D Geometry
def extrude(base, num_steps, start_z, end_z):
    base = np.hstack([base, np.zeros((base.shape[0], 1))]) # 2d - 3d conversion
    base = base + np.array([0, 0, start_z])

    points = base
    for point in base:
        start = point
        end = point + np.array([0, 0, end_z - start_z])
        line = make_line(start, end, num_steps)
        points = np.vstack([points, line])

    return points


def plot_3d_element(edge_points, inner_points):
    start_z = -0.1
    end_z = 0.1

    poly = Polygon(edge_points)
    x_bounds = (poly.bounds[0] - 0.1, poly.bounds[2] + 0.1)
    y_bounds = (poly.bounds[1] - 0.1, poly.bounds[3] + 0.1)

    plot_height = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1])
    z_bounds = (-plot_height / 2, plot_height / 2)

    walls = extrude(edge_points, num_steps=3, start_z=start_z, end_z=end_z)
    face = np.hstack([inner_points, np.zeros((inner_points.shape[0], 1))])
    top_face = face + np.array([0, 0, end_z])
    bottom_face = face + np.array([0, 0, start_z])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(walls[:, 0], walls[:, 1], walls[:, 2], s=4, color="slategrey", alpha=0.7)
    ax.scatter(top_face[:, 0], top_face[:, 1], top_face[:, 2], s=4, color="royalblue", alpha=1)
    ax.scatter(bottom_face[:, 0], bottom_face[:, 1], bottom_face[:, 2], s=4, color="royalblue", alpha=0.4)
    ax.set_xlim(x_bounds[0], x_bounds[1])
    ax.set_ylim(y_bounds[0], y_bounds[1])
    ax.set_zlim(z_bounds[0], z_bounds[1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud")

    return fig, ax


# SDF Grid Generation
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


# Geometry Helpers
def in_area(p, s):
    tri = Delaunay(s)
    return tri.find_simplex(p) >= 0


def remove_in_area_points(points, s):
    inside = np.array([in_area(p, s) for p in points])
    return points[~inside]


def make_line(p1, p2, num_steps):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    m = p2 - p1
    total_dist = np.linalg.norm(m)
    if total_dist < 1e-8:
        unit_step = np.zeros(shape=(0, len(p1)))
    else:
        unit_step = m / total_dist  # unit vector in direction of line

    num_steps = max(1, num_steps)
    step_size = total_dist / (num_steps - 1)
    local_step = unit_step * step_size

    points = np.array([p1])
    for i in range(1, num_steps):
        points = np.append(points, [points[i - 1] + local_step], axis=0)
    return points


def fetch_neighbors(p, mesh_points, n):
    tree = KDTree(mesh_points)
    neighbor_distances, neighbor_indices = tree.query(p, k=n)
    mask = np.where(neighbor_distances > 1e-8)[0]
    return np.array([mesh_points[i] for i in neighbor_indices[mask]])