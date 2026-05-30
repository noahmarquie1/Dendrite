from particle_sim.solver import PointCloudSolver
from shapely.geometry import Polygon
from mesh_generation.mesh import Mesh
import numpy as np
from data.poly_management import load_polygon, save_polygon
import matplotlib.pyplot as plt

DPI = 75
N_BODIES = 100
DRAG_COEF = 40
FPS = 15

vertices = load_polygon("./data/polygons/basic_square.csv")
square_polygon = Polygon(vertices)

# make a hexagonal initial distribution
square_mesh = Mesh(square_polygon)
solver = PointCloudSolver(
    dpi=DPI,
    n_bodies=N_BODIES,
    plots=None,
    polygon=square_polygon,
    fps=FPS,
)

sol = solver.solve(
    steps=int(3e3), 
)
solver.animate(out="anim.gif")

plt.close('all')
final_points = np.vstack([square_mesh.edge_points, sol[-1][:N_BODIES]])
save_polygon(final_points, "./data/meshes/basic_ex.csv")
plt.scatter(final_points[:, 0], final_points[:, 1])
plt.savefig("mesh.png")


