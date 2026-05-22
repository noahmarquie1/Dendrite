import shapely
from shapely.geometry import Polygon, Point
from shapely.affinity import translate, rotate
import numpy as np
from particle_sim.solver import PointCloudSolver
from data.poly_management import load_polygon, save_polygon
import matplotlib.pyplot as plt
from mesh_generation.mesh import Mesh

# Define constants
DPI = 75
N_BODIES = 300
DRAG_COEF = 40
FPS = 15
FORCE_MULTIPLIER = 1

T = 6
step = T * 1e-2

# Generate geometry
rect1 = load_polygon("./data/polygons/int_rect_1.csv")
rect2 = load_polygon("./data/polygons/int_rect_2.csv")

poly1 = Polygon(rect1)
poly2 = Polygon(rect2)
poly2 = translate(poly2, 0, 1)
poly2 = rotate(poly2, 45, origin=Point([1/4, 1]))

composite_rect = shapely.union(poly1, poly2)
composite_rect_mesh = Mesh(composite_rect)

# Compute and animate solution
solver = PointCloudSolver(
    dpi=DPI,
    n_bodies=N_BODIES,
    force_multiplier=FORCE_MULTIPLIER,
    width=6,
    height=6,
    drag_coeff=DRAG_COEF,
    plots=None,
    polygon=composite_rect,
    fps=FPS,
)

sol = solver.solve(
    max_step=step, 
    steps=int(1e3),
)

final_points = np.vstack([composite_rect_mesh.edge_points, sol[-1][:N_BODIES]])
save_polygon(final_points, "./data/meshes/int_ex.csv")
solver.animate(out="./anim.gif")

plt.close('all')
plt.scatter(final_points[:, 0], final_points[:, 1])
plt.savefig("./mesh.png")