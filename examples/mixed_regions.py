from particle_sim.solver import PointCloudSolver
from mesh_generation.geometry import remove_in_area_points, make_square_edges
from shapely.geometry import Polygon
from mesh_generation.mesh import Mesh
import numpy as np
import matplotlib.pyplot as plt
from data.poly_management import save_polygon, load_polygon

# Constants and Global Variables
DPI = 75
N_BODIES = 100
DRAG_COEF = 100
FPS = 15
FORCE_MULTIPLIER = 1

T = 6
step = T * 1e-2


# Static Geometry 
square_1 = load_polygon("./data/polygons/mix_square_1.csv")
square_2 = load_polygon("./data/polygons/mix_square_2.csv")


mesh = Mesh(Polygon(square_1))
mesh.add_shape(Polygon(square_2))
mesh.inner_points = remove_in_area_points(mesh.inner_points, square_2)
mesh.inner_points = np.vstack([mesh.inner_points, mesh.edge_fill(Polygon(square_2))])


# Dynamic Geometry 
solver = PointCloudSolver(
    dpi=DPI,
    n_bodies=N_BODIES,
    force_multiplier=FORCE_MULTIPLIER,
    width=6,
    height=6,
    drag_coeff=DRAG_COEF,
    plots=None,
    polygon=Polygon(square_2),
    fps=FPS,
)

sol = solver.solve(
    max_step=step, 
    steps=int(1e3), 
)
solver.anim.add_static_points(mesh.inner_points, "red")
solver.anim.add_static_points(mesh.edge_points, "blue")

plt.close("all")
fig, ax = plt.subplots(1,1)
solver.animate(out="./anim.gif", color="purple")

final_points = np.vstack([sol[-1][:N_BODIES], mesh.inner_points, mesh.edge_points])
save_polygon(final_points, "./data/meshes/mix_ex.csv")

ax.set_aspect("equal")
mesh.visualize(ax=ax)
ax.scatter(sol[-1][:N_BODIES, 0], sol[-1][:N_BODIES, 1], s=4, c='purple')
plt.savefig("./mesh.png")


