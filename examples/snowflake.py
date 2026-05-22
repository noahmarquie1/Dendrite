from mesh_generation.snowflake_mesh import generate_full_snowflake
from particle_sim.solver import PointCloudSolver
from data.poly_management import load_polygon, save_polygon
from mesh_generation.mesh import Mesh
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

N_BODIES = 100
FORCE_MULTIPLIER = 10
DPI = 75
DRAG_COEF = 2

snowflake_mesh = generate_full_snowflake(1, 2, 3, 0.1)
width = snowflake_mesh.polygon.bounds[2] - snowflake_mesh.polygon.bounds[0]
height = snowflake_mesh.polygon.bounds[3] - snowflake_mesh.polygon.bounds[1]

T = 6
step = T * 1e-2

solver = PointCloudSolver(
    dpi=DPI,
    n_bodies=N_BODIES,
    force_multiplier=FORCE_MULTIPLIER,
    drag_coeff=DRAG_COEF,
    plots=['pdf-anim', 'max-vel', 'pdf-final'],
    polygon=snowflake_mesh.polygon,
)

sol = solver.solve(
    max_step=step, 
    steps=int(1e3), 
)

solver.animate(out="./anim.gif")
