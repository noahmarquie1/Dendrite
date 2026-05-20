from particle_sim.solver import PointCloudSolver
from shapely.geometry import Polygon
from mesh_generation.mesh import Mesh
import numpy as np

DPI = 75
N_BODIES = 50
DRAG_COEF = 100
FPS = 15
FORCE_MULTIPLIER = 1

T = 6
step = T * 1e-2

vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
square_polygon = Polygon(vertices)

# make a hexagonal initial distribution
square_mesh = Mesh(square_polygon)
hex_points = square_mesh.hex_fill_precise(n_bodies_ideal=N_BODIES)
hex_points *= 0.95 # Protection against points on edge

N_POINTS_ACTUAL = hex_points.shape[0]
print(N_POINTS_ACTUAL)

state0 = np.vstack([
    hex_points, 
    np.random.uniform(low=-1000.0, high=1000.0, size=hex_points.shape),
])


solver = PointCloudSolver(
    dpi=DPI,
    n_bodies=N_BODIES,
    force_multiplier=FORCE_MULTIPLIER,
    width=6,
    height=6,
    drag_coeff=DRAG_COEF,
    plots=['pdf-anim', 'max-vel-dynamic', 'pdf-comparison'],
    polygon=square_polygon,
    fps=FPS,
    deg=2,
)

solver.solve(
    max_step=step, 
    steps=int(2e3), 
    out="./animation.mp4",
    state0=state0,
)
solver.animate()

