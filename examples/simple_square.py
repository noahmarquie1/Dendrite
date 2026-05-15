from particle_sim.solver import PointCloudSolver
from shapely.geometry import Polygon
import numpy as np

DPI = 75
N_BODIES = 16
DRAG_COEF = 40
FPS = 15
FORCE_MULTIPLIER = 1

T = 6
step = T * 1e-2

vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
square_polygon = Polygon(vertices)
starting_points = np.array([
    [0.09, 0.09], [0.1, 0.1], [0.11, 0.11], [0.12, 0.12]
])
vel = np.random.random((vertices.shape[0], 2))
state0 = np.concatenate((starting_points, vel))

solver = PointCloudSolver(
    dpi=DPI,
    n_bodies=N_BODIES,
    force_multiplier=FORCE_MULTIPLIER,
    width=6,
    height=6,
    drag_coefficient=DRAG_COEF,
    pdfs=False,
    pdf_interval=FPS,
    polygon=square_polygon,
    fps=FPS,
    deg=2,
)

solver.solve(
    max_step=step, 
    steps=int(1e3), 
    #state0=state0
)
solver.animate()