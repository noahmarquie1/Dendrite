import shapely
from shapely.geometry import Polygon, Point
from shapely.affinity import translate, rotate
import numpy as np
from particle_sim.solver import PointCloudSolver

# Define constants
DPI = 75
N_BODIES = 16
DRAG_COEF = 40
FPS = 15
FORCE_MULTIPLIER = 1

T = 6
step = T * 1e-2

# Generate geometry
rect1 = np.array([
    [0, 0], [1, 0],
    [1, 2], [0, 2]
])
rect2 = rect1 * 1/2
poly1 = Polygon(rect1)
poly2 = Polygon(rect2)
poly2 = translate(poly2, 0, 1)
poly2 = rotate(poly2, 45, origin=Point([1/4, 1]))

composite_rect = shapely.union(poly1, poly2)

# Compute and animate solution
solver = PointCloudSolver(
    dpi=DPI,
    n_bodies=N_BODIES,
    force_multiplier=FORCE_MULTIPLIER,
    width=6,
    height=6,
    drag_coefficient=DRAG_COEF,
    pdfs=False,
    pdf_interval=FPS,
    polygon=composite_rect,
    fps=FPS,
    deg=2,
)

solver.solve(
    max_step=step, 
    steps=int(1e3), 
    #state0=state0
)
solver.animate()