from shapely.plotting import plot_polygon
import numpy as np
import matplotlib.pyplot as plt
from mesh_generation.square_mesh import Square, SquareMesh

# Constants
STEP_SIZE = 0.1


# Testing
s1 = Square(1, 2, step_size=STEP_SIZE)
mesh = SquareMesh(s1)

s2 = Square(0.5, 1.5, step_size=STEP_SIZE)
s2.transform([-0.5, 0.25], np.pi / 4)
mesh.add_square(s2)

mesh.dynamic_regions[0].fill_region()
