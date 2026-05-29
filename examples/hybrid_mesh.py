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


plt.close('all')
fig, ax = plt.subplots(1,1)
ax.set_aspect(1)
for region in mesh.static_regions.values():
    region.visualize(ax)
ax.scatter(mesh.dynamic_regions[0].filled_points[:, 0], mesh.dynamic_regions[0].filled_points[:, 1], c="green", alpha=0.3)
ax.scatter(mesh.dynamic_regions[0].boundary_points[:, 0], mesh.dynamic_regions[0].boundary_points[:, 1], c='purple', alpha=0.3)


plt.show()
