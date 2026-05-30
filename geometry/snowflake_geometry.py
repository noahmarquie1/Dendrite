import matplotlib.pyplot as plt
from mesh_generation.mesh import MultiMesh
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from shapely.plotting import plot_polygon

STEP_SIZE = 0.2

# Define base transformations
angles = np.array([
        90 - np.degrees(np.arctan(1 / 2)),   # tl
        0,                                   # t
        -90 + np.degrees(np.arctan(1 / 2)),  # tr
        -90 - np.degrees(np.arctan(1 / 2)),  # br
        180,                                 # b
        90 + np.degrees(np.arctan(1 / 2)),  # bl
    ])

offsets = np.array([
    [-3 / 4 +1e-7, 1 / 2 -1e-7],   # tl
    [0, 1],            # t
    [3 / 4 - 1e-7, 1 / 2 - 1e-7],    # tr
    [3 / 4, -1 / 2],   # br
    [0, -1],           # b
    [-3 / 4, -1 / 2],  # bl
])

# creates one snowflake branch with no root rotation
def create_snowflake_branch(width, length, n, step_size, out=None):
    square = np.array([ # basic box for snowflake partition
        [-width / 2, 0], [width / 2, 0],
        [width / 2, length], [-width / 2, length]
    ])
    square = Polygon(square)
    mesh = MultiMesh()
    mesh.add_shape(square)

    if n > 0:
        left_branch_mesh = create_snowflake_branch(width / 2, length / 2, n - 1, step_size)
        right_branch_mesh = create_snowflake_branch(width / 2, length / 2, n - 1, step_size)

        poly_l = [translate(rotate(poly, 45, origin=(0,0)), xoff=0, yoff=length/2) for poly in left_branch_mesh.poly_list]
        poly_r = [translate(rotate(poly, -45, origin=(0,0)), xoff=0, yoff=length/2) for poly in right_branch_mesh.poly_list]

        mesh.add_shapes(poly_l)
        mesh.add_shapes(poly_r)

    return mesh


def generate_full_snowflake(side_length, branch_length, n, step_size, out=None):
    snowflake_mesh = MultiMesh()
    base = Polygon(np.array([
        [side_length/2, side_length], [side_length, 0],
        [side_length/2, -side_length], [-side_length / 2, -side_length],
        [-side_length, 0], [-side_length / 2, side_length],
    ]))

    snowflake_mesh.add_shape(base)

    for i in range(6):
        branch_mesh = create_snowflake_branch(side_length, branch_length, n, step_size)
        branch_poly_list = [
            translate(
                rotate(poly, angles[i],  origin=(0, 0)), 
                xoff=offsets[i, 0]*side_length,  yoff=offsets[i, 1]*side_length,
            ) for poly in branch_mesh.poly_list                
        ]

        snowflake_mesh.add_shapes(branch_poly_list)

    return snowflake_mesh


if __name__ == "__main__":
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    snowflake_mesh = generate_full_snowflake(1, 3, 2, STEP_SIZE)

    #snowflake_mesh.visualize(ax)
    #ax.set_title("Hexagonal Snowflake Mesh")

    snowflake_mesh.save_geometry("data/polygons/snowflake/")
    #plt.show()