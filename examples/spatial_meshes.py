from mesh_generation.extrude import *
from mesh_generation.snowflake_mesh import generate_full_snowflake
from mesh_generation.geometry import *
from mesh_generation.stats import *

STEP_SIZE = 0.2

# 3d line
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
p1 = np.array([0, 0, 0])
p2 = np.array([1, 1, 1])
line = make_line(p1, p2, STEP_SIZE)
plot_3d_element(line, ax=ax)
plt.show()

# 3d box
start_z = -1
end_z = 1

square = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
square_edges = make_square_edges(square, STEP_SIZE)
walls = extrude(square_edges, STEP_SIZE, start_z, end_z)

square_area = fill_in_square(square, STEP_SIZE)
square_area = np.hstack([square_area, np.zeros((square_area.shape[0], 1))])
bottom_face = square_area + np.array([0, 0, start_z])
top_face = square_area + np.array([0, 0, end_z])

cube_mesh = np.vstack([bottom_face, top_face, walls])
plot_3d_element(cube_mesh, ax=ax)
plt.show()

# 3d snowflake
snowflake_mesh = generate_full_snowflake(2, 4, 2, STEP_SIZE)

start_z = -0.2
end_z = 0.2

walls = extrude(snowflake_mesh.edge_points, 0.05, start_z, end_z)
face = np.hstack([snowflake_mesh.inner_points, np.zeros((snowflake_mesh.inner_points.shape[0], 1))])
top_face = face + np.array([0, 0, end_z])
bottom_face = face + np.array([0, 0, start_z])
plot_width = 4

fig = plt.figure(figsize=(6, 12))
ax1 = fig.add_subplot(211, projection='3d')
plot_3d_element(walls, boundaries=[(-plot_width, plot_width), (-plot_width, plot_width), (start_z*4, end_z*4)], ax=ax1, color="slategrey", alpha=0.5)
plot_3d_element(top_face, boundaries=[(-plot_width, plot_width), (-plot_width, plot_width), (start_z*4, end_z*4)], ax=ax1, color="royalblue", alpha=1)
plot_3d_element(bottom_face, boundaries=[(-plot_width, plot_width), (-plot_width, plot_width), (start_z*4, end_z*4)], ax=ax1, color="royalblue", alpha=0.5)
ax1.set_title("Snowflake Point Cloud")

# print snowflake statistics
ax2 = fig.add_subplot(212)
all_snowflake_points = np.vstack([walls, top_face, bottom_face])
plot_mesh_pdf(all_snowflake_points, STEP_SIZE, bin_amt=50, ax=ax2, color="royalblue")
ax2.set_title("Snowflake PDF")
plt.show()