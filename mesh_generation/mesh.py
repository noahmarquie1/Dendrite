import numpy as np
from shapely.geometry import Polygon, Point
from shapely.plotting import plot_polygon
from shapely import unary_union
import matplotlib.pyplot as plt
from hexalattice.hexalattice import create_hex_grid
from mesh_generation.geometry import make_line
from mesh_generation.stats import plot_mesh_pdf

class Mesh:
    def __init__(self, polygon, step_size=0.1):
        self.inner_points = np.zeros((0, 2))
        self.edge_points = np.zeros((0, 2))
        self.polygon = polygon
        self.step_size = step_size

        self.add_shape(polygon)

    def hex_fill(self, step_size=None):
        if step_size is None:
            step_size = self.step_size

        max_radius = 0
        for point in self.edge_points:
            point_dist = np.linalg.norm(point)
            max_radius = max(max_radius, point_dist)

        nx = int(max_radius / step_size) * 2
        ny = int(max_radius / (step_size * np.sqrt(3) / 2)) * 2
        nx = max(nx, 1)
        ny = max(ny, 1)

        hex_centers, _ = create_hex_grid(nx=nx, ny=ny, min_diam=step_size)
        mask = np.array([self.polygon.contains(Point(p)) for p in hex_centers])
        self.inner_points = hex_centers[mask]
        return self.inner_points
    
    def hex_fill_precise(self, n_bodies_ideal):
        step_size = 3
        hex_points = np.zeros((0,2))
        while hex_points.shape[0] < n_bodies_ideal:
            hex_points = self.hex_fill(step_size)
            num = hex_points.shape[0]
            step_size -= 0.01
        return hex_points
        

    def edge_fill(self):
        temp_vertices = np.append(self.edge_points, [self.edge_points[0]], axis=0)
        for i in range(len(temp_vertices) - 1):
            line = make_line(temp_vertices[i], temp_vertices[i+1], step_size=self.step_size)
            self.edge_points = np.append(self.edge_points, line, axis=0)

    def add_shape(self, polygon):
        self.edge_points = np.array([v for v in self.edge_points if not polygon.contains(Point(v))])
        self.polygon = unary_union([self.polygon, polygon])
        self.edge_points = self.polygon.exterior.coords
        self.edge_fill()
        self.hex_fill()

    def visualize(self, ax=None):
        if ax is None:
            ax = plt.gca()
        plot_polygon(self.polygon, ax=ax)
        ax.scatter(self.inner_points[:, 0], self.inner_points[:, 1], s=4, color="red")
        ax.scatter(self.edge_points[:, 0], self.edge_points[:, 1], s=4, color="blue")


# test code
if __name__ == "__main__":
    m = Mesh(Polygon([[0, 0], [1, 0], [1, 1], [0, 1]]), step_size=0.2)
    m.add_shape(Polygon([[0, 0], [1, 0], [1, 1], [0, 1]]))
    m.add_shape(Polygon([[0.5, 0.5], [1.5, 0.5], [1.5, -0.5], [0.5, -0.5]]))

    # visualize basic example
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    m.visualize(ax[0])
    ax[0].set_title("Intersecting Cubes, Hexagonal Filling")

    plot_mesh_pdf(m.inner_points, m.step_size, bin_amt=40, ax=ax[1])
    plt.tight_layout()
    plt.show()



