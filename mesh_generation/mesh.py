import numpy as np
from shapely.geometry import Polygon, Point
from shapely.plotting import plot_polygon
from shapely import unary_union
import matplotlib.pyplot as plt
from hexalattice.hexalattice import create_hex_grid
from geometry.base_geometry import make_line
from mesh_generation.stats import plot_mesh_pdf
from data.poly_management import save_polygon, load_polygon
import os

class Mesh:
    def __init__(self, polygon=None, step_size=0.1):
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

    def edge_fill(self, poly: Polygon, step_size=0.1):
        edge_vertices = np.array(poly.exterior.coords)
        edge_points = edge_vertices
        for i in range(len(edge_vertices) - 1):
            edge_length = np.linalg.norm(edge_vertices[i+1]-edge_vertices[i])
            step_size = step_size if step_size is not None else self.step_size
            num_steps = int(edge_length / step_size)

            line = make_line(edge_vertices[i], edge_vertices[i+1], num_steps=num_steps)
            edge_points = np.vstack([edge_points, line])
        return edge_points

    def add_shape(self, poly):
        self.polygon = unary_union([self.polygon, poly])
        self.edge_points = self.edge_fill(self.polygon)
        self.hex_fill()

    def visualize(self, ax=None):
        if ax is None:
            ax = plt.gca()
        plot_polygon(self.polygon, ax=ax)
        ax.scatter(self.inner_points[:, 0], self.inner_points[:, 1], s=4, color="red")
        ax.scatter(self.edge_points[:, 0], self.edge_points[:, 1], s=4, color="blue")


class MultiMesh(Mesh):
    def __init__(self, dir=None):
        self.poly_list = []
        self.polygon = None
        if dir is not None:
            self.load_geometry(dir)

    
    def edge_fill(self, step_size=0.1):
        self.update_polygon()
        return super().edge_fill(self.polygon, step_size)
    

    def add_shape(self, poly):
        self.poly_list.append(poly)


    def add_shapes(self, shapes):
        self.poly_list.extend(shapes)


    def load_geometry(self, dir):
        if len(dir) > 0 and dir[-1] != "/":
            dir += "/"

        self.poly_list = []
        for filename in os.listdir(dir):
            if '.csv' in filename:
                self.poly_list.append(Polygon(load_polygon(dir + filename)))

        self.update_polygon()


    def save_geometry(self, dir):
        if len(dir) > 0 and dir[-1] != "/":
            dir += "/"

        os.makedirs(dir, exist_ok=True)
        for i, poly in enumerate(self.poly_list):
            save_polygon(np.array(poly.exterior.coords), dir + f"poly_{i}.csv")


    def update_polygon(self):
        self.polygon = unary_union([poly for poly in self.poly_list])


    def visualize(self, ax=None):
        self.update_polygon()
        plot_polygon(unary_union([poly for poly in self.poly_list]))


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



