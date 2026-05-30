import numpy as np
import shapely
from shapely import Polygon, Point
from shapely.plotting import plot_polygon
import matplotlib.pyplot as plt
from particle_sim.solver import PointCloudSolver
import jax.numpy as jnp
from geometry.rect_geometry import Rect


# Static Region Class
#  - starts off with points from a Rect
#  - points can be refreshed when a new square is added (often updating grid layouts)
#  - keeps track of intersecting dynamic regions as to avoid duplicate points
class StaticRegion:
    def __init__(self, points=None):
        self.points = points

    def visualize(self, ax):
        ax.scatter(self.points[:, 0], self.points[:, 1], alpha=0.5)


# Dynamic Region Classes
class DynamicRegionSolver(PointCloudSolver):
    def __init__(self, dpi=100, width=6, height=6, n_bodies=1, plots=None, polygon=None, fps=15, region=None):
        super().__init__(dpi=dpi, width=width, height=height, n_bodies=n_bodies, plots=plots, polygon=polygon, fps=fps)
        self.region = region


    def calculate_derivatives(self, state):
        state = state.reshape(-1, 2)
        n_bodies = int(state.shape[0] / 2)
        pos_i = state[:n_bodies]
        pos_i = jnp.vstack([pos_i, self.region.boundary_points])
        vel_i = state[n_bodies:]

        delta = pos_i[:, None, :] - pos_i[None, :, :]
        p_forces = self.point_vmap(delta)
        p_forces = jnp.sum(p_forces, axis=1) / 2
        p_forces = p_forces[:n_bodies]

        w_forces = self.wall_vmap(pos_i)[:n_bodies]
        drag = self.drag_vmap(vel_i)
        total_force = p_forces + drag + w_forces

        return jnp.vstack([vel_i, total_force]).flatten()



class DynamicRegion:
    def __init__(self, boundary_points, n_bodies):
        self.boundary_points = boundary_points
        self.mesh = shapely.convex_hull(Polygon(boundary_points))
        self.n_bodies = n_bodies
        self.solver = DynamicRegionSolver(polygon=self.mesh, n_bodies=self.n_bodies, region=self)
        self.filled_points = None

    def visualize(self):
        plt.close()
        plt.scatter(self.boundary_points[:, 0], self.boundary_points[:, 1], c='red')
        plt.scatter(self.filled_points[:, 0], self.filled_points[:, 1], c='blue')
        plot_polygon(self.mesh)

    def fill_region(self):
        sol = self.solver.solve(steps=int(2e3))
        self.filled_points = sol[-1][:self.n_bodies]
        self.visualize()
        plt.show()


# Mesh Class
class SquareMesh:
    def __init__(self, rect):
        self.rects: list[Rect] = [rect] 
        self.mesh = rect.mesh
        self.static_regions: dict[Rect, StaticRegion] = { rect: StaticRegion(rect.points) }
        self.intersections: dict[Rect, list[DynamicRegion]] = { rect: [] }
        self.dynamic_regions: list[DynamicRegion] = []

    
    def update_static_regions(self, rects: list[Rect]):
        for rect in rects:
            static_points = rect.points
            if rect in self.intersections.keys():
                for dynamic_region in self.intersections[rect]:
                    mask = [
                        (dynamic_region.mesh.distance(Point(pt)) > 1e-7) and 
                        (not self.pt_in_array(pt, dynamic_region.boundary_points)) 
                        for pt in static_points
                    ]
                    static_points = static_points[mask]

            self.static_regions[rect].points = static_points

        
    def add_intersection(self, rects, dynamic_region):
        for rect in rects:
            if not rect in self.intersections.keys():
                self.intersections[rect] = []

            self.intersections[rect].append(dynamic_region)
                    
    
    def pt_in_array(self, pt, arr):
        return any(np.allclose(pt, arr_p) for arr_p in arr)


    def pt_outside(self, pt, poly):
        return (not poly.contains(pt)) and poly.distance(pt) > 1e-7


    def create_dynamic_region(self, s1: Rect, s2: Rect, connecting_points):
        boundary_points = connecting_points
        intersection = s1.mesh.intersection(s2.mesh)
        s1_overlapping = s1.points[[intersection.contains(Point(pt)) for pt in s1.points]]
        s2_overlapping = s2.points[[intersection.contains(Point(pt)) for pt in s2.points]]
        overlapping = np.vstack([s1_overlapping, s2_overlapping])

        for pt in overlapping:
            squares = [s1, s2]
            for s in squares:
                if self.pt_in_array(pt, s.points):
                    ri, rj = s.find_in_row(pt)
                    ci, cj = s.find_in_col(pt)

                    candidates = [
                        s.rows[ri][min(rj+1, len(s.rows[ri]) - 1)], s.rows[ri][max(0, rj-1)],
                        s.cols[ci][min(cj+1, len(s.cols[ci]) - 1)], s.cols[ci][max(0, cj-1)],
                    ]

                    for cand in candidates:
                        if intersection.distance(Point(cand)) >= 1e-7:
                            boundary_points = np.append(boundary_points, [cand], axis=0)

        dynamic_region = DynamicRegion(boundary_points=boundary_points, n_bodies = s1_overlapping.shape[0])
        self.dynamic_regions.append(dynamic_region)
        self.add_intersection([s1, s2], dynamic_region)
        

    def add_rect(self, rect_n: Rect): # Main function used outside class
        self.static_regions[rect_n] = StaticRegion(rect_n.points)

        for rect in self.rects:
            edges1 = rect.mesh.exterior
            edges2 = rect_n.mesh.exterior
            intersection = edges2.intersection(edges1)
            connecting_points = shapely.get_coordinates(intersection)

            if connecting_points.shape[0] > 0:
                for point in connecting_points:
                    rect.add_edge_point(Point(point))
                    rect_n.add_edge_point(Point(point))

                self.create_dynamic_region(rect, rect_n, connecting_points)
                self.update_static_regions([rect, rect_n])
        
        self.rects.append(rect_n)



if __name__ == "__main__":
    square = Rect(1, 2, step_size=0.2)
    square.add_edge_point(Point([0.5, 0.1]))

    pt = np.array([0.5, 0.1])
    square.find_in_row(pt)
    quit()
    plot_polygon(square.mesh)
    plt.show()