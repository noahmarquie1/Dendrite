import numpy as np
import shapely
from shapely import Polygon, Point, LineString
from shapely.plotting import plot_polygon
import matplotlib.pyplot as plt
from particle_sim.solver import PointCloudSolver
import jax.numpy as jnp


# Geometry Classes
class Edge:
    def __init__(self, start, end):
        self.linestrings = [LineString([start, end])]
        self.points_sorted = [start, end]
        self.total_linestring = LineString([start, end])


    def update_points_sorted(self):
        self.points_sorted = []
        start = np.array(self.linestrings[0].coords)[0]
        self.points_sorted.append(start)
        for linestring in self.linestrings:
            end = np.array(linestring.coords)[1]
            self.points_sorted.append(end)

    
    def add_point(self, point: Point):
        for i, line in enumerate(self.linestrings):
            if line.distance(point) < 1e-8:
                start = np.array(line.coords)[0]
                end = np.array(line.coords)[1]
                midpoint = np.array(point.coords)[0]

                self.linestrings[i] = LineString([start, midpoint])
                self.linestrings.insert(i+1, LineString([midpoint, end]))
                break;

        self.update_points_sorted()


class Square:
    def __init__(self, width, height, step_size=0.1):

        self.corners = np.array([
            [-width / 2, -height / 2], [width / 2, -height / 2], 
            [width / 2, height / 2], [-width / 2, height / 2]
        ])
        self.step_size = step_size

        n_step_x = round(width / step_size)
        n_step_y = round(height / step_size)

        x_start = np.array([-width / 2, -height / 2])
        x_end = np.array([width / 2, -height / 2])

        y_start = np.array([width / 2, -height / 2])
        y_end = np.array([width / 2, height / 2])

        self.x_pts = np.linspace(x_start, x_end, n_step_x)
        self.y_pts = np.linspace(y_start, y_end, n_step_y)

        self.b = Edge(self.corners[0], self.corners[1])
        self.r = Edge(self.corners[1], self.corners[2])
        self.t = Edge(self.corners[3], self.corners[2])
        self.l = Edge(self.corners[0], self.corners[3])
        self.edges = [self.b, self.r, self.t, self.l]
        self.opposite_edges = { self.b: self.t, self.t: self.b, self.l: self.r, self.r: self.l }
        self.axes = { self.b: 'x_pts', self.t: 'x_pts', self.l: 'y_pts', self.r: 'y_pts' }

        self.make_grid()
        self.make_mesh()


    def find_in_row(self, pt):
        for i, row in enumerate(self.rows):
            for j, point in enumerate(row):
                if np.allclose(point, pt, atol=1e-7):
                    return i, j
        return None, None


    def find_in_col(self, pt):
        for i, col in enumerate(self.cols):
            for j, point in enumerate(col):
                if np.allclose(point, pt, atol=1e-7):
                    return i, j
                
        return None, None


    def make_grid(self):
        base = None
        for p1 in self.x_pts:
            for p2 in self.y_pts:
                dist = np.linalg.norm(p1 - p2)
                if dist < 1e-7:
                    base = p1
                    break;

        if base is None:
            raise ValueError("Axes not aligned.")

        x = self.x_pts[np.newaxis, :, :] # x points form columns
        y = self.y_pts[:, np.newaxis, :] # y points form rows
        self.grid = x + y - base

        self.rows = [self.grid[i, :, :] for i in range(len(self.y_pts))]
        self.cols = [self.grid[:, i, :] for i in range(len(self.x_pts))]
        self.points = np.vstack([row for row in self.rows])


    def make_mesh(self):
        self.mesh = Polygon(self.corners)


    def transform(self, offset, theta):
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [c, -s],
            [s,  c]
        ])

        self.points = self.points @ rotation_matrix.T
        self.points[:, 0] += offset[0]
        self.points[:, 1] += offset[1]

        self.x_pts = self.x_pts @ rotation_matrix.T
        self.x_pts[:, 0] += offset[0]
        self.x_pts[:, 1] += offset[1]

        self.y_pts = self.y_pts @ rotation_matrix.T
        self.y_pts[:, 0] += offset[0]
        self.y_pts[:, 1] += offset[1]

        self.corners = self.corners @ rotation_matrix.T
        self.corners[:, 0] += offset[0]
        self.corners[:, 1] += offset[1]

        self.b = Edge(self.corners[0], self.corners[1])
        self.r = Edge(self.corners[1], self.corners[2])
        self.t = Edge(self.corners[3], self.corners[2])
        self.l = Edge(self.corners[0], self.corners[3])
        self.edges = [self.b, self.r, self.t, self.l]
        self.opposite_edges = { self.b: self.t, self.t: self.b, self.l: self.r, self.r: self.l }
        self.axes = { self.b: 'x_pts', self.t: 'x_pts', self.l: 'y_pts', self.r: 'y_pts' }

        self.make_grid()
        self.make_mesh()

    
    def add_edge_point(self, point: Point):
        line_points = np.zeros((0,2))
        self.points = np.zeros((0,2))
        tolerance = 1e-7

        edge_found = False
        for edge in self.edges:
            if edge.total_linestring.distance(point) < tolerance:
                edge_found = True
                edge.add_point(point)

                opposite_edge = self.opposite_edges[edge]
                opposite_point = opposite_edge.total_linestring.interpolate(opposite_edge.total_linestring.project(point))
                opposite_edge.add_point(opposite_point)

                for linestring in edge.linestrings:
                    start = np.array(linestring.coords)[0]
                    end = np.array(linestring.coords)[1]
                    n_step = max(2, round(np.linalg.norm(end - start) / self.step_size))
                    line_seg = np.linspace(start, end, n_step)
                    line_points = np.append(line_points, line_seg, axis=0)

                setattr(self, self.axes[edge], line_points)
                break
        
        if not edge_found:
            raise ValueError("Point entered is not on an edge of the square")

        self.make_grid()
        self.make_mesh()


# Region Classes and Helper Classes
class StaticRegion:
    def __init__(self, points=None):
        self.points = points


    def visualize(self):
        plt.scatter(self.points[:, 0], self.points[:, 1])


class DynamicRegionSolver(PointCloudSolver):
    def __init__(self, dpi=100, width=6, height=6, n_bodies=1, plots=None, polygon=None, fps=15, region=None):
        super().__init__(dpi=dpi, width=width, height=height, n_bodies=n_bodies, plots=plots, polygon=polygon, fps=fps)
        self.region = region


    def calculate_derivatives(self, state):
        state = state.reshape(-1, 2)
        num_bodies = int(state.shape[0] / 2)
        pos_i = state[:num_bodies]
        pos_i = jnp.vstack([pos_i, self.region.boundary_points])

        vel_i = state[num_bodies:]

        # Apply forces through vectorized transformations
        delta = pos_i[:, None, :] - pos_i[None, :, :]
        p_forces = self.point_vmap(delta)
        p_forces = jnp.sum(p_forces, axis=1) / 2
        p_forces = p_forces[:num_bodies]

        drag = self.drag_vmap(vel_i)
        total_force = p_forces + drag

        # Combine with velocity and flatten
        return jnp.vstack([vel_i, total_force]).flatten()


class DynamicRegion:
    def __init__(self, boundary_points, mesh):
        self.boundary_points = boundary_points
        self.mesh = mesh
        self.solver = DynamicRegionSolver(polygon=mesh, n_bodies=20, region=self)


    def visualize(self):
        plt.scatter(self.boundary_points[:, 0], self.boundary_points[:, 1])
        #plot_polygon(self.mesh)


    def fill_region(self):
        self.solver.solve(steps=int(2e3))
        self.visualize()
        self.solver.animate(out="anim.gif")



# Mesh Class
class SquareMesh:
    def __init__(self, square):
        self.squares: list[Square] = [square] 
        self.mesh = square.mesh
        self.static_regions: dict[Square, StaticRegion] = { square: StaticRegion(square.points) }
        self.intersections = { square: [] }
        self.dynamic_regions: list[DynamicRegion] = []

    
    def update_static_regions(self, square):
        static_points = square.points
        for s2 in self.intersections[square]:
            intersection = square.mesh.intersection(s2.mesh)
            mask = [intersection.distance(Point(pt)) > 1e-7 for pt in static_points]
            static_points = static_points[mask]

        self.static_regions[square].points = static_points
                    


    def dynamic_region(self, s1: Square, s2: Square):
        boundary_points = []
        intersection = s1.mesh.intersection(s2.mesh)
        mesh = intersection

        s1_overlapping = s1.points[[intersection.distance(Point(pt)) < 1e-7 for pt in s1.points]]
        s2_overlapping = s2.points[[intersection.distance(Point(pt)) < 1e-7 for pt in s2.points]]

        for pt in s1_overlapping:
            ri, rj = s1.find_in_row(pt)
            ci, cj = s1.find_in_col(pt)

            if rj + 1 < len(s1.rows[ri]) - 1 and not intersection.contains(Point(s1.rows[ri][rj+1])):
                boundary_points.append(s1.rows[ri][rj+1])
            if rj > 0 and not intersection.contains(Point(s1.rows[ri][rj-1])):
                boundary_points.append(s1.rows[ri][rj-1])
            if cj + 1 < len(s1.rows[ri]) - 1 and not intersection.contains(Point(s1.cols[ci][cj+1])):
                boundary_points.append(s1.cols[ci][cj+1])
            if cj > 0 and not intersection.contains(Point(s1.cols[ci][cj-1])):
                boundary_points.append(s1.cols[ci][cj-1])

        for pt in s2_overlapping:
            ri, rj = s2.find_in_row(pt)
            ci, cj = s2.find_in_col(pt)

            if rj + 1 < len(s2.rows[ri]) - 1 and not intersection.contains(Point(s2.rows[ri][rj+1])):
                boundary_points.append(s2.rows[ri][rj+1])
            if rj > 0 and not intersection.contains(Point(s2.rows[ri][rj-1])):
                boundary_points.append(s2.rows[ri][rj-1])
            if cj + 1 < len(s2.rows[ri]) - 1 and not intersection.contains(Point(s2.cols[ci][cj+1])):
                boundary_points.append(s2.cols[ci][cj+1])
            if cj > 0 and not intersection.contains(Point(s2.cols[ci][cj-1])):
                boundary_points.append(s2.cols[ci][cj-1])

        boundary_points = np.array(boundary_points)
        return DynamicRegion(boundary_points=boundary_points, mesh=mesh)



    def add_square(self, s_new: Square):
        self.intersections[s_new] = []
        for s in self.squares:
            edges1 = s.mesh.exterior
            edges2 = s_new.mesh.exterior

            intersections = edges2.intersection(edges1)
            points = shapely.get_coordinates(intersections)
            for point in points:
                s.add_edge_point(Point(point))
                s_new.add_edge_point(Point(point))

            if points.shape[0] != 0:
                self.intersections[s].append(s_new)
                self.intersections[s_new].append(s)
                self.update_static_regions(s)
                self.dynamic_regions.append(self.dynamic_region(s, s_new))
        
        self.squares.append(s_new)
        self.static_regions[s_new] = StaticRegion(s_new.points)
        self.update_static_regions(s_new)


if __name__ == "__main__":
    square = Square(1, 2, step_size=0.2)
    square.add_edge_point(Point([0.5, 0.1]))

    pt = np.array([0.5, 0.1])
    square.find_in_row(pt)
    quit()
    plot_polygon(square.mesh)
    plt.show()