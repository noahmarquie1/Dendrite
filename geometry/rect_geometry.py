import numpy as np
from shapely.geometry import LineString, Polygon, Point


# Edge Helper Class
class Edge:
    def __init__(self, start, end):
        self.linestrings: list[LineString] = [LineString([start, end])]
        self.points_sorted: np.ndarray = [start, end]
        self.total_linestring: LineString = LineString([start, end])


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


# Rect Class
class Rect:
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
        self.update_geometry()


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


    def update_geometry(self):
        self.update_edges()
        self.make_grid()
        self.make_mesh()


    def update_edges(self):
        self.b = Edge(self.corners[0], self.corners[1])
        self.r = Edge(self.corners[1], self.corners[2])
        self.t = Edge(self.corners[3], self.corners[2])
        self.l = Edge(self.corners[0], self.corners[3])
        self.edges = [self.b, self.r, self.t, self.l]
        self.opposite_edges = { self.b: self.t, self.t: self.b, self.l: self.r, self.r: self.l }
        self.axes = { self.b: 'x_pts', self.t: 'x_pts', self.l: 'y_pts', self.r: 'y_pts' }


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


    def transform_points(self, points, rot, offset):
        points = points @ rot.T
        points[:, 0] += offset[0]
        points[:, 1] += offset[1]
        return points


    def transform_square(self, offset, theta):
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [c, -s],
            [s,  c]
        ])

        self.points = self.transform_points(self.points, rotation_matrix, offset)
        self.x_pts = self.transform_points(self.x_pts, rotation_matrix, offset)
        self.y_pts = self.transform_points(self.y_pts, rotation_matrix, offset)
        self.corners = self.transform_points(self.corners, rotation_matrix, offset)
        self.update_geometry()

    
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
                line_points = np.append(line_points, np.array([edge.linestrings[0].coords[0]]), axis=0)

                for linestring in edge.linestrings:
                    start = np.array(linestring.coords)[0]
                    end = np.array(linestring.coords)[1]
                    n_step = max(2, round(np.linalg.norm(end - start) / self.step_size))
                    line_seg = np.linspace(start, end, n_step)[1:] 
                    line_points = np.append(line_points, line_seg, axis=0)

                setattr(self, self.axes[edge], line_points)
                break
        
        if not edge_found:
            raise ValueError("Point entered is not on an edge of the square")

        self.make_grid()
        self.make_mesh()