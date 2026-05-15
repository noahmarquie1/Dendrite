import numpy as np
from random import choice
from shapely.geometry import Polygon, Point, LineString
from particle_sim.geometry import generate_sdf, sample_sdf

import jax 
import jax.numpy as jnp


class PhysicsHandler:
    def __init__(self, n_bodies, force_multiplier=100, drag_coefficient=0.01, deg=2, width=6, height=4, polygon: Polygon=None):
        self.n_bodies = n_bodies
        self.force_multiplier = force_multiplier
        self.drag_coefficient = drag_coefficient
        self.width = width
        self.height = height
        self.deg = deg

        # Geometry setup
        if polygon is None:
            polygon = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        
        self.polygon = polygon
        sdf_grid, grad_x, grad_y, min_p, max_p = generate_sdf(polygon)

        # JAX setup
        self.lj_kernel = jax.jit(jax.grad(self.lj_potential))
        self.sdf = jnp.array(sdf_grid)
        self.grad_x = jnp.array(grad_x)
        self.grad_y = jnp.array(grad_y)
        self.min_p = min_p
        self.max_p = max_p


    # Base Logic and Physics
    def unit(self, start, end):
        return (end - start) / np.linalg.norm(end - start)


    def standard_repulsion(self, this, other):
        dir = self.unit(start=other, end=this)
        mag = 1 / ((np.linalg.norm(this - other) + 1e-3) ** 2)
        return dir * mag


    def smooth_clamp(self, force, max):
        f_norm = np.linalg.norm(force)
        clamped = force / np.sqrt(1.0 + (f_norm / max) ** 2)
        return clamped


    def lj_potential(self, this, other):
        r2 = jnp.sum((this - other) ** 2)
        attr = (r2 + 1e-2) ** -6
        rep = (r2 + 1e-2) ** -2
        return -4 * 4 * rep


    def lj_force(self, this, other):
        force = np.array(self.lj_kernel(this, other))
        return force
    

    def standard_wall_repulsion(self, this):
        p = Point(this)
        if not self.polygon.contains(p):
            center = np.array([self.polygon.centroid.x, self.polygon.centroid.y])
            dir = (center - this) / np.linalg.norm(center - this)
            dist = p.distance(self.polygon)
            return 1e2 * self.force_multiplier * dist * dir # roughly one order of magnitude higher than other forces
        return np.zeros_like(this)


    def soft_wall_repulsion(self, this):
        dist = sample_sdf(grid=self.sdf, this=this, min_p=self.min_p, max_p=self.max_p)

        nx = sample_sdf(grid=self.grad_x, this=this, min_p=self.min_p, max_p=self.max_p)
        ny = sample_sdf(grid=self.grad_y, this=this, min_p=self.min_p, max_p=self.max_p)
        normal = jnp.array([nx, ny])
        
        mag = (10*dist + 1e-7) ** -6
        force = -mag * normal
        return force


    # Functions called each iteration
    def calculate_repulsive_force(self, this, other):
        return self.lj_force(this, other)
    

    def calculate_wall_force(self, this):
        return self.soft_wall_repulsion(this)


    def calculate_drag_force(self, vel):
        speed = np.linalg.norm(vel)
        return -self.drag_coefficient * vel * speed