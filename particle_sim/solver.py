import numpy as np
from particle_sim.animations import AnimationHandler
from scipy.integrate import RK45
from mesh_generation.geometry import sample_sdf, generate_sdf
import matplotlib.pyplot as plt
import shapely
import jax
import jax.numpy as jnp


# CONSTANTS
DRAG_COEFF = 100

# JAX-Optimized Forces
@jax.jit
def inter_point_repulsion(delta):
        r2 = jnp.sum(delta ** 2)
        rep = (r2 + 1e-7) ** -3 # Without the sqrt(), this acts as a force with 6 exponent (highly dissipative)

        norm = jnp.linalg.norm(delta)
        dir = jnp.where(norm > 1e-8, delta / norm, jnp.zeros_like(delta))
        return rep * dir

@jax.jit
def soft_wall_repulsion(this, sdf, grad_x, grad_y, min_p, max_p):
    dist = sample_sdf(grid=sdf, this=this, min_p=min_p, max_p=max_p)
    nx = sample_sdf(grid=grad_x, this=this, min_p=min_p, max_p=max_p)
    ny = sample_sdf(grid=grad_y, this=this, min_p=min_p, max_p=max_p)
    normal = jnp.array([nx, ny])

    dist_sq = dist ** 2    
    mag = (dist_sq + 1e-7) ** -3
    force = -mag * normal
    return force

@jax.jit
def drag(vel):
    speed = jnp.linalg.norm(vel)
    return -DRAG_COEFF * vel * speed


# Point Cloud Solver Class
class PointCloudSolver:
    def __init__(self, dpi=100, width=5, height=4, n_bodies=1, force_multiplier=100, drag_coeff=0.01,
                 plots=None, polygon=None, fps=15):

        self.solution = np.empty((0, 2))
        self.dpi = dpi
        self.width = width
        self.height = height
        self.n_bodies = n_bodies
        self.force_multiplier = force_multiplier
        self.drag_coefficient = drag_coeff
        self.scatter = plt.scatter(np.zeros((self.n_bodies, 1)), np.zeros((self.n_bodies, 1)), c='blue', marker='o')
        self.polygon = polygon

        self.vel_threshold = 0.5
        self.anim = AnimationHandler(
            n_bodies=self.n_bodies, width=self.width, height=self.height,
            polygon=polygon, dpi=self.dpi, plots=plots, fps=fps, vel_threshold=self.vel_threshold
        )

        sdf_grid, grad_x, grad_y, min_p, max_p = generate_sdf(polygon)

        # JAX setup
        #self.lj_kernel = jax.jit(jax.grad(self.lj_potential))
        sdf = jnp.array(sdf_grid)
        grad_x = jnp.array(grad_x)
        grad_y = jnp.array(grad_y)

        point_force = lambda v: inter_point_repulsion(v)
        wall_force = lambda v: soft_wall_repulsion(v, sdf, grad_x, grad_y, min_p, max_p)

        self.point_vmap = jax.vmap(jax.vmap(point_force))
        self.wall_vmap = jax.vmap(wall_force)
        self.drag_vmap = jax.vmap(drag)


    def add_polygon_bounds(self, polygon):
        self.phys.polygon = polygon


    def generate_random_initial_state(self):
        rng = np.random.default_rng()

        mic_line = shapely.maximum_inscribed_circle(self.polygon)
        radius = mic_line.length
        center_x, center_y = mic_line.coords[0]
        offset = radius / (2**0.5)  # R / sqrt(2)

        min_x = center_x - offset
        min_y = center_y - offset
        max_x = center_x + offset
        max_y = center_y + offset

        pos = rng.uniform(
            [min_x + 0.2, min_y + 0.2], 
            [max_x - 0.2, max_y - 0.2], 
            size=(self.n_bodies, 2)
        )
        random_deviation = np.random.uniform(-0.1, 0.1, size=(self.n_bodies, 2))
        pos += random_deviation

        vel = np.zeros((self.n_bodies, 2))
        state = np.concatenate((pos, vel))
        return jnp.array(state)
    

    def find_max_vel_at_state(self, state):
        vel_series = state[self.n_bodies:, :]
        vel_series = [np.linalg.norm(vel) for vel in vel_series]
        max_vel = np.max(vel_series)
        return max_vel


    def calculate_derivatives(self, state): 
        state = state.reshape(-1, 2)
        num_bodies = int(state.shape[0] / 2)
        pos_i = state[:num_bodies]
        vel_i = state[num_bodies:]

        # Apply forces through vectorized transformations
        delta = pos_i[:, None, :] - pos_i[None, :, :]
        p_forces = self.point_vmap(delta)
        p_forces = jnp.sum(p_forces, axis=1) / 2

        w_forces = self.wall_vmap(pos_i)
        drag = self.drag_vmap(vel_i)
        total_force = p_forces + w_forces + drag

        # Combine with velocity and flatten
        return jnp.vstack([vel_i, total_force]).flatten()


    def solve(self, state0=None, max_step=0.05, steps=5, out=None):
        print("Beginning simulation.")
        state0 = self.generate_random_initial_state() if state0 is None else state0
        y0 = state0.flatten()

        self.solution = np.zeros((steps, state0.shape[0], state0.shape[1]))
        func = jax.jit(lambda t,y: self.calculate_derivatives(y))
        solver = RK45(func, 0, y0=y0, t_bound=max_step * (steps + 1), max_step=max_step)

        for i in range(steps):
            if (i + 1) % (steps / 10) == 0:
                print(f"Completed {i + 1}/{steps} steps. {(i+1)/steps*100:.1f}% complete.")

            solver.step()
            y = solver.y.reshape(state0.shape)
            self.solution[i] = y

            if (i + 1) % int(steps / 100) == 0:
                vel_series = self.solution[i, self.n_bodies:, :]
                vel_series = [np.linalg.norm(vel) for vel in vel_series]
                max_vel = np.max(vel_series)
                if max_vel <= self.vel_threshold:
                    all_below = True
                    for iter in range(i - 10, i + 1):
                        iter = max(0, iter)
                        if self.find_max_vel_at_state(self.solution[iter]) > self.vel_threshold:
                            all_below = False
                    
                    if all_below:
                        print(f"Convergence Reached ({i+1}/{steps})")
                        self.solution = self.solution[:(i+1)]
                        return self.solution

        print("Solution did not converge")
        return self.solution


    def animate(self, out=None, color="blue"):
        if self.solution.shape[0] == 0:
            print("No solution to animate.")
            return
        
        if out:
            self.anim.out = out

        print("Beginning animation.")
        self.anim.animate(self.solution, color)


if __name__ == '__main__':
    N_BODIES = 37
    REPULSIVE_STRENGTH = 10
    WIDTH = 6
    HEIGHT = 4

    FPS = 20
    DPI = 50
    RUNTIME = 10
    DRAG_COEF = 0.7

    h = 1 / FPS
    steps = int(RUNTIME / h)

    solver = PointCloudSolver(
        n_bodies=N_BODIES,
        force_multiplier=REPULSIVE_STRENGTH,
        width=WIDTH,
        height=HEIGHT,
        drag_coeff=DRAG_COEF,
    )
    solver.solve(h=h, steps=steps)
    solver.animate()

