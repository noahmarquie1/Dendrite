from particle_sim.solver import PointCloudSolver
import jax.numpy as jnp
import jax
import numpy as np
from particle_sim.physics import PhysicsHandler
from particle_sim.geometry import generate_sdf, sample_sdf
from scipy.integrate import RK45

# CONSTANTS
DRAG_COEFF = 100

# JAX-Optimized Forces
@jax.jit
def inter_point_repulsion(delta):
        r2 = jnp.sum(delta ** 2)
        rep = (r2 + 1e-2) ** -3 # Without the sqrt(), this acts as a force with 6 exponent (highly dissipative)

        norm = jnp.linalg.norm(delta)
        dir = jnp.where(norm > 1e-8, delta / norm, jnp.zeros_like(delta))
        return rep * dir

@jax.jit
def soft_wall_repulsion(this, sdf, grad_x, grad_y, min_p, max_p):
    dist = sample_sdf(grid=sdf, this=this, min_p=min_p, max_p=max_p)
    nx = sample_sdf(grid=grad_x, this=this, min_p=min_p, max_p=max_p)
    ny = sample_sdf(grid=grad_y, this=this, min_p=min_p, max_p=max_p)
    normal = jnp.array([nx, ny])
        
    mag = (dist + 1e-2) ** -6
    force = -mag * normal * (1/20)
    return force

@jax.jit
def drag(vel):
    speed = jnp.linalg.norm(vel)
    return -DRAG_COEFF * vel * speed


class DiffraxSolver(PointCloudSolver):
    def __init__(self, dpi=100, width=5, height=4, n_bodies=1, force_multiplier=100, drag_coeff=0.01,
                 plots=None, polygon=None, fps=15, deg=2):
        
        super().__init__(dpi=dpi, width=width, height=height, n_bodies=n_bodies, force_multiplier=force_multiplier, drag_coeff=drag_coeff,
                 plots=plots, polygon=polygon, fps=fps, deg=deg)
        
        sdf = self.phys.sdf
        grad_x = self.phys.grad_x
        grad_y = self.phys.grad_y
        min_p = self.phys.min_p
        max_p = self.phys.max_p
        
        point_force = lambda v: inter_point_repulsion(v)
        wall_force = lambda v: soft_wall_repulsion(v, sdf, grad_x, grad_y, min_p, max_p)

        self.point_vmap = jax.vmap(jax.vmap(point_force))
        self.wall_vmap = jax.vmap(wall_force)
        self.drag_vmap = jax.vmap(drag)


    def generate_random_initial_state(self):
        state = super().generate_random_initial_state()
        return jnp.array(state)
    

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
        func = jax.jit(lambda _,y: self.calculate_derivatives(y))
        #func = lambda _,y: self.calculate_derivatives(y)
        solver = RK45(func, 0, y0=y0, t_bound=max_step * (steps + 1), max_step=max_step)

        if out:
            self.anim.out = out

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