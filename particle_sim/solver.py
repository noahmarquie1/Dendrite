import numpy as np
from particle_sim.physics import PhysicsHandler
from particle_sim.animations import AnimationHandler
from scipy.integrate import RK45
import matplotlib.pyplot as plt
import shapely


class PointCloudSolver:
    def __init__(self, dpi=100, width=5, height=4, n_bodies=1,
                 force_multiplier=100, drag_coefficient=0.01,
                 pdfs=False, pdf_interval=0, polygon=None, 
                 fps=15, deg=2):

        self.solution = np.empty((0, 2))
        self.dpi = dpi
        self.width = width
        self.height = height
        self.n_bodies = n_bodies
        self.force_multiplier = force_multiplier
        self.drag_coefficient = drag_coefficient
        self.scatter = plt.scatter(np.zeros((self.n_bodies, 1)), np.zeros((self.n_bodies, 1)), c='blue', marker='o')
        self.polygon = polygon

        self.phys = PhysicsHandler(
            n_bodies = self.n_bodies,
            force_multiplier = self.force_multiplier,
            drag_coefficient = self.drag_coefficient,
            width = self.width,
            height = self.height,
            polygon=self.polygon,
            deg=deg
        )

        self.anim = AnimationHandler(
            n_bodies=self.n_bodies,
            width=self.width,
            height=self.height,
            polygon=polygon,
            dpi=self.dpi,
            pdfs=pdfs,
            pdf_interval=pdf_interval,
            fps=fps
        )


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
        return state


    def calculate_derivatives(self, state):
        state = state.reshape(-1, 2)
        num_bodies = int(state.shape[0] / 2)

        initial_positions = state[:num_bodies]
        initial_velocities = state[num_bodies:]
        res = np.zeros_like(state)

        for i in range(num_bodies):
            for j in range(i+1, num_bodies):
                paired_repulsion = self.phys.calculate_repulsive_force(this=initial_positions[i], other=initial_positions[j])
                res[num_bodies + i] += paired_repulsion
                res[num_bodies + j] -= paired_repulsion # same repulsion in opposite direction (all masses equal)
            #calculate repulsion from walls
            res[num_bodies + i] += self.phys.calculate_wall_force(this=initial_positions[i])
            #calculate drag force
            res[num_bodies + i] += self.phys.calculate_drag_force(vel=initial_velocities[i])

        # derivatives of each position are equal to velocity
        res[0:num_bodies] = initial_velocities
        return res.flatten()


    def solve(self, state0=None, max_step=0.05, steps=5):
        print("Beginning simulation.")
        state0 = self.generate_random_initial_state() if state0 is None else state0
        y0 = state0.flatten()
        self.solution = np.zeros((steps, state0.shape[0], state0.shape[1]))

        solver = RK45(
            lambda t, y: self.calculate_derivatives(state=y),
            0, y0=y0, t_bound=max_step * (steps + 1), max_step=max_step,
        )
        for i in range(steps):
            if (i + 1) % (steps / 10) == 0:
                print(f"Completed {i + 1}/{steps} steps. {(i+1)/steps*100:.1f}% complete.")
            solver.step()
            y = solver.y.reshape(state0.shape)
            self.solution[i] = y

        return self.solution


    def animate(self):
        if self.solution.shape[0] == 0:
            print("No solution to animate.")
            return

        print("Beginning animation.")
        self.anim.animate(self.solution)


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
        drag_coefficient=DRAG_COEF,
    )
    solver.solve(h=h, steps=steps)
    solver.animate()

