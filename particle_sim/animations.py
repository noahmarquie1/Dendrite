import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon
from shapely.geometry import Point
from mesh_generation.stats import plot_mesh_pdf, update_mesh_pdf
import seaborn as sns


class Graphic:
    def __init__(self, anim: bool, ax, func):
        self.ax = ax
        self.anim = anim
        if anim:
            self.func = lambda setup, idx: func(setup=setup, ax=ax, idx=idx)
        else:
            self.func = lambda: func(ax=ax)



class AnimationHandler:
    def __init__(self, n_bodies=1, width=6, height=4, dpi=100, fps=30, plots=None, polygon=None, vel_threshold=2):
        self.n_bodies = n_bodies
        self.dpi = dpi
        self.fps = fps
        self.polygon = polygon
        self.width = width
        self.height = height
        self.scatter = None
        self.sol = np.empty((0, n_bodies, 2))
        self.out = "./animation.mp4"

        self.plots = plots # list - allows "pdf-anim", "pdf-final", "max-vel" (max 3 allowed)
        self.graphics = []
        self.plot_interval = int(self.fps / 4)
        self.primary_ax = self.configure_plot()
        self.mesh_pdf_bars = None
        self.approx_dist = max(self.width, self.height) / np.sqrt(n_bodies) # rough approximate of ideal distance - used for plotting

        self.interval = 0
        self.frames = 0
        self.vel_threshold = vel_threshold


    def configure_plot(self):
        if self.plots:
            self.fig, self.ax = plt.subplots(2, 2, dpi=self.dpi)
            primary_ax = self.ax[0,0]
            primary_ax.set_aspect('equal')
            self.fig.set_size_inches(self.width * 2, self.height * 2)

            positions = [[0, 1], [1, 0], [1, 1]] 
            if 'pdf-anim' in self.plots:
                self.graphics.append(Graphic(anim=True, func=self.update_pdf_anim_graphic, ax=self.ax[positions[0][0], positions[0][1]]))
                positions = positions[1:]
            if 'pdf-final' in self.plots:
                self.graphics.append(Graphic(anim=False, func=self.setup_pdf_final_graphic, ax=self.ax[positions[0][0], positions[0][1]]))
                positions = positions[1:]
            if 'max-vel-static' in self.plots:
                self.graphics.append(Graphic(anim=False, func=self.setup_max_vel_static_graphic, ax=self.ax[positions[0][0], positions[0][1]]))
                positions = positions[1:]
            if 'max-vel-dynamic' in self.plots:
                self.graphics.append(Graphic(anim=True, func=self.update_max_vel_dynamic_graphic, ax=self.ax[positions[0][0], positions[0][1]]))
                positions = positions[1:]
            if 'pdf-comparison' in self.plots:
                self.graphics.append(Graphic(anim=False, func=self.setup_pdf_comparison_graphic, ax=self.ax[positions[0][0], positions[0][1]]))
                positions = positions[1:]

        else:
            self.fig, self.ax = plt.subplots(1, 1, dpi=self.dpi)
            primary_ax = self.ax
            self.fig.set_size_inches(self.width, self.height)
            self.ax.set_aspect('equal')
        
        primary_ax.set_xlim(self.polygon.bounds[0] - 0.1, self.polygon.bounds[2] + 0.1)
        primary_ax.set_ylim(self.polygon.bounds[1] - 0.1, self.polygon.bounds[3] + 0.1)
        primary_ax.set_title(f'{self.n_bodies}-Body Simulation')

        if self.polygon is not None:
            plot_polygon(self.polygon, ax=primary_ax, color='lightgray', edgecolor='black')

        return primary_ax
    

    def add_static_points(self, points: np.ndarray, color="blue"):
        min_x = min(min(points[:, 0]) - 0.1, self.polygon.bounds[0] - 0.1)
        max_x = max(max(points[:, 0]) + 0.1, self.polygon.bounds[2] - 0.1)
        min_y = min(min(points[:, 1]) - 0.1, self.polygon.bounds[1] - 0.1)
        max_y = max(max(points[:, 1]) + 0.1, self.polygon.bounds[3] - 0.1)

        self.primary_ax.set_xlim(min_x, max_x)
        self.primary_ax.set_ylim(min_y, max_y)
        self.primary_ax.scatter(points[:, 0], points[:, 1], c=color)


    def update_pdf_anim_graphic(self, setup: bool, ax, idx=None):
        if setup:
            ax.set_title("Mesh PDF Evolution")
            ax.set_xlabel("Distance")
            ax.set_ylabel("Probability")
            mesh_pdf_bars = plot_mesh_pdf(
                mesh_points=self.sol[0, :self.n_bodies, :], 
                approx_step=self.approx_dist, 
                ax=ax, 
                color='lightblue', 
                chart_details=False
            )
            ax.set_ylim(0, 1)
            self.mesh_pdf_bars = mesh_pdf_bars
        else:
            update_mesh_pdf(bars=self.mesh_pdf_bars, mesh_points=self.sol[idx*self.interval, :self.n_bodies, :], approx_step=self.approx_dist)
                

    def update_max_vel_dynamic_graphic(self, setup: bool, ax, idx=None):
        if setup and self.vel_threshold:
            self.setup_max_vel_static_graphic(ax)
            x = np.linspace(0, self.sol.shape[0], 2)
            y = np.full_like(x, self.vel_threshold)
            ax.fill_between(x, y, color="tab:blue", alpha=0.3)
            ax.plot(x, y, color="black")
            ax.scatter([0], [self.find_max_vel_at_idx(0)], c="red")
        else:
            scatter_point = ax.collections[1] # index of scatter point - after filled in box
            max_vel = self.find_max_vel_at_idx(idx*self.interval)
            point_color = 'green' if max_vel < self.vel_threshold else 'red'
            scatter_point.set_facecolors([point_color])
            scatter_point.set_offsets([[idx*self.interval, max_vel]])


    def get_dist_at_idx(self, idx):
        return plot_mesh_pdf(self.sol[idx, :self.n_bodies, :], approx_step=self.approx_dist) # will return x and y as arrays


    def setup_pdf_final_graphic(self, ax):
        ax.set_title("Final PDF")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        plot_mesh_pdf(self.sol[-1, :self.n_bodies, :], approx_step=self.approx_dist, ax=ax, color='lightblue', chart_details=False)


    def setup_pdf_comparison_graphic(self, ax):
        ax.set_title("Initial v. Final PDF")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)

        # plot initial distribution
        x, y = self.get_dist_at_idx(0)
        sns.histplot(
            x=x,
            weights=y,
            ax=ax,
            bins=x.shape[0],
            element='poly',
            label="Initial",
            color='red',
            alpha=0.3,
        )

        # plot final distribution
        x, y = self.get_dist_at_idx(-1)
        sns.histplot(
            x=x,
            weights=y,
            ax=ax,
            bins=x.shape[0],
            element='poly',
            label="Final",
            color='blue',
            alpha=0.3,
        )

        # plot legend
        ax.legend(loc="upper right")



    def find_max_vel_at_idx(self, i):
        vel_series = self.sol[i, self.n_bodies:, :]
        vel_series = [np.linalg.norm(vel) for vel in vel_series]
        max_vel = np.max(vel_series)
        return max_vel


    def setup_max_vel_static_graphic(self, ax):
        ax.set_title("Max Velocity vs. Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Velocity")

        t = range(self.frames)
        max_vels = np.array([self.find_max_vel_at_idx(i*self.interval) for i in t])
        t = np.array(t) * self.interval
        ax.plot(t, max_vels)


    def _animate(self, i):
        positions = self.sol[i*self.interval, :self.n_bodies, :]
        self.scatter.set_offsets(positions)
        self.info_text.set_text((
            f"Step: {i*self.interval}/{len(self.sol)}, {i*self.interval/len(self.sol)*100:.1f}%\n"
        ))

        if self.plots and i % self.plot_interval == 0:
            for graphic in self.graphics:
                graphic.func(setup=False, idx=i)
                
        return self.scatter,


    def animate(self, solution, color="blue"):
        self.sol = solution
        self.ax_map = {}
        self.scatter = self.primary_ax.scatter(solution[0, :self.n_bodies, 0], solution[0, :self.n_bodies, 1], c=color, marker='o')

        desired_length = 20
        self.frames = int(desired_length * self.fps) - 1
        self.interval = self.sol.shape[0] // self.frames
        self.interval = max(self.interval, 1)
        self.frames = self.sol.shape[0] // self.interval

        if self.interval == 0: # handles edge case for small animations
            self.interval = 1
            self.frames = self.sol.shape[0]

        for graphic in self.graphics:
            if graphic.anim:
                graphic.func(setup=True, idx=None)
            else:
                graphic.func()
        self.graphics = [graphic for graphic in self.graphics if graphic.anim] # retains only animated graphics for later use

        self.info_text = self.primary_ax.text(
            0.95, 0.95, '', transform=self.primary_ax.transAxes, 
            ha='right', va='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

        format = self.out.split('.')[1]
        writer = 'pillow' if format == 'gif' else 'ffmpeg'
        anim = FuncAnimation(self.fig, lambda i: self._animate(i), frames=self.frames)
        anim.save(filename=self.out, writer=writer, dpi=self.dpi, fps=self.fps)
        print(f"Animation saved as {self.out}.")
