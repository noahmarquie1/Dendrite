import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon
from shapely.geometry import Point
from mesh_generation.stats import plot_mesh_pdf, update_mesh_pdf

class AnimationHandler:
    def __init__(self, n_bodies=1, width=6, height=4, dpi=100, fps=30, pdfs=True, pdf_interval=1, polygon=None):
        self.n_bodies = n_bodies
        self.dpi = dpi
        self.fps = fps
        self.polygon = polygon
        self.width = width
        self.height = height
        self.scatter = None
        self.solution = np.empty((0, n_bodies, 2))
        self.pdfs = pdfs
        self.pdf_interval = pdf_interval
        self.primary_ax = self.configure_plot()

        self.approx_dist = max(self.width, self.height) / np.sqrt(n_bodies) # rough approximate of ideal distance

    def configure_plot(self):
        if self.pdfs:
            self.fig, self.ax = plt.subplots(2, 2, dpi=self.dpi)
            primary_ax = self.ax[0,0]
            primary_ax.set_aspect('equal')
            self.fig.set_size_inches(self.width * 2, self.height * 2)
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
    

    def avg_dist_deviation(self, i):
        diff = self.solution[i, :self.n_bodies, :] - self.solution[i-1, :self.n_bodies, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        avg_dist = np.mean(dists)
        return avg_dist


    def _animate(self, i, interval, mesh_pdf_bars=None):
        positions = self.solution[i*interval, :self.n_bodies, :]
        self.scatter.set_offsets(positions)
        self.info_text.set_text((
            f"Step: {i*interval}/{len(self.solution)}, {i*interval/len(self.solution)*100:.1f}%\n"
        ))
        if self.pdfs and (i+1) % self.pdf_interval == 0:
            update_mesh_pdf(bars=mesh_pdf_bars, mesh_points=positions, approx_step=self.approx_dist)
                
        return self.scatter,


    def animate(self, solution):
        self.solution = solution
        mesh_pdf_bars = None

        self.scatter = self.primary_ax.scatter(solution[0, :self.n_bodies, 0], solution[0, :self.n_bodies, 1], c='blue', marker='o')
        if self.pdfs:
            # Plot initial PDF in evolution chart
            self.ax[0,1].set_title("Mesh PDF Evolution")
            self.ax[0,1].set_xlabel("Distance")
            self.ax[0,1].set_ylabel("Probability")
            mesh_pdf_bars = plot_mesh_pdf(solution[0, :self.n_bodies, :], approx_step=self.approx_dist, ax=self.ax[0,1], color='lightblue', chart_details=False)
            self.ax[0,1].set_ylim(0, 1)
            # Plot distance evolution
            dist_evolution_points = [i*self.pdf_interval + 1 for i in range(int(len(solution) / self.pdf_interval) - 1)]
            self.ax[1,0].plot(dist_evolution_points, [self.avg_dist_deviation(i) for i in dist_evolution_points], color='green')
            self.ax[1,0].set_title("Rate of Change of Positions vs. Time")
            self.ax[1,0].set_xlabel("Time")
            self.ax[1,0].set_ylabel("Average Distance Deviation")

            # Plot final pdf in bottom left
            self.ax[1,1].set_title("Final PDF")
            self.ax[1,1].set_xlabel("Distance")
            self.ax[1,1].set_ylabel("Probability")
            plot_mesh_pdf(solution[-1, :self.n_bodies, :], approx_step=self.approx_dist, ax=self.ax[1,1], color='lightblue', chart_details=False)
            self.ax[1,1].set_ylim(0, 1)

        self.info_text = self.primary_ax.text(0.95, 0.95, '', transform=self.primary_ax.transAxes, 
                    ha='right', va='top', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        desired_length = 20
        frames = desired_length * self.fps
        interval = int(self.solution.shape[0] / frames)
        out = './animation.mp4'

        anim = FuncAnimation(self.fig, lambda i: self._animate(i, interval, mesh_pdf_bars), frames=frames)
        anim.save(out, writer='ffmpeg', dpi=self.dpi, fps=self.fps)
        print(f"Animation saved as {out}.")
