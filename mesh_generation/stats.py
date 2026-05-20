from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mesh_generation.geometry import fetch_neighbors


def _calculate_pdf_data(mesh_points, approx_step, bin_amt):
    """Helper function to quickly calculate the PDF using NumPy vectorization."""
    mesh_tree = KDTree(mesh_points)
    
    # 1. Vectorized Query: Ask for all neighbors of all points simultaneously
    distances, _ = mesh_tree.query(mesh_points, k=8)
    
    # Flatten the array and mask out distances effectively equal to 0
    distances = distances.flatten()
    distances = distances[distances > 1e-8]
    
    max_dist = approx_step
    counts, _ = np.histogram(distances, bins=bin_amt, range=(0, max_dist))
    
    # Convert raw counts to probabilities
    if len(distances) > 0:
        probabilities = counts / len(distances)
    else:
        probabilities = np.zeros(bin_amt)
        
    return probabilities


def get_single_point_avg_dist(center, neighbors):
    dist_sum = 0
    for neighbor in neighbors:
        dist = np.linalg.norm(neighbor - center)
        dist_sum += dist
    return dist_sum / neighbors.shape[0]


def get_mesh_avg_dist(mesh_points):
    mesh_tree = KDTree(mesh_points)
    distances = np.array([])
    for point in mesh_points:
        neighbor_distances, _ = mesh_tree.query(point, k=8)
        distances = np.append(distances, neighbor_distances)
    return np.mean(distances)


def plot_single_point_pdf(p, mesh_points, approx_step, ax=None, color="lightblue"):
    neighbors = fetch_neighbors(p, mesh_points=mesh_points, n=8)
    distances = []
    for neighbor in neighbors:
        distance = np.sqrt((p[0]-neighbor[0])**2 + (p[1]-neighbor[1])**2)
        distances.append(distance)

    bin_amt = 20 # should be divisible by two (half below delta)
    total_samples = 0
    bin_size = approx_step / (bin_amt / 2)
    bins = np.zeros(bin_amt)
    for dist in distances:
        for i in range(bin_amt):
            if i*bin_size <= dist < (i+1)*bin_size:
                bins[i] += 1
                total_samples += 1

    # plot pdf
    if ax is None:
        ax = plt.gca()
    for i, bin in enumerate(bins):
        ax.bar(i*bin_size, bin / total_samples, width=bin_size, color=color, edgecolor="black")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.set_title("Single Point PDF")
    return ax


def plot_mesh_pdf(mesh_points, approx_step, bin_amt=20, ax=None, color="lightblue", chart_details=True):
    mesh_tree = KDTree(mesh_points)
    distances = np.array([])
    for point in mesh_points:
        neighbor_distances, _ = mesh_tree.query(point, k=8)
        mask = neighbor_distances > 1e-8
        distances = np.append(distances, neighbor_distances[mask])

    total_samples = 0
    bin_size = (approx_step) / bin_amt
    bins = np.zeros(bin_amt)
    for dist in distances:
        for i in range(bin_amt):
            if i * bin_size <= dist < (i + 1) * bin_size:
                bins[i] += 1
                total_samples += 1

    probabilities = _calculate_pdf_data(mesh_points, approx_step, bin_amt)
    max_dist = approx_step * 2
    bin_size = max_dist / bin_amt
    x_positions = np.arange(bin_amt) * bin_size

    # plot PDF
    if ax is None:
        return x_positions, probabilities
    
    bars = ax.bar(x_positions, probabilities, width=bin_size, color=color, edgecolor="black", align='edge')
    if chart_details:
        ax.set_xlabel("Distance")
        ax.set_ylabel("Count")
        ax.set_title("Mesh PDF")
        
    return bars


def update_mesh_pdf(bars, mesh_points, approx_step, bin_amt=20):
    """Calculates new data and instantly updates the heights of existing bars."""
    new_probabilities = _calculate_pdf_data(mesh_points, approx_step, bin_amt)
    
    # Instantly update the height of every existing rectangle
    for bar, new_height in zip(bars, new_probabilities):
        bar.set_height(new_height)
