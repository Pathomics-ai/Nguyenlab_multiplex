import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from anndata import AnnData
import pandas as pd
import numpy as np
import seaborn as sns
import colorcet as cc
import scanpy as sc
import igraph as ig
import colorcet as cc
import leidenalg
import umap
import copy
import os

def glasbey(n_colours):
    """ Return a list of n RGB tuples representing colours from the Glasbey colour map"""
    cm = cc.cm.glasbey_bw_minc_20
    return [cm(val)[:3] for val in np.linspace(0, 1, cm.N)][:n_colours]

def get_colours_from(colourmap_name, n):
    """ Return a list of n colours from the specified Matplotlib colourmap."""
    cmap = plt.get_cmap(colourmap_name)
    colours = [cmap(i / n) for i in range(n)]
    
    return colours

# function to plot the intensity of each marker on the dimensional reduction plots
def map_scatter(x, y, c, **kwargs):
    """ <x> and <y> are the names of the two dimensions obtained from dimensional reduction.
    <c> is the name of the marker of interest
    Pass in the color and dataframe as well.
    """
    df = kwargs.pop("data")
    marker = df['Marker'].iloc[0]  # Get the marker name for the current facet
    vmin = df[c].min()
    vmax = df[c].max()
    kwargs.pop("color", None)  # Remove 'color' from kwargs if it exists
    kwargs.pop("vmin", None)
    kwargs.pop("vmax", None)
    scatter = plt.scatter(df[x], df[y], c=df[c], vmin=vmin, vmax=vmax, **kwargs)
    plt.colorbar(scatter, ax=plt.gca(), label=f'{marker} Intensity')

def leiden_cluster(embedding, res):
    """ Return a numpy array <labels>, which contains the Leiden cluster labels for every data point in embedding. 
    Code from Shamini. 
    - embedding: a tuple of NDarrays
    - res: the resolution parameter. 
    """
    # make a k-nearest neighbors graph
    k = 15  # can adjust this value
    knn_graph = kneighbors_graph(embedding, n_neighbors=k, mode='connectivity', include_self=False)
    sources, targets = knn_graph.nonzero()
    edges = list(zip(sources.tolist(), targets.tolist()))

    # Create an igraph graph from the knn graph
    g = ig.Graph(edges=edges)

    # Perform Leiden clustering
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res)

    # Extract cluster labels
    labels = np.array(partition.membership)
    return labels

def plot_zoomed_triangulation_centered(ax, points, filtered_edges, title, zoom_factor=0.01):
    """
    Plot a zoomed-in section of the Delaunay triangulation with filtered edges (2D array).

    Parameters:
    ax: matplotlib axis object to plot on
    points: numpy array of shape (n, 2), representing the coordinates of points
    filtered_edges: list of edges to plot, where each edge is a tuple of two point indices
    title: title of the plot
    zoom_factor: float, fraction of the graph to zoom in on, centered at the middle
    """
    # zoom in on center of the graph
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    width = (max_x - min_x) * zoom_factor
    height = (max_y - min_y) * zoom_factor

    x_min_zoom = center_x - width / 2
    x_max_zoom = center_x + width / 2
    y_min_zoom = center_y - height / 2
    y_max_zoom = center_y + height / 2

    # filter points within the zoomed bounding box
    zoomed_points = points[
        (points[:, 0] >= x_min_zoom) & (points[:, 0] <= x_max_zoom) &
        (points[:, 1] >= y_min_zoom) & (points[:, 1] <= y_max_zoom)
    ]
    
    # map original points to their indices
    point_index_map = {tuple(point): idx for idx, point in enumerate(points)}
    zoomed_point_indices = {point_index_map[tuple(point)] for point in zoomed_points}

    # filter edges within the zoomed area
    zoomed_edges = [
        edge for edge in filtered_edges
        if edge[0] in zoomed_point_indices and edge[1] in zoomed_point_indices
    ]

    # plot
    for edge in zoomed_edges:
        x_values = [points[edge[0]][0], points[edge[1]][0]]
        y_values = [points[edge[0]][1], points[edge[1]][1]]
        ax.plot(x_values, y_values, 'r-', lw=1)
    ax.scatter(zoomed_points[:, 0], zoomed_points[:, 1], c='blue', s=20)

    # set plot limits to the zoomed area
    ax.set_xlim(x_min_zoom, x_max_zoom)
    ax.set_ylim(y_min_zoom, y_max_zoom)
    ax.set_title(title)

def plot_zoomed_triangulation(ax, points, filtered_edges, title, x_min_zoom, x_max_zoom, y_min_zoom, y_max_zoom):
    """
    Plot a zoomed-in section of the Delaunay triangulation with filtered edges (2D array).

    Parameters:
    ax: matplotlib axis object to plot on
    points: numpy array of shape (n, 2), representing the coordinates of points
    filtered_edges: list of edges to plot, where each edge is a tuple of two point indices
    title: title of the plot
    x_min_zoom: minimum x value
    WLOG for x_max_zoom, y_min_zoom, y_max_zoom
    """
    # filter points within the zoomed bounding box
    zoomed_points = points[
        (points[:, 0] >= x_min_zoom) & (points[:, 0] <= x_max_zoom) &
        (points[:, 1] >= y_min_zoom) & (points[:, 1] <= y_max_zoom)
    ]
    
    # map original points to their indices
    point_index_map = {tuple(point): idx for idx, point in enumerate(points)}
    zoomed_point_indices = {point_index_map[tuple(point)] for point in zoomed_points}

    # filter edges within the zoomed area
    zoomed_edges = [
        edge for edge in filtered_edges
        if edge[0] in zoomed_point_indices and edge[1] in zoomed_point_indices
    ]

    # plot
    for edge in zoomed_edges:
        x_values = [points[edge[0]][0], points[edge[1]][0]]
        y_values = [points[edge[0]][1], points[edge[1]][1]]
        ax.plot(x_values, y_values, 'r-', lw=1)
    ax.scatter(zoomed_points[:, 0], zoomed_points[:, 1], c='blue', s=20)

    # set plot limits to the zoomed area
    ax.set_xlim(x_min_zoom, x_max_zoom)
    ax.set_ylim(y_min_zoom, y_max_zoom)
    ax.set_title(title)


def plot_filtered_triangulation(ax, points, filtered_edges, title):
    """ Plot the Delaunay triangulation with filtered edges (2D array) """
    for edge in filtered_edges:
        x_values = [points[edge[0]][0], points[edge[1]][0]]
        y_values = [points[edge[0]][1], points[edge[1]][1]]
        ax.plot(x_values, y_values, 'r-', lw=1)
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=20)
    ax.set_title(title)

def filter_edges(tri, distance_cutoff, phenotype_mask):
    """ Filter edges based on a single distance cutoff and phenotype mask. """
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edges.add(tuple(sorted([simplex[i], simplex[(i + 1) % 3]])))
    
    points = tri.points
    filtered_edges = []

    # create a mask to determine valid vertices
    valid_vertices = set()
    for vertex_index in range(len(points)):
        if phenotype_mask[vertex_index]:
            valid_vertices.add(vertex_index)

    for edge in edges:
        if edge[0] in valid_vertices and edge[1] in valid_vertices:
            p1, p2 = points[edge[0]], points[edge[1]]
            distance = np.linalg.norm(p1 - p2)
            if distance <= distance_cutoff:
                filtered_edges.append(edge)
    
    return filtered_edges

def analyze_slide(df, distance_cutoffs, phenotypes_oi, boundaries: dict):
    """ Analyze each slide, perform triangulation, filter edges for each cutoff, and plot within boundaries. 
    <boundaries> should map strings from the Parent column to a 4-element tuple of bounds with 
    the following format: (x_min, x_max, y_min, y_max)
    """
    grouped = df.groupby('Parent')
    
    for name, group in grouped:
        coords = group[['Centroid X µm', 'Centroid Y µm']].values
        phenotypes = group['Phenotype'].values

        phenotype_mask = np.array([phenotype in phenotypes_oi for phenotype in phenotypes])
        
        # Delaunay triangulation
        tri = Delaunay(coords)
        
        # plot for each distance cutoff
        num_cutoffs = len(distance_cutoffs)
        fig, axs = plt.subplots(1, num_cutoffs, figsize=(15, 5))
        if num_cutoffs == 1:
            axs = [axs]
        
        for i, cutoff in enumerate(distance_cutoffs):
            filtered_edges = filter_edges(tri, cutoff, phenotype_mask) # get filtered graph
            x_min, x_max, y_min, y_max = boundaries[name]
            plot_zoomed_triangulation(ax=axs[i], 
                                      points=coords, 
                                      filtered_edges=filtered_edges, 
                                      title=f'Cutoff: {cutoff} µm', 
                                      x_min_zoom=x_min,
                                      x_max_zoom=x_max,
                                      y_min_zoom=y_min,
                                      y_max_zoom=y_max)
        
        plt.suptitle(f'Slide ID: {name}')
        plt.show()

def analyze_slide_zoomed_out(df, distance_cutoffs, phenotypes_oi):
    """ Analyze each slide, perform triangulation, filter edges for each cutoff, and plot. """
    grouped = df.groupby('Parent')
    
    for name, group in grouped:
        coords = group[['Centroid X µm', 'Centroid Y µm']].values
        phenotypes = group['Phenotype'].values

        phenotype_mask = np.array([phenotype in phenotypes_oi for phenotype in phenotypes])
        
        # Delaunay triangulation
        tri = Delaunay(coords)
        
        # plot for each distance cutoff
        num_cutoffs = len(distance_cutoffs)
        fig, axs = plt.subplots(1, num_cutoffs, figsize=(15, 5))
        if num_cutoffs == 1:
            axs = [axs]
        
        for i, cutoff in enumerate(distance_cutoffs):
            filtered_edges = filter_edges(tri, cutoff, phenotype_mask) # get filtered graph
            plot_filtered_triangulation(axs[i], coords, filtered_edges, f'Cutoff: {cutoff} µm')
        
        plt.suptitle(f'Slide ID: {name}')
        plt.show()

