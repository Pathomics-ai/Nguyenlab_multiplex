import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA

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


