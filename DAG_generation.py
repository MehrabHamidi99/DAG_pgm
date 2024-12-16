from abc import abstractmethod
import os
import re
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple
import random

# import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# import torch
# from torch_geometric.utils import to_dense_adj


def set_seed(seed=33):
    random.seed(seed)
    np.random.seed(seed)


def random_dag_generation(d: int, edge_prob: float, graph_mode: str) -> nx.DiGraph:
    '''
    TODO: expand ways in which we generate adjacency matrices for DAGs, currently a very narrow definition
    '''
    adj_matrix = None
    if graph_mode == 'er':
        # prob = float(degree) / (d - 1)
        adj_matrix = np.tril((np.random.rand(d, d) < edge_prob).astype(float), k=-1)
        # graph_edges = nx.erdos_renyi_graph(d, edge_prob).edges
    
    dag_graph = nx.DiGraph(adj_matrix)

    assert nx.is_directed_acyclic_graph(dag_graph)
    return dag_graph, adj_matrix

# TODO: generated weighted adjacency matrix from regular adjacency matrix
def generate_weighted_adjacency_matrix(B, w_ranges):
    W = None
    return W

def visualize_graph(G: nx.Graph):
    edge_labels = nx.get_edge_attributes(G,'weight')

    pos = nx.spring_layout(G)

    ax = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

    return ax


def simulate_single_equation(X, w, scale, noise):
    '''
    TODO: maybe add more types of noise?
    '''
    n = X.shape[0]
    if noise == "gauss":
        z = np.random.normal(scale = scale, size = n)
    elif noise == "exp":
        z = np.random.exponential(scale = scale, size = n)
    elif noise == "gumbel":
        z = np.random.gumbel(scale = scale, size = n)
    elif noise == "uniform":
        z = np.random.uniform(low = -scale, high = scale, size = n)
    
    return (X @ w) + z
    
def simulate_variable(G: nx.Graph, n: int, noise_type: str, noise_scale: float = 1.0, zeros:bool = False):

    W = nx.to_numpy_array(G)
    d = W.shape[0]
    if zeros:
        X = np.zeros([n, d])
    else:
        X = np.random.randn(n, d)
    scale_vec = np.ones(d) * noise_scale

    topological_order = list(nx.topological_sort(G))

    for node in topological_order:
        ancesters = list(G.predecessors(node)) #p
        X[:, node] = simulate_single_equation(X[:, ancesters], W[ancesters, node], scale_vec[node], noise_type)

    return X, W

def generate_single_dataset(G: nx.Graph, n: int, noise_type: str, noise_scale: float):
    return simulate_variable(G, n, noise_type, noise_scale)
    
def mix_datasets(datasets: List):
    '''
    The idea is to generate different datasets with generate_single_dataset() [potentially different noise types with the same underlying graph, 
    or the same noise type with different graphs (could be a different weighted adjacency matrix but the same adjacency matrix, or a completely different
    adjacency matrix)].
    '''
    datasets = np.array(datasets)
    d = datasets[0].shape[0]
    for dataset in datasets:
        assert dataset.shape[0] == d # ensure all graphs in the dataset have the same number of nodes
    datasets.reshape(-1, d)
    np.random.shuffle(datasets)
    return datasets

