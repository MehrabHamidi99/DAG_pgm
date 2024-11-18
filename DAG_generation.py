from abc import abstractmethod
import os
import re
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# import torch
# from torch_geometric.utils import to_dense_adj


def set_seed(seed=33):
    random.seed(seed)
    np.random.seed(seed)


def random_dag_generation(d: int, edge_prob: float, graph_mode: str) -> nx.DiGraph:
    
    adj_matrix = None
    if graph_mode == 'er':
        # prob = float(degree) / (d - 1)
        adj_matrix = np.tril((np.random.rand(d, d) < edge_prob).astype(float), k=-1)
        # graph_edges = nx.erdos_renyi_graph(d, edge_prob).edges
    
    dag_graph = nx.DiGraph(adj_matrix)



    assert nx.is_directed_acyclic_graph(dag_graph)
    return dag_graph, adj_matrix

def visualize_graph(G: nx.Graph):
    edge_labels = nx.get_edge_attributes(G,'weight') # key is edge, pls check for your case

    ax = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

