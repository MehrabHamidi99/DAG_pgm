import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import time
import sys
import os
import math
import copy
import igraph
import pickle
from sklearn.gaussian_process.kernels import RBF

def create_graphs():
    nodes = list(range(5)) # 0, 1, 2, 3, 4
    edges =[(0,1),(0,4),(1,2),(1,3),(2,4),(4,3)]
    edges_2 = [(0,1),(0,2),(2,3),(1,3),(2,3),(3,4)]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    G2 = nx.DiGraph()
    G2.add_nodes_from(nodes)
    G2.add_edges_from(edges_2)
    #check if no cycles
    if nx.is_directed_acyclic_graph(G2):
        print("Graph 2 is a DAG")
    else:
        print("Graph 2 is not a DAG")
    if nx.is_directed_acyclic_graph(G):
        print("Graph 1 is a DAG")
    else:
        print("Graph 1 is not a DAG")
    A = nx.adjacency_matrix(G).toarray()
    A2 = nx.adjacency_matrix(G2).toarray()
    plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.title("Graph 1")
    plt.subplot(122)
    nx.draw(G2, with_labels=True, font_weight='bold')
    plt.title("Graph 2")
    plt.savefig("data/graphs.png")
    with open('data/G1.gpickle', 'wb') as f:
        pickle.dump(G, f)
    with open('data/G2.gpickle', 'wb') as f:
        pickle.dump(G2, f)
    return (G, A), (G2, A2)



def load_graphs(g_path):
    with open(g_path, 'rb') as f:
        G = pickle.load(f)
    A = nx.adjacency_matrix(G).toarray()
    return G, A

def gaussian_anm(G,A,n_points,sem_type='non_linear_gaussian',sigma_min=0.4,sigma_max=0.8):
    n,d = n_points, A.shape[0]
    #generate random data according to the non linear gaussian additive model
    X = np.zeros((n,d))
    parents_dict = {
    node: [parent for parent, is_parent in zip(G.nodes, A[:, i]) if is_parent] for i, node in enumerate(G.nodes)}
    ordered_vertices = list(nx.topological_sort(G))
    if sem_type == 'non_linear_gaussian':
        sigmas = {}
        kernel = RBF(length_scale=1.0)
        for i in ordered_vertices:
            parents = parents_dict[i]
            if len(parents) == 0:
                sigma_i = random.uniform(1,2)
                X[:, i] = np.random.normal(0, 1, n)
            else:
                sigma_i = random.uniform(sigma_min, sigma_max)
                noise_i = np.random.normal(0, sigma_i, n)
                parent_values = X[:, parents]
                K = kernel(parent_values, parent_values) + np.eye(n)*1e-9
                fj = np.random.multivariate_normal(np.zeros(n), K)
                X[:, i] = fj + noise_i
            sigmas[i] = sigma_i
    return X,sigmas





def main():
    seed = 40
    np.random.seed(seed)
    random.seed(seed)
    (graph1, A1), (graph2, A2) = create_graphs()
    X1, sigmas1 = gaussian_anm(graph1, A1, 10000)
    X2, sigmas2 = gaussian_anm(graph2, A2, 10000)
    np.save("data/train_graph1.npy", X1)
    np.save("data/train_graph2.npy", X2)
    np.save("data/sigmas1.npy", sigmas1)
    np.save("data/sigmas2.npy", sigmas2)


if __name__ == "__main__":
    main()