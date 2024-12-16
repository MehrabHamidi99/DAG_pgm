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
import argparse

def create_two_graphs_five_nodes():
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



def create_three_graphs_ten_nodes():
    nodes = list(range(10))  # 0, 1, ..., 9
    edges_1 = [
        (0, 1), (0, 2), (1, 3), (1, 4), (3, 5), (4, 5), (5, 6),
        (6, 7), (7, 8), (8, 9), (2, 6), (3, 7)
    ]
    edges_2 = [
        (0, 1), (0, 2), (1, 3), (1, 4), (3, 5), (5, 6), (6, 8),
        (6, 9), (2, 5), (4, 8), (3, 7), (7, 9), (8, 9)
    ]
    edges_3 = [
        (0, 1), (0, 3), (1, 3), (3, 5), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 9), (2, 6), (3, 7), (4, 8), (6, 9)
    ]

    G1 = nx.DiGraph()
    G1.add_nodes_from(nodes)
    G1.add_edges_from(edges_1)
    
    G2 = nx.DiGraph()
    G2.add_nodes_from(nodes)
    G2.add_edges_from(edges_2)
    
    G3 = nx.DiGraph()
    G3.add_nodes_from(nodes)
    G3.add_edges_from(edges_3)
    
    def ensure_dag(graph, name):
        if nx.is_directed_acyclic_graph(graph):
            print(f"{name} is a DAG")
        else:
            print(f"{name} is not a DAG. Removing cycles...")
            graph = nx.DiGraph([(u, v) for u, v in nx.topological_sort(graph)])
            if nx.is_directed_acyclic_graph(graph):
                print(f"{name} is now a DAG")
        return graph
    
    G1 = ensure_dag(G1, "Graph 1")
    G2 = ensure_dag(G2, "Graph 2")
    G3 = ensure_dag(G3, "Graph 3")
    
    A1 = nx.adjacency_matrix(G1).toarray()
    A2 = nx.adjacency_matrix(G2).toarray()
    A3 = nx.adjacency_matrix(G3).toarray()
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    nx.draw(G1, with_labels=True, font_weight='bold')
    plt.title("Graph 1")
    
    plt.subplot(132)
    nx.draw(G2, with_labels=True, font_weight='bold')
    plt.title("Graph 2")
    
    plt.subplot(133)
    nx.draw(G3, with_labels=True, font_weight='bold')
    plt.title("Graph 3")
    
    os.makedirs("data", exist_ok=True)
    
    plt.savefig("data/three_graphs_complex.pdf")
    plt.show()
    
    # Save the graphs as gpickle files
    with open('data/10_nodes_G1.gpickle', 'wb') as f:
        pickle.dump(G1, f)
    with open('data/10_nodes_G2.gpickle', 'wb') as f:
        pickle.dump(G2, f)
    with open('data/10_nodes_G3.gpickle', 'wb') as f:
        pickle.dump(G3, f)
    
    return (G1, A1), (G2, A2), (G3, A3)







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
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-graphs', type=int, default=3)
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    if args.mode == 'train':
        n_samples = 10000
    else:
        n_samples = 1000
    if args.num_graphs == 2:
        (graph1, A1), (graph2, A2) = create_two_graphs_five_nodes()
        (graph1, A1), (graph2, A2) = create_two_graphs_five_nodes()
        X1, sigmas1 = gaussian_anm(graph1, A1, n_samples)
        X2, sigmas2 = gaussian_anm(graph2, A2, n_samples)
        np.save("data/train_graph1.npy", X1)
        np.save("data/train_graph2.npy", X2)
        np.save("data/sigmas1.npy", sigmas1)
        np.save("data/sigmas2.npy", sigmas2)
    else:
        (graph1, A1), (graph2, A2), (graph3, A3) = create_three_graphs_ten_nodes()
        X1, sigmas1 = gaussian_anm(graph1, A1, n_samples)
        X2, sigmas2 = gaussian_anm(graph2, A2, n_samples)
        X3, sigmas3 = gaussian_anm(graph3, A3, n_samples)
        np.save(f"data/10_nodes_{args.mode}_graph1.npy", X1)
        np.save(f"data/10_nodes_{args.mode}_graph2.npy", X2)
        np.save(f"data/10_nodes_{args.mode}_graph3.npy", X3)
        np.save(f"data/10_nod'es_sigmas1.npy", sigmas1)
        np.save(f"data/10_nodes_sigmas2.npy", sigmas2)
        np.save(f"data/10_nodes_sigmas3.npy", sigmas3)

if __name__ == "__main__":
    main()