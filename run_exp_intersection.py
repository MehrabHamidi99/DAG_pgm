import networkx as nx
import numpy as np

from Methods import *
from Evaluation import *
from DAG_generation import *

set_seed()

import argparse
import time
import multiprocessing 
import os
from functools import partial
import pickle
import argparse
import json 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    parser_arg = argparse.ArgumentParser(description='DAG Generation Hyperparameters')

    parser_arg.add_argument('--noise_types', metavar='noise_types', required=True,
                        help='', type=str, choices=['gauss', 'exp', 'gumbel', 'uniform'])

    parser_arg.add_argument('--graph_type', metavar='graph_type', required=True,
                        help='', type=str, default='er', choices=['er'])

    parser_arg.add_argument('--num_nodes', metavar='num_nodes', required=True, type=int)

    parser_arg.add_argument('--num_samples', metavar='num_samples', required=True, type=int)

    parser_arg.add_argument('--prob', metavar='prob', required=True, type=float)

    parser = vars(parser_arg.parse_args())

    graph_type = parser['graph_type']
    d = parser['num_nodes']
    n = parser['num_samples']
    prob = parser['prob']
    noise_types = parser['noise_types']

    save_path = f'mixed_data_experiment_{graph_type}/nodes_{d}_samples_{n}/prob_{prob}/{noise_types}'
    os.makedirs(save_path, exist_ok=True)

    # Step 1: Generate two different DAGs with the same nodes but different edges
    g_dag1, adj_dag1 = random_dag_generation(d, prob, graph_type)
    g_dag2, adj_dag2 = random_dag_generation(d, prob, graph_type)

    # Step 2: Generate datasets from both DAGs
    X1 = generate_single_dataset(g_dag1, n, noise_types, 1)
    X2 = generate_single_dataset(g_dag2, n, noise_types, 1)

    # Step 3: Combine the datasets into a mixed dataset
    X_mixed = np.vstack((X1, X2))

    # Step 4: Estimate a DAG from the mixed dataset
    W_est = notears_linear(X_mixed, lambda1=0.1, loss_type='l2')

    # Step 5: Compute the intersection of adj_dag1 and adj_dag2
    adj_intersection = np.logical_and(adj_dag1, adj_dag2).astype(int)

    # Save the true DAGs, intersection DAG, and estimated DAG
    np.savetxt(f'{save_path}/W_true_1.csv', adj_dag1, delimiter=',')
    np.savetxt(f'{save_path}/W_true_2.csv', adj_dag2, delimiter=',')
    np.savetxt(f'{save_path}/W_intersection.csv', adj_intersection, delimiter=',')
    np.savetxt(f'{save_path}/X_mixed.csv', X_mixed, delimiter=',')
    np.savetxt(f'{save_path}/W_est.csv', W_est, delimiter=',')

    # Step 6: Evaluate the estimated DAG against the intersection DAG
    acc = count_accuracy(adj_intersection, W_est != 0)

    # Save the results
    with open(f'{save_path}/res.txt', "w") as f:
        f.write(json.dumps(parser))
        f.write('\n#######\n#######\n#######\n')
        f.write(json.dumps(acc))

    # Optional: Plot the estimated DAG and intersection DAG
    # Create graphs from adjacency matrices
    G_est = nx.DiGraph(W_est != 0)
    G_intersection = nx.DiGraph(adj_intersection)

    # Plotting
    import matplotlib.pyplot as plt

    # Define a layout for consistent node positions
    pos = nx.spring_layout(G_est, seed=42)

    # Plot estimated DAG
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G_est, pos, node_size=300)
    nx.draw_networkx_edges(G_est, pos, arrowstyle='->', arrowsize=10)
    nx.draw_networkx_labels(G_est, pos)
    plt.title('Estimated DAG from Mixed Data')
    plt.savefig(f'{save_path}/Estimated_DAG.png')
    plt.close()

    # Plot intersection DAG
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G_intersection, pos, node_size=300)
    nx.draw_networkx_edges(G_intersection, pos, arrowstyle='->', arrowsize=10, edge_color='r')
    nx.draw_networkx_labels(G_intersection, pos)
    plt.title('Intersection DAG of True Graphs')
    plt.savefig(f'{save_path}/Intersection_DAG.png')
    plt.close()

    G_dag1 = nx.DiGraph(adj_dag1)
    G_dag2 = nx.DiGraph(adj_dag2)

     # Plot intersection DAG
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G_dag1, pos, node_size=300)
    nx.draw_networkx_edges(G_dag1, pos, arrowstyle='->', arrowsize=10)
    nx.draw_networkx_labels(G_dag1, pos)
    plt.title('DAG1')
    plt.savefig(f'{save_path}/DAG1.png')
    plt.close()

        # Plot intersection DAG
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G_dag2, pos, node_size=300)
    nx.draw_networkx_edges(G_dag2, pos, arrowstyle='->', arrowsize=10)
    nx.draw_networkx_labels(G_dag2, pos)
    plt.title('DAG2')
    plt.savefig(f'{save_path}/DAG2.png')
    plt.close()
