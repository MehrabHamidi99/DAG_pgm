import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import argparse
import time
import multiprocessing
import os
from functools import partial
import pickle
import argparse
import json

def show_graph_with_labels(adjacency_matrix, save_path):
    plt.cla()
    rows, cols = np.where(adjacency_matrix > 0)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()
    gr.add_edges_from(edges)

    nx.draw(gr, with_labels=True, node_size=500, node_color='lightblue', arrowsize=20)
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser(description='DAG Generation Hyperparameters')

    parser_arg.add_argument('--noise_types', metavar='noise_types', required=True,
                            help='', type=str, choices=['gauss', 'exp', 'gumbel', 'uniform'])
    parser_arg.add_argument('--graph_type', metavar='graph_type', required=True,
                            help='', type=str, default='er', choices=['er'])
    parser_arg.add_argument('--num_nodes', metavar='num_nodes', required=True, type=int)
    parser_arg.add_argument('--num_samples', metavar='num_samples', required=True, type=int)
    parser_arg.add_argument('--prob', metavar='prob', required=True, type=float)
    parser_arg.add_argument('--exp_mode', metavar='exp_mode', required=True, type=str)
    parser_arg.add_argument('--num_datasets', metavar='num_datasets', required=False, type=int, default=1,
                            help='Number of datasets to visualize')

    parser = vars(parser_arg.parse_args())

    graph_type = parser['graph_type']
    d = parser['num_nodes']
    n = parser['num_samples']
    prob = parser['prob']
    noise_types = parser['noise_types']
    exp_mode = parser['exp_mode']
    K = parser['num_datasets']

    save_path = f'{exp_mode}_{graph_type}/nodes_{d}_samples_{n}/prob_{prob}/{noise_types}'

    # Loop over each dataset index and visualize W_est_k and W_true_k
    for k in range(K):
        W_est_path = os.path.join(save_path, f'W_est_{k}.csv') if K > 1 else os.path.join(save_path, 'W_est.csv')
        W_true_path = os.path.join(save_path, f'W_true_{k}.csv') if K > 1 else os.path.join(save_path, 'W_true.csv')

        if not (os.path.exists(W_est_path) and os.path.exists(W_true_path)):
            print(f"Files for dataset {k} not found. Expected {W_est_path} and {W_true_path}. Skipping.")
            continue

        W_est = np.loadtxt(W_est_path, delimiter=',')
        show_graph_with_labels(W_est, os.path.join(save_path, f'w_est_{k}.png'))

        W_true = np.loadtxt(W_true_path, delimiter=',')
        show_graph_with_labels(W_true, os.path.join(save_path, f'w_true_{k}.png'))

    # Check if union-selected results exist
    W_est_union_selected_path = os.path.join(save_path, 'W_est_union_selected.csv')
    if os.path.exists(W_est_union_selected_path):
        W_est_union_selected = np.loadtxt(W_est_union_selected_path, delimiter=',')
        show_graph_with_labels(W_est_union_selected, os.path.join(save_path, 'w_est_union_selected.png'))
        print("Visualized W_est_union_selected.")

    # Check if full no-selection results exist
    W_est_full_no_selection_path = os.path.join(save_path, 'W_est_full_no_selection.csv')
    if os.path.exists(W_est_full_no_selection_path):
        W_est_full_no_selection = np.loadtxt(W_est_full_no_selection_path, delimiter=',')
        show_graph_with_labels(W_est_full_no_selection, os.path.join(save_path, 'w_est_full_no_selection.png'))
        print("Visualized W_est_full_no_selection.")
