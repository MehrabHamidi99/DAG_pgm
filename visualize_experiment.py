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
    
    nx.draw(gr)
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

    parser = vars(parser_arg.parse_args())

    graph_type = parser['graph_type']
    d = parser['num_nodes']
    n = parser['num_samples']
    prob = parser['prob']
    noise_types = parser['noise_types']

    save_path = f'no_tears_res_{graph_type}/nodes_{d}_samples_{n}/prob_{prob}/{noise_types}'

    W_est = np.loadtxt(f'{save_path}/W_est.csv', delimiter=',')
    show_graph_with_labels(W_est, f'{save_path}/w_est.png')

    W_true = np.loadtxt(f'{save_path}/W_true.csv', delimiter=',')
    show_graph_with_labels(W_true, f'{save_path}/w_true.png')

