import networkx as nx
import numpy as np
import argparse
import time
import multiprocessing 
import os
from functools import partial
import pickle
import json 

from Methods import *
from Evaluation import *
from DAG_generation import *
from knockoff_alg import *

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

    parser_arg.add_argument('--noise_types', required=True, choices=['gauss', 'exp', 'gumbel', 'uniform'], type=str)
    parser_arg.add_argument('--graph_type', required=True, type=str, default='er', choices=['er'])
    parser_arg.add_argument('--num_nodes', required=True, type=int)
    parser_arg.add_argument('--num_samples', required=True, type=int)
    parser_arg.add_argument('--prob', required=True, type=float)
    parser_arg.add_argument('--num_datasets', required=False, type=int, default=2, help='Number of related datasets')
    parser_arg.add_argument('--exp_mode', metavar='exp_mode', required=False, type=str, default='knockoff')

    parser = vars(parser_arg.parse_args())

    graph_type = parser['graph_type']
    d = parser['num_nodes']
    n = parser['num_samples']
    prob = parser['prob']
    noise_types = parser['noise_types']
    K = parser['num_datasets']

    exp_mode = parser['exp_mode']

    save_path = f'{exp_mode}_{graph_type}/nodes_{d}_samples_{n}/prob_{prob}/{noise_types}'
    os.makedirs(save_path, exist_ok=True)

    # Generate K datasets and their true DAGs
    dags = []
    X_list = []
    for k in range(K):
        g_dag, adj_dag = random_dag_generation(d, prob, graph_type)
        X = generate_single_dataset(g_dag, n, noise_types, 1)
        dags.append(adj_dag)
        X_list.append(X)
        np.savetxt(f'{save_path}/W_true_{k}.csv', adj_dag, delimiter=',')
        np.savetxt(f'{save_path}/X_{k}.csv', X, delimiter=',')

    # ---------------------------
    # Approach 1: Knockoff + NO TEARS on Union of Selected Features
    # ---------------------------

    # Perform knockoff filtering on each dataset
    S_list = []
    q = 0.6
    for k in range(K):
        S_k = perform_knockoff_filtering(X_list[k], q=q)
        S_list.append(set(S_k))

    # Compute the union of all selected variables
    S_union = set.union(*S_list)
    S_union = sorted(list(S_union))  # Convert to a sorted list for indexing

    # Restrict each dataset to the unioned variables and combine
    X_union_list = [X[:, S_union] for X in X_list]
    # Stack all datasets vertically to form one combined dataset
    X_combined_selected = np.vstack(X_union_list)  # shape: (K*n) x |S_union|

    # Time NO TEARS on the reduced dataset
    start_time_knockoff = time.time()
    W_est_restricted = notears_linear(X_combined_selected, lambda1=0.1, loss_type='l2')
    end_time_knockoff = time.time()
    notears_time_knockoff = end_time_knockoff - start_time_knockoff

    # Map W_est_restricted back to full set of variables
    p_union = len(S_union)
    W_est_full_selected = np.zeros((d, d))
    for i, vi in enumerate(S_union):
        for j, vj in enumerate(S_union):
            W_est_full_selected[vi, vj] = W_est_restricted[i, j]

    np.savetxt(f'{save_path}/W_est_union_selected.csv', W_est_full_selected, delimiter=',')

    # Evaluate accuracy on each dataset
    results_selected = {}
    for k in range(K):
        acc = count_accuracy(dags[k], W_est_full_selected != 0)
        results_selected[k] = acc

    with open(f'{save_path}/results_union_selected.txt', "w") as f:
        f.write(json.dumps(results_selected, indent=2))

    # ---------------------------
    # Approach 2: NO TEARS on Union of All Variables (No Knockoff)
    # ---------------------------

    # Combine all datasets over the full set of variables
    X_combined_full = np.vstack(X_list)  # shape: (K*n) x d

    # Time NO TEARS on the full dataset
    start_time_full = time.time()
    W_est_full = notears_linear(X_combined_full, lambda1=0.1, loss_type='l2')
    end_time_full = time.time()
    notears_time_full = end_time_full - start_time_full

    np.savetxt(f'{save_path}/W_est_full_no_selection.csv', W_est_full, delimiter=',')

    results_full = {}
    for k in range(K):
        acc = count_accuracy(dags[k], W_est_full != 0)
        results_full[k] = acc

    with open(f'{save_path}/results_full_no_selection.txt', "w") as f:
        f.write(json.dumps(results_full, indent=2))

    # ---------------------------
    # Compare Results and Times
    # ---------------------------
    comparison = {
        "knockoff_selected_results": results_selected,
        "full_no_selection_results": results_full,
        "notears_time_knockoff_selected": notears_time_knockoff,
        "notears_time_full_no_selection": notears_time_full
    }

    with open(f'{save_path}/comparison.txt', "w") as f:
        f.write(json.dumps(comparison, indent=2))

    print("Comparison Results:")
    print(json.dumps(comparison, indent=2))