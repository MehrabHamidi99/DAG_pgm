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

    save_path = f'no_tears_res_{graph_type}/nodes_{d}_samples_{n}/prob_{prob}/{noise_types}'
    os.makedirs(save_path, exist_ok=True)

    g_dag, adj_dag = random_dag_generation(d, prob, graph_type)
    X = generate_single_dataset(g_dag, n, noise_types, 1)

    np.savetxt(f'{save_path}/W_true.csv', adj_dag, delimiter=',')
    np.savetxt(f'{save_path}/X.csv', X, delimiter=',')
    W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
    # assert is_dag(W_est)
    np.savetxt(f'{save_path}/W_est.csv', W_est, delimiter=',')

    acc = count_accuracy(adj_dag, W_est != 0)
    f = open(f'{save_path}/res.txt', "w+")
    f.write(json.dumps(parser))
    f.write('\n#######\n#######\n#######\n')
    f.write(json.dumps(acc))
    f.close()