import numpy as np
import pandas as pd
import knockpy
from knockpy import KnockoffFilter
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from DAG_generation import random_dag_generation, generate_single_dataset
from Methods import notears_linear, notears_knockoff_regularizer
from Evaluation import count_accuracy
from Visualization import visualize_dag, visualize_adjacency_matrices

def edges_to_adj_matrix(edges, p):
    """Convert a list of edges to an adjacency matrix."""
    adj_matrix = np.zeros((p, p))
    for i, j in edges:
        adj_matrix[i, j] = 1
    return adj_matrix

def save_results_and_plots(W_true_union, W_true_intersection, pred_list, save_dir):
    """Save results, plots, and metrics to the specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    notears_edges, notears_knockoff_reg_edges = pred_list
    predictions = [notears_edges, notears_knockoff_reg_edges]
    titles = ["NO TEARS Learned DAG", "NO TEARS with Knockoff Regularizer"]

    # Visualize and save DAGs
    visualize_dag(W_true_union, predictions, titles, save_path=os.path.join(save_dir, "dag_comparison_union.png"))
    visualize_dag(W_true_intersection, predictions, titles, save_path=os.path.join(save_dir, "dag_comparison_intersection.png"))

    # Visualize and save adjacency matrices
    visualize_adjacency_matrices(W_true_union, predictions, titles, save_path=os.path.join(save_dir, "adjacency_heatmaps_union.png"))
    visualize_adjacency_matrices(W_true_intersection, predictions, titles, save_path=os.path.join(save_dir, "adjacency_heatmaps_intersection.png"))

    # Compute and save metrics
    metrics_union = [count_accuracy(W_true_union, edges_to_adj_matrix(edges, W_true_union.shape[0])) for edges in predictions]
    metrics_intersection = [count_accuracy(W_true_intersection, edges_to_adj_matrix(edges, W_true_intersection.shape[0])) for edges in predictions]

    with open(os.path.join(save_dir, "res.txt"), "w") as f:
        for title, metrics_u, metrics_i in zip(titles, metrics_union, metrics_intersection):
            f.write(f"{title} (Union Ground Truth):\n{metrics_u}\n\n")
            f.write(f"{title} (Intersection Ground Truth):\n{metrics_i}\n\n")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="DAG Learning with Knockoff and NO TEARS")
    parser.add_argument("--samples", type=int, required=True, help="Number of samples per graph")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes (variables)")
    parser.add_argument("--fdr", type=float, required=True, help="Target false discovery rate")
    parser.add_argument("--graph_type", type=str, required=True, choices=['er', 'ba', 'ws'], help="Graph type")
    parser.add_argument("--graph_prob", type=float, required=True, help="Graph probability")
    parser.add_argument("--noise_type", type=str, required=True, choices=['gauss', 'exp', 'gumbel', 'uniform'], help="Noise type")
    parser.add_argument("--noise_scale", type=float, required=True, help="Noise scale")
    parser.add_argument("--lambda_val", type=float, required=True, help="Lambda value for NO TEARS")
    parser.add_argument("--groups", type=int, required=True, help="Number of groups (graphs) to generate")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Assign variables from arguments
    n = args.samples
    p = args.nodes
    fdr = args.fdr
    graph_type = args.graph_type
    graph_prob = args.graph_prob
    noise_type = args.noise_type
    noise_scale = args.noise_scale
    lambda_val = args.lambda_val
    groups = args.groups
    seed = args.seed

    # Set random seed
    np.random.seed(seed)

    # Generate g graphs
    datasets, true_dags = [], []
    for _ in range(args.groups):
        g_dag, adj_dag = random_dag_generation(p, graph_prob, graph_type)
        X, _ = generate_single_dataset(g_dag, n, noise_type, noise_scale, zeros=False, adj=True)
        datasets.append(X)
        true_dags.append(adj_dag)

    # Combine datasets and compute union and intersection of ground truth graphs
    X_combined = np.vstack(datasets)
    W_true_union = np.maximum.reduce(true_dags)
    W_true_intersection = np.minimum.reduce(true_dags)

    # NO TEARS
    W_notears = notears_linear(X_combined, lambda1=args.lambda_val, loss_type='l2')
    notears_edges = [(i, j) for i in range(args.nodes) for j in range(args.nodes) if W_notears[i, j] != 0]

    # NO TEARS with Knockoff Regularizer
    W_notears_knockoff_reg = notears_knockoff_regularizer(X_combined, lambda1=args.lambda_val, loss_type='l2')
    notears_knockoff_reg_edges = [(i, j) for i in range(args.nodes) for j in range(args.nodes) if W_notears_knockoff_reg[i, j] != 0]

    # Define save directory
    save_dir = f"results_multiple/notears_knockoff_regularizer/groups_{args.groups}_samples_{args.samples}_nodes_{args.nodes}_fdr_{args.fdr}_seed_{args.seed}"
    
    # Save results and plots
    save_results_and_plots(W_true_union, W_true_intersection, (notears_edges, notears_knockoff_reg_edges), save_dir)
