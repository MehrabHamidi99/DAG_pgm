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
from Methods import notears_linear
from Evaluation import count_accuracy
from Visualization import visualize_dag, visualize_adjacency_matrices

def knockoff_feature_selection(X, y, fdr=0.1):
    """Select features using knockoff filtering."""
    print("\nPerforming feature selection using knockoff filtering...")
    Sigma = knockpy.dgp.AR1(p=X.shape[1], rho=0.1)
    kfilter = KnockoffFilter(ksampler='gaussian', fstat='lasso')
    rejections = kfilter.forward(X=X, y=y, Sigma=Sigma, fdr=fdr)
    selected_features = np.where(rejections == 1)[0]
    return selected_features

def nodewise_knockoff_selection(X, fdr=0.1):
    """Perform knockoff-based parent selection for each node."""
    n, p = X.shape
    selected_edges = []
    ancestor_sets = [set() for _ in range(p)]
    for j in range(p):
        predictors_indices = [k for k in range(p) if k != j and j not in ancestor_sets[k]]
        if not predictors_indices:
            continue
        predictors = X[:, predictors_indices]
        target = X[:, j]
        Sigma = knockpy.dgp.AR1(p=predictors.shape[1], rho=0.5)
        kfilter = KnockoffFilter(ksampler='gaussian', fstat='lasso')
        rejections = kfilter.forward(X=predictors, y=target, Sigma=Sigma, fdr=fdr)
        for idx, reject in enumerate(rejections):
            if reject == 1 and not creates_cycle(selected_edges, predictors_indices[idx], j, p):
                selected_edges.append((predictors_indices[idx], j))
                ancestor_sets[j].add(predictors_indices[idx])
    return selected_edges

def creates_cycle(edges, parent, child, p):
    """Check if adding the edge (parent -> child) creates a cycle."""
    G = nx.DiGraph()
    G.add_edges_from(edges)
    G.add_edge(parent, child)
    try:
        nx.find_cycle(G, orientation='original')
        return True
    except nx.NetworkXNoCycle:
        return False

def edges_to_adj_matrix(edges, p):
    """Convert a list of edges to an adjacency matrix."""
    adj_matrix = np.zeros((p, p))
    for i, j in edges:
        adj_matrix[i, j] = 1
    return adj_matrix

def save_results_and_plots(W_true_union, W_true_intersection, pred_list, save_dir):
    """Save results, plots, and metrics to the specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    knockoff_edges, notears_edges, intersection_edges, notears_knockoff_edges = pred_list
    predictions = [knockoff_edges, notears_edges, intersection_edges, notears_knockoff_edges]
    titles = ["Knockoff Learned DAG", "NO TEARS Learned DAG", "Intersection Learned DAG", "Knockoff + NO TEARS"]
    
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

    # Knockoff-based parent selection
    knockoff_edges = nodewise_knockoff_selection(X_combined, fdr=args.fdr)

    # NO TEARS
    W_notears = notears_linear(X_combined, lambda1=lambda_val, loss_type='l2')
    notears_edges = [(i, j) for i in range(args.nodes) for j in range(args.nodes) if W_notears[i, j] != 0]

    # Intersection of Knockoff and NO TEARS edges
    intersection_edges = list(set(knockoff_edges) & set(notears_edges))

    # Knockoff feature selection followed by NO TEARS
    
    beta = knockpy.dgp.create_sparse_coefficients(p=p, sparsity=fdr)
    y = np.dot(X_combined, beta)

    selected_features = knockoff_feature_selection(X_combined, y, fdr=fdr)
    X_selected = X_combined[:, selected_features]
    W_notears_knockoff = notears_linear(X_selected, lambda1=args.lambda_val, loss_type='l2')
    notears_knockoff_edges = [(selected_features[i], selected_features[j]) for i in range(len(selected_features)) for j in range(len(selected_features)) if W_notears_knockoff[i, j] != 0]

    # Define save directory
    save_dir = f"results_multiple/notears_knockoff_multiple_graphs/groups_{args.groups}_samples_{args.samples}_nodes_{args.nodes}_fdr_{args.fdr}_seed_{args.seed}"
    
    # Save results and plots
    save_results_and_plots(W_true_union, W_true_intersection, (knockoff_edges, notears_edges, intersection_edges, notears_knockoff_edges), save_dir)
