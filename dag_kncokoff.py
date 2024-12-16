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

# Set random seed for reproducibility
np.random.seed(42)


def knockoff_feature_selection(X, y, fdr=0.1):
    """Select features using knockoff filtering."""
    print("\nPerforming feature selection using knockoff filtering...")
    
    # Generate covariance matrix (AR1 process with correlation 0.5)
    Sigma = knockpy.dgp.AR1(p=X.shape[1], rho=0.1)
    
    # Knockoff filter
    kfilter = KnockoffFilter(
        ksampler='gaussian',
        fstat='lasso',
    )
    
    # Run knockoff filter to select significant features
    rejections = kfilter.forward(
        X=X,
        y=y,  # Random target for unsupervised setting
        Sigma=Sigma,
        fdr=fdr
    )    
    selected_features = np.where(rejections == 1)[0]
    
    return selected_features


def nodewise_knockoff_selection(X, fdr=0.1):
    """Perform knockoff-based parent selection for each node."""
    n, p = X.shape
    selected_edges = []
    ancestor_sets = [set() for _ in range(p)]  # Track ancestors for each node

    for j in range(p):
        print(f"\nSelecting parents for node {j}")
        
        # Skip nodes that are already known ancestors of others
        predictors_indices = [k for k in range(p) if k != j and j not in ancestor_sets[k]]
        
        if not predictors_indices:
            continue  # No valid predictors, skip this node

        # Prepare predictors and target
        predictors = X[:, predictors_indices]
        target = X[:, j]

        # Standardize predictors
        # scaler = StandardScaler()
        # predictors = scaler.fit_transform(predictors)

        # Generate covariance matrix (AR1 process with correlation 0.5)
        Sigma = knockpy.dgp.AR1(p=predictors.shape[1], rho=0.5)

        # Knockoff filter
        kfilter = KnockoffFilter(
            ksampler='gaussian',
            fstat='lasso',
        )

        # Run knockoff filter to select significant predictors
        rejections = kfilter.forward(
            X=predictors,
            y=target,
            Sigma=Sigma,
            fdr=fdr  # Desired level of false discovery rate control
        )

        # Record selected edges (j is the target, i are the parents)
        for idx, reject in enumerate(rejections):
            if reject == 1:
                parent = predictors_indices[idx]
                # Check if adding the edge creates a cycle
                if not creates_cycle(selected_edges, parent, j, p):
                    selected_edges.append((parent, j))
                    ancestor_sets[j].add(parent)

    return selected_edges

def creates_cycle(edges, parent, child, p):
    """Check if adding the edge (parent -> child) creates a cycle."""
    G = nx.DiGraph()
    G.add_edges_from(edges)
    G.add_edge(parent, child)

    try:
        # If there's a cycle, this will raise an exception
        nx.find_cycle(G, orientation='original')
        return True
    except nx.NetworkXNoCycle:
        return False

def save_results_and_plots(W_true, pred_list, save_dir):
    """Save results, plots, and metrics to the specified directory."""

    knockoff_edges, notears_edges, intersection_edges, notears_knockoff_edges = pred_list

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save adjacency matrices
    np.savetxt(os.path.join(save_dir, "W_true.csv"), W_true, delimiter=",")
    np.savetxt(os.path.join(save_dir, "W_knockoff.csv"), edges_to_adj_matrix(knockoff_edges, W_true.shape[0]), delimiter=",")
    np.savetxt(os.path.join(save_dir, "W_notears.csv"), edges_to_adj_matrix(notears_edges, W_true.shape[0]), delimiter=",")
    np.savetxt(os.path.join(save_dir, "W_intersection.csv"), edges_to_adj_matrix(intersection_edges, W_true.shape[0]), delimiter=",")

    # List of predictions and titles
    predictions = [knockoff_edges, notears_edges, intersection_edges, notears_knockoff_edges]
    titles = ["Knockoff Learned Adjacency Matrix", "NO TEARS Learned Adjacency Matrix", "Intersection Learned Adjacency Matrix", "Knockoff then NoTEARS"]

    
    # Visualize and save DAGs
    visualize_dag(W_true, predictions, titles, save_path=os.path.join(save_dir, "dag_comparison.png"))

    # Visualize and save adjacency matrices
    visualize_adjacency_matrices(W_true, predictions, titles, save_path=os.path.join(save_dir, "adjacency_heatmaps.png"))

    # Compute and save metrics
    metrics_knockoff = count_accuracy(W_true, edges_to_adj_matrix(knockoff_edges, W_true.shape[0]))
    metrics_notears = count_accuracy(W_true, edges_to_adj_matrix(notears_edges, W_true.shape[0]))
    metrics_intersection = count_accuracy(W_true, edges_to_adj_matrix(intersection_edges, W_true.shape[0]))
    metrics_knoc_not = count_accuracy(W_true, edges_to_adj_matrix(notears_knockoff_edges, W_true.shape[0]))

    # Save all results in res.txt
    with open(os.path.join(save_dir, "res.txt"), "w") as f:
        f.write("Knockoff Results:\n")
        f.write(str(metrics_knockoff) + "\n\n")

        f.write("NO TEARS Results:\n")
        f.write(str(metrics_notears) + "\n\n")

        f.write("Intersection Results:\n")
        f.write(str(metrics_intersection) + "\n\n")

        f.write("Knockoff followed by NOTEARS Results:\n")
        f.write(str(metrics_knoc_not) + "\n")

def edges_to_adj_matrix(edges, p):
    """Convert a list of edges to an adjacency matrix."""
    adj_matrix = np.zeros((p, p))
    for i, j in edges:
        adj_matrix[i, j] = 1
    return adj_matrix

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="DAG Learning with Knockoff and NO TEARS")
    parser.add_argument("--samples", type=int, required=True, help="Number of samples")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes (variables)")
    parser.add_argument("--fdr", type=float, required=True, help="Target false discovery rate")
    parser.add_argument("--graph_type", type=str, required=True, choices=['er', 'ba', 'ws'], help="Graph type")
    parser.add_argument("--graph_prob", type=float, required=True, help="Graph probability")
    parser.add_argument("--noise_type", type=str, required=True, choices=['gauss', 'exp', 'gumbel', 'uniform'], help="Noise type")
    parser.add_argument("--noise_scale", type=float, required=True, help="Noise scale")
    parser.add_argument("--lambda_val", type=float, required=True, help="Lambda value for NO TEARS")

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

    # Generate synthetic data
    g_dag, adj_dag = random_dag_generation(p, graph_prob, graph_type)
    X, W_true = generate_single_dataset(g_dag, n, noise_type, noise_scale, zeros=False, adj=True)
    # Create random sparse coefficients
    beta = knockpy.dgp.create_sparse_coefficients(p=p, sparsity=fdr)
    y = np.dot(X, beta) + np.random.randn(n)

    # Perform nodewise knockoff selection
    knockoff_edges = nodewise_knockoff_selection(X, fdr=0.9)

    # Run NO TEARS
    W_notears = notears_linear(X, lambda1=lambda_val, loss_type='l2')
    notears_edges = [(i, j) for i in range(p) for j in range(p) if W_notears[i, j] != 0]

    # Compute intersection of knockoff and NO TEARS edges
    intersection_edges = list(set(knockoff_edges) & set(notears_edges))


    # Perform feature selection using knockoff and run NO TEARS on selected features
    selected_features = knockoff_feature_selection(X, y, fdr=fdr)
    if len(selected_features) != 0:
        X_selected = X[:, selected_features]
    else:
        X_selected = X
    print(selected_features)
    W_notears_knockoff = notears_linear(X_selected, lambda1=lambda_val, loss_type='l2')
    notears_knockoff_edges = [(selected_features[i], selected_features[j]) for i in range(len(selected_features)) for j in range(len(selected_features)) if W_notears_knockoff[i, j] != 0]

    # Define save directory
    save_dir = f"results/notears_knockoff/samples_{n}_nodes_{p}_fdr_{fdr}/graph_type_{graph_type}_graph_prob_{graph_prob}_noise_type_{noise_type}_noise_scale_{noise_scale}_lambda_{lambda_val}"
    
    # Save results and plots
    save_results_and_plots(W_true, (knockoff_edges, notears_edges, intersection_edges, notears_knockoff_edges), save_dir)
