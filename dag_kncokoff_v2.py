import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import knockpy
from knockpy import KnockoffFilter
from sklearn.preprocessing import StandardScaler
from DAG_generation import random_dag_generation, generate_single_dataset
from Methods import notears_linear
from Evaluation import count_accuracy
from Visualization import visualize_dag, visualize_adjacency_matrices
import os
from Methods import notears_linear, notears_knockoff_regularizer


def save_results_and_plots(W_true, pred_list, save_dir):
    """Save results, plots, and metrics to the specified directory."""

    notears_knockoff_edges, notears_knockoff_edges_normal = pred_list

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # List of predictions and titles
    predictions = [notears_knockoff_edges, notears_knockoff_edges_normal]
    titles = ["NOTEARS Knockoff regularizer", "NOTEARS"]

    print(notears_knockoff_edges)
    print(notears_knockoff_edges_normal)

    # Visualize and save DAGs
    visualize_dag(W_true, predictions, titles, save_path=os.path.join(save_dir, "dag_comparison.png"))

    # Visualize and save adjacency matrices
    visualize_adjacency_matrices(W_true, predictions, titles, save_path=os.path.join(save_dir, "adjacency_heatmaps.png"))

    # Compute and save metrics
    metrics_knockoff = count_accuracy(W_true, edges_to_adj_matrix(notears_knockoff_edges, W_true.shape[0]))
    metrics_knockoff_normal = count_accuracy(W_true, edges_to_adj_matrix(notears_knockoff_edges_normal, W_true.shape[0]))

    # Save all results in res.txt
    with open(os.path.join(save_dir, "res.txt"), "w") as f:
        f.write("NOTEARS Knockoff regularizer Results:\n")
        f.write(str(metrics_knockoff) + "\n\n")
        
        f.write("NOTEARS Results:\n")
        f.write(str(metrics_knockoff_normal) + "\n\n")


def edges_to_adj_matrix(edges, p):
    """Convert a list of edges to an adjacency matrix."""
    adj_matrix = np.zeros((p, p))
    for i, j in edges:
        adj_matrix[i, j] = 1
    return adj_matrix

if __name__ == "__main__":
    # # Argument parser
    # parser = argparse.ArgumentParser(description="DAG Learning with Knockoff and NO TEARS")
    # parser.add_argument("--samples", type=int, required=True, help="Number of samples")
    # parser.add_argument("--nodes", type=int, required=True, help="Number of nodes (variables)")
    # parser.add_argument("--fdr", type=float, required=True, help="Target false discovery rate")
    # parser.add_argument("--graph_type", type=str, required=True, choices=['er', 'ba', 'ws'], help="Graph type")
    # parser.add_argument("--graph_prob", type=float, required=True, help="Graph probability")
    # parser.add_argument("--noise_type", type=str, required=True, choices=['gauss', 'exp', 'gumbel', 'uniform'], help="Noise type")
    # parser.add_argument("--noise_scale", type=float, required=True, help="Noise scale")
    # parser.add_argument("--lambda_val", type=float, required=True, help="Lambda value for NO TEARS")

    # args = parser.parse_args()

    # # Assign variables from arguments
    # n = args.samples
    # p = args.nodes
    # fdr = args.fdr
    # graph_type = args.graph_type
    # graph_prob = args.graph_prob
    # noise_type = args.noise_type
    # noise_scale = args.noise_scale
    # lambda_val = args.lambda_val

    n = 500
    p = 30
    graph_type = 'er'
    graph_prob = 0.1
    noise_type = 'gauss'
    noise_scale = 1
    lambda_val = 0.0

    # Generate synthetic data
    g_dag, adj_dag = random_dag_generation(p, graph_prob, graph_type)
    X, W_true = generate_single_dataset(g_dag, n, noise_type, noise_scale, zeros=False, adj=True)
    
    # Run NO TEARS
    W_notears = notears_knockoff_regularizer(X, lambda1=lambda_val, loss_type='l2')
    notears_edges = [(i, j) for i in range(p) for j in range(p) if W_notears[i, j] != 0]

    W_notears_normal = notears_linear(X, lambda1=lambda_val, loss_type='l2')
    notears_edges_normal = [(i, j) for i in range(p) for j in range(p) if W_notears_normal[i, j] != 0]


    # Define save directory
    save_dir = f"results/notears_knockoff_penalty__/samples_{n}_nodes_{p}/graph_type_{graph_type}_graph_prob_{graph_prob}_noise_type_{noise_type}_noise_scale_{noise_scale}_lambda_{lambda_val}____"
    
    # Save results and plots
    save_results_and_plots(W_true, (notears_edges, notears_edges_normal), save_dir)
