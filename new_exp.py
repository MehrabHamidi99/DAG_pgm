import numpy as np
import os
import json
import matplotlib.pyplot as plt
import networkx as nx
import time
import sys

from ..Methods import notears_linear

sys.path.append('/path/to/directory/containing/Evaluation')    # **Update this path**
from Evaluation import count_accuracy

sys.path.append('/path/to/directory/containing/DAG_generation')  # **Update this path**
from DAG_generation import random_dag_generation, generate_single_dataset

from julia.api import Julia # type: ignore
jl = Julia(compiled_modules=False)  # Initialize Julia with compiled modules disabled

from knockoffspy import ko # type: ignore
from scipy import linalg
from sklearn.linear_model import Lasso

raise Exception("TESTIng")

def perform_knockoff_filtering(X, Y, groups, m=5, q=0.1):
    """
    Perform knockoff filtering using modelX_gaussian_group_knockoffs from knockoffspy.
    
    Steps:
    1. Define groups for features.
    2. Estimate mu and Sigma from X.
    3. Generate model-X Gaussian group knockoffs for X.
    4. Fit a Lasso model on [X, X_knock1, ..., X_knockm].
    5. Compute W statistics: |beta_j| - median(|beta_knock1_j|, ..., |beta_knockm_j|).
    6. Determine κ: which among original and knockoffs has the highest |beta|.
    7. Apply mk_threshold to find the cutoff T.
    8. Select variables where W_j >= T.
    
    Returns:
    - selected: array of selected variable indices
    """
    n, p = X.shape
    num_groups = len(np.unique(groups))
    
    # Estimate mu and Sigma from X
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False)
    
    # Generate group knockoffs
    solver = "maxent"  # Choices: "maxent", "mvr", "sdp", "equi"
    try:
        result = ko.modelX_gaussian_group_knockoffs(X, solver, groups, mu, Sigma, m=m, verbose=False)
    except Exception as e:
        print(f"Error generating knockoffs: {e}")
        return np.array([])
    
    X_knock = result.Xko  # Shape: (n, p * m)
    
    # Fit Lasso on [X, X_knock]
    XX = np.hstack([X, X_knock])
    lasso = Lasso(alpha=0.01, fit_intercept=True)
    lasso.fit(XX, Y)
    beta_hat = lasso.coef_  # Shape: (p + p*m,)
    
    # Compute W statistics and κ
    tau = np.zeros(p)
    kappa = np.zeros(p, dtype=int)
    
    for j in range(p):
        # Original feature coefficient
        beta_original = np.abs(beta_hat[j])
        
        # Knockoff coefficients for feature j
        knockoff_start = j + 1 + j * m
        knockoff_end = knockoff_start + m
        beta_knockoffs = np.abs(beta_hat[knockoff_start:knockoff_end])
        
        # Compute tau[j] as original - median of knockoffs
        tau[j] = beta_original - np.median(beta_knockoffs)
        
        # Determine which has the maximum coefficient
        all_betas = np.concatenate(([beta_original], beta_knockoffs))
        max_idx = np.argmax(all_betas)
        kappa[j] = max_idx  # 0 for original, 1 to m for knockoffs
    
    # Apply knockoff thresholding
    try:
        # mk_threshold expects lists
        T = ko.mk_threshold(tau.tolist(), kappa.tolist(), m, q)
    except Exception as e:
        print(f"Error computing threshold: {e}")
        return np.array([])
    
    # Select variables
    selected = np.where(tau >= T)[0]
    
    return selected

def visualize_comparison(G_true, W_est, save_path, title):
    """
    Visualize a single graph comparison.
    
    Parameters:
    - G_true: Adjacency matrix of the true DAG
    - W_est: Adjacency matrix of the estimated DAG
    - save_path: File path to save the visualization
    - title: Title of the plot
    """
    plt.figure(figsize=(12, 12))
    
    # Identify True Positives (TP), False Positives (FP), False Negatives (FN)
    TP = np.where((G_true == 1) & (W_est == 1))
    FP = np.where((G_true == 0) & (W_est == 1))
    FN = np.where((G_true == 1) & (W_est == 0))
    
    G = nx.DiGraph()
    rows_true, cols_true = np.where(G_true > 0)
    G.add_edges_from(zip(rows_true.tolist(), cols_true.tolist()))
    
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Draw True Positive edges
    edges_tp = list(zip(TP[0], TP[1]))
    nx.draw_networkx_edges(G, pos, edgelist=edges_tp, edge_color='green', arrows=True, arrowsize=20, width=2, label='True Positive')
    
    # Draw False Positive edges
    edges_fp = list(zip(FP[0], FP[1]))
    if edges_fp:
        G.add_edges_from(edges_fp)  # Temporarily add FP edges for visualization
        nx.draw_networkx_edges(G, pos, edgelist=edges_fp, edge_color='red', arrows=True, arrowsize=20, style='dashed', label='False Positive')
        G.remove_edges_from(edges_fp)  # Remove after drawing
    
    # Draw False Negative edges
    edges_fn = list(zip(FN[0], FN[1]))
    if edges_fn:
        nx.draw_networkx_edges(G, pos, edgelist=edges_fn, edge_color='orange', arrows=True, arrowsize=20, width=3, label='False Negative')
    
    # Create legend
    if edges_fp or edges_fn:
        plt.legend(scatterpoints=1)
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    # Experiment parameters
    d = 30              # Number of variables
    n = 200             # Number of samples per dataset
    prob = 0.2          # Edge probability for DAG generation
    graph_type = 'er'   # Type of graph ('er' for Erdős–Rényi)
    noise_types = 'gauss'  # Type of noise
    K = 3               # Number of datasets
    m = 5               # Number of knockoffs per variable
    
    save_path = 'experiment_knockoffspy_group_ko_example'
    os.makedirs(save_path, exist_ok=True)
    
    # Create datasets
    dags = []
    X_list = []
    Y_list = []
    for k in range(K):
        g_dag, adj_dag = random_dag_generation(d, prob, graph_type)
        X = generate_single_dataset(g_dag, n, noise_types, 1)
        # Define Y as a linear combination of the first two variables plus noise for demonstration
        Y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
        
        dags.append(adj_dag)
        X_list.append(X)
        Y_list.append(Y)
        
        # Save true DAG and data
        np.savetxt(f'{save_path}/W_true_{k}.csv', adj_dag, delimiter=',')
        np.savetxt(f'{save_path}/X_{k}.csv', X, delimiter=',')
        np.savetxt(f'{save_path}/Y_{k}.csv', Y, delimiter=',')
    
    # Define groups for group knockoffs
    group_size = 5
    num_groups = d // group_size
    groups = np.repeat(np.arange(num_groups), group_size)
    if d % group_size != 0:
        groups = np.concatenate([groups, np.full(d % group_size, num_groups)])
    
    # Knockoff + NO TEARS union approach
    S_list = []
    for k in range(K):
        print(f"Performing knockoff filtering for dataset {k+1}/{K}...")
        selected = perform_knockoff_filtering(X_list[k], Y_list[k], groups=groups, m=m, q=0.1)
        print(f"Selected features: {selected}")
        S_list.append(set(selected))
    
    # Compute the union of selected features across all datasets
    S_union = set.union(*S_list)
    S_union = sorted(list(S_union))
    print(f"Union of selected features across all datasets: {S_union}")
    
    # Combine selected features from all datasets
    if S_union:
        X_union_list = [X[:, S_union] for X in X_list]
        X_combined_selected = np.vstack(X_union_list)
        
        # Run NO TEARS on the combined selected dataset
        print("Running NO TEARS on the combined selected dataset...")
        start_time_knockoff = time.time()
        W_est_restricted = notears_linear(X_combined_selected, lambda1=0.1, loss_type='l2')
        end_time_knockoff = time.time()
        notears_time_knockoff = end_time_knockoff - start_time_knockoff
        print(f"NO TEARS (Union Selected) completed in {notears_time_knockoff:.2f} seconds.")
        
        # Map W_est_restricted back to full set of variables
        W_est_full_selected = np.zeros((d, d))
        for i, vi in enumerate(S_union):
            for j, vj in enumerate(S_union):
                W_est_full_selected[vi, vj] = W_est_restricted[i, j]
        np.savetxt(f'{save_path}/W_est_union_selected.csv', W_est_full_selected, delimiter=',')
    else:
        print("No features were selected by knockoff filtering.")
        W_est_full_selected = np.zeros((d, d))
    
    # NO TEARS on full data (no selection)
    print("Running NO TEARS on the full dataset (no selection)...")
    X_combined_full = np.vstack(X_list)
    start_time_full = time.time()
    W_est_full = notears_linear(X_combined_full, lambda1=0.1, loss_type='l2')
    end_time_full = time.time()
    notears_time_full = end_time_full - start_time_full
    np.savetxt(f'{save_path}/W_est_full_no_selection.csv', W_est_full, delimiter=',')
    print(f"NO TEARS (No Selection) completed in {notears_time_full:.2f} seconds.")
    
    # Evaluate and visualize
    # Create a union-of-true DAG: edges that appear in any dataset's ground truth
    W_true_union = np.zeros((d, d))
    for k in range(K):
        W_true_union = np.maximum(W_true_union, dags[k])
    
    # Visualize the results
    if S_union:
        visualize_comparison(
            G_true=W_true_union,
            W_est=W_est_full_selected,
            save_path=os.path.join(save_path, 'compare_union_selected.png'),
            title="Knockoff + NO TEARS (Union Selected) vs True Union"
        )
    else:
        print("Skipping visualization for Knockoff + NO TEARS as no features were selected.")
    
    visualize_comparison(
        G_true=W_true_union,
        W_est=W_est_full,
        save_path=os.path.join(save_path, 'compare_full_no_selection.png'),
        title="NO TEARS (No Selection) vs True Union"
    )
    
    # Optionally, visualize against each dataset's true DAG
    for k in range(K):
        if S_union:
            visualize_comparison(
                G_true=dags[k],
                W_est=W_est_full_selected,
                save_path=os.path.join(save_path, f'compare_union_selected_k{k}.png'),
                title=f"Knockoff + NO TEARS vs Dataset {k} True"
            )
        visualize_comparison(
            G_true=dags[k],
            W_est=W_est_full,
            save_path=os.path.join(save_path, f'compare_full_no_selection_k{k}.png'),
            title=f"NO TEARS (No Selection) vs Dataset {k} True"
        )
    
    # Evaluate accuracy
    results_selected = {}
    if S_union:
        for k in range(K):
            acc = count_accuracy(dags[k], W_est_full_selected != 0)
            results_selected[k] = acc
    
    results_full = {}
    for k in range(K):
        acc = count_accuracy(dags[k], W_est_full != 0)
        results_full[k] = acc
    
    # Record comparison results and timing
    comparison = {
        "knockoff_selected_results": results_selected,
        "full_no_selection_results": results_full,
        "notears_time_knockoff_selected": notears_time_knockoff if S_union else None, # type: ignore
        "notears_time_full_no_selection": notears_time_full
    }
    
    with open(f'{save_path}/comparison.txt', "w") as f:
        f.write(json.dumps(comparison, indent=2))
    
    print("Comparison Results:")
    print(json.dumps(comparison, indent=2))
