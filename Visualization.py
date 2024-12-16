import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

def visualize_dag(W_true, predictions, titles, save_path="dag_comparison.png"):
    """
    Visualize the true DAG and the learned DAGs with selected edges.

    Parameters:
    - W_true: numpy array, the adjacency matrix of the true DAG.
    - predictions: list of lists, each containing predicted edges (as tuples).
    - titles: list of strings, titles for each prediction subplot.
    - save_path: string, the path to save the plot.
    """
    # True DAG
    G_true = nx.from_numpy_array(W_true, create_using=nx.DiGraph)
    
    # Create consistent node positions
    pos = nx.spring_layout(G_true, seed=42)

    # Plot the True DAG and all predictions side by side
    num_plots = len(predictions) + 1
    plt.figure(figsize=(6 * num_plots, 6))
    
    # Plot True DAG
    plt.subplot(1, num_plots, 1)
    nx.draw(G_true, pos, with_labels=True, node_color='lightblue', node_size=500, arrowsize=20)
    plt.title("True DAG")
    
    # Plot each predicted DAG
    colors = ['lightgreen', 'lightcoral', 'gold', 'lightpink', 'lightskyblue']
    for i, (edges, title) in enumerate(zip(predictions, titles)):
        G_pred = nx.DiGraph()
        G_pred.add_edges_from(edges)
        
        plt.subplot(1, num_plots, i + 2)
        nx.draw(G_pred, pos, with_labels=True, node_color=colors[i % len(colors)], node_size=500, arrowsize=20)
        plt.title(title)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



def visualize_adjacency_matrices(W_true, predictions, titles, save_path="adjacency_heatmaps.png"):
    """
    Visualize the adjacency matrices of the true DAG and the learned DAGs as heatmaps.

    Parameters:
    - W_true: numpy array, the adjacency matrix of the true DAG.
    - predictions: list of lists, each containing predicted edges (as tuples).
    - titles: list of strings, titles for each prediction subplot.
    - save_path: string, the path to save the plot.
    """
    p = W_true.shape[0]

    # Create adjacency matrices for the predictions
    W_predictions = []
    for edges in predictions:
        W_pred = np.zeros((p, p))
        for i, j in edges:
            W_pred[i, j] = 1
        W_predictions.append(W_pred)

    # Plot the True adjacency matrix and all predictions side by side
    num_plots = len(W_predictions) + 1
    plt.figure(figsize=(6 * num_plots, 6))

    # Plot True Adjacency Matrix
    plt.subplot(1, num_plots, 1)
    sns.heatmap(W_true, annot=True, cmap="Blues", cbar=True, square=True)
    plt.title("True Adjacency Matrix")

    # Plot each predicted adjacency matrix
    colormaps = ['Greens', 'Reds', 'Oranges', 'Purples', 'BuPu']
    for i, (W_pred, title) in enumerate(zip(W_predictions, titles)):
        plt.subplot(1, num_plots, i + 2)
        sns.heatmap(W_pred, annot=True, cmap=colormaps[i % len(colormaps)], cbar=True, square=True)
        plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
