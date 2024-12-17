import os
import json
import re
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def parse_res_file(file_path):
    """Parse the res.txt file and extract configurations and metrics for each method."""
    results = []
    config = {}

    # Extract configuration from the path
    path_pattern = re.compile(
        r"samples_(\d+)_nodes_(\d+)_fdr_([\d.]+)/graph_type_(\w+)_graph_prob_([\d.]+)_noise_type_(\w+)_noise_scale_([\d.]+)_lambda_([\d.]+)"
    )
    match = path_pattern.search(file_path)
    if match:
        config['samples'] = int(match.group(1))
        config['nodes'] = int(match.group(2))
        config['fdr'] = float(match.group(3))
        config['graph_type'] = match.group(4)
        config['graph_prob'] = float(match.group(5))
        config['noise_type'] = match.group(6)
        config['noise_scale'] = float(match.group(7))
        config['lambda_val'] = float(match.group(8))
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract metrics for each method
    current_method = None
    current_metrics = {}
    for line in lines:
        if "Results:" in line:
            if current_method and current_metrics:
                results.append((current_method, current_metrics))
                current_metrics = {}
            current_method = line.strip().replace(" Results:", "")
        elif line.strip().startswith("{"):
            try:
                current_metrics = json.loads(line.strip().replace("'", "\""))
            except json.JSONDecodeError:
                continue

    if current_method and current_metrics:
        results.append((current_method, current_metrics))

    return config, results

def collect_all_results(base_dir):
    """Recursively collect all results from res.txt files in the base directory."""
    all_data = []

    for root, _, files in os.walk(base_dir):
        if 'res.txt' in files:
            file_path = os.path.join(root, 'res.txt')
            config, results = parse_res_file(file_path)
            for method, metrics in results:
                all_data.append({
                    "method": method,
                    "config": config,
                    "metrics": metrics
                })

    return all_data

def plot_metric(all_data, metric_name, save_dir):
    """Generate plots for a specific metric across different configurations."""
    os.makedirs(save_dir, exist_ok=True)

    # Create DataFrame for plotting
    data = []
    for entry in all_data:
        config = entry['config']
        metrics = entry['metrics']
        method = entry['method']
        if metric_name in metrics:
            data.append({
                'method': method,
                'nodes': config.get('nodes'),
                'samples': config.get('samples'),
                'fdr': config.get('fdr'),
                'graph_prob': config.get('graph_prob'),
                'noise_type': config.get('noise_type'),
                'lambda_val': config.get('lambda_val'),
                'value': metrics[metric_name]
            })

    df = pd.DataFrame(data)

    print(df)

    # Plot based on different configurations
    def plot_by_group(group_by, title, filename):
        plt.figure(figsize=(10, 6))
        for method, group in df.groupby('method'):
            grouped = group.groupby(group_by)['value'].mean().reset_index()
            plt.plot(grouped[group_by], grouped['value'], marker='o', label=method)
        
        plt.xlabel(group_by)
        plt.ylabel(metric_name)
        plt.title(f"{metric_name.upper()} vs {group_by} ({title})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    # Plot for different configurations
    plot_by_group('nodes', 'Different Node Sizes', f"{metric_name}_nodes.png")
    plot_by_group('samples', 'Different Sample Sizes', f"{metric_name}_samples.png")
    plot_by_group('fdr', 'Different Initial FDR', f"{metric_name}_fdr.png")
    plot_by_group('graph_prob', 'Different Graph Probabilities', f"{metric_name}_graph_prob.png")
    plot_by_group('noise_type', 'Different Noise Types', f"{metric_name}_noise_type.png")

def main():
    base_dir = "/home/mila/m/mehrab.hamidi/scratch/pgm_results/results/notears_knockoff"  # Base directory containing results
    save_dir = "/home/mila/m/mehrab.hamidi/scratch/pgm_results/results/analysis"  # Directory to save the plots

    # Collect all results
    all_data = collect_all_results(base_dir)

    # Metrics to plot
    metrics = ['fdr', 'tpr', 'fpr', 'structural_hamming_du']  # 'structural_hamming_du' stands for Structural Hamming Distance

    # Generate plots for each metric
    for metric in metrics:
        plot_metric(all_data, metric, save_dir)
        print(f"Plots for {metric} saved in {save_dir}")

if __name__ == "__main__":
    main()
