#!/bin/bash

# Navigate to the script directory
cd ~/pgm_project/DAG_pgm/scripts/

# Parameter arrays
samples=(100 1000)
nodes=(10 20 50)
fdr_values=(0.05 0.1 0.9)
graph_types=('er')
graph_probs=(0.1 0.2 0.5)
noise_types=('gauss' 'exp' 'uniform')
noise_scales=(0.1 1.0)
lambda_vals=(0.0 0.1)


# Create logs directory if it doesn't exist
mkdir -p $SCRATH/logs

# Submit jobs for all combinations of parameters
for s in "${samples[@]}"; do
  for n in "${nodes[@]}"; do
    for fdr in "${fdr_values[@]}"; do
      for g_t in "${graph_types[@]}"; do
        for g_p in "${graph_probs[@]}"; do
          for n_t in "${noise_types[@]}"; do
            for n_s in "${noise_scales[@]}"; do
              for l_val in "${lambda_vals[@]}"; do
                sbatch dag_knockoff.sh $s $n $fdr $g_t $g_p $n_t $n_s $l_val
              done
            done
          done
        done
      done
    done
  done
done
