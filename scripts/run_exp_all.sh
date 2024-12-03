#!/bin/bash

noise_types=('gauss' 'exp' 'gumbel' 'uniform')
graph_type=('er')
num_nodes=(20 50 100)
num_samples=(50 100)
prob=(0.01 0.05 0.1 0.2)


# Loop through each combination of parameters
for n_t in "${noise_types[@]}"; do
  for g_t in "${graph_type[@]}"; do
    for n in "${num_nodes[@]}"; do
      for n_s in "${num_samples[@]}"; do
        for p in "${prob[@]}"; do
            # Submit job with the current combination of parameters
            sbatch run_exp.sh $n_t $g_t $n $n_s $p
        done
      done
    done
  done
done