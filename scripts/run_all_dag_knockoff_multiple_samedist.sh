#!/bin/bash

# Navigate to the script directory
cd ~/pgm_project/DAG_pgm/scripts/

# Parameter arrays
samples=(100 1000)
nodes=(20 50 100)
fdr_values=(0.9)
graph_types=('er')
s0s=(40 100 200 500)
noise_types=('gauss')
noise_scales=(0.1 1.0)
lambda_vals=(0.0)
groups=(2 3 4)
seeds=(12345 14583 14952)


# Create logs directory if it doesn't exist
mkdir -p $SCRATCH/logs_multiples_samedist

# Submit jobs for all combinations of parameters
for s in "${samples[@]}"; do
  for n in "${nodes[@]}"; do
    for fdr in "${fdr_values[@]}"; do
      for g_t in "${graph_types[@]}"; do
        for s0 in "${s0s[@]}"; do
          for n_t in "${noise_types[@]}"; do
            for n_s in "${noise_scales[@]}"; do
              for l_val in "${lambda_vals[@]}"; do
                for g_n in "${groups[@]}"; do
                  for seed in "${seeds[@]}"; do
                    ./dag_knockoff_multiple_samedist.sh $s $n $fdr $g_t $s0 $n_t $n_s $l_val $g_n $seed
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
