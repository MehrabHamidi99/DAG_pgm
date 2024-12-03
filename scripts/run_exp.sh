#!/bin/bash
#SBATCH --job-name=dag_no_tears
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=5

module load miniconda
conda activate dag_pgm

cd ~/pgm_project/DAG_pgm/scripts/

python ~/pgm_project/DAG_pgm/run_experiment.py --noise_types=$1 \
                        --graph_type=$2 \
                        --num_nodes=$3 \
                        --num_samples=${4} \
                        --prob=$5 \


module load miniconda
conda activate dag_pgm

echo "Visualizing"

python ~/pgm_project/DAG_pgm/visualize_experiment.py --noise_types=$1 \
                        --graph_type=$2 \
                        --num_nodes=$3 \
                        --num_samples=${4} \
                        --prob=$5 \
