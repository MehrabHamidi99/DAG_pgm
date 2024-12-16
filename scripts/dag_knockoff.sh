#!/bin/bash
#SBATCH --job-name=dag_knockoff
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=5
#SBATCH --output=$SCRATH/logs/dag_knockoff_%j.out

# Load necessary modules and activate the conda environment
module load miniconda
conda activate dag_pgm

# Navigate to the script directory
cd $SCRATH/pgm_results/

# Run the Python script with the provided arguments
python ~/pgm_project/DAG_pgm/dag_kncokoff.py --samples=${1} \
                        --nodes=${2} \
                        --fdr=${3} \
                        --graph_type=${4} \
                        --graph_prob=${5} \
                        --noise_type=${6} \
                        --noise_scale=${7} \
                        --lambda_val=${8}

# Deactivate conda environment
conda deactivate
