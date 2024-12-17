#!/bin/bash
#SBATCH --job-name=dag_knockoff_same_dist
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=5
#SBATCH --output=/home/mila/m/mehrab.hamidi/scratch/logs_multiples_samedist/dag_knockoff_%j.out

# Load necessary modules and activate the conda environment
module load miniconda
conda activate dag_pgm

# Navigate to the script directory
cd $SCRATCH/pgm_results_multiple/

# Run the Python script with the provided arguments
python ~/pgm_project/DAG_pgm/dag_kncokoff_multiple_grpahs_samedist.py --samples=${1} \
                        --nodes=${2} \
                        --fdr=${3} \
                        --graph_type=${4} \
                        --edges=${5} \
                        --noise_type=${6} \
                        --noise_scale=${7} \
                        --lambda_val=${8} \
                        --groups=${9} \
                        --seed=${10} 

# Deactivate conda environment
conda deactivate
