#!/bin/bash

out_dir="output"
job="test_gran_dag"
job_name=$job
dataset="toy_5_nodes"

mkdir -p $out_dir/$job_name

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=$job_name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=$out_dir/$job_name/%t_out_%j.txt
#SBATCH --error=$out_dir/$job_name/%t_err_%j.txt
#SBATCH --time=0:30:00

module purge
module load miniconda/3
conda activate dag
set -x
srun python main.py
EOT
