#!/bin/bash
outdir = "test_data"


sbatch <<EOT
#!/bin/bash

#SBATCH --partition=long-cpu         # Specify the partition
#SBATCH --job-name=data_generator_test    # Job name
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                    # Memory allocation
#SBATCH --time=05:00:00              # Max runtime (hh:mm:ss)
#SBATCH --output=logs/test/%x_%j.out      # Standard output log
#SBATCH --error=logs/test/%x_%j.err       # Standard error log

module purge
module load miniconda/3
conda activate dag             

mkdir -p logs

# Run the Python script
set -x
srun python generate_data.py --num-graphs 3 --mode test
EOT