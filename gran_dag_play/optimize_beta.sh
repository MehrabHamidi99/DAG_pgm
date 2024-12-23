#!/bin/bash

out_dir="outputs"
job="train_3_graphs_10_nodes"
NUM_VARS=10
TRAIN_FLAG="--train"  
DATA_PATH="data/"
G_PATH1="data/10_nodes_G1.gpickle"
G_PATH2="data/10_nodes_G2.gpickle"
G_PATH3="data/10_nodes_G3.gpickle"
EXP_PATH="exp/10_nodes_3_graphs/"
OUTPUT_CKPT="models/"

SEM_TYPE="non_linear_gaussian"
SIGMA_MIN=0.4
SIGMA_MAX=0.8
LOAD_DATA="True"
N_POINTS=10000

NUM_NEIGHBORS="None"
if [[ "$NUM_NEIGHBORS" == "None" ]]; then
    NUM_NEIGHBORS_FLAG=""
else
    NUM_NEIGHBORS_FLAG="--num-neighbors ${NUM_NEIGHBORS}"
fi

PNS_THRESH=0.75
PNS_FLAG="--pns"               

TO_DAG_FLAG="--to-dag"                 
JAC_THRESH_FLAG=""             

CAM_PRUNING_FLAG=""           
CAM_PRUNING_CUTOFF="1e-6 1e-4 1e-2"

RISK_EXTRAPOLATION_FLAG="--risk-extrapolation"
BETA=2

NUM_LAYERS=3
HID_DIM=16
NONLIN="leaky-relu"

EPOCHS=30
BATCH_SIZE=128
LR=0.0005
NUM_TRAIN_ITER=300000
SEED=40
DEVICE="cuda"
STOP_CRIT_WIN=100
PLOT_FREQ=10000

H_THRESHOLD=1e-8
MU_INIT=1e-3
LAMBDA_INIT=0

OMEGA_LAMBDA=1e-4
OMEGA_MU=0.9
LR_REINIT=1e-3
EDGE_CLAMP_RANGE=1e-4

NORM_PROD="paths"
SQUARE_PROD_FLAG=""

# Beta values for the loop
betas=(
    "0.0"
    "0.5"
    "1.0"
    "2.0"
    "5.0"
)

# Loop through beta values and submit jobs
for beta in "${betas[@]}"; do
    job_name="${job}/model_beta_${beta}_3_graphs_10_nodes"
    echo "Launching job $job_name"

    sbatch <<EOT
#!/bin/bash
#SBATCH --partition=long       # Specify the partition
#SBATCH --job-name=$job_name   # Job name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4      # Number of CPU cores per task
#SBATCH --mem=16G              # Memory allocation
#SBATCH --time=05:00:00        # Max runtime (hh:mm:ss)
#SBATCH --output=${out_dir}/${job_name}/logs/%x_%j.out      # Standard output log
#SBATCH --error=${out_dir}/${job_name}/logs/%x_%j.err       # Standard error log

module purge
module load miniconda/3
conda activate dag             

# Ensure output directory exists
mkdir -p ${out_dir}/${job_name}/logs

# Run the Python script
set -x
srun python main.py \
    --num-vars ${NUM_VARS} \
    ${TRAIN_FLAG} \
    --data-path ${DATA_PATH} \
    --g-path1 ${G_PATH1} \
    --g-path2 ${G_PATH2} \
    --g-path3 ${G_PATH3} \
    --exp_path ${EXP_PATH} \
    --output-ckpt ${OUTPUT_CKPT} \
    --sem_type ${SEM_TYPE} \
    --sigma-min ${SIGMA_MIN} \
    --sigma-max ${SIGMA_MAX} \
    --load-data ${LOAD_DATA} \
    --n_points ${N_POINTS} \
    ${NUM_NEIGHBORS_FLAG} \
    --pns-thresh ${PNS_THRESH} \
    ${PNS_FLAG} \
    ${TO_DAG_FLAG} \
    ${JAC_THRESH_FLAG} \
    ${CAM_PRUNING_FLAG} \
    --cam-pruning-cutoff ${CAM_PRUNING_CUTOFF} \
    ${RISK_EXTRAPOLATION_FLAG} \
    --beta ${beta} \
    --num-layers ${NUM_LAYERS} \
    --hid-dim ${HID_DIM} \
    --nonlin ${NONLIN} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --num-train-iter ${NUM_TRAIN_ITER} \
    --seed ${SEED} \
    --device ${DEVICE} \
    --stop-crit-win ${STOP_CRIT_WIN} \
    --plot_freq ${PLOT_FREQ} \
    --h-threshold ${H_THRESHOLD} \
    --mu-init ${MU_INIT} \
    --lambda-init ${LAMBDA_INIT} \
    --omega-lambda ${OMEGA_LAMBDA} \
    --omega-mu ${OMEGA_MU} \
    --lr-reinit ${LR_REINIT} \
    --edge-clamp-range ${EDGE_CLAMP_RANGE} \
    --norm-prod ${NORM_PROD} \
    ${SQUARE_PROD_FLAG}
EOT
done
