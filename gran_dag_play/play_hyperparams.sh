#!/usr/bin/env bash

# Example: Modify these variables as desired
NUM_VARS=10
TRAIN_FLAG="--train"          # Use "--train" to enable training mode, or "" to disable
DATA_PATH="data/"
G_PATH1="data/custom_G1.gpickle"
G_PATH2="data/custom_G2.gpickle"
EXP_PATH="experiments/"
OUTPUT_CKPT="models/"

SEM_TYPE="non_linear_gaussian"
SIGMA_MIN=0.4
SIGMA_MAX=0.8
LOAD_DATA="True"
N_POINTS=2000

NUM_NEIGHBORS=5
PNS_THRESH=0.7
PNS_FLAG="--pns"               # Use "--pns" or "" to toggle

TO_DAG_FLAG=""                 # e.g. "--to-dag" if you want to enable it
JAC_THRESH_FLAG=""             # e.g. "--jac_thresh" if you want to enable it

CAM_PRUNING_FLAG=""            # e.g. "--cam-pruning"
CAM_PRUNING_CUTOFF="1e-6 1e-4 1e-2"

RISK_EXTRAPOLATION_FLAG=""     # e.g. "--risk-extrapolation"
BETA=0.5

NUM_LAYERS=3
HID_DIM=16
NONLIN="leaky-relu"

EPOCHS=30
BATCH_SIZE=128
LR=0.0005
NUM_TRAIN_ITER=50000
SEED=42
DEVICE="cuda"
STOP_CRIT_WIN=200
PLOT_FREQ=5000

H_THRESHOLD=1e-8
MU_INIT=1e-3
LAMBDA_INIT=0

OMEGA_LAMBDA=1e-4
OMEGA_MU=0.9
LR_REINIT=1e-3
EDGE_CLAMP_RANGE=1e-4

NORM_PROD="paths"
SQUARE_PROD_FLAG="--square-prod"  # Use "--square-prod" or "" to toggle


# Run the python script with the specified arguments
python train.py \
    --num-vars ${NUM_VARS} \
    ${TRAIN_FLAG} \
    --data-path ${DATA_PATH} \
    --g-path1 ${G_PATH1} \
    --g-path2 ${G_PATH2} \
    --exp_path ${EXP_PATH} \
    --output-ckpt ${OUTPUT_CKPT} \
    --sem_type ${SEM_TYPE} \
    --sigma-min ${SIGMA_MIN} \
    --sigma-max ${SIGMA_MAX} \
    --load-data ${LOAD_DATA} \
    --n_points ${N_POINTS} \
    --num-neighbors ${NUM_NEIGHBORS} \
    --pns-thresh ${PNS_THRESH} \
    ${PNS_FLAG} \
    ${TO_DAG_FLAG} \
    ${JAC_THRESH_FLAG} \
    ${CAM_PRUNING_FLAG} \
    --cam-pruning-cutoff ${CAM_PRUNING_CUTOFF} \
    ${RISK_EXTRAPOLATION_FLAG} \
    --beta ${BETA} \
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
