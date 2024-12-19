#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4GB
#SBATCH --account=math026082
#SBATCH --time=02:00:00
#SBATCH --array=0-10
#SBATCH --output=%x_%A_%a.out

LOSS_NAME="$1"
WIDTH_SIZE=$2
LEARNING_RATE=$3

sleep 5
source ~/miniforge3/bin/activate
sleep 5
conda activate softcvi_validation_env
python -m scripts.run_bnn --seed=$SLURM_ARRAY_TASK_ID --learning-rate=$LEARNING_RATE --loss-name=$LOSS_NAME --width-size=$WIDTH_SIZE --steps=100000
