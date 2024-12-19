#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4GB
#SBATCH --account=math026082
#SBATCH --array=0-50
#SBATCH --output=%x_%A_%a.out

TASK_NAME=$1
NEGATIVE_DIST=$2
N_PARTICLES=$3

source ~/miniforge3/bin/activate
sleep 5  # Ensure time to activate
conda activate softcvi_validation_env
python -m scripts.run_task --seed=$SLURM_ARRAY_TASK_ID --task-name=$TASK_NAME --negative-distribution=$NEGATIVE_DIST --n-particles=$N_PARTICLES
