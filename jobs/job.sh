#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4GB
#SBATCH --account=math026082
#SBATCH --array=0-20
#SBATCH --output=%x_%A_%a.out

TASK_NAME=$1
NEGATIVE_DIST=$2
module load lang/python/miniconda/3.9.7
source activate softcvi_env
python -m scripts.run_task --seed=$SLURM_ARRAY_TASK_ID --task-name=$TASK_NAME --negative-distribution=$NEGATIVE_DIST
