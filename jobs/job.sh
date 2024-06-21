#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4GB
#SBATCH --account=math026082
#SBATCH --array=0-20
#SBATCH --output=%x_%A_%a.out

# Note contructing a compatible environment, have both softce and softce_validation in your current directory and run:
# conda create --name softce_env python
# conda activate softce_env
# pip install -e softce
# pip install -e softce_validation

TASK_NAME=$1
module load lang/python/miniconda/3.9.7
source activate softce_env
python -m scripts.run_task --seed=$SLURM_ARRAY_TASK_ID --task-name=$TASK_NAME
