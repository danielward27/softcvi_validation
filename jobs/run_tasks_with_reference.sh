#!/bin/bash

#SBATCH --job-name=with_reference_tasks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --time=4:00:00
#SBATCH --account=math026082
#SBATCH --array=0-20

# Example submission: sbatch jobs/run_tasks_with_reference.sh
module load lang/python/miniconda/3.9.7
source activate softce_env

# Note contructing a compatible environment, have both softce and softce_validation in your current directory and run:
# conda create --name softce_env python
# conda activate softce_env
# pip install -e softce
# pip install -e softce_validation


task_names=("eight_schools" "multimodal_gaussian_flexible" "multimodal_gaussian_inflexible" "slcp" "linear_regression")

for task_name in "${task_names[@]}"; do
    python -m scripts.run_task --seed=$SLURM_ARRAY_TASK_ID --task-name=$task_name
done
