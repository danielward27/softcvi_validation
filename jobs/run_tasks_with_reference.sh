#!/bin/bash

#SBATCH --job-name=run_tasks_with_reference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --account=math026082
#SBATCH --array=0-20

# Example submission: sbatch jobs/run_tasks_with_reference.sh
module load lang/python/miniconda/3.9.7
source activate flowjax_env

task_names=("eight_schools" "multimodal_gaussian_flexible" "multimodal_gaussian_inflexible" "slcp")

for task_name in "${task_names[@]}"; do
    python -m scripts.run_task --seed=$SLURM_ARRAY_TASK_ID --task-name=$task_name
done
