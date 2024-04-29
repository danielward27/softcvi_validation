#!/bin/bash

#SBATCH --job-name=run_two_moons
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=30:00
#SBATCH --mem=1GB
#SBATCH --account=math026082
#SBATCH --array=0-100

# Example submission: sbatch jobs/run_two_moons.sh
module load lang/python/miniconda/3.9.7
source activate flowjax_env
python -m scripts.run_task --seed=$SLURM_ARRAY_TASK_ID --task-name="two_moons"