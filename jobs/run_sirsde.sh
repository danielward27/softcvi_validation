#!/bin/bash

#SBATCH --job-name=run_task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00
#SBATCH --mem=1GB
#SBATCH --account=math026082
#SBATCH --array=0-100

# Example submission: sbatch jobs/run_sirsde.sh
module load lang/python/miniconda/3.9.7
source activate flowjax_env
python -m scripts.run_task --seed=$SLURM_ARRAY_TASK_ID --task-name="sirsde"