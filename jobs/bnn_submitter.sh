#!/bin/bash

# Example submission script for softcvi_validation root
# chmod +x jobs/bnn_submitter.sh && ./jobs/bnn_submitter.sh

loss_names=("SoftCVI(a=0.75)" "SoftCVI(a=1)" "SNIS-fKL" "ELBO") 
width_sizes=(50 100)  
learning_rates=(5e-5 1e-4 5e-4 1e-3)

for loss_name in "${loss_names[@]}"; do
    for width in "${width_sizes[@]}"; do
      for learning_rate in "${learning_rates[@]}"; do
        sbatch --job-name="${loss_name// /_}_${learning_rate}_${width}" jobs/bnn_job.sh "$loss_name" "$width" "$learning_rate"
      done
    done
  done
