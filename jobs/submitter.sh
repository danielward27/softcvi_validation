#!/bin/bash

# Example submission from softcvi_validation root
# chmod +x jobs/submitter.sh && ./jobs/submitter.sh

# Declare tasks with their runtime
declare -A tasks=(
  ["garch"]="07:00:00"
  ["eight_schools"]="06:00:00"
  ["linear_regression"]="04:00:00"
  ["slcp"]="07:00:00"
)

# Define distributions and particle counts
distributions=("proposal", "posterior")
n_particles=(2 4 8 16 32 64)

# Iterate over tasks, distributions, and particle counts
for task in "${!tasks[@]}"; do
  run_time="${tasks[$task]}"
  for dist in "${distributions[@]}"; do
    for k in "${n_particles[@]}"; do
      sbatch --job-name="${task}_${dist}" --time="$run_time" jobs/job.sh "$task" "$dist" "$k"
    done
  done
done
