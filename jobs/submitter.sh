#!/bin/bash

# Example submission from softce_validation root
# chmod +x jobs/submitter.sh && ./jobs/submitter.sh

declare -A tasks
tasks=(
  ["garch"]="01:30:00"
  # ["eight_schools"]="01:00:00"
  # ["linear_regression"]="01:00:00"
  # ["slcp"]="01:30:00"
)

for task in "${!tasks[@]}"; do
  run_time=${tasks[$task]}
  sbatch --job-name=$task --time=$run_time jobs/job.sh $task
done
