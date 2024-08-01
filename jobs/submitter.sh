#!/bin/bash

# Example submission from softcvi_validation root
# chmod +x jobs/submitter.sh && ./jobs/submitter.sh

declare -A tasks
tasks=(
  ["garch"]="01:30:00"
  ["eight_schools"]="01:15:00"
  ["linear_regression"]="00:45:00"
  ["slcp"]="01:30:00"
)

distributions=("proposal" "posterior")

for task in "${!tasks[@]}"; do
  run_time=${tasks[$task]}
  for dist in "${distributions[@]}"; do
    sbatch --job-name="${task}_${dist}" --time=$run_time jobs/job.sh $task $dist
  done
done