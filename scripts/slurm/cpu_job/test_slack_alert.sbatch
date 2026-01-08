#!/bin/bash
# --- SLURM Directives ---
#SBATCH -J prepare               # Job name
#SBATCH -p cpu-farm               # Partition name
#SBATCH --qos=normal      # Quality of Service
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH -o logs/%x_%j.log         # Standard output file path

set -a
source .env
set +a


slack_notify () {
  local msg="$1"
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"${msg}\"}" \
    "$SLACK_WEBHOOK_URL" >/dev/null
}

# Send “job started” alert
slack_notify ":rocket: Job STARTED jobname=${SLURM_JOB_NAME} jobid=${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
