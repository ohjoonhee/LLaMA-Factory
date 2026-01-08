#!/bin/bash
#SBATCH -J launch
#SBATCH -p gpu-4farm
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log
#SBATCH --account=gpu

set -a
source .env
set +a


slack_notify () {
  local msg="$1"
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"${msg}\"}" \
    "$SLACK_WEBHOOK_URL" >/dev/null
}

# Send slack alert
slack_notify ":rocket: JOB START | ${SLURM_JOB_NAME} | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}"

# Actual job starting
uv run llamafactory-cli train configs/slurm/qwen3vl_4b.yaml
rc=$?

# Send completion status
if [ $rc -eq 0 ]; then
  slack_notify ":white_check_mark: DONE ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID})"
else
  slack_notify ":x: FAIL ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID}) exit=${rc}"
fi

exit $rc