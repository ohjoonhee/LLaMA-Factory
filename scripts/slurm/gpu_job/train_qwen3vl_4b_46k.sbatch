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

LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-${SLURM_SUBMIT_SCRIPT:-<unknown>}}"

slack_notify ":rocket: JOB START | ${SLURM_JOB_NAME} | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH}\`
log: \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`
status: \`scontrol show job ${SLURM_JOB_ID} -dd\`"

# Prepare modules
module load cuda/12.8
export CUDA_HOME="/opt/ohpc/pub/apps/cuda/12.8"

# Actual job starting
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
uv run llamafactory-cli train configs/slurm/qwen3vl_4b_46k.yaml
uv run llamafactory-cli train configs/slurm/qwen3vl_4b_46k_hpo1.yaml
uv run llamafactory-cli train configs/slurm/qwen3vl_4b_46k_hpo2.yaml
rc=$?

# Send completion status
if [ $rc -eq 0 ]; then
  slack_notify ":white_check_mark: DONE ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID})"
else
  slack_notify ":x: FAIL ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID}) exit=${rc}"
fi

exit $rc