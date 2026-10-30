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


notify () {   # Telegram (Lab Alerts bot) via tg-notify, lab-servers skill; no-op if not installed
  [ -x "$HOME/.local/bin/tg-notify" ] || return 0
  "$HOME/.local/bin/tg-notify" "$@" || true
}

LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-${SLURM_SUBMIT_SCRIPT:-<unknown>}}"

notify --silent ":rocket: JOB START | ${SLURM_JOB_NAME} | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH}\`
log: \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`
status: \`scontrol show job ${SLURM_JOB_ID} -dd\`"

# Prepare modules
module load cuda/12.6
# export CUDA_HOME="/opt/ohpc/pub/apps/cuda/12.8"

# Actual job starting
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
uv run llamafactory-cli train configs/qwen3vl_2b_46k_v1.yaml
rc=$?

# Send completion status
if [ $rc -eq 0 ]; then
  notify ":white_check_mark: DONE ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID})"
else
  notify ":x: FAIL ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID}) exit=${rc}"
fi

exit $rc