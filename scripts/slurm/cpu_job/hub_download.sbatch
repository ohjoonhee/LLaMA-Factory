#!/bin/bash
# --- SLURM Directives ---
#SBATCH -J launch
#SBATCH -p cpu-farm
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x_%j.log

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
SCRIPT_ARGS="${ORIG_SBATCH_ARGS:-<none>}"

slack_notify ":rocket: JOB START | ${SLURM_JOB_NAME} | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH} ${SCRIPT_ARGS}\`
log: \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`
status: \`scontrol show job ${SLURM_JOB_ID} -dd\`"

uvx --from huggingface_hub hf download $@
rc=$?

# Send completion status
if [ $rc -eq 0 ]; then
  slack_notify ":white_check_mark: DONE ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID})"
else
  slack_notify ":x: FAIL ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID}) exit=${rc}"
fi

exit $rc
