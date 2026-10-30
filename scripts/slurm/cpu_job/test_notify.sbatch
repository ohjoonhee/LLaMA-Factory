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


notify () {   # Telegram (Lab Alerts bot) via tg-notify, lab-servers skill; no-op if not installed
  [ -x "$HOME/.local/bin/tg-notify" ] || return 0
  "$HOME/.local/bin/tg-notify" "$@" || true
}

LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-${SLURM_SUBMIT_SCRIPT:-<unknown>}}"
SCRIPT_ARGS="${ORIG_SBATCH_ARGS:-<none>}"

notify --silent ":rocket: JOB START | ${SLURM_JOB_NAME} | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH} ${SCRIPT_ARGS}\`
log: \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`
status: \`scontrol show job ${SLURM_JOB_ID} -dd\`"