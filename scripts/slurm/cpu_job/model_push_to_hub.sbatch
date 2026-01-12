#!/bin/bash
# --- SLURM Directives ---
#SBATCH -J launch
#SBATCH -p cpu-farm
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x_%j.log

set -e

uv run python utils/model_push_to_hub.py $@