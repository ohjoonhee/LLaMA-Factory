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

uv run llamafactory-cli train configs/slurm/qwen3vl_4b.yaml