#!/bin/sh
# Adapt the partition, account, image cache, and data path to your cluster.
# Apptainer is used because Docker is commonly unavailable on compute nodes.
#SBATCH --job-name=truezero
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -eu

IMAGE="${TRUEZERO_IMAGE:-docker://ghcr.io/andreaslam/truezero:latest}"
DATA="${TRUEZERO_DATA_DIR:?Set TRUEZERO_DATA_DIR to persistent storage}"
mkdir -p "$DATA/logs"

apptainer exec --nv \
  --bind "$DATA:/data" \
  "$IMAGE" \
  /opt/truezero/bin/main 2>&1 | tee "$DATA/logs/slurm-${SLURM_JOB_ID:-local}.log"
