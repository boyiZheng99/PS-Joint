#!/usr/bin/env bash
#SBATCH --job-name=ps-joint
#SBATCH --account=YOUR_PROJECT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --partition=standard-g
#SBATCH --time=20:00:00
#SBATCH --output=ps_joint_train.out
set -euo pipefail
module load LUMI/23.09 partition/G rocm/5.6.1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export PYTHONNOUSERSITE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
mkdir -p "$MIOPEN_USER_DB_PATH"
: "${PSJOINT_DATA_ROOT:?Set PSJOINT_DATA_ROOT}"
: "${PSJOINT_PRETRAINED_ROOT:?Set PSJOINT_PRETRAINED_ROOT}"
GPUS=${GPUS:-8} bash "$(dirname "$0")/train.sh"
