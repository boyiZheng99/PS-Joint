#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
GPUS=${GPUS:-8}
WORK_DIR=${WORK_DIR:-$PROJECT_ROOT/work_dirs/ps_joint_retinal}
cd "$PROJECT_ROOT"
bash "$PROJECT_ROOT/tools/dist_train.sh" \
  "$PROJECT_ROOT/configs/ps_joint/ps_joint_retinal.py" "$GPUS" \
  --work-dir "$WORK_DIR" "$@"
