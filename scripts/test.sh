#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CHECKPOINT=${1:?Usage: test.sh CHECKPOINT [additional test.py arguments]}
shift
cd "$PROJECT_ROOT"
python "$PROJECT_ROOT/tools/test.py" \
  "$PROJECT_ROOT/configs/ps_joint/ps_joint_retinal.py" "$CHECKPOINT" "$@"
