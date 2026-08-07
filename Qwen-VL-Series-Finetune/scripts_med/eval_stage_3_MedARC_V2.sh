#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VOLUME_DINS_VERSION=2
export VOLUME_DINS_ENTRY="src/eval/eval_med_MedARC_V2.py"
exec "$SCRIPT_DIR/eval_stage_3_volume_dins_common.sh" "$@"
