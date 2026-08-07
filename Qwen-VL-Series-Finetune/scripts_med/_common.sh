#!/usr/bin/env bash

# Shared, location-independent setup for medical evaluation scripts.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEDARC_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
SCRIPT_DIR="$PROJECT_ROOT/scripts_med"
DATA_ASSET_ROOT="${MEDARC_DATA_ROOT:-$MEDARC_ROOT/data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Prefer an externally cloned, unmodified 3DINO checkout. The legacy local
# layout remains supported for existing research environments.
if [[ -n "${THREE_DINO_ROOT:-}" ]]; then
    DEFAULT_THREE_DINO_ROOT="$THREE_DINO_ROOT"
elif [[ -d "$MEDARC_ROOT/third_party/3DINO/dinov2" ]]; then
    DEFAULT_THREE_DINO_ROOT="$MEDARC_ROOT/third_party/3DINO"
elif [[ -d "$MEDARC_ROOT/3DINO/dinov2" ]]; then
    DEFAULT_THREE_DINO_ROOT="$MEDARC_ROOT/3DINO"
else
    DEFAULT_THREE_DINO_ROOT="$MEDARC_ROOT/third_party/3DINO"
fi
