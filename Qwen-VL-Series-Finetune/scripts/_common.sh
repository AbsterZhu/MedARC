#!/usr/bin/env bash

# Shared, location-independent setup for all training helper scripts.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEDARC_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"

require_env() {
    local variable_name="$1"
    if [[ -z "${!variable_name:-}" ]]; then
        echo "Required environment variable is not set: $variable_name" >&2
        exit 2
    fi
}

require_path() {
    local path="$1"
    local description="$2"
    if [[ ! -e "$path" ]]; then
        echo "$description does not exist: $path" >&2
        exit 2
    fi
}
