#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "$SCRIPT_DIR/_common.sh"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-VL-7B-Instruct}"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

MODEL_PATH="${MODEL_PATH:-}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-$PROJECT_ROOT/output/merged_lora}"
require_env MODEL_PATH
require_path "$MODEL_PATH" "LoRA checkpoint"

"$PYTHON_BIN" "$PROJECT_ROOT/src/merge_lora_weights.py" \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_NAME" \
    --save-model-path "$SAVE_MODEL_PATH" \
    --safe-serialization
