#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts_med/_common.sh
source "$SCRIPT_DIR/_common.sh"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/output/mri_stage_3/checkpoint-2040}"
MODEL_BASE="${MODEL_BASE:-Qwen/Qwen3-VL-4B-Instruct}"
DATA_ROOT="${DATA_ROOT:-$MEDARC_ROOT/dataset/MR-RATE}"
REPORT_DATA_PATH="${REPORT_DATA_PATH:-$DATA_ROOT/mr_rate_med_test_readable.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-$DATA_ASSET_ROOT/MR-RATE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/output/mri_stage_3_eval}"
MODE="${MODE:-all}"
GPU="${GPU:-3}"

export CUDA_VISIBLE_DEVICES="$GPU"

for required_path in "$MODEL_PATH" "$REPORT_DATA_PATH" "$IMAGE_FOLDER"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path does not exist: $required_path" >&2
        exit 1
    fi
done

COMMON_ARGS=(
    --mode "$MODE"
    --model-path "$MODEL_PATH"
    --model-base "$MODEL_BASE"
    --image-folder "$IMAGE_FOLDER"
    --dtype "${DTYPE:-bf16}"
    --device-map "${DEVICE_MAP:-auto}"
    --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
    --num-beams "${NUM_BEAMS:-1}"
    --temperature "${TEMPERATURE:-0.0}"
    --top-p "${TOP_P:-0.9}"
    --nframes "${NFRAMES:-128}"
    --video-min-pixels "${VIDEO_MIN_PIXELS:-100352}"
    --video-max-pixels "${VIDEO_MAX_PIXELS:-147456}"
    --bertscore-model "${BERTSCORE_MODEL:-roberta-large}"
    --bertscore-batch-size "${BERTSCORE_BATCH_SIZE:-16}"
    --num-save-samples "${NUM_SAVE_SAMPLES:-32}"
    --sample-seed "${SAMPLE_SEED:-42}"
)

if [[ -n "${LIMIT:-}" ]]; then
    COMMON_ARGS+=(--limit "$LIMIT")
fi

"$PYTHON_BIN" src/eval/eval_med.py \
    "${COMMON_ARGS[@]}" \
    --data-path "$REPORT_DATA_PATH" \
    --output-dir "$OUTPUT_ROOT/reports" \
    --max-new-tokens "${REPORT_MAX_NEW_TOKENS:-1024}"
