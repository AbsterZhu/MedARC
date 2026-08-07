#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts_med/_common.sh
source "$SCRIPT_DIR/_common.sh"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/output/ct_stage_3/checkpoint-4600}"
MODEL_BASE="${MODEL_BASE:-$PROJECT_ROOT/output/ct_stage_2/checkpoint-600}"
DATA_ROOT="${DATA_ROOT:-$MEDARC_ROOT/dataset/CT-RATE}"
REPORT_DATA_PATH="${REPORT_DATA_PATH:-$DATA_ROOT/validation_reports_cleaned_med_readable.json}"
VQA_DATA_PATH="${VQA_DATA_PATH:-$DATA_ROOT/valid_vqa_cleaned_med_readable.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-$DATA_ASSET_ROOT/CT-RATE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/output/ct_stage_3_eval}"
EVAL_TARGET="${EVAL_TARGET:-all}"
MODE="${MODE:-all}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"

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

VQA_TYPE_ARGS=()
if [[ -n "${QUESTION_TYPES:-}" ]]; then
    VQA_TYPE_ARGS+=(--question-types "$QUESTION_TYPES")
fi

run_reports() {
    "$PYTHON_BIN" src/eval/eval_med.py \
        "${COMMON_ARGS[@]}" \
        --data-path "$REPORT_DATA_PATH" \
        --output-dir "$OUTPUT_ROOT/reports" \
        --max-new-tokens "${REPORT_MAX_NEW_TOKENS:-1024}"
}

run_vqa() {
    "$PYTHON_BIN" src/eval/eval_med.py \
        "${COMMON_ARGS[@]}" \
        "${VQA_TYPE_ARGS[@]}" \
        --data-path "$VQA_DATA_PATH" \
        --output-dir "$OUTPUT_ROOT/vqa" \
        --max-new-tokens "${VQA_MAX_NEW_TOKENS:-256}"
}

case "$EVAL_TARGET" in
    all)
        run_reports
        run_vqa
        ;;
    reports)
        run_reports
        ;;
    vqa)
        run_vqa
        ;;
    *)
        echo "EVAL_TARGET must be one of: all, reports, vqa" >&2
        exit 2
        ;;
esac
