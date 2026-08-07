#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts_med/_common.sh
source "$SCRIPT_DIR/_common.sh"

: "${VOLUME_DINS_VERSION:?VOLUME_DINS_VERSION must be set by the version wrapper}"
: "${VOLUME_DINS_ENTRY:?VOLUME_DINS_ENTRY must be set by the version wrapper}"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/output/ct_stage_3/checkpoint-4600}"
MODEL_BASE="${MODEL_BASE:-$PROJECT_ROOT/output/ct_stage_2/checkpoint-600}"
DATA_ROOT="${DATA_ROOT:-$MEDARC_ROOT/dataset/CT-RATE}"
REPORT_DATA_PATH="${REPORT_DATA_PATH:-$DATA_ROOT/validation_reports_cleaned_med_readable.json}"
VQA_DATA_PATH="${VQA_DATA_PATH:-$DATA_ROOT/valid_vqa_cleaned_med_readable.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-$DATA_ASSET_ROOT/CT-RATE}"
TAU="${TAU:-0.9}"
CONTEXTUAL_RATIO="${CONTEXTUAL_RATIO:-0.05}"
ATTENTION_TEMPERATURE="${ATTENTION_TEMPERATURE:-2.0}"
TEXT_WEIGHT="${TEXT_WEIGHT:-0.3}"
TEXT_TEMPERATURE="${TEXT_TEMPERATURE:-0.07}"
VFM_WEIGHT="${VFM_WEIGHT:-0.3}"
VFM_TEMPERATURE="${VFM_TEMPERATURE:-0.2}"
OUTPUT_TAG="tau_${TAU}_contextual_${CONTEXTUAL_RATIO}_attention_temp_${ATTENTION_TEMPERATURE}"
if (( VOLUME_DINS_VERSION >= 2 )); then
    OUTPUT_TAG+="_text_${TEXT_WEIGHT}_text_temp_${TEXT_TEMPERATURE}"
fi
if (( VOLUME_DINS_VERSION >= 3 )); then
    OUTPUT_TAG+="_vfm_${VFM_WEIGHT}_vfm_temp_${VFM_TEMPERATURE}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-output_volume_dins/v${VOLUME_DINS_VERSION}/${OUTPUT_TAG}}"
EVAL_TARGET="${EVAL_TARGET:-all}"
MODE="${MODE:-all}"
GPU="${GPU:-0}"

THREE_DINO_ROOT="$DEFAULT_THREE_DINO_ROOT"
THREE_DINO_CONFIG_FILE="${THREE_DINO_CONFIG_FILE:-$THREE_DINO_ROOT/dinov2/configs/train/vit3d_highres}"
THREE_DINO_CHECKPOINT="${THREE_DINO_CHECKPOINT:-$THREE_DINO_ROOT/3dino_vit_weights.pth}"
THREE_DINO_DEVICE="${THREE_DINO_DEVICE:-cuda}"
THREE_DINO_OUTPUT_DTYPE="${THREE_DINO_OUTPUT_DTYPE:-float16}"
if [[ "$THREE_DINO_CONFIG_FILE" == *.yaml ]]; then
    THREE_DINO_CONFIG_YAML="$THREE_DINO_CONFIG_FILE"
else
    THREE_DINO_CONFIG_YAML="$THREE_DINO_CONFIG_FILE.yaml"
fi

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
    --volume-dins-tau "$TAU"
    --volume-dins-temperature "$ATTENTION_TEMPERATURE"
    --volume-dins-contextual-ratio "$CONTEXTUAL_RATIO"
    --volume-dins-text-weight "$TEXT_WEIGHT"
    --volume-dins-text-temperature "$TEXT_TEMPERATURE"
    --volume-dins-vfm-weight "$VFM_WEIGHT"
    --volume-dins-vfm-temperature "$VFM_TEMPERATURE"
    --three-dino-root "$THREE_DINO_ROOT"
    --three-dino-config-file "$THREE_DINO_CONFIG_FILE"
    --three-dino-checkpoint "$THREE_DINO_CHECKPOINT"
    --three-dino-device "$THREE_DINO_DEVICE"
    --three-dino-output-dtype "$THREE_DINO_OUTPUT_DTYPE"
)

if [[ -n "${LIMIT:-}" ]]; then
    COMMON_ARGS+=(--limit "$LIMIT")
fi

VQA_TYPE_ARGS=()
if [[ -n "${QUESTION_TYPES:-}" ]]; then
    VQA_TYPE_ARGS+=(--question-types "$QUESTION_TYPES")
fi

required_paths=("$IMAGE_FOLDER")
if [[ "$MODE" == "all" || "$MODE" == "infer" ]]; then
    required_paths+=("$MODEL_PATH")
fi
if [[ "$EVAL_TARGET" == "all" || "$EVAL_TARGET" == "reports" ]]; then
    required_paths+=("$REPORT_DATA_PATH")
fi
if [[ "$EVAL_TARGET" == "all" || "$EVAL_TARGET" == "vqa" ]]; then
    required_paths+=("$VQA_DATA_PATH")
fi
if [[ "$VOLUME_DINS_VERSION" == "3" ]]; then
    required_paths+=(
        "$THREE_DINO_ROOT"
        "$THREE_DINO_CONFIG_YAML"
        "$THREE_DINO_CHECKPOINT"
    )
fi
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path does not exist: $required_path" >&2
        exit 1
    fi
done

run_reports() {
    "$PYTHON_BIN" "$VOLUME_DINS_ENTRY" \
        "${COMMON_ARGS[@]}" \
        "$@" \
        --data-path "$REPORT_DATA_PATH" \
        --output-dir "$OUTPUT_ROOT/reports" \
        --max-new-tokens "${REPORT_MAX_NEW_TOKENS:-1024}"
}

run_vqa() {
    "$PYTHON_BIN" "$VOLUME_DINS_ENTRY" \
        "${COMMON_ARGS[@]}" \
        "${VQA_TYPE_ARGS[@]}" \
        "$@" \
        --data-path "$VQA_DATA_PATH" \
        --output-dir "$OUTPUT_ROOT/vqa" \
        --max-new-tokens "${VQA_MAX_NEW_TOKENS:-256}"
}

case "$EVAL_TARGET" in
    all)
        run_reports "$@"
        run_vqa "$@"
        ;;
    reports)
        run_reports "$@"
        ;;
    vqa)
        run_vqa "$@"
        ;;
    *)
        echo "EVAL_TARGET must be one of: all, reports, vqa" >&2
        exit 2
        ;;
esac
