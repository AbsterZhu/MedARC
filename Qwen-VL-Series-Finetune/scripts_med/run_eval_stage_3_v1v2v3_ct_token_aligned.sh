#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts_med/_common.sh
source "$SCRIPT_DIR/_common.sh"

TARGET_VISUAL_KEEP_RATE="${TARGET_VISUAL_KEEP_RATE:-0.787}"

DATA_ROOT="${DATA_ROOT:-$MEDARC_ROOT/dataset/CT-RATE}"
REPORT_DATA_PATH="${REPORT_DATA_PATH:-$DATA_ROOT/validation_reports_cleaned_med_readable.json}"
VQA_DATA_PATH="${VQA_DATA_PATH:-$DATA_ROOT/valid_vqa_cleaned_med_readable.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-$DATA_ASSET_ROOT/CT-RATE}"
THREE_DINO_ROOT="$DEFAULT_THREE_DINO_ROOT"
THREE_DINO_CONFIG_FILE="${THREE_DINO_CONFIG_FILE:-$THREE_DINO_ROOT/dinov2/configs/train/vit3d_highres}"
THREE_DINO_CHECKPOINT="${THREE_DINO_CHECKPOINT:-$THREE_DINO_ROOT/3dino_vit_weights.pth}"
if [[ "$THREE_DINO_CONFIG_FILE" == *.yaml ]]; then
    THREE_DINO_CONFIG_YAML="$THREE_DINO_CONFIG_FILE"
else
    THREE_DINO_CONFIG_YAML="$THREE_DINO_CONFIG_FILE.yaml"
fi

# Keep aligned runs separate from the original tau=0.9 experiments so that the
# evaluator cannot resume predictions produced with a different token budget.
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/ct_stage_3_ct_volume_dins_token_aligned}"
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/output_volume_dins/ct_stage_3_ct_token_aligned}"

export PYTHON_BIN
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/output/ct_stage_3/checkpoint-4600}"
export MODEL_BASE="${MODEL_BASE:-$PROJECT_ROOT/output/ct_stage_2/checkpoint-600}"
export DATA_ROOT REPORT_DATA_PATH VQA_DATA_PATH IMAGE_FOLDER
export THREE_DINO_ROOT THREE_DINO_CONFIG_FILE THREE_DINO_CHECKPOINT
export EVAL_TARGET="${EVAL_TARGET:-all}"
export MODE="${MODE:-all}"
export DTYPE="${DTYPE:-bf16}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export NFRAMES="${NFRAMES:-128}"
export VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-100352}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-147456}"
export REPORT_MAX_NEW_TOKENS="${REPORT_MAX_NEW_TOKENS:-1024}"
export VQA_MAX_NEW_TOKENS="${VQA_MAX_NEW_TOKENS:-256}"
export NUM_BEAMS="${NUM_BEAMS:-1}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-0.9}"
export SAMPLE_SEED="${SAMPLE_SEED:-42}"
export BERTSCORE_MODEL="${BERTSCORE_MODEL:-roberta-large}"
export BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-16}"
export NUM_SAVE_SAMPLES="${NUM_SAVE_SAMPLES:-32}"
export THREE_DINO_DEVICE="${THREE_DINO_DEVICE:-cuda}"
export THREE_DINO_OUTPUT_DTYPE="${THREE_DINO_OUTPUT_DTYPE:-float16}"
export PYTHONUNBUFFERED=1

export CONTEXTUAL_RATIO="${CONTEXTUAL_RATIO:-0.05}"
export ATTENTION_TEMPERATURE="${ATTENTION_TEMPERATURE:-2.0}"
export TEXT_TEMPERATURE="${TEXT_TEMPERATURE:-0.07}"
export VFM_TEMPERATURE="${VFM_TEMPERATURE:-0.2}"

# tau is a cumulative importance-mass threshold, not a direct keep ratio.  The
# fused v2/v3 distributions are flatter than v1, so they need lower thresholds.
# These defaults are inferred from the completed full-set tau=0.9 runs:
#   v1: 78.55% kept; v2: 89.70% kept; v3: 88.34% kept.
# They are initial full-set alignment values and can be overridden independently.
V1_TAU="${V1_TAU:-0.90}"
V2_TAU="${V2_TAU:-0.82}"
V3_TAU="${V3_TAU:-0.84}"
V2_TEXT_WEIGHT="${V2_TEXT_WEIGHT:-0.5}"
V3_TEXT_WEIGHT="${V3_TEXT_WEIGHT:-0.3}"
V3_VFM_WEIGHT="${V3_VFM_WEIGHT:-0.3}"

read -r -a GPU_IDS <<< "${GPUS:-0 1 2}"
if [[ "${#GPU_IDS[@]}" -ne 3 ]]; then
    echo "GPUS must contain exactly three GPU IDs, for example: GPUS='0 1 2'" >&2
    exit 2
fi

for required_path in \
    "$REPORT_DATA_PATH" \
    "$VQA_DATA_PATH" \
    "$IMAGE_FOLDER" \
    "$THREE_DINO_CONFIG_YAML" \
    "$THREE_DINO_CHECKPOINT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path does not exist: $required_path" >&2
        exit 1
    fi
done

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

launch() {
    local method="$1"
    local gpu="$2"
    local script="$3"
    local tau="$4"
    local output_root="$5"
    local text_weight="$6"
    local vfm_weight="$7"
    local log_file="$LOG_DIR/$method.log"
    local pid_file="$LOG_DIR/$method.pid"

    nohup env \
        GPU="$gpu" \
        TAU="$tau" \
        OUTPUT_ROOT="$output_root" \
        TEXT_WEIGHT="$text_weight" \
        VFM_WEIGHT="$vfm_weight" \
        bash "$SCRIPT_DIR/$script" \
        >"$log_file" 2>&1 &
    local pid=$!
    printf '%s\n' "$pid" >"$pid_file"
    echo "$method: GPU $gpu, tau $tau, PID $pid, log $log_file"
}

launch "Volume_DINS_V1_aligned" "${GPU_IDS[0]}" \
    "eval_stage_3_MedARC_V1.sh" "$V1_TAU" \
    "$OUTPUT_BASE/v1/target_keep_${TARGET_VISUAL_KEEP_RATE}_tau_${V1_TAU}_contextual_${CONTEXTUAL_RATIO}_attention_temp_${ATTENTION_TEMPERATURE}" \
    "0.0" "0.0"

launch "Volume_DINS_V2_aligned" "${GPU_IDS[1]}" \
    "eval_stage_3_MedARC_V2.sh" "$V2_TAU" \
    "$OUTPUT_BASE/v2/target_keep_${TARGET_VISUAL_KEEP_RATE}_tau_${V2_TAU}_contextual_${CONTEXTUAL_RATIO}_attention_temp_${ATTENTION_TEMPERATURE}_text_${V2_TEXT_WEIGHT}_text_temp_${TEXT_TEMPERATURE}" \
    "$V2_TEXT_WEIGHT" "0.0"

launch "Volume_DINS_V3_aligned" "${GPU_IDS[2]}" \
    "eval_stage_3_MedARC_V3.sh" "$V3_TAU" \
    "$OUTPUT_BASE/v3/target_keep_${TARGET_VISUAL_KEEP_RATE}_tau_${V3_TAU}_contextual_${CONTEXTUAL_RATIO}_attention_temp_${ATTENTION_TEMPERATURE}_text_${V3_TEXT_WEIGHT}_text_temp_${TEXT_TEMPERATURE}_vfm_${V3_VFM_WEIGHT}_vfm_temp_${VFM_TEMPERATURE}" \
    "$V3_TEXT_WEIGHT" "$V3_VFM_WEIGHT"

echo "All three token-aligned volume-DINS evaluations have been started."
echo "Target visual keep rate: $TARGET_VISUAL_KEEP_RATE"
echo "Configured taus: v1=$V1_TAU, v2=$V2_TAU, v3=$V3_TAU"
