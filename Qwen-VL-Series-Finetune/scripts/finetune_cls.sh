#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "$SCRIPT_DIR/_common.sh"

# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen3.5-4B"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATA_PATH="${DATA_PATH:-}"
IMAGE_FOLDER="${IMAGE_FOLDER:-}"
EVAL_PATH="${EVAL_PATH:-$DATA_PATH}"
EVAL_IMAGE_FOLDER="${EVAL_IMAGE_FOLDER:-$IMAGE_FOLDER}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/qwen_cls}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$PROJECT_ROOT/scripts/zero3.json}"
require_env DATA_PATH
require_env IMAGE_FOLDER
require_path "$DATA_PATH" "Training manifest"
require_path "$IMAGE_FOLDER" "Training image root"
require_path "$EVAL_PATH" "Evaluation manifest"
require_path "$EVAL_IMAGE_FOLDER" "Evaluation image root"

# If you want to set the min pixels and max pixels for Qwen3-VL, You should set as (N * 32 * 32)

#  loss_type should be one of "cross_entropy", "focal_loss", "class_balanced_cross_entropy", or "class_balanced_focal_loss".
# If you switch MODEL_NAME to a Qwen3.5 model, set `--disable_flash_attn2 True`.
# Flash Attention 2 raised CUDA errors for the Qwen3.5 series in local tests, so SDPA is the stable path for now.

"$DEEPSPEED_BIN" "$PROJECT_ROOT/src/train/train_cls.py" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --eval_path "$EVAL_PATH" \
    --eval_image_folder "$EVAL_IMAGE_FOLDER" \
    --freeze_llm True \
    --freeze_vision_tower False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --loss_type "cross_entropy" \
    --num_labels 2 \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --head_lr 4e-5 \
    --vision_lr 6e-6 \
    --merger_lr 2e-5 \
    --weight_decay 0.02 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --eval_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_f1" \
    --greater_is_better True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --dataloader_num_workers 4
