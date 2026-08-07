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
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/test_grpo}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$PROJECT_ROOT/scripts/zero3.json}"
require_env DATA_PATH
require_env IMAGE_FOLDER
require_path "$DATA_PATH" "Training manifest"
require_path "$IMAGE_FOLDER" "Image/video root"

# If you want to set the min pixels and max pixels for Qwen3-VL, You should set as (N * 32 * 32)

# If you switch MODEL_NAME to a Qwen3.5 model, set `--disable_flash_attn2 True`.
# Flash Attention 2 raised CUDA errors for the Qwen3.5 series in local tests, so SDPA is the stable path for now.

# Optional Liger GRPO loss variant (requires --use_liger_loss True and liger-kernel >= 0.8.0):
#   --liger_grpo_loss_type <name>   where <name> is one of:
#     grpo | bnpo | dr_grpo | dapo (default) | cispo | sapo | luspo
# Note: dr_grpo additionally requires --max_completion_length to be set.

"$DEEPSPEED_BIN" "$PROJECT_ROOT/src/train/train_grpo.py" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --use_liger_loss True \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_completion_length 256 \
    --max_prompt_length 512 \
    --image_min_pixels $((128 * 28 * 28)) \
    --image_max_pixels $((256 * 28 * 28)) \
    --learning_rate 5e-6 \
    --remove_unused_columns False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --dataloader_num_workers 4
