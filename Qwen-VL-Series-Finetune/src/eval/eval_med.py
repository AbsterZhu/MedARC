#!/usr/bin/env python3
"""Generate and score CT-RATE VQA or report-generation predictions.

The evaluator understands the LLaVA-style ``conversations`` records used by
both medical validation files.  VQA records are flattened into independent
question/answer examples; every question is paired with the record's CT volume.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class EvalExample:
    sample_id: str
    record_id: str
    turn_index: int
    video: str
    question: str
    reference: str
    question_type: str


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}.")
    return data


def load_eval_examples(
    data_path: str | Path,
    question_types: set[str] | None = None,
    limit: int | None = None,
) -> tuple[list[EvalExample], dict[str, Any]]:
    """Parse both CT-RATE validation schemas into one example representation."""
    path = Path(data_path).expanduser().resolve()
    records = _read_json(path)
    examples: list[EvalExample] = []
    seen_ids: Counter[str] = Counter()

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {record_index} in {path} is not an object.")
        record_id = str(record.get("id", record_index))
        video = record.get("video")
        conversations = record.get("conversations")
        if not isinstance(video, str) or not video.strip():
            raise ValueError(f"Record {record_id!r} has no valid 'video' path.")
        if not isinstance(conversations, list) or len(conversations) % 2:
            raise ValueError(
                f"Record {record_id!r} must contain an even-length 'conversations' list."
            )

        for turn_index in range(0, len(conversations), 2):
            user_turn = conversations[turn_index]
            assistant_turn = conversations[turn_index + 1]
            if user_turn.get("from") != "human" or assistant_turn.get("from") != "gpt":
                raise ValueError(
                    f"Record {record_id!r}, turns {turn_index}-{turn_index + 1} are not "
                    "an alternating human/gpt pair."
                )
            question = user_turn.get("value")
            reference = assistant_turn.get("value")
            if not isinstance(question, str) or not isinstance(reference, str):
                raise ValueError(f"Record {record_id!r}, turn {turn_index} has non-string text.")

            question_type = str(user_turn.get("type") or "report")
            if question_types and question_type not in question_types:
                continue

            # The volume is explicitly added again when constructing every independent prompt.
            question = re.sub(r"\s*<video>\s*", "\n", question).strip()
            base_sample_id = f"{record_id}::turn_{turn_index // 2}"
            duplicate_number = seen_ids[base_sample_id]
            seen_ids[base_sample_id] += 1
            sample_id = (
                base_sample_id
                if duplicate_number == 0
                else f"{base_sample_id}::duplicate_{duplicate_number}"
            )
            examples.append(
                EvalExample(
                    sample_id=sample_id,
                    record_id=record_id,
                    turn_index=turn_index // 2,
                    video=video,
                    question=question,
                    reference=reference.strip(),
                    question_type=question_type,
                )
            )
            if limit is not None and len(examples) >= limit:
                break
        if limit is not None and len(examples) >= limit:
            break

    summary = {
        "data_path": str(path),
        "num_records": len(records),
        "num_examples": len(examples),
        "num_unique_videos": len({example.video for example in examples}),
        "question_types": dict(sorted(Counter(e.question_type for e in examples).items())),
        "limit": limit,
    }
    return examples, summary


def _resolve_video_path(video: str, image_folder: Path) -> Path:
    video_path = Path(video).expanduser()
    if not video_path.is_absolute():
        video_path = image_folder / video_path
    return video_path.resolve()


def _build_prompt(question: str, model_type: str) -> str:
    from constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        LLAVA_VIDEO_TOKEN,
        SYSTEM_MESSAGE,
    )
    from dataset.data_utils import replace_image_tokens, use_default_system_message

    user_content = replace_image_tokens(
        f"{LLAVA_VIDEO_TOKEN}\n{question}", is_video=True
    )
    pieces: list[str] = []
    if SYSTEM_MESSAGE and use_default_system_message(model_type):
        pieces.append(
            f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        )
    pieces.append(
        f"{DEFAULT_IM_START_TOKEN}user\n{user_content}{DEFAULT_IM_END_TOKEN}\n"
        f"{DEFAULT_IM_START_TOKEN}assistant\n"
    )
    return "".join(pieces)


def _normalise_non_lora_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    state_dict = {
        (key[11:] if key.startswith("base_model.") else key): value
        for key, value in state_dict.items()
    }
    if any(key.startswith("model.model.") for key in state_dict):
        state_dict = {
            (key[6:] if key.startswith("model.") else key): value
            for key, value in state_dict.items()
        }
    return state_dict


def _prepare_qwen3_vl_generation_position_ids(
    model, inputs: dict[str, Any], model_type: str
) -> None:
    """Prepare Qwen3-VL RoPE positions without changing the visual grid input.

    Qwen3-VL represents one video as a separate token group per temporal grid
    step, while ``video_grid_thw`` contains one aggregate row per video.  The
    training forward patch expands that row before computing RoPE positions,
    but Transformers computes generation positions before entering the patched
    forward.  Use an expanded copy for that calculation and retain the original
    grid for the visual encoder.
    """
    if model_type not in {"qwen3_vl", "qwen3_vl_moe"}:
        return
    if "position_ids" in inputs or "video_grid_thw" not in inputs:
        return

    video_grid_thw = inputs["video_grid_thw"]
    if video_grid_thw is None or video_grid_thw.numel() == 0:
        return

    temporal_steps = video_grid_thw[:, 0].long()
    rope_video_grid_thw = video_grid_thw.repeat_interleave(temporal_steps, dim=0).clone()
    rope_video_grid_thw[:, 0] = 1

    rope_inputs = dict(inputs)
    rope_inputs["video_grid_thw"] = rope_video_grid_thw
    inputs["position_ids"] = model._prepare_position_ids_for_generation(
        inputs["input_ids"], rope_inputs
    )


def load_eval_model(args: argparse.Namespace):
    """Load a full checkpoint or merge a stage-3 LoRA plus non-LoRA weights."""
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoProcessor

    from model.load_model import load_qwen_vl_generation_model

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    dtype: Any = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    load_kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "dtype": dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
    }
    adapter_config_path = model_path / "adapter_config.json"
    adapter_weights_exist = any(
        (model_path / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )

    if adapter_config_path.exists() and adapter_weights_exist:
        with adapter_config_path.open("r", encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        configured_base = adapter_config.get("base_model_name_or_path")
        model_base = args.model_base or configured_base
        if not model_base:
            raise ValueError(
                "This is a LoRA checkpoint. Pass --model-base or set "
                "base_model_name_or_path in adapter_config.json."
            )
        base_path = Path(model_base).expanduser()
        if not base_path.is_absolute() and not base_path.exists():
            project_candidate = Path(__file__).resolve().parents[2] / base_path
            if project_candidate.exists():
                base_path = project_candidate
        config = AutoConfig.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)
        print(f"Loading base model: {base_path}", flush=True)
        model = load_qwen_vl_generation_model(
            str(base_path), config=config, **load_kwargs
        )

        non_lora_path = model_path / "non_lora_state_dict.bin"
        if not non_lora_path.exists():
            raise FileNotFoundError(
                f"Stage-3 checkpoint is missing required non-LoRA merger weights: {non_lora_path}"
            )
        non_lora_state = torch.load(non_lora_path, map_location="cpu", weights_only=True)
        non_lora_state = _normalise_non_lora_state_dict(non_lora_state)
        incompatible = model.load_state_dict(non_lora_state, strict=False)
        print(
            "Loaded non-LoRA weights "
            f"({len(non_lora_state)} tensors; unexpected={len(incompatible.unexpected_keys)}).",
            flush=True,
        )
        model = PeftModel.from_pretrained(model, model_path)
        if not args.no_merge_lora:
            model = model.merge_and_unload()
    else:
        config = AutoConfig.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)
        model = load_qwen_vl_generation_model(
            str(model_path), config=config, **load_kwargs
        )

    model.eval()
    model.config.use_cache = True
    return processor, model, config.model_type, model_path


def _model_input_device(model):
    import torch

    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")


def _move_inputs_to_device(inputs: dict[str, Any], device) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _sync_cuda(torch_module: Any) -> None:
    """Synchronize queued CUDA work so generation wall time is accurate."""
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _clean_prediction(text: str, keep_reasoning: bool) -> str:
    text = text.strip()
    if not keep_reasoning:
        text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.replace("<|im_end|>", "").strip()


def run_inference(
    args: argparse.Namespace,
    examples: Sequence[EvalExample],
    predictions_path: Path,
) -> None:
    existing = _load_prediction_map(predictions_path) if args.resume else {}
    if predictions_path.exists() and not args.resume and not args.overwrite_predictions:
        raise FileExistsError(
            f"{predictions_path} already exists. Use --resume or --overwrite-predictions."
        )
    file_mode = "w" if args.overwrite_predictions and not args.resume else "a"
    pending = [example for example in examples if example.sample_id not in existing]
    print(
        f"Inference examples: total={len(examples)}, existing={len(existing)}, pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return

    import torch
    from tqdm import tqdm

    from dataset.data_utils import get_qwen_multimodal_settings
    from dataset.sft_dataset_med import get_nifti_video_info

    processor, model, model_type, loaded_model_path = load_eval_model(args)
    _, image_patch_size, return_video_metadata = get_qwen_multimodal_settings(
        str(loaded_model_path)
    )
    input_device = _model_input_device(model)
    image_folder = Path(args.image_folder).expanduser().resolve()

    @lru_cache(maxsize=args.video_cache_size)
    def load_video(video_path_string: str):
        return get_nifti_video_info(
            video_path_string,
            args.video_min_pixels,
            args.video_max_pixels,
            args.video_resized_width,
            args.video_resized_height,
            args.fps,
            image_patch_size,
            nframes=args.nframes,
            return_video_metadata=return_video_metadata,
        )

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open(file_mode, encoding="utf-8") as output_handle:
        for example in tqdm(pending, desc="Generating", unit="example"):
            video_path = _resolve_video_path(example.video, image_folder)
            if not video_path.exists():
                raise FileNotFoundError(
                    f"CT volume for {example.sample_id} does not exist: {video_path}"
                )
            video_payload, video_kwargs = load_video(str(video_path))
            processor_kwargs: dict[str, Any] = {
                "text": [_build_prompt(example.question, model_type)],
                "images": None,
                "padding": False,
                "do_resize": False,
                "return_tensors": "pt",
                **video_kwargs,
            }
            if return_video_metadata:
                video_data, video_metadata = video_payload
                processor_kwargs["videos"] = [video_data]
                processor_kwargs["video_metadata"] = [video_metadata]
            else:
                processor_kwargs["videos"] = [video_payload]

            inputs = processor(**processor_kwargs)
            inputs = _move_inputs_to_device(dict(inputs), input_device)
            _prepare_qwen3_vl_generation_position_ids(model, inputs, model_type)
            prompt_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                _sync_cuda(torch)
                generation_start = time.perf_counter()
                generated_ids = model.generate(**inputs, **generation_kwargs)
                _sync_cuda(torch)
                generation_seconds = time.perf_counter() - generation_start
            new_ids = generated_ids[:, prompt_length:]
            prediction = processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            generated_tokens = int(new_ids.shape[1])
            row = {
                **asdict(example),
                "video_path": str(video_path),
                "prediction": _clean_prediction(prediction, args.keep_reasoning),
                "generation": {
                    "generation_seconds": generation_seconds,
                    "generated_tokens": generated_tokens,
                    "generated_tokens_per_second": (
                        generated_tokens / generation_seconds
                        if generation_seconds > 0
                        else None
                    ),
                },
            }
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_handle.flush()

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            for field in ("sample_id", "reference", "prediction"):
                if not isinstance(row.get(field), str):
                    raise ValueError(f"Missing string field {field!r} on {path}:{line_number}")
            rows.append(row)
    return rows


def _load_prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _load_predictions(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {path}")
        result[sample_id] = row
    return result


class _NoWordNet:
    @staticmethod
    def synsets(_word: str) -> list[Any]:
        return []


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def compute_metrics(
    rows: Sequence[dict[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    try:
        from bert_score import score as bert_score
        from nltk.corpus import wordnet
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import wordpunct_tokenize
        from rouge_score.rouge_scorer import RougeScorer
    except ImportError as exc:
        raise ImportError(
            "Metric dependencies are missing. Install requirements.txt in the arc environment "
            "(bert-score, nltk, and rouge-score)."
        ) from exc

    if not rows:
        raise ValueError("Cannot compute metrics for an empty predictions file.")

    references = [row["reference"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    ref_tokens = [wordpunct_tokenize(text.lower()) for text in references]
    pred_tokens = [wordpunct_tokenize(text.lower()) for text in predictions]
    smoothing = SmoothingFunction().method1
    rouge_scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    use_wordnet = args.meteor_wordnet
    if use_wordnet:
        try:
            wordnet.ensure_loaded()
        except LookupError:
            print(
                "Warning: NLTK WordNet data is unavailable; METEOR will run without synonym matching.",
                file=sys.stderr,
            )
            use_wordnet = False
    meteor_wordnet = wordnet if use_wordnet else _NoWordNet()

    per_sample: list[dict[str, float]] = []
    for reference, prediction, reference_tokens, prediction_tokens in zip(
        references, predictions, ref_tokens, pred_tokens
    ):
        rouge = rouge_scorer.score(reference, prediction)
        if reference_tokens and prediction_tokens:
            meteor = float(
                meteor_score([reference_tokens], prediction_tokens, wordnet=meteor_wordnet)
            )
            bleu_1 = float(
                sentence_bleu(
                    [reference_tokens], prediction_tokens, weights=(1.0,), smoothing_function=smoothing
                )
            )
            bleu_2 = float(
                sentence_bleu(
                    [reference_tokens],
                    prediction_tokens,
                    weights=(0.5, 0.5),
                    smoothing_function=smoothing,
                )
            )
        else:
            meteor = bleu_1 = bleu_2 = 0.0
        per_sample.append(
            {
                "BL-1": bleu_1,
                "BL-2": bleu_2,
                "ROUGE-1": float(rouge["rouge1"].fmeasure),
                "ROUGE-2": float(rouge["rouge2"].fmeasure),
                "ROUGE-L": float(rouge["rougeL"].fmeasure),
                "ROUGE": float(rouge["rougeL"].fmeasure),
                "METEOR": meteor,
            }
        )

    print(f"Computing BERTScore for {len(rows)} predictions...", flush=True)
    bert_p, bert_r, bert_f1 = bert_score(
        predictions,
        references,
        model_type=args.bertscore_model,
        batch_size=args.bertscore_batch_size,
        device=args.bertscore_device,
        verbose=True,
        rescale_with_baseline=args.bertscore_rescale_with_baseline,
    )
    for index, scores in enumerate(zip(bert_p.tolist(), bert_r.tolist(), bert_f1.tolist())):
        precision, recall, f1 = scores
        per_sample[index].update(
            {
                "BERTScore-P": float(precision),
                "BERTScore-R": float(recall),
                "BERTScore-F1": float(f1),
                "BERTScore": float(f1),
            }
        )

    def aggregate(indices: Sequence[int]) -> dict[str, Any]:
        group_references = [[ref_tokens[index]] for index in indices]
        group_predictions = [pred_tokens[index] for index in indices]
        valid = [
            position
            for position, (reference, prediction) in enumerate(
                zip(group_references, group_predictions)
            )
            if reference[0] and prediction
        ]
        if valid:
            bleu_1 = corpus_bleu(
                [group_references[i] for i in valid],
                [group_predictions[i] for i in valid],
                weights=(1.0,),
                smoothing_function=smoothing,
            )
            bleu_2 = corpus_bleu(
                [group_references[i] for i in valid],
                [group_predictions[i] for i in valid],
                weights=(0.5, 0.5),
                smoothing_function=smoothing,
            )
        else:
            bleu_1 = bleu_2 = 0.0
        result = {
            "num_samples": len(indices),
            "BL-1": float(bleu_1),
            "BL-2": float(bleu_2),
        }
        for metric_name in (
            "ROUGE-1",
            "ROUGE-2",
            "ROUGE-L",
            "ROUGE",
            "METEOR",
            "BERTScore-P",
            "BERTScore-R",
            "BERTScore-F1",
            "BERTScore",
        ):
            result[metric_name] = _mean(per_sample[index][metric_name] for index in indices)
        return result

    metrics: dict[str, Any] = {
        "overall": aggregate(list(range(len(rows)))),
        "by_question_type": {},
        "config": {
            "bleu_level": "corpus",
            "lowercase_wordpunct_tokenization": True,
            "rouge_use_stemmer": True,
            "rouge_alias": "ROUGE is ROUGE-L F1",
            "meteor_wordnet": use_wordnet,
            "bertscore_model": args.bertscore_model,
            "bertscore_rescale_with_baseline": args.bertscore_rescale_with_baseline,
        },
    }
    type_to_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        type_to_indices.setdefault(str(row.get("question_type") or "unknown"), []).append(index)
    metrics["by_question_type"] = {
        question_type: aggregate(indices)
        for question_type, indices in sorted(type_to_indices.items())
    }
    return metrics, per_sample


def _select_rows_for_dataset(
    rows: Sequence[dict[str, Any]], examples: Sequence[EvalExample]
) -> list[dict[str, Any]]:
    prediction_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["sample_id"] in prediction_map:
            raise ValueError(f"Duplicate prediction sample_id: {row['sample_id']}")
        prediction_map[row["sample_id"]] = row

    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for example in examples:
        row = prediction_map.get(example.sample_id)
        if row is None:
            missing.append(example.sample_id)
            continue
        if row["reference"] != example.reference:
            raise ValueError(
                f"Reference mismatch for {example.sample_id}; predictions may belong to another dataset."
            )
        selected.append(row)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Predictions are incomplete: {len(missing)} of {len(examples)} samples are missing "
            f"(first: {preview})."
        )
    return selected


def save_reference_samples(
    rows: Sequence[dict[str, Any]],
    per_sample_metrics: Sequence[dict[str, float]],
    output_path: Path,
    num_samples: int,
    seed: int,
) -> None:
    count = min(num_samples, len(rows))
    random_indices = sorted(random.Random(seed).sample(range(len(rows)), count))
    samples = []
    for index in random_indices:
        samples.append({**rows[index], "sample_metrics": per_sample_metrics[index]})
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, ensure_ascii=False, indent=2)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate stage-3 Qwen-VL on CT-RATE VQA or report generation."
    )
    parser.add_argument("--mode", choices=("all", "infer", "metrics", "dry-run"), default="all")
    parser.add_argument("--model-path", help="Full model or stage-3 LoRA checkpoint directory.")
    parser.add_argument("--model-base", help="Base checkpoint for a stage-3 LoRA checkpoint.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--image-folder",
        default=".",
        help="Root used to resolve relative image/video paths (default: current directory).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--predictions-file", help="Defaults to OUTPUT_DIR/predictions.jsonl.")
    parser.add_argument(
        "--question-types",
        help="Comma-separated VQA types to retain (for example free_response,description).",
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N flattened examples.")

    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--no-merge-lora", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--keep-reasoning", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-predictions", action="store_true")

    parser.add_argument("--video-min-pixels", type=int, default=100352)
    parser.add_argument("--video-max-pixels", type=int, default=384 * 384)
    parser.add_argument("--video-resized-width", type=int)
    parser.add_argument("--video-resized-height", type=int)
    parser.add_argument("--nframes", type=int, default=128)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--video-cache-size", type=int, default=1)

    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--bertscore-device")
    parser.add_argument("--bertscore-rescale-with-baseline", action="store_true")
    parser.add_argument("--meteor-wordnet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-save-samples", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    if args.nframes == 0:
        args.nframes = None
    if args.nframes is not None and args.fps is not None:
        parser.error("--nframes and --fps are mutually exclusive; pass --nframes 0 to use fps.")
    if (args.video_resized_width is None) != (args.video_resized_height is None):
        parser.error("--video-resized-width and --video-resized-height must be set together.")
    if args.mode in {"all", "infer"} and not args.model_path:
        parser.error("--model-path is required for inference.")
    if args.overwrite_predictions and args.resume:
        parser.error("--overwrite-predictions requires --no-resume.")

    question_types = None
    if args.question_types:
        question_types = {item.strip() for item in args.question_types.split(",") if item.strip()}
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = (
        Path(args.predictions_file).expanduser().resolve()
        if args.predictions_file
        else output_dir / "predictions.jsonl"
    )

    examples, dataset_summary = load_eval_examples(
        args.data_path, question_types=question_types, limit=args.limit
    )
    _write_json(output_dir / "dataset_summary.json", dataset_summary)
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)
    if args.mode == "dry-run":
        return

    if args.mode in {"all", "infer"}:
        run_inference(args, examples, predictions_path)
    if args.mode == "infer":
        return

    rows = _select_rows_for_dataset(_load_predictions(predictions_path), examples)
    metrics, per_sample_metrics = compute_metrics(rows, args)
    metrics["dataset"] = dataset_summary
    _write_json(output_dir / "metrics.json", metrics)
    save_reference_samples(
        rows,
        per_sample_metrics,
        output_dir / "samples.json",
        args.num_save_samples,
        args.sample_seed,
    )
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
