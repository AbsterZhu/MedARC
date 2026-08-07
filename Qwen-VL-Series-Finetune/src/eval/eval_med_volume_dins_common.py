#!/usr/bin/env python3
"""Shared evaluator for three volume-level DINS variants.

The variants deliberately share the same token selection, contextual merging,
position handling, generation path, and metrics so that an ablation changes
only the source of token importance:

* v1: final-vision-block attention, selected and merged over the whole volume;
* v2: v1 plus question-token relevance in the post-projector LLM space;
* v3: v2 plus an aligned pre-projector saliency prior generated online by 3DINO.

Pruning happens after the vision tower.
It shortens the language-model prefix and KV cache but does not skip vision
tower computation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import eval_med as baseline
import eval_med_merge_baseline as compact_baseline


SUPPORTED_VERSIONS = (1, 2, 3)


@dataclass(frozen=True)
class VolumeAttention:
    """Post-spatial-merge attention importance and keys for one volume."""

    importance: Any
    keys: Any
    raw_groups: int


@dataclass(frozen=True)
class VolumeDINSPlan:
    """Global primary selection and contextual merge assignment."""

    select_mask: Any
    primary_indices: Any
    target_indices: Any
    source_indices: Any
    assignments: Any

    @property
    def primary_tokens(self) -> int:
        return int(self.primary_indices.numel())

    @property
    def contextual_tokens(self) -> int:
        return int(self.target_indices.numel())


class LastVisionVolumeAttentionCapture(AbstractContextManager):
    """Capture final-block Q/K and materialize one volume-level token metric."""

    def __init__(self, visual) -> None:
        self.visual = visual
        self._handle = None
        self._groups: list[tuple[Any, Any]] = []

    def __enter__(self):
        blocks = getattr(self.visual, "blocks", None)
        if not blocks:
            raise TypeError("Qwen3-VL visual tower has no vision blocks.")
        self._handle = blocks[-1].attn.register_forward_pre_hook(
            self._capture, with_kwargs=True
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def _capture(self, module, args, kwargs) -> None:
        import torch
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb_vision,
        )

        if self._groups:
            raise RuntimeError(
                "The final Qwen3-VL vision attention ran more than once."
            )
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        cu_seqlens = kwargs.get("cu_seqlens")
        position_embeddings = kwargs.get("position_embeddings")
        if hidden_states is None or cu_seqlens is None or position_embeddings is None:
            raise RuntimeError("Could not capture Qwen3-VL vision attention inputs.")

        sequence_length = hidden_states.shape[0]
        query_states, key_states, _ = (
            module.qkv(hidden_states)
            .reshape(sequence_length, 3, module.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cosine, sine = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cosine, sine
        )

        boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # [patches, heads, dim] -> [heads, patches, dim]
            query = query_states[start:end].permute(1, 0, 2).detach()
            key = key_states[start:end].permute(1, 0, 2).detach()
            self._groups.append((query, key))

    def materialize(self, spatial_merge_size: int) -> VolumeAttention:
        """Concatenate frame-local attention metrics into one 3D volume metric.

        Qwen3-VL itself attends inside temporal groups.  We retain that trained
        attention definition, but nucleus selection and contextual key matching
        downstream operate once over all groups in the volume.
        """
        import torch

        merge_unit = spatial_merge_size**2
        importance_parts = []
        key_parts = []
        for query, key in self._groups:
            if query.shape[1] % merge_unit:
                raise ValueError(
                    f"Vision group has {query.shape[1]} patches, which is not divisible "
                    f"by spatial merge unit {merge_unit}."
                )
            attention = torch.softmax(
                torch.matmul(query.float(), key.float().transpose(-1, -2))
                / math.sqrt(query.shape[-1]),
                dim=-1,
            )
            importance = attention.mean(dim=0).sum(dim=0)
            importance_parts.append(importance.reshape(-1, merge_unit).mean(dim=-1))
            merged_keys = key.float().reshape(
                key.shape[0], -1, merge_unit, key.shape[-1]
            )
            key_parts.append(merged_keys.mean(dim=2).mean(dim=0))
            del attention

        raw_groups = len(self._groups)
        self._groups.clear()
        if not importance_parts:
            raise RuntimeError("No final-block vision attention groups were captured.")
        return VolumeAttention(
            importance=torch.cat(importance_parts, dim=0),
            keys=torch.cat(key_parts, dim=0),
            raw_groups=raw_groups,
        )


def _probability_from_scores(scores, temperature: float):
    import torch

    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError(
            "Importance scores must be a non-empty one-dimensional tensor."
        )
    return torch.softmax(scores.float() / temperature, dim=-1)


def _question_relevance_scores(frame_features, question_input_ids, embedding_layer):
    """Max cosine relevance between each projected visual token and question tokens."""
    import torch
    import torch.nn.functional as functional

    if question_input_ids is None or question_input_ids.numel() == 0:
        raise ValueError(
            "Text-guided volume DINS requires non-empty question token ids."
        )
    question_features = embedding_layer(question_input_ids).reshape(
        -1, frame_features.shape[-1]
    )
    visual = functional.normalize(frame_features.float(), p=2, dim=-1, eps=1e-6)
    text = functional.normalize(question_features.float(), p=2, dim=-1, eps=1e-6)
    similarities = torch.matmul(visual, text.transpose(0, 1))
    return similarities.max(dim=-1).values


def _vfm_distinctiveness_saliency(features):
    """Return a scalar 3DINO saliency map using distance from volume semantics."""
    import torch
    import torch.nn.functional as functional

    if features.ndim != 4:
        raise ValueError(
            f"Expected VFM features shaped [T,H,W,D], got {tuple(features.shape)}."
        )
    feature_grid = features.float()
    normalized = functional.normalize(feature_grid, p=2, dim=-1, eps=1e-6)
    centroid = functional.normalize(
        normalized.mean(dim=(0, 1, 2), keepdim=True), p=2, dim=-1, eps=1e-6
    )
    # Rare semantic regions have lower cosine similarity to the volume centroid.
    return 1.0 - (normalized * centroid).sum(dim=-1)


def _align_vfm_saliency_to_projector_tokens(
    saliency,
    *,
    temporal: int,
    patch_height: int,
    patch_width: int,
    spatial_merge_size: int,
    device,
):
    """Trilinearly align a VFM 3D map to Qwen pre-projector patch order."""
    import torch.nn.functional as functional

    if saliency.ndim != 3:
        raise ValueError(
            f"Expected a 3D VFM saliency map, got {tuple(saliency.shape)}."
        )
    resized = functional.interpolate(
        saliency.float()[None, None],
        size=(temporal, patch_height, patch_width),
        mode="trilinear",
        align_corners=False,
    )[0, 0].to(device=device)

    merge = spatial_merge_size
    if patch_height % merge or patch_width % merge:
        raise ValueError("Qwen patch grid is not divisible by spatial_merge_size.")
    # Qwen's patch sequence is spatial-merge-major: [t,mh,mw,s,s].
    pre_projector = (
        resized.reshape(
            temporal,
            patch_height // merge,
            merge,
            patch_width // merge,
            merge,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(-1)
    )
    return pre_projector.reshape(-1, merge**2).mean(dim=-1)


def _mixed_importance_probabilities(
    attention_scores,
    *,
    attention_temperature: float,
    text_scores=None,
    text_weight: float = 0.0,
    text_temperature: float = 0.07,
    vfm_scores=None,
    vfm_weight: float = 0.0,
    vfm_temperature: float = 0.2,
):
    """Mix normalized importance distributions so heterogeneous scales are safe."""
    attention_weight = 1.0 - text_weight - vfm_weight
    if attention_weight < -1e-8:
        raise ValueError("Text and VFM importance weights must sum to at most one.")
    probabilities = max(attention_weight, 0.0) * _probability_from_scores(
        attention_scores, attention_temperature
    )
    if text_weight:
        if text_scores is None:
            raise ValueError("A positive text weight requires text relevance scores.")
        probabilities = probabilities + text_weight * _probability_from_scores(
            text_scores, text_temperature
        )
    if vfm_weight:
        if vfm_scores is None:
            raise ValueError("A positive VFM weight requires aligned VFM scores.")
        probabilities = probabilities + vfm_weight * _probability_from_scores(
            vfm_scores, vfm_temperature
        )
    return probabilities / probabilities.sum().clamp_min(1e-12)


def _build_volume_dins_plan(
    probabilities,
    keys,
    *,
    tau: float,
    contextual_ratio: float,
) -> VolumeDINSPlan:
    """Select and cluster tokens once across the complete 3D volume."""
    import torch
    import torch.nn.functional as functional

    token_count = int(probabilities.numel())
    if keys.shape[0] != token_count:
        raise ValueError("Volume attention keys and importance have different lengths.")

    sorted_probabilities, sorted_indices = torch.sort(
        probabilities.float(), descending=True
    )
    cumulative = torch.cumsum(sorted_probabilities, dim=0)
    reached = torch.nonzero(cumulative >= tau, as_tuple=False)
    primary_count = int(reached[0].item()) + 1 if reached.numel() else token_count
    primary_indices = sorted_indices[:primary_count]

    primary_mask = torch.zeros(token_count, dtype=torch.bool, device=keys.device)
    primary_mask[primary_indices] = True
    redundant_indices = torch.nonzero(~primary_mask, as_tuple=False).flatten()
    if redundant_indices.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=keys.device)
        return VolumeDINSPlan(
            select_mask=primary_mask,
            primary_indices=primary_indices,
            target_indices=empty,
            source_indices=empty,
            assignments=empty,
        )

    requested_contextual = max(int(contextual_ratio * token_count), 1)
    contextual_count = min(requested_contextual, int(redundant_indices.numel()))
    step = max(1, int(redundant_indices.numel()) // contextual_count)
    target_local_indices = torch.arange(
        0, redundant_indices.numel(), step, device=keys.device
    )[:contextual_count]
    target_indices = redundant_indices[target_local_indices]

    non_target_mask = torch.ones(
        redundant_indices.numel(), dtype=torch.bool, device=keys.device
    )
    non_target_mask[target_local_indices] = False
    source_indices = redundant_indices[non_target_mask]
    if source_indices.numel():
        normalized_keys = functional.normalize(keys.float(), p=2, dim=-1, eps=1e-6)
        similarities = torch.matmul(
            normalized_keys[source_indices],
            normalized_keys[target_indices].transpose(0, 1),
        )
        assignments = similarities.argmax(dim=-1)
    else:
        assignments = torch.empty(0, dtype=torch.long, device=keys.device)

    select_mask = primary_mask.clone()
    select_mask[target_indices] = True
    return VolumeDINSPlan(
        select_mask=select_mask,
        primary_indices=primary_indices,
        target_indices=target_indices,
        source_indices=source_indices,
        assignments=assignments,
    )


def _apply_volume_contextual_merge(features, plan: VolumeDINSPlan):
    """Apply one volume assignment consistently to projector and DeepStack features."""
    import torch

    updated = features.clone()
    if plan.source_indices.numel():
        contextual_count = plan.contextual_tokens
        accumulated = torch.zeros(
            (contextual_count, features.shape[-1]),
            dtype=torch.float32,
            device=features.device,
        )
        accumulated.index_add_(
            0, plan.assignments, features[plan.source_indices].float()
        )
        counts = torch.bincount(plan.assignments, minlength=contextual_count).float()
        nonempty = counts > 0
        means = torch.zeros_like(accumulated)
        means[nonempty] = accumulated[nonempty] / counts[nonempty, None]
        updated[plan.target_indices] = (
            features[plan.target_indices].float() + means
        ).to(features.dtype)
    return updated[plan.select_mask]


class OnlineVFMExtractor:
    """Keep one frozen 3DINO teacher loaded and extract each sample on demand."""

    def __init__(self, args) -> None:
        three_dino_root = Path(args.three_dino_root).expanduser().resolve()
        root_string = str(three_dino_root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        from precompute_3dino_med_align import ThreeDINOMedAlignExtractor

        self.extractor = ThreeDINOMedAlignExtractor(
            three_dino_root=three_dino_root,
            config_file=args.three_dino_config_file,
            checkpoint=args.three_dino_checkpoint,
            device=args.three_dino_device,
            output_dtype=args.three_dino_output_dtype,
            nframes=128,
            teacher_input_size=192,
        )

    def load_saliency(self, video_path: str | Path):
        extraction_start = time.perf_counter()
        features, metadata = self.extractor.extract(video_path)
        saliency = _vfm_distinctiveness_saliency(features)
        metadata["generation_seconds"] = time.perf_counter() - extraction_start
        del features
        return saliency, metadata


def _prepare_volume_dins_generation_inputs(
    model,
    inputs: dict[str, Any],
    args,
    *,
    version: int,
    question_input_ids=None,
    vfm_payload=None,
):
    """Run vision once, perform global DINS, and build a compact LLM prefix."""
    import torch

    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("Volume DINS currently requires batch size 1.")
    if (
        inputs.get("pixel_values") is not None
        or inputs.get("image_grid_thw") is not None
    ):
        raise ValueError("Volume DINS currently supports CT video prompts only.")
    video_grid_thw = inputs.get("video_grid_thw")
    pixel_values_videos = inputs.get("pixel_values_videos")
    if video_grid_thw is None or pixel_values_videos is None:
        raise ValueError("A Qwen3-VL video and video_grid_thw are required.")
    if video_grid_thw.shape[0] != 1:
        raise ValueError(
            "Volume DINS currently requires exactly one volume per prompt."
        )

    backbone = model.model
    baseline._prepare_qwen3_vl_generation_position_ids(model, inputs, "qwen3_vl")
    original_position_ids = inputs["position_ids"]
    original_rope_deltas = getattr(backbone, "rope_deltas", None)
    spatial_merge_size = int(backbone.visual.spatial_merge_size)

    compact_baseline._sync_cuda(torch)
    vision_start = time.perf_counter()
    with LastVisionVolumeAttentionCapture(backbone.visual) as capture:
        vision_outputs = backbone.get_video_features(
            pixel_values_videos, video_grid_thw, return_dict=True
        )
    compact_baseline._sync_cuda(torch)
    vision_seconds = time.perf_counter() - vision_start

    input_ids = inputs["input_ids"]
    inputs_embeds = backbone.get_input_embeddings()(input_ids)
    final_features = compact_baseline._flatten_pooler_output(vision_outputs, torch).to(
        device=inputs_embeds.device, dtype=inputs_embeds.dtype
    )
    deepstack_features = getattr(vision_outputs, "deepstack_features", None)
    if not deepstack_features:
        raise ValueError("Qwen3-VL returned no DeepStack visual features.")
    deepstack_features = [
        feature.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        for feature in deepstack_features
    ]

    temporal, patch_height, patch_width = [
        int(item) for item in video_grid_thw[0].tolist()
    ]
    if patch_height % spatial_merge_size or patch_width % spatial_merge_size:
        raise ValueError("Video patch grid is not divisible by spatial_merge_size.")
    merged_height = patch_height // spatial_merge_size
    merged_width = patch_width // spatial_merge_size
    tokens_per_temporal_grid = merged_height * merged_width
    expected_visual_tokens = temporal * tokens_per_temporal_grid
    if final_features.shape[0] != expected_visual_tokens:
        raise ValueError(
            f"Vision projector returned {final_features.shape[0]} tokens; expected "
            f"{expected_visual_tokens}."
        )
    for layer_index, feature in enumerate(deepstack_features):
        if feature.shape[0] != expected_visual_tokens:
            raise ValueError(
                f"DeepStack tensor {layer_index} has {feature.shape[0]} tokens; expected "
                f"{expected_visual_tokens}."
            )

    prune_start = time.perf_counter()
    volume_attention = capture.materialize(spatial_merge_size)
    if volume_attention.importance.shape[0] != expected_visual_tokens:
        raise ValueError(
            "Captured attention does not match the full projector token volume."
        )
    if volume_attention.raw_groups != temporal:
        raise ValueError(
            f"Captured {volume_attention.raw_groups} temporal attention groups; "
            f"expected {temporal}."
        )

    text_scores = None
    if version >= 2:
        if final_features.shape[-1] != backbone.get_input_embeddings().weight.shape[-1]:
            raise ValueError(
                "Projected vision and text embedding dimensions do not match."
            )
        text_scores = _question_relevance_scores(
            final_features,
            question_input_ids,
            backbone.get_input_embeddings(),
        )

    vfm_scores = None
    vfm_metadata = None
    if version >= 3:
        if vfm_payload is None:
            raise ValueError("Volume DINS v3 requires an online VFM payload.")
        vfm_saliency, vfm_metadata = vfm_payload
        vfm_scores = _align_vfm_saliency_to_projector_tokens(
            vfm_saliency,
            temporal=temporal,
            patch_height=patch_height,
            patch_width=patch_width,
            spatial_merge_size=spatial_merge_size,
            device=final_features.device,
        )
        if vfm_scores.shape[0] != expected_visual_tokens:
            raise ValueError(
                "Aligned VFM importance does not match projector token count."
            )

    text_weight = args.volume_dins_text_weight if version >= 2 else 0.0
    vfm_weight = args.volume_dins_vfm_weight if version >= 3 else 0.0
    probabilities = _mixed_importance_probabilities(
        volume_attention.importance.to(final_features.device),
        attention_temperature=args.volume_dins_temperature,
        text_scores=text_scores,
        text_weight=text_weight,
        text_temperature=args.volume_dins_text_temperature,
        vfm_scores=vfm_scores,
        vfm_weight=vfm_weight,
        vfm_temperature=args.volume_dins_vfm_temperature,
    )
    plan = _build_volume_dins_plan(
        probabilities,
        volume_attention.keys.to(final_features.device),
        tau=args.volume_dins_tau,
        contextual_ratio=args.volume_dins_contextual_ratio,
    )
    if not torch.any(plan.select_mask):
        raise RuntimeError("Volume DINS removed every visual token.")

    selected_final_features = _apply_volume_contextual_merge(final_features, plan)
    selected_deepstack_features = [
        _apply_volume_contextual_merge(feature, plan) for feature in deepstack_features
    ]

    video_token_id = int(model.config.video_token_id)
    video_positions = torch.nonzero(
        input_ids[0] == video_token_id, as_tuple=False
    ).flatten()
    if video_positions.numel() != expected_visual_tokens:
        raise ValueError(
            f"Prompt has {video_positions.numel()} video placeholders, but vision returned "
            f"{expected_visual_tokens} tokens."
        )
    selected_feature_indices = torch.nonzero(plan.select_mask, as_tuple=False).flatten()
    selected_sequence_positions = video_positions[selected_feature_indices]

    keep_mask = torch.ones(
        input_ids.shape[1], dtype=torch.bool, device=input_ids.device
    )
    keep_mask[video_positions] = False
    keep_mask[selected_sequence_positions] = True
    compact_inputs_embeds = inputs_embeds[:, keep_mask, :].clone()
    original_to_compact = torch.full(
        (input_ids.shape[1],), -1, dtype=torch.long, device=input_ids.device
    )
    original_to_compact[keep_mask] = torch.arange(
        int(keep_mask.sum().item()), device=input_ids.device
    )
    compact_visual_positions = original_to_compact[selected_sequence_positions]
    compact_inputs_embeds[0, compact_visual_positions] = selected_final_features

    compact_visual_pos_masks = torch.zeros(
        (1, compact_inputs_embeds.shape[1]),
        dtype=torch.bool,
        device=inputs_embeds.device,
    )
    compact_visual_pos_masks[0, compact_visual_positions] = True
    compact_attention_mask = inputs["attention_mask"][:, keep_mask]
    compact_position_ids = original_position_ids[..., keep_mask]

    removed_tokens = int(input_ids.shape[1] - compact_inputs_embeds.shape[1])
    if original_rope_deltas is not None:
        backbone.rope_deltas = original_rope_deltas + removed_tokens

    compact_baseline._sync_cuda(torch)
    pruning_seconds = time.perf_counter() - prune_start
    selected_visual_tokens = int(plan.select_mask.sum().item())
    selected_per_temporal_grid = (
        plan.select_mask.reshape(temporal, tokens_per_temporal_grid).sum(dim=1).tolist()
    )
    statistics = {
        "version": version,
        "importance_sources": (
            ["vision_attention"]
            + (["question_text"] if version >= 2 else [])
            + (["vfm_3dino"] if version >= 3 else [])
        ),
        "tau": args.volume_dins_tau,
        "attention_temperature": args.volume_dins_temperature,
        "contextual_ratio": args.volume_dins_contextual_ratio,
        "attention_weight": 1.0 - text_weight - vfm_weight,
        "text_weight": text_weight,
        "text_temperature": args.volume_dins_text_temperature if version >= 2 else None,
        "vfm_weight": vfm_weight,
        "vfm_temperature": args.volume_dins_vfm_temperature if version >= 3 else None,
        "scope": "whole_3d_volume",
        "original_visual_tokens": expected_visual_tokens,
        "selected_visual_tokens": selected_visual_tokens,
        "removed_visual_tokens": expected_visual_tokens - selected_visual_tokens,
        "overall_visual_keep_rate": selected_visual_tokens / expected_visual_tokens,
        "primary_visual_tokens": plan.primary_tokens,
        "contextual_visual_tokens": plan.contextual_tokens,
        "original_prompt_tokens": int(input_ids.shape[1]),
        "compressed_prompt_tokens": int(compact_inputs_embeds.shape[1]),
        "temporal_grid": temporal,
        "post_projector_height": merged_height,
        "post_projector_width": merged_width,
        "tokens_per_temporal_grid_before": tokens_per_temporal_grid,
        "selected_tokens_per_temporal_grid": selected_per_temporal_grid,
        "vision_projector_seconds": vision_seconds,
        "volume_dins_and_compact_seconds": pruning_seconds,
        "vfm": vfm_metadata,
    }
    generation_inputs = {
        "inputs_embeds": compact_inputs_embeds,
        "attention_mask": compact_attention_mask,
        "position_ids": compact_position_ids,
        "merge_baseline_active": True,
        "merge_baseline_visual_pos_masks": compact_visual_pos_masks,
        "merge_baseline_deepstack_visual_embeds": selected_deepstack_features,
    }
    return generation_inputs, statistics


def run_inference(
    args,
    examples: Sequence[baseline.EvalExample],
    predictions_path: Path,
    *,
    version: int,
):
    existing = baseline._load_prediction_map(predictions_path) if args.resume else {}
    if predictions_path.exists() and not args.resume and not args.overwrite_predictions:
        raise FileExistsError(
            f"{predictions_path} already exists. Use --resume or --overwrite-predictions."
        )
    file_mode = "w" if args.overwrite_predictions and not args.resume else "a"
    pending = [example for example in examples if example.sample_id not in existing]
    print(
        f"Volume DINS v{version}: total={len(examples)}, existing={len(existing)}, "
        f"pending={len(pending)}, tau={args.volume_dins_tau}",
        flush=True,
    )
    if not pending:
        return

    import torch
    from tqdm import tqdm

    from dataset.data_utils import get_qwen_multimodal_settings
    from dataset.sft_dataset_med import get_nifti_video_info

    if args.num_beams != 1:
        raise ValueError("Volume DINS evaluation currently requires --num-beams 1.")
    if args.no_merge_lora:
        raise ValueError("Volume DINS evaluation requires merged LoRA weights.")

    vfm_extractor = OnlineVFMExtractor(args) if version >= 3 else None
    processor, model, model_type, loaded_model_path = baseline.load_eval_model(args)
    if model_type != "qwen3_vl":
        raise ValueError(f"Expected dense qwen3_vl, got {model_type!r}.")
    compact_baseline._install_precomputed_merge_forward(model)
    _, image_patch_size, return_video_metadata = get_qwen_multimodal_settings(
        str(loaded_model_path)
    )
    input_device = baseline._model_input_device(model)
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
        "num_beams": 1,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    tokenizer = getattr(processor, "tokenizer", None)
    if version >= 2 and tokenizer is None:
        raise TypeError(
            "The Qwen3-VL processor exposes no tokenizer for question guidance."
        )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open(file_mode, encoding="utf-8") as output_handle:
        for example in tqdm(
            pending, desc=f"Generating volume DINS v{version}", unit="example"
        ):
            video_path = baseline._resolve_video_path(example.video, image_folder)
            if not video_path.exists():
                raise FileNotFoundError(f"CT volume does not exist: {video_path}")
            video_payload, video_kwargs = load_video(str(video_path))
            processor_kwargs: dict[str, Any] = {
                "text": [baseline._build_prompt(example.question, model_type)],
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
            inputs = baseline._move_inputs_to_device(dict(inputs), input_device)
            question_input_ids = None
            if version >= 2:
                question_tokens = tokenizer(
                    example.question,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                question_input_ids = question_tokens["input_ids"].to(input_device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            vfm_payload = (
                vfm_extractor.load_saliency(video_path)
                if vfm_extractor is not None
                else None
            )
            with torch.inference_mode():
                generation_inputs, statistics = _prepare_volume_dins_generation_inputs(
                    model,
                    inputs,
                    args,
                    version=version,
                    question_input_ids=question_input_ids,
                    vfm_payload=vfm_payload,
                )
                del inputs
                compact_baseline._sync_cuda(torch)
                generation_start = time.perf_counter()
                generated_ids = model.generate(**generation_inputs, **generation_kwargs)
                compact_baseline._sync_cuda(torch)
                generation_seconds = time.perf_counter() - generation_start

            prediction = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            statistics["generation_seconds"] = generation_seconds
            statistics["generated_tokens"] = int(generated_ids.shape[1])
            if torch.cuda.is_available():
                statistics["peak_cuda_memory_bytes"] = int(
                    torch.cuda.max_memory_allocated()
                )
            row = {
                **asdict(example),
                "video_path": str(video_path),
                "prediction": baseline._clean_prediction(
                    prediction, args.keep_reasoning
                ),
                "volume_dins": statistics,
            }
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_handle.flush()

    del model, processor, vfm_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mean(values: Sequence[float]):
    return sum(values) / len(values) if values else None


def compute_volume_dins_metrics(rows, args, *, version: int):
    metrics, per_sample_metrics = baseline.compute_metrics(rows, args)
    statistics = [row.get("volume_dins") for row in rows]
    if any(not isinstance(item, dict) for item in statistics):
        raise ValueError("Every prediction must contain volume_dins statistics.")

    original_visual = [int(item["original_visual_tokens"]) for item in statistics]
    selected_visual = [int(item["selected_visual_tokens"]) for item in statistics]
    primary_visual = [int(item["primary_visual_tokens"]) for item in statistics]
    contextual_visual = [int(item["contextual_visual_tokens"]) for item in statistics]
    original_prompt = [int(item["original_prompt_tokens"]) for item in statistics]
    compressed_prompt = [int(item["compressed_prompt_tokens"]) for item in statistics]
    vision_seconds = [float(item["vision_projector_seconds"]) for item in statistics]
    prune_seconds = [
        float(item["volume_dins_and_compact_seconds"]) for item in statistics
    ]
    vfm_generation_seconds = [
        float(item["vfm"].get("generation_seconds", 0.0))
        if isinstance(item.get("vfm"), dict)
        else 0.0
        for item in statistics
    ]
    generation_seconds = [float(item["generation_seconds"]) for item in statistics]
    generated_tokens = [int(item["generated_tokens"]) for item in statistics]
    peak_memory = [
        int(item["peak_cuda_memory_bytes"])
        for item in statistics
        if item.get("peak_cuda_memory_bytes") is not None
    ]

    total_original_visual = sum(original_visual)
    total_selected_visual = sum(selected_visual)
    total_original_prompt = sum(original_prompt)
    total_compressed_prompt = sum(compressed_prompt)
    total_generation_seconds = sum(generation_seconds)
    metrics["volume_dins_efficiency"] = {
        "version": version,
        "num_samples": len(statistics),
        "tau": args.volume_dins_tau,
        "contextual_ratio": args.volume_dins_contextual_ratio,
        "attention_temperature": args.volume_dins_temperature,
        "text_weight": args.volume_dins_text_weight if version >= 2 else 0.0,
        "text_temperature": (
            args.volume_dins_text_temperature if version >= 2 else None
        ),
        "vfm_weight": args.volume_dins_vfm_weight if version >= 3 else 0.0,
        "vfm_temperature": (args.volume_dins_vfm_temperature if version >= 3 else None),
        "visual_keep_rate_achieved": total_selected_visual / total_original_visual,
        "visual_token_reduction_rate": 1
        - total_selected_visual / total_original_visual,
        "prompt_token_reduction_rate": 1
        - total_compressed_prompt / total_original_prompt,
        "mean_original_visual_tokens": _mean(original_visual),
        "mean_selected_visual_tokens": _mean(selected_visual),
        "mean_primary_visual_tokens": _mean(primary_visual),
        "mean_contextual_visual_tokens": _mean(contextual_visual),
        "mean_original_prompt_tokens": _mean(original_prompt),
        "mean_compressed_prompt_tokens": _mean(compressed_prompt),
        "mean_vision_projector_seconds": _mean(vision_seconds),
        "mean_online_3dino_seconds": (
            _mean(vfm_generation_seconds) if version >= 3 else None
        ),
        "mean_volume_dins_and_compact_seconds": _mean(prune_seconds),
        "mean_generation_seconds": _mean(generation_seconds),
        "mean_end_to_end_model_seconds": _mean(
            [
                a + b + c + d
                for a, b, c, d in zip(
                    vfm_generation_seconds,
                    vision_seconds,
                    prune_seconds,
                    generation_seconds,
                )
            ]
        ),
        "generated_tokens_per_second": (
            sum(generated_tokens) / total_generation_seconds
            if total_generation_seconds > 0
            else None
        ),
        "mean_peak_cuda_memory_bytes": _mean(peak_memory),
        "max_peak_cuda_memory_bytes": max(peak_memory) if peak_memory else None,
    }
    return metrics, per_sample_metrics


def build_parser(version: int) -> argparse.ArgumentParser:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported volume DINS version: {version}")
    parser = baseline.build_parser()
    parser.description = f"Evaluate stage-3 Qwen3-VL with volume DINS v{version}."
    parser.add_argument("--volume-dins-tau", type=float, default=0.9)
    parser.add_argument("--volume-dins-temperature", type=float, default=2.0)
    parser.add_argument("--volume-dins-contextual-ratio", type=float, default=0.05)
    parser.add_argument("--volume-dins-text-weight", type=float, default=0.3)
    parser.add_argument("--volume-dins-text-temperature", type=float, default=0.07)
    parser.add_argument("--volume-dins-vfm-weight", type=float, default=0.3)
    parser.add_argument("--volume-dins-vfm-temperature", type=float, default=0.2)
    medarc_root = Path(__file__).resolve().parents[3]
    configured_root = os.environ.get("THREE_DINO_ROOT")
    if configured_root:
        three_dino_root = Path(configured_root).expanduser()
    elif (medarc_root / "third_party/3DINO/dinov2").is_dir():
        three_dino_root = medarc_root / "third_party/3DINO"
    elif (medarc_root / "3DINO/dinov2").is_dir():
        # Backward compatibility for older local checkouts.
        three_dino_root = medarc_root / "3DINO"
    else:
        three_dino_root = medarc_root / "third_party/3DINO"
    three_dino_config = os.environ.get(
        "THREE_DINO_CONFIG_FILE",
        str(three_dino_root / "dinov2/configs/train/vit3d_highres"),
    )
    three_dino_checkpoint = os.environ.get(
        "THREE_DINO_CHECKPOINT",
        str(three_dino_root / "3dino_vit_weights.pth"),
    )
    parser.add_argument(
        "--three-dino-root",
        default=str(three_dino_root),
    )
    parser.add_argument(
        "--three-dino-config-file",
        default=three_dino_config,
    )
    parser.add_argument(
        "--three-dino-checkpoint",
        default=three_dino_checkpoint,
    )
    parser.add_argument("--three-dino-device", default="cuda")
    parser.add_argument(
        "--three-dino-output-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    return parser


def main(version: int) -> None:
    parser = build_parser(version)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    if not 0 < args.volume_dins_tau <= 1:
        parser.error("--volume-dins-tau must be in (0, 1].")
    if args.volume_dins_temperature <= 0:
        parser.error("--volume-dins-temperature must be positive.")
    if not 0 < args.volume_dins_contextual_ratio <= 1:
        parser.error("--volume-dins-contextual-ratio must be in (0, 1].")
    if not 0 <= args.volume_dins_text_weight <= 1:
        parser.error("--volume-dins-text-weight must be in [0, 1].")
    if args.volume_dins_text_temperature <= 0:
        parser.error("--volume-dins-text-temperature must be positive.")
    if not 0 <= args.volume_dins_vfm_weight <= 1:
        parser.error("--volume-dins-vfm-weight must be in [0, 1].")
    if args.volume_dins_vfm_temperature <= 0:
        parser.error("--volume-dins-vfm-temperature must be positive.")
    active_text_weight = args.volume_dins_text_weight if version >= 2 else 0.0
    active_vfm_weight = args.volume_dins_vfm_weight if version >= 3 else 0.0
    if active_text_weight + active_vfm_weight > 1:
        parser.error("Active text and VFM weights must sum to at most 1.")
    if version >= 3:
        three_dino_root = Path(args.three_dino_root).expanduser()
        checkpoint = Path(args.three_dino_checkpoint).expanduser()
        config_file = Path(args.three_dino_config_file).expanduser()
        config_yaml = (
            config_file
            if config_file.suffix == ".yaml"
            else config_file.with_suffix(".yaml")
        )
        if not three_dino_root.is_dir():
            parser.error(f"3DINO source root does not exist: {three_dino_root}")
        if not checkpoint.is_file():
            parser.error(f"3DINO checkpoint does not exist: {checkpoint}")
        if not config_yaml.is_file():
            parser.error(f"3DINO config does not exist: {config_yaml}")
    if args.nframes == 0:
        args.nframes = None
    if args.nframes is not None and args.fps is not None:
        parser.error("--nframes and --fps are mutually exclusive.")
    if (args.video_resized_width is None) != (args.video_resized_height is None):
        parser.error("Video width and height must be set together.")
    if args.mode in {"all", "infer"} and not args.model_path:
        parser.error("--model-path is required for inference.")
    if args.overwrite_predictions and args.resume:
        parser.error("--overwrite-predictions requires --no-resume.")

    question_types = None
    if args.question_types:
        question_types = {
            item.strip() for item in args.question_types.split(",") if item.strip()
        }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = (
        Path(args.predictions_file).expanduser().resolve()
        if args.predictions_file
        else output_dir / f"predictions_volume_dins_v{version}.jsonl"
    )

    examples, dataset_summary = baseline.load_eval_examples(
        args.data_path, question_types=question_types, limit=args.limit
    )
    dataset_summary["volume_dins"] = {
        "version": version,
        "scope": "whole_3d_volume",
        "tau": args.volume_dins_tau,
        "contextual_ratio": args.volume_dins_contextual_ratio,
        "attention_temperature": args.volume_dins_temperature,
        "text_weight": active_text_weight,
        "text_temperature": (
            args.volume_dins_text_temperature if version >= 2 else None
        ),
        "vfm_weight": active_vfm_weight,
        "vfm_temperature": (args.volume_dins_vfm_temperature if version >= 3 else None),
        "vfm_generation": "online_per_inference" if version >= 3 else None,
        "three_dino_root": (
            str(Path(args.three_dino_root).expanduser()) if version >= 3 else None
        ),
        "three_dino_config_file": (
            str(Path(args.three_dino_config_file).expanduser())
            if version >= 3
            else None
        ),
        "three_dino_checkpoint": (
            str(Path(args.three_dino_checkpoint).expanduser())
            if version >= 3
            else None
        ),
        "three_dino_device": args.three_dino_device if version >= 3 else None,
        "three_dino_output_dtype": (
            args.three_dino_output_dtype if version >= 3 else None
        ),
        "attention_layer": "last_vision_block",
        "question_relevance": (
            "max_cosine_in_post_projector_llm_space" if version >= 2 else None
        ),
        "vfm_importance": (
            "3dino_cosine_distance_from_volume_centroid" if version >= 3 else None
        ),
        "vfm_alignment": (
            "trilinear_to_qwen_pre_projector_grid" if version >= 3 else None
        ),
        "contextual_similarity": "global_final_key_cosine",
        "deepstack_policy": "same_global_contextual_merge",
        "position_policy": "original_selected_token_position",
    }
    baseline._write_json(
        output_dir / f"dataset_summary_volume_dins_v{version}.json", dataset_summary
    )
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)
    if args.mode == "dry-run":
        return

    if args.mode in {"all", "infer"}:
        run_inference(args, examples, predictions_path, version=version)
    if args.mode == "infer":
        return

    rows = baseline._select_rows_for_dataset(
        baseline._load_predictions(predictions_path), examples
    )
    metrics, per_sample_metrics = compute_volume_dins_metrics(
        rows, args, version=version
    )
    metrics["dataset"] = dataset_summary
    baseline._write_json(output_dir / f"metrics_volume_dins_v{version}.json", metrics)
    baseline.save_reference_samples(
        rows,
        per_sample_metrics,
        output_dir / f"samples_volume_dins_v{version}.json",
        args.num_save_samples,
        args.sample_seed,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
