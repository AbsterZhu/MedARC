"""Shared compact-generation helpers used by the volume-DINS evaluators."""

from __future__ import annotations

import types


def _sync_cuda(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _flatten_pooler_output(vision_outputs, torch_module):
    pooled = getattr(vision_outputs, "pooler_output", vision_outputs)
    if isinstance(pooled, torch_module.Tensor):
        return pooled
    if isinstance(pooled, (tuple, list)):
        return torch_module.cat(list(pooled), dim=0)
    raise TypeError(f"Unsupported vision pooler output type: {type(pooled)!r}")


def _install_precomputed_merge_forward(model) -> None:
    """Teach one loaded Qwen3-VL instance to generate from compact embeddings."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModelOutputWithPast,
    )

    if getattr(model, "_merge_baseline_forward_installed", False):
        return
    if getattr(model.config, "model_type", None) != "qwen3_vl":
        raise ValueError("Compact generation currently supports only dense Qwen3-VL.")
    if not hasattr(model, "model") or not hasattr(model.model, "language_model"):
        raise TypeError(f"Unsupported Qwen3-VL generation wrapper: {type(model)!r}")

    backbone = model.model
    original_top_forward = model.forward
    original_backbone_forward = backbone.forward

    def compact_backbone_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        cache_position=None,
        merge_baseline_active=False,
        merge_baseline_visual_pos_masks=None,
        merge_baseline_deepstack_visual_embeds=None,
        **kwargs,
    ):
        if not merge_baseline_active:
            return original_backbone_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                cache_position=cache_position,
                **kwargs,
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        past_length = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            elif isinstance(past_key_values, tuple) and past_key_values:
                past_length = past_key_values[0][0].shape[-2]

        is_prefill = past_length == 0 and inputs_embeds.shape[1] > 1
        language_outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=(
                merge_baseline_visual_pos_masks if is_prefill else None
            ),
            deepstack_visual_embeds=(
                merge_baseline_deepstack_visual_embeds if is_prefill else None
            ),
            **kwargs,
        )
        return Qwen3VLModelOutputWithPast(
            last_hidden_state=language_outputs.last_hidden_state,
            past_key_values=language_outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    def compact_top_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        cache_position=None,
        logits_to_keep=0,
        merge_baseline_active=False,
        merge_baseline_visual_pos_masks=None,
        merge_baseline_deepstack_visual_embeds=None,
        **kwargs,
    ):
        return original_top_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            merge_baseline_active=merge_baseline_active,
            merge_baseline_visual_pos_masks=merge_baseline_visual_pos_masks,
            merge_baseline_deepstack_visual_embeds=(
                merge_baseline_deepstack_visual_embeds
            ),
            **kwargs,
        )

    backbone.forward = types.MethodType(compact_backbone_forward, backbone)
    model.forward = types.MethodType(compact_top_forward, model)
    model._merge_baseline_forward_installed = True
