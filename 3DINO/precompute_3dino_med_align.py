#!/usr/bin/env python3
"""Precompute frozen 3DINO patch targets for Qwen medical alignment."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


CACHE_FORMAT_VERSION = "2"
NORMALIZATION_PERCENTILES = (0.05, 99.95)
DEFAULT_NFRAMES = 128
DEFAULT_INPUT_SIZE = 192
DEFAULT_GRID_SIZE = 12
DEFAULT_TEACHER_DIM = 1024


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    medarc_root = script_dir.parent
    configured_root = os.environ.get("THREE_DINO_ROOT")
    if configured_root:
        default_three_dino_root = Path(configured_root).expanduser()
    elif (medarc_root / "third_party/3DINO/dinov2").is_dir():
        default_three_dino_root = medarc_root / "third_party/3DINO"
    elif (script_dir / "dinov2").is_dir():
        # Backward compatibility for older local checkouts that vendored 3DINO.
        default_three_dino_root = script_dir
    else:
        default_three_dino_root = medarc_root / "third_party/3DINO"

    default_config = os.environ.get(
        "THREE_DINO_CONFIG_FILE",
        str(default_three_dino_root / "dinov2/configs/train/vit3d_highres"),
    )
    default_checkpoint = os.environ.get(
        "THREE_DINO_CHECKPOINT",
        str(default_three_dino_root / "3dino_vit_weights.pth"),
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        action="append",
        required=True,
        help="Training JSON path. Repeat this option to precompute the union of multiple datasets.",
    )
    parser.add_argument(
        "--image-folder",
        default=None,
        help="Root used to resolve relative video paths (must match Qwen --image_folder).",
    )
    parser.add_argument("--output-dir", required=True, help="Cache root to create.")
    parser.add_argument("--three-dino-root", default=str(default_three_dino_root))
    parser.add_argument(
        "--config-file",
        default=default_config,
        help="3DINO config path, with or without the .yaml suffix.",
    )
    parser.add_argument(
        "--checkpoint",
        default=default_checkpoint,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--nframes", type=int, default=DEFAULT_NFRAMES)
    parser.add_argument("--teacher-input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for future batching; currently must be 1.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit after de-duplication.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(source_relative_path: str) -> str:
    return hashlib.sha256(source_relative_path.encode("utf-8")).hexdigest()


def normalize_source_relative_path(source_path: str, image_folder: str | None) -> str:
    source = Path(source_path).expanduser()
    if source.is_absolute():
        if image_folder:
            folder = Path(image_folder).expanduser().resolve(strict=False)
            try:
                source = source.resolve(strict=False).relative_to(folder)
            except ValueError:
                source = source.resolve(strict=False)
        else:
            source = source.resolve(strict=False)
    normalized = PurePosixPath(source.as_posix()).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def resolve_source_path(source_path: str, image_folder: str | None) -> Path:
    path = Path(source_path).expanduser()
    if not path.is_absolute() and image_folder:
        path = Path(image_folder).expanduser() / path
    return path.resolve(strict=False)


def collect_unique_videos(data_paths: list[str], image_folder: str | None) -> list[tuple[str, Path]]:
    unique: dict[str, Path] = {}
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"Dataset JSON must contain a list: {data_path}")
        for record_index, record in tqdm(enumerate(records)):
            if "video" not in record:
                raise ValueError(f"Record {record_index} in {data_path} has no 'video' entry.")
            videos = record["video"]
            if isinstance(videos, str):
                videos = [videos]
            if not isinstance(videos, list):
                raise ValueError(f"Invalid video entry at record {record_index} in {data_path}.")
            for video in videos:
                relative = normalize_source_relative_path(video, image_folder)
                resolved = resolve_source_path(video, image_folder)
                previous = unique.get(relative)
                if previous is not None and previous != resolved:
                    raise ValueError(
                        f"Stable source id collision for {relative!r}: {previous} vs {resolved}."
                    )
                unique[relative] = resolved
    return sorted(unique.items())


def sample_third_axis(volume: np.ndarray, nframes: int) -> np.ndarray:
    if nframes <= 0:
        raise ValueError("nframes must be positive for deterministic cache generation.")
    indices = np.linspace(0, volume.shape[2] - 1, nframes)
    indices = np.round(indices).astype(np.int64)
    return volume[:, :, indices]


def normalize_volume(volume: np.ndarray) -> torch.Tensor:
    volume = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(volume, NORMALIZATION_PERCENTILES)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        normalized = np.zeros_like(volume, dtype=np.float32)
    else:
        normalized = np.clip((volume - low) / (high - low), 0.0, 1.0)
        normalized = normalized * 2.0 - 1.0
    # Keep the semantic order [H,W,D]. Conv3d treats these as its three spatial axes.
    # np.percentile returns float64 scalars, which can promote the normalized
    # array back to float64. 3DINO's reference preprocessing uses float32.
    normalized = np.ascontiguousarray(normalized, dtype=np.float32)
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)


def prepare_teacher_input(volume: np.ndarray, nframes: int, input_size: int) -> torch.Tensor:
    sampled = sample_third_axis(volume, nframes)
    tensor = normalize_volume(sampled)
    return F.interpolate(
        tensor,
        size=(input_size, input_size, input_size),
        mode="trilinear",
        align_corners=False,
    )


def atomic_save_safetensors(tensors: dict[str, torch.Tensor], metadata: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(file_descriptor)
    try:
        save_file(tensors, temporary_name, metadata=metadata)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_model(args: argparse.Namespace):
    root = Path(args.three_dino_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    from dinov2.configs import load_and_merge_config_3d
    from dinov2.models import build_model_from_cfg
    from dinov2.utils.utils import load_pretrained_weights

    config_name = str(Path(args.config_file).expanduser())
    if config_name.endswith(".yaml"):
        config_name = config_name[:-5]
    config = load_and_merge_config_3d(config_name)
    model, _ = build_model_from_cfg(config, only_teacher=True)
    load_pretrained_weights(model, str(Path(args.checkpoint).expanduser()), "teacher")
    model.eval().to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config


class ThreeDINOMedAlignExtractor:
    """Reusable frozen 3DINO extractor for cache-free evaluation."""

    def __init__(
        self,
        *,
        three_dino_root: str | Path,
        config_file: str | Path,
        checkpoint: str | Path,
        device: str = "cuda",
        output_dtype: str = "float16",
        nframes: int = DEFAULT_NFRAMES,
        teacher_input_size: int = DEFAULT_INPUT_SIZE,
    ) -> None:
        if nframes != DEFAULT_NFRAMES:
            raise ValueError(
                f"The alignment design requires nframes={DEFAULT_NFRAMES}; got {nframes}."
            )
        if teacher_input_size != DEFAULT_INPUT_SIZE:
            raise ValueError(
                "The alignment protocol requires teacher input size "
                f"{DEFAULT_INPUT_SIZE}; got {teacher_input_size}."
            )
        if output_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"Unsupported 3DINO output dtype: {output_dtype}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for 3DINO but is not available.")

        self.three_dino_root = Path(three_dino_root).expanduser().resolve()
        self.config_file = Path(config_file).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.three_dino_root.is_dir():
            raise FileNotFoundError(
                f"3DINO source root does not exist: {self.three_dino_root}"
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"3DINO checkpoint does not exist: {self.checkpoint}"
            )
        config_yaml = (
            self.config_file
            if self.config_file.suffix == ".yaml"
            else self.config_file.with_suffix(".yaml")
        )
        if not config_yaml.is_file():
            raise FileNotFoundError(f"3DINO config does not exist: {config_yaml}")

        self.device = device
        self.output_dtype = getattr(torch, output_dtype)
        self.nframes = nframes
        self.teacher_input_size = teacher_input_size
        model_args = argparse.Namespace(
            three_dino_root=str(self.three_dino_root),
            config_file=str(self.config_file),
            checkpoint=str(self.checkpoint),
            device=self.device,
        )
        self.model, self.config = load_model(model_args)

        self._autocast_context = contextlib.nullcontext
        self._autocast_kwargs: dict[str, Any] = {}
        if self.device.startswith("cuda"):
            self._autocast_context = torch.autocast
            self._autocast_kwargs = {
                "device_type": "cuda",
                "dtype": (
                    torch.bfloat16
                    if self.output_dtype == torch.bfloat16
                    else torch.float16
                ),
                "enabled": self.output_dtype != torch.float32,
            }

    def extract(self, source_path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
        """Generate one CPU feature grid without writing any local cache."""
        resolved_source = Path(source_path).expanduser().resolve()
        if not resolved_source.is_file():
            raise FileNotFoundError(f"Source medical volume does not exist: {resolved_source}")

        image = nib.load(str(resolved_source))
        volume = np.squeeze(np.asanyarray(image.dataobj))
        if volume.ndim != 3:
            raise ValueError(
                f"Expected a 3D NIfTI volume, got {volume.shape} from {resolved_source}."
            )
        source_shape = tuple(int(value) for value in volume.shape)
        source_spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        teacher_input = prepare_teacher_input(
            volume, self.nframes, self.teacher_input_size
        ).to(device=self.device, dtype=torch.float32)
        with torch.inference_mode(), self._autocast_context(**self._autocast_kwargs):
            patch_tokens = self.model.forward_features(teacher_input)[
                "x_norm_patchtokens"
            ]

        expected_tokens = DEFAULT_GRID_SIZE**3
        expected_shape = (1, expected_tokens, DEFAULT_TEACHER_DIM)
        if tuple(patch_tokens.shape) != expected_shape:
            raise ValueError(
                "Unexpected 3DINO output shape: "
                f"got {tuple(patch_tokens.shape)}, expected {expected_shape}."
            )
        # 3DINO flattening is [H,W,D]; Qwen video alignment expects [D,H,W].
        features = (
            patch_tokens[0]
            .reshape(
                DEFAULT_GRID_SIZE,
                DEFAULT_GRID_SIZE,
                DEFAULT_GRID_SIZE,
                DEFAULT_TEACHER_DIM,
            )
            .permute(2, 0, 1, 3)
            .contiguous()
            .to(device="cpu", dtype=self.output_dtype)
        )
        metadata = {
            "mode": "online",
            "source_path": str(resolved_source),
            "source_shape": list(source_shape),
            "source_spacing": list(source_spacing),
            "source_grid_thw": list(features.shape[:3]),
            "teacher_dim": int(features.shape[-1]),
            "nframes": self.nframes,
            "teacher_input_size": self.teacher_input_size,
            "output_dtype": str(self.output_dtype).removeprefix("torch."),
            "checkpoint": str(self.checkpoint),
            "config_file": str(self.config_file),
        }
        return features, metadata


def merged_config_sha256(config) -> str:
    from omegaconf import OmegaConf

    serialized = OmegaConf.to_yaml(config, resolve=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def existing_cache_is_compatible(path: Path, expected_metadata: dict[str, str]) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            feature_shape = tuple(handle.get_slice("features").get_shape())
        return feature_shape == (DEFAULT_GRID_SIZE,) * 3 + (DEFAULT_TEACHER_DIM,) and all(
            metadata.get(key) == value for key, value in expected_metadata.items()
        )
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("The first implementation intentionally supports --batch-size 1 only.")
    if args.teacher_input_size != DEFAULT_INPUT_SIZE:
        raise ValueError(
            f"The alignment cache protocol requires teacher input size {DEFAULT_INPUT_SIZE}."
        )
    if args.nframes != DEFAULT_NFRAMES:
        raise ValueError(
            f"The alignment design requires nframes={DEFAULT_NFRAMES}; got {args.nframes}."
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"3DINO checkpoint does not exist: {checkpoint_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    features_dir = output_dir / "features"
    index_path = output_dir / "index.json"
    videos = collect_unique_videos(args.data_path, args.image_folder)
    if args.limit is not None:
        videos = videos[: args.limit]
    print(f"Found {len(videos)} unique CT volumes.", flush=True)

    model, config = load_model(args)
    checkpoint_sha = sha256_file(checkpoint_path)
    config_sha = merged_config_sha256(config)
    output_dtype = getattr(torch, args.output_dtype)
    index: dict[str, Any] = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "teacher_checkpoint_sha256": checkpoint_sha,
        "teacher_config_sha256": config_sha,
        "entries": {},
    }
    if index_path.is_file() and not args.overwrite:
        with index_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("cache_format_version") == CACHE_FORMAT_VERSION:
            index["entries"].update(previous.get("entries", {}))

    autocast_context = contextlib.nullcontext
    autocast_kwargs = {}
    if args.device.startswith("cuda"):
        autocast_context = torch.autocast
        autocast_kwargs = {
            "device_type": "cuda",
            "dtype": torch.bfloat16 if output_dtype == torch.bfloat16 else torch.float16,
            "enabled": output_dtype != torch.float32,
        }

    for item_index, (source_relative_path, source_path) in tqdm(enumerate(videos, start=1), total=len(videos)):
        if not source_path.is_file():
            raise FileNotFoundError(f"Source CT does not exist: {source_path}")
        key = cache_key(source_relative_path)
        relative_cache_file = f"features/{key[:2]}/{key}.safetensors"
        cache_file = output_dir / relative_cache_file
        base_metadata = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "source_relative_path": source_relative_path,
            "nframes": str(args.nframes),
            "teacher_input_size": str(args.teacher_input_size),
            "grid_thw": f"{DEFAULT_GRID_SIZE} {DEFAULT_GRID_SIZE} {DEFAULT_GRID_SIZE}",
            "teacher_dim": str(DEFAULT_TEACHER_DIM),
            "normalization_percentiles": f"{NORMALIZATION_PERCENTILES[0]} {NORMALIZATION_PERCENTILES[1]}",
            "teacher_checkpoint_sha256": checkpoint_sha,
            "teacher_config_sha256": config_sha,
        }
        if not args.overwrite and existing_cache_is_compatible(cache_file, base_metadata):
            index["entries"][source_relative_path] = {
                "cache_key": key,
                "file": relative_cache_file,
                "metadata": base_metadata,
            }
            print(f"[{item_index}/{len(videos)}] cached {source_relative_path}", flush=True)
            continue

        image = nib.load(str(source_path))
        volume = np.squeeze(np.asanyarray(image.dataobj))
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D NIfTI volume, got {volume.shape} from {source_path}.")
        source_shape = tuple(int(value) for value in volume.shape)
        source_spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        teacher_input = prepare_teacher_input(volume, args.nframes, args.teacher_input_size).to(
            device=args.device,
            dtype=torch.float32,
        )
        with torch.inference_mode(), autocast_context(**autocast_kwargs):
            patch_tokens = model.forward_features(teacher_input)["x_norm_patchtokens"]
        expected_tokens = DEFAULT_GRID_SIZE**3
        if tuple(patch_tokens.shape) != (1, expected_tokens, DEFAULT_TEACHER_DIM):
            raise ValueError(
                "Unexpected 3DINO output shape: "
                f"got {tuple(patch_tokens.shape)}, expected {(1, expected_tokens, DEFAULT_TEACHER_DIM)}."
            )
        # 3DINO flattening is [H,W,D]; cache in Qwen video order [D,H,W].
        features = (
            patch_tokens[0]
            .reshape(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, DEFAULT_TEACHER_DIM)
            .permute(2, 0, 1, 3)
            .contiguous()
            .to(device="cpu", dtype=output_dtype)
        )
        metadata = dict(base_metadata)
        metadata.update(
            source_shape=" ".join(str(value) for value in source_shape),
            source_spacing=" ".join(f"{value:.9g}" for value in source_spacing),
        )
        atomic_save_safetensors(
            {
                "features": features,
                "grid_thw": torch.tensor([DEFAULT_GRID_SIZE] * 3, dtype=torch.int64),
            },
            metadata,
            cache_file,
        )
        index["entries"][source_relative_path] = {
            "cache_key": key,
            "file": relative_cache_file,
            "metadata": metadata,
        }
        atomic_save_json(index, index_path)
        print(f"[{item_index}/{len(videos)}] wrote {source_relative_path}", flush=True)

    atomic_save_json(index, index_path)
    print(f"Cache index written to {index_path}", flush=True)


if __name__ == "__main__":
    main()
