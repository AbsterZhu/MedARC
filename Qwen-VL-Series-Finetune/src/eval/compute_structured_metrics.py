#!/usr/bin/env python3
"""Compute reusable structured metrics from saved medical VQA predictions.

The script does not run inference.  It recursively discovers
``vqa/predictions_*.jsonl`` files and writes one sibling
``structured_metrics_*.json`` file for each prediction file.

The three reported scores are micro-aggregates over every object that can be
parsed without a model-based evaluator:

* ACC: exact-match categorical objects plus tolerance-based numeric objects.
* F1: label-set micro F1 for categorical objects.
* MAE: mean absolute error, in millimetres, for unambiguous measurements.

The output also records supports and parser coverage so that scores cannot be
misinterpreted when a generated answer is not parseable.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


_TASK_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9.]+")
_CHOICE_MARK_RE = re.compile(r"\(([a-d])\)", re.IGNORECASE)
_OPTION_MARK_RE = re.compile(r"\(([a-d])\)\s*", re.IGNORECASE)
_YES_NO_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
_MEASUREMENT_RE = re.compile(
    r"(?<![a-z0-9.])"
    r"(?P<values>\d+(?:\.\d+)?(?:\s*[x×]\s*\d+(?:\.\d+)?){0,2})"
    r"\s*(?P<unit>mm|cm)\b",
    re.IGNORECASE,
)

_DIRECT_LABEL_TYPES = {"presence", "location", "multiple_choice"}
_ENTITY_SET_TYPES = {"abnormality", "disorders"}
_LONG_ANSWER_TYPES = {"conversation", "description", "free_response"}


def _normalise_text(text: str) -> str:
    text = _TASK_TAG_RE.sub(" ", text.lower())
    text = text.replace("lungs", "lung")
    return " ".join(_NON_ALNUM_RE.sub(" ", text).split()).strip(" .")


def _parse_yes_no(text: str) -> str | None:
    match = _YES_NO_RE.match(text)
    return match.group(1).lower() if match else None


def _parse_marked_values(text: str) -> dict[str, str]:
    matches = list(_OPTION_MARK_RE.finditer(text))
    options: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = _normalise_text(text[match.end() : end])
        if value:
            options[match.group(1).lower()] = value
    return options


def _choice_labels(answer: str, question: str) -> set[str]:
    letters = {match.lower() for match in _CHOICE_MARK_RE.findall(answer)}
    options = _parse_marked_values(question)
    if letters:
        answer_values = _parse_marked_values(answer)
        return {
            f"choice:{options[letter]}"
            if letter in options
            else (
                f"choice:{answer_values[letter]}"
                if letter in answer_values
                else f"choice_option:{letter}"
            )
            for letter in letters
        }

    normalised = _normalise_text(answer)
    return {f"choice:{normalised}"} if normalised else set()


def _split_entity_labels(text: str, question_type: str) -> set[str]:
    normalised = _normalise_text(text)
    if not normalised:
        return set()
    no_finding_answers = {
        "no",
        "none",
        "normal",
        "no finding",
        "no findings",
        "no abnormal finding",
        "no abnormal findings",
        "no abnormality",
        "no abnormalities",
        "no disorder",
        "no disorders",
    }
    if normalised in no_finding_answers or re.fullmatch(
        r"(?:there (?:is|are) )?no (?:abnormal )?findings?", normalised
    ):
        return {f"{question_type}:no_finding"}

    # These two short-answer types are written as comma/semicolon-separated
    # findings.  Splitting on "and" is intentionally avoided because it often
    # occurs inside a single medical concept.
    parts = (_normalise_text(part) for part in re.split(r"\s*[,;]\s*", text))
    return {f"{question_type}:{part}" for part in parts if part}


def _categorical_labels(row: dict[str, Any], field: str) -> set[str] | None:
    question_type = str(row.get("question_type") or "")
    text = str(row.get(field) or "")

    if question_type == "presence":
        label = _parse_yes_no(text)
        return {f"presence:{label}"} if label else set()

    if question_type == "multiple_choice":
        return _choice_labels(text, str(row.get("question") or ""))

    if question_type == "location":
        label = _normalise_text(text)
        return {f"location:{label}"} if label else set()

    if question_type in _ENTITY_SET_TYPES:
        return _split_entity_labels(text, question_type)

    if question_type in _LONG_ANSWER_TYPES:
        label = _parse_yes_no(text)
        return {f"{question_type}_yes_no:{label}"} if label else set()

    return None


def _has_categorical_reference(row: dict[str, Any]) -> bool:
    question_type = str(row.get("question_type") or "")
    reference_labels = _categorical_labels(row, "reference")
    if question_type in _LONG_ANSWER_TYPES:
        return bool(reference_labels)
    return question_type in _DIRECT_LABEL_TYPES | _ENTITY_SET_TYPES and bool(reference_labels)


def _parse_measurements_mm(text: str) -> list[list[float]]:
    measurements: list[list[float]] = []
    for match in _MEASUREMENT_RE.finditer(text):
        scale = 10.0 if match.group("unit").lower() == "cm" else 1.0
        values = [
            float(value) * scale
            for value in re.split(r"\s*[x×]\s*", match.group("values"))
        ]
        measurements.append(values)
    return measurements


def _numeric_reference_value(row: dict[str, Any]) -> float | None:
    measurements = _parse_measurements_mm(str(row.get("reference") or ""))
    if not measurements:
        return None
    if row.get("question_type") == "size":
        return max(value for measurement in measurements for value in measurement)
    if len(measurements) == 1:
        return max(measurements[0])
    return None


def _numeric_prediction_value(row: dict[str, Any]) -> float | None:
    measurements = _parse_measurements_mm(str(row.get("prediction") or ""))
    if not measurements:
        return None
    if row.get("question_type") == "size":
        return max(value for measurement in measurements for value in measurement)
    if len(measurements) == 1:
        return max(measurements[0])
    return None


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    prediction_path = Path(path)
    rows: list[dict[str, Any]] = []
    with prediction_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object at {prediction_path}:{line_number}."
                )
            for key in ("question_type", "reference", "prediction"):
                if key not in value:
                    raise ValueError(
                        f"Missing {key!r} at {prediction_path}:{line_number}."
                    )
            rows.append(value)
    if not rows:
        raise ValueError(f"No predictions found in {prediction_path}.")
    return rows


def compute_structured_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    absolute_tolerance_mm: float = 1.0,
    relative_tolerance: float = 0.20,
) -> dict[str, Any]:
    """Return one aggregate ACC, F1 and MAE plus their supports."""
    if absolute_tolerance_mm < 0:
        raise ValueError("absolute_tolerance_mm must be non-negative.")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative.")

    question_type_counts: Counter[str] = Counter()
    acc_eligible_by_type: Counter[str] = Counter()
    f1_eligible_by_type: Counter[str] = Counter()
    mae_eligible_by_type: Counter[str] = Counter()

    acc_correct = 0
    acc_objects = 0
    f1_tp = f1_fp = f1_fn = 0
    f1_objects = 0
    absolute_errors_mm: list[float] = []
    numeric_reference_objects = 0

    for row in rows:
        question_type = str(row.get("question_type") or "unknown")
        question_type_counts[question_type] += 1

        if _has_categorical_reference(row):
            reference_labels = _categorical_labels(row, "reference") or set()
            prediction_labels = _categorical_labels(row, "prediction") or set()
            acc_objects += 1
            f1_objects += 1
            acc_eligible_by_type[question_type] += 1
            f1_eligible_by_type[question_type] += 1
            acc_correct += int(reference_labels == prediction_labels)
            f1_tp += len(reference_labels & prediction_labels)
            f1_fp += len(prediction_labels - reference_labels)
            f1_fn += len(reference_labels - prediction_labels)

        reference_value = _numeric_reference_value(row)
        if reference_value is None:
            continue
        numeric_reference_objects += 1
        prediction_value = _numeric_prediction_value(row)

        # A numeric reference always contributes an ACC object.  An
        # unparseable/ambiguous prediction is counted as incorrect rather than
        # silently disappearing from the denominator.
        acc_objects += 1
        acc_eligible_by_type[question_type] += 1
        if prediction_value is None:
            continue

        error_mm = abs(prediction_value - reference_value)
        tolerance_mm = max(
            absolute_tolerance_mm,
            relative_tolerance * abs(reference_value),
        )
        acc_correct += int(error_mm <= tolerance_mm)
        absolute_errors_mm.append(error_mm)
        mae_eligible_by_type[question_type] += 1

    f1_denominator = 2 * f1_tp + f1_fp + f1_fn
    return {
        "ACC": acc_correct / acc_objects if acc_objects else None,
        "F1": (2 * f1_tp / f1_denominator) if f1_denominator else None,
        "MAE": (
            sum(absolute_errors_mm) / len(absolute_errors_mm)
            if absolute_errors_mm
            else None
        ),
        "support": {
            "ACC": {
                "num_objects": acc_objects,
                "num_correct": acc_correct,
                "eligible_by_question_type": dict(sorted(acc_eligible_by_type.items())),
            },
            "F1": {
                "num_objects": f1_objects,
                "true_positive": f1_tp,
                "false_positive": f1_fp,
                "false_negative": f1_fn,
                "eligible_by_question_type": dict(sorted(f1_eligible_by_type.items())),
            },
            "MAE": {
                "num_objects": len(absolute_errors_mm),
                "unit": "mm",
                "eligible_by_question_type": dict(sorted(mae_eligible_by_type.items())),
            },
        },
        "coverage": {
            "num_samples": len(rows),
            "question_types": dict(sorted(question_type_counts.items())),
            "numeric_reference_objects": numeric_reference_objects,
            "numeric_predictions_parsed": len(absolute_errors_mm),
        },
        "config": {
            "aggregation": "micro_over_parsed_structured_objects",
            "ACC_objects": [
                "categorical_exact_match",
                "numeric_within_tolerance",
            ],
            "F1_objects": "categorical_label_sets",
            "MAE_objects": "unambiguous_measurements_converted_to_mm",
            "categorical_question_types": sorted(
                _DIRECT_LABEL_TYPES | _ENTITY_SET_TYPES | _LONG_ANSWER_TYPES
            ),
            "long_answer_rule": "only_reference_answers_starting_with_yes_or_no",
            "entity_set_rule": "normalised_comma_or_semicolon_separated_short_answers",
            "non_size_numeric_rule": "exactly_one_measurement_in_each_answer",
            "size_numeric_rule": "largest_reported_dimension",
            "absolute_tolerance_mm": absolute_tolerance_mm,
            "relative_tolerance": relative_tolerance,
        },
    }


def _prediction_files(inputs: Iterable[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for raw_path in inputs:
        path = Path(raw_path).expanduser()
        if path.is_file():
            if path.name.startswith("predictions_") and path.suffix == ".jsonl":
                files.add(path.resolve())
            else:
                raise ValueError(f"Not a predictions_*.jsonl file: {path}")
        elif path.is_dir():
            files.update(
                candidate.resolve()
                for candidate in path.rglob("predictions_*.jsonl")
                if candidate.parent.name == "vqa"
            )
        else:
            raise FileNotFoundError(path)
    return sorted(files)


def _output_path(prediction_path: Path) -> Path:
    method = prediction_path.stem.removeprefix("predictions_")
    return prediction_path.with_name(f"structured_metrics_{method}.json")


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute ACC, F1 and MAE from saved structured VQA answers."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["output_four_methods"],
        help="Prediction JSONL files or directories to scan (default: output_four_methods).",
    )
    parser.add_argument(
        "--absolute-tolerance-mm",
        type=float,
        default=1.0,
        help="Absolute component of numeric ACC tolerance (default: 1 mm).",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.20,
        help="Relative component of numeric ACC tolerance (default: 0.20).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print metrics without writing structured_metrics_*.json files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prediction_files = _prediction_files(args.inputs)
    if not prediction_files:
        raise SystemExit("No predictions_*.jsonl files found.")

    for prediction_path in prediction_files:
        rows = read_predictions(prediction_path)
        metrics = compute_structured_metrics(
            rows,
            absolute_tolerance_mm=args.absolute_tolerance_mm,
            relative_tolerance=args.relative_tolerance,
        )
        metrics["source"] = str(prediction_path)
        output_path = _output_path(prediction_path)
        if not args.dry_run:
            _write_json(output_path, metrics)
        print(
            f"{prediction_path}: ACC={metrics['ACC']:.6f}, "
            f"F1={metrics['F1']:.6f}, MAE={metrics['MAE']:.6f} mm"
            + (" (dry run)" if args.dry_run else f" -> {output_path}")
        )


if __name__ == "__main__":
    main()
