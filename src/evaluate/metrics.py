"""Deterministic metrics for perovskite performance prediction outputs."""

from __future__ import annotations

import json
import math
import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.prepare_outputs import BIN_EDGES, make_bin
from src.evaluate.physics import verify_prediction_payload


METRIC_FIELDS = ("pce", "voc", "jsc", "ff")


@dataclass(frozen=True)
class FieldMetrics:
    """Regression metrics for one predicted device-performance field."""

    count: int
    mae: float | None
    rmse: float | None
    r2: float | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def write_flat_csv(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = flatten_metrics(payload)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        for key in sorted(flattened):
            writer.writerow((key, flattened[key]))
    return path


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    stripped = strip_json_fences(text.strip())
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def strip_json_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def prediction_payload(record: Mapping[str, Any]) -> dict[str, Any] | None:
    parsed = record.get("prediction_json")
    if isinstance(parsed, dict):
        return parsed
    return parse_json_object(record.get("prediction_text"))


def ground_truth_payload(record: Mapping[str, Any]) -> dict[str, Any] | None:
    parsed = record.get("ground_truth_json")
    if isinstance(parsed, dict):
        return parsed
    return parse_json_object(record.get("ground_truth_text"))


def extract_prediction_values(payload: Mapping[str, Any] | None) -> dict[str, float | None]:
    prediction = payload.get("prediction") if isinstance(payload, Mapping) else None
    if not isinstance(prediction, Mapping):
        return {field: None for field in METRIC_FIELDS}
    return {field: parse_float(prediction.get(field)) for field in METRIC_FIELDS}


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def compute_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    total = len(rows)
    predicted_payloads = [prediction_payload(row) for row in rows]
    truth_payloads = [ground_truth_payload(row) for row in rows]

    predictions = [extract_prediction_values(payload) for payload in predicted_payloads]
    truths = [extract_prediction_values(payload) for payload in truth_payloads]
    physics_checks = [verify_prediction_payload(payload) for payload in predicted_payloads]

    field_metrics = {
        field: regression_metrics(
            [truth[field] for truth in truths],
            [prediction[field] for prediction in predictions],
        ).__dict__
        for field in METRIC_FIELDS
    }

    return {
        "rows": total,
        "json_validity_rate": safe_rate(sum(payload is not None for payload in predicted_payloads), total),
        "ground_truth_validity_rate": safe_rate(sum(payload is not None for payload in truth_payloads), total),
        "field_validity": field_validity_metrics(predictions, truths),
        "field_metrics": field_metrics,
        "physical_consistency": physical_consistency_metrics(predictions),
        "physics_verification": physics_verification_metrics(physics_checks, total),
        "bin_accuracy": bin_accuracy_metrics(predictions, truths),
        "bin_confusion": bin_confusion_matrices(predictions, truths),
        "pce_subgroups": pce_subgroup_metrics(predictions, truths),
        "missing_prediction_rate": missing_prediction_rates(predictions),
    }


def regression_metrics(y_true: list[float | None], y_pred: list[float | None]) -> FieldMetrics:
    pairs = [(truth, pred) for truth, pred in zip(y_true, y_pred) if truth is not None and pred is not None]
    if not pairs:
        return FieldMetrics(count=0, mae=None, rmse=None, r2=None)

    errors = [pred - truth for truth, pred in pairs]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    truth_values = [truth for truth, _ in pairs]
    mean_truth = sum(truth_values) / len(truth_values)
    ss_tot = sum((truth - mean_truth) ** 2 for truth in truth_values)
    ss_res = sum(error * error for error in errors)
    r2 = None if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return FieldMetrics(count=len(pairs), mae=mae, rmse=rmse, r2=r2)


def physical_consistency_metrics(predictions: list[Mapping[str, float | None]]) -> dict[str, float | int | None]:
    errors = []
    for prediction in predictions:
        pce = prediction.get("pce")
        voc = prediction.get("voc")
        jsc = prediction.get("jsc")
        ff = prediction.get("ff")
        if None not in (pce, voc, jsc, ff):
            errors.append(abs(float(pce) - float(voc) * float(jsc) * float(ff)))

    if not errors:
        return {"count": 0, "mae": None, "max": None}
    return {
        "count": len(errors),
        "mae": sum(errors) / len(errors),
        "max": max(errors),
    }


def physics_verification_metrics(checks: list[Mapping[str, Any]], total: int) -> dict[str, Any]:
    inconsistent = sum(bool(check.get("physically_inconsistent")) for check in checks)
    schema_invalid = sum(not bool(check.get("schema_valid")) for check in checks)
    consistency_errors = [
        float(check["pce_consistency_error"])
        for check in checks
        if check.get("pce_consistency_error") is not None
    ]
    range_violations = violation_counts(checks, "range_violations")
    invalid_bins = violation_counts(checks, "invalid_bin_labels")
    bin_mismatches = violation_counts(checks, "bin_mismatches")

    return {
        "physically_inconsistent_count": inconsistent,
        "physically_inconsistent_rate": safe_rate(inconsistent, total),
        "schema_invalid_count": schema_invalid,
        "schema_invalid_rate": safe_rate(schema_invalid, total),
        "pce_consistency_count": len(consistency_errors),
        "pce_consistency_mae": safe_mean(consistency_errors),
        "pce_consistency_max": max(consistency_errors) if consistency_errors else None,
        "range_violations": range_violations,
        "invalid_bin_labels": invalid_bins,
        "bin_mismatches": bin_mismatches,
    }


def violation_counts(checks: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        for violation in check.get(key, []) or []:
            field = str(violation.get("field", "unknown"))
            counts[field] = counts.get(field, 0) + 1
    return counts


def bin_accuracy_metrics(
    predictions: list[Mapping[str, float | None]],
    truths: list[Mapping[str, float | None]],
) -> dict[str, dict[str, float | int | None]]:
    results = {}
    for field in METRIC_FIELDS:
        correct = 0
        count = 0
        for prediction, truth in zip(predictions, truths):
            pred_value = prediction.get(field)
            truth_value = truth.get(field)
            if pred_value is None or truth_value is None:
                continue
            count += 1
            correct += make_bin(field, float(pred_value)) == make_bin(field, float(truth_value))
        results[field] = {
            "count": count,
            "accuracy": safe_rate(correct, count),
        }
    return results


def field_validity_metrics(
    predictions: list[Mapping[str, float | None]],
    truths: list[Mapping[str, float | None]],
) -> dict[str, dict[str, float | int | None]]:
    total = len(predictions)
    results = {}
    for field in METRIC_FIELDS:
        predicted_count = sum(prediction.get(field) is not None for prediction in predictions)
        truth_count = sum(truth.get(field) is not None for truth in truths)
        comparable_count = sum(
            prediction.get(field) is not None and truth.get(field) is not None
            for prediction, truth in zip(predictions, truths)
        )
        results[field] = {
            "predicted_count": predicted_count,
            "truth_count": truth_count,
            "comparable_count": comparable_count,
            "prediction_validity_rate": safe_rate(predicted_count, total),
            "truth_validity_rate": safe_rate(truth_count, total),
            "comparable_rate": safe_rate(comparable_count, total),
        }
    return results


def bin_confusion_matrices(
    predictions: list[Mapping[str, float | None]],
    truths: list[Mapping[str, float | None]],
) -> dict[str, dict[str, dict[str, int]]]:
    matrices = {}
    for field in METRIC_FIELDS:
        labels = bin_labels(field)
        matrix = {truth_label: {pred_label: 0 for pred_label in labels} for truth_label in labels}
        for prediction, truth in zip(predictions, truths):
            pred_value = prediction.get(field)
            truth_value = truth.get(field)
            if pred_value is None or truth_value is None:
                continue
            truth_bin = make_bin(field, float(truth_value))
            pred_bin = make_bin(field, float(pred_value))
            matrix.setdefault(truth_bin, {})
            matrix[truth_bin][pred_bin] = matrix[truth_bin].get(pred_bin, 0) + 1
        matrices[field] = matrix
    return matrices


def bin_labels(field: str) -> list[str]:
    edges = BIN_EDGES[field]
    labels = [f"{low:g}-{high:g}" for low, high in zip(edges, edges[1:])]
    return [f"<{edges[0]:g}", *labels, f"{edges[-1]:g}+"]


def pce_subgroup_metrics(
    predictions: list[Mapping[str, float | None]],
    truths: list[Mapping[str, float | None]],
) -> dict[str, dict[str, Any]]:
    subgroups = {
        "low_pce_lt_5": [],
        "mid_pce_5_to_15": [],
        "high_pce_gte_15": [],
    }
    for prediction, truth in zip(predictions, truths):
        truth_pce = truth.get("pce")
        if truth_pce is None:
            continue
        if truth_pce < 5.0:
            subgroups["low_pce_lt_5"].append((truth, prediction))
        elif truth_pce < 15.0:
            subgroups["mid_pce_5_to_15"].append((truth, prediction))
        else:
            subgroups["high_pce_gte_15"].append((truth, prediction))

    results = {}
    for name, pairs in subgroups.items():
        subgroup_truths = [truth for truth, _ in pairs]
        subgroup_predictions = [prediction for _, prediction in pairs]
        results[name] = {
            "rows": len(pairs),
            "field_metrics": {
                field: regression_metrics(
                    [truth[field] for truth in subgroup_truths],
                    [prediction[field] for prediction in subgroup_predictions],
                ).__dict__
                for field in METRIC_FIELDS
            },
            "bin_accuracy": bin_accuracy_metrics(subgroup_predictions, subgroup_truths),
        }
    return results


def missing_prediction_rates(predictions: list[Mapping[str, float | None]]) -> dict[str, float | None]:
    total = len(predictions)
    return {
        field: safe_rate(sum(prediction.get(field) is None for prediction in predictions), total)
        for field in METRIC_FIELDS
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def flatten_metrics(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_metrics(value, name))
        elif isinstance(value, (int, float, str)) or value is None:
            flattened[name] = value
    return flattened


def evaluate_predictions_file(
    predictions_path: Path,
    output_path: Path | None = None,
    csv_output_path: Path | None = None,
) -> dict[str, Any]:
    metrics = compute_metrics(read_jsonl(predictions_path))
    if output_path is not None:
        write_json(output_path, metrics)
    if csv_output_path is not None:
        write_flat_csv(csv_output_path, metrics)
    return metrics
