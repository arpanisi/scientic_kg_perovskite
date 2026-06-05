"""Physics and schema verification for generated device-performance predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from src.data.prepare_outputs import BIN_EDGES, PHYSICAL_LIMITS, make_bin


METRIC_FIELDS = ("pce", "voc", "jsc", "ff")
DEFAULT_PCE_CONSISTENCY_THRESHOLD = 5.0


def verify_prediction_payload(
    payload: Mapping[str, Any] | None,
    pce_consistency_threshold: float = DEFAULT_PCE_CONSISTENCY_THRESHOLD,
) -> dict[str, Any]:
    """Verify generated prediction values independent of model confidence."""
    prediction = payload.get("prediction") if isinstance(payload, Mapping) else None
    if not isinstance(prediction, Mapping):
        return {
            "physically_inconsistent": True,
            "schema_valid": False,
            "pce_consistency_error": None,
            "range_violations": [],
            "invalid_bin_labels": [],
            "bin_mismatches": [],
            "missing_fields": list(METRIC_FIELDS),
        }

    values = {field: parse_float(prediction.get(field)) for field in METRIC_FIELDS}
    missing_fields = [field for field, value in values.items() if value is None]
    range_violations = range_violations_for(values)
    invalid_bin_labels = invalid_bin_labels_for(prediction)
    bin_mismatches = bin_mismatches_for(prediction, values)
    consistency_error = pce_consistency_error(values)
    consistency_violation = (
        consistency_error is not None
        and consistency_error > pce_consistency_threshold
    )

    return {
        "physically_inconsistent": bool(
            range_violations
            or invalid_bin_labels
            or bin_mismatches
            or consistency_violation
        ),
        "schema_valid": not missing_fields,
        "pce_consistency_error": consistency_error,
        "range_violations": range_violations,
        "invalid_bin_labels": invalid_bin_labels,
        "bin_mismatches": bin_mismatches,
        "missing_fields": missing_fields,
    }


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def range_violations_for(values: Mapping[str, float | None]) -> list[dict[str, Any]]:
    violations = []
    for field, value in values.items():
        if value is None:
            continue
        low, high = PHYSICAL_LIMITS[field]
        if value < low or value > high:
            violations.append({"field": field, "value": value, "low": low, "high": high})
    return violations


def invalid_bin_labels_for(prediction: Mapping[str, Any]) -> list[dict[str, Any]]:
    invalid = []
    for field in METRIC_FIELDS:
        key = f"{field}_bin"
        if key not in prediction:
            continue
        label = prediction.get(key)
        if label not in allowed_bins(field):
            invalid.append({"field": key, "value": label, "allowed": sorted(allowed_bins(field))})
    return invalid


def bin_mismatches_for(
    prediction: Mapping[str, Any],
    values: Mapping[str, float | None],
) -> list[dict[str, Any]]:
    mismatches = []
    for field, value in values.items():
        key = f"{field}_bin"
        label = prediction.get(key)
        if value is None or label is None or label not in allowed_bins(field):
            continue
        expected = make_bin(field, float(value))
        if label != expected:
            mismatches.append({"field": key, "value": label, "expected": expected})
    return mismatches


def pce_consistency_error(values: Mapping[str, float | None]) -> float | None:
    pce = values.get("pce")
    voc = values.get("voc")
    jsc = values.get("jsc")
    ff = values.get("ff")
    if None in (pce, voc, jsc, ff):
        return None
    return abs(float(pce) - float(voc) * float(jsc) * float(ff))


def allowed_bins(field: str) -> set[str]:
    edges = BIN_EDGES[field]
    labels = {f"{low:g}-{high:g}" for low, high in zip(edges, edges[1:])}
    labels.add(f"<{edges[0]:g}")
    labels.add(f"{edges[-1]:g}+")
    return labels
