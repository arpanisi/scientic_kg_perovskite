"""
Output target builders for fine-tuning datasets.

This module converts raw CSV rows into strict JSON-serializable targets.
It intentionally does not know about file paths or train/test splitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


TARGET_COLUMNS = {
    "pce": "JV_default_PCE",
    "voc": "JV_default_Voc",
    "jsc": "JV_default_Jsc",
    "ff": "JV_default_FF",
}

BIN_EDGES = {
    "pce": (0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
    "voc": (0.0, 0.6, 0.8, 1.0, 1.1, 1.2),
    "jsc": (0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
    "ff": (0.0, 0.4, 0.5, 0.6, 0.7, 0.8),
}

PHYSICAL_LIMITS = {
    "pce": (0.0, 35.0),
    "voc": (0.0, 2.0),
    "jsc": (0.0, 35.0),
    "ff": (0.0, 1.0),
}


@dataclass(frozen=True)
class OutputBuildConfig:
    """Configuration for row-to-target conversion."""

    schema: str = "performance_with_justification"
    max_pce_consistency_error: float | None = 5.0


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "nan", "none", "null", "n/a"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def target_values(row: Mapping[str, str]) -> dict[str, float] | None:
    values = {name: parse_float(row.get(column)) for name, column in TARGET_COLUMNS.items()}
    if any(value is None for value in values.values()):
        return None
    return {key: float(value) for key, value in values.items() if value is not None}


def within_physical_limits(values: Mapping[str, float]) -> bool:
    for name, value in values.items():
        low, high = PHYSICAL_LIMITS[name]
        if value < low or value > high:
            return False
    return True


def pce_consistency_error(values: Mapping[str, float]) -> float:
    calculated = values["voc"] * values["jsc"] * values["ff"]
    return abs(values["pce"] - calculated)


def is_valid_target(row: Mapping[str, str], config: OutputBuildConfig) -> bool:
    values = target_values(row)
    if values is None:
        return False
    if not within_physical_limits(values):
        return False
    if config.max_pce_consistency_error is not None:
        if pce_consistency_error(values) > config.max_pce_consistency_error:
            return False
    return True


def make_bin(metric: str, value: float) -> str:
    edges = BIN_EDGES[metric]
    for low, high in zip(edges, edges[1:]):
        if low <= value < high:
            return f"{low:g}-{high:g}"
    if value < edges[0]:
        return f"<{edges[0]:g}"
    return f"{edges[-1]:g}+"


def confidence_label(row: Mapping[str, str], values: Mapping[str, float]) -> str:
    missing_process_fields = 0
    for column in (
        "Perovskite_deposition_procedure",
        "Perovskite_deposition_solvents",
        "Perovskite_deposition_thermal_annealing_temperature",
        "Perovskite_deposition_thermal_annealing_time",
        "ETL_stack_sequence",
        "HTL_stack_sequence",
        "Backcontact_stack_sequence",
    ):
        value = str(row.get(column, "")).strip().lower()
        if not value or value in {"unknown", "nan", "none", "null", "n/a"}:
            missing_process_fields += 1

    consistency_error = pce_consistency_error(values)
    if consistency_error <= 0.5 and missing_process_fields <= 2:
        return "high"
    if consistency_error <= 2.0 and missing_process_fields <= 4:
        return "medium"
    return "low"


def compact_justification(row: Mapping[str, str], values: Mapping[str, float]) -> list[str]:
    justifications = []

    architecture = str(row.get("Cell_architecture", "")).strip()
    stack = str(row.get("Cell_stack_sequence", "")).strip()
    composition = (
        str(row.get("Perovskite_composition_long_form", "")).strip()
        or str(row.get("Perovskite_composition_short_form", "")).strip()
    )
    deposition = str(row.get("Perovskite_deposition_procedure", "")).strip()
    anneal_temp = str(row.get("Perovskite_deposition_thermal_annealing_temperature", "")).strip()
    anneal_time = str(row.get("Perovskite_deposition_thermal_annealing_time", "")).strip()

    if architecture:
        justifications.append(f"The device architecture is {architecture}.")
    if composition:
        justifications.append(f"The absorber composition is recorded as {composition}.")
    if stack:
        justifications.append("The layer stack provides device context for transport and contact layers.")
    if deposition:
        justifications.append(f"The perovskite deposition procedure is {deposition}.")
    if anneal_temp or anneal_time:
        justifications.append("Annealing conditions are present in the input recipe.")

    consistency_error = pce_consistency_error(values)
    justifications.append(
        f"The predicted metrics have a PCE consistency error of {consistency_error:.3g} from Voc*Jsc*FF."
    )

    return justifications[:5]


def build_performance_target(row: Mapping[str, str], config: OutputBuildConfig) -> dict:
    values = target_values(row)
    if values is None:
        raise ValueError("Cannot build output target for row with missing target values.")

    prediction = {
        "pce_bin": make_bin("pce", values["pce"]),
        "voc_bin": make_bin("voc", values["voc"]),
        "jsc_bin": make_bin("jsc", values["jsc"]),
        "ff_bin": make_bin("ff", values["ff"]),
        "pce": round(values["pce"], 4),
        "voc": round(values["voc"], 4),
        "jsc": round(values["jsc"], 4),
        "ff": round(values["ff"], 4),
    }

    if config.schema == "performance_only":
        return {"prediction": prediction}

    if config.schema == "performance_with_justification":
        return {
            "prediction": prediction,
            "confidence": confidence_label(row, values),
            "justification": compact_justification(row, values),
        }

    raise ValueError(f"Unknown output schema: {config.schema}")


def build_output(row: Mapping[str, str], config: OutputBuildConfig) -> dict:
    return build_performance_target(row, config)
