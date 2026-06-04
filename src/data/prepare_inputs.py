"""
Input representation builders for fine-tuning datasets.

This module converts raw CSV rows into compact, leakage-safe text prompts.
It intentionally does not know about file paths or train/test splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


UNKNOWN_VALUES = {"", "unknown", "nan", "none", "null", "n/a"}

LEAKAGE_PREFIXES = (
    "Ref_",
    "JV_",
    "Stabilised_performance_",
    "EQE_",
    "Stability_",
    "Outdoor_",
    "Stability_PCE_",
    "Outdoor_PCE_",
    "Outdoor_power_generated",
)

METADATA_COLUMNS = {
    "Ref_ID",
    "Ref_ID_temp",
    "Ref_name_of_person_entering_the_data",
    "Ref_data_entered_by_author",
    "Ref_DOI_number",
    "Ref_lead_author",
    "Ref_journal",
    "Ref_original_filename_data_upload",
    "Ref_free_text_comment",
    "Ref_internal_sample_id",
    "JV_link_raw_data",
    "Stabilised_performance_link_raw_data",
    "EQE_link_raw_data",
    "Stability_link_raw_data_for_stability_trace",
    "Outdoor_link_raw_data_for_outdoor_trace",
}

PRIMARY_COLUMNS = (
    "Cell_architecture",
    "Cell_stack_sequence",
    "Cell_area_measured",
    "Cell_flexible",
    "Cell_semitransparent",
    "Perovskite_composition_short_form",
    "Perovskite_composition_long_form",
    "Perovskite_composition_a_ions",
    "Perovskite_composition_a_ions_coefficients",
    "Perovskite_composition_b_ions",
    "Perovskite_composition_b_ions_coefficients",
    "Perovskite_composition_c_ions",
    "Perovskite_composition_c_ions_coefficients",
    "Perovskite_composition_inorganic",
    "Perovskite_composition_leadfree",
    "Perovskite_thickness",
    "Perovskite_band_gap",
    "Perovskite_deposition_procedure",
    "Perovskite_deposition_synthesis_atmosphere",
    "Perovskite_deposition_thermal_annealing_temperature",
    "Perovskite_deposition_thermal_annealing_time",
    "Perovskite_deposition_thermal_annealing_atmosphere",
    "ETL_stack_sequence",
    "ETL_deposition_procedure",
    "HTL_stack_sequence",
    "HTL_additives_compounds",
    "Backcontact_stack_sequence",
    "Backcontact_thickness_list",
    "JV_test_atmosphere",
    "JV_test_temperature",
    "JV_light_source_type",
    "JV_light_intensity",
    "JV_default_PCE_scan_direction",
)

SECONDARY_COLUMNS = (
    "Substrate_stack_sequence",
    "Substrate_cleaning_procedure",
    "Substrate_surface_roughness_rms",
    "ETL_thickness",
    "ETL_additives_compounds",
    "ETL_additives_concentrations",
    "ETL_deposition_solvents",
    "ETL_deposition_solvents_mixing_ratios",
    "ETL_deposition_reaction_solutions_compounds",
    "ETL_deposition_reaction_solutions_concentrations",
    "ETL_deposition_thermal_annealing_temperature",
    "ETL_deposition_thermal_annealing_time",
    "Perovskite_additives_compounds",
    "Perovskite_additives_concentrations",
    "Perovskite_deposition_solvents",
    "Perovskite_deposition_solvents_mixing_ratios",
    "Perovskite_deposition_reaction_solutions_compounds",
    "Perovskite_deposition_reaction_solutions_concentrations",
    "Perovskite_deposition_substrate_temperature",
    "Perovskite_deposition_thermal_annealing_relative_humidity",
    "Perovskite_storage_atmosphere",
    "Perovskite_storage_relative_humidity",
    "HTL_thickness_list",
    "HTL_additives_concentrations",
    "HTL_deposition_procedure",
    "HTL_deposition_solvents",
    "HTL_deposition_solvents_mixing_ratios",
    "HTL_deposition_thermal_annealing_temperature",
    "HTL_deposition_thermal_annealing_time",
    "Backcontact_deposition_procedure",
    "JV_light_masked_cell",
    "JV_light_mask_area",
    "JV_scan_speed",
    "JV_scan_delay_time",
    "JV_preconditioning_protocol",
)

RARE_KEYWORDS = (
    "additives",
    "quenching",
    "after_treatment",
    "surface_treatment",
    "storage",
    "humidity",
    "atmosphere",
    "encapsulation",
    "flexible",
    "semitransparent",
    "module",
    "solvent_annealing",
    "pressure",
    "passivation",
)

SECTION_PREFIXES = {
    "Device": ("Cell_", "Module_"),
    "Substrate": ("Substrate_",),
    "ETL": ("ETL_",),
    "Perovskite absorber": ("Perovskite_",),
    "HTL": ("HTL_",),
    "Back contact": ("Backcontact_",),
    "Additional front layer": ("Add_lay_front",),
    "Additional back layer": ("Add_lay_back",),
    "Encapsulation": ("Encapsulation",),
    "JV measurement": ("JV_",),
}


@dataclass(frozen=True)
class InputBuildConfig:
    """Configuration for row-to-prompt conversion."""

    representation: str = "core_rare"
    rare_column_coverage_threshold: float = 0.20
    rare_value_frequency_threshold: float = 0.02
    max_activated_features: int = 60


@dataclass(frozen=True)
class ColumnStats:
    """Global column/value frequencies used by activated-feature selection."""

    row_count: int
    nonempty_counts: Mapping[str, int]
    value_counts: Mapping[str, Mapping[str, int]]


def normalize_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    lower = text.lower()
    if lower in UNKNOWN_VALUES:
        return ""
    if ";" in lower or "|" in lower:
        parts = [part.strip() for part in lower.replace("|", ";").split(";")]
        if parts and all(part in UNKNOWN_VALUES for part in parts):
            return ""
    return text


def is_true_value(value: str) -> bool:
    return value.strip().lower() == "true"


def is_inactive_value(value: str) -> bool:
    return value.strip().lower() in {"false", "0", "0.0"}


def is_leakage_column(column: str) -> bool:
    return column in METADATA_COLUMNS or any(column.startswith(prefix) for prefix in LEAKAGE_PREFIXES)


def format_label(column: str) -> str:
    return column.replace("_", " ").strip().lower()


def format_field(column: str, value: str) -> str:
    return f"- {format_label(column)}: {value}"


def meaningful_items(row: Mapping[str, str], columns: Sequence[str]) -> list[tuple[str, str]]:
    items = []
    for column in columns:
        value = normalize_value(row.get(column))
        if value:
            items.append((column, value))
    return items


def build_core_input(row: Mapping[str, str]) -> str:
    lines = ["Task: Predict perovskite solar-cell JV performance.", "", "Core device recipe:"]
    for column, value in meaningful_items(row, PRIMARY_COLUMNS):
        lines.append(format_field(column, value))
    return "\n".join(lines)


def build_core_secondary_input(row: Mapping[str, str]) -> str:
    lines = [build_core_input(row), "", "Secondary process and measurement details:"]
    secondary = meaningful_items(row, SECONDARY_COLUMNS)
    lines.extend(format_field(column, value) for column, value in secondary)
    return "\n".join(lines)


def is_rare_activated_feature(
    column: str,
    value: str,
    stats: ColumnStats,
    config: InputBuildConfig,
) -> bool:
    if is_leakage_column(column):
        return False
    if column in PRIMARY_COLUMNS or column in SECONDARY_COLUMNS:
        return False

    lower_column = column.lower()
    if is_inactive_value(value):
        return False

    nonempty = stats.nonempty_counts.get(column, 0)
    column_coverage = nonempty / max(stats.row_count, 1)
    value_frequency = stats.value_counts.get(column, {}).get(value, 0) / max(stats.row_count, 1)

    return (
        column_coverage < config.rare_column_coverage_threshold
        or value_frequency < config.rare_value_frequency_threshold
        or is_true_value(value)
        or any(keyword in lower_column for keyword in RARE_KEYWORDS)
    )


def activated_features(
    row: Mapping[str, str],
    stats: ColumnStats,
    config: InputBuildConfig,
) -> list[tuple[str, str]]:
    features = []
    for column, raw_value in row.items():
        value = normalize_value(raw_value)
        if not value:
            continue
        if is_rare_activated_feature(column, value, stats, config):
            nonempty = stats.nonempty_counts.get(column, 0)
            value_count = stats.value_counts.get(column, {}).get(value, 0)
            features.append((nonempty, value_count, column, value))

    features.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(column, value) for _, _, column, value in features[: config.max_activated_features]]


def build_core_rare_input(row: Mapping[str, str], stats: ColumnStats, config: InputBuildConfig) -> str:
    lines = [build_core_input(row)]
    features = activated_features(row, stats, config)
    if features:
        lines.extend(["", "Activated rare or special features:"])
        lines.extend(format_field(column, value) for column, value in features)
    return "\n".join(lines)


def build_hierarchical_input(row: Mapping[str, str], stats: ColumnStats, config: InputBuildConfig) -> str:
    lines = ["Task: Predict perovskite solar-cell JV performance."]
    used_columns = set()

    for section, prefixes in SECTION_PREFIXES.items():
        section_items = []
        for column, raw_value in row.items():
            if is_leakage_column(column):
                continue
            if not column.startswith(prefixes):
                continue
            value = normalize_value(raw_value)
            if not value:
                continue
            if (
                column in PRIMARY_COLUMNS
                or column in SECONDARY_COLUMNS
                or is_rare_activated_feature(column, value, stats, config)
            ):
                section_items.append((column, value))
                used_columns.add(column)

        if section_items:
            lines.extend(["", f"{section}:"])
            lines.extend(format_field(column, value) for column, value in section_items)

    leftovers = [
        (column, value)
        for column, value in activated_features(row, stats, config)
        if column not in used_columns
    ]
    if leftovers:
        lines.extend(["", "Other activated features:"])
        lines.extend(format_field(column, value) for column, value in leftovers)

    return "\n".join(lines)


def build_input(row: Mapping[str, str], stats: ColumnStats, config: InputBuildConfig) -> str:
    if config.representation == "core":
        return build_core_input(row)
    if config.representation == "core_secondary":
        return build_core_secondary_input(row)
    if config.representation == "core_rare":
        return build_core_rare_input(row, stats, config)
    if config.representation == "hierarchical":
        return build_hierarchical_input(row, stats, config)
    raise ValueError(f"Unknown input representation: {config.representation}")
