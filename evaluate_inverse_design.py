"""End-to-end evaluator for generated perovskite inverse-design candidates.

This script evaluates generated recipe JSONL files from train_llm.py and reports:

- predicted PCE from a leakage-safe tabular oracle
- novelty and nearest-neighbor similarity to training recipes
- physical/chemistry validity heuristics
- synthesis feasibility heuristics
- optional similarity to an external holdout recipe set

Example:
    python evaluate_inverse_design.py \
      --predictions runs/qwen_0_5b_recipe_generation_10k_v1/evaluation/test_predictions.jsonl \
      --train-jsonl runs/qwen_0_5b_recipe_generation_10k_v1/generated_data/recipe_generation.train.jsonl \
      --csv data/raw/Perovskite_database_content_all_data.csv \
      --oracle-model xgboost \
      --oracle-representation hierarchical \
      --output-json runs/qwen_0_5b_recipe_generation_10k_v1/evaluation/inverse_design_eval.json \
      --output-jsonl runs/qwen_0_5b_recipe_generation_10k_v1/evaluation/inverse_design_candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.model.regression import build_regression_model, load_perovskite_dataframe, prepare_regression_data


FORBIDDEN_TOXIC_ELEMENTS = {"Hg", "Cd", "Tl", "As"}
CHLORINATED_SOLVENT_TERMS = (
    "chlorobenzene",
    "dichlorobenzene",
    "chloroform",
    "dichloromethane",
    "dcm",
)
KNOWN_HALIDES = {"F", "Cl", "Br", "I"}
COMMON_A_SITE = {"MA", "FA", "Cs", "Rb", "K", "Na", "BA", "PEA", "EA"}
COMMON_B_SITE = {"Pb", "Sn", "Ge", "Bi", "Sb", "Cu", "Ag"}
REQUIRED_RECIPE_SECTIONS = {"composition", "device_stack", "deposition", "transport_layers"}
REQUIRED_CONSTRAINT_KEYS = {
    "lead_free",
    "no_chlorinated_solvents",
    "has_composition",
    "has_device_stack",
    "has_deposition_process",
}


@dataclass(frozen=True)
class OracleBundle:
    model: Any
    feature_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated perovskite recipe candidates.")
    parser.add_argument("--predictions", required=True, type=Path, help="Generated predictions JSONL.")
    parser.add_argument("--train-jsonl", required=True, type=Path, help="Recipe-generation train JSONL.")
    parser.add_argument("--csv", required=True, type=Path, help="Raw PDB CSV used to train the oracle.")
    parser.add_argument("--oracle-model", default="xgboost")
    parser.add_argument("--oracle-representation", default="hierarchical")
    parser.add_argument("--external-holdout-jsonl", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = read_jsonl(args.predictions)
    if args.limit is not None:
        predictions = predictions[: args.limit]

    train_recipes = read_recipe_texts(args.train_jsonl)
    external_recipes = (
        read_recipe_texts(args.external_holdout_jsonl)
        if args.external_holdout_jsonl is not None
        else []
    )
    oracle = train_oracle(args.csv, args.oracle_model, args.oracle_representation, args.seed)

    rows = []
    for record in predictions:
        rows.append(evaluate_prediction_record(record, oracle, train_recipes, external_recipes))

    summary = summarize(rows)
    result = {
        "predictions": str(args.predictions),
        "train_jsonl": str(args.train_jsonl),
        "csv": str(args.csv),
        "oracle_model": args.oracle_model,
        "oracle_representation": args.oracle_representation,
        "summary": summary,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")


def train_oracle(csv_path: Path, model_name: str, representation: str, seed: int) -> OracleBundle:
    dataframe = load_perovskite_dataframe(str(csv_path))
    data = prepare_regression_data(dataframe, representation=representation, random_state=seed)
    model = build_regression_model(model_name, data.x_train, random_state=seed)
    model.fit(data.x_train, data.y_train)
    return OracleBundle(model=model, feature_columns=list(data.x_train.columns))


def evaluate_prediction_record(
    record: dict[str, Any],
    oracle: OracleBundle,
    train_recipes: list[str],
    external_recipes: list[str],
) -> dict[str, Any]:
    payload = record.get("prediction_json")
    valid_json = isinstance(payload, dict)
    recipe = payload.get("recipe") if valid_json else None
    constraints = payload.get("constraints_satisfied") if valid_json else None

    schema = schema_checks(payload)
    chemistry = chemistry_validity(recipe)
    feasibility = synthesis_feasibility(recipe)
    recipe_text = canonical_recipe_text(recipe)
    nearest = nearest_similarity(recipe_text, train_recipes)
    external_nearest = nearest_similarity(recipe_text, external_recipes) if external_recipes else None
    oracle_prediction = predict_recipe_performance(recipe, oracle) if isinstance(recipe, dict) else None

    return {
        "index": record.get("index"),
        "valid_json": valid_json,
        "schema": schema,
        "chemistry_validity": chemistry,
        "synthesis_feasibility": feasibility,
        "similarity_to_training": nearest,
        "novelty_score": None if nearest is None else 1.0 - nearest["similarity"],
        "similarity_to_external_holdout": external_nearest,
        "oracle_prediction": oracle_prediction,
        "constraints_satisfied": constraints if isinstance(constraints, dict) else None,
        "raw_prediction_text": record.get("prediction_text", ""),
    }


def schema_checks(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "valid_top_level": False,
            "fixed_recipe_sections": False,
            "fixed_constraint_keys": False,
            "required_recipe_fields": False,
        }
    recipe = payload.get("recipe")
    constraints = payload.get("constraints_satisfied")
    return {
        "valid_top_level": set(payload) <= {"recipe", "constraints_satisfied"},
        "fixed_recipe_sections": isinstance(recipe, dict) and set(recipe) == REQUIRED_RECIPE_SECTIONS,
        "fixed_constraint_keys": isinstance(constraints, dict) and set(constraints) == REQUIRED_CONSTRAINT_KEYS,
        "required_recipe_fields": recipe_has_minimum_content(recipe),
    }


def chemistry_validity(recipe: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(recipe, dict):
        return {"valid": False, "issues": ["missing_recipe"]}
    composition = recipe.get("composition", {})
    issues = []
    a_ions = split_value(composition.get("a_ions"))
    b_ions = split_value(composition.get("b_ions"))
    c_ions = split_value(composition.get("c_ions"))
    all_elements = set(a_ions + b_ions + c_ions)

    if not (composition.get("long_form") or composition.get("short_form")):
        issues.append("missing_composition_formula")
    if not b_ions:
        issues.append("missing_b_site")
    if b_ions and not any(ion in COMMON_B_SITE for ion in b_ions):
        issues.append("unusual_b_site")
    if c_ions and not any(ion in KNOWN_HALIDES for ion in c_ions):
        issues.append("no_halide_detected")
    if any(element in FORBIDDEN_TOXIC_ELEMENTS for element in all_elements):
        issues.append("forbidden_toxic_element")

    lead_free = composition.get("lead_free")
    contains_pb = "Pb" in all_elements or "Pb" in str(composition.get("long_form", ""))
    if lead_free is True and contains_pb:
        issues.append("lead_free_conflicts_with_pb")
    if not a_ions:
        issues.append("missing_a_site")
    elif not any(ion in COMMON_A_SITE for ion in a_ions):
        issues.append("unusual_a_site")

    return {"valid": not issues, "issues": issues}


def synthesis_feasibility(recipe: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(recipe, dict):
        return {"feasible": False, "issues": ["missing_recipe"]}
    deposition = recipe.get("deposition", {})
    stack = recipe.get("device_stack", {})
    transport = recipe.get("transport_layers", {})
    issues = []

    if not deposition.get("perovskite_method"):
        issues.append("missing_perovskite_deposition_method")
    solvents = split_value(deposition.get("solvents"))
    if not solvents:
        issues.append("missing_solvents")
    if any(is_chlorinated_solvent(solvent) for solvent in solvents):
        issues.append("chlorinated_solvent")
    anneal_temp = parse_float(deposition.get("annealing_temperature_c"))
    if anneal_temp is not None and (anneal_temp < 0 or anneal_temp > 250):
        issues.append("annealing_temperature_outside_common_range")
    if not stack.get("architecture"):
        issues.append("missing_architecture")
    if not stack.get("stack_sequence"):
        issues.append("missing_stack_sequence")
    if not stack.get("etl") and not stack.get("htl"):
        issues.append("missing_transport_layers")
    if not stack.get("backcontact") and not transport.get("backcontact_deposition"):
        issues.append("missing_backcontact")

    return {"feasible": not issues, "issues": issues}


def predict_recipe_performance(recipe: dict[str, Any] | None, oracle: OracleBundle) -> dict[str, float] | None:
    if not isinstance(recipe, dict):
        return None
    row = recipe_to_pdb_row(recipe)
    frame = pd.DataFrame([{column: row.get(column, "") for column in oracle.feature_columns}])
    prediction = oracle.model.predict(frame)[0]
    fields = ("pce", "voc", "jsc", "ff")
    return {field: round(float(value), 4) for field, value in zip(fields, prediction)}


def recipe_to_pdb_row(recipe: dict[str, Any]) -> dict[str, Any]:
    composition = recipe.get("composition", {})
    stack = recipe.get("device_stack", {})
    deposition = recipe.get("deposition", {})
    transport = recipe.get("transport_layers", {})
    return {
        "Perovskite_composition_short_form": composition.get("short_form", ""),
        "Perovskite_composition_long_form": composition.get("long_form", ""),
        "Perovskite_composition_a_ions": join_value(composition.get("a_ions")),
        "Perovskite_composition_a_ions_coefficients": join_value(composition.get("a_ion_coefficients")),
        "Perovskite_composition_b_ions": join_value(composition.get("b_ions")),
        "Perovskite_composition_b_ions_coefficients": join_value(composition.get("b_ion_coefficients")),
        "Perovskite_composition_c_ions": join_value(composition.get("c_ions")),
        "Perovskite_composition_c_ions_coefficients": join_value(composition.get("c_ion_coefficients")),
        "Perovskite_composition_leadfree": composition.get("lead_free", ""),
        "Perovskite_composition_inorganic": composition.get("inorganic", ""),
        "Perovskite_additives_compounds": join_value(composition.get("additives")),
        "Cell_architecture": stack.get("architecture", ""),
        "Cell_stack_sequence": stack.get("stack_sequence", ""),
        "Substrate_stack_sequence": stack.get("substrate", ""),
        "ETL_stack_sequence": stack.get("etl", ""),
        "HTL_stack_sequence": stack.get("htl", ""),
        "Backcontact_stack_sequence": stack.get("backcontact", ""),
        "Perovskite_deposition_procedure": deposition.get("perovskite_method", ""),
        "Perovskite_deposition_solvents": join_value(deposition.get("solvents")),
        "Perovskite_deposition_solvents_mixing_ratios": deposition.get("solvent_ratios", ""),
        "Perovskite_deposition_thermal_annealing_temperature": deposition.get("annealing_temperature_c", ""),
        "Perovskite_deposition_thermal_annealing_time": deposition.get("annealing_time_min", ""),
        "Perovskite_deposition_thermal_annealing_atmosphere": deposition.get("annealing_atmosphere", ""),
        "Perovskite_deposition_synthesis_atmosphere": deposition.get("synthesis_atmosphere", ""),
        "ETL_deposition_procedure": transport.get("etl_deposition", ""),
        "ETL_additives_compounds": join_value(transport.get("etl_additives")),
        "HTL_deposition_procedure": transport.get("htl_deposition", ""),
        "HTL_additives_compounds": join_value(transport.get("htl_additives")),
        "Backcontact_deposition_procedure": transport.get("backcontact_deposition", ""),
    }


def nearest_similarity(recipe_text: str, references: list[str]) -> dict[str, Any] | None:
    if not recipe_text or not references:
        return None
    corpus = references + [recipe_text]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    if sims.size == 0:
        return None
    index = int(np.argmax(sims))
    return {"similarity": round(float(sims[index]), 4), "nearest_index": index}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    valid_rows = [row for row in rows if row["valid_json"]]
    pce_values = [
        row["oracle_prediction"]["pce"]
        for row in rows
        if isinstance(row.get("oracle_prediction"), dict)
        and math.isfinite(float(row["oracle_prediction"]["pce"]))
    ]
    novelty_values = [
        float(row["novelty_score"])
        for row in rows
        if row.get("novelty_score") is not None
        and math.isfinite(float(row["novelty_score"]))
    ]
    return {
        "rows": total,
        "json_validity_rate": rate(len(valid_rows), total),
        "schema_required_recipe_rate": rate(
            sum(bool(row["schema"]["required_recipe_fields"]) for row in rows),
            total,
        ),
        "chemistry_validity_rate": rate(
            sum(bool(row["chemistry_validity"]["valid"]) for row in rows),
            total,
        ),
        "synthesis_feasibility_rate": rate(
            sum(bool(row["synthesis_feasibility"]["feasible"]) for row in rows),
            total,
        ),
        "predicted_pce_mean": safe_mean(pce_values),
        "predicted_pce_max": max(pce_values) if pce_values else None,
        "novelty_mean": safe_mean(novelty_values),
        "novelty_min": min(novelty_values) if novelty_values else None,
        "novelty_max": max(novelty_values) if novelty_values else None,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_recipe_texts(path: Path) -> list[str]:
    texts = []
    for row in read_jsonl(path):
        payload = row.get("prediction_json")
        if not isinstance(payload, dict):
            payload = assistant_payload(row)
        recipe = payload.get("recipe") if isinstance(payload, dict) else None
        text = canonical_recipe_text(recipe)
        if text:
            texts.append(text)
    return texts


def assistant_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages", [])
    text = next((message.get("content", "") for message in messages if message.get("role") == "assistant"), "")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def canonical_recipe_text(recipe: Any) -> str:
    if not isinstance(recipe, dict):
        return ""
    return json.dumps(recipe, ensure_ascii=False, sort_keys=True)


def recipe_has_minimum_content(recipe: Any) -> bool:
    if not isinstance(recipe, dict):
        return False
    composition = recipe.get("composition", {})
    stack = recipe.get("device_stack", {})
    deposition = recipe.get("deposition", {})
    return bool(
        isinstance(composition, dict)
        and isinstance(stack, dict)
        and isinstance(deposition, dict)
        and (composition.get("long_form") or composition.get("short_form"))
        and stack.get("stack_sequence")
        and deposition.get("perovskite_method")
    )


def split_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.replace("|", ";").split(";") if part.strip()]


def join_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def is_chlorinated_solvent(solvent: str) -> bool:
    lower = solvent.lower()
    return any(term in lower for term in CHLORINATED_SOLVENT_TERMS)


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


if __name__ == "__main__":
    main()
