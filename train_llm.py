"""
Simple supervised fine-tuning entry point for generated JSONL datasets.

Expected JSONL format per line:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

The script trains on assistant completion tokens only; system/user prompt tokens
are masked with -100 labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.sft_dataset import JsonlSFTDataset, SFTDataCollator, data_paths
from src.data.build_datasets import DEFAULT_SPLIT_KEY, split_name_for_key, validate_split_ratios
from src.data.prepare_inputs import normalize_value
from src.data.prepare_outputs import OutputBuildConfig, is_valid_target, parse_float
from src.evaluate.generate import generate_predictions_from_model
from src.evaluate.metrics import evaluate_predictions_file
from src.model.setup import configure_fine_tuning, load_model, should_use_bf16, should_use_fp16
from src.tracking.experiment import import_mlflow, log_metrics, log_mlflow_params, log_optional_artifact
from src.tracking.versioning import fine_tuning_manifest, write_manifest
from src.training.distributed import is_distributed, is_main_process
from src.training.fine_tuning import FineTuningMethod, list_strategy_names
from src.training.trainer import WeightedSFTTrainer
from src.training.weight_update import WeightUpdateMethod, get_weight_update, list_weight_update_names


DEFAULT_DATA_DIR = Path("data/processed/fine_tuning")
DEFAULT_DATASET_PREFIX = "core_rare__performance_with_justification"
DEFAULT_RECIPE_GENERATION_PREFIX = "recipe_generation"
RECIPE_TOP_LEVEL_KEYS = {"recipe", "constraints_satisfied"}
RECIPE_SECTION_KEYS = {"composition", "device_stack", "deposition", "transport_layers"}
CONSTRAINT_KEYS = {
    "lead_free",
    "no_chlorinated_solvents",
    "has_composition",
    "has_device_stack",
    "has_deposition_process",
}
CHLORINATED_SOLVENT_TERMS = (
    "chlorobenzene",
    "dichlorobenzene",
    "chloroform",
    "dichloromethane",
    "dcm",
    "chlorinated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LLM on generated JSONL splits.")
    parser.add_argument("--model-name", required=True, help="Hugging Face model name or local model path.")
    parser.add_argument(
        "--task",
        choices=("performance_prediction", "recipe_generation"),
        default="performance_prediction",
        help="Training task. recipe_generation builds target-constraints -> recipe JSONL from --source-csv.",
    )
    parser.add_argument("--source-csv", type=Path, default=None, help="Raw CSV used to build recipe-generation JSONL.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset-prefix", default=DEFAULT_DATASET_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/llm_sft"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fine-tuning-method", choices=list_strategy_names(), default=FineTuningMethod.FULL.value)
    parser.add_argument("--weight-update-method", choices=list_weight_update_names(), default=WeightUpdateMethod.SFT.value)
    parser.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=("hf_default", "eager", "sdpa", "flash_attention_2"),
        default="hf_default",
        help="Non-invasive Hugging Face attention backend ablation.",
    )
    parser.add_argument("--attn-implementation", default=None, help="Raw HF attn_implementation override.")
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-target-modules", nargs="+", default=None)
    parser.add_argument("--num-prefix-tokens", type=int, default=None)
    parser.add_argument("--num-virtual-tokens", type=int, default=None)
    parser.add_argument("--prompt-init-text", default=None)
    parser.add_argument("--mlflow-experiment", default="perovskite-performance-llm")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--registered-model-name", default=None)
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    parser.add_argument("--run-generation-eval", action="store_true")
    parser.add_argument("--generation-limit", type=int, default=None)
    parser.add_argument("--generation-max-input-length", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--generation-temperature", type=float, default=0.0)
    parser.add_argument("--generation-top-p", type=float, default=1.0)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--metrics-csv-output", type=Path, default=None)
    parser.add_argument("--loss-default-weight", type=float, default=None)
    parser.add_argument("--loss-field-weight", action="append", default=[])
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument("--split-key", default=DEFAULT_SPLIT_KEY)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--min-source-pce", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    weight_update = get_weight_update(args.weight_update_method)
    if args.task == "recipe_generation":
        args.data_dir, args.dataset_prefix = build_recipe_generation_data(args)
    train_path, validation_path, test_path = data_paths(args.data_dir, args.dataset_prefix)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = configure_fine_tuning(model, args)

    loss_config = loss_weight_config(args, weight_update)
    train_dataset = JsonlSFTDataset(train_path, tokenizer, args.max_length, **loss_config)
    validation_dataset = JsonlSFTDataset(validation_path, tokenizer, args.max_length, **loss_config)
    manifest = fine_tuning_manifest(args.data_dir, args.dataset_prefix, train_path, validation_path, test_path)
    manifest_path = None
    if is_main_process():
        manifest_path = write_manifest(manifest, args.output_dir / "dataset_manifest.json")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        fp16=should_use_fp16(args),
        bf16=should_use_bf16(args),
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters if is_distributed() else None,
    )

    trainer_class = trainer_for_weight_update(args.weight_update_method)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=SFTDataCollator(tokenizer),
    )
    if args.no_mlflow:
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        if is_main_process():
            trainer.save_model(str(args.output_dir / "final"))
            tokenizer.save_pretrained(str(args.output_dir / "final"))
        wait_for_processes(trainer)
        generation_metrics = run_generation_eval(trainer.model, tokenizer, test_path, args)
        if is_main_process():
            print({"train_metrics": train_result.metrics, "eval_metrics": eval_metrics, "generation_metrics": generation_metrics})
        return

    if not is_main_process():
        trainer.train()
        trainer.evaluate()
        wait_for_processes(trainer)
        return

    mlflow = import_mlflow()
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run() as run:
        log_mlflow_params(mlflow, args, manifest)
        if manifest_path is not None:
            mlflow.log_artifact(str(manifest_path), artifact_path="data")
        log_optional_artifact(mlflow, Path("config/fine_tuning.yaml"), "config")
        log_optional_artifact(mlflow, Path("config/weight_update.yaml"), "config")
        log_optional_artifact(mlflow, args.data_dir / f"{args.dataset_prefix}.summary.json", "data")

        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        log_metrics(mlflow, train_result.metrics, prefix="train")
        log_metrics(mlflow, eval_metrics, prefix="eval")

        final_dir = args.output_dir / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        mlflow.log_artifacts(str(final_dir), artifact_path="model")
        wait_for_processes(trainer)
        generation_metrics = run_generation_eval(trainer.model, tokenizer, test_path, args, mlflow=mlflow)
        if generation_metrics is not None:
            log_generation_metrics(mlflow, generation_metrics)

        if args.registered_model_name:
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=args.registered_model_name,
            )


def build_recipe_generation_data(args: argparse.Namespace) -> tuple[Path, str]:
    if args.source_csv is None:
        raise ValueError("--source-csv is required when --task recipe_generation.")
    validate_split_ratios(args.train_ratio, args.validation_ratio)

    output_dir = args.output_dir / "generated_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = DEFAULT_RECIPE_GENERATION_PREFIX
    output_config = OutputBuildConfig(max_pce_consistency_error=5.0)
    records_by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    rows_read = 0
    rows_skipped = 0

    with args.source_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_read += 1
            if args.max_source_rows is not None and rows_read > args.max_source_rows:
                break
            pce = parse_float(row.get("JV_default_PCE"))
            if pce is None or pce < args.min_source_pce:
                rows_skipped += 1
                continue
            if not is_valid_target(row, output_config):
                rows_skipped += 1
                continue
            split_key = normalize_value(row.get(args.split_key)) or normalize_value(row.get(DEFAULT_SPLIT_KEY))
            if not split_key:
                rows_skipped += 1
                continue
            recipe = recipe_payload_from_row(row)
            if not recipe_has_minimum_content(recipe):
                rows_skipped += 1
                continue
            split = split_name_for_key(split_key, args.train_ratio, args.validation_ratio, args.seed)
            records_by_split[split].append(recipe_generation_example(row, recipe, split_key))

    counts = {
        split: write_local_jsonl(output_dir / f"{prefix}.{split}.jsonl", records)
        for split, records in records_by_split.items()
    }
    summary = {
        "task": "recipe_generation",
        "source_csv": str(args.source_csv),
        "output_dir": str(output_dir),
        "dataset_prefix": prefix,
        "split_key": args.split_key,
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "rows_read": rows_read,
        "rows_skipped": rows_skipped,
        "rows_written": sum(counts.values()),
        "split_counts": counts,
        "min_source_pce": args.min_source_pce,
    }
    (output_dir / f"{prefix}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if is_main_process():
        print(json.dumps(summary, indent=2, sort_keys=True))
    return output_dir, prefix


def recipe_generation_example(row: dict[str, str], recipe: dict, split_key: str) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate perovskite solar-cell recipe candidates from target constraints. "
                    "Return exactly one valid JSON object and nothing else."
                ),
            },
            {"role": "user", "content": recipe_generation_prompt(row)},
            {"role": "assistant", "content": json.dumps(recipe, ensure_ascii=False, sort_keys=True)},
        ],
        "metadata": {
            "task": "recipe_generation",
            "split_key": split_key,
            "source_pce": parse_float(row.get("JV_default_PCE")),
        },
    }


def recipe_generation_prompt(row: dict[str, str]) -> str:
    pce = parse_float(row.get("JV_default_PCE"))
    pce_min = max(0.0, round((pce or 0.0) - 0.5, 1))
    constraints = {
        "target_pce_min": pce_min,
        "architecture": normalize_value(row.get("Cell_architecture")) or "any",
        "lead_free": parse_boolish(row.get("Perovskite_composition_leadfree")),
        "inorganic": parse_boolish(row.get("Perovskite_composition_inorganic")),
        "no_chlorinated_solvents": not row_mentions_chlorinated_solvents(row),
        "allowed_deposition": normalize_value(row.get("Perovskite_deposition_procedure")) or "any",
    }
    lines = [
        "Task: Design a perovskite solar-cell recipe candidate.",
        "",
        "Target constraints:",
    ]
    for key, value in constraints.items():
        if value is not None and value != "":
            lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "Return exactly one JSON object.",
            "Do not write markdown.",
            "Do not write code.",
            "Do not write comments.",
            "Do not write text after the JSON.",
            "Use only these top-level keys: recipe, constraints_satisfied.",
            "recipe must contain only: composition, device_stack, deposition, transport_layers.",
            "constraints_satisfied must contain only: lead_free, no_chlorinated_solvents, has_composition, has_device_stack, has_deposition_process.",
            "Do not include measured JV target values in the recipe.",
        ]
    )
    return "\n".join(lines)


def recipe_payload_from_row(row: dict[str, str]) -> dict:
    lead_free = parse_boolish(row.get("Perovskite_composition_leadfree"))
    chlorinated = row_mentions_chlorinated_solvents(row)
    return {
        "recipe": {
            "composition": {
                "short_form": normalize_value(row.get("Perovskite_composition_short_form")),
                "long_form": normalize_value(row.get("Perovskite_composition_long_form")),
                "a_ions": normalize_value(row.get("Perovskite_composition_a_ions")),
                "a_ion_coefficients": normalize_value(row.get("Perovskite_composition_a_ions_coefficients")),
                "b_ions": normalize_value(row.get("Perovskite_composition_b_ions")),
                "b_ion_coefficients": normalize_value(row.get("Perovskite_composition_b_ions_coefficients")),
                "c_ions": normalize_value(row.get("Perovskite_composition_c_ions")),
                "c_ion_coefficients": normalize_value(row.get("Perovskite_composition_c_ions_coefficients")),
                "additives": split_list_value(row.get("Perovskite_additives_compounds")),
                "lead_free": lead_free,
                "inorganic": parse_boolish(row.get("Perovskite_composition_inorganic")),
            },
            "device_stack": {
                "architecture": normalize_value(row.get("Cell_architecture")),
                "stack_sequence": normalize_value(row.get("Cell_stack_sequence")),
                "substrate": normalize_value(row.get("Substrate_stack_sequence")),
                "etl": normalize_value(row.get("ETL_stack_sequence")),
                "htl": normalize_value(row.get("HTL_stack_sequence")),
                "backcontact": normalize_value(row.get("Backcontact_stack_sequence")),
            },
            "deposition": {
                "perovskite_method": normalize_value(row.get("Perovskite_deposition_procedure")),
                "solvents": split_list_value(row.get("Perovskite_deposition_solvents")),
                "solvent_ratios": normalize_value(row.get("Perovskite_deposition_solvents_mixing_ratios")),
                "annealing_temperature_c": parse_float(row.get("Perovskite_deposition_thermal_annealing_temperature")),
                "annealing_time_min": parse_float(row.get("Perovskite_deposition_thermal_annealing_time")),
                "annealing_atmosphere": normalize_value(row.get("Perovskite_deposition_thermal_annealing_atmosphere")),
                "synthesis_atmosphere": normalize_value(row.get("Perovskite_deposition_synthesis_atmosphere")),
            },
            "transport_layers": {
                "etl_deposition": normalize_value(row.get("ETL_deposition_procedure")),
                "etl_additives": split_list_value(row.get("ETL_additives_compounds")),
                "htl_deposition": normalize_value(row.get("HTL_deposition_procedure")),
                "htl_additives": split_list_value(row.get("HTL_additives_compounds")),
                "backcontact_deposition": normalize_value(row.get("Backcontact_deposition_procedure")),
            },
        },
        "constraints_satisfied": {
            "lead_free": lead_free,
            "no_chlorinated_solvents": not chlorinated,
            "has_composition": bool(
                normalize_value(row.get("Perovskite_composition_long_form"))
                or normalize_value(row.get("Perovskite_composition_short_form"))
            ),
            "has_device_stack": bool(normalize_value(row.get("Cell_stack_sequence"))),
            "has_deposition_process": bool(normalize_value(row.get("Perovskite_deposition_procedure"))),
        },
    }


def recipe_has_minimum_content(recipe: dict) -> bool:
    payload = recipe.get("recipe", {})
    composition = payload.get("composition", {})
    stack = payload.get("device_stack", {})
    deposition = payload.get("deposition", {})
    return bool(
        (composition.get("long_form") or composition.get("short_form"))
        and stack.get("stack_sequence")
        and deposition.get("perovskite_method")
    )


def row_mentions_chlorinated_solvents(row: dict[str, str]) -> bool:
    solvent_columns = (
        "Perovskite_deposition_solvents",
        "ETL_deposition_solvents",
        "HTL_deposition_solvents",
        "Perovskite_deposition_anti_solvent",
    )
    text = " ".join(normalize_value(row.get(column)).lower() for column in solvent_columns)
    return any(term in text for term in CHLORINATED_SOLVENT_TERMS)


def parse_boolish(value: object) -> bool | None:
    text = normalize_value(value).lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def split_list_value(value: object) -> list[str]:
    text = normalize_value(value)
    if not text:
        return []
    return [part.strip() for part in text.replace("|", ";").split(";") if part.strip()]


def write_local_jsonl(path: Path, records: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return len(records)


def run_generation_eval(model, tokenizer, test_path: Path, args: argparse.Namespace, mlflow=None) -> dict | None:
    if not args.run_generation_eval or not is_main_process():
        return None

    predictions_path = args.predictions_output or args.output_dir / "evaluation" / "test_predictions.jsonl"
    metrics_path = args.metrics_output or args.output_dir / "evaluation" / "test_metrics.json"
    metrics_csv_path = args.metrics_csv_output or args.output_dir / "evaluation" / "test_metrics.csv"
    max_input_length = args.generation_max_input_length or args.max_length

    generate_predictions_from_model(
        model=model,
        tokenizer=tokenizer,
        test_jsonl=test_path,
        output_jsonl=predictions_path,
        max_input_length=max_input_length,
        max_new_tokens=args.generation_max_new_tokens,
        temperature=args.generation_temperature,
        top_p=args.generation_top_p,
        limit=args.generation_limit,
        batch_size=args.generation_batch_size,
    )
    if args.task == "recipe_generation":
        metrics = evaluate_recipe_generation_predictions(predictions_path, metrics_path)
        metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_csv_path.write_text("", encoding="utf-8")
    else:
        metrics = evaluate_predictions_file(predictions_path, metrics_path, metrics_csv_path)
    print({"generation_eval_metrics": metrics})

    if mlflow is not None:
        mlflow.log_artifact(str(predictions_path), artifact_path="evaluation")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        mlflow.log_artifact(str(metrics_csv_path), artifact_path="evaluation")
    return metrics


def evaluate_recipe_generation_predictions(predictions_path: Path, metrics_path: Path) -> dict:
    total = 0
    valid_json = 0
    required_recipe = 0
    constraints_present = 0
    all_constraint_flags_true = 0
    allowed_top_level_keys = 0
    fixed_recipe_sections = 0
    fixed_constraint_keys = 0
    no_chlorinated_solvents = 0
    lead_free_true = 0

    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            payload = row.get("prediction_json")
            if not isinstance(payload, dict):
                continue
            valid_json += 1
            if set(payload) <= RECIPE_TOP_LEVEL_KEYS:
                allowed_top_level_keys += 1
            recipe = payload.get("recipe")
            if isinstance(recipe, dict):
                if set(recipe) == RECIPE_SECTION_KEYS:
                    fixed_recipe_sections += 1
                if recipe_has_minimum_content(payload):
                    required_recipe += 1
            constraints = payload.get("constraints_satisfied")
            if isinstance(constraints, dict):
                constraints_present += 1
                if set(constraints) == CONSTRAINT_KEYS:
                    fixed_constraint_keys += 1
                if constraints and all(value is True for value in constraints.values() if isinstance(value, bool)):
                    all_constraint_flags_true += 1
                if constraints.get("no_chlorinated_solvents") is True:
                    no_chlorinated_solvents += 1
                if constraints.get("lead_free") is True:
                    lead_free_true += 1

    metrics = {
        "rows": total,
        "json_validity_rate": safe_rate(valid_json, total),
        "allowed_top_level_keys_rate": safe_rate(allowed_top_level_keys, total),
        "required_recipe_rate": safe_rate(required_recipe, total),
        "fixed_recipe_sections_rate": safe_rate(fixed_recipe_sections, total),
        "constraints_present_rate": safe_rate(constraints_present, total),
        "fixed_constraint_keys_rate": safe_rate(fixed_constraint_keys, total),
        "all_constraint_flags_true_rate": safe_rate(all_constraint_flags_true, total),
        "no_chlorinated_solvents_rate": safe_rate(no_chlorinated_solvents, total),
        "lead_free_true_rate": safe_rate(lead_free_true, total),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def safe_rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def wait_for_processes(trainer) -> None:
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        accelerator.wait_for_everyone()


def trainer_for_weight_update(method: str):
    if WeightUpdateMethod(method) == WeightUpdateMethod.SFT:
        return Trainer
    if WeightUpdateMethod(method) == WeightUpdateMethod.WEIGHTED_SFT:
        return WeightedSFTTrainer
    raise NotImplementedError(
        f"weight_update_method={method} is configured, but train_llm.py currently implements sft and weighted_sft."
    )


def loss_weight_config(args: argparse.Namespace, weight_update) -> dict:
    if weight_update.method != WeightUpdateMethod.WEIGHTED_SFT:
        return {}

    defaults = weight_update.default_hyperparameters
    field_weights = dict(defaults.get("field_weights", {}))
    field_weights.update(parse_field_weight_overrides(args.loss_field_weight))
    default_weight = (
        args.loss_default_weight
        if args.loss_default_weight is not None
        else float(defaults.get("default_weight", 1.0))
    )
    return {
        "default_loss_weight": float(default_weight),
        "field_loss_weights": {key: float(value) for key, value in field_weights.items()},
        "emit_loss_weights": True,
    }


def parse_field_weight_overrides(overrides: list[str]) -> dict[str, float]:
    parsed = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError("--loss-field-weight must use FIELD=WEIGHT format.")
        field, value = override.split("=", 1)
        parsed[field.strip()] = float(value)
    return parsed


def log_generation_metrics(mlflow, metrics: dict) -> None:
    for key, value in flatten_metrics(metrics).items():
        if isinstance(value, int | float):
            mlflow.log_metric(f"generation_{key}", float(value))


def flatten_metrics(payload: dict, prefix: str = "") -> dict[str, int | float | None]:
    flattened = {}
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, name))
        elif isinstance(value, int | float) or value is None:
            flattened[name] = value
    return flattened


if __name__ == "__main__":
    main()
