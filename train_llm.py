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
from pathlib import Path

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.sft_dataset import JsonlSFTDataset, SFTDataCollator, data_paths
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LLM on generated JSONL splits.")
    parser.add_argument("--model-name", required=True, help="Hugging Face model name or local model path.")
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
    parser.add_argument("--generation-temperature", type=float, default=0.0)
    parser.add_argument("--generation-top-p", type=float, default=1.0)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--metrics-csv-output", type=Path, default=None)
    parser.add_argument("--loss-default-weight", type=float, default=None)
    parser.add_argument("--loss-field-weight", action="append", default=[])
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    weight_update = get_weight_update(args.weight_update_method)
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
    )
    metrics = evaluate_predictions_file(predictions_path, metrics_path, metrics_csv_path)
    print({"generation_eval_metrics": metrics})

    if mlflow is not None:
        mlflow.log_artifact(str(predictions_path), artifact_path="evaluation")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        mlflow.log_artifact(str(metrics_csv_path), artifact_path="evaluation")
    return metrics


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
