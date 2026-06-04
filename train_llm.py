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
import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.training.data_versioning import fine_tuning_manifest, write_manifest


DEFAULT_DATA_DIR = Path("data/processed/fine_tuning")
DEFAULT_DATASET_PREFIX = "core_rare__performance_with_justification"


class JsonlSFTDataset(Dataset):
    """Completion-only SFT dataset for chat-style JSONL records."""

    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        self.records = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        messages = self.records[index]["messages"]
        prompt_text, full_text = format_sft_text(messages)

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SFTDataCollator:
    """Pad input IDs, masks, and completion-only labels."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(feature["input_ids"].numel() for feature in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_masks = []
        labels = []
        for feature in features:
            pad_length = max_length - feature["input_ids"].numel()
            input_ids.append(pad_tensor(feature["input_ids"], pad_length, pad_id))
            attention_masks.append(pad_tensor(feature["attention_mask"], pad_length, 0))
            labels.append(pad_tensor(feature["labels"], pad_length, -100))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"JSONL file is empty: {path}")
    return records


def format_sft_text(messages: Sequence[dict[str, str]]) -> tuple[str, str]:
    system = next((message["content"] for message in messages if message["role"] == "system"), "")
    user = next((message["content"] for message in messages if message["role"] == "user"), "")
    assistant = next((message["content"] for message in messages if message["role"] == "assistant"), "")

    prompt = (
        "<|system|>\n"
        f"{system}\n"
        "<|user|>\n"
        f"{user}\n"
        "<|assistant|>\n"
    )
    full = f"{prompt}{assistant}"
    return prompt, full


def pad_tensor(tensor: torch.Tensor, pad_length: int, pad_value: int) -> torch.Tensor:
    if pad_length <= 0:
        return tensor
    padding = torch.full((pad_length,), pad_value, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=0)


def data_paths(data_dir: Path, dataset_prefix: str) -> tuple[Path, Path, Path]:
    return (
        data_dir / f"{dataset_prefix}.train.jsonl",
        data_dir / f"{dataset_prefix}.validation.jsonl",
        data_dir / f"{dataset_prefix}.test.jsonl",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LLM on generated JSONL splits.")
    parser.add_argument("--model-name", required=True, help="Hugging Face model name or local model path.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset-prefix", default=DEFAULT_DATASET_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/llm_sft"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fine-tuning-method", default="full")
    parser.add_argument("--weight-update-method", default="sft")
    parser.add_argument("--mlflow-experiment", default="perovskite-performance-llm")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--registered-model-name", default=None)
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path, validation_path, test_path = data_paths(args.data_dir, args.dataset_prefix)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = JsonlSFTDataset(train_path, tokenizer, args.max_length)
    validation_dataset = JsonlSFTDataset(validation_path, tokenizer, args.max_length)
    manifest = fine_tuning_manifest(args.data_dir, args.dataset_prefix, train_path, validation_path, test_path)
    manifest_path = write_manifest(manifest, args.output_dir / "dataset_manifest.json")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=SFTDataCollator(tokenizer),
    )
    if args.no_mlflow:
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        trainer.save_model(str(args.output_dir / "final"))
        tokenizer.save_pretrained(str(args.output_dir / "final"))
        print({"train_metrics": train_result.metrics, "eval_metrics": eval_metrics})
        return

    mlflow = import_mlflow()
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run() as run:
        log_mlflow_params(mlflow, args, manifest)
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

        if args.registered_model_name:
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=args.registered_model_name,
            )


def import_mlflow():
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow logging is enabled, but mlflow is not installed. "
            "Install requirements.txt or rerun with --no-mlflow."
        ) from exc
    return mlflow


def log_mlflow_params(mlflow, args: argparse.Namespace, manifest: dict) -> None:
    params = {
        "model_name": args.model_name,
        "dataset_prefix": args.dataset_prefix,
        "data_dir": str(args.data_dir),
        "fine_tuning_method": args.fine_tuning_method,
        "weight_update_method": args.weight_update_method,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "git_commit": manifest.get("git_commit"),
        "dvc_dataset_hash": manifest.get("dvc_dataset_hash"),
    }
    for split, info in manifest["splits"].items():
        params[f"{split}_sha256"] = info["sha256"]
        params[f"{split}_rows"] = info["rows"]
    for key, value in params.items():
        if value is not None:
            mlflow.log_param(key, value)


def log_metrics(mlflow, metrics: dict, prefix: str) -> None:
    for key, value in metrics.items():
        if isinstance(value, int | float):
            mlflow.log_metric(f"{prefix}_{key}", float(value))


def log_optional_artifact(mlflow, path: Path, artifact_path: str) -> None:
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


if __name__ == "__main__":
    main()
