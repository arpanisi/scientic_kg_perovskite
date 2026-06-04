"""MLflow experiment tracking helpers."""

from __future__ import annotations

import argparse
from pathlib import Path


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
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "attention_backend": getattr(args, "attention_backend", None),
        "attn_implementation": getattr(args, "attn_implementation", None),
        "loss_default_weight": getattr(args, "loss_default_weight", None),
        "num_prefix_tokens": getattr(args, "num_prefix_tokens", None),
        "num_virtual_tokens": getattr(args, "num_virtual_tokens", None),
        "prompt_init_text": getattr(args, "prompt_init_text", None),
        "seed": args.seed,
        "git_commit": manifest.get("git_commit"),
        "dvc_dataset_hash": manifest.get("dvc_dataset_hash"),
    }
    for index, override in enumerate(getattr(args, "loss_field_weight", []) or []):
        params[f"loss_field_weight_override_{index}"] = override
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
