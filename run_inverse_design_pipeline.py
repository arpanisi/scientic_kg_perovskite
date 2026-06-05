"""Run the current inverse-design pipeline end to end.

This script orchestrates:

1. Recipe-generation SFT via train_llm.py
2. Held-out generation evaluation from train_llm.py
3. Inverse-design evaluation via evaluate_inverse_design.py

It is a root-level experiment runner so Kaggle can execute it after git pull.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full perovskite inverse-design experiment.")

    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument(
        "--fine-tuning-method",
        choices=("full", "partial", "lora", "qlora", "dora", "prefix_tuning", "prompt_tuning"),
        default="qlora",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("hf_default", "eager", "sdpa", "flash_attention_2"),
        default="hf_default",
    )
    parser.add_argument("--attn-implementation", default=None)

    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-source-rows", type=int, default=10000)
    parser.add_argument("--min-source-pce", type=float, default=0.0)
    parser.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-target-modules", nargs="+", default=None)

    parser.add_argument("--generation-limit", type=int, default=100)
    parser.add_argument("--generation-max-new-tokens", type=int, default=512)
    parser.add_argument("--generation-batch-size", type=int, default=4)

    parser.add_argument(
        "--oracle-model",
        choices=("ridge", "random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting", "xgboost"),
        default="xgboost",
    )
    parser.add_argument(
        "--oracle-representation",
        choices=("core", "core_secondary", "core_rare", "hierarchical"),
        default="hierarchical",
    )
    parser.add_argument("--external-holdout-jsonl", type=Path, default=None)

    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-inverse-eval", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=17)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()

    if not args.skip_training:
        run_command(train_command(args), cwd=REPO_ROOT)

    if not args.skip_inverse_eval:
        predictions = args.output_dir / "evaluation" / "test_predictions.jsonl"
        train_jsonl = args.output_dir / "generated_data" / "recipe_generation.train.jsonl"
        if not predictions.exists():
            raise FileNotFoundError(f"Missing generated predictions: {predictions}")
        if not train_jsonl.exists():
            raise FileNotFoundError(f"Missing generated train JSONL: {train_jsonl}")
        run_command(inverse_eval_command(args, predictions, train_jsonl), cwd=REPO_ROOT)


def train_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "train_llm.py",
        "--task",
        "recipe_generation",
        "--source-csv",
        str(args.source_csv),
        "--model-name",
        args.model_name,
        "--fine-tuning-method",
        args.fine_tuning_method,
        "--attention-backend",
        args.attention_backend,
        "--output-dir",
        str(args.output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--max-length",
        str(args.max_length),
        "--max-source-rows",
        str(args.max_source_rows),
        "--min-source-pce",
        str(args.min_source_pce),
        "--generation-max-new-tokens",
        str(args.generation_max_new_tokens),
        "--generation-batch-size",
        str(args.generation_batch_size),
        "--run-generation-eval",
        "--generation-limit",
        str(args.generation_limit),
        "--torch-dtype",
        args.torch_dtype,
        "--seed",
        str(args.seed),
        "--no-mlflow",
    ]
    if args.eval_batch_size is not None:
        command.extend(["--eval-batch-size", str(args.eval_batch_size)])
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if args.attn_implementation:
        command.extend(["--attn-implementation", args.attn_implementation])
    if args.lora_r is not None:
        command.extend(["--lora-r", str(args.lora_r)])
    if args.lora_alpha is not None:
        command.extend(["--lora-alpha", str(args.lora_alpha)])
    if args.lora_dropout is not None:
        command.extend(["--lora-dropout", str(args.lora_dropout)])
    if args.lora_target_modules:
        command.append("--lora-target-modules")
        command.extend(args.lora_target_modules)
    return command


def inverse_eval_command(args: argparse.Namespace, predictions: Path, train_jsonl: Path) -> list[str]:
    command = [
        sys.executable,
        "evaluate_inverse_design.py",
        "--predictions",
        str(predictions),
        "--train-jsonl",
        str(train_jsonl),
        "--csv",
        str(args.source_csv),
        "--oracle-model",
        args.oracle_model,
        "--oracle-representation",
        args.oracle_representation,
        "--output-json",
        str(args.output_dir / "evaluation" / "inverse_design_eval.json"),
        "--output-jsonl",
        str(args.output_dir / "evaluation" / "inverse_design_candidates.jsonl"),
        "--seed",
        str(args.seed),
    ]
    if args.external_holdout_jsonl is not None:
        command.extend(["--external-holdout-jsonl", str(args.external_holdout_jsonl)])
    return command


def run_command(command: list[str], cwd: Path) -> None:
    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(command))
    print("=" * 100 + "\n")
    subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    main()
