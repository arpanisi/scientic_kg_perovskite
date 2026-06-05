"""RLVR/GRPO training for perovskite recipe inverse design.

This is the post-SFT stage:

1. Start from a supervised recipe-generation adapter produced by train_llm.py.
2. Generate multiple completions per target-constraint prompt.
3. Score completions with verifiable rewards:
   - valid JSON and fixed schema
   - requested constraint satisfaction
   - chemistry/synthesis sanity checks
   - DOI-split tabular oracle predicted PCE
   - novelty with a high-similarity penalty
4. Update the adapter with TRL GRPO.

The goal is not to replace the oracle. The oracle is the reward model used to
push a recipe generator away from imitation and toward higher-scoring candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from evaluate_inverse_design import (  # noqa: E402
    canonical_recipe_text,
    chemistry_validity,
    nearest_similarity,
    predict_recipe_performance,
    read_recipe_texts,
    schema_checks,
    synthesis_feasibility,
    train_oracle,
)
from src.data.sft_dataset import format_sft_text, read_jsonl  # noqa: E402
from src.evaluate.generate import generate_predictions_from_model  # noqa: E402


@dataclass(frozen=True)
class RewardContext:
    oracle: Any
    train_recipes: list[str]
    target_pce_weight: float
    validity_weight: float
    constraint_weight: float
    chemistry_weight: float
    feasibility_weight: float
    novelty_weight: float
    similarity_penalty_weight: float
    invalid_json_penalty: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO/RLVR on an SFT recipe generator.")
    parser.add_argument("--model-name", required=True, help="Base Hugging Face model name.")
    parser.add_argument("--sft-adapter-dir", required=True, type=Path, help="Path to train_llm.py final adapter.")
    parser.add_argument("--train-jsonl", required=True, type=Path, help="Recipe-generation train JSONL.")
    parser.add_argument("--source-csv", required=True, type=Path, help="Raw PDB CSV for the DOI-split oracle.")
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--oracle-model", default="xgboost")
    parser.add_argument("--oracle-representation", default="hierarchical")
    parser.add_argument("--max-prompts", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model with 4-bit quantization.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--run-generation-eval", action="store_true")
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--generation-limit", type=int, default=100)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--generation-max-new-tokens", type=int, default=512)
    parser.add_argument("--generation-temperature", type=float, default=0.0)
    parser.add_argument("--generation-top-p", type=float, default=1.0)

    parser.add_argument("--target-pce-weight", type=float, default=2.0)
    parser.add_argument("--validity-weight", type=float, default=1.0)
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--chemistry-weight", type=float, default=1.0)
    parser.add_argument("--feasibility-weight", type=float, default=1.0)
    parser.add_argument("--novelty-weight", type=float, default=0.5)
    parser.add_argument("--similarity-penalty-weight", type=float, default=1.0)
    parser.add_argument("--invalid-json-penalty", type=float, default=-2.0)
    parser.add_argument("--dry-run-rewards", action="store_true", help="Score gold completions without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reward_context = RewardContext(
        oracle=train_oracle(args.source_csv, args.oracle_model, args.oracle_representation, args.seed),
        train_recipes=read_recipe_texts(args.train_jsonl),
        target_pce_weight=args.target_pce_weight,
        validity_weight=args.validity_weight,
        constraint_weight=args.constraint_weight,
        chemistry_weight=args.chemistry_weight,
        feasibility_weight=args.feasibility_weight,
        novelty_weight=args.novelty_weight,
        similarity_penalty_weight=args.similarity_penalty_weight,
        invalid_json_penalty=args.invalid_json_penalty,
    )
    dataset = build_prompt_dataset(args.train_jsonl, args.max_prompts)

    if args.dry_run_rewards:
        dry_run_rewards(dataset, reward_context)
        return

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "RLVR requires TRL with GRPO support. Install requirements.txt on the GPU runtime "
            "or run: pip install 'trl>=0.12,<1'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_sft_adapter_model(args)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        fp16=should_use_fp16(args),
        bf16=should_use_bf16(args),
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[make_reward_function(reward_context)],
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    if args.run_generation_eval:
        if args.eval_jsonl is None:
            raise ValueError("--eval-jsonl is required with --run-generation-eval.")
        predictions_output = args.predictions_output or args.output_dir / "evaluation" / "test_predictions.jsonl"
        generate_predictions_from_model(
            model=trainer.model,
            tokenizer=tokenizer,
            test_jsonl=args.eval_jsonl,
            output_jsonl=predictions_output,
            max_input_length=args.max_prompt_length,
            max_new_tokens=args.generation_max_new_tokens,
            temperature=args.generation_temperature,
            top_p=args.generation_top_p,
            limit=args.generation_limit,
            batch_size=args.generation_batch_size,
        )


def load_sft_adapter_model(args: argparse.Namespace):
    model_kwargs: dict[str, Any] = {}
    dtype = resolve_torch_dtype(args.torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype or torch.float16,
        )

    base = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    return PeftModel.from_pretrained(base, str(args.sft_adapter_dir), is_trainable=True)


def build_prompt_dataset(train_jsonl: Path, max_prompts: int | None) -> Dataset:
    rows = read_jsonl(train_jsonl)
    if max_prompts is not None:
        rows = rows[:max_prompts]

    records = []
    for row in rows:
        messages = row["messages"]
        prompt, gold_completion = format_sft_text(messages)
        records.append(
            {
                "prompt": prompt,
                "gold_completion": gold_completion,
                "target_pce_min": extract_target_pce_min(prompt),
                "requires_lead_free": prompt_has_bool_constraint(prompt, "lead_free", True),
                "requires_no_chlorinated_solvents": prompt_has_bool_constraint(
                    prompt,
                    "no_chlorinated_solvents",
                    True,
                ),
            }
        )
    return Dataset.from_list(records)


def make_reward_function(context: RewardContext):
    def reward_function(*args: Any, **kwargs: Any) -> list[float]:
        completions = reward_completions_from_call(args, kwargs)
        target_pces = field_values(kwargs, "target_pce_min", len(completions), default=None)
        require_lead_free = field_values(kwargs, "requires_lead_free", len(completions), default=False)
        require_no_chlorinated = field_values(
            kwargs,
            "requires_no_chlorinated_solvents",
            len(completions),
            default=False,
        )
        return [
            score_completion(
                completion=completion,
                target_pce_min=target_pces[index],
                requires_lead_free=bool(require_lead_free[index]),
                requires_no_chlorinated_solvents=bool(require_no_chlorinated[index]),
                context=context,
            )
            for index, completion in enumerate(completions)
        ]

    return reward_function


def reward_completions_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    if "completions" in kwargs:
        completions = kwargs["completions"]
    elif len(args) == 1:
        completions = args[0]
    elif len(args) >= 2:
        completions = args[1]
    else:
        completions = []
    return list(completions)


def score_completion(
    completion: Any,
    target_pce_min: float | None,
    requires_lead_free: bool,
    requires_no_chlorinated_solvents: bool,
    context: RewardContext,
) -> float:
    text = completion_text(completion)
    payload = parse_json_object(text)
    if not isinstance(payload, dict):
        return context.invalid_json_penalty

    recipe = payload.get("recipe")
    constraints = payload.get("constraints_satisfied")
    schema = schema_checks(payload)
    chemistry = chemistry_validity(recipe)
    feasibility = synthesis_feasibility(recipe)
    oracle_prediction = predict_recipe_performance(recipe, context.oracle) if isinstance(recipe, dict) else None
    nearest = nearest_similarity(canonical_recipe_text(recipe), context.train_recipes)

    reward = 0.0
    reward += context.validity_weight
    reward += context.validity_weight * bool(schema["valid_top_level"])
    reward += context.validity_weight * bool(schema["fixed_recipe_sections"])
    reward += context.validity_weight * bool(schema["fixed_constraint_keys"])
    reward += context.validity_weight * bool(schema["required_recipe_fields"])

    if isinstance(constraints, dict):
        if requires_lead_free:
            reward += context.constraint_weight if constraints.get("lead_free") is True else -context.constraint_weight
        if requires_no_chlorinated_solvents:
            reward += (
                context.constraint_weight
                if constraints.get("no_chlorinated_solvents") is True
                else -context.constraint_weight
            )
        reward += context.constraint_weight * bool(constraints.get("has_composition"))
        reward += context.constraint_weight * bool(constraints.get("has_device_stack"))
        reward += context.constraint_weight * bool(constraints.get("has_deposition_process"))

    reward += context.chemistry_weight if chemistry["valid"] else -0.5 * len(chemistry["issues"])
    reward += context.feasibility_weight if feasibility["feasible"] else -0.5 * len(feasibility["issues"])

    if isinstance(oracle_prediction, dict):
        predicted_pce = float(oracle_prediction["pce"])
        reward += context.target_pce_weight * max(0.0, min(predicted_pce / 30.0, 1.5))
        if target_pce_min is not None:
            reward += context.target_pce_weight if predicted_pce >= target_pce_min else -0.5

    if nearest is not None:
        similarity = float(nearest["similarity"])
        novelty = 1.0 - similarity
        reward += context.novelty_weight * novelty
        if similarity > 0.98:
            reward -= context.similarity_penalty_weight * (similarity - 0.98) / 0.02

    return float(reward)


def dry_run_rewards(dataset: Dataset, context: RewardContext) -> None:
    rows = list(dataset)
    completions = [row["gold_completion"] for row in rows[: min(16, len(rows))]]
    kwargs = {
        "target_pce_min": [row["target_pce_min"] for row in rows[: len(completions)]],
        "requires_lead_free": [row["requires_lead_free"] for row in rows[: len(completions)]],
        "requires_no_chlorinated_solvents": [
            row["requires_no_chlorinated_solvents"] for row in rows[: len(completions)]
        ],
    }
    rewards = make_reward_function(context)(completions=completions, **kwargs)
    print(
        json.dumps(
            {
                "rows_scored": len(rewards),
                "reward_min": min(rewards) if rewards else None,
                "reward_mean": sum(rewards) / len(rewards) if rewards else None,
                "reward_max": max(rewards) if rewards else None,
                "rewards": rewards,
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(text)
    except json.JSONDecodeError:
        return None
    if text[end:].strip():
        return None
    return payload if isinstance(payload, dict) else None


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(item.get("content", "") if isinstance(item, dict) else str(item) for item in completion)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def field_values(kwargs: dict[str, Any], key: str, length: int, default: Any) -> list[Any]:
    value = kwargs.get(key, default)
    if isinstance(value, list | tuple):
        return list(value)
    return [value for _ in range(length)]


def extract_target_pce_min(prompt: str) -> float | None:
    match = re.search(r"target_pce_min:\s*([0-9]+(?:\.[0-9]+)?)", prompt)
    if not match:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) else None


def prompt_has_bool_constraint(prompt: str, key: str, expected: bool) -> bool:
    expected_text = "True" if expected else "False"
    return f"{key}: {expected_text}" in prompt


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "auto":
        if not torch.cuda.is_available():
            return None
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def should_use_fp16(args: argparse.Namespace) -> bool:
    return torch.cuda.is_available() and resolve_torch_dtype(args.torch_dtype) == torch.float16


def should_use_bf16(args: argparse.Namespace) -> bool:
    return torch.cuda.is_available() and resolve_torch_dtype(args.torch_dtype) == torch.bfloat16


if __name__ == "__main__":
    main()
