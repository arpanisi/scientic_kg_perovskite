"""Generate model predictions from held-out JSONL examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.sft_dataset import format_sft_text, read_jsonl
from src.evaluate.metrics import parse_json_object
from src.evaluate.physics import verify_prediction_payload


def generate_predictions(
    model_path: str | Path,
    test_jsonl: Path,
    output_jsonl: Path,
    base_model_name: str | None = None,
    max_input_length: int = 1024,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    torch_dtype: str = "auto",
    device: str | None = None,
    limit: int | None = None,
    batch_size: int = 1,
) -> Path:
    tokenizer = load_tokenizer(model_path, base_model_name)
    model = load_generation_model(model_path, base_model_name, torch_dtype=torch_dtype, device=device)
    model.eval()

    rows = read_jsonl(test_jsonl)
    if limit is not None:
        rows = rows[:limit]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), batch_size):
            for result in generate_batch(
                rows=rows[start : start + batch_size],
                start_index=start,
                model=model,
                tokenizer=tokenizer,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                handle.write(json.dumps(result, sort_keys=True) + "\n")
    return output_jsonl


def generate_predictions_from_model(
    model,
    tokenizer,
    test_jsonl: Path,
    output_jsonl: Path,
    max_input_length: int = 1024,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    limit: int | None = None,
    batch_size: int = 1,
) -> Path:
    """Generate predictions using an already-loaded model."""
    model.eval()
    rows = read_jsonl(test_jsonl)
    if limit is not None:
        rows = rows[:limit]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), batch_size):
            for result in generate_batch(
                rows=rows[start : start + batch_size],
                start_index=start,
                model=model,
                tokenizer=tokenizer,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                handle.write(json.dumps(result, sort_keys=True) + "\n")
    return output_jsonl


def load_tokenizer(model_path: str | Path, base_model_name: str | None = None):
    tokenizer_source = base_model_name or str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_generation_model(
    model_path: str | Path,
    base_model_name: str | None = None,
    torch_dtype: str = "auto",
    device: str | None = None,
):
    dtype = resolve_dtype(torch_dtype)
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    if base_model_name:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(selected_device)


def generate_one(
    row: dict[str, Any],
    index: int,
    model,
    tokenizer,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    return generate_batch(
        rows=[row],
        start_index=index,
        model=model,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )[0]


def generate_batch(
    rows: Sequence[dict[str, Any]],
    start_index: int,
    model,
    tokenizer,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    prompt_truth_pairs = [prompt_and_truth(row["messages"]) for row in rows]
    prompt_texts = [prompt for prompt, _ in prompt_truth_pairs]
    ground_truth_texts = [truth for _, truth in prompt_truth_pairs]

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    tokenizer.padding_side = original_padding_side
    encoded = {key: value.to(model_device(model)) for key, value in encoded.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        generated = model.generate(**encoded, **generation_kwargs)

    generated_tokens = generated[:, encoded["input_ids"].shape[-1] :]
    prediction_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    results = []
    for offset, (row, prompt_text, ground_truth_text, prediction_text) in enumerate(
        zip(rows, prompt_texts, ground_truth_texts, prediction_texts)
    ):
        prediction_text = prediction_text.strip()
        prediction_json = parse_json_object(prediction_text)
        ground_truth_json = parse_json_object(ground_truth_text)
        prediction_verification = verify_prediction_payload(prediction_json)

        results.append(
            {
                "index": start_index + offset,
                "metadata": row.get("metadata", {}),
                "prompt_text": prompt_text,
                "ground_truth_text": ground_truth_text,
                "ground_truth_json": ground_truth_json,
                "prediction_text": prediction_text,
                "prediction_json": prediction_json,
                "prediction_verification": prediction_verification,
            }
        )
    return results


def prompt_and_truth(messages: Sequence[dict[str, str]]) -> tuple[str, str]:
    prompt_messages = [message for message in messages if message["role"] in {"system", "user"}]
    ground_truth = next((message["content"] for message in messages if message["role"] == "assistant"), "")
    prompt, _ = format_sft_text([*prompt_messages, {"role": "assistant", "content": ""}])
    return prompt, ground_truth


def resolve_dtype(dtype_name: str) -> torch.dtype | None:
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


def model_device(model) -> torch.device:
    return next(model.parameters()).device
