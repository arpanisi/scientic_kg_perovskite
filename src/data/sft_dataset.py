"""Dataset and collation utilities for completion-only SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset


class JsonlSFTDataset(Dataset):
    """Completion-only SFT dataset for chat-style JSONL records."""

    def __init__(
        self,
        path: Path,
        tokenizer,
        max_length: int,
        default_loss_weight: float = 1.0,
        field_loss_weights: dict[str, float] | None = None,
        emit_loss_weights: bool = False,
    ) -> None:
        self.records = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.default_loss_weight = default_loss_weight
        self.field_loss_weights = field_loss_weights or {}
        self.emit_loss_weights = emit_loss_weights

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        messages = self.records[index]["messages"]
        prompt_text, assistant_text = format_sft_text(messages)

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        assistant_ids = self.tokenizer(
            assistant_text,
            add_special_tokens=False,
        )["input_ids"]
        assistant_weights = assistant_token_weights(
            assistant_text,
            self.tokenizer,
            self.default_loss_weight,
            self.field_loss_weights,
        )
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            assistant_ids = assistant_ids + [eos_id]
            assistant_weights = assistant_weights + [self.default_loss_weight]

        input_ids, labels, loss_weights = completion_only_sequence(
            prompt_ids,
            assistant_ids,
            self.max_length,
            assistant_weights=assistant_weights,
            default_loss_weight=self.default_loss_weight,
        )
        attention_mask = [1] * len(input_ids)

        output = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if self.emit_loss_weights:
            output["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float)
        return output


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
        loss_weights = []
        for feature in features:
            pad_length = max_length - feature["input_ids"].numel()
            input_ids.append(pad_tensor(feature["input_ids"], pad_length, pad_id))
            attention_masks.append(pad_tensor(feature["attention_mask"], pad_length, 0))
            labels.append(pad_tensor(feature["labels"], pad_length, -100))
            if "loss_weights" in feature:
                loss_weights.append(pad_tensor(feature["loss_weights"], pad_length, 0.0))

        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }
        if loss_weights:
            batch["loss_weights"] = torch.stack(loss_weights)
        return batch


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
    return prompt, assistant


def completion_only_sequence(
    prompt_ids: Sequence[int],
    assistant_ids: Sequence[int],
    max_length: int,
    assistant_weights: Sequence[float] | None = None,
    default_loss_weight: float = 1.0,
) -> tuple[list[int], list[int], list[float]]:
    """Build labels while preserving assistant tokens under truncation."""
    if max_length < 2:
        raise ValueError("max_length must be at least 2 for completion-only SFT.")

    completion_ids = list(assistant_ids)
    completion_weights = (
        list(assistant_weights)
        if assistant_weights is not None
        else [default_loss_weight] * len(completion_ids)
    )
    if not completion_ids:
        raise ValueError("Assistant completion is empty; cannot build SFT labels.")
    if len(completion_weights) != len(completion_ids):
        raise ValueError("assistant_weights must match assistant_ids length.")

    if len(completion_ids) >= max_length:
        input_ids = completion_ids[:max_length]
        return input_ids, list(input_ids), completion_weights[:max_length]

    prompt_budget = max_length - len(completion_ids)
    kept_prompt_ids = list(prompt_ids)[-prompt_budget:] if prompt_budget > 0 else []
    input_ids = kept_prompt_ids + completion_ids
    labels = [-100] * len(kept_prompt_ids) + completion_ids
    loss_weights = [0.0] * len(kept_prompt_ids) + completion_weights
    return input_ids, labels, loss_weights


def assistant_token_weights(
    assistant_text: str,
    tokenizer,
    default_weight: float,
    field_weights: dict[str, float],
) -> list[float]:
    """Map configured JSON field weights onto assistant completion tokens."""
    encoded = tokenizer(
        assistant_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if not offsets:
        return [default_weight] * len(input_ids)

    char_weights = [default_weight] * len(assistant_text)
    for field_path, weight in field_weights.items():
        field_name = field_path.split(".")[-1]
        for start, end in field_value_spans(assistant_text, field_name):
            for index in range(start, min(end, len(char_weights))):
                char_weights[index] = max(char_weights[index], float(weight))

    token_weights = []
    for start, end in offsets:
        if end <= start:
            token_weights.append(default_weight)
            continue
        token_weights.append(max(char_weights[start:end], default=default_weight))
    return token_weights


def field_value_spans(json_text: str, field_name: str) -> list[tuple[int, int]]:
    """Return approximate character spans for a JSON field key and value."""
    spans = []
    decoder = json.JSONDecoder()
    search_from = 0
    key_text = json.dumps(field_name)
    while True:
        key_start = json_text.find(key_text, search_from)
        if key_start == -1:
            break
        colon = json_text.find(":", key_start + len(key_text))
        if colon == -1:
            break
        value_start = colon + 1
        while value_start < len(json_text) and json_text[value_start].isspace():
            value_start += 1
        try:
            _, value_end = decoder.raw_decode(json_text[value_start:])
            spans.append((key_start, value_start + value_end))
        except json.JSONDecodeError:
            spans.append((key_start, fallback_value_end(json_text, value_start)))
        search_from = colon + 1
    return spans


def fallback_value_end(text: str, start: int) -> int:
    comma = text.find(",", start)
    brace = text.find("}", start)
    candidates = [index for index in (comma, brace) if index != -1]
    return min(candidates) if candidates else len(text)


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
