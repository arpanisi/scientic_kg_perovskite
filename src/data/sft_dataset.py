"""Dataset and collation utilities for completion-only SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset


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
        prompt_text, assistant_text = format_sft_text(messages)

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        assistant_ids = self.tokenizer(
            assistant_text,
            add_special_tokens=False,
        )["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            assistant_ids = assistant_ids + [eos_id]

        input_ids, labels = completion_only_sequence(prompt_ids, assistant_ids, self.max_length)
        attention_mask = [1] * len(input_ids)

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
    return prompt, assistant


def completion_only_sequence(
    prompt_ids: Sequence[int],
    assistant_ids: Sequence[int],
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Build labels while preserving assistant tokens under truncation."""
    if max_length < 2:
        raise ValueError("max_length must be at least 2 for completion-only SFT.")

    completion_ids = list(assistant_ids)
    if not completion_ids:
        raise ValueError("Assistant completion is empty; cannot build SFT labels.")

    if len(completion_ids) >= max_length:
        input_ids = completion_ids[:max_length]
        return input_ids, list(input_ids)

    prompt_budget = max_length - len(completion_ids)
    kept_prompt_ids = list(prompt_ids)[-prompt_budget:] if prompt_budget > 0 else []
    input_ids = kept_prompt_ids + completion_ids
    labels = [-100] * len(kept_prompt_ids) + completion_ids
    return input_ids, labels


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
