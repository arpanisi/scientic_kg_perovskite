"""
Build fine-tuning JSONL datasets from a CSV source.

The CLI orchestrates input representation builders and output schema builders,
then writes DOI-grouped train/validation/test splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from src.data.prepare_inputs import (
    ColumnStats,
    InputBuildConfig,
    build_input,
    normalize_value,
)
from src.data.prepare_outputs import OutputBuildConfig, build_output, is_valid_target


DEFAULT_INPUT_REPRESENTATION = "core_rare"
DEFAULT_OUTPUT_SCHEMA = "performance_with_justification"
DEFAULT_SPLIT_KEY = "Ref_DOI_number"

INPUT_REPRESENTATIONS = ("core", "core_secondary", "core_rare", "hierarchical")
OUTPUT_SCHEMAS = ("performance_only", "performance_with_justification")


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_column_stats(rows: Iterable[Mapping[str, str]]) -> ColumnStats:
    row_list = list(rows)
    nonempty_counts: Counter[str] = Counter()
    value_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in row_list:
        for column, raw_value in row.items():
            value = normalize_value(raw_value)
            if not value:
                continue
            nonempty_counts[column] += 1
            value_counts[column][value] += 1

    return ColumnStats(
        row_count=len(row_list),
        nonempty_counts=dict(nonempty_counts),
        value_counts={column: dict(counts) for column, counts in value_counts.items()},
    )


def stable_bucket(value: str, seed: int) -> float:
    payload = f"{seed}:{value}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def split_name_for_key(
    key: str,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> str:
    bucket = stable_bucket(key, seed)
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + validation_ratio:
        return "validation"
    return "test"


def make_messages(input_text: str, output_payload: Mapping) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You predict perovskite solar-cell JV performance from structured device "
                "recipes. Return only valid JSON matching the requested schema."
            ),
        },
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": json.dumps(output_payload, ensure_ascii=False, sort_keys=True)},
    ]


def build_example(
    row: Mapping[str, str],
    stats: ColumnStats,
    input_config: InputBuildConfig,
    output_config: OutputBuildConfig,
) -> dict:
    input_text = build_input(row, stats, input_config)
    output_payload = build_output(row, output_config)
    return {
        "messages": make_messages(input_text, output_payload),
        "metadata": {
            "split_key": row.get(DEFAULT_SPLIT_KEY, ""),
            "input_representation": input_config.representation,
            "output_schema": output_config.schema,
        },
    }


def write_jsonl(path: Path, records: Iterable[Mapping]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fine-tuning JSONL datasets from a CSV file.")
    parser.add_argument("--csv", required=True, type=Path, help="Input CSV path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for JSONL outputs.")
    parser.add_argument(
        "--input-representation",
        choices=INPUT_REPRESENTATIONS,
        default=DEFAULT_INPUT_REPRESENTATION,
        help=f"Input representation variant. Default: {DEFAULT_INPUT_REPRESENTATION}.",
    )
    parser.add_argument(
        "--output-schema",
        choices=OUTPUT_SCHEMAS,
        default=DEFAULT_OUTPUT_SCHEMA,
        help=f"Output target schema. Default: {DEFAULT_OUTPUT_SCHEMA}.",
    )
    parser.add_argument(
        "--split-key",
        default=DEFAULT_SPLIT_KEY,
        help=f"Column used for grouped splitting. Default: {DEFAULT_SPLIT_KEY}.",
    )
    parser.add_argument("--train-ratio", type=positive_float, default=0.80)
    parser.add_argument("--validation-ratio", type=positive_float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for smoke tests.")
    parser.add_argument("--max-activated-features", type=int, default=60)
    parser.add_argument("--rare-column-coverage-threshold", type=float, default=0.20)
    parser.add_argument("--rare-value-frequency-threshold", type=float, default=0.02)
    parser.add_argument(
        "--max-pce-consistency-error",
        type=float,
        default=5.0,
        help="Drop rows whose abs(PCE - Voc*Jsc*FF) exceeds this value. Use a negative value to disable.",
    )
    return parser.parse_args()


def validate_split_ratios(train_ratio: float, validation_ratio: float) -> None:
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Split ratios must satisfy train_ratio > 0, validation_ratio >= 0, and sum < 1.")


def main() -> None:
    args = parse_args()
    build_from_args(args)


def build_from_args(args: argparse.Namespace) -> dict:
    validate_split_ratios(args.train_ratio, args.validation_ratio)

    rows = read_rows(args.csv)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    stats = build_column_stats(rows)
    input_config = InputBuildConfig(
        representation=args.input_representation,
        rare_column_coverage_threshold=args.rare_column_coverage_threshold,
        rare_value_frequency_threshold=args.rare_value_frequency_threshold,
        max_activated_features=args.max_activated_features,
    )
    output_config = OutputBuildConfig(
        schema=args.output_schema,
        max_pce_consistency_error=(
            None if args.max_pce_consistency_error < 0 else args.max_pce_consistency_error
        ),
    )

    records_by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    skipped = 0
    for row in rows:
        if not is_valid_target(row, output_config):
            skipped += 1
            continue

        split_key = normalize_value(row.get(args.split_key)) or normalize_value(row.get(DEFAULT_SPLIT_KEY))
        if not split_key:
            skipped += 1
            continue

        example = build_example(row, stats, input_config, output_config)
        example["metadata"]["split_key"] = split_key
        split = split_name_for_key(split_key, args.train_ratio, args.validation_ratio, args.seed)
        records_by_split[split].append(example)

    prefix = f"{args.input_representation}__{args.output_schema}"
    counts = {
        split: write_jsonl(args.output_dir / f"{prefix}.{split}.jsonl", records)
        for split, records in records_by_split.items()
    }

    summary = {
        "input_csv": str(args.csv),
        "output_dir": str(args.output_dir),
        "input_representation": args.input_representation,
        "output_schema": args.output_schema,
        "split_key": args.split_key,
        "rows_read": len(rows),
        "rows_skipped": skipped,
        "rows_written": sum(counts.values()),
        "split_counts": counts,
    }
    summary_path = args.output_dir / f"{prefix}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
