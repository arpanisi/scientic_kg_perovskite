"""
Convenience CLI for building fine-tuning train/validation/test JSONL splits.

Defaults are intentionally conservative for smoke testing:
  - input CSV: data/raw/Perovskite_database_content_all_data.csv
  - output dir: data/processed/fine_tuning
  - max rows: 10
  - split key: Ref_ID
  - split ratios: 60/20/20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import build_datasets


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "Perovskite_database_content_all_data.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "fine_tuning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare fine-tuning JSONL train/validation/test splits from a raw CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Input CSV path. Default: {DEFAULT_CSV.relative_to(REPO_ROOT)}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)}.",
    )
    parser.add_argument(
        "--input-representation",
        choices=build_datasets.INPUT_REPRESENTATIONS,
        default=build_datasets.DEFAULT_INPUT_REPRESENTATION,
        help=f"Input representation variant. Default: {build_datasets.DEFAULT_INPUT_REPRESENTATION}.",
    )
    parser.add_argument(
        "--output-schema",
        choices=build_datasets.OUTPUT_SCHEMAS,
        default=build_datasets.DEFAULT_OUTPUT_SCHEMA,
        help=f"Output schema. Default: {build_datasets.DEFAULT_OUTPUT_SCHEMA}.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10,
        help="Maximum rows to read. Default: 10. Use a negative value to process all rows.",
    )
    parser.add_argument(
        "--split-key",
        default="Ref_ID",
        help="Column used for grouped splitting. Default: Ref_ID for the 10-row smoke dataset.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=17)
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


def main() -> None:
    args = parse_args()
    max_rows = None if args.max_rows < 0 else args.max_rows

    build_args = argparse.Namespace(
        csv=args.csv,
        output_dir=args.output_dir,
        input_representation=args.input_representation,
        output_schema=args.output_schema,
        split_key=args.split_key,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        max_rows=max_rows,
        max_activated_features=args.max_activated_features,
        rare_column_coverage_threshold=args.rare_column_coverage_threshold,
        rare_value_frequency_threshold=args.rare_value_frequency_threshold,
        max_pce_consistency_error=args.max_pce_consistency_error,
    )
    build_datasets.build_from_args(build_args)


if __name__ == "__main__":
    main()
