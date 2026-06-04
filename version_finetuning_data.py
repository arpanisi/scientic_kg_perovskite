"""
Version the processed fine-tuning dataset with DVC.

This tracks only data/processed/fine_tuning, not raw PDFs, raw CSV files, or
extraction outputs. Run this after generating train/validation/test JSONL files.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_DATA_DIR = Path("data/processed/fine_tuning")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track fine-tuning JSONL splits with DVC.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-init", action="store_true", help="Skip `dvc init`.")
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Fine-tuning data directory not found: {args.data_dir}")

    if not args.no_init and not Path(".dvc").exists():
        run(["dvc", "init"])

    run(["dvc", "add", "--force", str(args.data_dir)])
    print(f"Tracked with DVC: {args.data_dir}")
    print("Next steps:")
    print(f"  git add {args.data_dir.with_suffix('.dvc')} .gitignore .dvc")
    print('  git commit -m "Track fine-tuning dataset version"')
    print("  dvc push  # if a DVC remote is configured")


if __name__ == "__main__":
    main()
