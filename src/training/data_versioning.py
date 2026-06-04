"""Dataset versioning helpers for fine-tuning JSONL splits."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def dvc_hash_for_path(path: Path) -> str | None:
    dvc_file = candidate_dvc_file(path)
    if dvc_file is None or not dvc_file.exists():
        return None
    try:
        import yaml
    except ModuleNotFoundError:
        return None

    data = yaml.safe_load(dvc_file.read_text(encoding="utf-8")) or {}
    outs = data.get("outs") or []
    if not outs:
        return None
    return outs[0].get("md5") or outs[0].get("hash")


def candidate_dvc_file(path: Path) -> Path | None:
    if path.is_dir():
        return path.with_suffix(".dvc")
    return Path(f"{path}.dvc")


def fine_tuning_manifest(
    data_dir: Path,
    dataset_prefix: str,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
) -> dict:
    split_paths = {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
    }
    split_files = {}
    for split, path in split_paths.items():
        split_files[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": count_jsonl_rows(path),
            "bytes": path.stat().st_size,
        }

    summary_path = data_dir / f"{dataset_prefix}.summary.json"
    summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "dataset_prefix": dataset_prefix,
        "data_dir": str(data_dir),
        "git_commit": git_commit(),
        "dvc_dataset_hash": dvc_hash_for_path(data_dir),
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "summary_sha256": sha256_file(summary_path) if summary_path.exists() else None,
        "summary": summary,
        "splits": split_files,
    }


def write_manifest(manifest: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
