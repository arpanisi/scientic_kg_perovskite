"""
Normalize per-paper output.yaml into normalized.json in the same directory.

Strips the value/source/grounded envelope from each field, leaving plain values.
Three input shapes handled:
  1. Minimal  — {filename, is_perovskite_synthesis: false}  (skipped paper)
  2. Failed   — {error: ...} or {raw_output: ...}
  3. Full     — complete value/source/grounded extraction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import yaml


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _extract_value(node):
    """
    Recursively strip the value/source/grounded envelope.

    {value: X, source: ..., grounded: ..., source_doc: ...}  ->  X
      (source_doc is preserved as a sibling key on the parent, not here)
    {name: X, concentration: Y, source: ...}  ->  {name: X, concentration: Y}
    [item, ...]  ->  [normalised item, ...]
    scalar  ->  scalar
    """
    if isinstance(node, dict):
        if "value" in node and "source" in node:
            return node["value"]
        return {
            k: _extract_value(v)
            for k, v in node.items()
            if k not in ("source", "grounded")
        }
    if isinstance(node, list):
        return [_extract_value(item) for item in node]
    return node


def _extract_source_doc(node) -> str | None:
    """Return source_doc from a leaf {value, source, source_doc} node, or None."""
    if isinstance(node, dict) and "source" in node:
        return node.get("source_doc")
    return None


def _count_grounding(node) -> tuple[int, int]:
    """Return (grounded_count, total_count) across all grounded leaf nodes."""
    if isinstance(node, dict):
        if "grounded" in node:
            return (1 if node.get("grounded") is True else 0, 1)
        g, t = 0, 0
        for v in node.values():
            dg, dt = _count_grounding(v)
            g += dg
            t += dt
        return g, t
    if isinstance(node, list):
        g, t = 0, 0
        for item in node:
            dg, dt = _count_grounding(item)
            g += dg
            t += dt
        return g, t
    return 0, 0


def _clean_precursors(precursors: list) -> list:
    """Drop null-placeholder entries left when the LLM found no precursors."""
    return [p for p in precursors if any(v is not None for v in p.values())]


# ---------------------------------------------------------------------------
# Fixed schema template
# ---------------------------------------------------------------------------

FIXED_SCHEMA: dict = {
    "source_file": None,
    "title": None,
    "authors": [],
    "doi": None,
    "is_perovskite_synthesis": None,
    "material": None,
    "precursors": [],
    "synthesis_method": None,
    "process_conditions": {
        "temperature": None,
        "duration": None,
        "atmosphere": None,
        "annealing": None,
    },
    "substrate": None,
    "characterization": {
        "xrd": None,
        "sem": None,
        "bandgap": None,
        "film_thickness": None,
    },
    "device_performance": {
        "pce": None,
        "voc": None,
        "jsc": None,
        "ff": None,
    },
    "key_findings": None,
    "figures": [],
    "_meta": {
        "extraction_complete": False,
        "reason": None,
        "paper_type": None,
        "extraction_model": None,
        "extraction_timestamp": None,
        "grounding_score": None,
        "si_referenced": False,
        "si_merged": False,
        "has_si": False,
        "si_fields": [],
    },
}


def _merge_schema(base: dict, data: dict) -> dict:
    """Deep-merge data into base, preserving base keys that are absent in data."""
    result = {}
    for key, default in base.items():
        if key not in data:
            result[key] = default
        elif isinstance(default, dict) and isinstance(data[key], dict):
            result[key] = _merge_schema(default, data[key])
        else:
            result[key] = data[key]
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_record(raw: dict, source_stem: str) -> dict:
    """
    Transform a raw output.yaml dict into a flat normalised record.
    Every record has the same fixed schema — missing fields are null/[].
    """
    extracted: dict = {"source_file": source_stem}

    # Determine extraction status
    if set(raw.keys()) <= {"filename", "is_perovskite_synthesis"}:
        # Minimal — paper was skipped (not perovskite)
        extracted["is_perovskite_synthesis"] = raw.get("is_perovskite_synthesis", False)
        extracted["_meta"] = {"extraction_complete": False, "reason": "not_perovskite"}

    elif "error" in raw or ("raw_output" in raw and "title" not in raw):
        # Failed extraction
        extracted["is_perovskite_synthesis"] = raw.get("is_perovskite_synthesis")
        extracted["_meta"] = {"extraction_complete": False, "reason": "extraction_failed"}

    else:
        # Full extraction
        figures = raw.get("figures", [])
        existing_meta = raw.get("_meta")
        synthesis = {k: v for k, v in raw.items() if k not in ("figures", "_meta")}

        grounded, total = _count_grounding(synthesis)
        normalised = _extract_value(synthesis)

        if isinstance(normalised.get("precursors"), list):
            normalised["precursors"] = _clean_precursors(normalised["precursors"])

        extracted.update(normalised)
        extracted["figures"] = figures

        if existing_meta:
            meta = dict(existing_meta)
            meta["extraction_complete"] = True
        else:
            meta = {
                "extraction_complete": True,
                "grounded_fields": grounded,
                "total_fields": total,
            }
        extracted["_meta"] = meta

    return _merge_schema(FIXED_SCHEMA, extracted)


def normalise_paper(paper_dir: Path) -> Path | None:
    """
    Read output.yaml from paper_dir, write normalized.json alongside it.
    Returns the path written, or None if output.yaml not found.
    """
    yaml_path = paper_dir / "output.yaml"
    if not yaml_path.exists():
        return None
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    record = normalize_record(raw, paper_dir.name)
    out_path = paper_dir / "normalized.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return out_path


def iter_paper_dirs(output_dir: Path) -> Iterator[Path]:
    """Yield every subdirectory of output_dir that contains an output.yaml."""
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and (d / "output.yaml").exists():
            yield d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "data" / "output"

    written, skipped = 0, 0
    for paper_dir in iter_paper_dirs(output_dir):
        out = normalise_paper(paper_dir)
        if out:
            print(f"  {paper_dir.name} -> normalized.json")
            written += 1
        else:
            skipped += 1

    print(f"\nDone: {written} normalised, {skipped} skipped.")


if __name__ == "__main__":
    main()
