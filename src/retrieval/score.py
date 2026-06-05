"""
Multi-dimensional extraction scoring.

Replaces the single grounding_score with three orthogonal metrics:

  synthesis_completeness  — fraction of key synthesis fields that are non-null
                            (measures omission; only meaningful for synthesis papers)
  grounding_precision     — grounded / verifiable_non_null
                            (measures citation quality; replaces old grounding_score)
  field_weighted_score    — weighted average over critical / important / minor fields
                            (single number that captures both completeness and grounding)

Plug-in to test_pipeline.py
----------------------------
  from src.retrieval.score import compute_scores

  # Replace the grounding_score block in main():
  scores = compute_scores(synthesis, is_synthesis_paper=synthesis.get("is_perovskite_synthesis", False))
  synthesis["_meta"].update(scores)

  # _meta will then contain:
  #   grounding_score           (backward-compat, same key as before)
  #   synthesis_completeness
  #   field_weighted_score
  #   missing_critical_fields   (list — most actionable signal)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Field weight map
# Fields are grouped by importance to synthesis extraction quality.
# Weights used in field_weighted_score (higher = matters more).
# ---------------------------------------------------------------------------

_CRITICAL = {
    "material":                      3.0,
    "precursors":                    3.0,
    "synthesis_method":              2.5,
    "process_conditions.temperature":2.0,
}

_IMPORTANT = {
    "process_conditions.duration":   1.5,
    "process_conditions.atmosphere": 1.0,
    "process_conditions.annealing":  1.0,
    "substrate":                     1.0,
    "characterization.bandgap":      1.0,
}

_MINOR = {
    "characterization.xrd":          0.5,
    "characterization.sem":          0.5,
    "characterization.film_thickness":0.5,
    "device_performance.pce":        0.5,
    "device_performance.voc":        0.3,
    "device_performance.jsc":        0.3,
    "device_performance.ff":         0.3,
    "key_findings":                  0.5,
}

_ALL_FIELDS: dict[str, float] = {**_CRITICAL, **_IMPORTANT, **_MINOR}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_leaf(d: dict, path: str) -> dict | None:
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur if isinstance(cur, dict) and "value" in cur else None


def _is_present(leaf: dict | None) -> bool:
    if leaf is None:
        return False
    v = leaf.get("value")
    if v is None or v == "" or v == []:
        return False
    return True


def _is_grounded(leaf: dict | None) -> bool:
    # "table" counts as grounded — value came from a pdfplumber table in the paper
    return bool(leaf and leaf.get("grounded") in (True, "table"))


def _precursors_present(extracted: dict) -> bool:
    lst = extracted.get("precursors")
    return isinstance(lst, list) and len(lst) > 0


def _precursors_grounded(extracted: dict) -> bool:
    lst = extracted.get("precursors") or []
    grounded = [p for p in lst if isinstance(p, dict) and p.get("grounded") in (True, "table")]
    return len(grounded) > 0


# ---------------------------------------------------------------------------
# Score components
# ---------------------------------------------------------------------------

def _synthesis_completeness(extracted: dict) -> tuple[float, list[str]]:
    """
    Fraction of critical + important fields that are non-null.
    Returns (score 0–1, list_of_missing_critical_fields).
    """
    fields = {**_CRITICAL, **_IMPORTANT}
    present = 0
    total = len(fields)
    missing_critical = []

    for path in fields:
        if path == "precursors":
            ok = _precursors_present(extracted)
        else:
            ok = _is_present(_get_leaf(extracted, path))

        if ok:
            present += 1
        elif path in _CRITICAL:
            missing_critical.append(path)

    score = round(present / total, 3) if total else 0.0
    return score, missing_critical


def _grounding_precision(extracted: dict) -> float:
    """
    grounded_count / verifiable_non_null_count across all tracked fields.
    Excludes null fields (they don't inflate the denominator).
    Excludes unverifiable nodes (grounded=None — source too short).
    """
    total_verifiable = 0
    total_grounded = 0

    for path in _ALL_FIELDS:
        if path == "precursors":
            lst = extracted.get("precursors") or []
            for p in lst:
                if not isinstance(p, dict):
                    continue
                if p.get("name") or p.get("source"):  # non-null entry
                    total_verifiable += 1
                    if p.get("grounded") is True:
                        total_grounded += 1
        else:
            leaf = _get_leaf(extracted, path)
            if not _is_present(leaf):
                continue
            if leaf.get("grounded") is None:  # unverifiable — exclude
                continue
            total_verifiable += 1
            if _is_grounded(leaf):  # True or "table" both count
                total_grounded += 1

    return round(total_grounded / total_verifiable, 3) if total_verifiable else 0.0


def _field_weighted_score(extracted: dict) -> float:
    """
    Weighted sum of (present AND grounded) / total_weight.
    Rewards fields that are both non-null and verified.
    """
    total_weight = sum(_ALL_FIELDS.values())
    earned = 0.0

    for path, weight in _ALL_FIELDS.items():
        if path == "precursors":
            present = _precursors_present(extracted)
            grounded = _precursors_grounded(extracted)
        else:
            leaf = _get_leaf(extracted, path)
            present = _is_present(leaf)
            grounded = _is_grounded(leaf)

        if present and grounded:
            earned += weight
        elif present:
            earned += weight * 0.5   # partial credit: present but not grounded

    return round(earned / total_weight, 3) if total_weight else 0.0


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def compute_scores(
    extracted: dict,
    is_synthesis_paper: bool = True,
) -> dict:
    """
    Compute all three score dimensions and return a dict ready to merge
    into synthesis["_meta"].

    Parameters
    ----------
    extracted           : synthesis dict after grounding has been run
    is_synthesis_paper  : if False, synthesis_completeness is set to null
                          (reviews and non-synthesis papers shouldn't be penalised)

    Returns
    -------
    {
      "grounding_score":          float,   # backward-compat key
      "grounding_precision":      float,   # same value, explicit name
      "synthesis_completeness":   float | None,
      "field_weighted_score":     float,
      "missing_critical_fields":  list[str],
    }
    """
    precision = _grounding_precision(extracted)
    weighted  = _field_weighted_score(extracted)

    if is_synthesis_paper:
        completeness, missing_critical = _synthesis_completeness(extracted)
    else:
        completeness, missing_critical = None, []

    return {
        "grounding_score":         precision,      # keep old key for backward compat
        "grounding_precision":     precision,
        "synthesis_completeness":  completeness,
        "field_weighted_score":    weighted,
        "missing_critical_fields": missing_critical,
    }
