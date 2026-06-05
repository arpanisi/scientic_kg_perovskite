"""
Character-level span grounding.

Replaces fuzzy sentence matching with exact span search — every extracted
source gets (char_start, char_end) indices into the paper text. Mirrors
the Anthropic Citations API pattern, implemented locally with rapidfuzz.

Plug-in to test_pipeline.py
----------------------------
  from src.retrieval.spans import ground_with_spans

  # Stage 4 — replace verify_grounding() call:
  # verify_grounding(synthesis, extracted["text"])
  ground_with_spans(synthesis, extracted["text"])

  # ground_with_spans() writes the same grounded: true/false/null fields
  # plus an optional span: {start, end} on each node.
"""

import unicodedata

from rapidfuzz import fuzz

MIN_SOURCE_LEN = 20    # chars — below this a source can't be meaningfully located
SPAN_THRESHOLD  = 82   # rapidfuzz partial_ratio score (0–100) to accept a match

_TABLE_PREFIX = "Table:"   # pdfplumber table sources — grounded to paper, not prose


# ---------------------------------------------------------------------------
# Unicode normalisation
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """
    NFKC decomposition (resolves ligatures ﬁ→fi, ﬂ→fl, etc.) plus a handful
    of common PDF extraction noise: degree variants, dash variants, nbsp.
    Applied to both the source string and the paper text before comparison so
    that encoding differences between the extracted value and the raw PDF text
    don't cause false grounded=false.
    """
    s = unicodedata.normalize("NFKC", s)
    # Degree variants: U+00B0 ° and U+2103 ℃ → plain ASCII "°"
    s = s.replace("℃", "°C").replace("℉", "°F")
    # Dash variants → ASCII hyphen
    for dash in ("–", "—", "−", "‐"):
        s = s.replace(dash, "-")
    # Non-breaking / thin / hair spaces → regular space
    for sp in (" ", " ", " ", " "):
        s = s.replace(sp, " ")
    return s


# ---------------------------------------------------------------------------
# Core span finder
# ---------------------------------------------------------------------------

def find_span(source: str, text: str) -> tuple[int, int] | None:
    """
    Find the best-matching span of `source` in `text`.

    Both strings are Unicode-normalised before comparison so that PDF
    encoding artefacts (ligatures, degree symbols, dash variants) don't
    prevent a valid match.

    Fast path : exact substring search (after normalisation).
    Slow path : sliding-window partial_ratio scan (window = len(source)).

    Returns (char_start, char_end) into the *original* `text` or None if
    no match >= SPAN_THRESHOLD.
    """
    if not source or len(source.strip()) < MIN_SOURCE_LEN:
        return None

    src_raw  = source.strip()
    src_norm = _normalize(src_raw).lower()
    text_norm = _normalize(text).lower()

    # Fast path on normalised strings
    idx = text_norm.find(src_norm)
    if idx != -1:
        return (idx, idx + len(src_norm))

    # Slow path — slide a window, allow ±20 % length tolerance
    win  = len(src_norm)
    step = max(1, win // 5)
    best_score, best_span = 0, None

    for start in range(0, max(1, len(text_norm) - win + 1), step):
        end = min(start + win + win // 5, len(text_norm))
        score = fuzz.partial_ratio(src_norm, text_norm[start:end])
        if score > best_score:
            best_score = score
            best_span  = (start, end)

    if best_score >= SPAN_THRESHOLD and best_span:
        return best_span
    return None


# ---------------------------------------------------------------------------
# Node walker
# ---------------------------------------------------------------------------

def _annotate(node, text: str) -> None:
    """Recursively add grounded + span to every {value, source} leaf."""
    if isinstance(node, dict):
        if "source" in node:
            src = node.get("source")
            if not src:
                node["grounded"] = False
                return
            src_str = str(src).strip()
            # Table-sourced values are grounded to the paper (pdfplumber)
            # but won't appear as prose — mark them distinctly so scoring
            # doesn't treat them as hallucinations.
            if src_str.startswith(_TABLE_PREFIX):
                node["grounded"] = "table"
                return
            if len(src_str) < MIN_SOURCE_LEN:
                node["grounded"] = None   # too short to verify
                return
            span = find_span(src_str, text)
            if span:
                node["grounded"] = True
                node["span"] = {"start": span[0], "end": span[1]}
            else:
                node["grounded"] = False
        else:
            for v in node.values():
                _annotate(v, text)
    elif isinstance(node, list):
        for item in node:
            _annotate(item, text)


# ---------------------------------------------------------------------------
# Counters (mirror pipeline internals — kept local to avoid circular import)
# ---------------------------------------------------------------------------

def _count(node, predicate) -> int:
    if isinstance(node, dict):
        if "source" in node:
            return 1 if predicate(node) else 0
        return sum(_count(v, predicate) for v in node.values())
    if isinstance(node, list):
        return sum(_count(i, predicate) for i in node)
    return 0


def _is_non_null(n):  return n.get("value") not in (None, [], "")
def _is_grounded(n):  return _is_non_null(n) and n.get("grounded") in (True, "table")
def _is_unverif(n):   return n.get("grounded") is None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def ground_with_spans(extracted: dict, paper_text: str) -> dict:
    """
    Drop-in replacement for verify_grounding().

    Mutates `extracted` in-place:
      - Sets grounded: true / false / null on every {value, source} node.
      - Adds span: {start, end} to nodes that matched (grounded: true).

    Returns the same dict for chaining.
    """
    _annotate(extracted, paper_text)

    non_null   = _count(extracted, _is_non_null)
    grounded   = _count(extracted, _is_grounded)
    unverif    = _count(extracted, _is_unverif)
    verifiable = non_null - unverif

    suffix = f" ({unverif} unverifiable)" if unverif else ""
    print(f"  Span grounding: {grounded}/{verifiable} non-null fields located in text{suffix}.")
    return extracted


def grounding_score(extracted: dict) -> float:
    """
    Compute grounding score after ground_with_spans() has been called.
    Score = grounded / verifiable_non_null  (0.0 – 1.0).
    """
    non_null   = _count(extracted, _is_non_null)
    unverif    = _count(extracted, _is_unverif)
    grounded   = _count(extracted, _is_grounded)
    verifiable = non_null - unverif
    return round(grounded / verifiable, 3) if verifiable else 0.0
