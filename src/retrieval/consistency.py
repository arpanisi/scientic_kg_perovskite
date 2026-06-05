"""
Self-consistency extraction + PHCS (Paraphrased Hallucination Consistency Score).

Two complementary verification layers:

  1. Temperature-based consistency (Wang et al. 2023)
     Run extraction N times: run 1 at temp=0 (deterministic), runs 2..N at
     temp=0.3.  Majority vote per field.  Catches single-run greedy-decode
     errors.

  2. PHCS — Paraphrased Hallucination Consistency Score (HalluMat, 2025)
     For each high-risk field (temperature, precursors, annealing, …), ask
     2 paraphrased probe questions and check whether the LLM's short answer
     matches the value already extracted.  If the answers are inconsistent
     with the extraction, the field is flagged phcs_stable=False.
     Logic: a value grounded in the experimental section is stable across
     phrasings; a hallucinated value (e.g. from introduction or
     characterisation) is not, because the LLM "finds" a different passage
     depending on how the question is phrased.

References
----------
  Wang et al. 2023 — arXiv:2203.11171  (self-consistency)
  HalluMat       — arXiv:2512.22396   (PHCS)

Plug-in to test_pipeline.py
----------------------------
  from src.retrieval.consistency import consistent_extraction

  synthesis, model = consistent_extraction(
      extract_synthesis,
      synthesis_candidates,
      extracted["text"],
      figure_descriptions,
      tables=extracted.get("tables", []),
      n=2,          # consistency runs
      run_phcs=True,
  )
"""

from __future__ import annotations

import copy
from collections import Counter


# ---------------------------------------------------------------------------
# Majority-vote configuration
# ---------------------------------------------------------------------------

_VOTE_PATHS = [
    "material",
    "synthesis_method",
    "substrate",
    "key_findings",
    "process_conditions.temperature",
    "process_conditions.duration",
    "process_conditions.atmosphere",
    "process_conditions.annealing",
    "characterization.xrd",
    "characterization.sem",
    "characterization.bandgap",
    "characterization.film_thickness",
    "device_performance.pce",
    "device_performance.voc",
    "device_performance.jsc",
    "device_performance.ff",
]


# ---------------------------------------------------------------------------
# PHCS configuration
# ---------------------------------------------------------------------------

# Short prompt sent per probe — deliberately minimal to avoid guiding the LLM
_PHCS_PROBE_TEMPLATE = """\
Below is an excerpt from a scientific paper on perovskite synthesis.

TEXT:
{text}

Question: {question}

Answer in ONE short sentence using only information explicitly stated in the \
text above. If the answer is not clearly stated, reply exactly: NOT FOUND"""

# High-risk fields and their paraphrased probe questions.
# Two probes per field: if both agree with the extracted value → stable.
# If either disagrees → unstable (likely wrong-section hallucination).
_PHCS_PROBES: dict[str, list[str]] = {
    "process_conditions.temperature": [
        "What temperature was used during the perovskite synthesis or film annealing? "
        "Give only the numeric value and unit.",
        "At what temperature was the perovskite film or material heat-treated after "
        "deposition in the experimental procedure?",
    ],
    "process_conditions.annealing": [
        "What were the annealing conditions (temperature and duration) applied to the "
        "perovskite film after deposition?",
        "Describe the post-deposition thermal treatment: what temperature and for how long?",
    ],
    "process_conditions.duration": [
        "How long did the main synthesis or annealing step last? Give only the duration.",
        "What processing time was reported for the key thermal or deposition step?",
    ],
    "process_conditions.atmosphere": [
        "What atmosphere or gas environment was used during synthesis "
        "(e.g. nitrogen, argon, air, vacuum)?",
        "Was the synthesis or annealing carried out in an inert atmosphere? If so, which gas?",
    ],
    "material": [
        "What perovskite compound was synthesised in this paper? Give the chemical formula.",
        "What is the chemical formula of the primary material reported in this study?",
    ],
    "precursors": [
        "List the chemical precursors or starting materials used to synthesise the perovskite.",
        "What reagents or salts were used in the synthesis procedure described in the text?",
    ],
}

_PHCS_MATCH_THRESHOLD = 72   # rapidfuzz partial_ratio (0–100)
_PHCS_TEXT_WINDOW     = 6000  # chars sent per probe — shorter than full extraction


# ---------------------------------------------------------------------------
# Helpers for reading / writing dotted paths
# ---------------------------------------------------------------------------

def _get_leaf(d: dict, path: str) -> dict | None:
    """Return the {value, source} leaf dict at a dotted path, or None."""
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur if isinstance(cur, dict) and "value" in cur else None


def _set_leaf(d: dict, path: str, leaf: dict) -> None:
    """Write a leaf dict at a dotted path, creating intermediate dicts."""
    keys = path.split(".")
    cur = d
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = leaf


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------

def _majority_leaf(leaves: list[dict | None]) -> dict | None:
    """
    Given a list of {value, source} dicts from N runs, return the leaf whose
    value appears most often. Non-null values beat nulls; ties go to run 0.
    """
    non_null = [l for l in leaves if l and l.get("value") not in (None, "", [])]
    if not non_null:
        return leaves[0] if leaves else None

    counts: Counter = Counter()
    for leaf in non_null:
        counts[str(leaf["value"]).strip().lower()] += 1

    winner_key, _ = counts.most_common(1)[0]
    for leaf in non_null:
        if str(leaf["value"]).strip().lower() == winner_key:
            return leaf
    return non_null[0]


def _vote_precursors(runs: list[dict]) -> list:
    """Return the run with the most precursor entries — most complete wins."""
    lists = [r.get("precursors") for r in runs if isinstance(r.get("precursors"), list)]
    if not lists:
        return []
    return max(lists, key=len)


# ---------------------------------------------------------------------------
# Temperature adjustment
# ---------------------------------------------------------------------------

def _at_temperature(llm, temperature: float):
    """Return a copy of the LLM instance with a different temperature."""
    try:
        return llm.bind(temperature=temperature)
    except Exception:
        return llm


# ---------------------------------------------------------------------------
# PHCS helpers
# ---------------------------------------------------------------------------

def _invoke_probe(llm_candidates: list, prompt: str) -> str | None:
    """Run a short probe prompt through the LLM list. Returns text or None."""
    for llm in llm_candidates:
        try:
            response = llm.invoke(prompt).content.strip()
            if response:
                return response
        except Exception:
            continue
    return None


def _probe_matches(main_value: str, probe_answer: str) -> bool:
    """
    True if probe_answer is consistent with main_value.
    Consistent means the probe didn't return NOT FOUND and the main value
    string appears (fuzzily) inside the probe answer.
    """
    if not main_value or not probe_answer:
        return False
    if "NOT FOUND" in probe_answer.upper():
        return False
    try:
        from rapidfuzz import fuzz
        return fuzz.partial_ratio(str(main_value).lower(), probe_answer.lower()) >= _PHCS_MATCH_THRESHOLD
    except ImportError:
        return str(main_value).lower() in probe_answer.lower()


# ---------------------------------------------------------------------------
# PHCS public interface
# ---------------------------------------------------------------------------

def phcs_verify(
    extracted: dict,
    llm_candidates: list,
    paper_text: str,
    sections: dict | None = None,
) -> dict:
    """
    Run PHCS verification on high-risk fields in the extracted synthesis dict.

    For each field in _PHCS_PROBES with a non-null extracted value, sends
    2 paraphrased probe questions to the LLM and checks whether the short
    answers match the main extracted value.

    Adds to each probed leaf node:
      phcs_stable : True  — both probes agree with extraction
                    False — at least one probe disagrees (review recommended)
                    None  — field is null or field not in probe list

    Adds extracted["_phcs_summary"]: {field: stability_score (0.0–1.0)}

    Parameters
    ----------
    extracted      : synthesis dict after grounding (modified in place)
    llm_candidates : same LLM list used for extraction
    paper_text     : full paper text; BM25-retrieved synthesis context is used
                     for probes so the probe sees the experimental section
                     regardless of paper length, not just the abstract.

    Returns
    -------
    extracted dict with phcs_stable annotations added
    """
    if sections and sections.get("experimental"):
        abstract = sections.get("abstract", "")[:800]
        expt = sections["experimental"]
        text_excerpt = (abstract + "\n\n" + expt).strip()[:_PHCS_TEXT_WINDOW]
    else:
        try:
            from src.retrieval.bm25 import bm25_retrieve
            text_excerpt = bm25_retrieve(paper_text, max_chars=_PHCS_TEXT_WINDOW)
        except Exception:
            text_excerpt = paper_text[:_PHCS_TEXT_WINDOW]
    summary: dict[str, float] = {}

    for path, questions in _PHCS_PROBES.items():
        # --- get main extracted value ---
        if path == "precursors":
            prec_list = extracted.get("precursors") or []
            if not prec_list:
                continue
            main_value = ", ".join(
                p.get("name", "") for p in prec_list
                if isinstance(p, dict) and p.get("name")
            )
            leaf = None
        else:
            leaf = _get_leaf(extracted, path)
            if not leaf or leaf.get("value") in (None, "", []):
                continue
            main_value = str(leaf["value"])

        if not main_value.strip():
            continue

        # --- run each probe ---
        matches = 0
        total   = 0
        for question in questions:
            prompt = _PHCS_PROBE_TEMPLATE.format(
                text=text_excerpt, question=question
            )
            answer = _invoke_probe(llm_candidates, prompt)
            if answer is not None:
                total += 1
                if _probe_matches(main_value, answer):
                    matches += 1

        if total == 0:
            continue

        # Require both probes to have answered — a single probe that fails
        # (e.g. model gives different phrasing or one call is rate-limited)
        # is not enough evidence to flag hallucination.  With total==1 we
        # record the stability score but leave phcs_stable=None (unverified).
        stability  = round(matches / total, 2)
        n_expected = len(questions)
        is_stable  = stability >= 0.5 if total >= n_expected else None
        summary[path] = stability

        # write phcs_stable back to the leaf (or precursor entries)
        if leaf is not None:
            leaf["phcs_stable"] = is_stable
        else:
            for p in extracted.get("precursors", []):
                if isinstance(p, dict):
                    p["phcs_stable"] = is_stable

        if is_stable is None:
            verdict = "UNVERIFIED (only 1/2 probes answered)"
            flag    = ""
        elif is_stable:
            verdict, flag = "stable", ""
        else:
            verdict, flag = "UNSTABLE", "  ← UNSTABLE"
        print(f"  [PHCS] {path}: {matches}/{total} probes match → "
              f"{verdict} ({stability:.2f}){flag}")

    n_unstable = sum(1 for s in summary.values() if s < 0.5)
    n_probed   = len(summary)
    print(f"  [PHCS] {n_probed} fields probed, {n_unstable} unstable.")
    extracted["_phcs_summary"] = summary
    return extracted


# ---------------------------------------------------------------------------
# Public interface — consistent_extraction
# ---------------------------------------------------------------------------

def consistent_extraction(
    extractor,
    llm_candidates: list,
    text: str,
    figure_descriptions: list,
    tables: list | None = None,
    sections: dict | None = None,
    n: int = 2,
    temperature: float = 0.3,
    run_phcs: bool = True,
    is_si: bool = False,
) -> tuple[dict, str]:
    """
    Drop-in replacement for extract_synthesis() with two verification layers.

    Layer 1 — temperature-based consistency (majority vote over N runs).
    Layer 2 — PHCS: paraphrased probe questions on high-risk fields.

    Parameters
    ----------
    extractor         : the extract_synthesis callable from test_pipeline
    llm_candidates    : same list passed to extract_synthesis
    text              : full paper text (used for grounding; BM25 fallback)
    figure_descriptions : same list passed to extract_synthesis
    tables            : pdfplumber table dicts (forwarded to extractor)
    sections          : GROBID section dict forwarded to extractor and PHCS;
                        when experimental section is present, it replaces BM25
    n                 : number of extraction runs (1 = single run, no voting)
    temperature       : temperature for runs 2..n (run 1 is always temp=0)
    run_phcs          : if True, run PHCS after majority vote
    is_si             : forwarded to extractor

    Returns
    -------
    (merged_dict, model_name) — same shape as extract_synthesis()
    merged_dict has phcs_stable on probed fields and _phcs_summary in root.
    """
    results: list[dict] = []
    model_used = "unknown"

    for i in range(n):
        t = 0.0 if i == 0 else temperature
        candidates = (
            llm_candidates if t == 0.0
            else [_at_temperature(llm, t) for llm in llm_candidates]
        )
        try:
            result, m = extractor(
                candidates, text, figure_descriptions,
                tables=tables, is_si=is_si, sections=sections,
            )
            results.append(result)
            if i == 0:
                model_used = m
            print(f"  Consistency run {i+1}/{n} completed ({m}).")
        except Exception as e:
            print(f"  Consistency run {i+1}/{n} failed: {e}")

    if not results:
        return {"error": "all_consistency_runs_failed"}, model_used

    if len(results) == 1:
        merged = results[0]
        print("  Consistency: only 1 run succeeded — no voting applied.")
    else:
        merged = copy.deepcopy(results[0])
        for path in _VOTE_PATHS:
            leaves = [_get_leaf(r, path) for r in results]
            winner = _majority_leaf(leaves)
            if winner is not None:
                _set_leaf(merged, path, copy.deepcopy(winner))
        merged["precursors"] = _vote_precursors(results)
        print(f"  Consistency: {len(results)}/{n} runs, majority vote over "
              f"{len(_VOTE_PATHS)} fields.")

    if run_phcs:
        print("  [PHCS] Running paraphrased probe verification...")
        merged = phcs_verify(merged, llm_candidates, text, sections=sections)

    return merged, model_used
