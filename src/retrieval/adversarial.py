"""
Adversarial verifier — second LLM pass for extraction quality.

An Extractor LLM produces the structured synthesis record (existing pipeline).
This module provides the Verifier LLM that:
  1. Checks whether extracted values came from the experimental section
     or were hallucinated from the introduction / discussion.
  2. Identifies fields that should be present but are null (omissions).
  3. Flags product-as-precursor errors and characterisation-as-synthesis errors.
  4. Returns a corrected extraction and a structured critique.

Reference: "Dual-LLM Adversarial Framework for Information Extraction from
Research Literature" — bioRxiv 2025.09.11.675507

Plug-in to test_pipeline.py
----------------------------
  from src.retrieval.adversarial import adversarial_verify

  # After Stage 4 grounding, before saving:
  synthesis, critique = adversarial_verify(synthesis_candidates, synthesis, extracted["text"])
  synthesis["_meta"]["adversarial_critique"] = critique
"""

import json
import re


# ---------------------------------------------------------------------------
# Verifier prompt
# ---------------------------------------------------------------------------

_VERIFIER_PROMPT = """\
You are a critical reviewer of automated data extraction from scientific papers.

Below is a paper excerpt and the structured synthesis record extracted from it.
Your job is to find errors. Be specific and terse.

=== PAPER TEXT ===
{text}

=== EXTRACTED RECORD ===
{record}

Identify each of the following issues. Return ONLY valid JSON — no markdown.

{{
  "wrong_section_fields": [
    {{
      "field": "<dotted path, e.g. process_conditions.temperature>",
      "extracted_value": "<what was extracted>",
      "issue": "<one sentence: why this value is from the wrong section>",
      "correct_value": "<corrected value if you can infer it, else null>"
    }}
  ],
  "missing_fields": [
    {{
      "field": "<field name>",
      "evidence": "<exact quote from the text that contains this missing value>"
    }}
  ],
  "product_as_precursor": [
    {{
      "name": "<precursor name that is actually the product>",
      "reason": "<one sentence>"
    }}
  ],
  "overall_quality": "<one of: good | acceptable | poor>",
  "summary": "<2 sentences: main issues found>"
}}

Rules:
- wrong_section_fields: flag ONLY when you are certain the source sentence
  is from the introduction, abstract, or discussion — not from the
  experimental/methods section.
- missing_fields: flag ONLY if the text clearly states a value that is null
  in the extracted record.
- product_as_precursor: flag if a compound in the precursors list is the
  synthesised material itself (same formula as the 'material' field).
- If no issues exist for a category, return an empty list [].
- Do not invent issues that are not supported by the text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialise_record(extracted: dict) -> str:
    """Compact text representation of the extraction for the verifier prompt."""
    skip = {"figures", "_meta"}
    lines = []
    for k, v in extracted.items():
        if k in skip:
            continue
        if isinstance(v, dict) and "value" in v:
            lines.append(f"{k}: {v.get('value')} (source: {str(v.get('source',''))[:120]})")
        elif isinstance(v, list) and k == "precursors":
            for p in v:
                lines.append(f"  precursor: {p.get('name')} | conc: {p.get('concentration')} | "
                              f"source: {str(p.get('source',''))[:100]}")
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, dict) and "value" in sub_v:
                    lines.append(f"{k}.{sub_k}: {sub_v.get('value')} "
                                 f"(source: {str(sub_v.get('source',''))[:100]})")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _parse_critique(raw: str) -> dict:
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw).strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {
        "wrong_section_fields": [],
        "missing_fields": [],
        "product_as_precursor": [],
        "overall_quality": "unknown",
        "summary": raw[:300],
    }


def _apply_corrections(extracted: dict, critique: dict) -> dict:
    """
    Apply high-confidence corrections from the critique back into the extraction.
    Only overrides fields where the verifier provided a non-null correct_value.
    """
    for item in critique.get("wrong_section_fields", []):
        field = item.get("field", "")
        correct = item.get("correct_value")
        if not field or correct is None:
            continue
        # Navigate to the leaf and update value; keep original as source_original
        parts = field.split(".")
        cur = extracted
        for p in parts[:-1]:
            cur = cur.get(p, {}) if isinstance(cur, dict) else {}
        leaf_key = parts[-1]
        if isinstance(cur, dict) and isinstance(cur.get(leaf_key), dict):
            cur[leaf_key]["value_original"] = cur[leaf_key].get("value")
            cur[leaf_key]["value"] = correct
            cur[leaf_key]["corrected_by"] = "adversarial_verifier"

    # Flag product-as-precursor entries
    bad_names = {
        p.get("name", "").lower().strip()
        for p in critique.get("product_as_precursor", [])
    }
    for prec in extracted.get("precursors", []):
        if isinstance(prec, dict) and prec.get("name", "").lower().strip() in bad_names:
            prec["flagged_product_as_precursor"] = True

    return extracted


def _invoke(llm_candidates: list, prompt: str) -> str:
    for llm in llm_candidates:
        try:
            response = llm.invoke(prompt).content
            if response and response.strip():
                return response
        except Exception as e:
            print(f"  Verifier {llm.model_name} failed ({type(e).__name__}), trying next...")
    raise RuntimeError("All verifier models failed.")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def adversarial_verify(
    llm_candidates: list,
    extracted: dict,
    paper_text: str,
    text_window: int = 8000,
) -> tuple[dict, dict]:
    """
    Run an adversarial verifier LLM over the extracted synthesis record.

    Parameters
    ----------
    llm_candidates : LLM list from build_llm_list() — same models used for extraction
    extracted      : synthesis dict from extract_synthesis() (after grounding)
    paper_text     : full paper text (used to provide context to verifier)
    text_window    : how many chars of paper text to send (default 8000)

    Returns
    -------
    (corrected_extracted, critique_dict)
      corrected_extracted : extraction with high-confidence corrections applied
      critique_dict       : structured critique with wrong_section_fields,
                            missing_fields, product_as_precursor, summary
    """
    record_str = _serialise_record(extracted)
    try:
        from src.retrieval.bm25 import bm25_retrieve
        verifier_text = bm25_retrieve(paper_text, max_chars=text_window)
    except Exception:
        verifier_text = paper_text[:text_window]
    prompt = _VERIFIER_PROMPT.format(
        text=verifier_text,
        record=record_str,
    )

    print("  [Adversarial] Running verifier LLM...")
    try:
        raw = _invoke(llm_candidates, prompt)
    except RuntimeError:
        print("  [Adversarial] All verifier models failed — skipping.")
        return extracted, {"summary": "Verifier unavailable.", "overall_quality": "unknown"}

    critique = _parse_critique(raw)
    quality = critique.get("overall_quality", "unknown")
    summary = critique.get("summary", "")
    n_wrong  = len(critique.get("wrong_section_fields", []))
    n_miss   = len(critique.get("missing_fields", []))
    n_prod   = len(critique.get("product_as_precursor", []))
    print(f"  [Adversarial] quality={quality} | wrong_section={n_wrong} "
          f"| missing={n_miss} | product_as_precursor={n_prod}")
    if summary:
        print(f"  [Adversarial] {summary}")

    corrected = _apply_corrections(extracted, critique)
    return corrected, critique
