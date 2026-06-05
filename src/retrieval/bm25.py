"""
BM25 paragraph retrieval for synthesis context.

Replaces the fixed 12k character window sent to the LLM with a ranked
selection of synthesis-relevant paragraphs, assembled in document order.

Why this matters
----------------
Long papers (>50k chars) bury the experimental section well past the 12k
window.  BM25 retrieves synthesis paragraphs regardless of their position.
A paper where the intro runs 15k chars before the experimental section
currently sends the LLM only introduction text; BM25 sends the right section.

Measured impact (Section 11.1 audit)
  temperature missing in 42% of papers → expected drop to ~20% after BM25
  annealing missing in 64%             → expected drop to ~35%

Falls back gracefully to text[:max_chars] when rank-bm25 is not installed.

Reference: Robertson & Zaragoza 2009 — Okapi BM25 (rank-bm25 library)
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

# Broad synthesis query: hits experimental/methods paragraphs reliably.
# Includes both general synthesis terms and perovskite-specific vocabulary.
_SYNTHESIS_QUERY = (
    "temperature anneal precursor concentration spin coat substrate atmosphere "
    "dissolve solution solvothermal deposit fabricate synthesis procedure "
    "perovskite film crystal growth solvent DMF DMSO GBL antisolvent "
    "lead iodide methylammonium cesium bromide chloride spin speed rpm "
    "nitrogen argon glove box humidity stirred heated cooled washed"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_PARA_CHARS  = 80     # paragraphs shorter than this are headers / noise
_MAX_PARA_CHARS  = 3000   # paragraphs longer than this are likely concatenated;
                           # split them so BM25 can score sub-sections
_MAX_OUTPUT_CHARS = 10000  # hard cap on total returned text


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip non-alphanumeric, split on whitespace."""
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[tuple[int, str]]:
    """
    Split text into (original_index, paragraph) pairs.
    Long paragraphs are sub-split on single newlines to give BM25 finer
    granularity without losing document-order information.
    Returns only paragraphs >= _MIN_PARA_CHARS.
    """
    paras: list[tuple[int, str]] = []
    for i, block in enumerate(text.split('\n\n')):
        block = block.strip()
        if not block:
            continue
        if len(block) > _MAX_PARA_CHARS:
            # Sub-split on single newlines
            for sub in block.split('\n'):
                sub = sub.strip()
                if len(sub) >= _MIN_PARA_CHARS:
                    paras.append((i, sub))
        elif len(block) >= _MIN_PARA_CHARS:
            paras.append((i, block))
    return paras


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def bm25_retrieve(
    text: str,
    query: str = _SYNTHESIS_QUERY,
    top_k: int = 14,
    max_chars: int = _MAX_OUTPUT_CHARS,
) -> str:
    """
    Retrieve the top-k synthesis-relevant paragraphs from paper text.

    Paragraphs are ranked by BM25 against the synthesis query and returned
    in original document order (not score order) so the LLM sees coherent
    narrative context.

    Parameters
    ----------
    text      : full paper text (any length)
    query     : space-separated query terms (default: broad synthesis query)
    top_k     : number of paragraphs to retrieve (default 14)
    max_chars : hard cap on returned text length

    Returns
    -------
    str — top-k paragraphs in document order, separated by blank lines.
          Falls back to text[:max_chars] if rank-bm25 is not installed.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return text[:max_chars]

    paras = _split_paragraphs(text)
    if not paras:
        return text[:max_chars]

    corpus = [_tokenize(p) for _, p in paras]
    bm25   = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))

    # Pick top-k by score, then restore document order
    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]
    top_indices_doc_order = sorted(top_indices, key=lambda i: paras[i][0])

    # Assemble within char budget
    parts: list[str] = []
    used = 0
    for idx in top_indices_doc_order:
        para = paras[idx][1]
        if used + len(para) > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                parts.append(para[:remaining])
            break
        parts.append(para)
        used += len(para)

    return "\n\n".join(parts)
