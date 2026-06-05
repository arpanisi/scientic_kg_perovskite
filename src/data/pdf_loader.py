"""
PDF and config loading for the perovskite KG pipeline.

Text extraction strategy:
  Primary  — GROBID (public API on HuggingFace Spaces): ML-based scientific document
             segmentation. Returns section-segmented text with explicit experimental /
             introduction / results boundaries and structured metadata. No regex guessing.
  Fallback — PyMuPDF: raw text extraction if GROBID is unreachable, times out, or
             returns a malformed response. Behaviour identical to the original pipeline.

Figure extraction strategy (always PyMuPDF — GROBID does not return image bytes):
  Phase 1 — embedded bitmaps: get_images() + caption matching (fast, exact).
  Phase 2 — page rendering fallback: render the page region above any FIG. caption not
             covered by Phase 1 as PNG. Catches vector figures (Nature, Joule, ACS).

Table extraction strategy (pdfplumber — soft dependency):
  pdfplumber extracts table structure (rows, columns, headers) from PDF vector data.
  If pdfplumber is not installed, tables is returned as [].
  Tables are exposed alongside text and figures so downstream extraction can query them
  directly rather than relying on garbled cell fragments in the flat text stream.

Return contract (extract_pdf):
  {
    "text":     str,          # full paper text (GROBID body or PyMuPDF concatenation)
    "figures":  [...],        # deduplicated figures with image bytes (always PyMuPDF)
    "tables":   [...],        # structured tables from pdfplumber ([] if unavailable)
                              # each: {page, table_index, headers, rows, text}
    "sections": {             # only populated when GROBID succeeds; empty dict otherwise
        "abstract":     str,
        "introduction": str,
        "experimental": str,  # primary target — sent to LLM instead of regex window
        "results":      str,
        "conclusion":   str,
        "other":        str,  # all remaining sections concatenated
        "figure_captions": [str],  # structured captions from GROBID TEI
    },
    "metadata": {             # from GROBID teiHeader; empty dict if fallback
        "title":    str,
        "doi":      str,
        "abstract": str,
    },
    "parser": "grobid" | "pymupdf",
  }

Environment variables:
  GROBID_URL      override API base URL (default: HuggingFace Spaces public instance)
  GROBID_ENABLED  set to "0" or "false" to force PyMuPDF fallback without trying GROBID
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz   # PyMuPDF
import yaml

# ---------------------------------------------------------------------------
# GROBID configuration
# ---------------------------------------------------------------------------

GROBID_URL     = os.getenv("GROBID_URL", "https://kermitt2-grobid.hf.space")
GROBID_TIMEOUT = int(os.getenv("GROBID_TIMEOUT", "60"))   # seconds
_GROBID_ENABLED_ENV = os.getenv("GROBID_ENABLED", "1").strip().lower()
GROBID_ENABLED = _GROBID_ENABLED_ENV not in ("0", "false", "no", "off")

_TEI_NS = "http://www.tei-c.org/ns/1.0"
_T = f"{{{_TEI_NS}}}"   # shorthand: _T + "div" == "{http://www.tei-c.org/ns/1.0}div"

# Section head classifiers — matched against <head> text from GROBID
_EXPT_HEAD_RE = re.compile(
    r'\b(experimental|materials\s+and\s+methods|synthesis|sample\s+prep'
    r'|device\s+fab|fabrication|preparation|procedure)\b',
    re.IGNORECASE,
)
_RESULTS_HEAD_RE = re.compile(
    r'\b(results?|characterization|performance|optical\s+propert|discussion)\b',
    re.IGNORECASE,
)
_INTRO_HEAD_RE = re.compile(
    r'\b(introduction|background|motivation|overview)\b',
    re.IGNORECASE,
)
_CONCL_HEAD_RE = re.compile(
    r'\b(conclu\w*|summary|outlook|perspective)',   # prefix match: conclusion/conclusions/concluding
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Figure caption regex (PyMuPDF path — unchanged)
# ---------------------------------------------------------------------------

_FIG_RE = re.compile(
    r"^\s*fig(?:ure)?\.?\s*\d+[a-zA-Z]?\s*(?:[\.:\|–—\n]|\s{2,}|$)",
    re.IGNORECASE,
)
_MIN_RENDER_HEIGHT = 50
_RENDER_SCALE      = 2.0
_MIN_BITMAP_W      = 200
_MIN_BITMAP_H      = 200
_CAPTION_SEARCH_PX = 150


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# GROBID — TEI XML helpers
# ---------------------------------------------------------------------------

def _elem_text(el: ET.Element) -> str:
    """Collect all text within an element, collapsing whitespace."""
    return " ".join(el.itertext()).strip()


def _classify_head(head_text: str) -> str:
    """Map a section <head> string to one of our canonical section keys."""
    if _EXPT_HEAD_RE.search(head_text):
        return "experimental"
    if _RESULTS_HEAD_RE.search(head_text):
        return "results"
    if _INTRO_HEAD_RE.search(head_text):
        return "introduction"
    if _CONCL_HEAD_RE.search(head_text):
        return "conclusion"
    return "other"


def _parse_tei(tei_xml: str) -> dict:
    """
    Parse GROBID TEI XML into a sections dict and metadata dict.

    Returns:
        {
          "sections": {abstract, introduction, experimental, results,
                       conclusion, other, figure_captions},
          "metadata": {title, doi, abstract},
        }
    Raises ValueError on malformed XML so callers can fall back gracefully.
    """
    root = ET.fromstring(tei_xml.encode("utf-8"))

    sections: dict[str, object] = {
        "abstract":       "",
        "introduction":   "",
        "experimental":   "",
        "results":        "",
        "conclusion":     "",
        "other":          "",
        "figure_captions": [],
    }
    metadata: dict[str, str] = {"title": "", "doi": "", "abstract": ""}

    # ---- teiHeader: title, DOI, abstract -----------------------------------
    header = root.find(f".//{_T}teiHeader")
    if header is not None:
        title_el = header.find(f".//{_T}title[@level='a']")
        if title_el is not None:
            metadata["title"] = _elem_text(title_el)

        doi_el = header.find(f".//{_T}idno[@type='DOI']")
        if doi_el is not None and doi_el.text:
            metadata["doi"] = doi_el.text.strip()

        abstract_el = header.find(f".//{_T}abstract")
        if abstract_el is not None:
            abstract_text = _elem_text(abstract_el)
            sections["abstract"] = abstract_text
            metadata["abstract"] = abstract_text

    # ---- body: section divs ------------------------------------------------
    body = root.find(f".//{_T}body")
    if body is not None:
        other_parts: list[str] = []
        for div in body.findall(f"{_T}div"):
            head_el  = div.find(f"{_T}head")
            head_str = _elem_text(head_el) if head_el is not None else ""

            # Collect paragraph text from this div (skip nested sub-divs here;
            # they will appear as their own top-level divs in GROBID output)
            paras = [
                _elem_text(p)
                for p in div.findall(f"{_T}p")
                if _elem_text(p)
            ]
            div_text = "\n\n".join(paras).strip()
            if not div_text:
                continue

            key = _classify_head(head_str)
            if key == "other":
                label = f"[{head_str}]\n" if head_str else ""
                other_parts.append(label + div_text)
            else:
                existing = sections[key]
                sections[key] = (existing + "\n\n" + div_text).strip() if existing else div_text

        sections["other"] = "\n\n".join(other_parts)

    # ---- figures: structured captions --------------------------------------
    captions: list[str] = []
    for fig in root.iter(f"{_T}figure"):
        head_el = fig.find(f"{_T}head")
        desc_el = fig.find(f"{_T}figDesc")
        head_str = _elem_text(head_el) if head_el is not None else ""
        desc_str = _elem_text(desc_el) if desc_el is not None else ""
        caption  = f"{head_str} {desc_str}".strip()
        if caption:
            captions.append(caption)
    sections["figure_captions"] = captions

    return {"sections": sections, "metadata": metadata}


def _assemble_full_text(sections: dict) -> str:
    """
    Build a single text string from GROBID sections for backward-compat callers
    that expect extracted["text"] (e.g. grounding verification, keyword search).
    Order: abstract → introduction → experimental → results → conclusion → other.
    """
    order = ["abstract", "introduction", "experimental", "results", "conclusion", "other"]
    parts = [sections[k] for k in order if isinstance(sections.get(k), str) and sections[k].strip()]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# GROBID — API call
# ---------------------------------------------------------------------------

def _grobid_extract(pdf_path: Path) -> dict | None:
    """
    POST the PDF to the GROBID processFulltextDocument endpoint.
    Returns parsed TEI dict on success, None on any failure.

    consolidateHeader=1 : GROBID cross-references CrossRef/PubMed for DOI/authors.
    consolidateCitations=0 : skip reference resolution (slow, not needed here).
    """
    try:
        import requests  # soft dependency — already present via langchain
    except ImportError:
        print("  [GROBID] requests not installed — falling back to PyMuPDF.")
        return None

    url = f"{GROBID_URL}/api/processFulltextDocument"
    try:
        with open(pdf_path, "rb") as fh:
            resp = requests.post(
                url,
                files={"input": (pdf_path.name, fh, "application/pdf")},
                data={"consolidateHeader": "1", "consolidateCitations": "0",
                      "includeRawAffiliations": "0"},
                timeout=GROBID_TIMEOUT,
            )
    except Exception as e:
        print(f"  [GROBID] Request failed ({type(e).__name__}: {e}) — falling back to PyMuPDF.")
        return None

    if resp.status_code == 503:
        print("  [GROBID] Server busy (503) — falling back to PyMuPDF.")
        return None
    # 200 = full success; 206 = partial content (some elements unparseable, TEI still returned)
    if resp.status_code not in (200, 206):
        print(f"  [GROBID] Unexpected status {resp.status_code} — falling back to PyMuPDF.")
        return None
    # Guard: HuggingFace Spaces returns an HTML loading page when the Space is sleeping.
    # TEI XML always starts with the <TEI root element (after optional <?xml?> declaration).
    body = resp.text.strip()
    if not (body.startswith("<?xml") or "<TEI " in body[:200]):
        print("  [GROBID] Response is not TEI XML (Space may be sleeping) — falling back to PyMuPDF.")
        return None

    try:
        return _parse_tei(resp.text)
    except Exception as e:
        print(f"  [GROBID] TEI parse error ({type(e).__name__}: {e}) — falling back to PyMuPDF.")
        return None


# ---------------------------------------------------------------------------
# PyMuPDF helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _is_caption(text: str) -> bool:
    return bool(_FIG_RE.match(text))


def _find_caption_below(bbox_y1: float, text_blocks: list, margin: float = _CAPTION_SEARCH_PX) -> str:
    candidates = [b for b in text_blocks if bbox_y1 < b[1] < bbox_y1 + margin]
    for b in sorted(candidates, key=lambda b: b[1]):
        if _is_caption(b[4]):
            return b[4].strip()
    return candidates[0][4].strip() if candidates else ""


def _render_figure_region(page: fitz.Page, cap_block: tuple, text_blocks: list) -> dict | None:
    cap_y0       = cap_block[1]
    blocks_above = [b for b in text_blocks if b[3] < cap_y0 - 5]
    region_top   = max((b[3] for b in blocks_above), default=0) + 2
    if cap_y0 - region_top < _MIN_RENDER_HEIGHT:
        return None
    clip = fitz.Rect(0, region_top, page.rect.width, cap_y0)
    mat  = fitz.Matrix(_RENDER_SCALE, _RENDER_SCALE)
    try:
        pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
    except Exception:
        return None
    return {"ext": "png", "bytes": pix.tobytes("png"), "size": pix.width * pix.height}


def _pymupdf_extract_text_and_figures(pdf_path: Path) -> tuple[str, list]:
    """
    Full PyMuPDF extraction — text (layout-ordered) and figures.
    Used as both the fallback text source and the always-on figure source.
    Returns (full_text, figures_list).
    """
    doc        = fitz.open(str(pdf_path))
    pages_text = []
    figures    = []

    for page_num, page in enumerate(doc):
        blocks     = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
        page_width = page.rect.width
        mid_x      = page_width / 2

        left  = sorted([b for b in text_blocks if b[0] < mid_x],  key=lambda b: b[1])
        right = sorted([b for b in text_blocks if b[0] >= mid_x], key=lambda b: b[1])
        ordered = left if not right else left + right
        pages_text.append(
            f"[Page {page_num + 1}]\n" + "\n".join(b[4].strip() for b in ordered)
        )

        # Phase 1 — embedded bitmaps
        covered_captions: set[str] = set()
        for img_ref in page.get_images(full=True):
            xref = img_ref[0]
            try:
                img_data = doc.extract_image(xref)
                img_w, img_h = img_data.get("width", 0), img_data.get("height", 0)
                if img_w < _MIN_BITMAP_W or img_h < _MIN_BITMAP_H:
                    continue
                bbox    = page.get_image_bbox(img_ref)
                caption = _find_caption_below(bbox.y1, text_blocks)
                if not _is_caption(caption):
                    continue
                covered_captions.add(caption)
                figures.append({
                    "page": page_num + 1, "ext": img_data["ext"],
                    "bytes": img_data["image"], "size": img_w * img_h,
                    "caption": caption, "source": "bitmap",
                })
            except Exception:
                continue

        # Phase 2 — render vector figures for uncovered captions
        for cap_block in [b for b in text_blocks if _is_caption(b[4])]:
            caption = cap_block[4].strip()
            if caption in covered_captions:
                continue
            rendered = _render_figure_region(page, cap_block, text_blocks)
            if rendered is None:
                continue
            figures.append({
                "page": page_num + 1, "ext": rendered["ext"],
                "bytes": rendered["bytes"], "size": rendered["size"],
                "caption": caption, "source": "rendered",
            })

    doc.close()

    # Deduplicate — keep largest image per caption
    seen: dict = {}
    for fig in figures:
        cap = fig["caption"]
        if cap not in seen or fig["size"] > seen[cap]["size"]:
            seen[cap] = fig

    deduped = sorted(seen.values(), key=lambda f: (f["page"], f["caption"]))
    for i, fig in enumerate(deduped, start=1):
        fig["figure_number"] = i

    return "\n\n".join(pages_text), deduped


# ---------------------------------------------------------------------------
# Table extraction via pdfplumber (soft dependency)
# ---------------------------------------------------------------------------

def _pdfplumber_extract_tables(pdf_path: Path) -> list[dict]:
    """
    Extract tables from PDF using pdfplumber.
    Returns [] if pdfplumber is not installed or no tables are found.
    Each entry: {page, table_index, headers, rows, text}
      text — pipe-delimited string suitable for including in an LLM prompt.
    """
    try:
        import pdfplumber
    except ImportError:
        return []

    tables = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for t_idx, table in enumerate(page.extract_tables() or []):
                    if not table or len(table) < 2:
                        continue
                    # Skip near-empty tables
                    if not any(any(cell for cell in row) for row in table):
                        continue
                    headers = [str(c or "").strip() for c in table[0]]
                    rows    = [[str(c or "").strip() for c in row] for row in table[1:]]
                    sep = " | "
                    lines = [sep.join(headers), "-" * max(len(sep.join(headers)), 20)]
                    lines += [sep.join(row) for row in rows]
                    tables.append({
                        "page":        page_num,
                        "table_index": t_idx,
                        "headers":     headers,
                        "rows":        rows,
                        "text":        "\n".join(lines),
                    })
    except Exception as e:
        print(f"  [pdfplumber] Table extraction failed ({type(e).__name__}: {e}).")
    return tables


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: Path) -> dict:
    """
    Extract text, sections, metadata, and figures from a PDF.

    Text/section extraction:
      Tries GROBID first (unless GROBID_ENABLED=0).
      Falls back to PyMuPDF raw extraction on any failure.

    Figure extraction:
      Always uses PyMuPDF (GROBID does not return image bytes).

    If GROBID captions are available and a PyMuPDF figure has no good caption,
    the GROBID caption for that figure number is used as enrichment.

    Returns:
        {
          "text":     str,    full paper text for grounding / keyword search
          "figures":  list,   deduplicated figures with image bytes
          "sections": dict,   section-segmented text (empty if GROBID unavailable)
          "metadata": dict,   structured metadata (empty if GROBID unavailable)
          "parser":   str,    "grobid" | "pymupdf"
        }
    """
    # Always extract figures (and fallback text) with PyMuPDF
    pymupdf_text, figures = _pymupdf_extract_text_and_figures(pdf_path)

    # Always extract tables with pdfplumber (soft dependency — [] if not installed)
    tables = _pdfplumber_extract_tables(pdf_path)
    if tables:
        print(f"  [pdfplumber] {len(tables)} table(s) extracted.")

    # Try GROBID for text segmentation
    grobid_result = None
    if GROBID_ENABLED:
        print("  [GROBID] Sending to GROBID for section segmentation...")
        grobid_result = _grobid_extract(pdf_path)

    if grobid_result:
        sections = grobid_result["sections"]
        metadata = grobid_result["metadata"]
        full_text = _assemble_full_text(sections)

        # Enrich PyMuPDF figure captions with GROBID captions where available
        grobid_caps = sections.get("figure_captions", [])
        if grobid_caps:
            _enrich_figure_captions(figures, grobid_caps)

        print(f"  [GROBID] OK — experimental section: "
              f"{len(sections.get('experimental',''))} chars | "
              f"{len(figures)} figure(s)")
        return {
            "text":     full_text,
            "figures":  figures,
            "tables":   tables,
            "sections": sections,
            "metadata": metadata,
            "parser":   "grobid",
        }

    # Fallback
    print("  [GROBID] Using PyMuPDF fallback.")
    return {
        "text":     pymupdf_text,
        "figures":  figures,
        "tables":   tables,
        "sections": {},
        "metadata": {},
        "parser":   "pymupdf",
    }


# ---------------------------------------------------------------------------
# Caption enrichment helper
# ---------------------------------------------------------------------------

def _enrich_figure_captions(figures: list, grobid_captions: list[str]) -> None:
    """
    In-place: if a PyMuPDF figure has a short/missing caption and a GROBID caption
    for the same figure number exists, replace with the GROBID caption (richer text).
    """
    # Build fig_number → grobid caption index
    grobid_cap_map: dict[int, str] = {}
    fig_num_re = re.compile(r'fig(?:ure)?\.?\s*(\d+)', re.IGNORECASE)
    for cap in grobid_captions:
        m = fig_num_re.search(cap)
        if m:
            grobid_cap_map[int(m.group(1))] = cap

    for fig in figures:
        fn = fig.get("figure_number")
        if fn and fn in grobid_cap_map:
            existing = fig.get("caption", "")
            grobid_cap = grobid_cap_map[fn]
            # Replace only if GROBID caption is substantially longer
            if len(grobid_cap) > len(existing) + 20:
                fig["caption"] = grobid_cap
                fig["caption_source"] = "grobid"
