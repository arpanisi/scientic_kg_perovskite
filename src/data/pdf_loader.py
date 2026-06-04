"""
PDF and config loading for the perovskite KG pipeline.

Handles reading a PDF from disk (text + images) and loading the YAML config.
No LLM calls, no retrieval — pure I/O.

Figure extraction strategy:
  Phase 1 — embedded bitmaps: use get_images() + caption matching (fast, exact).
  Phase 2 — page rendering fallback: for any FIG. caption not matched in Phase 1,
             render the page region above the caption as a PNG. Catches vector figures
             (Nature, Joule, ACS) that get_images() misses entirely.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF
import yaml

# Matches caption formats:
#   "Figure 1." / "Fig. 2:" — punctuation after number (ACS, RSC, Nature)
#   "Fig. 1   a XRD..."    — 2+ spaces after number (Springer)
# Rejects inline refs like "Fig. 1a and 1b show..." (single space + text)
_FIG_RE = re.compile(
    r"^\s*fig(?:ure)?\.?\s*\d+[a-zA-Z]?\s*(?:[\.:\|–—]|\s{2,})",
    re.IGNORECASE,
)
_MIN_RENDER_HEIGHT = 50   # pixels — skip render if region too thin
_RENDER_SCALE = 2.0       # 2× DPI for readable figures
_MIN_BITMAP_W = 200
_MIN_BITMAP_H = 200
_CAPTION_SEARCH_PX = 150  # how far below a bbox to search for a caption


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_caption(text: str) -> bool:
    return bool(_FIG_RE.match(text))


def _find_caption_below(bbox_y1: float, text_blocks: list, margin: float = _CAPTION_SEARCH_PX) -> str:
    """Return the first FIG. caption text block within margin px below bbox_y1."""
    candidates = [
        b for b in text_blocks
        if b[1] > bbox_y1 and b[1] < bbox_y1 + margin
    ]
    for b in sorted(candidates, key=lambda b: b[1]):
        if _is_caption(b[4]):
            return b[4].strip()
    # Fallback: any close block, even if not "FIG."
    return candidates[0][4].strip() if candidates else ""


def _render_figure_region(page: fitz.Page, cap_block: tuple, text_blocks: list) -> dict | None:
    """
    Render the page region between the nearest text above the caption and the caption top.
    Returns figure dict with ext="png" and bytes, or None if region is too small.
    """
    cap_y0 = cap_block[1]
    # Find text blocks strictly above the caption
    blocks_above = [b for b in text_blocks if b[3] < cap_y0 - 5]
    region_top = max((b[3] for b in blocks_above), default=0) + 2
    if cap_y0 - region_top < _MIN_RENDER_HEIGHT:
        return None

    clip = fitz.Rect(0, region_top, page.rect.width, cap_y0)
    mat = fitz.Matrix(_RENDER_SCALE, _RENDER_SCALE)
    try:
        pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
    except Exception:
        return None
    return {
        "ext": "png",
        "bytes": pix.tobytes("png"),
        "size": pix.width * pix.height,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: Path) -> dict:
    """
    Extract text and figures from a PDF.

    Returns:
        {
            "text": str,          # full paper text, layout-ordered
            "figures": [          # deduplicated scientific figures
                {
                    "figure_number": int,
                    "page": int,
                    "ext": str,
                    "bytes": bytes,
                    "size": int,
                    "caption": str,
                    "source": str,   # "bitmap" | "rendered"
                }
            ]
        }
    """
    doc = fitz.open(str(pdf_path))
    pages_text = []
    figures = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
        page_width = page.rect.width
        mid_x = page_width / 2

        left  = sorted([b for b in text_blocks if b[0] < mid_x],  key=lambda b: b[1])
        right = sorted([b for b in text_blocks if b[0] >= mid_x], key=lambda b: b[1])
        ordered = left if not right else left + right
        pages_text.append(f"[Page {page_num + 1}]\n" + "\n".join(b[4].strip() for b in ordered))

        # -- Phase 1: embedded bitmaps ----------------------------------------
        covered_captions: set[str] = set()
        for img_ref in page.get_images(full=True):
            xref = img_ref[0]
            try:
                img_data = doc.extract_image(xref)
                img_w = img_data.get("width", 0)
                img_h = img_data.get("height", 0)
                if img_w < _MIN_BITMAP_W or img_h < _MIN_BITMAP_H:
                    continue

                bbox = page.get_image_bbox(img_ref)
                caption = _find_caption_below(bbox.y1, text_blocks)
                if not _is_caption(caption):
                    continue

                covered_captions.add(caption)
                figures.append({
                    "page": page_num + 1,
                    "ext": img_data["ext"],
                    "bytes": img_data["image"],
                    "size": img_w * img_h,
                    "caption": caption,
                    "source": "bitmap",
                })
            except Exception:
                continue

        # -- Phase 2: render vector figures for uncovered captions -------------
        fig_caption_blocks = [b for b in text_blocks if _is_caption(b[4])]
        for cap_block in fig_caption_blocks:
            caption = cap_block[4].strip()
            if caption in covered_captions:
                continue
            rendered = _render_figure_region(page, cap_block, text_blocks)
            if rendered is None:
                continue
            figures.append({
                "page": page_num + 1,
                "ext": rendered["ext"],
                "bytes": rendered["bytes"],
                "size": rendered["size"],
                "caption": caption,
                "source": "rendered",
            })

    doc.close()

    # Deduplicate multi-panel figures — keep largest image per caption
    seen: dict = {}
    for fig in figures:
        cap = fig["caption"]
        if cap not in seen or fig["size"] > seen[cap]["size"]:
            seen[cap] = fig

    deduped = sorted(seen.values(), key=lambda f: (f["page"], f["caption"]))
    for i, fig in enumerate(deduped, start=1):
        fig["figure_number"] = i

    return {
        "text": "\n\n".join(pages_text),
        "figures": deduped,
    }
