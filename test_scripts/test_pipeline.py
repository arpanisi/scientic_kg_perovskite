"""
End-to-end pipeline test — single PDF:
  1. Extract text + images from PDF (PyMuPDF)
  2. Describe each figure with a VLM
  3. Extract perovskite synthesis data with a text LLM
  4. Save structured output as YAML
"""

import base64
import json
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
CONFIG_PATH = REPO_ROOT / "config" / "llm_config.yaml"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
OUTPUT_DIR = REPO_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def build_llm_list(cfg: dict, model_key: str, fallback_key: str) -> list:
    """Return an ordered list of ChatOpenAI instances from config."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")
    return [
        ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=4096,
        )
        for model in [cfg[model_key]] + cfg.get(fallback_key, [])
    ]


def resolve_llm(cfg: dict, model_key: str, fallback_key: str, required: bool = True):
    """Ping each model in order, return the first one that responds."""
    candidates = build_llm_list(cfg, model_key, fallback_key)

    for llm in candidates:
        try:
            llm.invoke("ping")
            print(f"  [{model_key}] using: {llm.model_name}")
            return llm
        except Exception as e:
            print(f"  [{model_key}] {llm.model_name} unavailable ({type(e).__name__}), trying next...")

    if required:
        raise RuntimeError(f"No available model found for '{model_key}' in llm_config.yaml.")
    print(f"  [{model_key}] no available model found — stage will be skipped.")
    return None


def invoke_with_fallback(candidates: list, prompt: str) -> str:
    """Try each LLM in order, skip on rate-limit or provider errors."""
    for llm in candidates:
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            print(f"  {llm.model_name} failed during invoke ({type(e).__name__}), trying next...")
    raise RuntimeError("All models failed during invoke.")


# ---------------------------------------------------------------------------
# Stage 1 — PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: Path) -> dict:
    """
    Extract text and images from a PDF using PyMuPDF.
    Returns dict with 'text' (str) and 'figures' (list of dicts).
    """
    doc = fitz.open(str(pdf_path))
    pages_text = []
    figures = []

    for page_num, page in enumerate(doc):
        # --- Text: sort blocks top-to-bottom, detect two columns by x midpoint ---
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
        page_width = page.rect.width
        mid_x = page_width / 2

        # Split into left and right column, sort each by y0
        left  = sorted([b for b in text_blocks if b[0] < mid_x], key=lambda b: b[1])
        right = sorted([b for b in text_blocks if b[0] >= mid_x], key=lambda b: b[1])

        # If right column is empty the page is single-column; just use left order
        if not right:
            ordered = left
        else:
            ordered = left + right

        page_text = "\n".join(b[4].strip() for b in ordered)
        pages_text.append(f"[Page {page_num + 1}]\n{page_text}")

        # --- Images ---
        for img_ref in page.get_images(full=True):
            xref = img_ref[0]
            try:
                img_data = doc.extract_image(xref)
                img_bytes = img_data["image"]
                img_ext = img_data["ext"]
                img_w = img_data.get("width", 0)
                img_h = img_data.get("height", 0)

                # Skip publisher icons, logos, and UI elements — scientific
                # figures are at least 200×200 px; icons are typically <100 px
                if img_w < 200 or img_h < 200:
                    continue

                # Find caption: collect all text blocks below the image within
                # 150px, prefer any that starts with "FIG." over plain body text
                bbox = page.get_image_bbox(img_ref)
                caption = ""
                candidates = [
                    b for b in text_blocks
                    if b[1] > bbox.y1 and b[1] < bbox.y1 + 150
                    and b[0] < bbox.x1 and b[2] > bbox.x0
                ]
                for b in sorted(candidates, key=lambda b: b[1]):
                    if b[4].strip().upper().startswith("FIG"):
                        caption = b[4].strip()
                        break
                if not caption:
                    # fall back to first candidate if none starts with FIG.
                    caption = candidates[0][4].strip() if candidates else ""

                # Skip images with no FIG. caption (journal covers, headers, etc.)
                if not caption.strip().upper().startswith("FIG"):
                    continue

                figures.append({
                    "page": page_num + 1,
                    "ext": img_ext,
                    "bytes": img_bytes,
                    "caption": caption,
                })
            except Exception:
                continue

    doc.close()
    return {
        "text": "\n\n".join(pages_text),
        "figures": figures,
    }


# ---------------------------------------------------------------------------
# Stage 2 — Figure description
# ---------------------------------------------------------------------------

FIGURE_PROMPT = """\
You are analysing a figure extracted from a scientific paper about materials science.
Describe what this figure shows in precise scientific terms.
Focus on: type of measurement or diagram, key features, any values or trends visible.
Caption provided: {caption}
Be concise — 2-4 sentences."""


def describe_figures(vlm: ChatOpenAI, figures: list) -> list:
    """Send each figure to the VLM and return descriptions."""
    described = []
    for i, fig in enumerate(figures):
        b64 = base64.b64encode(fig["bytes"]).decode()
        mime = f"image/{fig['ext']}" if fig["ext"] != "jpg" else "image/jpeg"
        prompt = FIGURE_PROMPT.format(caption=fig["caption"] or "No caption available.")

        try:
            msg = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ])
            response = vlm.invoke([msg])
            description = response.content.strip()
        except Exception as e:
            description = f"[Description unavailable: {type(e).__name__}]"

        print(f"  Figure {i+1} (page {fig['page']}): described.")
        described.append({
            "figure_number": i + 1,
            "page": fig["page"],
            "caption": fig["caption"],
            "description": description,
        })

    return described


# ---------------------------------------------------------------------------
# Stage 3 — Synthesis extraction
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """\
You are extracting perovskite synthesis information from a scientific paper.

PAPER TEXT:
{text}

FIGURE DESCRIPTIONS:
{figures}

Extract the following fields. Use null for any field not found in the paper.
Return ONLY a valid JSON object — no markdown, no commentary.

{{
  "title": "...",
  "authors": ["..."],
  "doi": "...",
  "is_perovskite_synthesis": true or false,
  "material": "...",
  "precursors": [
    {{"name": "...", "concentration": "...", "solvent": "..."}}
  ],
  "synthesis_method": "...",
  "process_conditions": {{
    "temperature": "...",
    "duration": "...",
    "atmosphere": "...",
    "annealing": "..."
  }},
  "substrate": "...",
  "characterization": {{
    "xrd": "...",
    "sem": "...",
    "bandgap": "...",
    "film_thickness": "..."
  }},
  "device_performance": {{
    "pce": "...",
    "voc": "...",
    "jsc": "...",
    "ff": "..."
  }},
  "key_findings": "..."
}}"""


def extract_synthesis(llm_candidates: list, text: str, figure_descriptions: list) -> dict:
    figures_text = "\n".join(
        f"Figure {f['figure_number']} (page {f['page']}): {f['description']}"
        for f in figure_descriptions
    ) or "No figures extracted."

    prompt = SYNTHESIS_PROMPT.format(
        text=text[:12000],  # guard against context limits
        figures=figures_text,
    )

    raw = invoke_with_fallback(llm_candidates, prompt)

    # Try to pull out the first {...} JSON object regardless of surrounding text
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"raw_output": raw}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    print("Resolving models...")
    llm_candidates = build_llm_list(cfg, "model", "fallback_models")
    vlm_candidates = build_llm_list(cfg, "vlm_model", "vlm_fallback_models")

    llm = resolve_llm(cfg, model_key="model", fallback_key="fallback_models", required=True)
    vlm = resolve_llm(cfg, model_key="vlm_model", fallback_key="vlm_fallback_models", required=False)

    # Pick first PDF in corpus
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {CORPUS_DIR}")
        return
    pdf_path = pdfs[0]
    print(f"\nProcessing: {pdf_path.name}")

    # Stage 1 — extract
    print("\n[Stage 1] Extracting text and images...")
    extracted = extract_pdf(pdf_path)
    print(f"  {len(extracted['figures'])} figure(s) found.")

    # Stage 2 — describe figures (skipped if no VLM available)
    print("\n[Stage 2] Describing figures...")
    if vlm is None:
        print("  Skipped — no VLM available. Update vlm_model in llm_config.yaml.")
        figure_descriptions = []
    else:
        figure_descriptions = describe_figures(vlm, extracted["figures"])

    # Stage 3 — extract synthesis
    print("\n[Stage 3] Extracting synthesis data...")
    synthesis = extract_synthesis(llm_candidates, extracted["text"], figure_descriptions)

    # Attach figure descriptions to output
    synthesis["figures"] = figure_descriptions

    # Save as YAML
    out_path = OUTPUT_DIR / (pdf_path.stem + ".yaml")
    with open(out_path, "w") as f:
        yaml.dump(synthesis, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"\nOutput saved to: {out_path}")
    print(yaml.dump(synthesis, allow_unicode=True, sort_keys=False, default_flow_style=False))


if __name__ == "__main__":
    main()
