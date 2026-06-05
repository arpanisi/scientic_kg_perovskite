# Perovskite Synthesis Knowledge Graph

A two-pipeline system for perovskite materials science:

1. **Extraction pipeline** — reads scientific PDFs, extracts structured synthesis data into a knowledge graph, verifies every field against the source text.
2. **Training pipeline** — fine-tunes and RLVR-trains language models for performance prediction and constrained recipe generation.

---

## Table of Contents

- [Project Layout](#project-layout)
- [Setup](#setup)
- [Pipeline 1 — Extraction](#pipeline-1--extraction)
  - [Stage 0 — Document Classification](#stage-0--document-classification)
  - [Stage 1 — Text, Table, and Figure Extraction](#stage-1--text-table-and-figure-extraction)
  - [Stage 2a — Relevance Filter](#stage-2a--relevance-filter)
  - [Stage 2 — Figure Description](#stage-2--figure-description)
  - [Stage 3 — Synthesis Extraction](#stage-3--synthesis-extraction)
  - [Stage 4 — Span Grounding](#stage-4--span-grounding)
  - [Stage 4.5 — Adversarial Verification](#stage-45--adversarial-verification)
  - [Supporting Information Merging](#supporting-information-merging)
  - [Output Schema](#output-schema)
  - [On-the-Fly Quality Metrics](#on-the-fly-quality-metrics)
  - [Running the Pipeline](#running-the-pipeline)
- [Pipeline 2 — Training](#pipeline-2--training)
  - [Data Preparation](#data-preparation)
  - [SFT Training](#sft-training)
  - [GRPO / RLVR Training](#grpo--rlvr-training)
- [Configuration](#configuration)

---

## Project Layout

```
.
├── config/
│   ├── llm_config.yaml          # LLM provider, model, fallback chain, VLM
│   ├── fine_tuning.yaml         # Fine-tuning method taxonomy
│   └── weight_update.yaml       # Weight update strategies
├── data/
│   ├── corpus/                  # Input PDFs (98 papers)
│   ├── output/                  # Per-paper output.yaml + figure images
│   ├── processed/               # JSONL train/val/test splits
│   └── raw/                     # Source CSV (Perovskite database)
├── src/
│   ├── data/
│   │   ├── pdf_loader.py        # GROBID + PyMuPDF extraction
│   │   ├── normalize_outputs.py # Strip value/source envelope to flat JSON
│   │   ├── sft_dataset.py       # PyTorch Dataset + collator
│   │   ├── build_datasets.py    # CSV → JSONL splits
│   │   ├── prepare_inputs.py    # Input representation builders
│   │   └── prepare_outputs.py   # Output schema builders
│   ├── retrieval/
│   │   ├── bm25.py              # Synthesis-aware paragraph ranking
│   │   ├── consistency.py       # Self-consistency voting + PHCS
│   │   ├── adversarial.py       # Second-LLM error detection
│   │   ├── score.py             # Grounding and completeness metrics
│   │   └── spans.py             # Character-level span grounding
│   ├── evaluate/
│   │   ├── metrics.py           # MAE, RMSE, R² + JSON validation
│   │   └── generate.py          # Test-set inference
│   ├── model/
│   │   └── setup.py             # Model loading + fine-tuning strategy
│   └── training/
│       ├── trainer.py           # WeightedSFTTrainer
│       └── fine_tuning.py       # 8 fine-tuning method registry
├── test_scripts/
│   ├── test_pipeline.py         # Extraction pipeline entry point (single PDF)
│   └── run_batch.sh             # Batch runner for entire corpus
├── train_llm.py                 # SFT training entry point
├── train_rlvr.py                # GRPO/RLVR training entry point
└── prepare_training_dataset.py  # CLI: build JSONL from CSV
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your OpenRouter key:

```
OPENROUTER_API_KEY=sk-or-...
```

GROBID is used automatically via the public HuggingFace Spaces instance. Set `GROBID_ENABLED=0` to skip it and use PyMuPDF only. Override the endpoint with `GROBID_URL=http://localhost:8070`.

---

## Pipeline 1 — Extraction

Processes one PDF at a time through six stages, writing a single `output.yaml` per paper to `data/output/<stem>/`. Each stage is independently skippable or replaceable.

```
PDF
 │
 ├─ Stage 0:  Document classification (main paper vs. supporting information)
 ├─ Stage 1:  Text + section + table + figure extraction
 ├─ Stage 2a: Relevance filter (is this about perovskite?)
 ├─ Stage 2:  Vision-language figure description
 ├─ Stage 3:  Synthesis extraction — BM25/GROBID context → LLM → JSON
 │            └─ self-consistency voting over N runs
 │            └─ PHCS probe verification on high-risk fields
 ├─ Stage 4:  Span grounding (character-level, mirrors Citations API)
 └─ Stage 4.5: Adversarial verification (second LLM cross-check)
      │
      └─ output.yaml (+ fig_N.png files)
```

### Stage 0 — Document Classification

Classifies the PDF as a main paper or supporting information (SI) before any LLM calls.

**SI signals (in priority order)**:
- Filename contains `_si_`, `_supp_`, `_supplementary_`, etc.
- Opening 600 characters contain known SI header phrases (`"Supporting Information for"`, `"Electronic Supplementary Information"`, `"Supplementary Methods"`, etc.)
- ACS false-positive patterns are stripped before the keyword scan.

If the document is classified as SI, the pipeline attempts to match it to an already-processed main paper by DOI (exact) or title (fuzzy, threshold 80). If a match is found, it is queued for merging. If no match is found yet, it is queued by DOI or title and retried automatically when the corresponding main paper is processed later.

### Stage 1 — Text, Table, and Figure Extraction

Returns a unified dict used by all downstream stages:

```python
{
  "text":     str,       # full paper text (for grounding + fallback context)
  "sections": {          # populated when GROBID succeeds; {} otherwise
      "abstract":        str,
      "introduction":    str,
      "experimental":    str,   # primary synthesis context
      "results":         str,
      "conclusion":      str,
      "other":           str,
      "figure_captions": [str],
  },
  "metadata": {          # from GROBID teiHeader; {} otherwise
      "title": str,
      "doi":   str,
      "abstract": str,
  },
  "figures":  [...],     # deduplicated figures with image bytes
  "tables":   [...],     # structured tables (page, headers, rows, pipe-delimited text)
  "parser":   "grobid" | "pymupdf",
}
```

**Text extraction** — GROBID first, PyMuPDF fallback:

GROBID (`https://kermitt2-grobid.hf.space`) performs ML-based scientific document segmentation. It parses the TEI XML response into named sections, which eliminates the need for regex-based section detection. If GROBID is unavailable, times out (60 s), returns a non-200/206 status, or returns non-TEI content (e.g. a sleeping Spaces loading page), PyMuPDF takes over with layout-ordered column extraction.

**Figure extraction** — always PyMuPDF (two phases):

- *Phase 1*: Embedded bitmaps via `get_images()`. Matched to captions by proximity using a regex that handles `Figure 1.`, `Fig. 2:`, and multi-space variants.
- *Phase 2*: For any caption not covered by Phase 1 (vector graphics common in ACS, Nature, Joule), the page region above the caption is rendered at 2× scale as PNG.

Figures are deduplicated (largest image per caption wins) and numbered in document order. If GROBID returned structured captions that are substantially longer than the PyMuPDF caption for the same figure number, the GROBID caption replaces the PyMuPDF one.

**Table extraction** — pdfplumber (soft dependency):

pdfplumber extracts table structure from PDF vector data and returns each table as `{page, table_index, headers, rows, text}` where `text` is a pipe-delimited string for direct LLM consumption. If pdfplumber is not installed, `tables` is `[]`.

### Stage 2a — Relevance Filter

Two-layer check to skip non-perovskite papers without expensive LLM calls:

1. **Local keyword scan** — checks for 14 terms (`perovskite`, `mapbi`, `fapbi`, `cspbi`, `batio3`, `pzt`, etc.). No LLM needed; fast enough to run on every paper.
2. **LLM confirmation** — only invoked if the keyword scan passes. Sends first 3000 chars with a single-word prompt (`YES` / `NO`). If all models fail, defaults to relevant.

Papers classified as non-perovskite are saved as a minimal record `{filename, is_perovskite_synthesis: false}` and skipped.

### Stage 2 — Figure Description

Each figure is sent to a vision-language model with the figure caption and a short paper context snippet (first ~800 chars). The structured JSON response captures:

```json
{
  "figure_type": "XRD | SEM | TEM | PL | UV-Vis | J-V | EDS | XPS | FTIR | impedance | other",
  "key_values": ["peak 2θ = 28.5°", "efficiency 18.5%", ...],
  "phases_or_materials": ["Cs2TiBr6", "MAPbI3", ...],
  "key_observation": "XRD confirms tetragonal perovskite phase with no residual PbI2."
}
```

The primary VLM is `google/gemma-4-31b-it:free` (OpenRouter). Three fallback VLMs are tried in order on failure. Figure descriptions are formatted into a compact text block and included in the Stage 3 prompt.

### Stage 3 — Synthesis Extraction

The core extraction stage. The LLM receives a focused synthesis context, structured table data, and figure descriptions, then returns a JSON object with `{value, source}` pairs for every field.

**Context selection** (in priority order):

1. If GROBID succeeded and `sections["experimental"]` is non-empty: use `abstract[:1500] + experimental section` directly, capped at 12,000 characters. This sends the exact experimental section without any approximation.
2. If this is an SI document: strip table-of-contents noise first, then apply BM25.
3. Otherwise (no GROBID sections): BM25 paragraph retrieval over the full text.

**BM25 retrieval** selects the top-14 paragraphs by Okapi BM25 score against a fixed synthesis query covering temperature, precursors, solvents, spin coating, annealing, atmosphere, and related perovskite vocabulary. Paragraphs are returned in document order (not score order) so the LLM sees coherent narrative context. Output is capped at 10,000 characters.

**Extracted schema**:

| Field | Description |
|---|---|
| `title`, `authors`, `doi` | Paper metadata |
| `is_perovskite_synthesis` | Boolean classification |
| `material` | Perovskite formula (e.g. CH₃NH₃PbI₃, Cs₂TiBr₆) |
| `precursors` | List of `{name, concentration, solvent, source}` — includes salts, solvents, antisolvents, capping ligands; excludes wash solvents, HTL/ETL, electrode materials |
| `synthesis_method` | One-step spin coating, two-step, vapor deposition, colloidal, etc. |
| `process_conditions` | `temperature`, `duration`, `atmosphere`, `annealing` |
| `substrate` | Support during synthesis (null for colloidal or characterization-only) |
| `characterization` | XRD, SEM, bandgap, film thickness |
| `device_performance` | PCE, Voc, Jsc, FF |
| `key_findings` | One-sentence synthesis summary |

Every scalar field follows `{"value": "...", "source": "exact quote from paper text"}`.

**Self-consistency** (configured via `consistency_runs` in `llm_config.yaml`):

The extraction runs N times. Run 1 uses temperature=0 (deterministic). Runs 2..N use temperature=0.3. A majority vote is applied across 16 key paths. For `precursors`, the run with the most entries wins (completeness heuristic). The default is 2 runs.

**PHCS — Paraphrased Hallucination Consistency Score** (configured via `run_phcs`):

After majority vote, each high-risk field (temperature, annealing, duration, atmosphere, material, precursors) is probed with 2 independently phrased questions sent to the same LLM. The short answers are fuzzy-matched against the extracted value. If at least one probe disagrees, the field is marked `phcs_stable: false` — a signal that the value may have been pulled from the wrong section (e.g. a temperature mentioned in the introduction rather than the experimental procedure). Stability scores are collected in `_phcs_summary`.

When GROBID sections are available, PHCS probes are run against the experimental section directly (not a BM25 window).

**LLM fallback chain**: 16 alternates (Hermes-3 405B, Nemotron 120B, Qwen, Gemma-4, etc.) are tried in sequence on any error or empty response. The primary model is `meta-llama/llama-3.3-70b-instruct:free`.

If GROBID metadata (title, DOI) is available and the LLM returned null for those fields, the GROBID values are filled in automatically with `"source": "grobid_metadata"`.

### Stage 4 — Span Grounding

Every `{value, source}` pair in the extraction is located in the original paper text. This mirrors the Anthropic Citations API pattern implemented locally.

**Algorithm**:
1. Normalize both the source string and the paper text: NFKC decomposition (resolves ligatures ﬁ→fi, ﬂ→fl), degree symbol unification (℃→°C), dash normalization (–, —, − → -), space normalization.
2. Fast path: exact substring search on the normalized strings.
3. Slow path: sliding-window fuzzy match (rapidfuzz `partial_ratio ≥ 82`) over windows sized to the source length.

Each node receives `grounded: true/false/null` and, when grounded, `span: {start, end}` (character indices into the original text). Sources shorter than 20 characters receive `grounded: null` (too short to verify meaningfully). Table-sourced values receive `grounded: "table"`.

### Stage 4.5 — Adversarial Verification

A second LLM pass reviews the extraction against the paper text and flags three error classes:

- **wrong_section_fields** — values that appear to have been pulled from the introduction or discussion rather than the experimental section (e.g. a reference temperature from a comparison rather than the actual synthesis temperature).
- **missing_fields** — fields that are null in the extraction but present in the text, with an exact supporting quote.
- **product_as_precursor** — entries in the `precursors` list that are actually the synthesised material.

Where the verifier provides a `correct_value`, it overwrites the extractor's answer and the original is preserved as `value_original`. Suspected product-as-precursor entries are flagged with `flagged_product_as_precursor: true` rather than silently removed. The critique is stored in `_meta.adversarial_critique`.

### Supporting Information Merging

SI documents are detected in Stage 0 and processed separately. After the main paper output is written, the pipeline checks whether any queued SI matches the paper's DOI. If so, the SI is extracted and merged:

- **Experimental fields** (material, precursors, synthesis_method, process_conditions, substrate, characterization) prefer SI values — SI typically contains more detailed procedures.
- **Metadata fields** (title, doi, authors) prefer the main paper.
- **Precursors** are merged as a union by name; SI entries overwrite main-paper entries on name collision.

Every field is tagged with `source_doc: "main"` or `source_doc: "si"`. Merged papers have `_meta.si_merged: true` and `_meta.si_fields` listing which fields were updated from SI.

### Output Schema

Each paper produces `data/output/<stem>/output.yaml`:

```yaml
title:
  value: "Cesium Titanium(IV) Bromide Thin Films Based Stable Lead-free Perovskite Solar Cells"
  source: "Cesium Titanium(IV) Bromide..."
  grounded: true
  span: {start: 0, end: 81}
  source_doc: main

is_perovskite_synthesis: true

material:
  value: Cs2TiBr6
  source: high-quality, uniform thin films of Cs2TiBr6 HP can be prepared
  grounded: true
  phcs_stable: true

precursors:
  - name: CsBr
    concentration: null
    solvent: null
    source: CsBr thin films were deposited
    grounded: true
    source_doc: main
  - name: TiBr4
    concentration: null
    solvent: null
    source: annealed in a TiBr4-vapor atmosphere at 200 °C
    grounded: true
    source_doc: main

synthesis_method:
  value: low-temperature vapor-based method
  source: prepared through a facile low-temperature vapor-based method
  grounded: true

process_conditions:
  temperature:
    value: 200 °C
    source: annealed in a TiBr4-vapor atmosphere at 200 °C
    grounded: true
    phcs_stable: true
  duration:
    value: 24 h
    source: annealed in a TiBr4-vapor atmosphere at 200 °C for 24 h
    grounded: true
  atmosphere:
    value: TiBr4 vapor
    source: annealed in a TiBr4-vapor atmosphere
    grounded: true
  annealing:
    value: 200 °C for 24 h in TiBr4 vapor
    source: annealed in a TiBr4-vapor atmosphere at 200 °C for 24 h
    grounded: true

substrate:
  value: TiO2-coated glass
  source: deposited on compact TiO2-coated glass substrates
  grounded: true

characterization:
  bandgap:
    value: 1.8 eV
    source: These thin films exhibit a favorable bandgap of ≈1.8 eV
    grounded: true

device_performance:
  pce:
    value: 3.28%
    source: best PSCs show stabilized efficiency of up to 3.28%
    grounded: true

figures:
  - figure_number: 1
    page: 3
    caption: Figure 1. XRD patterns of Cs2TiBr6 thin films...
    filename: fig_1.png
    description:
      figure_type: XRD
      key_values: ["2θ = 28.5°", "2θ = 31.2°"]
      phases_or_materials: ["Cs2TiBr6"]
      key_observation: XRD confirms cubic perovskite phase formation.

_meta:
  paper_type: original
  extraction_model: meta-llama/llama-3.3-70b-instruct:free
  parser: grobid
  consistency_runs: 2
  extraction_timestamp: "2025-06-05T12:00:00Z"
  si_referenced: true
  si_merged: false
  adversarial_critique:
    wrong_section_fields: []
    missing_fields: []
    product_as_precursor: []
    overall_quality: good
    summary: Extraction is accurate and complete.
  phcs_summary:
    process_conditions.temperature: 1.0
    material: 1.0
    precursors: 0.75
  grounding_precision: 0.91
  synthesis_completeness: 0.82
  field_weighted_score: 0.74
  missing_critical_fields: []
```

A normalized flat JSON (`normalized.json`) is also written alongside each `output.yaml`, stripping the `{value, source}` envelope to plain values for downstream analysis.

### On-the-Fly Quality Metrics

No ground-truth labels are required. Quality is assessed automatically during extraction:

| Metric | Description |
|---|---|
| `grounding_precision` | `grounded_count / verifiable_non_null_count` — citation quality |
| `synthesis_completeness` | Fraction of critical + important fields that are non-null |
| `field_weighted_score` | Weighted average over critical (3×), important (1×), and minor (0.5×) fields |
| `missing_critical_fields` | Explicit list of critical fields that are null |
| `phcs_summary` | Per-field probe stability scores (0–1) |
| `adversarial_critique.overall_quality` | Second-LLM assessment: good / acceptable / poor |

### Running the Pipeline

**Single paper**:
```bash
source .venv/bin/activate
python test_scripts/test_pipeline.py my_paper.pdf
```

**Full batch** (skips already-processed papers):
```bash
bash test_scripts/run_batch.sh
```

**Limit to N papers** (useful for testing):
```bash
bash test_scripts/run_batch.sh --limit 5
```

**Force reprocessing**: delete `data/output/<stem>/output.yaml` and re-run.

**Skip GROBID** (use PyMuPDF only):
```bash
GROBID_ENABLED=0 python test_scripts/test_pipeline.py my_paper.pdf
```

---

## Pipeline 2 — Training

Two complementary training pipelines built on top of the same extraction outputs and the Perovskite database CSV.

### Data Preparation

**Source**: `data/raw/Perovskite_database_content_all_data.csv`

```bash
python prepare_training_dataset.py \
  --input data/raw/Perovskite_database_content_all_data.csv \
  --output-dir data/processed/fine_tuning \
  --input-repr core_rare \
  --output-schema performance_with_justification
```

**Input representations** — controls which columns become the prompt:

| Mode | Description |
|---|---|
| `core` | Device composition, stack, synthesis method only |
| `core_secondary` | Core + secondary measurements (bandgap, defect density, etc.) |
| `core_rare` *(default)* | Core + columns selected by activation frequency analysis (caps at 60 features) |
| `hierarchical` | Nested structure (composition.lead_content, stack.electron_transport, etc.) |

**Output schemas** — controls the assistant completion format:

| Schema | Description |
|---|---|
| `performance_only` | `{"pce": 15.2, "voc": 1.05, "jsc": 22.5, "ff": 0.77}` |
| `performance_with_justification` *(default)* | Predictions + per-field reasoning and limiting factor |

**Train/val/test split**: 80/10/10, stratified by DOI (`Ref_DOI_number`) to prevent data leakage between devices from the same paper.

### SFT Training

Supervised fine-tuning for performance prediction or recipe generation:

```bash
python train_llm.py \
  --model-name "meta-llama/llama-3.3-70b-instruct" \
  --task performance_prediction \
  --data-dir data/processed/fine_tuning \
  --dataset-prefix core_rare__performance_with_justification \
  --output-dir runs/llm_sft \
  --fine-tuning-method lora \
  --epochs 1.0 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --gradient-checkpointing
```

**Fine-tuning methods** (set with `--fine-tuning-method`):

| Method | Description |
|---|---|
| `full` | Update all weights — highest capacity, highest memory |
| `partial` | Freeze all but last N layers |
| `lora` | Low-rank adapters on q/k/v/o projections (r=16, α=32) |
| `qlora` | LoRA on 4-bit quantized base (NF4, double quant) |
| `dora` | Weight-decomposed LoRA — often stronger than plain LoRA |
| `adapters` | Bottleneck modules inserted between transformer layers |
| `prefix_tuning` | Trainable key/value prefix per layer |
| `prompt_tuning` | Soft prompt embeddings — minimal parameter count |

**Key training decisions**:

- *Completion-only loss*: prompt tokens are masked with `label=-100` so the model only learns to generate the answer, not to repeat the input.
- *Per-field loss weighting*: pass `--field-loss-weights '{"pce": 2.0, "ff": 0.5}'` to upweight critical outputs.
- *Attention backend*: `--attention-backend flash_attention_2` for speed on supported hardware.

**Evaluation** runs automatically after training: generates predictions on the test set, parses JSON, and reports MAE, RMSE, and R² per field (PCE, Voc, Jsc, FF).

**Recipe generation SFT**:

```bash
python train_llm.py \
  --model-name "meta-llama/llama-3.3-70b-instruct" \
  --task recipe_generation \
  --prompt-jsonl data/recipe_prompts.jsonl \
  --output-dir runs/recipe_sft \
  --fine-tuning-method qlora
```

The model learns to generate valid `{recipe, constraints_satisfied}` JSON from constraint descriptions (lead-free, no chlorinated solvents, target PCE, etc.).

### GRPO / RLVR Training

After SFT teaches the model to emit valid recipe JSON, GRPO refines it using verifiable rewards — no human labels required at inference time.

```bash
python train_rlvr.py \
  --model-name runs/recipe_sft/final \
  --prompt-jsonl data/recipe_prompts.jsonl \
  --output-dir runs/recipe_grpo \
  --num-generations 4 \
  --beta 0.02 \
  --epochs 1.0
```

**Reward function** — decomposed JSON schema compliance:

| Component | Reward |
|---|---|
| Valid JSON object | +1.0 |
| Correct top-level keys (`recipe`, `constraints_satisfied`) | +0.5 / −0.5 |
| `recipe` is a dict | +0.5 / −1.0 |
| Correct recipe sections (composition, device_stack, deposition, transport_layers) | +0.75 / −0.25 |
| Minimum content present (composition formula, stack sequence, deposition method) | +1.0 / −0.5 |
| `constraints_satisfied` is a dict | +0.5 / −1.0 |
| Correct constraint keys | +0.75 / −0.25 |
| `no_chlorinated_solvents: true` | +0.5 / −0.25 |

Range: approximately −4.0 to +4.0.

**GRPO setup**: generates 4 completions per prompt, scores each, computes normalized advantages, updates model with a KL penalty (`beta=0.02`) against the SFT baseline.

**Dry run** (score existing outputs without RL training):
```bash
python train_rlvr.py \
  --model-name runs/recipe_sft/final \
  --prompt-jsonl data/recipe_prompts.jsonl \
  --dry-run-rewards
```

---

## Configuration

**`config/llm_config.yaml`** — controls all LLM behaviour in the extraction pipeline:

```yaml
provider: openrouter

# Primary extraction model
model: "meta-llama/llama-3.3-70b-instruct:free"

# Fallback chain (tried in order on error or empty response)
fallback_models:
  - "nousresearch/hermes-3-llama-3.1-405b:free"
  - "openai/gpt-oss-120b:free"
  - "nvidia/nemotron-3-super-120b-a12b:free"
  # ... 13 more

# Self-consistency: 1 = disabled, 2+ = majority vote
consistency_runs: 2

# PHCS probe verification on high-risk fields after extraction
run_phcs: true

# Vision-language model for figure description
vlm_model: "google/gemma-4-31b-it:free"
vlm_fallback_models:
  - "google/gemma-4-26b-a4b-it:free"
  - "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
  - "nvidia/nemotron-nano-12b-v2-vl:free"
```

**Environment variables** for the extraction pipeline:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required. OpenRouter API key. |
| `GROBID_ENABLED` | `1` | Set to `0` to force PyMuPDF fallback. |
| `GROBID_URL` | `https://kermitt2-grobid.hf.space` | GROBID API base URL. |
| `GROBID_TIMEOUT` | `60` | Seconds before GROBID request times out. |
