"""
Agent that reads scientific PDFs from data/corpus/ and parses them into
semantic sections (Title, Abstract, Introduction, Methods, Results,
Discussion, Conclusion, References) using LangChain + OpenRouter.
"""

import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
CONFIG_PATH = REPO_ROOT / "config" / "llm_config.yaml"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def resolve_llm(cfg: dict) -> ChatOpenAI:
    """Try each model in config order; return the first one that responds."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")

    all_models = [cfg["model"]] + cfg.get("fallback_models", [])

    for model in all_models:
        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=4096,
        )
        try:
            llm.invoke("ping")
            print(f"Using model: {model}")
            return llm
        except Exception as e:
            print(f"  {model} unavailable ({type(e).__name__}), trying next...")

    raise RuntimeError("No available models found in llm_config.yaml.")

# ---------------------------------------------------------------------------
# Tools available to the agent
# ---------------------------------------------------------------------------

@tool
def list_corpus_files() -> str:
    """List all PDF files available in the scientific article corpus."""
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        return "No PDF files found in corpus."
    return "\n".join(f.name for f in pdfs)


@tool
def read_pdf(filename: str) -> str:
    """
    Load a PDF from the corpus and return its full text content.
    Pass only the filename (e.g. '001_Oriented_piezoelectric_films.pdf').
    """
    path = CORPUS_DIR / filename
    if not path.exists():
        return f"File not found: {filename}"
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    if not pages:
        return "PDF loaded but contained no extractable text."
    # Concatenate all pages with page markers
    parts = [f"[Page {i+1}]\n{p.page_content.strip()}" for i, p in enumerate(pages)]
    return "\n\n".join(parts)


TOOLS = [list_corpus_files, read_pdf]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a scientific document parser. Your task is to read a scientific
article from the corpus and extract its content into clearly labelled
semantic sections.

For each article you are given, follow these steps:
1. Call read_pdf with the filename to get the full text.
2. Identify and extract the following sections (use "N/A" if a section is
   absent): Title, Authors, Abstract, Introduction, Materials & Methods,
   Results, Discussion, Conclusion, References.
3. Return your output as a JSON object with those section names as keys and
   the extracted text as values. Keep each section concise but complete.

Only output valid JSON — no surrounding markdown, no commentary.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def build_agent(llm) -> AgentExecutor:
    agent = create_tool_calling_agent(llm, TOOLS, PROMPT)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=True, max_iterations=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_article(executor: AgentExecutor, filename: str) -> dict:
    print(f"\n{'='*70}")
    print(f"Parsing: {filename}")
    print("="*70)
    result = executor.invoke({"input": f"Parse this article into semantic sections: {filename}"})
    raw = result.get("output", "")
    try:
        # Strip markdown code fences if the model added them
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_output": raw}


def main():
    cfg = load_config()
    llm = resolve_llm(cfg)
    executor = build_agent(llm)

    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {CORPUS_DIR}")
        return

    # Process first 3 articles as a smoke test; remove the slice for all
    for pdf_path in pdfs[:3]:
        sections = parse_article(executor, pdf_path.name)
        print("\n--- Extracted Sections ---")
        print(json.dumps(sections, indent=2, ensure_ascii=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
