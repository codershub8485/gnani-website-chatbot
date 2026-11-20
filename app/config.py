"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# --- Data -------------------------------------------------------------------
DATA_DIR = Path(os.getenv("GNANI_DATA_DIR", ROOT_DIR / "data"))
PAGES_FILE = DATA_DIR / "pages.jsonl"  # output of the scraper, input of the indexer

# --- Crawler ----------------------------------------------------------------
SITE_ROOT = "https://www.gnani.ai"
SITEMAP_URL = f"{SITE_ROOT}/sitemap.xml"
# A plain, honest user agent (no impersonation of browsers or other bots).
CRAWLER_USER_AGENT = os.getenv("GNANI_CRAWLER_UA", "GnaniSiteQA-Assessment/1.0 (website Q&A demo)")
CRAWLER_WORKERS = int(os.getenv("GNANI_CRAWLER_WORKERS", "4"))
CRAWLER_DELAY_S = float(os.getenv("GNANI_CRAWLER_DELAY_S", "0.25"))  # per-request politeness delay

# --- Retrieval --------------------------------------------------------------
CHUNK_WORDS = int(os.getenv("GNANI_CHUNK_WORDS", "220"))
CHUNK_OVERLAP_WORDS = int(os.getenv("GNANI_CHUNK_OVERLAP_WORDS", "40"))
TOP_K = int(os.getenv("GNANI_TOP_K", "6"))
MAX_CHUNKS_PER_PAGE = int(os.getenv("GNANI_MAX_CHUNKS_PER_PAGE", "3"))

# --- LLM --------------------------------------------------------------------
# groq (default, free key) | ollama (local, no key) | anthropic | openai-compatible
LLM_PROVIDER = os.getenv("GNANI_LLM_PROVIDER", "groq").strip().lower()
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",  # open-weight (Apache-2.0) model, free on Groq
    "ollama": "qwen2.5:7b",
    "anthropic": "claude-opus-5-5",
}
MODEL = os.getenv("GNANI_MODEL", "")  # empty = the provider's default above
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_BASE_URL = os.getenv("GNANI_LLM_BASE_URL", "")  # only needed for openai-compatible
LLM_API_KEY = os.getenv("GNANI_LLM_API_KEY", "")  # only needed for openai-compatible
EFFORT = os.getenv("GNANI_EFFORT", "medium")  # Claude only: low | medium | high | xhigh | max
CLAUDE_MAX_TOKENS = int(os.getenv("GNANI_CLAUDE_MAX_TOKENS", "8000"))
# Output cap for Groq/Ollama. Keep it tight: Groq's free tier reserves prompt + this cap
# against its 8,000 tokens/minute budget, so a big cap means fewer questions per minute.
MAX_TOKENS = int(os.getenv("GNANI_MAX_TOKENS", "1000"))
# gpt-oss models only: low reasoning is plenty for grounded Q&A and answers faster.
REASONING_EFFORT = os.getenv("GNANI_REASONING_EFFORT", "low")
HISTORY_TURNS = int(os.getenv("GNANI_HISTORY_TURNS", "4"))  # past Q/A pairs sent as context
