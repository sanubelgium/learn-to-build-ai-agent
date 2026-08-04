"""
LangSmith evaluation target function. Calls rag_pipeline_design(query) and 
returns its structured dict output directly.

Args:
    inputs: dict with keys:
        - "question"   (str) — question to answer
        - "category"   (str) — metadata (not used by RAG logic)
        - "difficulty" (str) — metadata (not used by RAG logic)

Returns:
    dict: {"answer": str, "retrieved_contexts": list[str], "question": str}
"""
import os
import sys
from pathlib import Path
from config.settings import RAG_PIPELINE_DIR
from dotenv import load_dotenv

# ── project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Add the rag_pipeline directory so the module is importable by name
sys.path.append(str(RAG_PIPELINE_DIR))
load_dotenv(PROJECT_ROOT / ".env")

if "CHATBOT_API_KEY" in os.environ:
    os.environ["GROQ_API_KEY"] = os.environ["CHATBOT_API_KEY"]

from rag_pipeline_design import rag_pipeline_design


def rag_target(inputs: dict) -> dict:
    return rag_pipeline_design(inputs["question"])
