from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = str(PROJECT_ROOT / "data" / "raw_pdfs" / "budget_speech.pdf")
DB_DIR = str(PROJECT_ROOT / "data" / "vectorstore")   
RAG_PIPELINE_DIR = PROJECT_ROOT / "src" / "fundamentals" / "rag_pipeline"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
COLLECTION_NAME = "my_first_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_JSON = PROJECT_ROOT / "data" / "evaluation_dataset" / "rag_test_dataset.json"
LANGSMITH_DATASET_NAME = "rag-budget-speech-eval-v1.1"
TOP_K = 4
MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 1000
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the "
    "provided context from a PDF document. If the context does not contain "
    "enough information to answer, say so clearly instead of guessing."
)

SYSTEM_PROMPT_RAG_CHAIN = (
    "You are a helpful assistant that answers questions using only the "
    "provided context from a PDF document. If the context does not contain "
    "enough information to answer, say so clearly instead of guessing. "
    "Cite the page number(s) you used."
)
USE_FULL_CHAIN = False  # True: ingest + answer in one chain.invoke() call
                        # False (recommended): ingest once, reuse chain for many questions
