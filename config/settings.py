
PDF_PATH = "../../data/raw_pdfs/budget_speech.pdf"   
DB_DIR = "../../data/vectorstore"   
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
COLLECTION_NAME = "pdf_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
MODEL = "llama3.1"
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
USE_FULL_CHAIN = True  # True: ingest + answer in one chain.invoke() call
                        # False (recommended): ingest once, reuse chain for many questions
