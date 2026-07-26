
import sys
from pathlib import Path

# Add the project root to the python path so config can be found
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config.settings import PDF_PATH, DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, COLLECTION_NAME


def load_pdf(pdf_path: str):
    """Load a PDF and return one LangChain Document per page (via pypdf directly)."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))
    pages = [
        Document(page_content=page.extract_text() or "", metadata={"source": str(path), "page": i})
        for i, page in enumerate(reader.pages)
    ]
    print(f"[load_pdf] Loaded {len(pages)} page(s) from {path.name}")
    return pages


def chunk_documents(pages, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Split pages into overlapping chunks sized for embedding & retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"[chunk_documents] Split into {len(chunks)} chunk(s) "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def build_embedding_model(model_name: str = EMBEDDING_MODEL):
    """Load a local embedding model. Swap this for OpenAIEmbeddings etc. if preferred."""
    print(f"[build_embedding_model] Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)


def store_in_vector_db(chunks, embedding_model, persist_dir: str, collection_name: str = COLLECTION_NAME):
    """Embed chunks and persist them to a local Chroma vector store."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    # langchain-chroma persists automatically to persist_directory; no separate
    # .persist() call needed (that method was removed from the newer package).
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    print(f"[store_in_vector_db] Persisted {len(chunks)} chunk(s) to '{persist_dir}' "
          f"(collection='{collection_name}')")
    return vectordb


def ingest(pdf_path: str, persist_dir: str, chunk_size: int, chunk_overlap: int):
    """Load Pages"""
    pages = load_pdf(pdf_path)
    print("Pages: ", pages)
    """Split Pages"""
    chunks = chunk_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print("Chunks: ", chunks)
    """Load Embedding Model"""
    embedding_model = build_embedding_model()
    print("Embedding Model: ", embedding_model)
    """Store in Vector DB"""
    vectordb = store_in_vector_db(chunks, embedding_model, persist_dir)
    print("[ingest] Done. Vector store is ready for retrieval.")
    return vectordb


if __name__ == "__main__":
    try:
        print("Starting ingestion process...")
        ingest(PDF_PATH, DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)