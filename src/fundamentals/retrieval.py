import sys
from pathlib import Path

# Add the project root to the python path so config can be found
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config.settings import EMBEDDING_MODEL, DB_DIR, COLLECTION_NAME, TOP_K


def build_embedding_model(model_name: str = EMBEDDING_MODEL):
    """Load the same local embedding model used at ingestion time."""
    print(f"[build_embedding_model] Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)
 
 
def load_vector_db(persist_dir: str, embedding_model, collection_name: str = COLLECTION_NAME):
    """Load the persisted Chroma vector store created by ingest.py."""
    if not Path(persist_dir).exists():
        raise FileNotFoundError(
            f"Vector store not found at '{persist_dir}'. Run ingest.py first."
        )
 
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=collection_name,
    )
    print(f"[load_vector_db] Loaded vector store from '{persist_dir}' "
          f"(collection='{collection_name}')")
    return vectordb
 
 
def retrieve_top_k(vectordb, query: str, k: int = TOP_K):
    """Embed the query and return the k most similar chunks, with scores."""
    results = vectordb.similarity_search_with_score(query, k=k)
    print(f"[retrieve_top_k] Retrieved {len(results)} chunk(s) for query: {query!r}")
    return results
 
 
def format_results(results):
    """Pretty-print retrieved chunks with their source page and similarity score."""
    lines = []
    for i, (doc, score) in enumerate(results, start=1):
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        lines.append(
            f"[{i}] score={score:.4f} | source={Path(source).name} | page={page}\n{snippet}\n"
        )
    return "\n".join(lines)
 
 
def retrieve(query: str, db_dir: str = DB_DIR, k: int = TOP_K):
    """embedding model"""
    embedding_model = build_embedding_model()
    """Vector DB"""
    vectordb = load_vector_db(db_dir, embedding_model)
    """Top K chunks"""
    results = retrieve_top_k(vectordb, query, k=k)
    return results
 
 
if __name__ == "__main__":
    # --- Set your test query directly here ---
    QUERY = "What is the budget alloted for education sector?"
 
    try:
        print("Starting retrieval process...")
        results = retrieve(QUERY)
        print("\n" + format_results(results))
    except Exception as exc:
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        sys.exit(1)