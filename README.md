# RAG Ingestion Pipeline

RAG ingestion pipeline: PDF -> chunks -> embeddings -> vector store.

Covers the "Ingestion" phase of the RAG diagram:
  PDF document -> Chunking -> Embedding model -> Vector store

Stack used (all free / local, no API key required):
  - pypdf                 (PDF loading, no langchain-community dependency)
  - langchain-text-splitters (chunking)
  - langchain-huggingface (local embedding model, standalone package)
  - langchain-chroma      (local vector store, standalone package, persisted to disk)

Note: langchain-community is now sunset/archived upstream, so this avoids it
entirely in favor of the maintained standalone integration packages.

Install:
    pip install pypdf langchain-core langchain-text-splitters langchain-huggingface langchain-chroma

Usage:
    python ingest.py

# RAG Query & Retrieval Pipeline

RAG query & retrieval pipeline: user query -> query embedding -> top-k chunks.

Covers the "Query & retrieval" phase of the RAG diagram:
  User query -> Query embedding -> Retrieve top-k (similarity search)

Loads the vector store persisted by ingest.py and returns the most relevant
chunks for a given question. Must use the SAME embedding model as ingest.py,
or similarity scores become meaningless.

Usage:
    python retrieval.py

# RAG Generation Pipeline

RAG generation pipeline: retrieved chunks + query -> augmented prompt -> LLM -> answer

Covers the "Generation" phase of the RAG diagram:
  Augmented prompt -> LLM -> Answer

Ties the whole pipeline together by importing retrieve() from retrieval.py,
building a prompt grounded in the retrieved PDF chunks, and calling a LOCAL
LLM via Ollama. No API key, no internet connection needed once the model
is pulled.

Setup:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model, e.g.:  ollama pull llama3.1
    3. Make sure Ollama is running (it starts a local server on :11434)
    4. pip install ollama
       (plus everything retrieval.py needs: langchain, langchain-community,
        sentence-transformers, chromadb)

Usage:
    python generate.py

# LangChain LCEL Pipeline

RAG pipeline as a single LangChain chain (LCEL), using local Ollama.

Instead of manually gluing retrieve() -> build_prompt() -> generate_answer()
like generate.py does, this wires everything into one LangChain "Runnable"
using the pipe (|) operator:

    retriever | format_docs  ->  prompt  ->  llm  ->  output parser

Benefits over the manual version:
  - One object (`chain`) you can .invoke() or .stream() instead of three
    separate function calls to keep in sync.
  - Swapping the LLM, retriever, or prompt is a one-line change.
  - Streaming support comes for free via chain.stream(...).

Reuses ingest.py's vector store (built with langchain-huggingface +
langchain-chroma, no langchain-community involved).

Install:
    pip install langchain-core langchain-huggingface langchain-chroma langchain-ollama

Setup: same as generate.py — Ollama running locally with a model pulled
    (ollama pull llama3.1)

Usage:
    python combined_rag_chain.py

