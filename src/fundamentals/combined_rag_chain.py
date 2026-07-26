import sys
from pathlib import Path

# Add the project root to the python path so config can be found
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.documents import Document
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from config.settings import *
 
 
def load_vector_db(persist_dir: str = DB_DIR, collection_name: str = COLLECTION_NAME):
    """Load the persisted Chroma vector store created by ingest.py."""
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=collection_name,
    )
 
 
def format_docs(docs) -> str:
    """Turn retrieved chunks into one context string, tagged with page numbers."""
    return "\n\n".join(
        f"[page {doc.metadata.get('page', '?')}]\n{doc.page_content.strip()}"
        for doc in docs
    )
 
 
def build_chain(vectordb, model: str = MODEL, k: int = TOP_K):
    """Assemble the retriever, prompt, and local LLM into one LCEL chain.
 
    Use this when the PDF is already ingested and you're answering multiple
    questions against the same vectordb (the normal chatbot case: ingest
    once per upload, then reuse this chain for every chat turn).
    """
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
 
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_RAG_CHAIN),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
 
    llm = ChatOllama(model=model)
 
    # The chain: retrieve docs -> format as context, pass question through unchanged
    # -> fill the prompt -> call the LLM -> parse to a plain string.
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
 
 
def ingest_step(inputs: dict) -> dict:
    """Runnable step: load + chunk + embed + store a PDF, return vectordb + question.
 
    CAVEAT: this re-ingests (re-embeds the whole PDF) on every invocation.
    Fine for a one-off script or a demo; wasteful in a real chatbot where the
    same PDF gets many questions. See build_full_chain() below vs. the
    ingest-once-then-build_chain() pattern used in __main__.
    """
    pages = [
        Document(page_content=page.extract_text() or "", metadata={"source": inputs["pdf_path"], "page": i})
        for i, page in enumerate(PdfReader(inputs["pdf_path"]).pages)
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
 
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks, embedding=embedding_model,
        collection_name=COLLECTION_NAME, persist_directory=DB_DIR,
    )
    return {"vectordb": vectordb, "question": inputs["question"]}
 
 
def build_full_chain(model: str = MODEL, k: int = TOP_K):
    """Start the chain"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_RAG_CHAIN),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    llm = ChatOllama(model=model)
 
    def retrieve_and_format(state: dict) -> dict:
        retriever = state["vectordb"].as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(state["question"])
        return {"context": format_docs(docs), "question": state["question"]}
 
    return (
        RunnableLambda(ingest_step)
        | RunnableLambda(retrieve_and_format)
        | prompt
        | llm
        | StrOutputParser()
    )
 
 
if __name__ == "__main__": 
    QUESTION = "What is alloted budget for defence sector?" 
 
    try:
        if USE_FULL_CHAIN:
            chain = build_full_chain()
            print(f"[rag_chain] Ingesting + asking: {QUESTION!r}\n")
            for token in chain.stream({"pdf_path": PDF_PATH, "question": QUESTION}):
                print(token, end="", flush=True)
            print()
        else:
            vectordb = load_vector_db()
            chain = build_chain(vectordb)
            print(f"[rag_chain] Asking: {QUESTION!r}\n")
            for token in chain.stream(QUESTION):
                print(token, end="", flush=True)
            print()
 
    except Exception as exc:
        print(f"Chain failed: {exc}", file=sys.stderr)
        sys.exit(1)