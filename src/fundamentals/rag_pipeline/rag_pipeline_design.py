import os
import sys
import chromadb
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the python path so config can be found
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from config.settings import *
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
# Map the CHATBOT_API_KEY from .env to the standard OPENAI_API_KEY environment variable
if "CHATBOT_API_KEY" in os.environ:
    os.environ["GROQ_API_KEY"] = os.environ["CHATBOT_API_KEY"]


def rag_pipeline_design(query):
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    ###Embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL ,
        encode_kwargs={"normalize_embeddings": True},
    )
    if collection.count() == 0:
        ###Chunks Created
        pages = [
            Document(page_content=page.extract_text() or "", metadata={"source": PDF_PATH, "page": i})
            for i, page in enumerate(PdfReader(PDF_PATH).pages)
        ]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)
        ###Embeddings of chunks
        embedding_chunks = embeddings.embed_documents([doc.page_content for doc in chunks])

        ###Store chunks
        collection.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            documents=[doc.page_content for doc in chunks],
            embeddings=embedding_chunks,
            metadatas=[doc.metadata for doc in chunks]
        )
        print(f"[ingest] Embedded and stored {len(chunks)} chunks into '{COLLECTION_NAME}'.")
    else:
        print(f"[ingest] Collection '{COLLECTION_NAME}' already has {collection.count()} chunks — skipping ingestion.")  
    ###query vector
    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    ###Context Retreived
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context_lines = []
    for d, m in zip(docs, metas):
        page = m.get("page", "?")
        context_lines.append(f"[page {page}]\n{d.strip()}")
    context = "\n\n".join(context_lines)
    ###print("context", context)

    ### Prompt construction:
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant answering questions using only the provided context. If the context does not contain the answer, say so plainly instead of guessing. 
        Context: {context} 
        Question: {question}
        Answer:"""
    )   
    ###print("prompt", prompt)

    # Initialize the LLM
    groq_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    chain = prompt | groq_llm
    response = chain.invoke({"context": context, "question": query})
    # Return structured dict so callers (evaluators) get answer + retrieved chunks
    return {
        "answer": response.content,
        "retrieved_contexts": docs,
        "question": query,
    }

if __name__ == "__main__":
    query = "what is YUva shakti and 3 kartavya?"
    result = rag_pipeline_design(query)
    print("\n\nAnswer:", result["answer"])
