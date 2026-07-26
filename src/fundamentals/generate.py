import sys
from pathlib import Path

# Add the project root to the python path so config can be found
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Add the current directory of the script to sys.path to find retrieval
sys.path.append(str(Path(__file__).resolve().parent))

import ollama
from retrieval import retrieve, TOP_K
from config.settings import MAX_TOKENS, MODEL, SYSTEM_PROMPT


def build_prompt(query: str, results) -> str:
    """Assemble the retrieved chunks and the user's question into one prompt."""
    context_blocks = []
    for i, (doc, score) in enumerate(results, start=1):
        page = doc.metadata.get("page", "unknown")
        context_blocks.append(f"[Chunk {i} | page {page}]\n{doc.page_content.strip()}")

    context = "\n\n".join(context_blocks)

    prompt = (
        f"Context from the document:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using only the context above. "
        f"Cite the page number(s) you used."
    )
    return prompt


def generate_answer(prompt: str, model: str = MODEL, max_tokens: int = MAX_TOKENS) -> str:
    """Call the local Ollama model with the augmented prompt and return its answer."""
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"num_predict": max_tokens},
        )
    except Exception as exc:
        raise ConnectionError(
            f"Could not reach Ollama (is it running? try `ollama serve`), "
            f"or model '{model}' isn't pulled (try `ollama pull {model}`). "
            f"Original error: {exc}"
        )

    return response["message"]["content"]


def answer_question(query: str, k: int = TOP_K) -> str:
    """Run the full RAG pipeline end to end: retrieve -> build prompt -> generate."""
    print(f"[answer_question] Retrieving context for: {query!r}")
    results = retrieve(query, k=k)

    print(f"[answer_question] Building augmented prompt from {len(results)} chunk(s)")
    prompt = build_prompt(query, results)

    print("[answer_question] Calling LLM...")
    answer = generate_answer(prompt)

    return answer


if __name__ == "__main__":
    # --- Set your question directly here ---
    QUESTION = "What is budget alloted for education sector?"

    try:
        answer = answer_question(QUESTION)
        print("\n--- Answer ---\n")
        print(answer)
    except Exception as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        sys.exit(1)