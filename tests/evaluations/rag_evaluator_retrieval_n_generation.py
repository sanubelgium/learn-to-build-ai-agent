"""
rag_evaluator.py
----------------
Orchestrates a full LangSmith evaluation run against the budget-speech RAG
pipeline using four LLM-as-a-judge evaluators:

  1. faithfulness        — does the answer stay within the retrieved context?
  2. context_relevance   — are the retrieved chunks relevant to the question?
  3. answer_correctness  — does the answer match the ground truth?
  4. answer_relevance    — does the answer actually address the question asked?

Usage:
    python src/fundamentals/evaluations/rag_evaluator_retrieval_n_generation.py

Prerequisites:
    1. The vector store must be pre-populated:
           python src/fundamentals/rag_pipeline/rag_pipeline_design.py
    2. The dataset must be uploaded to LangSmith:
           python src/fundamentals/evaluations/upload_dataset.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

if "CHATBOT_API_KEY" in os.environ:
    os.environ["GROQ_API_KEY"] = os.environ["CHATBOT_API_KEY"]

from langsmith import Client, evaluate
from langsmith.schemas import Example, Run
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config.settings import LANGSMITH_DATASET_NAME
from src.fundamentals.evaluations.rag_target import rag_target

# Judge LLM
judge_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# Helper: call judge LLM and parse score + reason

def judge(system_prompt: str, user_message: str) -> tuple[float, str]:
    """
    Call the judge LLM.  Expects the model to respond with:
        Score: <0-10>
        Reason: <one sentence>
    Returns (normalised_score_0_to_1, reason_string).
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_message),
    ])
    chain = prompt | judge_llm
    response = chain.invoke({})
    text: str = response.content.strip()

    # ── parse score ──────────────────────────────────────────────────────────
    score = 0.5  # fallback
    reason = text
    for line in text.splitlines():
        if line.lower().startswith("score:"):
            try:
                raw = line.split(":", 1)[1].strip().split()[0]
                score = float(raw) / 10.0
                score = max(0.0, min(1.0, score))
            except (ValueError, IndexError):
                pass
        if line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    return score, reason


# Evaluator 1 — Faithfulness
FAITHFULNESS_SYSTEM = """
You are an expert evaluator assessing the faithfulness of a RAG system's answer.

FAITHFULNESS measures whether every claim in the answer is supported by the
retrieved context. An answer that introduces facts NOT found in the context
is unfaithful (hallucinated).

Scoring rubric (0–10):
  10 — every statement is directly supported by the context
   7 — mostly faithful; one minor unsupported detail
   5 — half the answer is grounded; rest is extrapolated
   2 — answer largely fabricates information
   0 — completely hallucinated; context ignored

Respond with EXACTLY:
  Score: <integer 0-10>
  Reason: <one concise sentence>
"""


def evaluate_faithfulness(run: Run, example: Example) -> dict:
    """LangSmith evaluator for faithfulness."""
    answer: str = (run.outputs or {}).get("answer", "")
    contexts: list[str] = (run.outputs or {}).get("retrieved_contexts", [])
    context_text = "\n\n".join(contexts) if contexts else "(no context retrieved)"

    user_msg = (
        f"Retrieved Context:\n{context_text}\n\n"
        f"Answer to evaluate:\n{answer}"
    )
    score, reason = judge(FAITHFULNESS_SYSTEM, user_msg)
    return {"key": "faithfulness", "score": score, "comment": reason}


# Evaluator 2 — Context Relevance
CONTEXT_RELEVANCE_SYSTEM = """
You are an expert evaluator assessing context relevance for a RAG system.

CONTEXT RELEVANCE measures how well the retrieved document chunks address the
user's question. High relevance means the chunks contain information that
directly helps answer the question.

Scoring rubric (0–10):
  10 — all retrieved chunks are directly relevant to the question
   7 — most chunks relevant; one or two are tangentially related
   5 — some relevant, some off-topic
   2 — chunks barely touch the question topic
   0 — completely irrelevant chunks returned

Respond with EXACTLY:
  Score: <integer 0-10>
  Reason: <one concise sentence>
"""


def evaluate_context_relevance(run: Run, example: Example) -> dict:
    question: str = (run.outputs or {}).get("question", "")
    contexts: list[str] = (run.outputs or {}).get("retrieved_contexts", [])
    context_text = "\n\n".join(contexts) if contexts else "(no context retrieved)"

    user_msg = (
        f"Question:\n{question}\n\n"
        f"Retrieved Context Chunks:\n{context_text}"
    )
    score, reason = judge(CONTEXT_RELEVANCE_SYSTEM, user_msg)
    return {"key": "context_relevance", "score": score, "comment": reason}


# Evaluator 3 — Answer Correctness
ANSWER_CORRECTNESS_SYSTEM = """
You are an expert evaluator assessing the correctness of a RAG system's answer
against a ground truth reference answer.

ANSWER CORRECTNESS measures factual agreement between the generated answer and
the reference. It does NOT penalise different wording; only incorrect or missing
facts matter.

Scoring rubric (0–10):
  10 — answer is fully correct; all key facts match the reference
   7 — mostly correct; minor details missing or slightly off
   5 — partially correct; captures half the key facts
   2 — mostly incorrect or contradicts the reference
   0 — completely wrong or completely irrelevant

Respond with EXACTLY:
  Score: <integer 0-10>
  Reason: <one concise sentence>
"""


def evaluate_answer_correctness(run: Run, example: Example) -> dict:
    """LangSmith evaluator for answer correctness (vs ground truth)."""
    answer: str = (run.outputs or {}).get("answer", "")
    ground_truth: str = (example.outputs or {}).get("ground_truth", "")

    user_msg = (
        f"Ground Truth Reference Answer:\n{ground_truth}\n\n"
        f"Generated Answer:\n{answer}"
    )
    score, reason = judge(ANSWER_CORRECTNESS_SYSTEM, user_msg)
    return {"key": "answer_correctness", "score": score, "comment": reason}


# Evaluator 4 — Answer Relevance
ANSWER_RELEVANCE_SYSTEM = """\
You are an expert evaluator assessing the relevance of a RAG system's answer
to the original question.

ANSWER RELEVANCE measures whether the answer directly addresses the user's
question, regardless of whether it matches a reference answer.

Scoring rubric (0-10):
  10 — answer directly and completely addresses the question
   7 — mostly on point; slight tangent or minor omission
   5 — partially answers the question; significant gaps
   2 — barely relevant; mostly off-topic
   0 — completely ignores the question

Respond with EXACTLY:
  Score: <integer 0-10>
  Reason: <one concise sentence>
"""


def evaluate_answer_relevance(run: Run, example: Example) -> dict:
    """LangSmith evaluator for answer relevance (vs question)."""
    question: str = (run.outputs or {}).get("question", "")
    answer: str = (run.outputs or {}).get("answer", "")

    user_msg = (
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}"
    )
    score, reason = judge(ANSWER_RELEVANCE_SYSTEM, user_msg)
    return {"key": "answer_relevance", "score": score, "comment": reason}


# Main — run the evaluation
def run_evaluation(
    dataset_name: str = LANGSMITH_DATASET_NAME,
    experiment_prefix: str = "rag-budget-eval",
) -> None:
    client = Client()
    datasets = [d for d in client.list_datasets() if d.name == dataset_name]
    if not datasets:
        raise RuntimeError(
            f"Dataset '{dataset_name}' not found in LangSmith.\n"
            f"Run upload_dataset.py first."
        )

    print(f"\n[eval] Starting evaluation on dataset: '{dataset_name}'")
    print(f"[eval] Project: {os.environ.get('LANGSMITH_PROJECT', 'default')}")
    print("[eval] Evaluators: faithfulness, context_relevance, answer_correctness, answer_relevance\n")

    results = evaluate(
        rag_target,
        data=dataset_name,
        evaluators=[
            evaluate_faithfulness,
            evaluate_context_relevance,
            evaluate_answer_correctness,
            evaluate_answer_relevance,
        ],
        experiment_prefix=experiment_prefix,
        metadata={
            "pipeline": "rag_pipeline_design.py",
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "llm": "openai/gpt-oss-120b",
            "vector_store": "chromadb",
        },
    )

    # ── print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)

    metric_scores: dict[str, list[float]] = {
        "faithfulness": [],
        "context_relevance": [],
        "answer_correctness": [],
        "answer_relevance": [],
    }

    for result in results:
        for eval_result in result.get("evaluation_results", {}).get("results", []):
            key = eval_result.key
            score = eval_result.score
            if key in metric_scores and score is not None:
                metric_scores[key].append(score)

    for metric, scores in metric_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {metric:<22}  avg={avg:.3f}  ({len(scores)} samples)")
        else:
            print(f"  {metric:<22}  (no scores recorded)")

    print("=" * 65)

if __name__ == "__main__":
    run_evaluation()
