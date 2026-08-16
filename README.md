# learn-to-build-ai-agent — Repository Guide

A personal learning repo tracking a progression from RAG fundamentals through
LangGraph orchestration, agentic tool use (MCP), a full weekend-project
chatbot, and LLM-as-judge evaluation. This document explains what lives where,
what each project does, and how the pieces connect.

> Note on the name: this file documents the repo as it stands; it does not
> alter or execute any code. Update it as new folders/projects are added.

---

## Repository map

```
config/                        Shared configuration (single source of truth)
data/                          Sample PDF + evaluation dataset (vectorstore excluded from git)
src/
  fundamentals/                RAG built from scratch: ingest -> retrieve -> generate
    rag_pipeline/               Self-contained variant used by the eval suite
  chatbot_langgraph/            LangGraph chat agent + persistence + Streamlit UI
  langgraph_pipelines/          7 focused LangGraph pattern demos (no shared theme by design)
  mcp/                          Model Context Protocol: a tool server + a client that uses it
  weekend_projects/week1/       End-to-end deployable agent: FastAPI backend + Streamlit frontend
tests/evaluations/              LangSmith-based RAG evaluation harness (4 LLM-as-judge metrics)
requirements.txt                Dependency list (see Known Gaps below)
```

---

## `config/`

**`settings.py`** — the single place all other scripts pull constants from:
paths (`PDF_PATH`, `DB_DIR`, `DATASET_JSON`), chunking (`CHUNK_SIZE`,
`CHUNK_OVERLAP`), retrieval (`TOP_K`, `COLLECTION_NAME`, `EMBEDDING_MODEL`),
and generation (`MODEL`, `MAX_TOKENS`, both system prompts). Centralizing
config here is a good instinct — most scripts import from it rather than
hardcoding values (a couple of exceptions are noted under Known Gaps).

## `data/`

- `raw_pdfs/budget_speech.pdf` — the sample document every RAG script in this
  repo ingests and queries.
- `evaluation_dataset/rag_test_dataset.json` — question/ground-truth pairs
  used by the evaluation harness in `tests/evaluations/`.
- `vectorstore/` (gitignored) — the persisted Chroma DB, generated locally by
  running ingestion; not checked into version control.

---

## `src/fundamentals/` — RAG built from first principles

The core teaching sequence: each script is a separate, runnable phase of a
RAG pipeline, designed to be read in order.

| File | Phase | What it does |
|---|---|---|
| `ingestion.py` | PDF → chunks → embeddings → vector store | Loads the PDF with `pypdf`, splits with `RecursiveCharacterTextSplitter`, embeds locally with `HuggingFaceEmbeddings`, persists to Chroma. Run standalone: `python ingestion.py` |
| `retrieval.py` | Query → embedding → top-k chunks | Loads the persisted vector store and returns the most similar chunks with scores. Must use the same embedding model as ingestion or scores are meaningless. |
| `generate.py` | Context + query → answer | Glues `retrieve()` to a **local Ollama model** — manual, three-function pipeline (retrieve → build prompt → generate). No API key needed once a model is pulled. |
| `combined_rag_chain.py` | Same pipeline, LangChain LCEL | Rebuilds the same flow as a single composable `Runnable` (`retriever \| format_docs \| prompt \| llm \| parser`), so it supports `.stream()` and one-line component swaps. Also demonstrates a "ingest-and-answer-in-one-chain" variant (`build_full_chain`) alongside the recommended "ingest once, reuse for many questions" pattern. |

**`rag_pipeline/rag_pipeline_design.py`** — a self-contained, dependency-light
rewrite of the same pipeline using the raw `chromadb` client instead of
`langchain-chroma`, plus **Groq** (`llama-3.3-70b-versatile`) instead of
Ollama for generation. It auto-skips re-ingestion if the collection is
already populated. This is the version imported by the evaluation harness
(`tests/evaluations/rag_target.py`), which is why it lives in its own
sub-folder rather than alongside the Ollama-based scripts above.

**Two pipeline flavors exist side by side on purpose**: `generate.py` /
`combined_rag_chain.py` are the "local-only, no API key" teaching path via
Ollama; `rag_pipeline_design.py` is the "cloud LLM, ready for evaluation"
path via Groq. They intentionally don't share a vector store or embedding
model — see Known Gaps for a caveat if you point them at the same
`DB_DIR`/`COLLECTION_NAME`.

---

## `src/chatbot_langgraph/` — stateful chat with LangGraph

A minimal LangGraph single-node chatbot, built up in two stages plus a UI:

| File | Adds |
|---|---|
| `chatbot_using_langgraph.py` | The base graph: one state (`conversations`, using `add_messages` reducer), one node (`chat_node` calling Groq), in-memory checkpointing via `MemorySaver`. Conversation state resets when the process restarts. |
| `chatbot_in_langgraph_with_database.py` | Same graph, swapped to `SqliteSaver` so conversation history survives restarts. Adds `find_all_thread_ids()` to enumerate past conversations from the checkpoint DB. |
| `chatbot_ui_with_langgraph.py` | Streamlit frontend for the SQLite-backed graph: sidebar lists past thread IDs (loaded from `find_all_thread_ids()`), "New Chat" starts a fresh thread, selecting a past thread reloads its history from the checkpointer, and responses stream token-by-token via `chatbot_workflow.stream(..., stream_mode="messages")`. |

This folder is a good self-contained demonstration of LangGraph's
checkpointing model: identity is a `thread_id`, and swapping the persistence
backend (`MemorySaver` → `SqliteSaver`) doesn't require touching the graph
definition at all.

---

## `src/langgraph_pipelines/` — LangGraph pattern gallery

Seven independent, runnable scripts, each demonstrating one core LangGraph
orchestration pattern in isolation (not a pipeline — each file is a complete,
standalone example you can run on its own). Two use plain Python only, five
call Groq.

| File | Pattern | No LLM needed? |
|---|---|---|
| `simple_bmi_workflow.py` | Basic sequential graph (2 nodes) | Yes — pure calculation |
| `parallel_workflow_batsman_summary.py` | Fan-out / fan-in: 3 parallel calculation nodes joined by a summary node | Yes — pure calculation |
| `sequential_workflow_langgraph.py` | Single-node "ask an LLM a question" graph | No (Groq) |
| `prompt_chaining_workflow.py` | 3-step sequential chain: outline → content → evaluation | No (Groq) |
| `conditional_workflow_review_sentiment_analysis.py` | Conditional branching: sentiment classifier routes to either a thank-you response or a diagnosis + empathetic-response sub-chain | No (Groq, structured output via Pydantic schemas) |
| `parallel_llm_based_workflow.py` | Fan-out to 3 parallel LLM evaluators (grammar/clarity/depth) joined by a summarizing node | No (Groq, structured output) |
| `iterative_workflow_llm.py` | Generator → evaluator → optimizer loop with a max-iteration cutoff (generate-critique-improve pattern) | No (Groq, structured output) |

These are ordered here by complexity, which is a reasonable way to read them
even though the files themselves don't share state or import each other.
The two non-LLM ones (`simple_bmi_workflow.py`,
`parallel_workflow_batsman_summary.py`) are useful specifically *because*
they isolate the graph mechanics (fan-out/fan-in, node wiring, `stream()`)
from any LLM behavior.

---

## `src/mcp/` — Model Context Protocol

Demonstrates giving a LangGraph agent tools via a real MCP server rather than
in-process Python functions.

- **`tools/calculator_mcp_server.py`** — a standalone `FastMCP` server
  exposing five arithmetic tools (`add`, `subtract`, `multiply`, `divide`,
  `modulo`) over stdio transport. Runs as its own process.
- **`simple_chatbot_mcp.py`** — a LangGraph agent that spawns the calculator
  server as a subprocess via `MultiServerMCPClient`, discovers its tools at
  runtime (`mcp_client.get_tools()`), binds them to a Groq model, and wires a
  standard `chat_node -> tools_condition -> tool_node` graph (LangGraph's
  built-in ReAct-style tool-calling loop). Demonstrates that MCP tools are a
  drop-in replacement for `@tool`-decorated Python functions from the
  graph's point of view.

## `src/weekend_projects/week1/` — deployable full-stack agent

The most "production-shaped" project in the repo: a real client/server split
with its own session model.

- **`chatbot_agent.py`** — the agent core. Builds a `create_react_agent`
  (Groq LLM + optional `TavilySearch` web tool), wraps it in
  `RunnableWithMessageHistory` for short-term memory, and layers a simple
  long-term memory (`long_term_store`, keeps the last 5 "notable" user
  messages per session) on top. Ships a long, carefully written system
  prompt covering search strategy, source prioritization, citation
  requirements, and uncertainty handling. `get_response_from_groq_agent()`
  takes a `session_id` so each conversation's history stays isolated.
- **`chatbot_api.py`** — a FastAPI wrapper exposing `POST /chat`. Validates
  the request with Pydantic models (`RequestState` includes `messages`,
  `allow_search`, and `session_id`) and delegates to `chatbot_agent.py`.
- **`chatbot_ui.py`** — a Streamlit frontend that generates a `session_id`
  per browser session (persisted in `st.session_state`, rotated on "Clear
  Chat History"), collects a web-search toggle, and calls the FastAPI
  backend over HTTP.

This is the only project in the repo with a genuine three-tier
separation (agent logic / API / UI) and the only one with per-session
isolation built in — a template worth reusing for future weekend projects.

---

## `tests/evaluations/` — RAG evaluation harness

A LangSmith-based evaluation suite for the `rag_pipeline_design.py` pipeline,
using four LLM-as-judge metrics rather than exact-match scoring:

1. **`upload_dataset.py`** — uploads `data/evaluation_dataset/rag_test_dataset.json`
   to LangSmith as a named dataset. Idempotent: re-running skips samples
   already present (matched by `external_id`), and reuses an existing
   dataset of the same name rather than duplicating it.
2. **`rag_target.py`** — the "system under test": a thin wrapper that calls
   `rag_pipeline_design(question)` and returns its `{answer,
   retrieved_contexts, question}` dict in the shape LangSmith expects.
3. **`rag_evaluator_retrieval_n_generation.py`** — orchestrates the run.
   Defines four judge evaluators, each with its own rubric and system
   prompt:
   - **Faithfulness** — does the answer stay grounded in retrieved context?
   - **Context relevance** — are the retrieved chunks actually on-topic?
   - **Answer correctness** — does the answer match the ground truth?
   - **Answer relevance** — does the answer address the question asked?

   Each judge call parses a `Score: <0-10>` / `Reason: <...>` response into a
   normalized 0–1 score, with a safe fallback if parsing fails. Running the
   file (`python rag_evaluator_retrieval_n_generation.py`) prints a summary
   table of average scores per metric.

**Run order**: ingest once (`rag_pipeline_design.py` or `ingestion.py`) →
`upload_dataset.py` → `rag_evaluator_retrieval_n_generation.py`.

This is the most rigorous part of the repo — separating retrieval quality
from generation quality, and correctness-vs-reference from
relevance-to-question, is a distinction a lot of RAG projects skip entirely.

---

## How the projects relate to each other

```
fundamentals/ingestion.py ──┐
                             ├─> shared Chroma vector store (data/vectorstore/)
fundamentals/retrieval.py ──┤        (via config.settings.DB_DIR)
combined_rag_chain.py ──────┘

rag_pipeline/rag_pipeline_design.py ──> its own ingestion path (chromadb client directly)
                                     ──> imported by tests/evaluations/rag_target.py

chatbot_langgraph/*.py ─── independent of the RAG scripts; pure chat + memory
langgraph_pipelines/*.py ── independent pattern demos; no shared state
mcp/*.py ─────────────────── independent; demonstrates tool-calling via MCP
weekend_projects/week1/*.py ─ independent; full-stack agent with web search
```

Nothing outside `fundamentals/` and `tests/evaluations/` depends on the
vector store — the LangGraph pattern demos, the MCP demo, and the weekend
project are all self-contained and can be run without ever running
`ingestion.py`.

---

## Known gaps to be aware of when reading/running this repo

- `requirements.txt` predates several imports actually used in `src/`
  (`langchain-groq`, `langchain-huggingface`, `langchain-chroma`,
  `langgraph-checkpoint-sqlite`, `langchain-mcp-adapters`, `langsmith`,
  `mcp`, `langchain-tavily` are all imported somewhere but not listed).
- `generate.py` calls Ollama with `config.settings.MODEL`, which is set to a
  Groq model id (`llama-3.3-70b-versatile`) — that constant is shared across
  both the Ollama and Groq code paths even though the two need different
  model names.
- `rag_pipeline_design.py` hardcodes `"sentence-transformers/all-mpnet-base-v2"`
  instead of importing `EMBEDDING_MODEL` from `config.settings`
  (`all-MiniLM-L6-v2`) — if it's ever pointed at the same `DB_DIR`/
  `COLLECTION_NAME` as the `fundamentals/` scripts, the two embedding
  dimensions would collide in one Chroma collection.
- Most scripts add the project root to `sys.path` manually
  (`sys.path.append(...)`) instead of the repo being an installable package;
  a `pyproject.toml` + `pip install -e .` would remove this across the board.

None of the above block reading or understanding the code — they're the
first things worth fixing if this repo moves from "learning project" to
"something other people run."