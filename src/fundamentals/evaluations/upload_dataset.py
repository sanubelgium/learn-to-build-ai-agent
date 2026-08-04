"""
upload_dataset.py
-----------------
Loads rag_test_dataset.json and uploads it to LangSmith as a named dataset.

Run once (or re-run to update):
    python src/fundamentals/evaluations/upload_dataset.py

The script is idempotent: if a dataset with the same name already exists it
will be reused; individual examples are checked by their external_id so
duplicates are never created.
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── project root on sys.path so `config` is importable ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
from config.settings import DATASET_JSON, LANGSMITH_DATASET_NAME
from langsmith import Client

def upload_dataset(json_path: Path = DATASET_JSON,
                   dataset_name: str = LANGSMITH_DATASET_NAME) -> str:
    """
    Upload (or update) the evaluation dataset to LangSmith.
    Returns the LangSmith dataset ID.
    """
    client = Client()

    #Load local JSON
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = raw["samples"]
    metadata = raw["metadata"]
    print(f"[upload] Loaded {len(samples)} samples from {json_path.name}")

    #Get or create dataset
    existing = [d for d in client.list_datasets() if d.name == dataset_name]
    if existing:
        dataset = existing[0]
        print(f"[upload] Reusing existing LangSmith dataset '{dataset_name}' (id={dataset.id})")
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=metadata.get("description", "RAG evaluation dataset"),
        )
        print(f"[upload] Created new LangSmith dataset '{dataset_name}' (id={dataset.id})")

    #Fetch already-uploaded example external_ids 
    existing_ext_ids = {
        ex.metadata.get("external_id")
        for ex in client.list_examples(dataset_id=dataset.id)
        if ex.metadata
    }

    #Upload examples
    uploaded = skipped = 0
    for sample in samples:
        ext_id = sample["id"]
        if ext_id in existing_ext_ids:
            skipped += 1
            continue

        # LangSmith example format:
        #   inputs  → what the target function receives
        #   outputs → ground truth (reference answer)
        client.create_example(
            dataset_id=dataset.id,
            inputs={
                "question": sample["question"],
                "category": sample["category"],
                "difficulty": sample["difficulty"],
            },
            outputs={
                "ground_truth": sample["ground_truth"],
            },
            metadata={
                "external_id": ext_id,
                "category": sample["category"],
                "difficulty": sample["difficulty"],
                "notes": sample.get("notes", ""),
            },
        )
        uploaded += 1

    print(f"[upload] Done — uploaded: {uploaded}, skipped (already exist): {skipped}")
    return str(dataset.id)


if __name__ == "__main__":
    dataset_id = upload_dataset()
    print(f"[upload] View dataset: https://smith.langchain.com/datasets/{dataset_id}")
