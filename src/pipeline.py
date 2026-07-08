"""Pipeline orchestration.

Wires together the five stages of the system end-to-end:

    1. Dataset generation
    2. Retrieval index building
    3. Response generation (retrieval-augmented, via Groq)
    4. Evaluation (composite LLM + deterministic metric)
    5. Reporting (per-example CSV + dataset summary statistics)

Stages requiring the LLM (3-5) degrade gracefully when ``GROQ_API_KEY``
is not set: the pipeline still runs stages 1-2 and exits cleanly.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from src.utils import OUTPUTS_DIR, PROCESSED_DIR

# A representative incoming email used to demo the single-example stages.
DEMO_QUERY = (
    "Hi, I ordered a laptop last week but it still hasn't arrived. "
    "Can you tell me when it will be delivered?"
)

TEST_PATH = PROCESSED_DIR / "test.jsonl"
EVAL_CSV_PATH = OUTPUTS_DIR / "evaluation_results.csv"

# Optional cap on how many test examples to evaluate (for quick runs).
# Set the EVAL_TEST_LIMIT env var to a positive integer; unset = full set.
_TEST_LIMIT_ENV = "EVAL_TEST_LIMIT"

EVAL_CSV_FIELDS = [
    "customer_email",
    "reference_reply",
    "generated_reply",
    "overall_score",
    "relevance",
    "completeness",
    "groundedness",
    "professional_tone",
    "actionability",
]


# --------------------------------------------------------------------------- #
# Stage 2: retrieval demo
# --------------------------------------------------------------------------- #

def _print_retrieved(results) -> None:
    print()
    print("Retrieved examples:")
    print()
    for rank, ex in enumerate(results, start=1):
        snippet = ex.customer_email.replace("\n", " ")[:120]
        print(f"{rank}. [{ex.category}] similarity={ex.similarity_score:.3f}")
        print(f"   {snippet}...")
        print()


# --------------------------------------------------------------------------- #
# Stage 4: single-example evaluation printout
# --------------------------------------------------------------------------- #

def _print_evaluation(result: dict[str, Any]) -> None:
    dims = result["dimension_scores"]
    reasoning = result["reasoning"]

    print()
    print("==========================")
    print("Evaluation")
    print("==========================")
    print()
    print(f"Overall Score: {result['overall_score']}/100")
    print()
    print("Dimension Scores")
    print()
    print(f"Relevance:         {dims['relevance']}/10")
    print(f"Completeness:      {dims['completeness']}/10")
    print(f"Groundedness:      {dims['groundedness']}/10")
    print(f"Professional Tone: {dims['professional_tone']}/10")
    print(f"Actionability:     {dims['actionability']}/10")
    print()
    print("Reasoning")
    print()
    for dim in ("relevance", "completeness", "groundedness", "professional_tone", "actionability"):
        print(f"- {dim.replace('_', ' ').title()}: {reasoning.get(dim, '')}")
    checks = reasoning.get("deterministic_checks", [])
    if checks:
        print(f"- Deterministic checks (adjustment {reasoning.get('adjustment', 0.0):+}): "
              + "; ".join(checks))
    print()


# --------------------------------------------------------------------------- #
# Stage 5: dataset-level evaluation + reporting
# --------------------------------------------------------------------------- #

def _load_test_records() -> list[dict[str, Any]]:
    if not TEST_PATH.exists():
        return []
    with TEST_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _test_limit() -> int | None:
    raw = os.getenv(_TEST_LIMIT_ENV)
    if not raw:
        return None
    try:
        val = int(raw)
        return val if val > 0 else None
    except ValueError:
        return None


def _evaluate_test_dataset(engine, generator, evaluator) -> None:
    """Generate + evaluate replies across the test set, write CSV, print stats."""
    from tqdm import tqdm

    from src.evaluate import summarize_scores

    records = _load_test_records()
    if not records:
        print("No test dataset found; skipping dataset-level evaluation.")
        return

    limit = _test_limit()
    if limit is not None:
        records = records[:limit]
        print(f"Evaluating {len(records)} test examples "
              f"(capped by {_TEST_LIMIT_ENV}={limit})...")
    else:
        print(f"Evaluating all {len(records)} test examples...")

    rows: list[dict[str, Any]] = []
    overall_scores: list[float] = []

    for rec in tqdm(records, desc="Evaluating test set"):
        customer_email = rec["customer_email"]
        reference_reply = rec["agent_reply"]
        try:
            retrieved = engine.retrieve(customer_email, k=5)
            generated_reply = generator.generate(customer_email, retrieved)
            result = evaluator.evaluate(customer_email, reference_reply, generated_reply)
        except Exception as exc:  # noqa: BLE001 - skip a bad example, keep going.
            tqdm.write(f"  Skipped one example: {exc}")
            continue

        dims = result["dimension_scores"]
        rows.append(
            {
                "customer_email": customer_email,
                "reference_reply": reference_reply,
                "generated_reply": generated_reply,
                "overall_score": result["overall_score"],
                "relevance": dims["relevance"],
                "completeness": dims["completeness"],
                "groundedness": dims["groundedness"],
                "professional_tone": dims["professional_tone"],
                "actionability": dims["actionability"],
            }
        )
        overall_scores.append(result["overall_score"])

    _write_eval_csv(rows)
    print(f"Saved per-example results: {EVAL_CSV_PATH} ({len(rows)} rows)")

    stats = summarize_scores(overall_scores)
    print()
    print("Summary Statistics (overall score, 0-100)")
    print()
    print(f"Average Score:      {stats['average']}")
    print(f"Median Score:       {stats['median']}")
    print(f"Minimum:            {stats['minimum']}")
    print(f"Maximum:            {stats['maximum']}")
    print(f"Standard Deviation: {stats['std']}")
    print()


def _write_eval_csv(rows: list[dict[str, Any]]) -> None:
    EVAL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _ensure_dataset() -> None:
    """Generate the dataset if the processed files are missing."""
    if TEST_PATH.exists() and (PROCESSED_DIR / "train.jsonl").exists():
        print("Dataset already present; reusing processed files.")
        return
    # Import here so `import pipeline` stays cheap.
    import importlib.util

    gen_path = Path(__file__).resolve().parents[1] / "data" / "generate_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_dataset", gen_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate_dataset()


def run() -> None:
    """Execute the full pipeline end-to-end."""
    # ---- Stage 1: dataset ------------------------------------------------- #
    print("Generating dataset...")
    _ensure_dataset()

    # ---- Stage 2: retrieval ---------------------------------------------- #
    print("Building retrieval index...")
    from src.retrieval import RetrievalEngine

    engine = RetrievalEngine()
    engine.build_index()
    results = engine.retrieve(DEMO_QUERY, k=5)
    _print_retrieved(results)

    # ---- Stages 3-5 need the LLM; skip gracefully without a key ---------- #
    print("Generating responses...")
    if not os.getenv("GROQ_API_KEY"):
        print("No GROQ_API_KEY found. Skipping response generation.")
        print("Evaluating responses...")
        print("No GROQ_API_KEY found. Skipping evaluation.")
        print("Done.")
        return

    from src.evaluate import EmailResponseEvaluator, EvaluationError
    from src.generator import GenerationError, ResponseGenerator

    try:
        generator = ResponseGenerator()
        reply = generator.generate(DEMO_QUERY, results)
    except GenerationError as exc:
        print(f"Response generation failed: {exc}")
        print("Evaluating responses...")
        print("Skipping evaluation (no generated response).")
        print("Done.")
        return

    print()
    print("==========================")
    print("Generated Response")
    print("==========================")
    print()
    print(reply)

    # ---- Stage 4: evaluate the single demo reply ------------------------- #
    print()
    print("Evaluating responses...")
    try:
        # Reuse the generator's Groq client for the judge.
        evaluator = EmailResponseEvaluator(client=generator.client)
        reference_reply = results[0].agent_reply
        single_eval = evaluator.evaluate(DEMO_QUERY, reference_reply, reply)
        _print_evaluation(single_eval)
    except EvaluationError as exc:
        print(f"Evaluation failed: {exc}")
        print("Done.")
        return

    # ---- Stage 5: dataset-level evaluation + report ---------------------- #
    _evaluate_test_dataset(engine, generator, evaluator)

    print("Done.")


if __name__ == "__main__":
    run()
