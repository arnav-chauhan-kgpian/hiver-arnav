"""Streamlit frontend for the AI Email Suggested Response System.

Interactive demo of the full pipeline — retrieval, generation, and the
composite evaluation — reusing the existing ``src`` modules unchanged:

    - RetrievalEngine        (src.retrieval)
    - ResponseGenerator      (src.generator)
    - EmailResponseEvaluator (src.evaluate)

Run locally with:

    streamlit run app.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import streamlit as st

from src.evaluate import DIMENSIONS, EmailResponseEvaluator, EvaluationError
from src.generator import DEFAULT_MODEL as GROQ_DEFAULT_MODEL
from src.generator import GenerationError, ResponseGenerator
from src.retrieval import DEFAULT_MODEL as EMBED_MODEL
from src.retrieval import RetrievalEngine
from src.utils import PROCESSED_DIR

TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
TEST_PATH = PROCESSED_DIR / "test.jsonl"
DATASET_PATH = PROCESSED_DIR / "email_dataset.jsonl"
REQUIRED_DATASET_FILES = [TRAIN_PATH, TEST_PATH, DATASET_PATH]

GENERATOR_SCRIPT = Path(__file__).resolve().parent / "data" / "generate_dataset.py"

DIMENSION_LABELS = {
    "relevance": "Relevance",
    "completeness": "Completeness",
    "groundedness": "Groundedness",
    "professional_tone": "Professional Tone",
    "actionability": "Actionability",
}

PREFILL_EMAIL = (
    "Hi,\n\n"
    "I placed an order for a coffee maker two weeks ago (order ORD-2024-583920) "
    "and it still hasn't arrived. The tracking page hasn't updated in over a week "
    "and still just says 'in transit'. I needed this before the weekend as a gift, "
    "and I'm starting to worry the package is lost. Could you check what's going on "
    "and give me a realistic delivery date?\n\n"
    "Thanks,\nMorgan"
)


# --------------------------------------------------------------------------- #
# API key resolution
# --------------------------------------------------------------------------- #

def get_server_key() -> str:
    """Return the server-configured Groq key, if any.

    Reads ``GROQ_API_KEY`` from the environment first (Streamlit Community
    Cloud injects configured secrets as environment variables), then falls
    back to ``st.secrets`` for local ``.streamlit/secrets.toml`` setups.
    The key itself is never logged or displayed.
    """
    key = os.getenv("GROQ_API_KEY", "")
    if key:
        return key
    try:
        return str(st.secrets.get("GROQ_API_KEY", "") or "")
    except Exception:  # noqa: BLE001 - no secrets file configured.
        return ""


# --------------------------------------------------------------------------- #
# Dataset bootstrap (first-launch auto-generation)
# --------------------------------------------------------------------------- #

def _run_dataset_generator() -> None:
    """Invoke the existing dataset generator without importing it as a package.

    ``data/generate_dataset.py`` is a standalone script (not part of the ``src``
    package), so it is loaded by file path and its ``generate_dataset()`` is
    called — no generation logic is duplicated here.
    """
    spec = importlib.util.spec_from_file_location("generate_dataset", GENERATOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate_dataset()


def ensure_dataset() -> tuple[bool, str | None]:
    """Ensure the processed dataset exists, generating it on first launch.

    Returns:
        ``(ready, error)`` where ``ready`` is True when all required files are
        present, and ``error`` is a message if automatic generation failed.
    """
    if all(p.exists() for p in REQUIRED_DATASET_FILES):
        return True, None

    try:
        with st.spinner("Generating dataset for first launch..."):
            _run_dataset_generator()
    except Exception as exc:  # noqa: BLE001 - surface any failure cleanly.
        return False, str(exc)

    missing = [p.name for p in REQUIRED_DATASET_FILES if not p.exists()]
    if missing:
        return False, f"Generation finished but these files are still missing: {', '.join(missing)}."

    st.success("Dataset generated successfully.")
    return True, None


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading embedding model and building FAISS index...")
def load_retrieval_engine() -> RetrievalEngine:
    """Build (once) and cache the retrieval engine.

    Caching the engine object caches the sentence-transformer model and the
    FAISS index it holds, so they are loaded a single time per session.
    """
    engine = RetrievalEngine()
    engine.build_index()
    return engine


# --------------------------------------------------------------------------- #
# Pipeline execution (reuses existing modules; no logic duplicated)
# --------------------------------------------------------------------------- #

def run_pipeline(
    email: str,
    api_key: str,
    model: str,
    temperature: float,
    k: int,
) -> dict:
    """Run retrieval -> generation -> evaluation for one email.

    Returns a dict of results stored in session state. Raises the
    underlying module errors so the caller can render them cleanly.
    """
    # The generator reads GROQ_API_KEY from the environment; set it here so
    # we don't modify the existing module interfaces.
    os.environ["GROQ_API_KEY"] = api_key

    engine = load_retrieval_engine()
    retrieved = engine.retrieve(email, k=k)

    generator = ResponseGenerator(model=model, temperature=temperature)
    t0 = time.perf_counter()
    reply = generator.generate(email, retrieved)
    gen_latency = time.perf_counter() - t0

    # Reuse the generator's Groq client for the judge (same as the pipeline).
    evaluator = EmailResponseEvaluator(client=generator.client, model=model)
    reference_reply = retrieved[0].agent_reply if retrieved else ""
    t1 = time.perf_counter()
    evaluation = evaluator.evaluate(email, reference_reply, reply)
    eval_latency = time.perf_counter() - t1

    return {
        "email": email,
        "retrieved": retrieved,
        "reply": reply,
        "evaluation": evaluation,
        "model": model,
        "temperature": temperature,
        "k": k,
        "gen_latency": gen_latency,
        "eval_latency": eval_latency,
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_retrieved(retrieved) -> None:
    st.subheader("Retrieved Similar Emails")
    if not retrieved:
        st.info("No similar emails retrieved.")
        return
    for i, ex in enumerate(retrieved, start=1):
        header = f"#{i} · {ex.category} · similarity {ex.similarity_score:.3f}"
        with st.expander(header, expanded=(i == 1)):
            st.caption(f"Category: **{ex.category}**  |  Similarity: **{ex.similarity_score:.3f}**")
            st.markdown("**Customer email**")
            st.text(ex.customer_email)
            st.markdown("**Historical agent reply**")
            st.text(ex.agent_reply)


def render_generated(reply: str) -> None:
    st.subheader("Generated Reply")
    st.code(reply, language="text")


def render_evaluation(evaluation: dict) -> None:
    st.subheader("Evaluation")
    dims = evaluation["dimension_scores"]
    reasoning = evaluation["reasoning"]

    st.metric("Overall Score", f"{evaluation['overall_score']} / 100")

    st.markdown("**Dimension scores**")
    for dim in DIMENSIONS:
        score = float(dims[dim])
        st.markdown(f"{DIMENSION_LABELS[dim]} — **{score:.1f} / 10**")
        st.progress(min(1.0, max(0.0, score / 10.0)))

    st.markdown("**Reasoning**")
    for dim in DIMENSIONS:
        just = reasoning.get(dim, "")
        if just:
            st.markdown(f"- **{DIMENSION_LABELS[dim]}:** {just}")
    checks = reasoning.get("deterministic_checks", [])
    if checks:
        adj = reasoning.get("adjustment", 0.0)
        st.markdown(f"- **Deterministic checks** (adjustment {adj:+}): " + "; ".join(checks))


def render_pipeline_details(run: dict) -> None:
    with st.expander("Pipeline Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Embedding model:** `{EMBED_MODEL}`")
            st.markdown("**Retrieval index:** `FAISS IndexFlatIP` (cosine)")
            st.markdown(f"**Retrieved examples:** {run['k']}")
        with col2:
            st.markdown(f"**LLM model:** `{run['model']}`")
            st.markdown(f"**Generation latency:** {run['gen_latency']:.2f} s")
            st.markdown(f"**Evaluation latency:** {run['eval_latency']:.2f} s")


def render_downloads(run: dict) -> None:
    st.subheader("Download")
    report = {
        "customer_email": run["email"],
        "generated_reply": run["reply"],
        "evaluation": run["evaluation"],
        "pipeline": {
            "embedding_model": EMBED_MODEL,
            "retrieval_index": "FAISS IndexFlatIP (cosine)",
            "retrieved_examples": run["k"],
            "llm_model": run["model"],
            "temperature": run["temperature"],
            "generation_latency_s": round(run["gen_latency"], 3),
            "evaluation_latency_s": round(run["eval_latency"], 3),
        },
    }
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download generated reply (.txt)",
            data=run["reply"],
            file_name="generated_reply.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download evaluation report (.json)",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name="evaluation_report.json",
            mime="application/json",
            use_container_width=True,
        )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(
        page_title="AI Email Suggested Response System",
        page_icon="📧",
        layout="wide",
    )

    st.title("AI Email Suggested Response System")
    st.write(
        "Generate customer-support email replies with retrieval-augmented "
        "generation (RAG) and score their quality with a composite "
        "evaluation framework (LLM judge + deterministic checks)."
    )

    # ---- Sidebar ------------------------------------------------------- #
    with st.sidebar:
        st.header("Settings")
        user_key = st.text_input(
            "Groq API Key (optional)",
            value="",
            type="password",
            placeholder="Leave blank to use the server configuration",
            help="Optional. Leave blank to use the server-configured key. "
            "A key entered here is used only for this session and is never saved.",
        )

        # Priority: user-entered key > server-configured key > none.
        server_key = get_server_key()
        if user_key.strip():
            effective_key = user_key.strip()
        elif server_key:
            effective_key = server_key
            st.success("Using server-configured API key.")
        else:
            effective_key = ""

        model = st.text_input(
            "Groq Model",
            value=os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL),
        )
        k = st.slider("Number of retrieved examples", min_value=1, max_value=5, value=3)
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05
        )
        generate = st.button("Generate", type="primary", use_container_width=True)
        st.caption(
            "Note: generation uses up to the top-3 retrieved examples as prompt "
            "context; the slider also controls how many are displayed."
        )

    # ---- Preconditions: dataset auto-generates on first launch --------- #
    dataset_ready, dataset_error = ensure_dataset()
    if not dataset_ready:
        st.error(
            "Could not prepare the dataset automatically: "
            f"{dataset_error} You can retry, or run `python data/generate_dataset.py` "
            "manually.",
            icon="⚠️",
        )

    # ---- Section 1: incoming email ------------------------------------- #
    st.subheader("Incoming Customer Email")
    email = st.text_area(
        "Customer email",
        value=PREFILL_EMAIL,
        height=220,
        label_visibility="collapsed",
    )

    # ---- Trigger ------------------------------------------------------- #
    if generate:
        if not dataset_ready:
            st.error("Cannot run: dataset is unavailable (see the error above).")
        elif not effective_key:
            st.warning(
                "No Groq API key available. Enter one in the sidebar, or configure "
                "GROQ_API_KEY on the server.",
                icon="🔑",
            )
        elif not email.strip():
            st.warning("Please enter a customer email.", icon="✉️")
        else:
            try:
                with st.spinner("Running retrieval → generation → evaluation..."):
                    st.session_state["run"] = run_pipeline(
                        email=email,
                        api_key=effective_key,
                        model=model.strip() or GROQ_DEFAULT_MODEL,
                        temperature=temperature,
                        k=k,
                    )
            except (GenerationError, EvaluationError) as exc:
                st.error(f"Pipeline failed: {exc}")
            except Exception as exc:  # noqa: BLE001 - present any failure cleanly.
                st.error(f"Unexpected error: {exc}")

    # ---- Results ------------------------------------------------------- #
    run = st.session_state.get("run")
    if run:
        st.divider()
        render_retrieved(run["retrieved"])
        st.divider()
        render_generated(run["reply"])
        st.divider()
        render_evaluation(run["evaluation"])
        st.divider()
        render_pipeline_details(run)
        st.divider()
        render_downloads(run)
    else:
        st.info("Configure the sidebar and click **Generate** to run the pipeline.")


if __name__ == "__main__":
    main()
