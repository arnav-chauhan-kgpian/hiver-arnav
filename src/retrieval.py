"""Retrieval stage.

Given an incoming customer email, retrieve the most similar historical
emails from the training set. Retrieved examples are later used as
few-shot grounding for the response generator.

Approach:
    - Embed text with sentence-transformers (all-MiniLM-L6-v2).
    - L2-normalize embeddings and index them with a FAISS
      ``IndexFlatIP`` so inner product == cosine similarity.
    - Cache both the embeddings (``outputs/train_embeddings.npy``) and
      the FAISS index (``outputs/faiss.index``) to avoid recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm

from src.utils import OUTPUTS_DIR, PROCESSED_DIR

# Default configuration.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
EMBEDDINGS_CACHE = OUTPUTS_DIR / "train_embeddings.npy"
INDEX_CACHE = OUTPUTS_DIR / "faiss.index"


@dataclass
class RetrievedExample:
    """A single retrieval hit.

    Attributes:
        customer_email: The retrieved historical customer email.
        agent_reply: The reference agent reply for that email.
        category: The support category of the retrieved example.
        similarity_score: Cosine similarity against the query.
    """

    customer_email: str
    agent_reply: str
    category: str
    similarity_score: float


class RetrievalEngine:
    """Semantic retrieval over historical support emails.

    Embeds the training emails, indexes them with FAISS, and answers
    nearest-neighbour queries by cosine similarity.
    """

    def __init__(
        self,
        train_path: str | Path = TRAIN_PATH,
        model_name: str = DEFAULT_MODEL,
        embeddings_cache: str | Path = EMBEDDINGS_CACHE,
        index_cache: str | Path = INDEX_CACHE,
    ) -> None:
        """Initialise the engine.

        Args:
            train_path: Path to the ``train.jsonl`` dataset.
            model_name: sentence-transformers model identifier.
            embeddings_cache: Where to cache/reuse embeddings (``.npy``).
            index_cache: Where to cache/reuse the FAISS index.
        """
        self.train_path = Path(train_path)
        self.model_name = model_name
        self.embeddings_cache = Path(embeddings_cache)
        self.index_cache = Path(index_cache)

        self._model: Any | None = None
        self.records: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.index: faiss.Index | None = None

    # ---------------------------------------------------------------- #
    # Lazy model loading
    # ---------------------------------------------------------------- #
    @property
    def model(self) -> Any:
        """Lazily load the sentence-transformers model on first use."""
        if self._model is None:
            # Imported lazily so simply importing this module stays cheap.
            from sentence_transformers import SentenceTransformer

            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ---------------------------------------------------------------- #
    # Data + embeddings
    # ---------------------------------------------------------------- #
    def _load_records(self) -> list[dict[str, Any]]:
        """Load the training records from the JSONL dataset."""
        import json

        if not self.train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {self.train_path}. "
                "Run data/generate_dataset.py first."
            )
        with self.train_path.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if not records:
            raise ValueError(f"No records found in {self.train_path}.")
        return records

    def _embed(self, texts: list[str], desc: str) -> np.ndarray:
        """Embed and L2-normalize a list of texts (with a progress bar)."""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        # Normalize in place so inner product == cosine similarity.
        faiss.normalize_L2(embeddings)
        return embeddings

    def _compute_embeddings(self) -> np.ndarray:
        """Compute training embeddings, reusing the cache when available."""
        if self.embeddings_cache.exists():
            print(f"Reusing cached embeddings: {self.embeddings_cache}")
            return np.load(self.embeddings_cache).astype("float32")

        texts = [r["customer_email"] for r in tqdm(self.records, desc="Preparing texts")]
        embeddings = self._embed(texts, desc="Embedding train emails")
        self.embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_cache, embeddings)
        print(f"Saved embeddings cache: {self.embeddings_cache}")
        return embeddings

    # ---------------------------------------------------------------- #
    # Index lifecycle
    # ---------------------------------------------------------------- #
    def build_index(self) -> None:
        """Build (or load from cache) the FAISS index over the train set.

        Loads the records and embeddings (reusing caches where possible),
        then either loads the cached FAISS index or builds a fresh
        ``IndexFlatIP`` and persists it.
        """
        self.records = self._load_records()
        self.embeddings = self._compute_embeddings()

        if self.index_cache.exists():
            print(f"Reusing cached FAISS index: {self.index_cache}")
            self.index = faiss.read_index(str(self.index_cache))
            return

        dim = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self.embeddings)
        self.index = index

        self.index_cache.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_cache))
        print(f"Built and saved FAISS index: {self.index_cache}")

    def save_index(self, path: str | Path) -> None:
        """Persist the FAISS index to an explicit path.

        Args:
            path: Destination path for the FAISS index.
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def load_index(self, path: str | Path) -> None:
        """Load a FAISS index and its records from disk.

        Records are still loaded from ``train_path`` so retrieved hits can
        be mapped back to their email/reply/category.

        Args:
            path: Path to a saved FAISS index.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index not found at {path}.")
        self.index = faiss.read_index(str(path))
        if not self.records:
            self.records = self._load_records()

    # ---------------------------------------------------------------- #
    # Query
    # ---------------------------------------------------------------- #
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedExample]:
        """Retrieve the top-k most similar historical emails.

        Args:
            query: The incoming customer email text.
            k: Number of examples to return.

        Returns:
            A list of ``RetrievedExample`` sorted by descending
            similarity.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if not self.records:
            self.records = self._load_records()

        query_vec = self._embed([query], desc="Embedding query")
        k = min(k, len(self.records))
        scores, indices = self.index.search(query_vec, k)

        results: list[RetrievedExample] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS pads with -1 when fewer than k hits exist.
                continue
            rec = self.records[int(idx)]
            results.append(
                RetrievedExample(
                    customer_email=rec["customer_email"],
                    agent_reply=rec["agent_reply"],
                    category=rec["category"],
                    similarity_score=float(score),
                )
            )
        # FAISS already returns descending IP order; sort defensively.
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results
