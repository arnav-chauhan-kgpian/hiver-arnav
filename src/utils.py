"""Shared utilities.

Common helpers used across the pipeline: configuration, path handling,
I/O for the raw and processed datasets, and (optionally) an
API-compatible LLM client factory.

Nothing here is implemented yet — these are stubs to be filled in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Project directory layout. Resolved relative to this file so the paths
# work regardless of the current working directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load run configuration (model names, paths, thresholds, ...).

    Args:
        path: Optional path to a config file. Defaults to project settings.

    Returns:
        A dictionary of configuration values.
    """
    # TODO: read configuration from a file or environment variables.
    raise NotImplementedError


def get_llm_client() -> Any:
    """Create an OpenAI API-compatible client.

    Returns:
        A client instance configured from environment variables
        (e.g. ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``).
    """
    # TODO: construct and return an openai-compatible client.
    raise NotImplementedError


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file into a list of records.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        A list of parsed records.
    """
    # TODO: parse the jsonl file line by line.
    raise NotImplementedError


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of records to a JSON Lines file.

    Args:
        records: Records to serialize.
        path: Destination path.
    """
    # TODO: serialize each record as one json object per line.
    raise NotImplementedError


def ensure_dirs() -> None:
    """Create the standard project directories if they do not exist."""
    # TODO: create RAW_DIR, PROCESSED_DIR and OUTPUTS_DIR.
    raise NotImplementedError
