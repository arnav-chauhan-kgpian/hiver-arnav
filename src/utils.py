"""Shared utilities.

Central definition of the project's directory layout. Paths are resolved
relative to this file so they work regardless of the current working
directory (portable across machines and when run from any folder).
"""

from __future__ import annotations

from pathlib import Path

# Project directory layout.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
