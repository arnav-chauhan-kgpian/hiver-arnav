"""CLI entry point for the email AI suggested-response system.

Usage:
    python run.py
"""

from src.pipeline import run


def main() -> None:
    """Run the end-to-end pipeline."""
    run()


if __name__ == "__main__":
    main()
