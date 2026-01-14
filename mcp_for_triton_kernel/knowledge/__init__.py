"""Knowledge base for Triton kernel development."""

from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent


def load_knowledge(filename: str) -> str:
    """Load a knowledge file as string."""
    filepath = KNOWLEDGE_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return f"Knowledge file not found: {filename}"

