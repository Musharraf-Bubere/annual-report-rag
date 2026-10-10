"""Source router for detecting target company reports from user questions."""

import re

from annual_report_rag.config import Settings


def _alias_matches(question: str, alias: str) -> bool:
    """Checks if an alias appears as a case-insensitive whole word or phrase in the question."""
    cleaned_alias = alias.strip()
    if not cleaned_alias:
        return False
    parts = [re.escape(part) for part in cleaned_alias.split()]
    pattern = r"\b" + r"\s+".join(parts) + r"\b"
    return bool(re.search(pattern, question, flags=re.IGNORECASE))


def detect_sources(question: str, settings: Settings) -> list[str]:
    """Detects matching source stems from a user question using whole-word alias matching.

    Args:
        question: The user's natural language question.
        settings: Application settings containing the sources alias registry.

    Returns:
        list[str]: Matched source stems (empty list if no alias matches, meaning search all).
    """
    if not question or not settings.sources:
        return []

    matched_sources: list[str] = []
    for source_stem, metadata in settings.sources.items():
        if any(_alias_matches(question, alias) for alias in metadata.aliases):
            matched_sources.append(source_stem)

    return matched_sources
