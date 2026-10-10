"""Non-blocking diagnostic checks for verifying numerical grounding in generated answers."""

import re
from collections.abc import Sequence

from annual_report_rag.retrieval.retriever import RetrievedChunk

_CITATION_MARKER_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_CANDIDATE_NUMBER_RE = re.compile(
    r"(?<!\d)(?:\d+(?:,\s*\d+)+(?:\.\d+)?|\d+\.\d+|\d{3,})(?!\d)"
)


def _strip_commas_and_spaces(text: str) -> str:
    """Removes all commas and whitespace characters from a string."""
    return re.sub(r"[,\s]+", "", text)


def extract_answer_numbers(answer: str) -> list[str]:
    """Extracts candidate numbers (>=3 digits, or containing a comma or decimal point) excluding [n] citations."""
    if not answer:
        return []
    cleaned = _CITATION_MARKER_RE.sub(" ", answer)
    matches = _CANDIDATE_NUMBER_RE.findall(cleaned)
    return [m.strip() for m in matches if m.strip()]


def find_unsupported_numbers(
    answer: str,
    cited_chunks: Sequence[RetrievedChunk],
) -> list[str]:
    """Identifies numbers in the answer that do not appear in any cited passage.

    Extracts numbers from the answer that have at least 3 digits or contain a
    decimal point or comma (ignoring bracketed citation markers like [1]), and
    checks whether each appears in at least one cited passage after removing
    commas and whitespace from both the number and the passage text.

    Diagnostic Limits (non-blocking):
    1. Small integers (1 or 2 digits without decimals/commas, e.g. '5' or '31')
       are intentionally ignored to avoid false positives on list counts or dates.
    2. Substring matching after stripping commas/spaces verifies character presence
       only; it cannot verify semantic attachment (whether the number refers to the
       same metric, unit, or fiscal year).
    3. User-requested derived arithmetic (e.g. sums, differences, or ratios
       calculated by the LLM) or rounded values (e.g. '25%' vs '25.0%') will be
       flagged as unsupported even when the calculation is valid.
    """
    candidates = extract_answer_numbers(answer)
    if not candidates:
        return []

    normalized_passages = [_strip_commas_and_spaces(c.text) for c in cited_chunks]

    unsupported: list[str] = []
    seen: set[str] = set()
    for num_str in candidates:
        norm_num = _strip_commas_and_spaces(num_str)
        if not norm_num:
            continue
        is_supported = any(norm_num in passage for passage in normalized_passages)
        if not is_supported and num_str not in seen:
            seen.add(num_str)
            unsupported.append(num_str)

    return unsupported
