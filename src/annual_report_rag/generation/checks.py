"""Non-blocking diagnostic checks for verifying numerical grounding in generated answers."""

import re
from collections.abc import Sequence

from annual_report_rag.retrieval.retriever import RetrievedChunk

_CITATION_MARKER_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_CANDIDATE_NUMBER_RE = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+\.\d+|\d{3,})(?!\d)"
)


_YEAR_4DIGIT_RE = re.compile(r"^(?:19|20)\d{2}$")


def _strip_commas_and_spaces(text: str) -> str:
    """Removes all commas and whitespace characters from a string."""
    return re.sub(r"[,\s]+", "", text)


def _is_fiscal_year_supported(norm_num: str, raw_passages: Sequence[str]) -> bool:
    """Returns True if norm_num is a 4-digit year (e.g. '2026') matched by '2025-26' or 'FY26' in any passage."""
    if not _YEAR_4DIGIT_RE.match(norm_num):
        return False
    year_int = int(norm_num)
    prev_year = str(year_int - 1)
    short_yy = norm_num[2:]
    fy_pattern = re.compile(
        rf"(?:(?<!\d){prev_year}\s*[-–—/]\s*{short_yy}(?!\d)|\bfy\s*['’\-]?\s*{short_yy}(?!\d))",
        re.IGNORECASE,
    )
    return any(bool(fy_pattern.search(p)) for p in raw_passages)


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
    commas and whitespace from both the number and the passage text (or in
    fiscal-year form such as '2025-26' / 'FY26' for a 4-digit year like '2026').

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

    raw_passages = [c.text for c in cited_chunks]
    normalized_passages = [_strip_commas_and_spaces(p) for p in raw_passages]

    unsupported: list[str] = []
    seen: set[str] = set()
    for num_str in candidates:
        norm_num = _strip_commas_and_spaces(num_str)
        if not norm_num:
            continue
        is_supported = (
            any(norm_num in passage for passage in normalized_passages)
            or _is_fiscal_year_supported(norm_num, raw_passages)
        )
        if not is_supported and num_str not in seen:
            seen.add(num_str)
            unsupported.append(num_str)

    return unsupported
