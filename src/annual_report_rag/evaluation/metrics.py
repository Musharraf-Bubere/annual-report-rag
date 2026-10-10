"""Pure deterministic evaluation metrics for retrieval and answer quality."""

import math
import re
from collections.abc import Sequence
from typing import Literal, Protocol

from annual_report_rag.evaluation.golden import ExpectedPage, GoldenQuestion
from annual_report_rag.generation.prompts import NOT_FOUND_MESSAGE

OutcomeType = Literal[
    "CORRECT",
    "WRONG",
    "FALSE_REFUSAL",
    "CORRECT_REFUSAL",
    "ANSWERED_ANYWAY",
]

_CITATION_MARKER_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_NUMERIC_TOKEN_RE = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?")
_PURE_NUMBER_ENTRY_RE = re.compile(
    r"^\s*[₹$€£]?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*%?\s*$"
)


class HasSourcePageAndText(Protocol):
    """Protocol for retrieved chunk objects used by retrieval metrics."""

    @property
    def source(self) -> str: ...

    @property
    def page(self) -> int: ...

    @property
    def text(self) -> str: ...


class HasAnswer(Protocol):
    """Protocol for answer result objects holding an answer string."""

    @property
    def answer(self) -> str: ...


def strip_citation_markers(text: str) -> str:
    """Removes inline bracket citation markers such as [1], [1][3], or [1, 3]."""
    if not text:
        return ""
    return _CITATION_MARKER_RE.sub("", text)


def numeric_tokens(text: str) -> list[str]:
    """Extracts numeric tokens from text with commas removed (e.g. '5,84,519' -> '584519', '4.6%' -> '4.6')."""
    if not text:
        return []
    without_citations = strip_citation_markers(text)
    raw_matches = _NUMERIC_TOKEN_RE.findall(without_citations)
    return [match.replace(",", "") for match in raw_matches]


def _extract_single_number_candidate(candidate: str) -> str | None:
    """Returns the normalized number string if candidate represents a single number, else None."""
    match = _PURE_NUMBER_ENTRY_RE.match(candidate)
    if not match:
        return None
    return match.group(1).replace(",", "")


def _alternative_matches(
    stripped_text: str,
    text_num_tokens: set[str],
    alternative: str,
) -> bool:
    """Checks whether a single alternative (numeric or string) matches stripped_text."""
    norm_num = _extract_single_number_candidate(alternative)
    if norm_num is not None:
        return norm_num in text_num_tokens
    return alternative.lower() in stripped_text.lower()


def entry_matches_text(text: str, entry: str) -> bool:
    """Checks whether a must_contain/must_not_contain entry (supporting 'a|b' any-of) matches text."""
    stripped_text = strip_citation_markers(text)
    text_num_tokens = set(numeric_tokens(stripped_text))
    alternatives = [alt.strip() for alt in entry.split("|") if alt.strip()]
    if not alternatives:
        return False
    return any(
        _alternative_matches(stripped_text, text_num_tokens, alt)
        for alt in alternatives
    )


def satisfies_must_contain(text: str, must_contain: Sequence[str]) -> bool:
    """Returns True if every entry in must_contain is satisfied by text."""
    return all(entry_matches_text(text, entry) for entry in must_contain)


def has_must_not_contain(text: str, must_not_contain: Sequence[str]) -> bool:
    """Returns True if any entry in must_not_contain matches text."""
    return any(entry_matches_text(text, entry) for entry in must_not_contain)


def page_hit_at_k(
    retrieved: Sequence[HasSourcePageAndText],
    expected_pages: Sequence[ExpectedPage],
    k: int,
) -> bool:
    """Returns True if top-k retrieved chunks contain at least one expected page from EACH expected source."""
    if not expected_pages or k <= 0:
        return False

    expected_by_source: dict[str, set[int]] = {}
    for ep in expected_pages:
        expected_by_source.setdefault(ep.source, set()).add(ep.page)

    top_k = retrieved[:k]
    for src, pages in expected_by_source.items():
        if not any(c.source == src and c.page in pages for c in top_k):
            return False
    return True


def reciprocal_rank(
    retrieved: Sequence[HasSourcePageAndText],
    expected_pages: Sequence[ExpectedPage],
) -> float:
    """Computes the reciprocal rank (1 / rank) of the first expected page across all sources."""
    if not expected_pages or not retrieved:
        return 0.0

    expected_set = {(ep.source, ep.page) for ep in expected_pages}
    for rank, chunk in enumerate(retrieved, start=1):
        if (chunk.source, chunk.page) in expected_set:
            return 1.0 / float(rank)
    return 0.0


def text_hit_at_k(
    retrieved: Sequence[HasSourcePageAndText],
    must_contain: Sequence[str],
    k: int,
) -> bool:
    """Returns True if any retrieved chunk in top-k satisfies a must_contain entry."""
    if not must_contain or k <= 0:
        return False
    top_k = retrieved[:k]
    return any(
        entry_matches_text(chunk.text, entry)
        for chunk in top_k
        for entry in must_contain
    )


def outcome(
    question: GoldenQuestion,
    result: HasAnswer | str,
) -> OutcomeType:
    """Classifies the answer outcome for an answerable, unanswerable, or adversarial question."""
    raw_answer = result if isinstance(result, str) else result.answer
    cleaned = raw_answer.strip()
    stripped = strip_citation_markers(cleaned).strip()
    is_refusal = cleaned == NOT_FOUND_MESSAGE or stripped == NOT_FOUND_MESSAGE

    if question.answerable:
        if is_refusal:
            return "FALSE_REFUSAL"
        if question.must_not_contain and has_must_not_contain(
            stripped, question.must_not_contain
        ):
            return "WRONG"
        if satisfies_must_contain(stripped, question.must_contain):
            return "CORRECT"
        return "WRONG"

    # Unanswerable or adversarial
    if question.must_not_contain and has_must_not_contain(
        stripped, question.must_not_contain
    ):
        return "ANSWERED_ANYWAY"
    if is_refusal:
        return "CORRECT_REFUSAL"
    return "ANSWERED_ANYWAY"


def percentile(values: Sequence[float], q: float) -> float:
    """Computes the q-th percentile (0..100) of a numeric sequence using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 2)
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return round(sorted_vals[lower], 2)
    weight = pos - lower
    interpolated = sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight
    return round(interpolated, 2)


def p50(values: Sequence[float]) -> float:
    """Computes the 50th percentile (median) of a numeric sequence."""
    return percentile(values, 50.0)


def p95(values: Sequence[float]) -> float:
    """Computes the 95th percentile of a numeric sequence."""
    return percentile(values, 95.0)
