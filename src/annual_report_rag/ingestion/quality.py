"""Text quality evaluation metrics for extracted document pages."""

import re
import string
from dataclasses import dataclass

# Normal character set: ASCII alphanumeric, punctuation, whitespace, and standard financial symbols
COMMON_EXTENDED_SYMBOLS = "₹€£¥¢–—−‘’“”‚„•·…°±×÷©®™"
NORMAL_CHARS: set[str] = set(
    string.ascii_letters
    + string.digits
    + string.punctuation
    + " \t\n\r"
    + COMMON_EXTENDED_SYMBOLS
)


@dataclass(frozen=True)
class PageQualityMetrics:
    """Holds calculated text quality metrics for an extracted page."""

    source: str
    page: int
    char_count: int
    single_letter_ratio: float
    odd_char_ratio: float
    odd_char_count: int
    number_split_count: int


def compute_single_letter_ratio(text: str) -> float:
    """Calculates the proportion of alphabetic tokens consisting of a single character."""
    tokens = re.findall(r"[A-Za-z]+", text)
    if not tokens:
        return 0.0
    single_letters = sum(1 for token in tokens if len(token) == 1)
    return single_letters / len(tokens)


def compute_odd_char_count(text: str) -> int:
    """Counts absolute number of characters outside the expected standard character set."""
    if not text:
        return 0
    return sum(1 for ch in text if ch not in NORMAL_CHARS)


def compute_odd_char_ratio(text: str) -> float:
    """Calculates the proportion of characters outside the expected standard character set."""
    if not text:
        return 0.0
    return compute_odd_char_count(text) / len(text)


SPLIT_NUMBER_PATTERN = re.compile(
    r"\b(?:\d{1,3},\s*)?\d(?: \d){3,}\b|\b\d{1,3},\s*\d(?: \d){2,}\b"
)


def compute_number_split_count(text: str) -> int:
    """Counts runs of split numbers requiring at least 4 single digits or comma-split groups."""
    if not text:
        return 0
    return len(SPLIT_NUMBER_PATTERN.findall(text))


def assess_page_quality(
    source: str,
    page: int,
    text: str,
    raw_text: str | None = None,
) -> PageQualityMetrics:
    """Evaluates text quality metrics for a page, identifying formatting and encoding defects."""
    eval_text = raw_text if raw_text is not None else text
    return PageQualityMetrics(
        source=source,
        page=page,
        char_count=len(text),
        single_letter_ratio=round(compute_single_letter_ratio(text), 4),
        odd_char_ratio=round(compute_odd_char_ratio(text), 4),
        odd_char_count=compute_odd_char_count(text),
        number_split_count=compute_number_split_count(eval_text),
    )
