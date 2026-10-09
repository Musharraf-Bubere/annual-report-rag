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


def compute_number_split_count(text: str) -> int:
    """Counts runs where single digits are separated by single spaces (e.g., '2 0 2 6')."""
    if not text:
        return 0
    return len(re.findall(r"\b\d(?: \d)+\b", text))


def assess_page_quality(
    source: str,
    page: int,
    text: str,
) -> PageQualityMetrics:
    """Evaluates text quality metrics for a page, identifying formatting and encoding defects."""
    return PageQualityMetrics(
        source=source,
        page=page,
        char_count=len(text),
        single_letter_ratio=round(compute_single_letter_ratio(text), 4),
        odd_char_ratio=round(compute_odd_char_ratio(text), 4),
        odd_char_count=compute_odd_char_count(text),
        number_split_count=compute_number_split_count(text),
    )
