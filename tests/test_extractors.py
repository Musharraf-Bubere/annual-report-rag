"""Unit tests for extractor helpers and word rejoining logic."""

import sys
from pathlib import Path

# Add project root to path so scripts can be imported cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_extractors import rejoin_words


def test_rejoin_words_empty() -> None:
    """Verifies that an empty list of words returns an empty string."""
    assert rejoin_words([]) == ""


def test_rejoin_words_tight_gap_merges() -> None:
    """Verifies that word fragments with horizontal gap <= tolerance are merged into one token."""
    # (x0, y0, x1, y1, word, block_no, line_no, word_no)
    words = [
        (10.0, 10.0, 20.0, 20.0, "Em", 0, 0, 0),
        (21.0, 10.0, 40.0, 20.0, "ployees", 0, 0, 1),  # gap is 21.0 - 20.0 = 1.0 <= 2.0
    ]
    result = rejoin_words(words, gap_tolerance=2.0)
    assert result == "Employees"


def test_rejoin_words_wide_gap_preserves_space() -> None:
    """Verifies that distinct words with horizontal gap > tolerance keep their separating space."""
    words = [
        (10.0, 10.0, 30.0, 20.0, "Annual", 0, 0, 0),
        (35.0, 10.0, 60.0, 20.0, "Report", 0, 0, 1),  # gap is 35.0 - 30.0 = 5.0 > 2.0
    ]
    result = rejoin_words(words, gap_tolerance=2.0)
    assert result == "Annual Report"


def test_rejoin_words_line_breaks() -> None:
    """Verifies that words on separate lines or blocks are separated by newlines."""
    words = [
        (10.0, 10.0, 30.0, 20.0, "Line1", 0, 0, 0),
        (10.0, 25.0, 30.0, 35.0, "Line2", 0, 1, 0),
    ]
    result = rejoin_words(words, gap_tolerance=2.0)
    assert result == "Line1\nLine2"
