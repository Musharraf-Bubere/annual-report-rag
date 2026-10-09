"""Unit tests for text quality metrics evaluation."""

from annual_report_rag.ingestion.quality import (
    assess_page_quality,
    compute_number_split_count,
    compute_odd_char_count,
    compute_odd_char_ratio,
    compute_single_letter_ratio,
)


def test_clean_text_metrics() -> None:
    """Verifies that well-formed English financial text produces low anomaly ratios."""
    clean_text = (
        "Consolidated revenue increased by 15.4% to ₹ 267,021 crore for FY26. "
        "Operating margins remained stable across European delivery centers."
    )
    metrics = assess_page_quality("test_doc", 1, clean_text)

    assert metrics.char_count == len(clean_text)
    assert metrics.single_letter_ratio < 0.05  # 'a', 'I' only
    assert metrics.odd_char_ratio == 0.0  # Rupee and standard punctuation are accepted
    assert metrics.odd_char_count == 0
    assert metrics.number_split_count == 0


def test_spaced_out_letters_detected() -> None:
    """Verifies that text with fragmented single-character spacing triggers high single-letter ratio."""
    spaced_text = "i n th i s b ack d rop o f technol og y sh i f ts"
    ratio = compute_single_letter_ratio(spaced_text)

    # Over 50% of the alphabetic tokens are single characters
    assert ratio > 0.50


def test_odd_corrupted_characters_detected() -> None:
    """Verifies that encoding glyph defects and non-standard symbols trigger high odd-character ratio and count."""
    garbled_text = "ĐonƐolŝdated ƌevenue oĨ ₹ 26ϳ,021 Đƌoƌe"
    ratio = compute_odd_char_ratio(garbled_text)
    count = compute_odd_char_count(garbled_text)

    # Multiple corrupted glyphs outside ASCII / standard symbol set
    assert ratio > 0.15
    assert count == 9


def test_clean_numbers_not_counted_as_split() -> None:
    """Verifies that standard formatted numbers are never counted as split numbers."""
    clean_sample = (
        "Revenues reached 2026 targets with 44,000 employees and 584,519 customers. "
        "Growth of 12.5% across 10 20 30 business divisions, page 1 of 2."
    )
    count = compute_number_split_count(clean_sample)
    assert count == 0


def test_split_numbers_detected() -> None:
    """Verifies that sequences of single digits separated by spaces are correctly detected."""
    split_sample = (
        "In fiscal 2 0 2 6, headcount was 44, 0 0 0 + and shares rose to 1 9 9."
    )
    count = compute_number_split_count(split_sample)
    assert count == 3


def test_page_8_flagged_by_odd_char_count_alone() -> None:
    """Verifies that a long text with >= 15 odd characters flags odd_char_count even if ratio is low."""
    filler = "Standard corporate report text with normal words. " * 150
    garbled_heading = "ĐonƐolŝdated ƌevenue oĨ ₹ 26ϳ,021 Đƌoƌe ǁŝth ϴ Ő ŵ ͘ й ͬ " * 2
    combined = filler + garbled_heading

    count = compute_odd_char_count(combined)
    assert count >= 15
    metrics = assess_page_quality("tcs_mock", 8, combined)
    assert metrics.odd_char_count >= 15


def test_empty_text_metrics() -> None:
    """Verifies that empty string inputs return zeros without raising division errors."""
    metrics = assess_page_quality("empty", 1, "")

    assert metrics.char_count == 0
    assert metrics.single_letter_ratio == 0.0
    assert metrics.odd_char_ratio == 0.0
    assert metrics.odd_char_count == 0
    assert metrics.number_split_count == 0
