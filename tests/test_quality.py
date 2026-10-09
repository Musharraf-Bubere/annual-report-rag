"""Unit tests for text quality metrics evaluation."""

from annual_report_rag.ingestion.quality import (
    assess_page_quality,
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


def test_spaced_out_letters_detected() -> None:
    """Verifies that text with fragmented single-character spacing triggers high single-letter ratio."""
    spaced_text = "i n th i s b ack d rop o f technol og y sh i f ts"
    ratio = compute_single_letter_ratio(spaced_text)

    # Over 50% of the alphabetic tokens are single characters
    assert ratio > 0.50


def test_odd_corrupted_characters_detected() -> None:
    """Verifies that encoding glyph defects and non-standard symbols trigger high odd-character ratio."""
    garbled_text = "ĐonƐolŝdated ƌevenue oĨ ₹ 26ϳ,021 Đƌoƌe"
    ratio = compute_odd_char_ratio(garbled_text)

    # Multiple corrupted glyphs outside ASCII / standard symbol set
    assert ratio > 0.15


def test_empty_text_metrics() -> None:
    """Verifies that empty string inputs return zeros without raising division errors."""
    metrics = assess_page_quality("empty", 1, "")

    assert metrics.char_count == 0
    assert metrics.single_letter_ratio == 0.0
    assert metrics.odd_char_ratio == 0.0
