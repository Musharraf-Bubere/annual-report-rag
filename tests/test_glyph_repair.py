"""Unit tests for glyph repair, corrupted digit recovery, and text quality states."""

from annual_report_rag.ingestion.glyph_repair import repair_glyphs
from annual_report_rag.ingestion.loader import assess_and_filter_page


def test_repair_single_corrupted_digit() -> None:
    """Verifies that '26ϳ,021' becomes '267,021'."""
    raw = "26ϳ,021"
    repaired, count = repair_glyphs(raw)
    assert repaired == "267,021"
    assert count == 1


def test_repair_indian_currency_with_spaces_and_combining_comma() -> None:
    """Verifies that '₹Ϯ͕ ϲϳ͕ ϬϮ1' becomes '₹2,67,021'."""
    raw = "₹Ϯ͕ ϲϳ͕ ϬϮ1"
    repaired, count = repair_glyphs(raw)
    assert repaired == "₹2,67,021"
    assert count == 7


def test_repair_four_digit_year() -> None:
    """Verifies that 'ϮϬϮϲ' becomes '2026'."""
    raw = "ϮϬϮϲ"
    repaired, count = repair_glyphs(raw)
    assert repaired == "2026"
    assert count == 4


def test_repair_tibetan_rupee_and_digits() -> None:
    """Verifies that '༤ϭϬϳ͕ϮϰϬ' becomes '₹1,07,240'."""
    raw = "༤ϭϬϳ͕ϮϰϬ"
    repaired, count = repair_glyphs(raw)
    assert repaired == "₹1,07,240"
    assert count == 8


def test_mondelez_survives_untouched_by_repair_and_filter() -> None:
    """Verifies that legitimate name 'Mondelēz' survives untouched by repair and token filter."""
    text = "Mondelēz International declared dividend distribution."
    repaired, count = repair_glyphs(text)
    assert repaired == text
    assert count == 0

    filtered, quality, record = assess_and_filter_page(
        source="infosys-ar-26",
        page_num=5,
        cleaned=text,
        raw=text,
        odd_count_threshold=1,
        keep_tokens=("Mondelēz",),
    )
    assert quality == "clean"
    assert filtered == text
    assert "Mondelēz" in filtered
    assert record is None


def test_letter_corrupted_token_is_still_dropped() -> None:
    """Verifies that letter-corrupted tokens like 'ĐonƐolŝdated' are still dropped."""
    text = "Company reported ĐonƐolŝdated revenue figures."
    filtered, quality, record = assess_and_filter_page(
        source="tcs-ar-26",
        page_num=8,
        cleaned=text,
        raw=text,
        odd_count_threshold=1,
    )
    assert quality == "degraded"
    assert "ĐonƐolŝdated" not in filtered
    assert record is not None
    assert record["dropped_tokens"] == 1


def test_text_quality_state_labels() -> None:
    """Verifies all three text_quality states: 'clean', 'repaired', and 'degraded'."""
    # 1. Clean page: no repairs, no dropped tokens
    clean_text = "Clean financial disclosure with zero strange glyphs."
    out_clean, q_clean, rec_clean = assess_and_filter_page(
        source="tcs", page_num=1, cleaned=clean_text, raw=clean_text
    )
    assert q_clean == "clean"
    assert out_clean == clean_text
    assert rec_clean is None

    # 2. Repaired page: digits repaired, no tokens dropped
    repaired_input = "Fiscal year ϮϬϮϲ revenue reached 26ϳ,021 crore."
    out_rep, q_rep, rec_rep = assess_and_filter_page(
        source="tcs", page_num=2, cleaned=repaired_input, raw=repaired_input
    )
    assert q_rep == "repaired"
    assert out_rep == "Fiscal year 2026 revenue reached 267,021 crore."
    assert rec_rep is not None
    assert rec_rep["repaired_chars"] == 5
    assert rec_rep["dropped_tokens"] == 0

    # 3. Degraded page: corrupted letters dropped
    degraded_input = "Fiscal year ϮϬϮϲ ĐonƐolŝdated balance sheets."
    out_deg, q_deg, rec_deg = assess_and_filter_page(
        source="tcs", page_num=3, cleaned=degraded_input, raw=degraded_input
    )
    assert q_deg == "degraded"
    assert "ĐonƐolŝdated" not in out_deg
    assert "2026" in out_deg
    assert rec_deg is not None
    assert rec_deg["repaired_chars"] == 4
    assert rec_deg["dropped_tokens"] == 1
