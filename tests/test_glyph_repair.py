"""Unit tests for glyph repair, corrupted digit/letter recovery, safety invariants, and text quality states."""

import re
from collections import Counter
from pathlib import Path

import pytest

from annual_report_rag.ingestion.glyph_repair import (
    CONFLICT_FREE_GLYPH_MAP,
    build_clean_vocabulary,
    repair_degraded_page_text,
    repair_glyphs,
)
from annual_report_rag.ingestion.loader import (
    assess_and_filter_page,
    extract_page_text,
    filter_corrupted_tokens,
)
from annual_report_rag.ingestion.quality import assess_page_quality


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


def test_repair_tibetan_rupee_and_digits_literal_decode() -> None:
    """Verifies that '༤ϭϬϳ͕ ϮϰϬ' decodes literally to '₹107,240' without grouping rewrites."""
    raw = "༤ϭϬϳ͕ ϮϰϬ"
    repaired, count = repair_glyphs(raw)
    assert repaired == "₹107,240"
    assert count == 8


def test_repair_percent_glyph() -> None:
    """Verifies that '4.6й', '25.0й', and '2.8й' have 'й' repaired to '%'."""
    raw = "Delivering growth of 4.6й and operating margin of 25.0й."
    repaired, count = repair_glyphs(raw)
    assert repaired == "Delivering growth of 4.6% and operating margin of 25.0%."
    assert count == 2


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


def test_mapped_letter_token_repaired_and_unmapped_dropped() -> None:
    """Verifies that mapped tokens ('ĐonƐolŝdated') are repaired while unmapped ('ĐƌŽƌe') are dropped."""
    text = "Company reported ĐonƐolŝdated revenue of 500 ĐƌŽƌe."
    filtered, quality, record = assess_and_filter_page(
        source="tcs-ar-26",
        page_num=8,
        cleaned=text,
        raw=text,
        odd_count_threshold=1,
    )
    assert quality == "degraded"
    assert "consolidated" in filtered
    assert "ĐƌŽƌe" not in filtered
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

    # 2. Repaired page: digits repaired, no letter tokens in U+0100..U+052F
    repaired_input = "Fiscal year ϮϬϮϲ revenue reached 26ϳ,021 crore."
    out_rep, q_rep, rec_rep = assess_and_filter_page(
        source="tcs", page_num=2, cleaned=repaired_input, raw=repaired_input
    )
    assert q_rep == "repaired"
    assert out_rep == "Fiscal year 2026 revenue reached 267,021 crore."
    assert rec_rep is not None
    assert rec_rep["repaired_chars"] == 5
    assert rec_rep["dropped_tokens"] == 0

    # 3. Degraded page: corrupted letters repaired and unmapped dropped
    degraded_input = "Fiscal year ϮϬϮϲ ĐonƐolŝdated balance in ĐƌŽƌe."
    out_deg, q_deg, rec_deg = assess_and_filter_page(
        source="tcs", page_num=3, cleaned=degraded_input, raw=degraded_input
    )
    assert q_deg == "degraded"
    assert "consolidated" in out_deg
    assert "ĐƌŽƌe" not in out_deg
    assert "2026" in out_deg
    assert rec_deg is not None
    assert rec_deg["dropped_tokens"] == 1


def test_phase5c_golden_repairs_and_idempotency() -> None:
    """Requirement 3(iii) & 3(iv): Golden unit test for target spans and idempotency."""
    assert len(CONFLICT_FREE_GLYPH_MAP) >= 31

    clean_corpus = [
        "In FY 2026, Operating Margin and Net Margin expanded as the Board declared dividend.",
        "FY 2025 revenue and Margin grew steadily. The Board recommended final dividend.",
        "FY 2024 Operating Margin and Net Margin remained resilient across markets.",
        "FY 2023 Margin and FY 2022 Margin were strong. The Foundation supported growth.",
        "The Operating Margin for FY 2026 was 25.0%.",
    ]
    exact_vocab, lower_vocab = build_clean_vocabulary(clean_corpus)
    rule_ii_log: Counter[tuple[str, str]] = Counter()
    join_log: Counter[tuple[str, str]] = Counter()

    raw_sample = (
        "The profit ;eǆĐůudinŐ eǆĐeptionaů iteŵsͿ for &z ϮϬϮϲ ǁas ₹52,820 Đrore "
        "ǁŝth an KpeƌatŝnŐ DaƌŐŝn oĨ 25͘0й and Net M argin of 19.8% and dividend of ₹110ͬ-."
    )
    repaired, rep_chars, joins = repair_degraded_page_text(
        raw_sample,
        clean_vocab_exact=exact_vocab,
        clean_vocab_lower=lower_vocab,
        join_spaced_capitals=True,
        rule_ii_counts=rule_ii_log,
        join_counts=join_log,
    )
    assert "(excluding exceptional items)" in repaired
    assert "FY 2026" in repaired
    assert "Operating Margin" in repaired
    assert "Net Margin" in repaired
    assert "₹110/-." in repaired
    assert rule_ii_log[("&z", "FY")] == 1
    assert join_log[("M argin", "Margin")] == 1
    assert rep_chars > 0
    assert joins == 1

    # Idempotency check: running repair a second time produces the exact same text with 0 changes
    repaired_twice, rep_chars_2, joins_2 = repair_degraded_page_text(
        repaired,
        clean_vocab_exact=exact_vocab,
        clean_vocab_lower=lower_vocab,
        join_spaced_capitals=True,
    )
    assert repaired_twice == repaired
    assert rep_chars_2 == 0
    assert joins_2 == 0

    # Verify join flag can be turned off
    repaired_no_join, _, joins_off = repair_degraded_page_text(
        raw_sample,
        clean_vocab_exact=exact_vocab,
        clean_vocab_lower=lower_vocab,
        join_spaced_capitals=False,
    )
    assert "Net M argin" in repaired_no_join
    assert joins_off == 0


def test_phase5c_pdf_safety_invariants() -> None:
    """Requirement 3(i)-(iii): Verifies non-degraded byte-identity, digit multiset preservation, and idempotency on PDFs."""
    tcs_path = Path("data/raw/tcs-ar-26.pdf")
    if not tcs_path.exists():
        pytest.skip("data/raw/tcs-ar-26.pdf not present in environment")

    import pymupdf

    doc = pymupdf.open(tcs_path)
    extracted: list[tuple[int, str, str]] = []
    clean_texts: list[str] = []

    for idx in range(len(doc)):
        page_num = idx + 1
        cleaned, raw, _ = extract_page_text(doc[idx])
        if len(cleaned) < 50:
            continue
        extracted.append((page_num, cleaned, raw))
        rep_num, rep_cnt = repair_glyphs(cleaned)
        if rep_cnt == 0:
            m = assess_page_quality("tcs-ar-26", page_num, rep_num, raw_text=raw)
            if m.single_letter_ratio <= 0.15:
                _, pre_drop = (
                    filter_corrupted_tokens(
                        rep_num, "\u0100", "\u052F", keep_tokens=("Mondelēz",)
                    )
                    if m.odd_char_count >= 1
                    else ("", 0)
                )
                if pre_drop == 0:
                    clean_texts.append(cleaned)
    doc.close()

    vocab_exact, vocab_lower = build_clean_vocabulary(clean_texts)
    degraded_count = 0
    non_degraded_count = 0

    for page_num, cleaned, raw in extracted:
        # Baseline-v3 behaviour before Phase 5c
        old_rep, _ = repair_glyphs(cleaned)
        old_m = assess_page_quality("tcs-ar-26", page_num, old_rep, raw_text=raw)
        if old_m.odd_char_count >= 1:
            old_final, old_dropped = filter_corrupted_tokens(
                old_rep, "\u0100", "\u052F", keep_tokens=("Mondelēz",)
            )
        else:
            old_final, old_dropped = old_rep, 0

        # New Phase 5c behaviour
        new_final, quality, _ = assess_and_filter_page(
            "tcs-ar-26",
            page_num,
            cleaned,
            raw,
            keep_tokens=("Mondelēz",),
            clean_vocab_exact=vocab_exact,
            clean_vocab_lower=vocab_lower,
        )

        if old_dropped == 0:
            # Safety (i): Every page not flagged degraded is byte-identical before and after
            non_degraded_count += 1
            assert quality in ("clean", "repaired")
            assert new_final == old_final, f"Non-degraded page {page_num} changed!"
        else:
            degraded_count += 1
            assert quality == "degraded"
            # Safety (ii): Multiset of digit tokens is unchanged vs old_final except tokens recovered
            old_digits = Counter(re.findall(r"\d+(?:[.,]\d+)*", old_final))
            new_digits = Counter(re.findall(r"\d+(?:[.,]\d+)*", new_final))
            lost_digits = old_digits - new_digits
            assert not lost_digits, f"Page {page_num} lost digit tokens: {lost_digits}"

            # Safety (iii): Idempotency
            idem_final, _, _ = assess_and_filter_page(
                "tcs-ar-26",
                page_num,
                new_final,
                new_final,
                keep_tokens=("Mondelēz",),
                clean_vocab_exact=vocab_exact,
                clean_vocab_lower=vocab_lower,
            )
            assert idem_final == new_final, f"Page {page_num} repair is not idempotent!"

    assert degraded_count == 30
    assert non_degraded_count == 330
