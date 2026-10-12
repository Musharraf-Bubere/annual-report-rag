"""Unit tests for PDF document loading and cleaning."""

from pathlib import Path

import pymupdf

from annual_report_rag.ingestion.loader import (
    assess_and_filter_page,
    choose_better_extraction,
    clean_text,
    filter_corrupted_tokens,
    get_pdf_page_count,
    load_pdf,
)


def test_clean_text() -> None:
    """Verifies whitespace normalization and edge trimming."""
    raw = "  Audited   results\n\nfor   FY26.\t  "
    assert clean_text(raw) == "Audited results for FY26."


def test_load_pdf_extraction_and_skipping(tmp_path: Path) -> None:
    """Creates a temporary PDF to test page extraction, min_page_chars filtering, and page counting."""
    pdf_path = tmp_path / "sample_report.pdf"

    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "Revenue increased by 15% year-over-year in cloud services.",
    )

    page2 = doc.new_page()
    page2.insert_text((50, 72), "Brief")  # 5 characters, below threshold

    page3 = doc.new_page()
    page3.insert_text(
        (50, 72),
        "Operating margins remained stable across European markets.",
    )
    doc.save(pdf_path)
    doc.close()

    assert get_pdf_page_count(pdf_path) == 3

    loaded = load_pdf(pdf_path, min_page_chars=20)

    assert len(loaded) == 2
    assert loaded[0].source == "sample_report"
    assert loaded[0].page == 1
    assert "Revenue increased" in loaded[0].text
    assert loaded[1].page == 3
    assert "Operating margins" in loaded[1].text


def test_load_pdf_limit_pages(tmp_path: Path) -> None:
    """Verifies that limit_pages properly restricts the number of evaluated pages."""
    pdf_path = tmp_path / "multi_page.pdf"
    doc = pymupdf.open()
    for i in range(5):
        p = doc.new_page()
        p.insert_text((50, 72), f"Page {i + 1} content with sufficient length.")
    doc.save(pdf_path)
    doc.close()

    loaded = load_pdf(pdf_path, min_page_chars=10, limit_pages=2)
    assert len(loaded) == 2
    assert loaded[0].page == 1
    assert loaded[1].page == 2


def test_filter_corrupted_tokens() -> None:
    """Verifies that tokens containing corrupted Unicode glyphs are dropped while clean tokens remain."""
    clean = "Annual report revenue increased to ₹ 50,000 crore."
    filtered, dropped = filter_corrupted_tokens(clean)
    assert filtered == clean
    assert dropped == 0

    corrupted = "ĐonƐolŝdated ƌevenue oĨ ₹ 26ϳ,021 Đƌoƌe for the year"
    filtered_corrupted, dropped_count = filter_corrupted_tokens(corrupted)
    # The corrupted tokens are dropped, ₹ and "for the year" remain
    assert dropped_count == 5
    assert filtered_corrupted == "₹ for the year"


def test_choose_better_extraction() -> None:
    """Verifies that choose_better_extraction selects the optimal method based on defect metrics."""
    # Clean text: default is kept
    clean = "Audited financial statements for the fiscal year ended March 31."
    text, method = choose_better_extraction(clean, clean)
    assert text == clean
    assert method == "default"

    # Spaced letters: inhibit is chosen
    def_spaced = "F ortune M ag az i ne S tev i e I nternati onal"
    inh_spaced = "Fortune Magazine Stevie International"
    text, method = choose_better_extraction(def_spaced, inh_spaced)
    assert text == inh_spaced
    assert method == "inhibit_spaces"

    # Split numbers: inhibit is chosen
    def_num = "Employees 584, 5 1 9 and 44, 0 0 0 and 2 0 2 6 in delivery centres"
    inh_num = "Employees 584,519 and 44,000 and 2026 in delivery centres"
    text, method = choose_better_extraction(def_num, inh_num)
    assert text == inh_num
    assert method == "inhibit_spaces"

    # Glued words regression in inhibit: default is kept
    inh_glued = "Employeeswithdeepadvisoryandconsultingexpertisegluednonsense"
    text, method = choose_better_extraction(def_num, inh_glued)
    assert text == def_num
    assert method == "default"


def test_load_pdf_quality_metadata_and_degraded_logging(tmp_path: Path) -> None:
    """Verifies that load_pdf tags degraded pages, cleans corrupted tokens, and writes audit CSV."""
    pdf_path = tmp_path / "quality_test.pdf"
    csv_path = tmp_path / "degraded.csv"

    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 72), "Audited consolidated revenue reached INR 250,000 crore.")

    p2 = doc.new_page()
    font_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    font_path = next((str(p) for p in font_candidates if p.exists()), None)

    corrupted_sentence = "ĐonƐolŝdated ƌevenue oĨ 26ϳ,021 Đƌoƌe " * 4
    if font_path:
        p2.insert_font(fontname="u_font", fontfile=font_path)
        p2.insert_text((50, 72), corrupted_sentence, fontname="u_font")
    else:
        p2.insert_text((50, 72), corrupted_sentence)
    doc.save(pdf_path)
    doc.close()

    loaded = load_pdf(pdf_path, min_page_chars=20, degraded_log_path=csv_path)

    assert len(loaded) == 2
    assert loaded[0].page == 1
    assert loaded[0].text_quality == "clean"
    assert loaded[0].extraction_method == "default"

    assert loaded[1].page == 2
    if font_path:
        assert loaded[1].text_quality == "degraded"
        # Corrupted tokens were filtered out; no corrupted glyphs remain in text
        assert all("\u0100" > ch or ch > "\u052F" for ch in loaded[1].text)
        assert csv_path.exists()


def test_filter_corrupted_tokens_keep_tokens() -> None:
    """Verifies that allowlisted keep_tokens are preserved while corrupted tokens are dropped."""
    text = "Mondelēz reported strong revenue with ĐonƐolŝdated balance sheets."
    # Without allowlist: both Mondelēz (ē is U+0113) and ĐonƐolŝdated are dropped
    no_allow, dropped_all = filter_corrupted_tokens(text)
    assert dropped_all == 2
    assert "Mondelēz" not in no_allow
    assert "ĐonƐolŝdated" not in no_allow

    # With allowlist: Mondelēz is preserved, ĐonƐolŝdated is dropped
    with_allow, dropped_corrupted = filter_corrupted_tokens(
        text, keep_tokens=("Mondelēz",)
    )
    assert dropped_corrupted == 1
    assert "Mondelēz" in with_allow
    assert "ĐonƐolŝdated" not in with_allow


def test_page_below_threshold_stays_clean() -> None:
    """Verifies that a page with odd characters below the threshold is left untouched and clean."""
    text = "Mondelēz International establishes an end-to-end traceability reporting capability."
    # odd_char_count is 1 (ē is U+0113), well below threshold 15
    cleaned, quality, record = assess_and_filter_page(
        source="infosys-ar-26",
        page_num=5,
        cleaned=text,
        raw=text,
        odd_count_threshold=15,
        keep_tokens=("Mondelēz",),
    )
    assert quality == "clean"
    assert record is None
    assert cleaned == text  # Completely untouched


def test_page_above_threshold_drops_corrupted_and_retains_allowlisted() -> None:
    """Verifies that a page above threshold repairs mapped tokens, drops unmapped corrupted tokens, and retains allowlisted names."""
    corrupted_part = "ĐonƐolŝdated ƌevenue oĨ 26ϳ,021 ĐƌŽƌe h^Ψ " * 4
    text = f"Mondelēz announced that {corrupted_part}"
    cleaned, quality, record = assess_and_filter_page(
        source="tcs-ar-26",
        page_num=8,
        cleaned=text,
        raw=text,
        odd_count_threshold=15,
        keep_tokens=("Mondelēz",),
    )
    assert quality == "degraded"
    assert record is not None
    assert record["dropped_tokens"] == 4
    assert "Mondelēz" in cleaned
    assert "consolidated revenue of 267,021 crore" in cleaned
    assert "h^Ψ" not in cleaned
