"""Unit tests for PDF document loading and cleaning."""

from pathlib import Path

import pymupdf

from annual_report_rag.ingestion.loader import clean_text, get_pdf_page_count, load_pdf


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
