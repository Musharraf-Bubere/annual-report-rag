"""PDF document loader and text extraction using PyMuPDF."""

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class LoadedPage:
    """Represents extracted text and metadata for a single PDF page."""

    source: str
    page: int
    text: str


def clean_text(text: str) -> str:
    """Normalizes consecutive whitespace characters into a single space and strips margins."""
    return re.sub(r"\s+", " ", text).strip()


def check_scanned_pdf(
    source_name: str,
    total_pages: int,
    extracted_pages: int,
    total_chars: int,
) -> None:
    """Prints a warning if a document appears to be scanned or image-only."""
    if total_pages > 0 and (extracted_pages == 0 or total_chars < 100):
        print(
            f"WARNING: '{source_name}' yielded almost no extractable text "
            f"({total_chars} chars across {total_pages} pages). "
            "Scanned/image-only PDFs are out of scope."
        )


def load_pdf(
    pdf_path: Path | str,
    min_page_chars: int = 50,
    limit_pages: int | None = None,
) -> list[LoadedPage]:
    """Extracts text page-by-page from a PDF, filtering out blank or sparse pages."""
    path = Path(pdf_path)
    source = path.stem
    doc = pymupdf.open(path)

    total_pages = len(doc)
    pages_to_process = min(total_pages, limit_pages) if limit_pages else total_pages

    loaded_pages: list[LoadedPage] = []
    skipped_count = 0
    total_chars = 0

    for idx in range(pages_to_process):
        page_num = idx + 1
        raw_text = doc[idx].get_text()
        cleaned = clean_text(raw_text)

        if len(cleaned) < min_page_chars:
            skipped_count += 1
            continue

        loaded_pages.append(LoadedPage(source=source, page=page_num, text=cleaned))
        total_chars += len(cleaned)

    doc.close()

    if skipped_count > 0:
        print(
            f"[{source}] Skipped {skipped_count} page(s) with < {min_page_chars} characters."
        )

    check_scanned_pdf(source, pages_to_process, len(loaded_pages), total_chars)
    return loaded_pages


def get_pdf_page_count(pdf_path: Path | str) -> int:
    """Returns the total number of pages in a PDF document."""
    doc = pymupdf.open(Path(pdf_path))
    count = len(doc)
    doc.close()
    return count

