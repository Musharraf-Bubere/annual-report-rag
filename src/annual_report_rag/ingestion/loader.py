"""PDF document loader with quality-aware best-of text extraction and token filtering."""

import csv
import re
import string
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from annual_report_rag.ingestion.quality import assess_page_quality


@dataclass(frozen=True)
class LoadedPage:
    """Represents extracted text and quality metadata for a single PDF page."""

    source: str
    page: int
    text: str
    extraction_method: str = "default"
    text_quality: str = "clean"


def clean_text(text: str) -> str:
    """Normalizes consecutive whitespace characters into a single space and strips margins."""
    return re.sub(r"\s+", " ", text).strip()


def filter_corrupted_tokens(
    text: str,
    range_start: str = "\u0100",
    range_end: str = "\u052F",
    keep_tokens: tuple[str, ...] | set[str] = (),
) -> tuple[str, int]:
    """Drops whitespace-separated tokens in Unicode range, preserving keep_tokens."""
    if not text:
        return "", 0
    keep_set = set(keep_tokens)
    tokens = text.split()
    clean_tokens: list[str] = []
    dropped_count = 0
    for token in tokens:
        stripped = token.strip(string.punctuation)
        if token in keep_set or stripped in keep_set:
            clean_tokens.append(token)
        elif any(range_start <= ch <= range_end for ch in token):
            dropped_count += 1
        else:
            clean_tokens.append(token)
    return " ".join(clean_tokens), dropped_count


def compute_extraction_defect_score(text: str, raw_text: str) -> float:
    """Computes composite defect score from single letters, split numbers, and glued tokens."""
    metrics = assess_page_quality("temp", 0, text, raw_text=raw_text)
    glued = len(re.findall(r"[A-Za-z]{25,}", text))
    return metrics.single_letter_ratio + (metrics.number_split_count * 0.05) + (glued * 0.25)


def choose_better_extraction(
    default_text: str,
    inhibit_text: str,
    raw_default: str | None = None,
    raw_inhibit: str | None = None,
    single_threshold: float = 0.15,
    number_split_threshold: int = 3,
) -> tuple[str, str]:
    """Selects best extraction between default and inhibit-spaces based on defect scores."""
    raw_def = raw_default if raw_default is not None else default_text
    m_default = assess_page_quality("temp", 0, default_text, raw_text=raw_def)
    is_flagged = (
        m_default.single_letter_ratio > single_threshold
        or m_default.number_split_count >= number_split_threshold
    )

    if not is_flagged:
        return default_text, "default"

    raw_inh = raw_inhibit if raw_inhibit is not None else inhibit_text
    score_default = compute_extraction_defect_score(default_text, raw_def)
    score_inhibit = compute_extraction_defect_score(inhibit_text, raw_inh)

    if score_inhibit < score_default:
        return inhibit_text, "inhibit_spaces"
    return default_text, "default"


def extract_page_text(
    page: pymupdf.Page,
    single_threshold: float = 0.15,
    number_split_threshold: int = 3,
) -> tuple[str, str, str]:
    """Extracts text for a page, comparing default vs inhibit-spaces if defect flags trigger."""
    raw_def = page.get_text("text", flags=pymupdf.TEXTFLAGS_TEXT)
    clean_def = clean_text(raw_def)
    m_def = assess_page_quality("temp", 0, clean_def, raw_text=raw_def)
    if (
        m_def.single_letter_ratio <= single_threshold
        and m_def.number_split_count < number_split_threshold
    ):
        return clean_def, raw_def, "default"

    inh_flags = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_INHIBIT_SPACES
    raw_inh = page.get_text("text", flags=inh_flags)
    clean_inh = clean_text(raw_inh)
    chosen_clean, method = choose_better_extraction(
        clean_def,
        clean_inh,
        raw_default=raw_def,
        raw_inhibit=raw_inh,
        single_threshold=single_threshold,
        number_split_threshold=number_split_threshold,
    )
    chosen_raw = raw_inh if method == "inhibit_spaces" else raw_def
    return chosen_clean, chosen_raw, method


def write_degraded_log(output_path: Path, records: list[dict[str, object]]) -> None:
    """Appends degraded page records to CSV audit log file avoiding duplicates."""
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys: set[tuple[str, str]] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_keys.add((str(row["pdf_name"]), str(row["page"])))

    new_records = [
        r for r in records if (str(r["pdf_name"]), str(r["page"])) not in existing_keys
    ]
    if not new_records and output_path.exists():
        return

    fieldnames = ["pdf_name", "page", "odd_char_count", "dropped_tokens", "remaining_chars"]
    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not output_path.exists() or len(existing_keys) == 0:
            writer.writeheader()
        writer.writerows(new_records)


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


def assess_and_filter_page(
    source: str,
    page_num: int,
    cleaned: str,
    raw: str,
    odd_count_threshold: int = 15,
    glyph_range_start: str = "\u0100",
    glyph_range_end: str = "\u052F",
    keep_tokens: tuple[str, ...] | set[str] = (),
) -> tuple[str, str, dict[str, object] | None]:
    """Filters corrupted tokens on pages meeting odd_char_count threshold, logging defects."""
    metrics = assess_page_quality(source, page_num, cleaned, raw_text=raw)
    if metrics.odd_char_count < odd_count_threshold:
        return cleaned, "clean", None

    filtered, dropped = filter_corrupted_tokens(
        cleaned, glyph_range_start, glyph_range_end, keep_tokens=keep_tokens
    )
    if dropped == 0:
        return cleaned, "clean", None

    record: dict[str, object] = {
        "pdf_name": source,
        "page": page_num,
        "odd_char_count": metrics.odd_char_count,
        "dropped_tokens": dropped,
        "remaining_chars": len(filtered),
    }
    return filtered, "degraded", record


def load_pdf(
    pdf_path: Path | str,
    min_page_chars: int = 50,
    limit_pages: int | None = None,
    single_threshold: float = 0.15,
    number_split_threshold: int = 3,
    odd_count_threshold: int = 15,
    glyph_range_start: str = "\u0100",
    glyph_range_end: str = "\u052F",
    keep_tokens: tuple[str, ...] | set[str] = (),
    degraded_log_path: Path | str | None = "data/processed/degraded_pages.csv",
) -> list[LoadedPage]:
    """Extracts text using best-of extraction, filters corrupted glyph tokens, and logs defects."""
    path = Path(pdf_path)
    source = path.stem
    doc = pymupdf.open(path)
    pages_to_process = min(len(doc), limit_pages) if limit_pages else len(doc)

    loaded_pages: list[LoadedPage] = []
    degraded_records: list[dict[str, object]] = []
    skipped_count = 0
    total_chars = 0

    for idx in range(pages_to_process):
        page_num = idx + 1
        cleaned, raw, method = extract_page_text(
            doc[idx], single_threshold, number_split_threshold
        )
        if len(cleaned) < min_page_chars:
            skipped_count += 1
            continue

        cleaned, quality, record = assess_and_filter_page(
            source,
            page_num,
            cleaned,
            raw,
            odd_count_threshold=odd_count_threshold,
            glyph_range_start=glyph_range_start,
            glyph_range_end=glyph_range_end,
            keep_tokens=keep_tokens,
        )
        if record:
            degraded_records.append(record)

        loaded_pages.append(
            LoadedPage(
                source=source,
                page=page_num,
                text=cleaned,
                extraction_method=method,
                text_quality=quality,
            )
        )
        total_chars += len(cleaned)

    doc.close()
    if degraded_log_path is not None:
        write_degraded_log(Path(degraded_log_path), degraded_records)
    if skipped_count > 0:
        print(f"[{source}] Skipped {skipped_count} page(s) with < {min_page_chars} characters.")
    check_scanned_pdf(source, pages_to_process, len(loaded_pages), total_chars)
    return loaded_pages


def get_pdf_page_count(pdf_path: Path | str) -> int:
    """Returns the total number of pages in a PDF document."""
    doc = pymupdf.open(Path(pdf_path))
    count = len(doc)
    doc.close()
    return count
