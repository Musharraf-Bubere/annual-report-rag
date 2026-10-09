"""Extractor comparison experiment evaluating PDF extraction candidates on flagged pages."""

import csv
import io
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
import pypdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

from annual_report_rag.config import Settings, load_settings
from annual_report_rag.ingestion.loader import clean_text
from annual_report_rag.ingestion.quality import (
    PageQualityMetrics,
    assess_page_quality,
)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class CandidateSummary:
    """Holds aggregated benchmark performance metrics for an extraction candidate."""

    name: str
    cleaned_count: int
    total_flagged: int
    median_single_letter: float
    median_odd_count: float
    median_number_split: float
    elapsed_seconds: float


def rejoin_words(
    words: list[tuple[Any, ...]],
    gap_tolerance: float = 2.0,
) -> str:
    """Re-joins PyMuPDF word tuples on the same line if horizontal gap <= gap_tolerance."""
    if not words:
        return ""
    lines: list[str] = []
    current_line: list[str] = []
    prev_x1: float | None = None
    prev_block: int | None = None
    prev_line_no: int | None = None

    for w in words:
        x0, _, x1, _, text, block_no, line_no = w[0], w[1], w[2], w[3], w[4], w[5], w[6]
        if prev_block is None:
            current_line.append(text)
        elif block_no == prev_block and line_no == prev_line_no:
            gap = x0 - (prev_x1 if prev_x1 is not None else x0)
            if gap <= gap_tolerance:
                current_line[-1] += text
            else:
                current_line.append(text)
        else:
            lines.append(" ".join(current_line))
            current_line = [text]
        prev_x1 = x1
        prev_block = block_no
        prev_line_no = line_no

    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines)


def extract_pymupdf_default(page: pymupdf.Page) -> str:
    """Extracts text using standard PyMuPDF layout device."""
    return page.get_text("text")


def extract_pymupdf_inhibit_spaces(page: pymupdf.Page) -> str:
    """Extracts text using PyMuPDF with TEXT_INHIBIT_SPACES to suppress gap spaces."""
    return page.get_text("text", flags=pymupdf.TEXT_INHIBIT_SPACES)


def extract_pymupdf_words_rejoined(
    page: pymupdf.Page, gap_tolerance: float = 2.0
) -> str:
    """Extracts word bounding boxes from PyMuPDF and re-joins them with gap tolerance."""
    words = page.get_text("words")
    return rejoin_words(words, gap_tolerance=gap_tolerance)


def extract_pypdf(reader: pypdf.PdfReader, page_idx: int) -> str:
    """Extracts text from a single PDF page using pypdf parser."""
    return reader.pages[page_idx].extract_text() or ""


def extract_pdfminer_page(
    page: PDFPage,
    rsrcmgr: PDFResourceManager,
    laparams: LAParams,
) -> str:
    """Extracts text from a single PDFPage object using pdfminer.six layout engine."""
    out = io.StringIO()
    device = TextConverter(rsrcmgr, out, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    interpreter.process_page(page)
    device.close()
    return out.getvalue()


def load_flagged_pages(csv_path: Path) -> dict[str, list[int]]:
    """Loads PDF names and 1-based flagged page numbers from the audit CSV."""
    flagged: dict[str, list[int]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("exceeds_threshold", "").strip().lower() == "true":
                pdf = row["pdf_name"]
                page = int(row["page_number"])
                flagged.setdefault(pdf, []).append(page)
    return flagged


def is_page_flagged(m: PageQualityMetrics, settings: Settings) -> bool:
    """Checks whether calculated page metrics violate any configured threshold."""
    return (
        m.single_letter_ratio > settings.quality.single_letter_ratio_threshold
        or m.odd_char_ratio > settings.quality.odd_char_ratio_threshold
        or m.odd_char_count >= settings.quality.odd_char_count_threshold
        or m.number_split_count >= settings.quality.number_split_threshold
    )


def run_pymupdf_candidate(
    extractor_func: Any,
    flagged_dict: dict[str, list[int]],
    settings: Settings,
    name: str,
) -> tuple[CandidateSummary, dict[str, dict[int, PageQualityMetrics]], dict[str, dict[int, str]]]:
    """Runs a PyMuPDF-based extraction candidate on all flagged pages."""
    start = time.perf_counter()
    all_metrics: list[PageQualityMetrics] = []
    metrics_by_pdf: dict[str, dict[int, PageQualityMetrics]] = {}
    texts_by_pdf: dict[str, dict[int, str]] = {}
    cleaned = 0

    for pdf_name, page_nums in flagged_dict.items():
        doc = pymupdf.open(Path(settings.storage.raw_data_dir) / pdf_name)
        metrics_by_pdf[pdf_name] = {}
        texts_by_pdf[pdf_name] = {}
        for p_num in page_nums:
            raw_text = extractor_func(doc[p_num - 1])
            cleaned_text = clean_text(raw_text)
            texts_by_pdf[pdf_name][p_num] = cleaned_text
            m = assess_page_quality(pdf_name, p_num, cleaned_text)
            all_metrics.append(m)
            metrics_by_pdf[pdf_name][p_num] = m
            if not is_page_flagged(m, settings):
                cleaned += 1
        doc.close()

    elapsed = time.perf_counter() - start
    summary = CandidateSummary(
        name=name,
        cleaned_count=cleaned,
        total_flagged=len(all_metrics),
        median_single_letter=statistics.median([m.single_letter_ratio for m in all_metrics]),
        median_odd_count=statistics.median([float(m.odd_char_count) for m in all_metrics]),
        median_number_split=statistics.median([float(m.number_split_count) for m in all_metrics]),
        elapsed_seconds=round(elapsed, 2),
    )
    return summary, metrics_by_pdf, texts_by_pdf


def run_pypdf_candidate(
    flagged_dict: dict[str, list[int]],
    settings: Settings,
) -> tuple[CandidateSummary, dict[str, dict[int, PageQualityMetrics]], dict[str, dict[int, str]]]:
    """Runs the pypdf extraction candidate across all flagged pages."""
    start = time.perf_counter()
    all_metrics: list[PageQualityMetrics] = []
    metrics_by_pdf: dict[str, dict[int, PageQualityMetrics]] = {}
    texts_by_pdf: dict[str, dict[int, str]] = {}
    cleaned = 0

    for pdf_name, page_nums in flagged_dict.items():
        reader = pypdf.PdfReader(Path(settings.storage.raw_data_dir) / pdf_name)
        metrics_by_pdf[pdf_name] = {}
        texts_by_pdf[pdf_name] = {}
        for p_num in page_nums:
            raw_text = extract_pypdf(reader, p_num - 1)
            cleaned_text = clean_text(raw_text)
            texts_by_pdf[pdf_name][p_num] = cleaned_text
            m = assess_page_quality(pdf_name, p_num, cleaned_text)
            all_metrics.append(m)
            metrics_by_pdf[pdf_name][p_num] = m
            if not is_page_flagged(m, settings):
                cleaned += 1

    elapsed = time.perf_counter() - start
    summary = CandidateSummary(
        name="pypdf",
        cleaned_count=cleaned,
        total_flagged=len(all_metrics),
        median_single_letter=statistics.median([m.single_letter_ratio for m in all_metrics]),
        median_odd_count=statistics.median([float(m.odd_char_count) for m in all_metrics]),
        median_number_split=statistics.median([float(m.number_split_count) for m in all_metrics]),
        elapsed_seconds=round(elapsed, 2),
    )
    return summary, metrics_by_pdf, texts_by_pdf


def run_pdfminer_candidate(
    flagged_dict: dict[str, list[int]],
    settings: Settings,
    word_margin: float = 0.1,
    char_margin: float = 2.0,
) -> tuple[CandidateSummary, dict[str, dict[int, PageQualityMetrics]], dict[str, dict[int, str]]]:
    """Runs the pdfminer.six extraction candidate across flagged pages using LAParams."""
    start = time.perf_counter()
    all_metrics: list[PageQualityMetrics] = []
    metrics_by_pdf: dict[str, dict[int, PageQualityMetrics]] = {}
    texts_by_pdf: dict[str, dict[int, str]] = {}
    cleaned = 0
    laparams = LAParams(word_margin=word_margin, char_margin=char_margin)

    for pdf_name, page_nums in flagged_dict.items():
        pdf_path = Path(settings.storage.raw_data_dir) / pdf_name
        metrics_by_pdf[pdf_name] = {}
        texts_by_pdf[pdf_name] = {}
        target_indices = {p - 1 for p in page_nums}
        rsrcmgr = PDFResourceManager()

        with pdf_path.open("rb") as fp:
            for idx, page in enumerate(PDFPage.get_pages(fp, pagenos=target_indices)):
                p_num = page_nums[idx] if idx < len(page_nums) else idx + 1
                raw_text = extract_pdfminer_page(page, rsrcmgr, laparams)
                cleaned_text = clean_text(raw_text)
                texts_by_pdf[pdf_name][p_num] = cleaned_text
                m = assess_page_quality(pdf_name, p_num, cleaned_text)
                all_metrics.append(m)
                metrics_by_pdf[pdf_name][p_num] = m
                if not is_page_flagged(m, settings):
                    cleaned += 1

    elapsed = time.perf_counter() - start
    summary = CandidateSummary(
        name=f"pdfminer.six (wm={word_margin}, cm={char_margin})",
        cleaned_count=cleaned,
        total_flagged=len(all_metrics),
        median_single_letter=statistics.median([m.single_letter_ratio for m in all_metrics]),
        median_odd_count=statistics.median([float(m.odd_char_count) for m in all_metrics]),
        median_number_split=statistics.median([float(m.number_split_count) for m in all_metrics]),
        elapsed_seconds=round(elapsed, 2),
    )
    return summary, metrics_by_pdf, texts_by_pdf


def get_excerpt(text: str, anchor: str, length: int = 300) -> str:
    """Extracts a slice of text centered around an anchor term or from start."""
    idx = text.lower().find(anchor.lower())
    if idx != -1:
        start = max(0, idx - 40)
        snippet = text[start : start + length]
    else:
        snippet = text[:length]
    return snippet.replace("\n", " ")


def print_comparison_table(summaries: list[CandidateSummary]) -> None:
    """Prints tabular performance summary across all candidate extractors."""
    print("\n" + "=" * 90)
    print("EXTRACTOR BENCHMARK COMPARISON ON FLAGGED PAGES")
    print("=" * 90)
    header = f"{'Candidate':<42} | {'Cleaned':<10} | {'Med Single':<10} | {'Med Odd':<8} | {'Med Split':<9} | {'Time (s)':<8}"
    print(header)
    print("-" * 90)
    for s in summaries:
        cleaned_str = f"{s.cleaned_count}/{s.total_flagged}"
        print(
            f"{s.name:<42} | {cleaned_str:<10} | {s.median_single_letter:<10.4f} | "
            f"{s.median_odd_count:<8.0f} | {s.median_number_split:<9.0f} | {s.elapsed_seconds:<8.2f}"
        )
    print("=" * 90)


def print_page_inspection(
    page_num: int,
    anchor: str,
    all_texts: dict[str, dict[str, dict[int, str]]],
) -> None:
    """Prints 300-char side-by-side excerpts for a specific page across candidates."""
    print(f"\n--- Side-by-Side Excerpt: TCS Page {page_num} (Anchor: '{anchor}') ---")
    for name, pdf_dict in all_texts.items():
        text = pdf_dict.get("tcs-ar-26.pdf", {}).get(page_num, "")
        snippet = get_excerpt(text, anchor, length=300)
        print(f"\n[{name}] (len {len(text)}):")
        print(f"  \"{snippet}...\"")
        if page_num == 8:
            has_num = "267,021" in text
            print(f"  Exact '267,021' present: {has_num}")


def inspect_winning_candidate(
    winner_name: str,
    metrics_by_pdf: dict[str, dict[int, PageQualityMetrics]],
    settings: Settings,
) -> None:
    """Analyzes and displays pages still flagged under the winning candidate."""
    print("\n=======================================================")
    print(f"RESIDUAL DEFECT ANALYSIS FOR WINNER: {winner_name}")
    print("=======================================================")
    still_flagged = []
    for pages in metrics_by_pdf.values():
        for m in pages.values():
            if is_page_flagged(m, settings):
                still_flagged.append(m)

    print(f"Total pages still flagged: {len(still_flagged)}")
    for m in still_flagged[:15]:
        reasons = []
        if m.single_letter_ratio > settings.quality.single_letter_ratio_threshold:
            reasons.append(f"single_letter={m.single_letter_ratio:.4f}")
        if m.odd_char_ratio > settings.quality.odd_char_ratio_threshold:
            reasons.append(f"odd_ratio={m.odd_char_ratio:.4f}")
        if m.odd_char_count >= settings.quality.odd_char_count_threshold:
            reasons.append(f"odd_count={m.odd_char_count}")
        if m.number_split_count >= settings.quality.number_split_threshold:
            reasons.append(f"num_split={m.number_split_count}")
        print(f"  {m.source} Page {m.page:03d} -> {', '.join(reasons)}")


def main() -> None:
    """Orchestrates comparison across all candidate extractors on flagged pages."""
    settings = load_settings()
    csv_path = Path("data/processed/text_quality_report.csv")
    if not csv_path.exists():
        print(f"Audit CSV '{csv_path}' not found. Run scripts/audit_text_quality.py first.")
        return

    flagged = load_flagged_pages(csv_path)
    total_flagged = sum(len(pages) for pages in flagged.values())
    print(f"Loaded {total_flagged} flagged pages across {len(flagged)} PDF(s).")

    summaries: list[CandidateSummary] = []
    all_texts: dict[str, dict[str, dict[int, str]]] = {}

    # 1. PyMuPDF default
    s1, _, t1 = run_pymupdf_candidate(
        extract_pymupdf_default, flagged, settings, "PyMuPDF (default)"
    )
    summaries.append(s1)
    all_texts[s1.name] = t1

    # 2. PyMuPDF with TEXT_INHIBIT_SPACES
    s2, m2, t2 = run_pymupdf_candidate(
        extract_pymupdf_inhibit_spaces,
        flagged,
        settings,
        "PyMuPDF (TEXT_INHIBIT_SPACES)",
    )
    summaries.append(s2)
    all_texts[s2.name] = t2

    # 3. PyMuPDF words re-joined with gap tolerance 2.0
    s3, _, t3 = run_pymupdf_candidate(
        lambda p: extract_pymupdf_words_rejoined(p, gap_tolerance=2.0),
        flagged,
        settings,
        "PyMuPDF (words rejoined, gap=2.0)",
    )
    summaries.append(s3)
    all_texts[s3.name] = t3

    # 4. pypdf
    s4, _, t4 = run_pypdf_candidate(flagged, settings)
    summaries.append(s4)
    all_texts[s4.name] = t4

    # 5. pdfminer.six
    s5, _, t5 = run_pdfminer_candidate(
        flagged, settings, word_margin=0.1, char_margin=2.0
    )
    summaries.append(s5)
    all_texts[s5.name] = t5

    # Print summary table
    print_comparison_table(summaries)

    # Print side-by-side excerpts on worst TCS pages: 8, 22, 25
    print_page_inspection(8, "revenue", all_texts)
    print_page_inspection(22, "Fortune", all_texts)
    print_page_inspection(25, "584", all_texts)

    # Winner analysis (PyMuPDF TEXT_INHIBIT_SPACES)
    inspect_winning_candidate(s2.name, m2, settings)


if __name__ == "__main__":
    main()
