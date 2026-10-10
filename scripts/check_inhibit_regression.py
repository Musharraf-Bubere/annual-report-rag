"""Regression audit evaluating if TEXT_INHIBIT_SPACES glues words or harms clean pages."""

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from annual_report_rag.config import Settings, load_settings
from annual_report_rag.ingestion.loader import clean_text
from annual_report_rag.ingestion.quality import assess_page_quality

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class PageRegressionResult:
    """Holds comparison metrics between default and inhibit-spaces extraction for one page."""

    pdf_name: str
    page: int
    default_words: int
    inhibit_words: int
    word_count_ratio: float
    default_glued: int
    inhibit_glued: int
    glued_diff: int
    was_clean_before: bool
    has_regression: bool
    default_excerpt: str
    inhibit_excerpt: str


def count_glued_tokens(text: str) -> int:
    """Counts alphabetic tokens consisting of 25 or more consecutive letters."""
    return len(re.findall(r"[A-Za-z]{25,}", text))


def evaluate_page_regression(
    pdf_name: str,
    page_num: int,
    page: pymupdf.Page,
    settings: Settings,
) -> PageRegressionResult:
    """Extracts page text under both modes and computes word count and glued token metrics."""
    def_flags = pymupdf.TEXTFLAGS_TEXT
    inh_flags = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_INHIBIT_SPACES

    raw_def = page.get_text("text", flags=def_flags)
    raw_inh = page.get_text("text", flags=inh_flags)

    w_def = len(raw_def.split())
    w_inh = len(raw_inh.split())
    ratio = round(w_inh / max(w_def, 1), 4)

    g_def = count_glued_tokens(raw_def)
    g_inh = count_glued_tokens(raw_inh)
    glued_diff = g_inh - g_def

    cleaned_def = clean_text(raw_def)
    cleaned_inh = clean_text(raw_inh)
    metrics_def = assess_page_quality(pdf_name, page_num, cleaned_def, raw_text=raw_def)

    was_clean = (
        metrics_def.single_letter_ratio <= settings.quality.single_letter_ratio_threshold
        and metrics_def.number_split_count < settings.quality.number_split_threshold
        and metrics_def.odd_char_count < settings.quality.odd_char_count_threshold
        and metrics_def.odd_char_ratio <= settings.quality.odd_char_ratio_threshold
    )

    regressed = (glued_diff >= 2) or (w_def >= 20 and ratio <= 0.85)

    return PageRegressionResult(
        pdf_name=pdf_name,
        page=page_num,
        default_words=w_def,
        inhibit_words=w_inh,
        word_count_ratio=ratio,
        default_glued=g_def,
        inhibit_glued=g_inh,
        glued_diff=glued_diff,
        was_clean_before=was_clean,
        has_regression=regressed,
        default_excerpt=cleaned_def[:200],
        inhibit_excerpt=cleaned_inh[:200],
    )


def print_regression_summary(
    pdf_name: str,
    results: list[PageRegressionResult],
) -> None:
    """Prints statistical breakdown and 10 worst pages for a document."""
    regressed_pages = [r for r in results if r.has_regression]
    clean_before_regressed = [r for r in regressed_pages if r.was_clean_before]

    print("\n" + "=" * 70)
    print(f"INHIBIT-SPACES REGRESSION AUDIT: {pdf_name}")
    print("=" * 70)
    print(f"Total pages audited:                  {len(results)}")
    print(f"Pages with regression:                {len(regressed_pages)}")
    print(f"  - Of which were clean before:       {len(clean_before_regressed)}")

    if not regressed_pages:
        print("No regressions detected on this document.")
        return

    worst = sorted(
        regressed_pages,
        key=lambda r: (r.glued_diff, -r.word_count_ratio),
        reverse=True,
    )[:10]

    print("\n--- Top 10 Most Regressed Pages ---")
    for rank, r in enumerate(worst, 1):
        clean_tag = "[WAS CLEAN]" if r.was_clean_before else "[WAS FLAGGED]"
        print(
            f"#{rank:02d} Page {r.page:03d} {clean_tag} | words: {r.default_words} -> {r.inhibit_words} "
            f"({r.word_count_ratio:.2f}) | glued: {r.default_glued} -> {r.inhibit_glued} (+{r.glued_diff})"
        )
        print(f"     Default: \"{r.default_excerpt.replace(chr(10), ' ')}\"")
        print(f"     Inhibit: \"{r.inhibit_excerpt.replace(chr(10), ' ')}\"")


def save_regression_csv(
    output_path: Path,
    all_results: list[PageRegressionResult],
) -> None:
    """Exports regression results for all pages to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdf_name",
        "page",
        "default_words",
        "inhibit_words",
        "word_count_ratio",
        "default_glued",
        "inhibit_glued",
        "glued_diff",
        "was_clean_before",
        "has_regression",
        "default_excerpt",
        "inhibit_excerpt",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(
                {
                    "pdf_name": r.pdf_name,
                    "page": r.page,
                    "default_words": r.default_words,
                    "inhibit_words": r.inhibit_words,
                    "word_count_ratio": r.word_count_ratio,
                    "default_glued": r.default_glued,
                    "inhibit_glued": r.inhibit_glued,
                    "glued_diff": r.glued_diff,
                    "was_clean_before": r.was_clean_before,
                    "has_regression": r.has_regression,
                    "default_excerpt": r.default_excerpt,
                    "inhibit_excerpt": r.inhibit_excerpt,
                }
            )
    print(f"\nFull regression report saved to: {output_path}")


def main() -> None:
    """Orchestrates regression check across all raw PDFs."""
    settings = load_settings()
    pdf_dir = Path(settings.storage.raw_data_dir)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir}")
        return

    all_results: list[PageRegressionResult] = []

    for path in pdf_paths:
        doc = pymupdf.open(path)
        doc_results: list[PageRegressionResult] = []
        for idx in range(len(doc)):
            res = evaluate_page_regression(path.name, idx + 1, doc[idx], settings)
            doc_results.append(res)
            all_results.append(res)
        doc.close()
        print_regression_summary(path.name, doc_results)

    out_csv = Path("data/processed/inhibit_regression.csv")
    save_regression_csv(out_csv, all_results)


if __name__ == "__main__":
    main()
