"""Audit script assessing text extraction quality across all raw PDF annual reports."""

import csv
import math
import statistics
from pathlib import Path

import pymupdf

from annual_report_rag.config import load_settings
from annual_report_rag.ingestion.loader import clean_text
from annual_report_rag.ingestion.quality import PageQualityMetrics, assess_page_quality


def calculate_percentile(values: list[float], pct: float) -> float:
    """Calculates the given percentile value from a sequence of numbers."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * (pct / 100.0)
    low, high = math.floor(idx), math.ceil(idx)
    if low == high:
        return sorted_vals[int(idx)]
    return sorted_vals[low] * (high - idx) + sorted_vals[high] * (idx - low)


def audit_pdf(
    pdf_path: Path,
) -> tuple[list[PageQualityMetrics], list[str]]:
    """Audits all pages in a single PDF, returning metrics and 150-char previews."""
    doc = pymupdf.open(pdf_path)
    source = pdf_path.stem
    all_metrics: list[PageQualityMetrics] = []
    excerpts: list[str] = []

    for idx in range(len(doc)):
        page_num = idx + 1
        raw_text = doc[idx].get_text()
        cleaned = clean_text(raw_text)
        metrics = assess_page_quality(source, page_num, cleaned)
        all_metrics.append(metrics)
        excerpts.append(cleaned[:150])

    doc.close()
    return all_metrics, excerpts


def print_pdf_audit_summary(
    pdf_name: str,
    metrics: list[PageQualityMetrics],
    excerpts: list[str],
    single_threshold: float,
    odd_threshold: float,
) -> None:
    """Prints statistical distributions, threshold violations, and top 10 worst pages."""
    print("\n=======================================================")
    print(f"AUDIT REPORT: {pdf_name}")
    print("=======================================================")
    print(f"Total pages examined: {len(metrics)}")

    single_ratios = [m.single_letter_ratio for m in metrics]
    odd_ratios = [m.odd_char_ratio for m in metrics]
    char_counts = [float(m.char_count) for m in metrics]

    print("\n--- Metric Distributions (Median / 95th Percentile) ---")
    print(
        f"  Character count:      Median: {int(statistics.median(char_counts))} | "
        f"P95: {int(calculate_percentile(char_counts, 95))}"
    )
    print(
        f"  Single-letter ratio:  Median: {statistics.median(single_ratios):.4f} | "
        f"P95: {calculate_percentile(single_ratios, 95):.4f} "
        f"(Threshold: >{single_threshold:.2f})"
    )
    print(
        f"  Odd character ratio:  Median: {statistics.median(odd_ratios):.4f} | "
        f"P95: {calculate_percentile(odd_ratios, 95):.4f} "
        f"(Threshold: >{odd_threshold:.2f})"
    )

    flagged_single = [m for m in metrics if m.single_letter_ratio > single_threshold]
    flagged_odd = [m for m in metrics if m.odd_char_ratio > odd_threshold]
    flagged_any = [
        m
        for m in metrics
        if m.single_letter_ratio > single_threshold or m.odd_char_ratio > odd_threshold
    ]

    print("\n--- Threshold Violations ---")
    print(f"  Pages exceeding single-letter ratio: {len(flagged_single)}")
    print(f"  Pages exceeding odd-character ratio: {len(flagged_odd)}")
    print(f"  Total flagged pages (any defect):    {len(flagged_any)}")

    ranked_indices = sorted(
        range(len(metrics)),
        key=lambda i: max(
            metrics[i].single_letter_ratio / max(single_threshold, 0.001),
            metrics[i].odd_char_ratio / max(odd_threshold, 0.001),
        ),
        reverse=True,
    )

    print("\n--- Top 10 Most Defective Pages ---")
    for rank, idx in enumerate(ranked_indices[:10], 1):
        m = metrics[idx]
        snippet = excerpts[idx].replace("\n", " ")
        safe_snippet = snippet.encode("ascii", errors="replace").decode("ascii")
        print(
            f"  #{rank:02d} Page {m.page:03d} | chars: {m.char_count:4d} | "
            f"single: {m.single_letter_ratio:.4f} | odd: {m.odd_char_ratio:.4f}"
        )
        print(f"       Snippet: \"{safe_snippet}...\"")


def write_csv_report(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Writes detailed per-page audit records to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdf_name",
        "page_number",
        "char_count",
        "single_letter_ratio",
        "odd_char_ratio",
        "exceeds_threshold",
        "excerpt",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull per-page audit written to: {output_path}")


def main() -> None:
    """Orchestrates quality audit across raw PDFs and outputs statistical report."""
    settings = load_settings()
    pdf_dir = Path(settings.storage.raw_data_dir)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDF files found in '{pdf_dir}'.")
        return

    csv_rows: list[dict[str, object]] = []
    single_thresh = settings.quality.single_letter_ratio_threshold
    odd_thresh = settings.quality.odd_char_ratio_threshold

    for pdf_path in pdf_paths:
        metrics, excerpts = audit_pdf(pdf_path)
        print_pdf_audit_summary(
            pdf_path.name, metrics, excerpts, single_thresh, odd_thresh
        )

        for m, exc in zip(metrics, excerpts, strict=True):
            violates = (
                m.single_letter_ratio > single_thresh
                or m.odd_char_ratio > odd_thresh
            )
            csv_rows.append(
                {
                    "pdf_name": pdf_path.name,
                    "page_number": m.page,
                    "char_count": m.char_count,
                    "single_letter_ratio": m.single_letter_ratio,
                    "odd_char_ratio": m.odd_char_ratio,
                    "exceeds_threshold": violates,
                    "excerpt": exc,
                }
            )

    report_csv = Path("data/processed/text_quality_report.csv")
    write_csv_report(report_csv, csv_rows)


if __name__ == "__main__":
    main()
