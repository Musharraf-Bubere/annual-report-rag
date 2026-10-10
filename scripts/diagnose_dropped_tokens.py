"""Diagnostic script to inspect dropped tokens across all PDF pages."""

import csv
import sys
from pathlib import Path

import pymupdf

sys.stdout.reconfigure(encoding="utf-8")
from annual_report_rag.config import load_settings
from annual_report_rag.ingestion.loader import extract_page_text
from annual_report_rag.ingestion.quality import assess_page_quality


def main() -> None:
    settings = load_settings()
    r_start = settings.quality.corrupted_glyph_range_start
    r_end = settings.quality.corrupted_glyph_range_end

    report_rows = []

    for pdf_path in [Path("data/raw/infosys-ar-26.pdf"), Path("data/raw/tcs-ar-26.pdf")]:
        if not pdf_path.exists():
            continue
        doc = pymupdf.open(pdf_path)
        source = pdf_path.stem
        for idx in range(len(doc)):
            page = doc[idx]
            clean_text, raw_text, _method = extract_page_text(page)
            tokens = clean_text.split()
            dropped = [t for t in tokens if any(r_start <= ch <= r_end for ch in t)]
            if dropped:
                metrics = assess_page_quality(source, idx + 1, clean_text, raw_text=raw_text)
                distinct_dropped = list(dict.fromkeys(dropped))[:8]
                report_rows.append(
                    {
                        "pdf": source,
                        "page": idx + 1,
                        "odd_char_count": metrics.odd_char_count,
                        "dropped_tokens_count": len(dropped),
                        "sample_dropped_tokens": " | ".join(distinct_dropped),
                    }
                )
        doc.close()

    out_path = Path("data/processed/dropped_tokens_report.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["pdf", "page", "odd_char_count", "dropped_tokens_count", "sample_dropped_tokens"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Total pages with dropped tokens: {len(report_rows)}\n")
    print(f"{'PDF':<16} | {'Page':>4} | {'Odd Chars':>9} | {'Dropped':>7} | {'Sample Dropped Tokens'}")
    print("-" * 90)
    for r in report_rows:
        print(f"{r['pdf']:<16} | {r['page']:4d} | {r['odd_char_count']:9d} | {r['dropped_tokens_count']:7d} | {r['sample_dropped_tokens']}")


if __name__ == "__main__":
    main()
