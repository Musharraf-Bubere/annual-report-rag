"""Diagnostic script to analyze most frequent characters in U+0100-U+0FFF in tcs-ar-26."""

import collections
import sys

import pymupdf

from annual_report_rag.ingestion.loader import extract_page_text


def main() -> None:
    """Extracts text and tallies top characters in U+0100-U+0FFF range."""
    sys.stdout.reconfigure(encoding="utf-8")
    doc = pymupdf.open("data/raw/tcs-ar-26.pdf")
    char_counts: collections.Counter[str] = collections.Counter()
    char_contexts: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)

    for page_idx, page in enumerate(doc):
        clean, _raw, _method = extract_page_text(page)
        for i, ch in enumerate(clean):
            code = ord(ch)
            if 0x0100 <= code <= 0x0FFF:
                char_counts[ch] += 1
                if len(char_contexts[ch]) < 2:
                    start = max(0, i - 25)
                    end = min(len(clean), i + 26)
                    ctx = clean[start:end].replace("\n", " ")
                    char_contexts[ch].append((page_idx + 1, ctx))
    doc.close()

    print("Top 25 characters in U+0100-U+0FFF in tcs-ar-26:\n")
    print(f"{'Rank':<4} | {'Char':<4} | {'Unicode':<8} | {'Count':<5} | {'Context 1'} | {'Context 2'}")
    print("-" * 120)
    for rank, (ch, count) in enumerate(char_counts.most_common(25), 1):
        c1 = f"p{char_contexts[ch][0][0]}: ...{char_contexts[ch][0][1]}..." if len(char_contexts[ch]) > 0 else ""
        c2 = f"p{char_contexts[ch][1][0]}: ...{char_contexts[ch][1][1]}..." if len(char_contexts[ch]) > 1 else ""
        print(f"{rank:<4d} | {ch!a:<4s} | U+{ord(ch):04X}   | {count:<5d} | {c1} | {c2}")


if __name__ == "__main__":
    main()
