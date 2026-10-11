"""Read-only CLI script to search source PDFs directly with PyMuPDF for must_contain tokens."""

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

from annual_report_rag.config import Settings, load_settings
from annual_report_rag.evaluation.golden import GoldenQuestion, load_golden_set
from annual_report_rag.ingestion.glyph_repair import repair_glyphs

_PURE_NUMBER_RE = re.compile(
    r"^\s*[₹$€£]?\s*(\d+(?:[\s,]\d+)*(?:\.\d+)?)\s*%?\s*$"
)
_NUMERIC_TOKEN_RE = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?")


@dataclass(frozen=True)
class PageLookupRow:
    """Represents a single row in the PDF page lookup table."""

    question_id: str
    token: str
    source: str
    matching_pages: list[int]
    expected_pages_for_source: list[int]


def parse_id_filter(raw_ids: Sequence[str] | None) -> list[str]:
    """Parses space-, comma-, or range-separated question IDs (e.g. q01-q09, q25) into a list of IDs."""
    if not raw_ids:
        return []
    parsed: list[str] = []
    for token in raw_ids:
        for part in token.split(","):
            cleaned = part.strip()
            if not cleaned:
                continue
            range_match = re.match(r"^q(\d+)\s*-\s*q?(\d+)$", cleaned, re.IGNORECASE)
            if range_match:
                start_n = int(range_match.group(1))
                end_n = int(range_match.group(2))
                step = 1 if end_n >= start_n else -1
                for n in range(start_n, end_n + step, step):
                    qid = f"q{n:02d}"
                    if qid not in parsed:
                        parsed.append(qid)
            else:
                if cleaned not in parsed:
                    parsed.append(cleaned)
    return parsed


def select_lookup_questions(
    golden: Sequence[GoldenQuestion],
    target_ids: Sequence[str] | None = None,
) -> list[GoldenQuestion]:
    """Selects questions matching target_ids, or all status=='draft_needs_fill' questions when target_ids is empty."""
    ids_list = parse_id_filter(target_ids)
    if ids_list:
        by_id = {q.id: q for q in golden}
        missing = [qid for qid in ids_list if qid not in by_id]
        if missing:
            raise ValueError(f"Unknown question IDs requested via --ids: {missing}")
        return [by_id[qid] for qid in ids_list]

    return [q for q in golden if q.status == "draft_needs_fill"]


def _extract_numeric_tokens_from_page(raw_text: str, is_tcs: bool) -> set[str]:
    """Extracts normalized numeric tokens from a PDF page (stripping commas and, for TCS, spaces inside digits)."""
    # Include both raw text and glyph-repaired text so TCS Coptic digits (U+03EC-U+03F5) are matched
    repaired, _ = repair_glyphs(raw_text)
    variants = [raw_text, repaired] if repaired != raw_text else [raw_text]

    tokens: set[str] = set()
    for text in variants:
        # Standard numeric tokens with thousands/Indian commas stripped
        for m in _NUMERIC_TOKEN_RE.findall(text):
            tokens.add(m.replace(",", ""))

        if is_tcs:
            # Collapse spaces adjacent to commas (e.g. "2, 67, 021" -> "2,67,021")
            comma_collapsed = re.sub(r"(?<=\d),\s+(?=\d)", ",", text)
            for m in _NUMERIC_TOKEN_RE.findall(comma_collapsed):
                tokens.add(m.replace(",", ""))

            # Collapse intra-line spaces between digits (e.g. "584, 51 9" -> "584519")
            space_collapsed = re.sub(r"(?<=\d)[ \t,]+(?=\d)", "", text)
            for m in _NUMERIC_TOKEN_RE.findall(space_collapsed):
                tokens.add(m.replace(",", ""))

    return tokens


def page_matches_token(raw_text: str, token_entry: str, source: str) -> bool:
    """Returns True if raw_text on a page from source contains any '|'-separated alternative in token_entry."""
    if not raw_text or not token_entry:
        return False

    is_tcs = source.lower().startswith("tcs")
    repaired, _ = repair_glyphs(raw_text)
    norm_raw = re.sub(r"\s+", " ", raw_text).strip()
    norm_rep = re.sub(r"\s+", " ", repaired).strip()

    page_num_tokens: set[str] | None = None
    tcs_digit_collapsed_raw: str | None = None
    tcs_digit_collapsed_rep: str | None = None

    alternatives = [alt.strip() for alt in token_entry.split("|") if alt.strip()]
    for alt in alternatives:
        num_match = _PURE_NUMBER_RE.match(alt)
        if num_match:
            target_num = re.sub(r"[\s,]+", "", num_match.group(1))
            if page_num_tokens is None:
                page_num_tokens = _extract_numeric_tokens_from_page(raw_text, is_tcs=is_tcs)
            if target_num in page_num_tokens:
                return True

            if is_tcs:
                if tcs_digit_collapsed_raw is None:
                    tcs_digit_collapsed_raw = re.sub(r"(?<=\d)[ \t,]+(?=\d)", "", norm_raw)
                    tcs_digit_collapsed_rep = re.sub(r"(?<=\d)[ \t,]+(?=\d)", "", norm_rep)
                pattern = re.compile(
                    rf"(?<!\d)(?<!\d\.){re.escape(target_num)}(?!\d)(?!\.\d)"
                )
                if pattern.search(tcs_digit_collapsed_raw) or (
                    tcs_digit_collapsed_rep and pattern.search(tcs_digit_collapsed_rep)
                ):
                    return True
        else:
            alt_norm = re.sub(r"\s+", " ", alt).strip().lower()
            if alt_norm in norm_raw.lower() or alt_norm in norm_rep.lower():
                return True

    return False


def load_pdf_pages_cache(
    sources: Sequence[str],
    raw_data_dir: Path,
) -> dict[str, list[str]]:
    """Loads raw text of all pages for each requested PDF source stem using PyMuPDF."""
    cache: dict[str, list[str]] = {}
    for src in sources:
        if src in cache:
            continue
        pdf_path = raw_data_dir / f"{src}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Source PDF not found at: {pdf_path}")
        doc = pymupdf.open(pdf_path)
        pages_raw: list[str] = []
        for idx in range(len(doc)):
            page = doc[idx]
            raw_def = page.get_text("text", flags=pymupdf.TEXTFLAGS_TEXT)
            if src.lower().startswith("tcs"):
                inh_flags = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_INHIBIT_SPACES
                raw_inh = page.get_text("text", flags=inh_flags)
                pages_raw.append(raw_def + "\n" + raw_inh)
            else:
                pages_raw.append(raw_def)
        doc.close()
        cache[src] = pages_raw
    return cache


def find_pages_for_questions(
    questions: Sequence[GoldenQuestion],
    raw_data_dir: Path,
) -> list[PageLookupRow]:
    """Searches source PDFs directly for each must_contain token in each question."""
    needed_sources: list[str] = []
    for q in questions:
        q_sources = list(q.expected_sources)
        if not q_sources:
            q_sources = sorted({ep.source for ep in q.expected_pages})
        if not q_sources:
            q_sources = sorted(p.stem for p in raw_data_dir.glob("*.pdf"))
        for s in q_sources:
            if s not in needed_sources:
                needed_sources.append(s)

    pdf_cache = load_pdf_pages_cache(needed_sources, raw_data_dir)
    rows: list[PageLookupRow] = []

    for q in questions:
        q_sources = list(q.expected_sources)
        if not q_sources:
            q_sources = sorted({ep.source for ep in q.expected_pages})
        if not q_sources:
            q_sources = list(needed_sources)

        tokens = list(q.must_contain) if q.must_contain else ["(none)"]
        for src in q_sources:
            expected_for_src = sorted(
                {ep.page for ep in q.expected_pages if ep.source == src}
            )
            pages_text = pdf_cache.get(src, [])
            for tok in tokens:
                if tok == "(none)":
                    rows.append(
                        PageLookupRow(
                            question_id=q.id,
                            token=tok,
                            source=src,
                            matching_pages=[],
                            expected_pages_for_source=expected_for_src,
                        )
                    )
                    continue

                matched: list[int] = []
                for idx, raw_page_text in enumerate(pages_text):
                    page_num = idx + 1
                    if page_matches_token(raw_page_text, tok, source=src):
                        matched.append(page_num)

                rows.append(
                    PageLookupRow(
                        question_id=q.id,
                        token=tok,
                        source=src,
                        matching_pages=matched,
                        expected_pages_for_source=expected_for_src,
                    )
                )

    return rows


def format_lookup_markdown(
    rows: Sequence[PageLookupRow],
    timestamp: str,
) -> str:
    """Formats the PDF page lookup results as a Markdown table."""
    lines: list[str] = [
        f"# Source PDF Page Lookup ({timestamp})",
        "",
        "> Read-only direct PyMuPDF scan over `data/raw/*.pdf` (1-based PDF page indices).",
        "",
        "| id | token | source | matching PDF pages | current expected_pages |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in rows:
        matched_str = ", ".join(str(p) for p in r.matching_pages) or "none"
        expected_str = ", ".join(str(p) for p in r.expected_pages_for_source) or "none"
        safe_tok = r.token.replace("|", "\\|")
        lines.append(
            f"| `{r.question_id}` | `{safe_tok}` | `{r.source}` | {matched_str} | {expected_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_page_lookup(
    golden: Sequence[GoldenQuestion],
    settings: Settings,
    *,
    target_ids: Sequence[str] | None = None,
    raw_data_dir: Path | str | None = None,
    eval_dir: Path | str | None = None,
) -> tuple[str, Path, list[PageLookupRow]]:
    """Executes the PDF page lookup and writes data/eval/runs/page_lookup_<timestamp>.md."""
    selected = select_lookup_questions(golden, target_ids=target_ids)
    active_raw_dir = (
        Path(raw_data_dir)
        if raw_data_dir is not None
        else Path(settings.storage.raw_data_dir)
    )
    active_eval_dir = (
        Path(eval_dir)
        if eval_dir is not None
        else Path(settings.storage.eval_data_dir)
    )

    rows = find_pages_for_questions(selected, active_raw_dir)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    md_text = format_lookup_markdown(rows, timestamp=timestamp)

    runs_dir = active_eval_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"page_lookup_{timestamp}.md"
    out_path.write_text(md_text, encoding="utf-8")

    return md_text, out_path, rows


def parse_args() -> argparse.Namespace:
    """Parses CLI flags for scripts/find_pages.py."""
    parser = argparse.ArgumentParser(
        description="Search source PDFs directly with PyMuPDF for must_contain tokens."
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Optional question IDs or ranges (e.g. --ids q25 q29 q31-q37 q01-q09 q16-q21). Defaults to status='draft_needs_fill'.",
    )
    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> int:
    """Loads settings and golden_questions.json, runs page lookup, and prints the Markdown table."""
    settings = load_settings()
    golden = load_golden_set(settings.evaluation.golden_set_path)

    md_text, out_path, _rows = run_page_lookup(
        golden=golden,
        settings=settings,
        target_ids=args.ids,
    )
    print(md_text)
    print(f"Saved page lookup report (gitignored): {out_path}")
    return 0


def main() -> None:
    """Entry point with UTF-8 console reconfiguration and safe error handling."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    try:
        sys.exit(run_cli(args))
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)[:300]
        print(f"Error ({type(exc).__name__}): {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
