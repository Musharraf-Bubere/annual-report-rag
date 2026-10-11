"""Read-only CLI helper to inspect top-6 retrieved candidate chunks for draft golden questions."""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from annual_report_rag.config import Settings, load_settings
from annual_report_rag.evaluation.golden import GoldenQuestion, load_golden_set
from annual_report_rag.providers.base import EmbeddingProvider
from annual_report_rag.providers.factory import get_embedder
from annual_report_rag.retrieval.retriever import RetrievedChunk, retrieve
from annual_report_rag.retrieval.store import ChromaStore

CANDIDATES_HEADER = (
    "Candidates only. Answers must be confirmed in the PDF, not copied from chunk text."
)


def parse_id_filter(raw_ids: Sequence[str] | None) -> list[str]:
    """Parses space- or comma-separated question IDs into a deduplicated list."""
    if not raw_ids:
        return []
    parsed: list[str] = []
    for token in raw_ids:
        for part in token.split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in parsed:
                parsed.append(cleaned)
    return parsed


def select_candidate_questions(
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


def format_question_candidates(
    question: GoldenQuestion,
    chunks: Sequence[RetrievedChunk],
    top_k: int = 6,
    max_chars: int = 400,
) -> str:
    """Formats a single question and its top-k retrieved candidate chunks as Markdown."""
    lines: list[str] = []
    pages_str = (
        ", ".join(f"{ep.source}:p{ep.page}" for ep in question.expected_pages)
        or "none"
    )
    sources_str = ", ".join(question.expected_sources) or "none"

    lines.append(
        f"## `{question.id}` [{question.type} | split=`{question.split}` | status=`{question.status}`]"
    )
    lines.append("")
    lines.append(f"- **Question**: {question.question}")
    lines.append(f"- **Expected sources**: `{sources_str}`")
    lines.append(f"- **Current expected_pages**: `{pages_str}`")
    lines.append(f"- **Current must_contain**: `{question.must_contain}`")
    if question.notes:
        lines.append(f"- **Notes**: {question.notes}")
    lines.append("")

    top_chunks = list(chunks[:top_k])
    if not top_chunks:
        lines.append("_No chunks retrieved._")
        lines.append("")
        return "\n".join(lines)

    for rank, chunk in enumerate(top_chunks, start=1):
        snippet = chunk.text[:max_chars].replace("\r\n", "\n").strip()
        if len(chunk.text) > max_chars:
            snippet += " ..."
        lines.append(
            f"### Rank {rank}: `{chunk.source}` | PDF page `{chunk.page}` | "
            f"quality=`{chunk.text_quality}` | score=`{chunk.score:.4f}` (`{chunk.chunk_id}`)"
        )
        lines.append("```text")
        lines.append(snippet)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def generate_candidates_report(
    golden: Sequence[GoldenQuestion],
    settings: Settings,
    *,
    target_ids: Sequence[str] | None = None,
    top_k: int = 6,
    max_chars: int = 400,
    store: ChromaStore | None = None,
    embedder: EmbeddingProvider | None = None,
    eval_dir: Path | str | None = None,
) -> tuple[str, Path]:
    """Runs read-only retrieval for selected questions and writes data/eval/runs/candidates_<timestamp>.md."""
    active_store = store or ChromaStore(
        persist_path=settings.storage.vector_store_path,
        collection_name=settings.storage.collection_name,
        embedding_provider=settings.embedding.provider,
        embedding_model=settings.embedding.model,
        embedding_dimension=settings.embedding.dimension,
    )
    active_embedder = embedder or get_embedder(settings)

    selected = select_candidate_questions(golden, target_ids=target_ids)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    sections: list[str] = [
        CANDIDATES_HEADER,
        "",
        f"# Candidate Chunks Report ({timestamp})",
        "",
        f"- **Questions inspected**: `{len(selected)}`",
        f"- **Top chunks shown per question**: `{top_k}`",
        f"- **Snippet length**: `~{max_chars} chars`",
        "",
    ]

    for q in selected:
        chunks = retrieve(
            question=q.question,
            embedder=active_embedder,
            store=active_store,
            settings=settings,
        )
        sections.append(
            format_question_candidates(
                question=q,
                chunks=chunks,
                top_k=top_k,
                max_chars=max_chars,
            )
        )

    markdown_text = "\n".join(sections)

    base_eval_dir = (
        Path(eval_dir)
        if eval_dir is not None
        else Path(settings.storage.eval_data_dir)
    )
    runs_dir = base_eval_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"candidates_{timestamp}.md"
    out_path.write_text(markdown_text, encoding="utf-8")

    return markdown_text, out_path


def parse_args() -> argparse.Namespace:
    """Parses CLI flags for scripts/show_candidates.py."""
    parser = argparse.ArgumentParser(
        description="Read-only candidate chunk helper for draft golden questions."
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Optional question IDs to inspect (e.g. --ids q13 q14 or --ids q13,q14). Defaults to all status='draft_needs_fill'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of top retrieved chunks to display per question (default 6).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=400,
        help="Approximate character limit per chunk snippet (default 400).",
    )
    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> int:
    """Executes the read-only candidate retrieval helper and prints output."""
    settings = load_settings()
    golden = load_golden_set(settings.evaluation.golden_set_path)

    md_text, out_path = generate_candidates_report(
        golden=golden,
        settings=settings,
        target_ids=args.ids,
        top_k=args.top_k,
        max_chars=args.max_chars,
    )
    print(md_text)
    print(f"Saved candidate chunks report (gitignored): {out_path}")
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
