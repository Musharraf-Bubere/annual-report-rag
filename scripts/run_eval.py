"""CLI script to validate the golden set and execute the RAG evaluation harness."""

import argparse
import sys

from annual_report_rag.config import load_settings
from annual_report_rag.evaluation.golden import (
    check_expected_pages_in_store,
    load_golden_set,
)
from annual_report_rag.evaluation.runner import (
    classify_expected_page_quality,
    run_eval,
)
from annual_report_rag.retrieval.store import ChromaStore


def parse_args() -> argparse.Namespace:
    """Parses command-line flags for scripts/run_eval.py."""
    parser = argparse.ArgumentParser(
        description="Run the RAG evaluation harness."
    )
    parser.add_argument(
        "--label",
        type=str,
        default="eval",
        help="Short label for the evaluation run (e.g. baseline-v1).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "test", "all"],
        default="dev",
        help="Dataset split to evaluate: dev (default), test, or all.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of generation repeats per question (default 1).",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run the LLM faithfulness judge on grounded answers.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate retrieval metrics only without calling the generator.",
    )
    parser.add_argument(
        "--only-type",
        type=str,
        choices=["fact", "comparison", "table", "unanswerable", "adversarial"],
        default=None,
        help="Filter golden questions to a single question type.",
    )
    parser.add_argument(
        "--include-drafts",
        action="store_true",
        help="Include questions with status='draft_needs_fill' in scored runs (excluded by default).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the evaluation answer and judge caches.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate golden_questions.json and check expected pages in ChromaStore without running queries.",
    )
    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> int:
    """Loads golden set, validates against ChromaStore, and optionally executes run_eval."""
    settings = load_settings()
    golden = load_golden_set(settings.evaluation.golden_set_path)

    store = ChromaStore(
        persist_path=settings.storage.vector_store_path,
        collection_name=settings.storage.collection_name,
        embedding_provider=settings.embedding.provider,
        embedding_model=settings.embedding.model,
        embedding_dimension=settings.embedding.dimension,
    )

    warn_msgs = check_expected_pages_in_store(golden, store)

    if args.validate_only:
        dev_qs = [q for q in golden if q.split == "dev"]
        test_qs = [q for q in golden if q.split == "test"]
        draft_qs = [q for q in golden if q.status == "draft_needs_fill"]
        ready_qs = [q for q in golden if q.status == "ready_unverified"]
        verified_qs = [q for q in golden if q.status == "verified" or q.human_verified]
        dev_ready = sum(1 for q in dev_qs if q.status != "draft_needs_fill")
        dev_draft = sum(1 for q in dev_qs if q.status == "draft_needs_fill")
        test_ready = sum(1 for q in test_qs if q.status != "draft_needs_fill")
        test_draft = sum(1 for q in test_qs if q.status == "draft_needs_fill")

        print(
            f"Validated {len(golden)} golden questions from '{settings.evaluation.golden_set_path}'."
        )
        print(
            f"Split counts:  dev={len(dev_qs)} (ready={dev_ready}, draft={dev_draft}) | "
            f"test={len(test_qs)} (ready={test_ready}, draft={test_draft})"
        )
        print(
            f"Status counts: ready_unverified={len(ready_qs)} | "
            f"draft_needs_fill={len(draft_qs)} | verified={len(verified_qs)}"
        )
        print(f"Store chunk count: {store.count()} | Store fingerprint: {store.fingerprint()[:16]}...")
        for q in golden:
            q_slice = classify_expected_page_quality(q, store)
            pages_str = ", ".join(f"{ep.source}:p{ep.page}" for ep in q.expected_pages) or "none"
            print(
                f"  - {q.id} [{q.type} | split={q.split} | status={q.status}] "
                f"(answerable={q.answerable}, quality_slice={q_slice}): "
                f"expected_pages=[{pages_str}], must_contain={q.must_contain}"
            )
        if warn_msgs:
            print("\nWarnings:")
            for w in warn_msgs:
                print(f"  {w}")
        else:
            print("\nAll expected pages have indexed chunks in the vector store (0 warnings).")
        return 0

    output = run_eval(
        golden=golden,
        settings=settings,
        label=args.label,
        retrieval_only=args.retrieval_only,
        only_type=args.only_type,
        limit=args.limit,
        use_cache=not args.no_cache,
        split=args.split,
        repeats=args.repeats,
        use_judge=args.judge,
        include_drafts=args.include_drafts,
        store=store,
    )

    print(output.markdown_report)
    print(f"Saved full run detail (gitignored): {output.run_json_path}")
    if output.judge_review_path is not None:
        print(f"Saved judge review (gitignored):    {output.judge_review_path}")
    print(f"Saved metrics JSON (committed):     {output.metrics_json_path}")
    print(f"Saved markdown report (committed):  {output.markdown_path}")

    if output.quota_exceeded:
        print(
            "\nStopped early due to DailyQuotaExceeded; partial results have been saved.",
            file=sys.stderr,
        )
        return 2

    return 0


def main() -> None:
    """Entry point with UTF-8 reconfiguration and safe error truncation."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    try:
        code = run_cli(args)
        sys.exit(code)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)[:300]
        print(f"Error ({type(exc).__name__}): {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
