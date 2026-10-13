"""CLI script to validate the golden set and execute the RAG evaluation harness."""

import argparse
import sys
from pathlib import Path
from typing import Any

from annual_report_rag.config import Settings, load_settings
from annual_report_rag.evaluation.golden import (
    check_expected_pages_in_store,
    load_golden_set,
)
from annual_report_rag.evaluation.runner import (
    classify_expected_page_quality,
    rescore_run,
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
        "--rescore",
        type=str,
        default=None,
        help="Rescore cached answers from a prior run label (e.g. baseline-v1-full) with the current scorer and zero API/judge calls.",
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
    parser.add_argument(
        "--neighbor-window",
        type=int,
        default=None,
        help="Override retrieval.neighbor_window for this evaluation run.",
    )
    parser.add_argument(
        "--neighbor-top-n",
        type=int,
        default=None,
        help="Override retrieval.neighbor_top_n for this evaluation run.",
    )
    parser.add_argument(
        "--preserve-timestamp",
        action="store_true",
        help="Preserve source run timestamp and overwrite output metrics/md files when rescoring.",
    )
    parser.add_argument(
        "--max-context-chunks",
        type=int,
        default=None,
        help="Override retrieval.max_context_chunks for this evaluation run.",
    )
    parser.add_argument(
        "--rules-v2",
        action="store_true",
        help="Enable prompt.rules_v2 for this evaluation run.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable retrieval.hybrid.enabled (BM25 + dense RRF) for this evaluation run.",
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=None,
        help="Override retrieval.hybrid.bm25_top_k for this evaluation run.",
    )
    parser.add_argument(
        "--fusion-weight",
        type=float,
        default=None,
        help="Override retrieval.hybrid.fusion_weight for this evaluation run.",
    )
    return parser.parse_args()


def compute_freeze_hash(
    settings: Settings,
    golden_path: Path | str,
    store_count: int,
) -> tuple[str, dict[str, Any]]:
    """Computes a deterministic SHA-256 freeze hash for verifying locked evaluation configuration."""
    import dataclasses
    import hashlib
    import json

    from annual_report_rag.generation.prompts import get_system_prompt

    settings_payload = dataclasses.asdict(settings)
    settings_json = json.dumps(settings_payload, sort_keys=True, default=str)
    settings_hash = hashlib.sha256(settings_json.encode("utf-8")).hexdigest()

    sys_prompt_text = get_system_prompt(settings)
    sys_prompt_hash = hashlib.sha256(sys_prompt_text.encode("utf-8")).hexdigest()

    golden_bytes = Path(golden_path).read_bytes()
    golden_sha256 = hashlib.sha256(golden_bytes).hexdigest()

    store_count_str = str(store_count)

    combined_payload = (
        f"settings:{settings_hash}\n"
        f"prompt:{sys_prompt_hash}\n"
        f"golden:{golden_sha256}\n"
        f"chunks:{store_count_str}\n"
    )
    freeze_hash = hashlib.sha256(combined_payload.encode("utf-8")).hexdigest()

    details = {
        "settings_hash": settings_hash,
        "sys_prompt_len": len(sys_prompt_text),
        "sys_prompt_hash": sys_prompt_hash,
        "golden_sha256": golden_sha256,
        "store_chunk_count": store_count,
        "freeze_hash": freeze_hash,
    }
    return freeze_hash, details


def verify_test_split_safeguard(
    settings: Settings,
    golden_path: Path | str,
    store_count: int,
    repo_dir: Path | str = ".",
    tag_name: str = "frozen-config-v1",
) -> str:
    """Enforces test-split execution safeguards.

    Refuses to run unless:
    (a) the working tree is clean
    (b) git tag frozen-config-v1 exists
    (c) config hash equals the one stored in that tag's annotation
    """
    import subprocess

    cwd = Path(repo_dir)

    # (a) Check clean working tree
    status_proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if status_proc.returncode != 0:
        raise RuntimeError(
            f"Git status failed ({status_proc.stderr.strip()}). Cannot verify working tree cleanliness."
        )
    if status_proc.stdout.strip():
        raise RuntimeError(
            "Refusing to run --split test: working tree is dirty. "
            "Commit or stash all changes before running test-split evaluation.\n"
            f"Dirty files:\n{status_proc.stdout.strip()}"
        )

    # (b) Check git tag exists
    tag_proc = subprocess.run(
        ["git", "tag", "-l", tag_name],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if tag_proc.returncode != 0 or tag_name not in tag_proc.stdout.split():
        raise RuntimeError(
            f"Refusing to run --split test: required git tag '{tag_name}' does not exist."
        )

    # Compute freeze hash
    freeze_hash, details = compute_freeze_hash(settings, golden_path, store_count)

    print(f"\n=== Test Split Freeze Verification (Tag: {tag_name}) ===")
    print(f"  Settings Hash:           {details['settings_hash']}")
    print(
        f"  System Prompt Length:    {details['sys_prompt_len']} chars "
        f"(SHA-256: {details['sys_prompt_hash'][:16]}...)"
    )
    print(f"  Golden File SHA-256:     {details['golden_sha256']}")
    print(f"  ChromaStore Chunk Count: {details['store_chunk_count']}")
    print(f"  Computed Freeze Hash:    {freeze_hash}")

    # (c) Check tag annotation matches config hash
    tag_msg_proc = subprocess.run(
        ["git", "tag", "-l", "--format=%(contents)", tag_name],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    tag_annotation = tag_msg_proc.stdout.strip()
    if freeze_hash not in tag_annotation:
        raise RuntimeError(
            f"Refusing to run --split test: config hash mismatch.\n"
            f"  Computed freeze hash:        {freeze_hash}\n"
            f"  Tag '{tag_name}' annotation: '{tag_annotation}'\n"
            "The current configuration does not match the frozen tag configuration."
        )

    # (d) Check HEAD equals the commit the tag points to
    head_rev_proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    tag_rev_proc = subprocess.run(
        ["git", "rev-parse", f"{tag_name}^{{commit}}"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    head_commit = head_rev_proc.stdout.strip()
    tag_commit = tag_rev_proc.stdout.strip()
    if head_commit != tag_commit:
        raise RuntimeError(
            f"Refusing to run --split test: HEAD does not match tagged commit.\n"
            f"  HEAD commit:       {head_commit}\n"
            f"  Tag commit:        {tag_commit}\n"
            f"Checkout the tag '{tag_name}' or commit {tag_commit} before evaluating test split."
        )

    print(f"  Freeze Hash Match:       VERIFIED (matches tag '{tag_name}' annotation)")
    print(f"  Tag Commit Match:        VERIFIED (HEAD is {head_commit[:10]})\n")
    return freeze_hash


def run_cli(args: argparse.Namespace) -> int:
    """Loads golden set, validates against ChromaStore, and optionally executes run_eval or rescore_run."""
    from dataclasses import replace

    settings = load_settings()
    if (
        args.neighbor_window is not None
        or args.neighbor_top_n is not None
        or args.max_context_chunks is not None
    ):
        new_ret = replace(
            settings.retrieval,
            neighbor_window=(
                args.neighbor_window
                if args.neighbor_window is not None
                else settings.retrieval.neighbor_window
            ),
            neighbor_top_n=(
                args.neighbor_top_n
                if args.neighbor_top_n is not None
                else settings.retrieval.neighbor_top_n
            ),
            max_context_chunks=(
                args.max_context_chunks
                if args.max_context_chunks is not None
                else settings.retrieval.max_context_chunks
            ),
        )
        settings = replace(settings, retrieval=new_ret)
    if args.hybrid or args.bm25_top_k is not None or args.fusion_weight is not None:
        new_hybrid = replace(
            settings.retrieval.hybrid,
            enabled=True if args.hybrid else settings.retrieval.hybrid.enabled,
            bm25_top_k=(
                args.bm25_top_k
                if args.bm25_top_k is not None
                else settings.retrieval.hybrid.bm25_top_k
            ),
            fusion_weight=(
                args.fusion_weight
                if args.fusion_weight is not None
                else settings.retrieval.hybrid.fusion_weight
            ),
        )
        settings = replace(
            settings,
            retrieval=replace(settings.retrieval, hybrid=new_hybrid),
        )
    if args.rules_v2:
        settings = replace(
            settings,
            prompt=replace(settings.prompt, rules_v2=True),
        )
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

    if args.rescore:
        target_label = args.label if args.label != "eval" else args.rescore
        output = rescore_run(
            golden=golden,
            settings=settings,
            source_run_label=args.rescore,
            label=target_label,
            store=store,
            preserve_timestamp=args.preserve_timestamp,
        )
    else:
        if args.split == "test":
            verify_test_split_safeguard(
                settings=settings,
                golden_path=settings.evaluation.golden_set_path,
                store_count=store.count(),
            )
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
