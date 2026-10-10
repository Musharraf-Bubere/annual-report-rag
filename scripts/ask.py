"""CLI script to run the RAG query pipeline directly without a running server."""

import argparse
import sys

from annual_report_rag.config import load_settings
from annual_report_rag.generation.answerer import QueryDeps, answer_question
from annual_report_rag.generation.prompts import format_context
from annual_report_rag.providers.factory import get_embedder, get_generator
from annual_report_rag.retrieval.store import ChromaStore


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for scripts/ask.py."""
    parser = argparse.ArgumentParser(
        description="Ask a grounded question against the annual report vector store."
    )
    parser.add_argument(
        "question",
        type=str,
        help="Question to ask about the indexed annual reports",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the full numbered context sent to the generator",
    )
    return parser.parse_args()


def print_result(question: str, show_context: bool) -> None:
    """Initializes dependencies, runs answer_question, and prints structured output."""
    settings = load_settings()
    store = ChromaStore(
        persist_path=settings.storage.vector_store_path,
        collection_name=settings.storage.collection_name,
        embedding_provider=settings.embedding.provider,
        embedding_model=settings.embedding.model,
        embedding_dimension=settings.embedding.dimension,
    )
    if store.count() == 0:
        print("Error: Vector store is empty. Run scripts/ingest.py first.")
        sys.exit(1)

    embedder = get_embedder(settings)
    generator = get_generator(settings)
    deps = QueryDeps(
        settings=settings,
        embedder=embedder,
        generator=generator,
        store=store,
    )

    result = answer_question(question, deps)

    detected_str = ", ".join(result.detected_sources) if result.detected_sources else "(all sources)"
    print(f"Question: {question}")
    print(f"Detected sources: {detected_str}")

    print("\nTop retrieved chunks (up to 3):")
    if not result.retrieved:
        print("  (none)")
    else:
        for idx, chunk in enumerate(result.retrieved[:3], start=1):
            print(
                f"  [{idx}] {chunk.source} | PDF page {chunk.page} | "
                f"score: {chunk.score:.4f} | text_quality: {chunk.text_quality}"
            )

    if show_context:
        print("\nNumbered Context:")
        print("-" * 60)
        print(format_context(result.retrieved, settings))
        print("-" * 60)

    print(f"\nAnswer:\n{result.answer}")

    print("\nCitations:")
    if not result.citations:
        print("  (none)")
    else:
        for cit in result.citations:
            print(
                f"  [{cit.n}] {cit.display_name} ({cit.source}), "
                f"PDF page {cit.page}, text quality: {cit.text_quality}"
            )

    print(f"\nGrounded: {result.grounded} (reason: {result.reason}, invalid_citations: {result.invalid_citations})")
    print(f"Unsupported numbers: {result.unsupported_numbers}")
    print(
        f"Timings: retrieve={result.timings.retrieve_ms}ms, "
        f"generate={result.timings.generate_ms}ms, "
        f"total={result.timings.total_ms}ms"
    )
    print(
        f"Tokens: input={result.input_tokens}, output={result.output_tokens}"
    )


def main() -> None:
    """Entry point with UTF-8 console reconfiguration and safe top-level exception truncation."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    try:
        print_result(args.question, args.show_context)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)[:300]
        print(f"Error ({type(exc).__name__}): {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
