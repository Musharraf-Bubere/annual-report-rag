"""Search demo script for manually querying the Chroma vector store."""

import argparse
import sys

from annual_report_rag.config import load_settings
from annual_report_rag.providers.factory import get_provider
from annual_report_rag.retrieval.store import ChromaStore


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the search demo."""
    parser = argparse.ArgumentParser(
        description="Query the Chroma vector store with a question."
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help="Question to search against annual report embeddings",
    )
    return parser.parse_args()


def run_search(question: str) -> None:
    """Embeds question and prints top retrieved chunks with metadata and preview."""
    settings = load_settings()
    store = ChromaStore(
        persist_path=settings.storage.vector_store_path,
        collection_name=settings.storage.collection_name,
        embedding_model=settings.provider.embedding_model,
        embedding_dimension=settings.provider.embedding_dimension,
    )

    if store.count() == 0:
        print("Vector store is empty. Please run scripts/ingest.py first.")
        return

    provider = get_provider(settings)
    print(f"Searching for: '{question}'")
    vectors = provider.embed([question], purpose="query")
    if not vectors:
        print("Failed to generate query embedding.")
        return

    results = store.query(vectors[0], top_k=settings.retrieval.top_k)
    print(f"\nFound {len(results)} matching chunk(s):\n")

    for i, res in enumerate(results, 1):
        snippet = res.text[:200].replace("\n", " ")
        print(f"[{i}] Source: {res.source} | Page: {res.page} | Score: {res.score}")
        print(f"    Excerpt: \"{snippet}...\"\n")


def main() -> None:
    """Validates CLI input and triggers search."""
    args = parse_args()
    if not args.question:
        print("Usage: uv run python scripts/search_demo.py \"<your question>\"")
        sys.exit(1)
    run_search(args.question)


if __name__ == "__main__":
    main()
