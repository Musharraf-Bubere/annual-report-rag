"""Smoke test script to verify environment configuration, provider setup, and API keys."""

from annual_report_rag.config import load_settings
from annual_report_rag.providers.factory import get_provider


def run_smoke_test() -> None:
    """Executes a sample embedding and generation call to verify provider functionality."""
    print("Loading application settings and environment...")
    settings = load_settings()

    print(f"Provider configured: {settings.provider.name}")
    print(f"Embedding model:     {settings.provider.embedding_model}")
    print(f"Generation model:    {settings.provider.generation_model}")

    provider = get_provider(settings)

    test_sentence = "Annual reports provide comprehensive audited financial disclosures."
    print(f"\n1. Embedding test sentence: '{test_sentence}'")
    embeddings = provider.embed([test_sentence])
    if not embeddings or not embeddings[0]:
        raise RuntimeError("Failed to generate embedding: empty result returned.")
    embedding_len = len(embeddings[0])
    print(f"   Success! Embedding vector dimension: {embedding_len}")

    test_prompt = "In one sentence, explain what a corporate annual report is."
    print(f"\n2. Generating completion for prompt: '{test_prompt}'")
    answer = provider.generate(test_prompt)
    print(f"   Success! Model answer:\n   \"{answer.strip()}\"")

    print("\nSmoke test passed successfully!")


if __name__ == "__main__":
    run_smoke_test()
