"""Smoke test script to verify environment configuration, provider setup, and API keys."""

from annual_report_rag.config import load_settings
from annual_report_rag.providers.factory import get_embedder, get_generator


def run_smoke_test() -> None:
    """Executes a sample embedding and generation call to verify provider functionality."""
    print("Loading application settings and environment...")
    settings = load_settings()

    print(f"Embedding configured:  {settings.embedding.provider} ({settings.embedding.model})")
    print(f"Generation configured: {settings.generation.provider} ({settings.generation.model})")

    embedder = get_embedder(settings)
    generator = get_generator(settings)

    test_sentence = "Annual reports provide comprehensive audited financial disclosures."
    print(f"\n1. Embedding test sentence: '{test_sentence}'")
    embeddings = embedder.embed([test_sentence])
    if not embeddings or not embeddings[0]:
        raise RuntimeError("Failed to generate embedding: empty result returned.")
    embedding_len = len(embeddings[0])
    print(f"   Success! Embedding vector dimension: {embedding_len}")

    test_prompt = "In one sentence, explain what a corporate annual report is."
    print(f"\n2. Generating completion for prompt: '{test_prompt}'")
    result = generator.generate(test_prompt)
    print(f'   Success! Model answer:\n   "{result.text.strip()}"')
    print(
        f"   Token usage: input={result.input_tokens}, output={result.output_tokens}"
    )

    print("\nSmoke test passed successfully!")


if __name__ == "__main__":
    run_smoke_test()
