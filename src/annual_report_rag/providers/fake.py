"""In-memory fake provider for offline testing and development."""

from annual_report_rag.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    """Simulates embedding and generation models deterministically without network calls."""

    def __init__(
        self,
        embedding_dim: int = 768,
        default_response: str = "This is a grounded fake answer from page 42.",
    ) -> None:
        """Initializes the fake provider with a target embedding dimension and default text."""
        self.embedding_dim = embedding_dim
        self.default_response = default_response
        self.embedded_texts: list[str] = []
        self.generated_prompts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns deterministic pseudo-embeddings based on text length."""
        self.embedded_texts.extend(texts)
        embeddings: list[list[float]] = []
        for text in texts:
            seed_val = float(len(text) % 100) / 100.0
            vector = [seed_val] * self.embedding_dim
            embeddings.append(vector)
        return embeddings

    def generate(self, prompt: str) -> str:
        """Records prompt and returns pre-configured mock answer."""
        self.generated_prompts.append(prompt)
        return self.default_response
