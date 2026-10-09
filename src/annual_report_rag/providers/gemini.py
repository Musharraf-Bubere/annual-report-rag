"""Gemini provider using the official google-genai SDK."""

from google import genai

from annual_report_rag.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Integrates Google Gemini models for embeddings and text generation."""

    def __init__(
        self,
        api_key: str | None,
        embedding_model: str,
        generation_model: str,
    ) -> None:
        """Initializes the Gemini client using API credentials and configured models."""
        if not api_key:
            raise ValueError(
                "Gemini API key is missing. Set GEMINI_API_KEY in .env or environment."
            )
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.client = genai.Client(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generates vector embeddings for a list of strings using the configured embedding model."""
        if not texts:
            return []

        response = self.client.models.embed_content(
            model=self.embedding_model,
            contents=texts,
        )

        if not response.embeddings:
            return []

        return [emb.values for emb in response.embeddings if emb.values is not None]

    def generate(self, prompt: str) -> str:
        """Generates a text completion for the prompt using the configured generation model."""
        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=prompt,
        )
        return response.text or ""
