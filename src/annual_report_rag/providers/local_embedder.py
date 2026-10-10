"""Local embedding provider using fastembed ONNX runtime."""

import math
from typing import Any

from annual_report_rag.providers.base import EmbeddingProvider


def l2_normalize(vector: list[float]) -> list[float]:
    """Scales a vector to unit Euclidean norm."""
    norm = math.sqrt(sum(val * val for val in vector))
    if norm == 0.0:
        return vector
    return [val / norm for val in vector]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Generates dense vector embeddings locally using FastEmbed ONNX models."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        embedding_dimension: int = 768,
        model: Any | None = None,
    ) -> None:
        """Initializes the local embedder with model name, dimension, and optional injected model."""
        self.model_name = model_name
        self.embedding_dimension = embedding_dimension
        self._model = model

    def _get_model(self) -> Any:
        """Loads and caches the fastembed TextEmbedding instance lazily."""
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(
        self,
        texts: list[str],
        purpose: str = "document",
    ) -> list[list[float]]:
        """Converts texts to normalized embeddings using query or passage methods."""
        if not texts:
            return []

        model = self._get_model()
        normalized_purpose = purpose.strip().lower()

        if normalized_purpose == "query":
            raw_iter = model.query_embed(texts)
        elif normalized_purpose == "document":
            raw_iter = model.passage_embed(texts)
        else:
            raise ValueError(
                f"Unknown purpose: '{purpose}'. Expected 'document' or 'query'."
            )

        raw_embeddings = [list(vec) for vec in raw_iter]

        if len(raw_embeddings) != len(texts):
            raise ValueError(
                f"Expected {len(texts)} embeddings from local embedder, "
                f"but received {len(raw_embeddings)}."
            )

        vectors: list[list[float]] = []
        for idx, vec in enumerate(raw_embeddings):
            if len(vec) != self.embedding_dimension:
                raise ValueError(
                    f"Embedding vector at index {idx} has dimension {len(vec)}, "
                    f"expected {self.embedding_dimension}."
                )
            vectors.append(l2_normalize(vec))

        return vectors
