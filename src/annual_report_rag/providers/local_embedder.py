"""Local embedding provider using fastembed ONNX runtime."""

from typing import Any

import numpy as np

from annual_report_rag.providers.base import EmbeddingProvider


def l2_normalize(vector: list[float] | Any) -> list[float]:
    """Scales a vector to unit Euclidean norm, returning standard Python floats."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = arr / norm
    return [float(x) for x in arr.tolist()]


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
        """Converts texts to normalized embeddings using query or passage methods.

        Returns a list of lists containing standard Python float scalars.
        """
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

        raw_embeddings = list(raw_iter)

        if len(raw_embeddings) != len(texts):
            raise ValueError(
                f"Expected {len(texts)} embeddings from local embedder, "
                f"but received {len(raw_embeddings)}."
            )

        vectors: list[list[float]] = []
        for idx, vec in enumerate(raw_embeddings):
            normed = l2_normalize(vec)
            if len(normed) != self.embedding_dimension:
                raise ValueError(
                    f"Embedding vector at index {idx} has dimension {len(normed)}, "
                    f"expected {self.embedding_dimension}."
                )
            vectors.append(normed)

        return vectors
