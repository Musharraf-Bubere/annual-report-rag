"""Persistent vector store wrapper around ChromaDB."""

import collections
from dataclasses import dataclass
from typing import Any

import chromadb
import numpy as np
from chromadb.errors import NotFoundError

from annual_report_rag.ingestion.chunker import Chunk


@dataclass(frozen=True)
class QueryResult:
    """Represents a matched chunk returned from a vector store query."""

    text: str
    source: str
    page: int
    score: float
    extraction_method: str = "default"
    text_quality: str = "clean"
    chunk_id: str = ""


class ChromaStore:
    """Encapsulates persistent ChromaDB operations using custom pre-computed embeddings."""

    def __init__(
        self,
        persist_path: str = "chroma_db",
        collection_name: str = "annual_reports",
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
        reset_existing: bool = False,
    ) -> None:
        """Initializes persistent Chroma client and ensures collection exists with cosine distance."""
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.client = chromadb.PersistentClient(path=persist_path)

        if reset_existing:
            self.reset()
        else:
            metadata: dict[str, Any] = {"hnsw:space": "cosine"}
            if embedding_provider is not None:
                metadata["embedding_provider"] = embedding_provider
            if embedding_model is not None:
                metadata["embedding_model"] = embedding_model
            if embedding_dimension is not None:
                metadata["embedding_dimension"] = embedding_dimension

            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=metadata,
            )
            self._validate_metadata()

    def _validate_metadata(self) -> None:
        """Validates that collection metadata matches current provider, model and dimension."""
        meta = self.collection.metadata or {}
        stored_provider = meta.get("embedding_provider")
        stored_model = meta.get("embedding_model")
        stored_dim = meta.get("embedding_dimension")

        if (
            stored_provider
            and self.embedding_provider
            and stored_provider != self.embedding_provider
        ):
            raise ValueError(
                f"Collection '{self.collection_name}' was indexed with provider '{stored_provider}', "
                f"but current setting is '{self.embedding_provider}'. "
                "Please re-index using --reset."
            )
        if stored_model and self.embedding_model and stored_model != self.embedding_model:
            raise ValueError(
                f"Collection '{self.collection_name}' was indexed with model '{stored_model}', "
                f"but current setting is '{self.embedding_model}'. "
                "Please re-index using --reset."
            )
        if (
            stored_dim is not None
            and self.embedding_dimension is not None
            and int(stored_dim) != int(self.embedding_dimension)
        ):
            raise ValueError(
                f"Collection '{self.collection_name}' was indexed with dimension {stored_dim}, "
                f"but current setting is {self.embedding_dimension}. "
                "Please re-index using --reset."
            )

    def reset(self) -> None:
        """Deletes and recreates the collection, safely handling missing collections."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except (NotFoundError, ValueError):
            # Collection does not exist yet; safe to proceed with recreation
            pass

        metadata: dict[str, Any] = {"hnsw:space": "cosine"}
        if self.embedding_provider is not None:
            metadata["embedding_provider"] = self.embedding_provider
        if self.embedding_model is not None:
            metadata["embedding_model"] = self.embedding_model
        if self.embedding_dimension is not None:
            metadata["embedding_dimension"] = self.embedding_dimension

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=metadata,
        )

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Inserts text chunks and pre-computed vectors into the collection."""
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks and vectors must match.")

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "extraction_method": chunk.extraction_method,
                "text_quality": chunk.text_quality,
            }
            for chunk in chunks
        ]

        converted_vectors: list[list[float]] = (
            np.asarray(vectors, dtype=np.float32).tolist()
        )

        self.collection.add(
            ids=ids,
            embeddings=converted_vectors,
            documents=documents,
            metadatas=metadatas,
        )

    def existing_ids(self, ids: list[str]) -> set[str]:
        """Queries the collection to identify which given IDs already exist."""
        if not ids:
            return set()
        result = self.collection.get(ids=ids)
        return set(result.get("ids", []))

    def count(self) -> int:
        """Returns the total number of indexed chunks currently in the store."""
        return self.collection.count()

    def source_counts(self) -> dict[str, int]:
        """Returns the number of indexed chunks per document source."""
        if self.count() == 0:
            return {}
        result = self.collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        counter = collections.Counter(
            str(meta.get("source", "")) for meta in metadatas if meta
        )
        return dict(sorted(counter.items()))

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        sources: list[str] | None = None,
    ) -> list[QueryResult]:
        """Queries the vector store for nearest neighbor chunks using cosine similarity."""
        total_items = self.count()
        if total_items == 0 or top_k <= 0:
            return []

        converted_vector: list[float] = (
            np.asarray(vector, dtype=np.float32).tolist()
        )

        where_filter: dict[str, Any] | None = None
        if sources:
            if len(sources) == 1:
                where_filter = {"source": sources[0]}
            else:
                where_filter = {"source": {"$in": list(sources)}}

        results = self.collection.query(
            query_embeddings=[converted_vector],
            n_results=min(top_k, total_items),
            where=where_filter,
        )

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        query_results: list[QueryResult] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, distances, strict=False):
            score = 1.0 - float(dist)
            query_results.append(
                QueryResult(
                    chunk_id=str(cid),
                    text=str(doc),
                    source=str(meta.get("source", "")),
                    page=int(meta.get("page", 0)),
                    score=round(score, 4),
                    extraction_method=str(meta.get("extraction_method", "default")),
                    text_quality=str(meta.get("text_quality", "clean")),
                )
            )
        return query_results
