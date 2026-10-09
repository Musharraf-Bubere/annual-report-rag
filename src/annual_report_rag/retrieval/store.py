"""Persistent vector store wrapper around ChromaDB."""

from dataclasses import dataclass

import chromadb

from annual_report_rag.ingestion.chunker import Chunk


@dataclass(frozen=True)
class QueryResult:
    """Represents a matched chunk returned from a vector store query."""

    text: str
    source: str
    page: int
    score: float


class ChromaStore:
    """Encapsulates persistent ChromaDB operations using custom pre-computed embeddings."""

    def __init__(
        self,
        persist_path: str = "chroma_db",
        collection_name: str = "annual_reports",
    ) -> None:
        """Initializes persistent Chroma client and ensures collection exists with cosine distance."""
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
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
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=vectors,
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

    def query(self, vector: list[float], top_k: int = 5) -> list[QueryResult]:
        """Queries the vector store for nearest neighbor chunks using cosine similarity."""
        total_items = self.count()
        if total_items == 0 or top_k <= 0:
            return []

        results = self.collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, total_items),
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        query_results: list[QueryResult] = []
        for doc, meta, dist in zip(docs, metas, distances, strict=False):
            score = 1.0 - float(dist)
            query_results.append(
                QueryResult(
                    text=str(doc),
                    source=str(meta.get("source", "")),
                    page=int(meta.get("page", 0)),
                    score=round(score, 4),
                )
            )
        return query_results
