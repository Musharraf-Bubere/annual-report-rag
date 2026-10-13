"""In-house Okapi BM25 lexical index and Reciprocal Rank Fusion (RRF) for hybrid retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

# Matches the ₹ symbol, comma/decimal formatted numbers (e.g. 13.7, 52,820, 1,78,650), or lowercase alphanumeric words
BM25_TOKEN_PATTERN = re.compile(r"₹|\d+(?:,\d+)*(?:\.\d+)?|[a-z0-9]+")

# Function/question stopwords ignored during BM25 scoring to prevent common-word dilution
BM25_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "many",
        "much",
        "of",
        "on",
        "or",
        "s",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
    }
)


@dataclass(frozen=True)
class BM25Document:
    """Represents a stored chunk record indexed by BM25Index."""

    chunk_id: str
    text: str
    source: str
    page: int
    extraction_method: str = "default"
    text_quality: str = "clean"


@dataclass(frozen=True)
class BM25Hit:
    """Represents a scored chunk returned from a BM25 lexical search."""

    chunk_id: str
    text: str
    source: str
    page: int
    score: float
    extraction_method: str = "default"
    text_quality: str = "clean"


def tokenize_bm25(text: str) -> list[str]:
    """Tokenizes text while keeping formatted numbers (13.7, 52,820, 1,78,650) and the ₹ symbol intact."""
    if not text:
        return []
    return BM25_TOKEN_PATTERN.findall(text.lower())


def _filter_content_tokens(tokens: list[str]) -> list[str]:
    """Filters out function/question stopwords while falling back to all tokens if all are stopwords."""
    filtered = [t for t in tokens if t not in BM25_STOPWORDS]
    return filtered if filtered else tokens


class BM25Index:
    """In-memory Okapi BM25 index built over stored Chroma chunk documents."""

    def __init__(
        self,
        documents: Sequence[BM25Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Precomputes term frequencies, document lengths, and IDF weights over the given documents."""
        self.documents: list[BM25Document] = list(documents)
        self.k1 = float(k1)
        self.b = float(b)

        self._doc_tfs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        df: Counter[str] = Counter()

        for doc in self.documents:
            raw_tokens = tokenize_bm25(doc.text)
            tokens = _filter_content_tokens(raw_tokens)
            tf = Counter(tokens)
            self._doc_tfs.append(tf)
            self._doc_lens.append(len(tokens))
            for term in tf:
                df[term] += 1

        n_docs = len(self.documents)
        self._avgdl: float = (
            sum(self._doc_lens) / n_docs if n_docs > 0 else 1.0
        )
        if self._avgdl <= 0.0:
            self._avgdl = 1.0

        self._idf: dict[str, float] = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def __len__(self) -> int:
        """Returns the number of indexed documents."""
        return len(self.documents)

    def search(
        self,
        query: str,
        top_k: int = 20,
        sources: Sequence[str] | None = None,
    ) -> list[BM25Hit]:
        """Scores indexed documents against the query using Okapi BM25 and returns the top-k positive hits."""
        if not self.documents or top_k <= 0:
            return []

        raw_q_tokens = tokenize_bm25(query)
        if not raw_q_tokens:
            return []
        q_tokens = _filter_content_tokens(raw_q_tokens)

        allowed_sources: set[str] | None = set(sources) if sources else None
        k1 = self.k1
        b = self.b
        avgdl = self._avgdl
        idf_map = self._idf

        scored: list[tuple[float, int]] = []
        for idx, doc in enumerate(self.documents):
            if allowed_sources is not None and doc.source not in allowed_sources:
                continue
            tf = self._doc_tfs[idx]
            dl = self._doc_lens[idx]
            score = 0.0
            for qt in q_tokens:
                freq = tf.get(qt, 0)
                if freq == 0:
                    continue
                term_idf = idf_map.get(qt, 0.0)
                denom = freq + k1 * (1.0 - b + b * (dl / avgdl))
                score += term_idf * (freq * (k1 + 1.0)) / denom
            if score > 0.0:
                scored.append((score, idx))

        scored.sort(key=lambda item: (-item[0], self.documents[item[1]].chunk_id))
        hits: list[BM25Hit] = []
        for score, idx in scored[:top_k]:
            doc = self.documents[idx]
            hits.append(
                BM25Hit(
                    chunk_id=doc.chunk_id,
                    text=doc.text,
                    source=doc.source,
                    page=doc.page,
                    score=round(score, 4),
                    extraction_method=doc.extraction_method,
                    text_quality=doc.text_quality,
                )
            )
        return hits
