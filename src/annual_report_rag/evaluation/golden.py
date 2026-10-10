"""Pydantic schemas and loader for the golden evaluation question set."""

import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from annual_report_rag.retrieval.store import ChromaStore

QuestionType = Literal["fact", "comparison", "unanswerable", "adversarial"]


class ExpectedPage(BaseModel):
    """Identifies a source document stem and 1-indexed PDF page number."""

    source: str = Field(..., min_length=1)
    page: int = Field(..., ge=1)


class GoldenQuestion(BaseModel):
    """Represents a single evaluation question with expected provenance and lexical checks."""

    id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    type: QuestionType
    answerable: bool
    expected_pages: list[ExpectedPage] = Field(default_factory=list)
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    notes: str = ""
    human_verified: bool = False

    @model_validator(mode="after")
    def validate_question_rules(self) -> "GoldenQuestion":
        """Enforces invariants for answerable vs unanswerable/adversarial questions."""
        if self.type in ("fact", "comparison") and not self.answerable:
            raise ValueError(
                f"Question '{self.id}' of type '{self.type}' must have answerable=True."
            )
        if self.type in ("unanswerable", "adversarial") and self.answerable:
            raise ValueError(
                f"Question '{self.id}' of type '{self.type}' must have answerable=False."
            )

        if self.answerable:
            if not self.expected_pages:
                raise ValueError(
                    f"Answerable question '{self.id}' must specify at least one expected_pages entry."
                )
            if not self.must_contain:
                raise ValueError(
                    f"Answerable question '{self.id}' must specify at least one must_contain entry."
                )
        else:
            if self.expected_pages:
                raise ValueError(
                    f"Unanswerable/adversarial question '{self.id}' must have no expected_pages."
                )

        return self


class GoldenSet(BaseModel):
    """Collection of GoldenQuestion items with uniqueness validation on IDs."""

    questions: list[GoldenQuestion]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "GoldenSet":
        """Ensures all question IDs in the golden set are unique."""
        seen: set[str] = set()
        duplicates: list[str] = []
        for q in self.questions:
            if q.id in seen:
                duplicates.append(q.id)
            seen.add(q.id)
        if duplicates:
            raise ValueError(f"Duplicate question IDs in golden set: {duplicates}")
        return self


def load_golden_set(path: Path | str = "data/eval/golden_questions.json") -> list[GoldenQuestion]:
    """Loads and validates the golden evaluation set from a JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Golden question file not found at: {file_path}")

    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        golden = GoldenSet(questions=raw)
    elif isinstance(raw, dict) and "questions" in raw:
        golden = GoldenSet(**raw)
    else:
        raise ValueError(
            "Golden JSON must be a list of question objects or a dict with a 'questions' key."
        )
    return golden.questions


def check_expected_pages_in_store(
    questions: Sequence[GoldenQuestion],
    store: ChromaStore,
) -> list[str]:
    """Checks that expected pages exist in the vector store, returning warnings without failing."""
    warn_messages: list[str] = []
    for q in questions:
        for ep in q.expected_pages:
            quality = store.page_text_quality(ep.source, ep.page)
            if quality is None:
                msg = (
                    f"Warning [{q.id}]: expected page ({ep.source}, PDF page {ep.page}) "
                    "has no chunks in the vector store."
                )
                warn_messages.append(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
    return warn_messages
