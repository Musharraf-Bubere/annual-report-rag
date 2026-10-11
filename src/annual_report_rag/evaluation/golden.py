"""Pydantic schemas and loader for the golden evaluation question set."""

import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from annual_report_rag.retrieval.store import ChromaStore

QuestionType = Literal[
    "fact", "comparison", "table", "unanswerable", "adversarial"
]
SplitType = Literal["dev", "test"]
StatusType = Literal["draft_needs_fill", "ready_unverified", "verified"]


class ExpectedPage(BaseModel):
    """Identifies a source document stem and 1-indexed PDF page number."""

    source: str = Field(..., min_length=1)
    page: int = Field(..., ge=1)


def normalize_question_dict(raw: dict[str, object]) -> dict[str, object]:
    """Maps a neutral draft question dict (expected_pdf_pages, expected_answer_all_of) to GoldenQuestion fields."""
    data = dict(raw)
    qtype = str(data.get("type", ""))

    if "answerable" not in data:
        data["answerable"] = qtype in ("fact", "comparison", "table")

    if "expected_pages" not in data and "expected_pdf_pages" in data:
        pdf_pages = data.pop("expected_pdf_pages")
        pages_list: list[dict[str, object]] = []
        if isinstance(pdf_pages, dict):
            for src, page_nums in pdf_pages.items():
                if isinstance(page_nums, list):
                    for p in page_nums:
                        pages_list.append({"source": str(src), "page": int(p)})
        data["expected_pages"] = pages_list
    else:
        data.pop("expected_pdf_pages", None)

    if "must_contain" not in data and "expected_answer_all_of" in data:
        all_of = data.pop("expected_answer_all_of")
        must_contain: list[str] = []
        if isinstance(all_of, list):
            for group in all_of:
                if isinstance(group, list):
                    cleaned = [str(item).strip() for item in group if str(item).strip()]
                    if cleaned:
                        must_contain.append("|".join(cleaned))
                elif isinstance(group, str) and group.strip():
                    must_contain.append(group.strip())
        data["must_contain"] = must_contain
    else:
        data.pop("expected_answer_all_of", None)

    data.pop("expected_refusal", None)

    if "expected_sources" not in data or not data["expected_sources"]:
        ep_list = data.get("expected_pages", [])
        if isinstance(ep_list, list):
            inferred: list[str] = []
            for ep in ep_list:
                src = ep.get("source") if isinstance(ep, dict) else getattr(ep, "source", None)
                if isinstance(src, str) and src and src not in inferred:
                    inferred.append(src)
            if inferred:
                data["expected_sources"] = inferred

    return data


class GoldenQuestion(BaseModel):
    """Represents a single evaluation question with expected provenance and lexical checks."""

    id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    type: QuestionType
    answerable: bool
    expected_sources: list[str] = Field(default_factory=list)
    expected_pages: list[ExpectedPage] = Field(default_factory=list)
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    notes: str = ""
    human_verified: bool = False
    split: SplitType = "dev"
    status: StatusType = "ready_unverified"

    @model_validator(mode="before")
    @classmethod
    def _normalize_draft_fields(cls, values: object) -> object:
        """Normalizes neutral draft schema fields into GoldenQuestion fields before validation."""
        if isinstance(values, dict):
            return normalize_question_dict(values)
        return values

    @model_validator(mode="after")
    def validate_question_rules(self) -> "GoldenQuestion":
        """Enforces invariants for answerable vs unanswerable/adversarial and draft vs ready questions."""
        if self.type in ("fact", "comparison", "table") and not self.answerable:
            raise ValueError(
                f"Question '{self.id}' of type '{self.type}' must have answerable=True."
            )
        if self.type in ("unanswerable", "adversarial") and self.answerable:
            raise ValueError(
                f"Question '{self.id}' of type '{self.type}' must have answerable=False."
            )

        if self.status == "draft_needs_fill" and self.human_verified:
            raise ValueError(
                f"Question '{self.id}' with status='draft_needs_fill' cannot have human_verified=True."
            )

        if self.answerable:
            if self.status != "draft_needs_fill":
                if not self.expected_pages:
                    raise ValueError(
                        f"Answerable question '{self.id}' (status='{self.status}') must specify at least one expected_pages entry."
                    )
                if not self.must_contain:
                    raise ValueError(
                        f"Answerable question '{self.id}' (status='{self.status}') must specify at least one must_contain entry."
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
