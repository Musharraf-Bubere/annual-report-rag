"""LLM-as-a-judge faithfulness and citation-support evaluator with caching and calibration."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from annual_report_rag.providers.base import GenerationProvider

JUDGE_SYSTEM_PROMPT = (
    "You grade whether an ANSWER is supported by the numbered PASSAGES it cites. "
    "Use only those passages. For each citation number, say whether that passage "
    "supports at least one claim in the answer that cites it. List every claim in "
    "the answer that no cited passage supports. A change of number format only "
    "(6,17,437 vs 617,437) is NOT a problem; a different number, label or unit IS. "
    "Output JSON only: "
    "{faithful: bool, unsupported_claims: [str], citations: [{n: int, supports: bool}]}"
)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class CitationSupport(BaseModel):
    """Indicates whether a cited passage [n] supports at least one claim citing it."""

    n: int
    supports: bool

    def __getitem__(self, key: str) -> Any:
        """Allows dict-style access (c['n'], c['supports']) in addition to attribute access."""
        return getattr(self, key)


class _RawJudgePayload(BaseModel):
    """Pydantic schema for validating raw JSON output from the judge model."""

    faithful: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    citations: list[CitationSupport] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    """Structured result returned by judge_answer."""

    faithful: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    citations: list[CitationSupport] = Field(default_factory=list)
    parse_ok: bool = True


class CalibrationPassage(BaseModel):
    """A numbered synthetic passage in judge_calibration.json."""

    n: int
    text: str


class CalibrationCase(BaseModel):
    """A synthetic test case for calibrating the LLM faithfulness judge."""

    id: str
    description: str
    question: str
    answer: str
    cited_passages: list[CalibrationPassage]
    expected_faithful: bool | None = None
    expected_citation_supports: dict[int, bool] = Field(default_factory=dict)


def normalize_cited_passages(
    cited_passages: (
        Mapping[int | str, str]
        | Sequence[tuple[int, str] | Mapping[str, Any] | CalibrationPassage | str]
    ),
) -> list[tuple[int, str]]:
    """Normalizes flexible cited_passages inputs into a sorted list of (n, full_text) pairs."""
    if isinstance(cited_passages, Mapping):
        return [(int(k), str(v)) for k, v in cited_passages.items()]

    pairs: list[tuple[int, str]] = []
    for idx, item in enumerate(cited_passages, start=1):
        if isinstance(item, CalibrationPassage):
            pairs.append((item.n, item.text))
        elif isinstance(item, tuple) and len(item) == 2:
            pairs.append((int(item[0]), str(item[1])))
        elif isinstance(item, Mapping):
            pairs.append((int(item["n"]), str(item["text"])))
        else:
            pairs.append((idx, str(item)))
    return pairs


def build_judge_prompt(
    question: str,
    answer: str,
    cited_passages: (
        Mapping[int | str, str]
        | Sequence[tuple[int, str] | Mapping[str, Any] | CalibrationPassage | str]
    ),
) -> str:
    """Formats the user prompt containing the question, answer, and full cited passages."""
    pairs = normalize_cited_passages(cited_passages)
    passage_blocks = "\n\n".join(f"[{n}]\n{text}" for n, text in pairs)
    return (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"PASSAGES:\n{passage_blocks}\n"
    )


def _extract_json_string(raw_text: str) -> str:
    """Extracts JSON text from raw LLM output, stripping optional Markdown code fences."""
    cleaned = raw_text.strip()
    fence_match = _CODE_FENCE_RE.search(cleaned)
    if fence_match:
        return fence_match.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def parse_judge_response(raw_text: str) -> JudgeVerdict | None:
    """Attempts to parse and Pydantic-validate a judge response, returning None on failure."""
    if not raw_text or not raw_text.strip():
        return None
    candidate = _extract_json_string(raw_text)
    try:
        data = json.loads(candidate)
        validated = _RawJudgePayload.model_validate(data)
        return JudgeVerdict(
            faithful=validated.faithful,
            unsupported_claims=validated.unsupported_claims,
            citations=validated.citations,
            parse_ok=True,
        )
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def judge_answer(
    question: str,
    answer: str,
    cited_passages: (
        Mapping[int | str, str]
        | Sequence[tuple[int, str] | Mapping[str, Any] | CalibrationPassage | str]
    ),
    provider: GenerationProvider,
) -> JudgeVerdict:
    """Grades whether an answer is supported by its cited passages, retrying once on parse failure."""
    prompt = build_judge_prompt(question, answer, cited_passages)

    for _attempt in range(2):
        gen_result = provider.generate(prompt, system=JUDGE_SYSTEM_PROMPT)
        verdict = parse_judge_response(gen_result.text)
        if verdict is not None:
            return verdict

    return JudgeVerdict(
        faithful=False,
        unsupported_claims=[],
        citations=[],
        parse_ok=False,
    )


def compute_judge_cache_key(
    question: str,
    answer: str,
    cited_passages: (
        Mapping[int | str, str]
        | Sequence[tuple[int, str] | Mapping[str, Any] | CalibrationPassage | str]
    ),
    judge_model: str,
    judge_prompt: str = JUDGE_SYSTEM_PROMPT,
) -> str:
    """Computes a deterministic cache key including the judge prompt hash and judge model."""
    prompt_hash = hashlib.sha256(judge_prompt.encode("utf-8")).hexdigest()
    pairs = normalize_cited_passages(cited_passages)
    payload = {
        "question": question,
        "answer": answer,
        "cited_passages": pairs,
        "judge_prompt_hash": prompt_hash,
        "judge_model": judge_model,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def judge_answer_cached(
    question_id: str,
    question: str,
    answer: str,
    cited_passages: (
        Mapping[int | str, str]
        | Sequence[tuple[int, str] | Mapping[str, Any] | CalibrationPassage | str]
    ),
    provider: GenerationProvider,
    judge_model: str,
    cache_dir: Path,
    use_cache: bool = True,
) -> tuple[JudgeVerdict, bool]:
    """Runs judge_answer with disk caching in cache_dir, returning (verdict, from_cache)."""
    jkey = compute_judge_cache_key(question, answer, cited_passages, judge_model)
    cache_file = cache_dir / f"judge_{question_id}_{jkey[:16]}.json"

    if use_cache and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("cache_key") == jkey:
                verdict = JudgeVerdict.model_validate(cached["verdict"])
                return verdict, True
        except (json.JSONDecodeError, ValidationError, KeyError):
            pass

    verdict = judge_answer(question, answer, cited_passages, provider)
    if use_cache and verdict.parse_ok:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(
                {"cache_key": jkey, "verdict": verdict.model_dump()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return verdict, False


def load_calibration_cases(
    path: Path | str = "data/eval/judge_calibration.json",
) -> list[CalibrationCase]:
    """Loads and validates the synthetic judge calibration cases from JSON."""
    file_path = Path(path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    return [CalibrationCase.model_validate(item) for item in raw]


def evaluate_calibration_case(
    case: CalibrationCase,
    verdict: JudgeVerdict,
) -> bool:
    """Checks whether a JudgeVerdict matches the expected faithfulness and citation support on a calibration case."""
    if not verdict.parse_ok:
        return False
    if case.expected_faithful is not None and verdict.faithful != case.expected_faithful:
        return False
    actual_cit_map = {c.n: c.supports for c in verdict.citations}
    for exp_n, exp_sup in case.expected_citation_supports.items():
        if actual_cit_map.get(int(exp_n)) != exp_sup:
            return False
    return True
