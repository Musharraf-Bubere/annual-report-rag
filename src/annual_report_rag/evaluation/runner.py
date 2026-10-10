"""Evaluation runner orchestrating retrieval, generation caching, metrics aggregation, and report writing."""

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from annual_report_rag.config import Settings
from annual_report_rag.evaluation.golden import GoldenQuestion
from annual_report_rag.evaluation.metrics import (
    outcome,
     p50,
    p95,
    page_hit_at_k,
    reciprocal_rank,
    text_hit_at_k,
)
from annual_report_rag.evaluation.reporting import (
    CitationRef,
    QuestionOutcomeRecord,
    RetrievedRef,
    format_markdown_report,
    is_question_failed,
)
from annual_report_rag.generation.answerer import (
    AnswerResult,
    Citation,
    QueryDeps,
    Timings,
    answer_question,
)
from annual_report_rag.generation.prompts import SYSTEM_PROMPT
from annual_report_rag.providers.base import (
    DailyQuotaExceeded,
    EmbeddingProvider,
    GenerationProvider,
)
from annual_report_rag.providers.factory import get_embedder, get_generator
from annual_report_rag.retrieval.retriever import RetrievedChunk, retrieve
from annual_report_rag.retrieval.store import ChromaStore


@dataclass(frozen=True)
class EvalRunOutput:
    """Returned summary and file paths from a completed or partial evaluation run."""

    label: str
    timestamp: str
    status: str
    quota_exceeded: bool
    summary: dict[str, Any]
    markdown_report: str
    run_json_path: Path
    metrics_json_path: Path
    markdown_path: Path


def compute_settings_hash(settings: Settings) -> str:
    """Computes a SHA-256 hash over all non-secret settings."""
    raw = asdict(settings)
    raw.pop("gemini_api_key", None)
    serialized = json.dumps(raw, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_cache_key(
    question: GoldenQuestion,
    settings_hash: str,
    prompt_hash: str,
    store_fingerprint: str,
) -> str:
    """Computes the deterministic cache key for a single question and pipeline state."""
    payload = {
        "id": question.id,
        "question": question.question,
        "settings_hash": settings_hash,
        "prompt_hash": prompt_hash,
        "store_fingerprint": store_fingerprint,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def classify_expected_page_quality(
    question: GoldenQuestion,
    store: ChromaStore,
) -> str:
    """Classifies an answerable question by whether its expected pages are all_clean, all_degraded, or mixed."""
    if not question.expected_pages:
        return "none"
    qualities = [
        store.page_text_quality(ep.source, ep.page) or "clean"
        for ep in question.expected_pages
    ]
    if all(q == "clean" for q in qualities):
        return "all_clean"
    if all(q == "degraded" for q in qualities):
        return "all_degraded"
    return "mixed"


def _serialize_answer_result(res: AnswerResult) -> dict[str, Any]:
    """Serializes an AnswerResult to a JSON-compatible dictionary for local caching."""
    return asdict(res)


def _deserialize_answer_result(data: dict[str, Any]) -> AnswerResult:
    """Reconstructs an AnswerResult from a cached JSON dictionary."""
    return AnswerResult(
        answer=str(data["answer"]),
        grounded=bool(data["grounded"]),
        citations=[Citation(**c) for c in data.get("citations", [])],
        retrieved=[RetrievedChunk(**r) for r in data.get("retrieved", [])],
        detected_sources=list(data.get("detected_sources", [])),
        invalid_citations=int(data.get("invalid_citations", 0)),
        unsupported_numbers=list(data.get("unsupported_numbers", [])),
        timings=Timings(**data["timings"]),
        model=str(data.get("model", "")),
        input_tokens=data.get("input_tokens"),
        output_tokens=data.get("output_tokens"),
        reason=data.get("reason"),
    )


def _evaluate_single_question(
    q: GoldenQuestion,
    deps: QueryDeps,
    retrieval_only: bool,
    use_cache: bool,
    cache_dir: Path,
    settings_hash: str,
    prompt_hash: str,
    store_fp: str,
) -> tuple[QuestionOutcomeRecord, dict[str, Any]]:
    """Evaluates a single question and returns its committed outcome record and full detail dict."""
    q_slice = classify_expected_page_quality(q, deps.store)

    if retrieval_only:
        t0 = time.perf_counter()
        chunks = retrieve(
            question=q.question,
            embedder=deps.embedder,
            store=deps.store,
            settings=deps.settings,
        )
        retrieve_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        ph1 = page_hit_at_k(chunks, q.expected_pages, 1) if q.answerable else None
        ph3 = page_hit_at_k(chunks, q.expected_pages, 3) if q.answerable else None
        ph6 = page_hit_at_k(chunks, q.expected_pages, 6) if q.answerable else None
        rr = round(reciprocal_rank(chunks, q.expected_pages), 4) if q.answerable else None
        th6 = text_hit_at_k(chunks, q.must_contain, 6) if q.answerable else None

        top6_refs = [
            RetrievedRef(
                chunk_id=c.chunk_id,
                source=c.source,
                page=c.page,
                score=c.score,
                text_quality=c.text_quality,
            )
            for c in chunks[:6]
        ]
        rec = QuestionOutcomeRecord(
            id=q.id,
            question=q.question,
            type=q.type,
            answerable=q.answerable,
            human_verified=q.human_verified,
            expected_quality_slice=q_slice,
            page_hit_at_1=ph1,
            page_hit_at_3=ph3,
            page_hit_at_6=ph6,
            reciprocal_rank=rr,
            text_hit_at_6=th6,
            outcome=None,
            grounded=None,
            reason=None,
            invalid_citations=0,
            unsupported_numbers=[],
            retrieve_ms=retrieve_ms,
            generate_ms=0.0,
            total_ms=retrieve_ms,
            input_tokens=None,
            output_tokens=None,
            from_cache=False,
            retrieved_top6=top6_refs,
            citations=[],
        )
        detail = {
            "question": q.model_dump(),
            "outcome_record": asdict(rec),
            "retrieved_chunks": [asdict(c) for c in chunks],
            "answer": None,
        }
        return rec, detail

    # Full mode (with optional cache)
    ckey = compute_cache_key(q, settings_hash, prompt_hash, store_fp)
    cache_file = cache_dir / f"{q.id}_{ckey[:16]}.json"
    from_cache = False

    if use_cache and cache_file.exists():
        cached_payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached_payload.get("cache_key") == ckey:
            ans_res = _deserialize_answer_result(cached_payload["answer_result"])
            from_cache = True
        else:
            ans_res = answer_question(q.question, deps)
    else:
        ans_res = answer_question(q.question, deps)
        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps(
                    {
                        "cache_key": ckey,
                        "answer_result": _serialize_answer_result(ans_res),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    chunks = ans_res.retrieved
    ph1 = page_hit_at_k(chunks, q.expected_pages, 1) if q.answerable else None
    ph3 = page_hit_at_k(chunks, q.expected_pages, 3) if q.answerable else None
    ph6 = page_hit_at_k(chunks, q.expected_pages, 6) if q.answerable else None
    rr = round(reciprocal_rank(chunks, q.expected_pages), 4) if q.answerable else None
    th6 = text_hit_at_k(chunks, q.must_contain, 6) if q.answerable else None
    q_outcome = outcome(q, ans_res)

    top6_refs = [
        RetrievedRef(
            chunk_id=c.chunk_id,
            source=c.source,
            page=c.page,
            score=c.score,
            text_quality=c.text_quality,
        )
        for c in chunks[:6]
    ]
    cit_refs = [
        CitationRef(
            n=c.n,
            chunk_id=c.chunk_id,
            source=c.source,
            page=c.page,
            text_quality=c.text_quality,
        )
        for c in ans_res.citations
    ]
    rec = QuestionOutcomeRecord(
        id=q.id,
        question=q.question,
        type=q.type,
        answerable=q.answerable,
        human_verified=q.human_verified,
        expected_quality_slice=q_slice,
        page_hit_at_1=ph1,
        page_hit_at_3=ph3,
        page_hit_at_6=ph6,
        reciprocal_rank=rr,
        text_hit_at_6=th6,
        outcome=q_outcome,
        grounded=ans_res.grounded,
        reason=ans_res.reason,
        invalid_citations=ans_res.invalid_citations,
        unsupported_numbers=ans_res.unsupported_numbers,
        retrieve_ms=ans_res.timings.retrieve_ms,
        generate_ms=ans_res.timings.generate_ms,
        total_ms=ans_res.timings.total_ms,
        input_tokens=ans_res.input_tokens,
        output_tokens=ans_res.output_tokens,
        from_cache=from_cache,
        retrieved_top6=top6_refs,
        citations=cit_refs,
    )
    detail = {
        "question": q.model_dump(),
        "outcome_record": asdict(rec),
        "answer": ans_res.answer,
        "retrieved_chunks": [asdict(c) for c in chunks],
        "citations_full": [asdict(c) for c in ans_res.citations],
    }
    return rec, detail


def _aggregate_summary(
    records: Sequence[QuestionOutcomeRecord],
    settings: Settings,
    label: str,
    timestamp: str,
    retrieval_only: bool,
    status: str,
    store_fp: str,
) -> dict[str, Any]:
    """Aggregates retrieval, answer, performance, slice, and failure metrics without any PDF text."""
    answerable_recs = [r for r in records if r.answerable]
    n_ans = len(answerable_recs)

    def _mean_bool(vals: Sequence[bool | None]) -> float:
        clean = [1.0 if v else 0.0 for v in vals if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else 0.0

    def _mean_float(vals: Sequence[float | None]) -> float:
        clean = [float(v) for v in vals if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else 0.0

    retrieval_metrics = {
        "answerable_count": n_ans,
        "page_hit_at_1": _mean_bool([r.page_hit_at_1 for r in answerable_recs]),
        "page_hit_at_3": _mean_bool([r.page_hit_at_3 for r in answerable_recs]),
        "page_hit_at_6": _mean_bool([r.page_hit_at_6 for r in answerable_recs]),
        "mrr": _mean_float([r.reciprocal_rank for r in answerable_recs]),
        "text_hit_at_6": _mean_bool([r.text_hit_at_6 for r in answerable_recs]),
    }

    outcome_counts = {
        "CORRECT": 0,
        "WRONG": 0,
        "FALSE_REFUSAL": 0,
        "CORRECT_REFUSAL": 0,
        "ANSWERED_ANYWAY": 0,
    }
    for r in records:
        if r.outcome in outcome_counts:
            outcome_counts[r.outcome] += 1

    unsupported_qids = [r.id for r in records if r.unsupported_numbers]
    answer_metrics = {
        "outcome_counts": outcome_counts,
        "grounded_rate": _mean_bool([r.grounded for r in answerable_recs]),
        "questions_with_unsupported_numbers": len(unsupported_qids),
        "unsupported_number_question_ids": unsupported_qids,
    }

    ret_lat = [r.retrieve_ms for r in records]
    gen_lat = [r.generate_ms for r in records if not retrieval_only]
    tot_lat = [r.total_ms for r in records]
    in_toks = [r.input_tokens for r in records if r.input_tokens is not None]
    out_toks = [r.output_tokens for r in records if r.output_tokens is not None]
    avg_in = round(sum(in_toks) / len(in_toks), 1) if in_toks else 0.0
    avg_out = round(sum(out_toks) / len(out_toks), 1) if out_toks else 0.0

    price_in = settings.evaluation.price_per_million_input_usd
    price_out = settings.evaluation.price_per_million_output_usd
    avg_cost: float | None = None
    if price_in is not None and price_out is not None and (in_toks or out_toks):
        avg_cost = round(
            (avg_in / 1_000_000.0) * price_in + (avg_out / 1_000_000.0) * price_out,
            6,
        )

    performance_metrics = {
        "retrieve_p50_ms": p50(ret_lat),
        "retrieve_p95_ms": p95(ret_lat),
        "generate_p50_ms": p50(gen_lat),
        "generate_p95_ms": p95(gen_lat),
        "total_p50_ms": p50(tot_lat),
        "total_p95_ms": p95(tot_lat),
        "avg_input_tokens": avg_in,
        "avg_output_tokens": avg_out,
        "avg_cost_per_query_usd": avg_cost,
    }

    by_type: dict[str, Any] = {}
    for qtype in ("fact", "comparison", "unanswerable", "adversarial"):
        sub = [r for r in records if r.type == qtype]
        if not sub:
            continue
        is_ans_type = qtype in ("fact", "comparison")
        pass_target = "CORRECT" if is_ans_type else "CORRECT_REFUSAL"
        by_type[qtype] = {
            "count": len(sub),
            "page_hit_at_6": _mean_bool([r.page_hit_at_6 for r in sub]) if is_ans_type else None,
            "text_hit_at_6": _mean_bool([r.text_hit_at_6 for r in sub]) if is_ans_type else None,
            "pass_count": sum(1 for r in sub if r.outcome == pass_target) if not retrieval_only else None,
        }

    by_quality: dict[str, Any] = {}
    for qslice in ("all_clean", "all_degraded", "mixed"):
        sub = [r for r in answerable_recs if r.expected_quality_slice == qslice]
        by_quality[qslice] = {
            "count": len(sub),
            "page_hit_at_6": _mean_bool([r.page_hit_at_6 for r in sub]) if sub else 0.0,
            "mrr": _mean_float([r.reciprocal_rank for r in sub]) if sub else 0.0,
            "text_hit_at_6": _mean_bool([r.text_hit_at_6 for r in sub]) if sub else 0.0,
            "correct_count": (
                sum(1 for r in sub if r.outcome == "CORRECT")
                if (sub and not retrieval_only)
                else None
            ),
        }

    failed_list = [
        asdict(r) for r in records if is_question_failed(r, retrieval_only)
    ]

    return {
        "label": label,
        "timestamp": timestamp,
        "retrieval_only": retrieval_only,
        "status": status,
        "store_fingerprint": store_fp,
        "num_questions": len(records),
        "all_human_verified": bool(records) and all(r.human_verified for r in records),
        "retrieval": retrieval_metrics,
        "answers": answer_metrics,
        "performance": performance_metrics,
        "slices": {
            "by_type": by_type,
            "by_expected_page_quality": by_quality,
        },
        "failed_questions": failed_list,
        "questions": [asdict(r) for r in records],
    }


def run_eval(
    golden: Sequence[GoldenQuestion],
    settings: Settings,
    label: str = "eval",
    retrieval_only: bool = False,
    only_type: str | None = None,
    limit: int | None = None,
    use_cache: bool = True,
    *,
    store: ChromaStore | None = None,
    embedder: EmbeddingProvider | None = None,
    generator: GenerationProvider | None = None,
    eval_dir: Path | str | None = None,
) -> EvalRunOutput:
    """Executes the evaluation harness, caches answers, and writes run and result files."""
    active_store = store or ChromaStore(
        persist_path=settings.storage.vector_store_path,
        collection_name=settings.storage.collection_name,
        embedding_provider=settings.embedding.provider,
        embedding_model=settings.embedding.model,
        embedding_dimension=settings.embedding.dimension,
    )
    active_embedder = embedder or get_embedder(settings)
    if retrieval_only:
        from annual_report_rag.providers.fake import FakeProvider

        active_generator = generator or FakeProvider(
            embedding_dim=settings.embedding.dimension
        )
    else:
        active_generator = generator or get_generator(settings)

    deps = QueryDeps(
        settings=settings,
        store=active_store,
        embedder=active_embedder,
        generator=active_generator,
    )

    questions = [
        q for q in golden if only_type is None or q.type == only_type
    ]
    if limit is not None and limit >= 0:
        questions = questions[:limit]

    # One throwaway query before timing so model initialization is not counted
    retrieve("warmup query", active_embedder, active_store, settings, top_k=1)

    base_eval_dir = Path(eval_dir) if eval_dir is not None else Path(settings.storage.eval_data_dir)
    runs_dir = base_eval_dir / "runs"
    results_dir = base_eval_dir / "results"
    cache_dir = base_eval_dir / "cache"
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    settings_hash = compute_settings_hash(settings)
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    store_fp = active_store.fingerprint()

    records: list[QuestionOutcomeRecord] = []
    details: list[dict[str, Any]] = []
    quota_exceeded = False
    status = "completed"

    for q in questions:
        try:
            rec, detail = _evaluate_single_question(
                q=q,
                deps=deps,
                retrieval_only=retrieval_only,
                use_cache=use_cache,
                cache_dir=cache_dir,
                settings_hash=settings_hash,
                prompt_hash=prompt_hash,
                store_fp=store_fp,
            )
            records.append(rec)
            details.append(detail)
        except DailyQuotaExceeded:
            quota_exceeded = True
            status = "partial_daily_quota_exceeded"
            break

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)
    stem = f"{timestamp}_{safe_label}"

    summary = _aggregate_summary(
        records=records,
        settings=settings,
        label=label,
        timestamp=timestamp,
        retrieval_only=retrieval_only,
        status=status,
        store_fp=store_fp,
    )
    md_report = format_markdown_report(summary)

    run_json_path = runs_dir / f"{stem}.json"
    metrics_json_path = results_dir / f"{stem}.metrics.json"
    markdown_path = results_dir / f"{stem}.md"

    run_json_path.write_text(
        json.dumps(
            {"summary": summary, "details": details},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(md_report, encoding="utf-8")

    return EvalRunOutput(
        label=label,
        timestamp=timestamp,
        status=status,
        quota_exceeded=quota_exceeded,
        summary=summary,
        markdown_report=md_report,
        run_json_path=run_json_path,
        metrics_json_path=metrics_json_path,
        markdown_path=markdown_path,
    )
