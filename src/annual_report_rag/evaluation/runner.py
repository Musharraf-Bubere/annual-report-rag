"""Evaluation runner orchestrating retrieval, generation caching, repeats, LLM judging, metrics aggregation, and report writing."""

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
from annual_report_rag.evaluation.judge import judge_answer_cached
from annual_report_rag.evaluation.metrics import (
    distinct_pages_at_k,
    flip_rate,
    max_val,
    outcome,
    p50,
    p95,
    page_hit_at_k,
    reciprocal_rank,
    text_hit_all_at_context,
    text_hit_all_at_k,
    text_hit_any_at_context,
    text_hit_any_at_k,
)
from annual_report_rag.evaluation.reporting import (
    CitationRef,
    QuestionOutcomeRecord,
    RetrievedRef,
    format_judge_review_markdown,
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
from annual_report_rag.generation.checks import find_unsupported_numbers
from annual_report_rag.generation.prompts import get_system_prompt
from annual_report_rag.providers.base import (
    DailyQuotaExceeded,
    EmbeddingProvider,
    GenerationProvider,
)
from annual_report_rag.providers.factory import get_embedder, get_generator
from annual_report_rag.retrieval.retriever import RetrievedChunk, retrieve
from annual_report_rag.retrieval.router import detect_sources
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
    judge_review_path: Path | None = None


def compute_settings_hash(settings: Settings) -> str:
    """Computes a SHA-256 hash over all non-secret settings."""
    raw = asdict(settings)
    raw.pop("gemini_api_key", None)
    ret_dict = raw.get("retrieval")
    if isinstance(ret_dict, dict):
        if ret_dict.get("query_prefix") is None:
            ret_dict.pop("query_prefix", None)
        if ret_dict.get("max_chunks_per_page") is None:
            ret_dict.pop("max_chunks_per_page", None)
        if not ret_dict.get("neighbor_window"):
            ret_dict.pop("neighbor_window", None)
        if ret_dict.get("neighbor_top_n", 3) == 3:
            ret_dict.pop("neighbor_top_n", None)
        if ret_dict.get("max_context_chunks") is None:
            ret_dict.pop("max_context_chunks", None)
        hybrid_dict = ret_dict.get("hybrid")
        if isinstance(hybrid_dict, dict) and not hybrid_dict.get("enabled"):
            ret_dict.pop("hybrid", None)
    prompt_dict = raw.get("prompt")
    if isinstance(prompt_dict, dict) and not prompt_dict.get("rules_v2"):
        raw.pop("prompt", None)
    serialized = json.dumps(raw, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_cache_key(
    question: GoldenQuestion,
    settings_hash: str,
    prompt_hash: str,
    store_fingerprint: str,
    repeat_index: int = 0,
) -> str:
    """Computes the deterministic cache key for a single question, pipeline state, and repeat index."""
    payload: dict[str, Any] = {
        "id": question.id,
        "question": question.question,
        "settings_hash": settings_hash,
        "prompt_hash": prompt_hash,
        "store_fingerprint": store_fingerprint,
    }
    if repeat_index > 0:
        payload["repeat_index"] = int(repeat_index)
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
    """Reconstructs an AnswerResult from a cached JSON dictionary, recomputing unsupported_numbers with the current checker."""
    answer = str(data["answer"])
    grounded = bool(data["grounded"])
    citations = [Citation(**c) for c in data.get("citations", [])]
    retrieved = [RetrievedChunk(**r) for r in data.get("retrieved", [])]
    if grounded and citations and retrieved:
        chunk_by_id = {c.chunk_id: c for c in retrieved}
        cited_chunks = [
            chunk_by_id[c.chunk_id] for c in citations if c.chunk_id in chunk_by_id
        ]
        unsupported_numbers = find_unsupported_numbers(answer, cited_chunks)
    else:
        unsupported_numbers = []

    return AnswerResult(
        answer=answer,
        grounded=grounded,
        citations=citations,
        retrieved=retrieved,
        detected_sources=list(data.get("detected_sources", [])),
        invalid_citations=int(data.get("invalid_citations", 0)),
        unsupported_numbers=unsupported_numbers,
        timings=Timings(**data["timings"]),
        model=str(data.get("model", "")),
        input_tokens=data.get("input_tokens"),
        output_tokens=data.get("output_tokens"),
        reason=data.get("reason"),
    )


def _compute_retrieval_fields(
    q: GoldenQuestion,
    chunks: Sequence[RetrievedChunk],
    num_sources: int,
) -> dict[str, Any]:
    """Computes primary-ranking retrieval hit metrics, distinct_pages@6, and full-context text_hit@context."""
    primary_chunks = [c for c in chunks if c.neighbor_of is None]
    dp6 = distinct_pages_at_k(primary_chunks, k=6)
    ctx_count = len(chunks)
    if not q.answerable:
        return {
            "page_hit_at_1": None,
            "page_hit_at_3": None,
            "page_hit_at_6": None,
            "reciprocal_rank": None,
            "text_hit_at_6": None,
            "text_hit_any_at_1": None,
            "text_hit_any_at_3": None,
            "text_hit_any_at_6": None,
            "text_hit_all_at_1": None,
            "text_hit_all_at_3": None,
            "text_hit_all_at_6": None,
            "text_hit_any_at_context": None,
            "text_hit_all_at_context": None,
            "context_chunks_count": ctx_count,
            "distinct_pages_at_6": dp6,
        }

    tha1 = text_hit_any_at_k(primary_chunks, q.must_contain, 1, num_sources=num_sources)
    tha3 = text_hit_any_at_k(primary_chunks, q.must_contain, 3, num_sources=num_sources)
    tha6 = text_hit_any_at_k(primary_chunks, q.must_contain, 6, num_sources=num_sources)
    return {
        "page_hit_at_1": page_hit_at_k(primary_chunks, q.expected_pages, 1, num_sources=num_sources),
        "page_hit_at_3": page_hit_at_k(primary_chunks, q.expected_pages, 3, num_sources=num_sources),
        "page_hit_at_6": page_hit_at_k(primary_chunks, q.expected_pages, 6, num_sources=num_sources),
        "reciprocal_rank": round(reciprocal_rank(primary_chunks, q.expected_pages), 4),
        "text_hit_at_6": tha6,
        "text_hit_any_at_1": tha1,
        "text_hit_any_at_3": tha3,
        "text_hit_any_at_6": tha6,
        "text_hit_all_at_1": text_hit_all_at_k(primary_chunks, q.must_contain, 1, num_sources=num_sources),
        "text_hit_all_at_3": text_hit_all_at_k(primary_chunks, q.must_contain, 3, num_sources=num_sources),
        "text_hit_all_at_6": text_hit_all_at_k(primary_chunks, q.must_contain, 6, num_sources=num_sources),
        "text_hit_any_at_context": text_hit_any_at_context(chunks, q.must_contain),
        "text_hit_all_at_context": text_hit_all_at_context(chunks, q.must_contain),
        "context_chunks_count": ctx_count,
        "distinct_pages_at_6": dp6,
    }


def _evaluate_single_question(
    q: GoldenQuestion,
    deps: QueryDeps,
    retrieval_only: bool,
    use_cache: bool,
    cache_dir: Path,
    settings_hash: str,
    prompt_hash: str,
    store_fp: str,
    repeat_index: int = 0,
    use_judge: bool = False,
    judge_provider: GenerationProvider | None = None,
) -> tuple[QuestionOutcomeRecord, dict[str, Any]]:
    """Evaluates a single question and repeat, returning its committed outcome record and full detail dict."""
    q_slice = classify_expected_page_quality(q, deps.store)
    det_sources = detect_sources(q.question, deps.settings)
    num_sources = max(1, len(det_sources))

    if retrieval_only:
        t0 = time.perf_counter()
        chunks = retrieve(
            question=q.question,
            embedder=deps.embedder,
            store=deps.store,
            settings=deps.settings,
        )
        retrieve_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        r_fields = _compute_retrieval_fields(q, chunks, num_sources)
        primary_chunks = [c for c in chunks if c.neighbor_of is None]

        top6_refs = [
            RetrievedRef(
                chunk_id=c.chunk_id,
                source=c.source,
                page=c.page,
                score=c.score,
                text_quality=c.text_quality,
                neighbor_of=c.neighbor_of,
            )
            for c in primary_chunks[:6]
        ]
        rec = QuestionOutcomeRecord(
            id=q.id,
            question=q.question,
            type=q.type,
            answerable=q.answerable,
            human_verified=q.human_verified,
            expected_quality_slice=q_slice,
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
            split=q.split,
            repeat_index=repeat_index,
            detected_sources_count=num_sources,
            status=q.status,
            **r_fields,
        )
        detail = {
            "question": q.model_dump(),
            "repeat_index": repeat_index,
            "outcome_record": asdict(rec),
            "retrieved_chunks": [asdict(c) for c in chunks],
            "answer": None,
            "judge_verdict": None,
            "cited_passages_full": [],
        }
        return rec, detail

    # Full mode (with optional cache)
    ckey = compute_cache_key(
        q, settings_hash, prompt_hash, store_fp, repeat_index=repeat_index
    )
    prefix = f"{q.id}_{ckey[:16]}" if repeat_index == 0 else f"{q.id}_r{repeat_index}_{ckey[:16]}"
    cache_file = cache_dir / f"{prefix}.json"
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
    num_sources = max(1, len(ans_res.detected_sources) or len(det_sources))
    r_fields = _compute_retrieval_fields(q, chunks, num_sources)
    q_outcome = outcome(q, ans_res)

    # Optional LLM judge on grounded=True answers
    judge_faithful: bool | None = None
    judge_parse_ok: bool | None = None
    supporting_cits = 0
    evaluated_cits = 0
    non_supporting_cits: list[int] = []
    judge_verdict_dict: dict[str, Any] | None = None
    cited_passages_full: list[dict[str, Any]] = []

    if use_judge and ans_res.grounded and judge_provider is not None:
        chunk_by_id = {c.chunk_id: c for c in chunks}
        passage_pairs: list[tuple[int, str]] = []
        for cit in ans_res.citations:
            full_chunk = chunk_by_id.get(cit.chunk_id)
            full_text = full_chunk.text if full_chunk is not None else cit.excerpt
            passage_pairs.append((cit.n, full_text))
            cited_passages_full.append(
                {
                    "n": cit.n,
                    "chunk_id": cit.chunk_id,
                    "source": cit.source,
                    "page": cit.page,
                    "text": full_text,
                }
            )

        verdict, _j_cached = judge_answer_cached(
            question_id=f"{q.id}_r{repeat_index}",
            question=q.question,
            answer=ans_res.answer,
            cited_passages=passage_pairs,
            provider=judge_provider,
            judge_model=deps.settings.generation.model,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
        judge_verdict_dict = verdict.model_dump()
        judge_parse_ok = verdict.parse_ok
        if verdict.parse_ok:
            judge_faithful = verdict.faithful
            support_map = {c.n: c.supports for c in verdict.citations}
            evaluated_cits = len(ans_res.citations)
            for cit in ans_res.citations:
                if support_map.get(cit.n, False):
                    supporting_cits += 1
                else:
                    non_supporting_cits.append(cit.n)

    primary_chunks = [c for c in chunks if c.neighbor_of is None]
    top6_refs = [
        RetrievedRef(
            chunk_id=c.chunk_id,
            source=c.source,
            page=c.page,
            score=c.score,
            text_quality=c.text_quality,
            neighbor_of=c.neighbor_of,
        )
        for c in primary_chunks[:6]
    ]
    cit_refs = [
        CitationRef(
            n=c.n,
            chunk_id=c.chunk_id,
            source=c.source,
            page=c.page,
            text_quality=c.text_quality,
            neighbor_of=c.neighbor_of,
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
        split=q.split,
        repeat_index=repeat_index,
        detected_sources_count=num_sources,
        judge_faithful=judge_faithful,
        judge_parse_ok=judge_parse_ok,
        supporting_citations_count=supporting_cits,
        evaluated_citations_count=evaluated_cits,
        non_supporting_citations=non_supporting_cits,
        status=q.status,
        **r_fields,
    )
    detail = {
        "question": q.model_dump(),
        "repeat_index": repeat_index,
        "outcome_record": asdict(rec),
        "answer": ans_res.answer,
        "retrieved_chunks": [asdict(c) for c in chunks],
        "citations_full": [asdict(c) for c in ans_res.citations],
        "judge_verdict": judge_verdict_dict,
        "cited_passages_full": cited_passages_full,
    }
    return rec, detail


def _aggregate_summary(
    all_records: Sequence[QuestionOutcomeRecord],
    settings: Settings,
    label: str,
    timestamp: str,
    retrieval_only: bool,
    status: str,
    store_fp: str,
    split: str,
    repeats: int,
    use_judge: bool,
    include_drafts: bool = False,
    skipped_draft_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Aggregates retrieval, answer, faithfulness, repeat, performance, slice, and failure metrics without any PDF text."""
    primary_records = [r for r in all_records if r.repeat_index == 0]
    if not primary_records and all_records:
        primary_records = list(all_records)

    answerable_recs = [r for r in primary_records if r.answerable]
    n_ans = len(answerable_recs)

    def _mean_bool(vals: Sequence[bool | None]) -> float:
        clean = [1.0 if v else 0.0 for v in vals if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else 0.0

    def _mean_float(vals: Sequence[float | int | None]) -> float:
        clean = [float(v) for v in vals if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else 0.0

    mean_ctx_chunks = round(
        _mean_float([r.context_chunks_count for r in primary_records]), 2
    )
    retrieval_metrics = {
        "answerable_count": n_ans,
        "page_hit_at_1": _mean_bool([r.page_hit_at_1 for r in answerable_recs]),
        "page_hit_at_3": _mean_bool([r.page_hit_at_3 for r in answerable_recs]),
        "page_hit_at_6": _mean_bool([r.page_hit_at_6 for r in answerable_recs]),
        "mrr": _mean_float([r.reciprocal_rank for r in answerable_recs]),
        "text_hit_at_6": _mean_bool([r.text_hit_at_6 for r in answerable_recs]),
        "text_hit_any_at_1": _mean_bool([r.text_hit_any_at_1 for r in answerable_recs]),
        "text_hit_any_at_3": _mean_bool([r.text_hit_any_at_3 for r in answerable_recs]),
        "text_hit_any_at_6": _mean_bool([r.text_hit_any_at_6 for r in answerable_recs]),
        "text_hit_all_at_1": _mean_bool([r.text_hit_all_at_1 for r in answerable_recs]),
        "text_hit_all_at_3": _mean_bool([r.text_hit_all_at_3 for r in answerable_recs]),
        "text_hit_all_at_6": _mean_bool([r.text_hit_all_at_6 for r in answerable_recs]),
        "text_hit_any_at_context": _mean_bool(
            [r.text_hit_any_at_context for r in answerable_recs]
        ),
        "text_hit_all_at_context": _mean_bool(
            [r.text_hit_all_at_context for r in answerable_recs]
        ),
        "distinct_pages_at_6": round(
            _mean_float([r.distinct_pages_at_6 for r in answerable_recs]), 2
        ),
        "mean_context_chunks": mean_ctx_chunks,
    }

    outcome_counts = {
        "CORRECT": 0,
        "WRONG": 0,
        "FALSE_REFUSAL": 0,
        "CORRECT_REFUSAL": 0,
        "ANSWERED_ANYWAY": 0,
    }
    for r in primary_records:
        if r.outcome in outcome_counts:
            outcome_counts[r.outcome] += 1

    unsupported_qids = [r.id for r in primary_records if r.unsupported_numbers]
    answer_metrics = {
        "outcome_counts": outcome_counts,
        "grounded_rate": _mean_bool([r.grounded for r in answerable_recs]),
        "questions_with_unsupported_numbers": len(unsupported_qids),
        "unsupported_number_question_ids": unsupported_qids,
    }

    # Faithfulness (LLM judge)
    faithfulness_metrics: dict[str, Any] = {"enabled": bool(use_judge and not retrieval_only)}
    if use_judge and not retrieval_only:
        judged_recs = [r for r in primary_records if r.grounded]
        judge_errors = sum(1 for r in judged_recs if r.judge_parse_ok is False)
        valid_judged = [r for r in judged_recs if r.judge_parse_ok is True]
        faith_rate = _mean_bool([r.judge_faithful for r in valid_judged])
        sup_cits = sum(r.supporting_citations_count for r in valid_judged)
        tot_cits = sum(r.evaluated_citations_count for r in valid_judged)
        cit_prec = round(sup_cits / tot_cits, 4) if tot_cits > 0 else 0.0
        non_sup_by_q = {
            r.id: list(r.non_supporting_citations) for r in judged_recs
        }
        faithfulness_metrics = {
            "enabled": True,
            "judged_count": len(judged_recs),
            "faithfulness_rate": faith_rate,
            "citation_precision": cit_prec,
            "supporting_citations": sup_cits,
            "total_citations": tot_cits,
            "judge_error_count": judge_errors,
            "non_supporting_citations_by_question": non_sup_by_q,
        }

    # Repeats summary
    outcomes_by_q: dict[str, list[str]] = {}
    for r in all_records:
        if r.outcome is not None:
            outcomes_by_q.setdefault(r.id, []).append(r.outcome)

    flipped_ids = [
        qid for qid, outs in outcomes_by_q.items() if len(set(outs)) > 1
    ]
    f_rate = flip_rate(outcomes_by_q)

    by_repeat_idx: dict[int, list[QuestionOutcomeRecord]] = {}
    for r in all_records:
        by_repeat_idx.setdefault(r.repeat_index, []).append(r)

    rep_accs: list[float] = []
    rep_ans_accs: list[float] = []
    for _r_idx, r_list in sorted(by_repeat_idx.items()):
        if not r_list or retrieval_only:
            continue
        passed = sum(
            1
            for r in r_list
            if (r.answerable and r.outcome == "CORRECT")
            or (not r.answerable and r.outcome == "CORRECT_REFUSAL")
        )
        rep_accs.append(round(passed / len(r_list), 4))
        ans_sub = [r for r in r_list if r.answerable]
        if ans_sub:
            ans_passed = sum(1 for r in ans_sub if r.outcome == "CORRECT")
            rep_ans_accs.append(round(ans_passed / len(ans_sub), 4))

    repeats_summary = {
        "repeats": repeats,
        "flip_rate": f_rate,
        "flipped_question_ids": flipped_ids,
        "outcomes_by_question": outcomes_by_q,
        "accuracy_mean": _mean_float(rep_accs),
        "accuracy_min": min(rep_accs) if rep_accs else 0.0,
        "accuracy_max": max(rep_accs) if rep_accs else 0.0,
        "answerable_accuracy_mean": _mean_float(rep_ans_accs),
        "answerable_accuracy_min": min(rep_ans_accs) if rep_ans_accs else 0.0,
        "answerable_accuracy_max": max(rep_ans_accs) if rep_ans_accs else 0.0,
    }

    ret_lat = [r.retrieve_ms for r in all_records]
    gen_lat = [r.generate_ms for r in all_records if not retrieval_only]
    tot_lat = [r.total_ms for r in all_records]
    in_toks = [r.input_tokens for r in all_records if r.input_tokens is not None]
    out_toks = [r.output_tokens for r in all_records if r.output_tokens is not None]
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
        "retrieve_n": len(ret_lat),
        "retrieve_p50_ms": p50(ret_lat),
        "retrieve_p95_ms": p95(ret_lat),
        "retrieve_max_ms": max_val(ret_lat),
        "generate_n": len(gen_lat),
        "generate_p50_ms": p50(gen_lat),
        "generate_p95_ms": p95(gen_lat),
        "generate_max_ms": max_val(gen_lat),
        "total_n": len(tot_lat),
        "total_p50_ms": p50(tot_lat),
        "total_p95_ms": p95(tot_lat),
        "total_max_ms": max_val(tot_lat),
        "mean_context_chunks": mean_ctx_chunks,
        "avg_input_tokens": avg_in,
        "avg_output_tokens": avg_out,
        "avg_cost_per_query_usd": avg_cost,
    }

    by_type: dict[str, Any] = {}
    for qtype in ("fact", "comparison", "table", "unanswerable", "adversarial"):
        sub = [r for r in primary_records if r.type == qtype]
        if not sub:
            continue
        is_ans_type = qtype in ("fact", "comparison", "table")
        pass_target = "CORRECT" if is_ans_type else "CORRECT_REFUSAL"
        tha6_val = _mean_bool([r.text_hit_any_at_6 for r in sub]) if is_ans_type else None
        by_type[qtype] = {
            "count": len(sub),
            "page_hit_at_6": _mean_bool([r.page_hit_at_6 for r in sub]) if is_ans_type else None,
            "text_hit_at_6": tha6_val,
            "text_hit_any_at_6": tha6_val,
            "text_hit_all_at_6": _mean_bool([r.text_hit_all_at_6 for r in sub]) if is_ans_type else None,
            "pass_count": sum(1 for r in sub if r.outcome == pass_target) if not retrieval_only else None,
        }

    by_quality: dict[str, Any] = {}
    for qslice in ("all_clean", "all_degraded", "mixed"):
        sub = [r for r in answerable_recs if r.expected_quality_slice == qslice]
        tha6_q = _mean_bool([r.text_hit_any_at_6 for r in sub]) if sub else 0.0
        by_quality[qslice] = {
            "count": len(sub),
            "page_hit_at_6": _mean_bool([r.page_hit_at_6 for r in sub]) if sub else 0.0,
            "mrr": _mean_float([r.reciprocal_rank for r in sub]) if sub else 0.0,
            "text_hit_at_6": tha6_q,
            "text_hit_any_at_6": tha6_q,
            "text_hit_all_at_6": _mean_bool([r.text_hit_all_at_6 for r in sub]) if sub else 0.0,
            "correct_count": (
                sum(1 for r in sub if r.outcome == "CORRECT")
                if (sub and not retrieval_only)
                else None
            ),
        }

    failed_list = [
        asdict(r) for r in primary_records if is_question_failed(r, retrieval_only)
    ]

    return {
        "label": label,
        "timestamp": timestamp,
        "retrieval_only": retrieval_only,
        "split": split,
        "repeats": repeats,
        "include_drafts": include_drafts,
        "skipped_draft_count": len(skipped_draft_ids),
        "skipped_draft_ids": list(skipped_draft_ids),
        "skipped_draft_reason": (
            "excluded by default because status='draft_needs_fill' (missing expected answers and/or PDF pages)"
            if not include_drafts
            else None
        ),
        "status": status,
        "store_fingerprint": store_fp,
        "num_questions": len(primary_records),
        "all_human_verified": bool(primary_records) and all(r.human_verified for r in primary_records),
        "retrieval": retrieval_metrics,
        "answers": answer_metrics,
        "faithfulness": faithfulness_metrics,
        "repeats_summary": repeats_summary,
        "performance": performance_metrics,
        "slices": {
            "by_type": by_type,
            "by_expected_page_quality": by_quality,
        },
        "failed_questions": failed_list,
        "questions": [asdict(r) for r in primary_records],
        "all_repeat_questions": [asdict(r) for r in all_records],
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
    split: str = "dev",
    repeats: int = 1,
    use_judge: bool = False,
    include_drafts: bool = False,
    store: ChromaStore | None = None,
    embedder: EmbeddingProvider | None = None,
    generator: GenerationProvider | None = None,
    judge_provider: GenerationProvider | None = None,
    eval_dir: Path | str | None = None,
) -> EvalRunOutput:
    """Executes the evaluation harness across splits and repeats, optionally judging faithfulness."""
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

    active_judge = judge_provider or active_generator

    deps = QueryDeps(
        settings=settings,
        store=active_store,
        embedder=active_embedder,
        generator=active_generator,
    )

    split_matched = [
        q
        for q in golden
        if (split == "all" or q.split == split)
        and (only_type is None or q.type == only_type)
    ]
    if not include_drafts:
        skipped_draft_ids = [
            q.id for q in split_matched if q.status == "draft_needs_fill"
        ]
        questions = [q for q in split_matched if q.status != "draft_needs_fill"]
    else:
        skipped_draft_ids = []
        questions = split_matched

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
    active_prompt = get_system_prompt(settings)
    prompt_hash = hashlib.sha256(active_prompt.encode("utf-8")).hexdigest()
    store_fp = active_store.fingerprint()

    all_records: list[QuestionOutcomeRecord] = []
    details: list[dict[str, Any]] = []
    quota_exceeded = False
    status = "completed"
    effective_repeats = 1 if retrieval_only else max(1, int(repeats))

    for r_idx in range(effective_repeats):
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
                    repeat_index=r_idx,
                    use_judge=use_judge and (r_idx == 0),
                    judge_provider=active_judge,
                )
                all_records.append(rec)
                details.append(detail)
            except DailyQuotaExceeded:
                quota_exceeded = True
                status = "partial_daily_quota_exceeded"
                break
        if quota_exceeded:
            break

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)
    stem = f"{timestamp}_{safe_label}"

    summary = _aggregate_summary(
        all_records=all_records,
        settings=settings,
        label=label,
        timestamp=timestamp,
        retrieval_only=retrieval_only,
        status=status,
        store_fp=store_fp,
        split=split,
        repeats=effective_repeats,
        use_judge=use_judge,
        include_drafts=include_drafts,
        skipped_draft_ids=skipped_draft_ids,
    )
    md_report = format_markdown_report(summary)

    run_json_path = runs_dir / f"{stem}.json"
    metrics_json_path = results_dir / f"{stem}.metrics.json"
    markdown_path = results_dir / f"{stem}.md"
    judge_review_path: Path | None = None

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

    if use_judge and not retrieval_only:
        judge_review_path = runs_dir / f"{stem}_judge_review.md"
        review_md = format_judge_review_markdown(details, label=label, timestamp=timestamp)
        judge_review_path.write_text(review_md, encoding="utf-8")

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
        judge_review_path=judge_review_path,
    )


def _find_source_run_json(runs_dir: Path, source_run_label: str) -> Path:
    """Locates the latest gitignored run JSON in runs_dir matching source_run_label."""
    direct = Path(source_run_label)
    if direct.is_file():
        return direct

    safe_src = "".join(c if c.isalnum() or c in "-_" else "-" for c in source_run_label)
    matches = sorted(runs_dir.glob(f"*_{safe_src}.json"))
    if matches:
        return matches[-1]

    by_summary_label: list[Path] = []
    for candidate in sorted(runs_dir.glob("*.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("summary", {}).get("label") == source_run_label:
            by_summary_label.append(candidate)

    if by_summary_label:
        return by_summary_label[-1]

    raise FileNotFoundError(
        f"No cached run JSON found in '{runs_dir}' for label '{source_run_label}'."
    )


def rescore_run(
    golden: Sequence[GoldenQuestion],
    settings: Settings,
    source_run_label: str,
    label: str = "baseline-v2",
    store: ChromaStore | None = None,
    eval_dir: Path | str | None = None,
) -> EvalRunOutput:
    """Rescores cached answers and retrieved chunks from a prior run using the current scorer, with zero API or judge calls."""
    base_eval_dir = (
        Path(eval_dir) if eval_dir is not None else Path(settings.storage.eval_data_dir)
    )
    runs_dir = base_eval_dir / "runs"
    results_dir = base_eval_dir / "results"
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    source_path = _find_source_run_json(runs_dir, source_run_label)
    raw_run = json.loads(source_path.read_text(encoding="utf-8"))
    src_summary: dict[str, Any] = raw_run["summary"]
    src_details: list[dict[str, Any]] = raw_run["details"]

    golden_by_id = {q.id: q for q in golden}
    retrieval_only = bool(src_summary.get("retrieval_only", False))
    use_judge = bool(src_summary.get("faithfulness", {}).get("enabled", False))
    split = str(src_summary.get("split", "dev"))
    repeats = int(src_summary.get("repeats", 1))
    include_drafts = bool(src_summary.get("include_drafts", False))
    skipped_draft_ids = list(src_summary.get("skipped_draft_ids", []))
    store_fp = (
        store.fingerprint()
        if store is not None
        else str(src_summary.get("store_fingerprint", ""))
    )

    all_records: list[QuestionOutcomeRecord] = []
    details: list[dict[str, Any]] = []

    for item in src_details:
        raw_q = item["question"]
        qid = str(raw_q["id"])
        q = golden_by_id.get(qid) or GoldenQuestion.model_validate(raw_q)
        r_idx = int(item.get("repeat_index", 0))
        old_rec: dict[str, Any] = item["outcome_record"]

        chunks = [RetrievedChunk(**c) for c in item.get("retrieved_chunks", [])]
        cits_full = [Citation(**c) for c in item.get("citations_full", [])]
        num_sources = max(1, int(old_rec.get("detected_sources_count", 1)))
        q_slice = (
            classify_expected_page_quality(q, store)
            if store is not None
            else str(old_rec.get("expected_quality_slice", "clean"))
        )
        r_fields = _compute_retrieval_fields(q, chunks, num_sources)

        ans_str = item.get("answer")
        grounded = old_rec.get("grounded")
        if not retrieval_only and ans_str is not None:
            q_outcome = outcome(q, str(ans_str))
            if grounded and cits_full and chunks:
                chunk_by_id = {c.chunk_id: c for c in chunks}
                cited_chunks = [
                    chunk_by_id[c.chunk_id]
                    for c in cits_full
                    if c.chunk_id in chunk_by_id
                ]
                unsup_nums = find_unsupported_numbers(str(ans_str), cited_chunks)
            else:
                unsup_nums = []
        else:
            q_outcome = None
            unsup_nums = []

        primary_chunks = [c for c in chunks if c.neighbor_of is None]
        top6_refs = [
            RetrievedRef(
                chunk_id=c.chunk_id,
                source=c.source,
                page=c.page,
                score=c.score,
                text_quality=c.text_quality,
                neighbor_of=c.neighbor_of,
            )
            for c in primary_chunks[:6]
        ]
        cit_refs = [
            CitationRef(
                n=c.n,
                chunk_id=c.chunk_id,
                source=c.source,
                page=c.page,
                text_quality=c.text_quality,
                neighbor_of=c.neighbor_of,
            )
            for c in cits_full
        ]

        rec = QuestionOutcomeRecord(
            id=q.id,
            question=q.question,
            type=q.type,
            answerable=q.answerable,
            human_verified=q.human_verified,
            expected_quality_slice=q_slice,
            outcome=q_outcome,
            grounded=grounded,
            reason=old_rec.get("reason"),
            invalid_citations=int(old_rec.get("invalid_citations", 0)),
            unsupported_numbers=unsup_nums,
            retrieve_ms=float(old_rec.get("retrieve_ms", 0.0)),
            generate_ms=float(old_rec.get("generate_ms", 0.0)),
            total_ms=float(old_rec.get("total_ms", 0.0)),
            input_tokens=old_rec.get("input_tokens"),
            output_tokens=old_rec.get("output_tokens"),
            from_cache=True,
            retrieved_top6=top6_refs,
            citations=cit_refs,
            split=q.split,
            repeat_index=r_idx,
            detected_sources_count=num_sources,
            judge_faithful=old_rec.get("judge_faithful"),
            judge_parse_ok=old_rec.get("judge_parse_ok"),
            supporting_citations_count=int(old_rec.get("supporting_citations_count", 0)),
            evaluated_citations_count=int(old_rec.get("evaluated_citations_count", 0)),
            non_supporting_citations=list(old_rec.get("non_supporting_citations", [])),
            status=q.status,
            **r_fields,
        )
        all_records.append(rec)
        details.append(
            {
                "question": q.model_dump(),
                "repeat_index": r_idx,
                "outcome_record": asdict(rec),
                "answer": ans_str,
                "retrieved_chunks": [asdict(c) for c in chunks],
                "citations_full": [asdict(c) for c in cits_full],
                "judge_verdict": item.get("judge_verdict"),
                "cited_passages_full": item.get("cited_passages_full", []),
            }
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)
    stem = f"{timestamp}_{safe_label}"
    status = str(src_summary.get("status", "completed"))

    summary = _aggregate_summary(
        all_records=all_records,
        settings=settings,
        label=label,
        timestamp=timestamp,
        retrieval_only=retrieval_only,
        status=status,
        store_fp=store_fp,
        split=split,
        repeats=repeats,
        use_judge=use_judge,
        include_drafts=include_drafts,
        skipped_draft_ids=skipped_draft_ids,
    )
    md_report = format_markdown_report(summary)

    run_json_path = runs_dir / f"{stem}.json"
    metrics_json_path = results_dir / f"{stem}.metrics.json"
    markdown_path = results_dir / f"{stem}.md"
    judge_review_path: Path | None = None

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

    if use_judge and not retrieval_only:
        judge_review_path = runs_dir / f"{stem}_judge_review.md"
        review_md = format_judge_review_markdown(
            details, label=label, timestamp=timestamp
        )
        judge_review_path.write_text(review_md, encoding="utf-8")

    return EvalRunOutput(
        label=label,
        timestamp=timestamp,
        status=status,
        quota_exceeded=False,
        summary=summary,
        markdown_report=md_report,
        run_json_path=run_json_path,
        metrics_json_path=metrics_json_path,
        markdown_path=markdown_path,
        judge_review_path=judge_review_path,
    )

