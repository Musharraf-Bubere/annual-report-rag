"""Report formatting, judge review generation, and comparison utilities for evaluation runs (zero PDF/chunk text in committed outputs)."""

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedRef:
    """Metadata-only reference to a retrieved chunk (safe to commit; contains no chunk text)."""

    chunk_id: str
    source: str
    page: int
    score: float
    text_quality: str


@dataclass(frozen=True)
class CitationRef:
    """Metadata-only reference to a resolved citation (safe to commit; contains no excerpt text)."""

    n: int
    chunk_id: str
    source: str
    page: int
    text_quality: str


@dataclass(frozen=True)
class QuestionOutcomeRecord:
    """Per-question evaluation outcome without any PDF, chunk, or unsupported-claim text."""

    id: str
    question: str
    type: str
    answerable: bool
    human_verified: bool
    expected_quality_slice: str
    page_hit_at_1: bool | None
    page_hit_at_3: bool | None
    page_hit_at_6: bool | None
    reciprocal_rank: float | None
    text_hit_at_6: bool | None
    outcome: str | None
    grounded: bool | None
    reason: str | None
    invalid_citations: int
    unsupported_numbers: list[str]
    retrieve_ms: float
    generate_ms: float
    total_ms: float
    input_tokens: int | None
    output_tokens: int | None
    from_cache: bool = False
    retrieved_top6: list[RetrievedRef] = field(default_factory=list)
    citations: list[CitationRef] = field(default_factory=list)
    split: str = "dev"
    repeat_index: int = 0
    detected_sources_count: int = 1
    text_hit_any_at_1: bool | None = None
    text_hit_any_at_3: bool | None = None
    text_hit_any_at_6: bool | None = None
    text_hit_all_at_1: bool | None = None
    text_hit_all_at_3: bool | None = None
    text_hit_all_at_6: bool | None = None
    distinct_pages_at_6: int = 0
    judge_faithful: bool | None = None
    judge_parse_ok: bool | None = None
    supporting_citations_count: int = 0
    evaluated_citations_count: int = 0
    non_supporting_citations: list[int] = field(default_factory=list)


def is_question_failed(rec: QuestionOutcomeRecord, retrieval_only: bool) -> bool:
    """Determines whether a question record failed retrieval or answer evaluation."""
    if retrieval_only:
        if not rec.answerable:
            return False
        th6 = rec.text_hit_all_at_6 if rec.text_hit_all_at_6 is not None else rec.text_hit_at_6
        return not bool(rec.page_hit_at_6) or not bool(th6)

    if rec.answerable:
        return rec.outcome != "CORRECT"
    return rec.outcome != "CORRECT_REFUSAL"


def format_markdown_report(summary: dict[str, Any]) -> str:
    """Renders the committed Markdown evaluation report from a metrics summary dictionary."""
    lines: list[str] = []
    label = summary.get("label", "eval")
    timestamp = summary.get("timestamp", "")
    retrieval_only = bool(summary.get("retrieval_only", False))
    status = summary.get("status", "completed")
    split = summary.get("split", "dev")
    repeats = int(summary.get("repeats", 1))
    num_questions = int(summary.get("num_questions", 0))
    all_human_verified = bool(summary.get("all_human_verified", False))

    lines.append(f"# Evaluation Report: `{label}` ({timestamp})")
    lines.append("")
    lines.append(
        f"- **Mode**: `{'retrieval-only' if retrieval_only else 'full (retrieval + generation)'}`"
    )
    lines.append(f"- **Split**: `{split}`")
    lines.append(f"- **Repeats**: `{repeats}`")
    lines.append(f"- **Status**: `{status}`")
    lines.append(f"- **Questions evaluated**: `{num_questions}`")
    lines.append("")

    if num_questions < 30 or not all_human_verified:
        reasons: list[str] = []
        if num_questions < 30:
            reasons.append(f"fewer than 30 questions ({num_questions})")
        if not all_human_verified:
            reasons.append("one or more questions have `human_verified=false`")
        lines.append(
            f"> [!WARNING]\n> **PRELIMINARY RESULTS**: {', '.join(reasons)}. "
            "Treat these metrics as directional engineering signals rather than statistically final benchmarks."
        )
        lines.append("")

    ret = summary.get("retrieval", {})
    lines.append("## 1. Retrieval Metrics (Answerable Questions)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| `page_hit@1` | {ret.get('page_hit_at_1', 0.0):.4f} |")
    lines.append(f"| `page_hit@3` | {ret.get('page_hit_at_3', 0.0):.4f} |")
    lines.append(f"| `page_hit@6` | {ret.get('page_hit_at_6', 0.0):.4f} |")
    lines.append(f"| `MRR` | {ret.get('mrr', 0.0):.4f} |")
    lines.append(f"| `text_hit_any@1` | {ret.get('text_hit_any_at_1', 0.0):.4f} |")
    lines.append(f"| `text_hit_any@3` | {ret.get('text_hit_any_at_3', 0.0):.4f} |")
    lines.append(f"| `text_hit_any@6` | {ret.get('text_hit_any_at_6', ret.get('text_hit_at_6', 0.0)):.4f} |")
    lines.append(f"| `text_hit_all@1` | {ret.get('text_hit_all_at_1', 0.0):.4f} |")
    lines.append(f"| `text_hit_all@3` | {ret.get('text_hit_all_at_3', 0.0):.4f} |")
    lines.append(f"| `text_hit_all@6` | {ret.get('text_hit_all_at_6', 0.0):.4f} |")
    lines.append(f"| `distinct_pages@6` | {ret.get('distinct_pages_at_6', 0.0):.2f} |")
    lines.append("")

    lines.append("## 2. Answer Metrics")
    lines.append("")
    if retrieval_only:
        lines.append("_Skipped in `--retrieval-only` mode._")
        lines.append("")
    else:
        ans = summary.get("answers", {})
        counts = ans.get("outcome_counts", {})
        lines.append("| Metric / Outcome | Value |")
        lines.append("| :--- | :--- |")
        lines.append(f"| `CORRECT` | {counts.get('CORRECT', 0)} |")
        lines.append(f"| `WRONG` | {counts.get('WRONG', 0)} |")
        lines.append(f"| `FALSE_REFUSAL` | {counts.get('FALSE_REFUSAL', 0)} |")
        lines.append(f"| `CORRECT_REFUSAL` | {counts.get('CORRECT_REFUSAL', 0)} |")
        lines.append(f"| `ANSWERED_ANYWAY` | {counts.get('ANSWERED_ANYWAY', 0)} |")
        lines.append(f"| `grounded_rate` (answerable) | {ans.get('grounded_rate', 0.0):.4f} |")
        lines.append(
            f"| `questions_with_unsupported_numbers` | {ans.get('questions_with_unsupported_numbers', 0)} "
            f"({ans.get('unsupported_number_question_ids', [])}) |"
        )
        lines.append("")

    faith = summary.get("faithfulness")
    if faith and faith.get("enabled"):
        lines.append("## 2b. Faithfulness & Citation Precision (LLM Judge)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| :--- | :--- |")
        lines.append(f"| `judged_count` (`grounded=True`) | {faith.get('judged_count', 0)} |")
        lines.append(f"| `faithfulness_rate` | {faith.get('faithfulness_rate', 0.0):.4f} |")
        lines.append(
            f"| `citation_precision` | {faith.get('citation_precision', 0.0):.4f} "
            f"({faith.get('supporting_citations', 0)}/{faith.get('total_citations', 0)}) |"
        )
        lines.append(f"| `judge_error` count | {faith.get('judge_error_count', 0)} |")
        lines.append("")
        lines.append("**Non-supporting citation numbers by question (`grounded=True`)**:")
        non_sup_map: dict[str, list[int]] = faith.get("non_supporting_citations_by_question", {})
        for qid, non_sup_nums in non_sup_map.items():
            lines.append(f"- `{qid}`: `{non_sup_nums}`")
        lines.append("")

    rep_data = summary.get("repeats_summary")
    if rep_data and repeats > 1:
        lines.append("## 2c. Repeats & Stability")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| :--- | :--- |")
        lines.append(f"| `repeats` | {repeats} |")
        lines.append(
            f"| `accuracy` (mean, min–max) | {rep_data.get('accuracy_mean', 0.0):.4f} "
            f"({rep_data.get('accuracy_min', 0.0):.4f}–{rep_data.get('accuracy_max', 0.0):.4f}) |"
        )
        lines.append(
            f"| `answerable_accuracy` (mean, min–max) | {rep_data.get('answerable_accuracy_mean', 0.0):.4f} "
            f"({rep_data.get('answerable_accuracy_min', 0.0):.4f}–{rep_data.get('answerable_accuracy_max', 0.0):.4f}) |"
        )
        lines.append(
            f"| `flip_rate` | {rep_data.get('flip_rate', 0.0):.4f} "
            f"(flipped IDs: {rep_data.get('flipped_question_ids', [])}) |"
        )
        lines.append("")
        lines.append("| Question ID | Outcomes across repeats | Flipped |")
        lines.append("| :--- | :--- | :--- |")
        per_q_outs: dict[str, list[str]] = rep_data.get("outcomes_by_question", {})
        for qid, outs in per_q_outs.items():
            flipped = len(set(outs)) > 1
            outs_str = ", ".join(f"`{o}`" for o in outs)
            lines.append(f"| `{qid}` | {outs_str} | `{flipped}` |")
        lines.append("")

    perf = summary.get("performance", {})
    ret_n = int(perf.get("retrieve_n", num_questions))
    gen_n = int(perf.get("generate_n", num_questions if not retrieval_only else 0))
    tot_n = int(perf.get("total_n", num_questions))
    lines.append("## 3. Performance & Token Usage")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| :--- | :--- |")
    lines.append(
        f"| `retrieve_ms` (n={ret_n}) p50 / p95 / max | "
        f"{perf.get('retrieve_p50_ms', 0.0):.2f} ms (n={ret_n}) / "
        f"{perf.get('retrieve_p95_ms', 0.0):.2f} ms (n={ret_n}) / "
        f"{perf.get('retrieve_max_ms', 0.0):.2f} ms (n={ret_n}) |"
    )
    if not retrieval_only:
        lines.append(
            f"| `generate_ms` (n={gen_n}) p50 / p95 / max | "
            f"{perf.get('generate_p50_ms', 0.0):.2f} ms (n={gen_n}) / "
            f"{perf.get('generate_p95_ms', 0.0):.2f} ms (n={gen_n}) / "
            f"{perf.get('generate_max_ms', 0.0):.2f} ms (n={gen_n}) |"
        )
        lines.append(
            f"| `total_ms` (n={tot_n}) p50 / p95 / max | "
            f"{perf.get('total_p50_ms', 0.0):.2f} ms (n={tot_n}) / "
            f"{perf.get('total_p95_ms', 0.0):.2f} ms (n={tot_n}) / "
            f"{perf.get('total_max_ms', 0.0):.2f} ms (n={tot_n}) |"
        )
        lines.append(f"| `avg_input_tokens` | {perf.get('avg_input_tokens', 0.0):.1f} |")
        lines.append(f"| `avg_output_tokens` | {perf.get('avg_output_tokens', 0.0):.1f} |")
        avg_cost = perf.get("avg_cost_per_query_usd")
        if avg_cost is not None:
            lines.append(f"| `avg_cost_per_query_usd` | ${avg_cost:.6f} |")
    lines.append("")
    lines.append(
        "> [!NOTE]\n> `p95` latency is not statistically meaningful below 20 samples (`n < 20`) "
        "and is heavily influenced by the single `max` sample."
    )
    lines.append("")

    slices = summary.get("slices", {})
    by_type = slices.get("by_type", {})
    lines.append("## 4. Slices by Question Type")
    lines.append("")
    lines.append("| Type | Count | `page_hit@6` | `text_hit_any@6` | `text_hit_all@6` | `CORRECT` / `CORRECT_REFUSAL` |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for qtype, sdata in by_type.items():
        ph6 = sdata.get("page_hit_at_6")
        tha6 = sdata.get("text_hit_any_at_6", sdata.get("text_hit_at_6"))
        thall6 = sdata.get("text_hit_all_at_6")
        ph6_str = f"{ph6:.4f}" if ph6 is not None else "N/A"
        tha6_str = f"{tha6:.4f}" if tha6 is not None else "N/A"
        thall6_str = f"{thall6:.4f}" if thall6 is not None else "N/A"
        pass_cnt = sdata.get("pass_count")
        pass_str = f"{pass_cnt}/{sdata.get('count', 0)}" if pass_cnt is not None else "N/A"
        lines.append(
            f"| `{qtype}` | {sdata.get('count', 0)} | {ph6_str} | {tha6_str} | {thall6_str} | {pass_str} |"
        )
    lines.append("")

    by_quality = slices.get("by_expected_page_quality", {})
    lines.append("## 5. Slices by Expected-Page Text Quality (Answerable Questions)")
    lines.append("")
    lines.append(
        "| Expected-Page Quality | Count | `page_hit@6` | `MRR` | `text_hit_any@6` | `text_hit_all@6` | `CORRECT` |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for qslice in ("all_clean", "all_degraded", "mixed"):
        sdata = by_quality.get(qslice, {"count": 0})
        cnt = int(sdata.get("count", 0))
        if cnt == 0:
            lines.append(f"| `{qslice}` | 0 | N/A | N/A | N/A | N/A | N/A |\n".rstrip())
        else:
            ph6 = sdata.get("page_hit_at_6", 0.0)
            mrr = sdata.get("mrr", 0.0)
            tha6 = sdata.get("text_hit_any_at_6", sdata.get("text_hit_at_6", 0.0))
            thall6 = sdata.get("text_hit_all_at_6", 0.0)
            corr = sdata.get("correct_count")
            corr_str = f"{corr}/{cnt}" if corr is not None else "N/A"
            lines.append(
                f"| `{qslice}` | {cnt} | {ph6:.4f} | {mrr:.4f} | {tha6:.4f} | {thall6:.4f} | {corr_str} |"
            )
    lines.append("")

    failed = summary.get("failed_questions", [])
    lines.append("## 6. Failed Questions")
    lines.append("")
    if not failed:
        lines.append("_None. All evaluated questions passed._")
    else:
        for fq in failed:
            top6_str = ", ".join(
                f"{r['source']} PDF p{r['page']} ({r['score']:.4f})"
                for r in fq.get("retrieved_top6", [])
            )
            lines.append(
                f"- **`{fq['id']}`** (`{fq['type']}`): {fq['question']}\n"
                f"  - Outcome: `{fq.get('outcome')}` | `page_hit@6`: `{fq.get('page_hit_at_6')}` | "
                f"`text_hit_any@6`: `{fq.get('text_hit_any_at_6', fq.get('text_hit_at_6'))}` | "
                f"`text_hit_all@6`: `{fq.get('text_hit_all_at_6')}`\n"
                f"  - Top 6 retrieved: {top6_str}"
            )
    lines.append("")
    return "\n".join(lines)


def format_judge_review_markdown(
    details: Sequence[dict[str, Any]],
    label: str,
    timestamp: str,
    max_samples: int = 10,
    seed: int = 42,
) -> str:
    """Renders the gitignored judge spot-check Markdown file with up to 10 random judged answers and full cited passages."""
    judged = [d for d in details if d.get("judge_verdict") is not None]
    rng = random.Random(seed)
    sampled = (
        rng.sample(judged, k=max_samples)
        if len(judged) > max_samples
        else list(judged)
    )

    lines: list[str] = []
    lines.append(f"# Judge Spot-Check Review: `{label}` ({timestamp})")
    lines.append("")
    lines.append(
        f"> **GITIGNORED DIAGNOSTIC FILE** — Contains full cited passage texts and unsupported-claim strings for manual spot-checking ({len(sampled)} of {len(judged)} judged answers)."
    )
    lines.append("")

    for idx, item in enumerate(sampled, start=1):
        q = item.get("question", {})
        verdict = item.get("judge_verdict", {})
        cited_passages = item.get("cited_passages_full", [])
        lines.append(f"## {idx}. `{q.get('id')}` (repeat `{item.get('repeat_index', 0)}`)")
        lines.append("")
        lines.append(f"- **Question**: {q.get('question')}")
        lines.append(f"- **Answer**: {item.get('answer')}")
        lines.append(
            f"- **Judge Verdict**: `faithful={verdict.get('faithful')}`, "
            f"`parse_ok={verdict.get('parse_ok')}`, "
            f"`unsupported_claims={verdict.get('unsupported_claims')}`, "
            f"`citations={verdict.get('citations')}`"
        )
        lines.append("")
        lines.append("### Cited Passages (Full Text)")
        lines.append("")
        for cp in cited_passages:
            lines.append(f"#### `[{cp.get('n')}]` (`{cp.get('source')}` PDF page `{cp.get('page')}`)")
            lines.append("```text")
            lines.append(str(cp.get("text", "")))
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def compare_metrics_dicts(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
) -> str:
    """Formats a side-by-side comparison table and per-question outcome changes between two runs."""
    label_a = metrics_a.get("label", "Run A")
    label_b = metrics_b.get("label", "Run B")

    def _get_nested(d: dict[str, Any], path: tuple[str, ...]) -> float | int | None:
        curr: Any = d
        for k in path:
            if not isinstance(curr, dict) or k not in curr:
                return None
            curr = curr[k]
        return curr if isinstance(curr, (int, float)) else None

    metric_rows: list[tuple[str, tuple[str, ...]]] = [
        ("page_hit@1", ("retrieval", "page_hit_at_1")),
        ("page_hit@3", ("retrieval", "page_hit_at_3")),
        ("page_hit@6", ("retrieval", "page_hit_at_6")),
        ("MRR", ("retrieval", "mrr")),
        ("text_hit_any@1", ("retrieval", "text_hit_any_at_1")),
        ("text_hit_any@3", ("retrieval", "text_hit_any_at_3")),
        ("text_hit_any@6", ("retrieval", "text_hit_any_at_6")),
        ("text_hit_all@1", ("retrieval", "text_hit_all_at_1")),
        ("text_hit_all@3", ("retrieval", "text_hit_all_at_3")),
        ("text_hit_all@6", ("retrieval", "text_hit_all_at_6")),
        ("distinct_pages@6", ("retrieval", "distinct_pages_at_6")),
        ("CORRECT", ("answers", "outcome_counts", "CORRECT")),
        ("WRONG", ("answers", "outcome_counts", "WRONG")),
        ("FALSE_REFUSAL", ("answers", "outcome_counts", "FALSE_REFUSAL")),
        ("CORRECT_REFUSAL", ("answers", "outcome_counts", "CORRECT_REFUSAL")),
        ("ANSWERED_ANYWAY", ("answers", "outcome_counts", "ANSWERED_ANYWAY")),
        ("grounded_rate", ("answers", "grounded_rate")),
        ("unsupported_number_questions", ("answers", "questions_with_unsupported_numbers")),
        ("faithfulness_rate", ("faithfulness", "faithfulness_rate")),
        ("citation_precision", ("faithfulness", "citation_precision")),
        ("retrieve_p50_ms", ("performance", "retrieve_p50_ms")),
        ("retrieve_p95_ms", ("performance", "retrieve_p95_ms")),
        ("retrieve_max_ms", ("performance", "retrieve_max_ms")),
        ("generate_p50_ms", ("performance", "generate_p50_ms")),
        ("generate_p95_ms", ("performance", "generate_p95_ms")),
        ("generate_max_ms", ("performance", "generate_max_ms")),
        ("total_p50_ms", ("performance", "total_p50_ms")),
        ("total_p95_ms", ("performance", "total_p95_ms")),
        ("total_max_ms", ("performance", "total_max_ms")),
    ]

    lines: list[str] = []
    lines.append(f"# Comparison: `{label_a}` vs `{label_b}`")
    lines.append("")
    lines.append(f"| Metric | `{label_a}` | `{label_b}` | Delta (`B - A`) |")
    lines.append("| :--- | :--- | :--- | :--- |")

    for name, path in metric_rows:
        va = _get_nested(metrics_a, path)
        vb = _get_nested(metrics_b, path)
        if va is None and vb is None:
            continue
        if isinstance(va, int) and isinstance(vb, int):
            delta = vb - va
            sign = "+" if delta > 0 else ""
            lines.append(f"| `{name}` | {va} | {vb} | {sign}{delta} |")
        else:
            fa = float(va) if va is not None else 0.0
            fb = float(vb) if vb is not None else 0.0
            delta_f = fb - fa
            sign = "+" if delta_f > 0 else ""
            lines.append(f"| `{name}` | {fa:.4f} | {fb:.4f} | {sign}{delta_f:.4f} |")

    lines.append("")
    lines.append("## Per-Question Outcome Changes")
    lines.append("")

    q_a = {q["id"]: q for q in metrics_a.get("questions", [])}
    q_b = {q["id"]: q for q in metrics_b.get("questions", [])}
    all_ids = sorted(set(q_a.keys()) | set(q_b.keys()))

    good_outcomes = {"CORRECT", "CORRECT_REFUSAL"}
    changes: list[str] = []
    for qid in all_ids:
        oa = q_a.get(qid, {}).get("outcome")
        ob = q_b.get(qid, {}).get("outcome")
        if oa != ob:
            if ob in good_outcomes and oa not in good_outcomes:
                direction = "IMPROVED"
            elif oa in good_outcomes and ob not in good_outcomes:
                direction = "REGRESSED"
            else:
                direction = "CHANGED"
            q_text = q_b.get(qid, q_a.get(qid, {})).get("question", "")
            changes.append(f"- **`{qid}`** [{direction}]: `{oa}` -> `{ob}` ({q_text})")

    if not changes:
        lines.append("_No per-question outcome changes between runs._")
    else:
        lines.extend(changes)

    lines.append("")
    return "\n".join(lines)
