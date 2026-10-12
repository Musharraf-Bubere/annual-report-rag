"""Unit tests for prompt builder, citation resolution, numerical grounding checks, and answerer."""

from pathlib import Path

from annual_report_rag.generation.answerer import (
    QueryDeps,
    answer_question,
    extract_citation_numbers,
    resolve_citations,
)
from annual_report_rag.generation.checks import find_unsupported_numbers
from annual_report_rag.generation.prompts import (
    NOT_FOUND_MESSAGE,
    SYSTEM_PROMPT,
    build_prompt,
    format_context,
)
from annual_report_rag.ingestion.chunker import Chunk
from annual_report_rag.providers.fake import FakeProvider
from annual_report_rag.retrieval.retriever import RetrievedChunk
from annual_report_rag.retrieval.store import ChromaStore
from tests.test_router_and_retriever import _build_test_settings


def _sample_chunks() -> list[RetrievedChunk]:
    """Creates sample RetrievedChunk records for prompt and citation tests."""
    return [
        RetrievedChunk(
            chunk_id="tcs-p8-c0",
            text="Consolidated revenue was ₹2,67,021 crore in FY26, growing 4.2%.",
            source="tcs-ar-26",
            page=8,
            score=0.88,
            extraction_method="default",
            text_quality="clean",
        ),
        RetrievedChunk(
            chunk_id="tcs-p224-c0",
            text="TCS received H583 crore and H1,073 crore from tax authorities.",
            source="tcs-ar-26",
            page=224,
            score=0.79,
            extraction_method="inhibit_spaces",
            text_quality="degraded",
        ),
        RetrievedChunk(
            chunk_id="infy-p12-c0",
            text="Infosys headcount stood at 317,240 employees.",
            source="infosys-ar-26",
            page=12,
            score=0.71,
            extraction_method="default",
            text_quality="clean",
        ),
    ]


def test_prompt_builder_and_system_prompt() -> None:
    """Verifies numbered context headers, PDF page labels, degraded quality tag, and system rules."""
    settings = _build_test_settings()
    chunks = _sample_chunks()
    ctx = format_context(chunks, settings)

    assert "[1] TCS Integrated Annual Report 2025-26, PDF page 8 (text quality: clean)" in ctx
    assert "[2] TCS Integrated Annual Report 2025-26, PDF page 224 (text quality: degraded)" in ctx
    assert "[3] Infosys Integrated Annual Report 2025-26, PDF page 12 (text quality: clean)" in ctx

    sys_prompt, user_prompt = build_prompt("What was TCS revenue?", chunks, settings)
    assert sys_prompt == SYSTEM_PROMPT
    assert "Accounting synonyms" not in sys_prompt
    assert "Question: What was TCS revenue?" in user_prompt
    assert NOT_FOUND_MESSAGE in SYSTEM_PROMPT
    assert "(Amount in H)" in SYSTEM_PROMPT

    # Verify prompt.rules_v2=True appends rules 8 and 9
    from dataclasses import replace

    from annual_report_rag.config import PromptConfig
    from annual_report_rag.generation.prompts import SYSTEM_PROMPT_V2

    settings_v2 = replace(settings, prompt=PromptConfig(rules_v2=True))
    sys_prompt_v2, _ = build_prompt("What was TCS revenue?", chunks, settings_v2)
    assert sys_prompt_v2 == SYSTEM_PROMPT_V2
    assert '8. Accounting synonyms: treat "operating income" = "operating profit" = "EBIT"' in sys_prompt_v2
    assert '9. Basis: unless the question says "standalone", answer with the consolidated figure' in sys_prompt_v2


def test_citation_parsing_and_resolution() -> None:
    """Verifies extraction and validation of [1], [1][3], [1, 3], out-of-range, and missing citations."""
    settings = _build_test_settings()
    chunks = _sample_chunks()

    # Valid single [1]
    nums = extract_citation_numbers("Revenue was ₹2,67,021 crore [1].")
    assert nums == [1]
    cits, _cited_chunks, inv = resolve_citations(nums, chunks, settings)
    assert inv == 0
    assert len(cits) == 1
    assert cits[0].n == 1
    assert cits[0].page == 8

    # Valid [1][3]
    nums = extract_citation_numbers("Both companies grew [1][3].")
    assert nums == [1, 3]
    cits, _cited_chunks, inv = resolve_citations(nums, chunks, settings)
    assert inv == 0
    assert [c.n for c in cits] == [1, 3]

    # Valid [1, 3]
    nums = extract_citation_numbers("Comparison across reports [1, 3].")
    assert nums == [1, 3]
    cits, _cited_chunks, inv = resolve_citations(nums, chunks, settings)
    assert inv == 0
    assert [c.n for c in cits] == [1, 3]

    # Invalid out-of-range [5]
    nums = extract_citation_numbers("Hallucinated citation [1][5].")
    assert nums == [1, 5]
    cits, _cited_chunks, inv = resolve_citations(nums, chunks, settings)
    assert inv == 1
    assert [c.n for c in cits] == [1]

    # None
    nums = extract_citation_numbers("Answer with no brackets at all.")
    assert nums == []
    cits, _cited_chunks, inv = resolve_citations(nums, chunks, settings)
    assert inv == 0
    assert cits == []


def test_find_unsupported_numbers() -> None:
    """Verifies numeric grounding check across Indian/Western commas, internal spaces, and unsupported numbers."""
    cited = [
        RetrievedChunk(
            chunk_id="c1",
            text="Revenue was 267,021 crore and headcount was 584, 519 in 2026 (growth 4.2%).",
            source="tcs-ar-26",
            page=8,
            score=0.9,
            extraction_method="default",
            text_quality="clean",
        )
    ]

    # Supported numbers with Indian vs Western commas and space-repaired numbers
    ans_ok = "In 2026, revenue was ₹2,67,021 crore (up 4.2%) with 584,519 employees [1]."
    assert find_unsupported_numbers(ans_ok, cited) == []

    # Unsupported number (999,000 and 12.5 not in cited chunk)
    ans_bad = "Revenue was ₹2,67,021 crore and net profit was ₹999,000 crore (12.5%) [1]."
    assert find_unsupported_numbers(ans_bad, cited) == ["999,000", "12.5"]

    # Fiscal-year forms "2025-26" and "FY26" support year "2026" without flagging it
    cited_fy_range = [
        RetrievedChunk(
            chunk_id="c_fy1",
            text="Regions 2025-26 North America 56.1% Europe 32.1%",
            source="infosys-ar-26",
            page=10,
            score=0.9,
            extraction_method="default",
            text_quality="clean",
        )
    ]
    assert (
        find_unsupported_numbers(
            "North America contributes the most in fiscal 2026, with a share of 56.1% [1].",
            cited_fy_range,
        )
        == []
    )

    cited_fy_short = [
        RetrievedChunk(
            chunk_id="c_fy2",
            text="In FY26, North America share was 56.1%.",
            source="infosys-ar-26",
            page=10,
            score=0.9,
            extraction_method="default",
            text_quality="clean",
        )
    ]
    assert (
        find_unsupported_numbers(
            "In 2026, North America share was 56.1% [1].",
            cited_fy_short,
        )
        == []
    )
    # Ungrounded year (2030) is still flagged
    assert find_unsupported_numbers(
        "By 2030, North America share will be 56.1% [1].",
        cited_fy_short,
    ) == ["2030"]

    # Dates like "March 31, 2025" or "31, 2025" must NOT be read as comma-grouped numbers ("31, 2025" -> 312025)
    cited_tax = [
        RetrievedChunk(
            chunk_id="tcs-ar-26-p224-c0",
            text=(
                "received from direct tax authorities in India and other jurisdictions "
                "is H583 crore and H1,073 crore as at March 31, 2026 and 2025, respectively."
            ),
            source="tcs-ar-26",
            page=224,
            score=0.9,
            extraction_method="default",
            text_quality="clean",
        )
    ]
    ans_date = (
        "The consolidated contingent liability in respect of tax demands received from "
        "direct tax authorities is ₹583 crore as at March 31, 2026, and ₹1,073 crore "
        "as at March 31, 2025 [1]."
    )
    assert find_unsupported_numbers(ans_date, cited_tax) == []
    assert find_unsupported_numbers("Balances as at 31, 2025 were ₹1,073 crore [1].", cited_tax) == []


def test_answerer_pipeline_cases(tmp_path: Path) -> None:
    """Verifies answer_question for grounded, not_found, invalid citation, and no_context."""
    db_path = str(tmp_path / "answerer_chroma")
    store = ChromaStore(persist_path=db_path, collection_name="ans_col")
    embedder = FakeProvider(embedding_dim=4)
    settings = _build_test_settings(min_score=0.0, top_k=2)

    # 1. Empty store -> no_context, generator must NOT be called
    gen_unused = FakeProvider(default_response="Should not be called [1].")
    deps_empty = QueryDeps(
        settings=settings, embedder=embedder, generator=gen_unused, store=store
    )
    res_empty = answer_question("What was TCS revenue?", deps_empty)
    assert res_empty.grounded is False
    assert res_empty.reason == "no_context"
    assert res_empty.answer == NOT_FOUND_MESSAGE
    assert gen_unused.generated_prompts == []

    # Populate store with 1 chunk
    q_vec = embedder.embed(["What was TCS revenue?"], purpose="query")[0]
    store.add_chunks(
        [
            Chunk(
                id="tcs-p8-c0",
                text="Consolidated revenue in FY26 was ₹2,67,021 crore.",
                source="tcs-ar-26",
                page=8,
                chunk_index=0,
            )
        ],
        [q_vec],
    )

    # 2. Grounded answer with valid citation [1]
    gen_ok = FakeProvider(default_response="TCS consolidated revenue was ₹2,67,021 crore [1].")
    deps_ok = QueryDeps(settings=settings, embedder=embedder, generator=gen_ok, store=store)
    res_ok = answer_question("What was TCS revenue?", deps_ok)
    assert res_ok.grounded is True
    assert res_ok.reason == "ok"
    assert len(res_ok.citations) == 1
    assert res_ok.citations[0].page == 8
    assert res_ok.unsupported_numbers == []

    # 3. NOT_FOUND_MESSAGE reply -> grounded=False, reason="not_found"
    gen_nf = FakeProvider(default_response=NOT_FOUND_MESSAGE)
    deps_nf = QueryDeps(settings=settings, embedder=embedder, generator=gen_nf, store=store)
    res_nf = answer_question("What was Wipro revenue?", deps_nf)
    assert res_nf.grounded is False
    assert res_nf.reason == "not_found"
    assert res_nf.answer == NOT_FOUND_MESSAGE
    assert res_nf.citations == []

    # 4. Invalid citation [9] -> grounded=False, reason="no_valid_citations", invalid_citations=1
    gen_bad_cit = FakeProvider(default_response="TCS revenue was ₹2,67,021 crore [9].")
    deps_bad_cit = QueryDeps(
        settings=settings, embedder=embedder, generator=gen_bad_cit, store=store
    )
    res_bad_cit = answer_question("What was TCS revenue?", deps_bad_cit)
    assert res_bad_cit.grounded is False
    assert res_bad_cit.reason == "no_valid_citations"
    assert res_bad_cit.invalid_citations == 1
