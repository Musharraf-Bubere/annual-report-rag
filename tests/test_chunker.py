"""Unit tests for document chunking logic."""

from annual_report_rag.ingestion.chunker import chunk_page, chunk_pages
from annual_report_rag.ingestion.loader import LoadedPage


def test_chunk_size_and_overlap() -> None:
    """Verifies that chunks adhere to target character bounds with overlapping windows."""
    long_text = " ".join([f"word{i}" for i in range(200)])
    page = LoadedPage(source="report2026", page=1, text=long_text)

    chunks = chunk_page(page, chunk_size=100, chunk_overlap=30)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 120  # Allows boundary padding for full words
        assert chunk.source == "report2026"
        assert chunk.page == 1


def test_no_word_is_cut() -> None:
    """Verifies that chunk splits strictly occur at whitespace boundaries."""
    text = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda"
    page = LoadedPage(source="alphabet", page=3, text=text)

    chunks = chunk_page(page, chunk_size=25, chunk_overlap=10)

    known_words = set(text.split())
    for chunk in chunks:
        chunk_words = chunk.text.split()
        for word in chunk_words:
            assert word in known_words


def test_no_cross_page_merging() -> None:
    """Verifies that text from different pages is never blended into a single chunk."""
    page1 = LoadedPage(source="infy", page=1, text="Text from page one.")
    page2 = LoadedPage(source="infy", page=2, text="Text from page two.")

    chunks = chunk_pages([page1, page2], chunk_size=500, chunk_overlap=100)

    assert len(chunks) == 2
    assert chunks[0].page == 1
    assert "page two" not in chunks[0].text
    assert chunks[1].page == 2
    assert "page one" not in chunks[1].text


def test_stable_ids() -> None:
    """Verifies that chunk IDs follow the predictable {source}-p{page}-c{chunk_index} format."""
    page = LoadedPage(
        source="tcs",
        page=42,
        text="Sample text that is intentionally long enough to split into multiple chunks.",
    )
    chunks1 = chunk_page(page, chunk_size=30, chunk_overlap=10)
    chunks2 = chunk_page(page, chunk_size=30, chunk_overlap=10)

    assert len(chunks1) > 1
    for idx, chunk in enumerate(chunks1):
        assert chunk.id == f"tcs-p42-c{idx}"
        assert chunk.chunk_index == idx
        assert chunk.id == chunks2[idx].id
