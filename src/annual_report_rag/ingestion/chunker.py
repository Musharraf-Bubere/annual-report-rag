"""Document chunking without cross-page merging or word cuts."""

from dataclasses import dataclass

from annual_report_rag.ingestion.loader import LoadedPage


@dataclass(frozen=True)
class Chunk:
    """Represents a discrete text passage belonging to a specific source page."""

    id: str
    text: str
    source: str
    page: int
    chunk_index: int
    extraction_method: str = "default"
    text_quality: str = "clean"


def find_word_boundary_end(text: str, start: int, target_end: int) -> int:
    """Finds the nearest word boundary at or adjacent to the target end position."""
    text_len = len(text)
    end = text.rfind(" ", start, target_end)
    if end == -1 or end <= start:
        end = text.find(" ", target_end)
        if end == -1:
            end = text_len
    return end


def split_text_into_slices(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Splits single-page text into overlapping character windows without cutting words."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    slices: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        if text_len - start <= chunk_size:
            slices.append(text[start:].strip())
            break

        end = find_word_boundary_end(text, start, start + chunk_size)
        slices.append(text[start:end].strip())

        next_start = text.find(" ", max(start + 1, end - chunk_overlap))
        if next_start == -1 or next_start >= end:
            start = end
        else:
            start = next_start + 1

    return [s for s in slices if s]


def chunk_page(
    page: LoadedPage,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Generates structured chunks with stable identifiers for a single page."""
    slices = split_text_into_slices(page.text, chunk_size, chunk_overlap)
    return [
        Chunk(
            id=f"{page.source}-p{page.page}-c{idx}",
            text=text_slice,
            source=page.source,
            page=page.page,
            chunk_index=idx,
            extraction_method=page.extraction_method,
            text_quality=page.text_quality,
        )
        for idx, text_slice in enumerate(slices)
    ]


def chunk_pages(
    pages: list[LoadedPage],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Transforms a collection of loaded pages into stable chunks without cross-page merging."""
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(chunk_page(page, chunk_size, chunk_overlap))
    return chunks
