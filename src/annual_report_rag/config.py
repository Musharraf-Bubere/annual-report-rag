"""Configuration loading and typed settings definition."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class RetryConfig:
    """Holds retry parameters for transient API failures."""

    max_retries: int
    max_rate_limit_retries: int
    initial_wait_seconds: float
    max_wait_seconds: float


@dataclass(frozen=True)
class EmbeddingConfig:
    """Holds configuration for embedding provider, dimension, batching, and retries."""

    provider: str
    model: str
    dimension: int
    batch_size: int
    max_items_per_minute: int
    retry: RetryConfig

    @property
    def name(self) -> str:
        """Alias for provider name."""
        return self.provider

    @property
    def embedding_model(self) -> str:
        """Alias for embedding model name."""
        return self.model

    @property
    def embedding_dimension(self) -> int:
        """Alias for embedding dimension."""
        return self.dimension


@dataclass(frozen=True)
class GenerationConfig:
    """Holds configuration for generation provider, model, and temperature."""

    provider: str
    model: str
    temperature: float | None = None

    @property
    def name(self) -> str:
        """Alias for generation provider name."""
        return self.provider

    @property
    def generation_model(self) -> str:
        """Alias for generation model name."""
        return self.model


@dataclass(frozen=True)
class IngestionConfig:
    """Holds chunking and ingestion parameters for document processing."""

    chunk_size: int
    chunk_overlap: int
    min_page_chars: int


@dataclass(frozen=True)
class RetrievalConfig:
    """Holds retrieval search parameters such as top-k results, score threshold, input limit, query prefix, per-page cap, and neighbour expansion."""

    top_k: int = 6
    min_score: float = 0.0
    max_question_chars: int = 1000
    query_prefix: str | None = None
    max_chunks_per_page: int | None = None
    neighbor_window: int = 0
    neighbor_top_n: int = 3
    max_context_chunks: int | None = None


@dataclass(frozen=True)
class PromptConfig:
    """Holds prompt construction feature switches."""

    rules_v2: bool = False


@dataclass(frozen=True)
class ApiConfig:
    """Holds FastAPI server configuration such as allowed CORS origins."""

    cors_origins: tuple[str, ...] = ("http://localhost:8501",)


@dataclass(frozen=True)
class SourceMetadata:
    """Holds human-readable display name and routing aliases for a document source."""

    display_name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class StorageConfig:
    """Holds filesystem paths and collection name for vector and document storage."""

    vector_store_path: str
    collection_name: str
    raw_data_dir: str
    eval_data_dir: str


@dataclass(frozen=True)
class QualityConfig:
    """Holds anomaly thresholds for text quality auditing."""

    single_letter_ratio_threshold: float = 0.15
    odd_char_ratio_threshold: float = 0.002
    odd_char_count_threshold: int = 1
    number_split_threshold: int = 3
    corrupted_glyph_range_start: str = "\u0100"
    corrupted_glyph_range_end: str = "\u052F"
    keep_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationConfig:
    """Holds evaluation harness configuration and optional token pricing."""

    golden_set_path: str = "data/eval/golden_questions.json"
    price_per_million_input_usd: float | None = None
    price_per_million_output_usd: float | None = None


@dataclass(frozen=True)
class Settings:
    """Aggregates all application configurations and secrets into a single typed object."""

    embedding: EmbeddingConfig
    generation: GenerationConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    storage: StorageConfig
    quality: QualityConfig = QualityConfig()
    prompt: PromptConfig = PromptConfig()
    api: ApiConfig = ApiConfig()
    sources: dict[str, SourceMetadata] = field(default_factory=dict)
    evaluation: EvaluationConfig = EvaluationConfig()
    gemini_api_key: str | None = None

    @property
    def provider(self) -> EmbeddingConfig:
        """Backward compatibility alias pointing to embedding settings."""
        return self.embedding


def load_yaml_file(file_path: Path) -> dict[str, Any]:
    """Reads and parses a YAML configuration file from the filesystem."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def parse_retry_config(retry_data: dict[str, Any]) -> RetryConfig:
    """Parses retry policy parameters into a typed RetryConfig instance."""
    return RetryConfig(
        max_retries=int(retry_data.get("max_retries", 3)),
        max_rate_limit_retries=int(retry_data.get("max_rate_limit_retries", 5)),
        initial_wait_seconds=float(retry_data.get("initial_wait_seconds", 1.0)),
        max_wait_seconds=float(retry_data.get("max_wait_seconds", 60.0)),
    )


def parse_embedding_config(data: dict[str, Any]) -> EmbeddingConfig:
    """Parses embedding provider, model, dimension, and retry configurations."""
    provider_name = str(data.get("provider", data.get("name", "local")))
    model_name = str(
        data.get("model", data.get("embedding_model", "BAAI/bge-base-en-v1.5"))
    )
    dimension = int(data.get("dimension", data.get("embedding_dimension", 768)))
    batch_size = int(data.get("batch_size", 20))
    max_items_per_minute = int(data.get("max_items_per_minute", 80))
    retry = parse_retry_config(data.get("retry", {}))

    return EmbeddingConfig(
        provider=provider_name,
        model=model_name,
        dimension=dimension,
        batch_size=batch_size,
        max_items_per_minute=max_items_per_minute,
        retry=retry,
    )


def parse_generation_config(data: dict[str, Any]) -> GenerationConfig:
    """Parses generation provider, model, and temperature configurations."""
    provider_name = str(data.get("provider", data.get("name", "gemini")))
    model_name = str(
        data.get("model", data.get("generation_model", "gemini-3.5-flash-lite"))
    )
    raw_temp = data.get("temperature")
    temperature = float(raw_temp) if raw_temp is not None else None
    return GenerationConfig(
        provider=provider_name,
        model=model_name,
        temperature=temperature,
    )


def parse_ingestion_config(data: dict[str, Any]) -> IngestionConfig:
    """Parses ingestion parameters into a typed IngestionConfig instance."""
    return IngestionConfig(
        chunk_size=int(data.get("chunk_size", 1000)),
        chunk_overlap=int(data.get("chunk_overlap", 200)),
        min_page_chars=int(data.get("min_page_chars", 50)),
    )


def parse_retrieval_config(data: dict[str, Any]) -> RetrievalConfig:
    """Parses retrieval parameters into a typed RetrievalConfig instance."""
    raw_prefix = data.get("query_prefix")
    query_prefix = str(raw_prefix) if raw_prefix is not None else None
    raw_max_per_page = data.get("max_chunks_per_page")
    max_chunks_per_page = (
        int(raw_max_per_page) if raw_max_per_page is not None else None
    )
    raw_neighbor_window = data.get("neighbor_window", 0)
    neighbor_window = int(raw_neighbor_window) if raw_neighbor_window is not None else 0
    raw_neighbor_top_n = data.get("neighbor_top_n", 3)
    neighbor_top_n = int(raw_neighbor_top_n) if raw_neighbor_top_n is not None else 3
    raw_max_ctx = data.get("max_context_chunks")
    max_context_chunks = int(raw_max_ctx) if raw_max_ctx is not None else None
    return RetrievalConfig(
        top_k=int(data.get("top_k", 6)),
        min_score=float(data.get("min_score", 0.0)),
        max_question_chars=int(data.get("max_question_chars", 1000)),
        query_prefix=query_prefix,
        max_chunks_per_page=max_chunks_per_page,
        neighbor_window=neighbor_window,
        neighbor_top_n=neighbor_top_n,
        max_context_chunks=max_context_chunks,
    )


def parse_prompt_config(data: dict[str, Any]) -> PromptConfig:
    """Parses prompt configuration switches into a typed PromptConfig instance."""
    if not isinstance(data, dict):
        return PromptConfig()
    return PromptConfig(rules_v2=bool(data.get("rules_v2", False)))


def parse_api_config(data: dict[str, Any]) -> ApiConfig:
    """Parses API server parameters into a typed ApiConfig instance."""
    raw_origins = data.get("cors_origins", ["http://localhost:8501"])
    origins = (
        tuple(str(o) for o in raw_origins)
        if isinstance(raw_origins, list)
        else ("http://localhost:8501",)
    )
    return ApiConfig(cors_origins=origins)


def parse_sources_config(data: dict[str, Any]) -> dict[str, SourceMetadata]:
    """Parses the sources registry mapping source stems to display names and aliases."""
    if not isinstance(data, dict):
        return {}
    sources: dict[str, SourceMetadata] = {}
    for stem, meta in data.items():
        if not isinstance(meta, dict):
            continue
        display_name = str(meta.get("display_name", stem))
        raw_aliases = meta.get("aliases", [])
        aliases = (
            tuple(str(a) for a in raw_aliases)
            if isinstance(raw_aliases, list)
            else ()
        )
        sources[str(stem)] = SourceMetadata(
            display_name=display_name,
            aliases=aliases,
        )
    return sources


def parse_storage_config(data: dict[str, Any]) -> StorageConfig:
    """Parses storage paths and collection configuration into StorageConfig."""
    return StorageConfig(
        vector_store_path=str(data.get("vector_store_path", "chroma_db")),
        collection_name=str(data.get("collection_name", "annual_reports")),
        raw_data_dir=str(data.get("raw_data_dir", "data/raw")),
        eval_data_dir=str(data.get("eval_data_dir", "data/eval")),
    )


def parse_quality_config(data: dict[str, Any]) -> QualityConfig:
    """Parses text quality thresholds into a typed QualityConfig instance."""
    raw_keep = data.get("keep_tokens", [])
    keep_tokens = (
        tuple(str(t) for t in raw_keep) if isinstance(raw_keep, list) else ()
    )
    return QualityConfig(
        single_letter_ratio_threshold=float(
            data.get("single_letter_ratio_threshold", 0.15)
        ),
        odd_char_ratio_threshold=float(
            data.get("odd_char_ratio_threshold", 0.002)
        ),
        odd_char_count_threshold=int(
            data.get("odd_char_count_threshold", 1)
        ),
        number_split_threshold=int(
            data.get("number_split_threshold", 3)
        ),
        corrupted_glyph_range_start=str(
            data.get("corrupted_glyph_range_start", "\u0100")
        ),
        corrupted_glyph_range_end=str(
            data.get("corrupted_glyph_range_end", "\u052F")
        ),
        keep_tokens=keep_tokens,
    )


def parse_evaluation_config(data: dict[str, Any]) -> EvaluationConfig:
    """Parses evaluation harness and optional token pricing settings."""
    if not isinstance(data, dict):
        return EvaluationConfig()
    raw_in = data.get("price_per_million_input_usd")
    raw_out = data.get("price_per_million_output_usd")
    return EvaluationConfig(
        golden_set_path=str(
            data.get("golden_set_path", "data/eval/golden_questions.json")
        ),
        price_per_million_input_usd=float(raw_in) if raw_in is not None else None,
        price_per_million_output_usd=float(raw_out) if raw_out is not None else None,
    )


def load_settings(
    config_path: Path | str = "config/settings.yaml",
    env_path: Path | str = ".env",
) -> Settings:
    """Loads YAML settings and .env secrets into an aggregated Settings instance."""
    resolved_env = Path(env_path)
    if resolved_env.exists():
        load_dotenv(dotenv_path=resolved_env, override=True)
    else:
        load_dotenv(override=False)

    data = load_yaml_file(Path(config_path))
    raw_emb = data.get("embedding", data.get("provider", {}))
    raw_gen = data.get("generation", data.get("provider", {}))

    return Settings(
        embedding=parse_embedding_config(raw_emb),
        generation=parse_generation_config(raw_gen),
        ingestion=parse_ingestion_config(data.get("ingestion", {})),
        retrieval=parse_retrieval_config(data.get("retrieval", {})),
        storage=parse_storage_config(data.get("storage", {})),
        quality=parse_quality_config(data.get("quality", {})),
        prompt=parse_prompt_config(data.get("prompt", {})),
        api=parse_api_config(data.get("api", {})),
        sources=parse_sources_config(data.get("sources", {})),
        evaluation=parse_evaluation_config(data.get("evaluation", {})),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )
