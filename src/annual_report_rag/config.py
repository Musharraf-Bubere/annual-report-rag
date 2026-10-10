"""Configuration loading and typed settings definition."""

import os
from dataclasses import dataclass
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
class ProviderConfig:
    """Holds configuration for model provider, dimensions, batching, and retries."""

    name: str
    embedding_model: str
    generation_model: str
    embedding_dimension: int
    batch_size: int
    max_items_per_minute: int
    retry: RetryConfig


@dataclass(frozen=True)
class IngestionConfig:
    """Holds chunking and ingestion parameters for document processing."""

    chunk_size: int
    chunk_overlap: int
    min_page_chars: int


@dataclass(frozen=True)
class RetrievalConfig:
    """Holds retrieval search parameters such as top-k results."""

    top_k: int


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
    odd_char_count_threshold: int = 15
    number_split_threshold: int = 3
    corrupted_glyph_range_start: str = "\u0100"
    corrupted_glyph_range_end: str = "\u052F"


@dataclass(frozen=True)
class Settings:
    """Aggregates all application configurations and secrets into a single typed object."""

    provider: ProviderConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    storage: StorageConfig
    quality: QualityConfig
    gemini_api_key: str | None = None


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


def parse_provider_config(data: dict[str, Any]) -> ProviderConfig:
    """Parses provider, rate limit, and retry parameters into a typed ProviderConfig."""
    return ProviderConfig(
        name=str(data.get("name", "gemini")),
        embedding_model=str(data.get("embedding_model", "gemini-embedding-2")),
        generation_model=str(data.get("generation_model", "gemini-3.5-flash-lite")),
        embedding_dimension=int(data.get("embedding_dimension", 768)),
        batch_size=int(data.get("batch_size", 20)),
        max_items_per_minute=int(data.get("max_items_per_minute", 80)),
        retry=parse_retry_config(data.get("retry", {})),
    )


def parse_ingestion_config(data: dict[str, Any]) -> IngestionConfig:
    """Parses ingestion parameters into a typed IngestionConfig instance."""
    return IngestionConfig(
        chunk_size=int(data.get("chunk_size", 1000)),
        chunk_overlap=int(data.get("chunk_overlap", 200)),
        min_page_chars=int(data.get("min_page_chars", 50)),
    )


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
    return QualityConfig(
        single_letter_ratio_threshold=float(
            data.get("single_letter_ratio_threshold", 0.15)
        ),
        odd_char_ratio_threshold=float(
            data.get("odd_char_ratio_threshold", 0.002)
        ),
        odd_char_count_threshold=int(
            data.get("odd_char_count_threshold", 15)
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
    return Settings(
        provider=parse_provider_config(data.get("provider", {})),
        ingestion=parse_ingestion_config(data.get("ingestion", {})),
        retrieval=RetrievalConfig(top_k=int(data.get("retrieval", {}).get("top_k", 5))),
        storage=parse_storage_config(data.get("storage", {})),
        quality=parse_quality_config(data.get("quality", {})),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )
