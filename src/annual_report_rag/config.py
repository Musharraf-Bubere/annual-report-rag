"""Configuration loading and typed settings definition."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class ProviderConfig:
    """Holds configuration for model provider name and specific model identifiers."""

    name: str
    embedding_model: str
    generation_model: str


@dataclass(frozen=True)
class IngestionConfig:
    """Holds chunking and ingestion parameters for document processing."""

    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class RetrievalConfig:
    """Holds retrieval search parameters such as top-k results."""

    top_k: int


@dataclass(frozen=True)
class StorageConfig:
    """Holds filesystem paths for raw files, eval sets, and vector storage."""

    vector_store_path: str
    raw_data_dir: str
    eval_data_dir: str


@dataclass(frozen=True)
class Settings:
    """Aggregates all application configurations and secrets into a single typed object."""

    provider: ProviderConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    storage: StorageConfig
    gemini_api_key: str | None = None


def load_yaml_file(file_path: Path) -> dict[str, Any]:
    """Reads and parses a YAML configuration file from the filesystem."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_settings(
    config_path: Path | str = "config/settings.yaml",
    env_path: Path | str = ".env",
) -> Settings:
    """Loads settings from YAML and secrets from environment variables into a typed Settings instance."""
    resolved_env_path = Path(env_path)
    if resolved_env_path.exists():
        load_dotenv(dotenv_path=resolved_env_path, override=True)
    else:
        load_dotenv(override=False)

    yaml_data = load_yaml_file(Path(config_path))
    provider_data = yaml_data.get("provider", {})
    ingestion_data = yaml_data.get("ingestion", {})
    retrieval_data = yaml_data.get("retrieval", {})
    storage_data = yaml_data.get("storage", {})

    return Settings(
        provider=ProviderConfig(
            name=provider_data.get("name", "gemini"),
            embedding_model=provider_data.get("embedding_model", "text-embedding-004"),
            generation_model=provider_data.get("generation_model", "gemini-2.5-flash"),
        ),
        ingestion=IngestionConfig(
            chunk_size=int(ingestion_data.get("chunk_size", 1000)),
            chunk_overlap=int(ingestion_data.get("chunk_overlap", 200)),
        ),
        retrieval=RetrievalConfig(
            top_k=int(retrieval_data.get("top_k", 5)),
        ),
        storage=StorageConfig(
            vector_store_path=storage_data.get("vector_store_path", "chroma_db"),
            raw_data_dir=storage_data.get("raw_data_dir", "data/raw"),
            eval_data_dir=storage_data.get("eval_data_dir", "data/eval"),
        ),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )
