"""Unit tests for configuration loading and settings parsing."""

from pathlib import Path

import pytest

from annual_report_rag.config import (
    EmbeddingConfig,
    GenerationConfig,
    IngestionConfig,
    QualityConfig,
    RetrievalConfig,
    RetryConfig,
    Settings,
    StorageConfig,
    load_settings,
    load_yaml_file,
)


def test_load_default_settings() -> None:
    """Verifies that the default project settings.yaml loads into typed Settings."""
    settings = load_settings("config/settings.yaml")

    assert isinstance(settings, Settings)
    assert isinstance(settings.embedding, EmbeddingConfig)
    assert isinstance(settings.generation, GenerationConfig)
    assert isinstance(settings.embedding.retry, RetryConfig)
    assert isinstance(settings.ingestion, IngestionConfig)
    assert isinstance(settings.retrieval, RetrievalConfig)
    assert isinstance(settings.storage, StorageConfig)
    assert isinstance(settings.quality, QualityConfig)

    assert settings.embedding.provider == "local"
    assert settings.embedding.model == "BAAI/bge-base-en-v1.5"
    assert settings.embedding.dimension == 768
    assert settings.embedding.batch_size == 20
    assert settings.embedding.max_items_per_minute == 80
    assert settings.embedding.retry.max_retries == 3
    assert settings.embedding.retry.max_rate_limit_retries == 5

    assert settings.generation.provider == "gemini"
    assert settings.generation.model == "gemini-3.5-flash-lite"

    assert settings.ingestion.chunk_size == 1000
    assert settings.ingestion.chunk_overlap == 200
    assert settings.ingestion.min_page_chars == 50
    assert settings.storage.collection_name == "annual_reports"
    assert settings.quality.single_letter_ratio_threshold == 0.15
    assert settings.quality.odd_char_ratio_threshold == 0.002
    assert settings.quality.odd_char_count_threshold == 1
    assert settings.quality.number_split_threshold == 3
    assert settings.quality.corrupted_glyph_range_start == "\u0100"
    assert settings.quality.corrupted_glyph_range_end == "\u052F"
    assert settings.quality.keep_tokens == ("Mondelēz",)


def test_missing_config_file_raises_error(tmp_path: Path) -> None:
    """Verifies that an error is raised when the configuration file does not exist."""
    missing_file = tmp_path / "non_existent.yaml"
    with pytest.raises(FileNotFoundError):
        load_yaml_file(missing_file)


def test_custom_yaml_and_env_loading(tmp_path: Path) -> None:
    """Verifies that settings can be loaded from custom temporary yaml and env files."""
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(
        """
embedding:
  provider: "fake"
  model: "fake-embed"
  dimension: 256
  batch_size: 15
  max_items_per_minute: 60
  retry:
    max_retries: 2
    max_rate_limit_retries: 4
    initial_wait_seconds: 0.5
    max_wait_seconds: 2.0
generation:
  provider: "fake"
  model: "fake-gen"
ingestion:
  chunk_size: 500
  chunk_overlap: 50
  min_page_chars: 10
retrieval:
  top_k: 3
storage:
  vector_store_path: "temp_db"
  collection_name: "temp_collection"
  raw_data_dir: "temp_raw"
  eval_data_dir: "temp_eval"
quality:
  single_letter_ratio_threshold: 0.20
  odd_char_ratio_threshold: 0.08
  odd_char_count_threshold: 2
        """,
        encoding="utf-8",
    )

    custom_env = tmp_path / ".env.test"
    custom_env.write_text("GEMINI_API_KEY=test_secret_key_123\n", encoding="utf-8")

    settings = load_settings(config_path=custom_yaml, env_path=custom_env)

    assert settings.embedding.provider == "fake"
    assert settings.embedding.model == "fake-embed"
    assert settings.embedding.dimension == 256
    assert settings.embedding.batch_size == 15
    assert settings.embedding.max_items_per_minute == 60
    assert settings.embedding.retry.max_retries == 2
    assert settings.embedding.retry.max_rate_limit_retries == 4

    assert settings.generation.provider == "fake"
    assert settings.generation.model == "fake-gen"

    assert settings.ingestion.chunk_size == 500
    assert settings.ingestion.min_page_chars == 10
    assert settings.storage.collection_name == "temp_collection"
    assert settings.quality.single_letter_ratio_threshold == 0.20
    assert settings.quality.odd_char_ratio_threshold == 0.08
    assert settings.quality.odd_char_count_threshold == 2
    assert settings.gemini_api_key == "test_secret_key_123"
