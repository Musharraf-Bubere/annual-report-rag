"""Unit tests for configuration loading and settings parsing."""

from pathlib import Path

import pytest

from annual_report_rag.config import (
    IngestionConfig,
    ProviderConfig,
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
    assert isinstance(settings.provider, ProviderConfig)
    assert isinstance(settings.provider.retry, RetryConfig)
    assert isinstance(settings.ingestion, IngestionConfig)
    assert isinstance(settings.retrieval, RetrievalConfig)
    assert isinstance(settings.storage, StorageConfig)

    assert settings.provider.name == "gemini"
    assert settings.provider.embedding_dimension == 768
    assert settings.provider.batch_size == 50
    assert settings.provider.retry.max_retries == 3
    assert settings.ingestion.chunk_size == 1000
    assert settings.ingestion.chunk_overlap == 200
    assert settings.ingestion.min_page_chars == 50
    assert settings.storage.collection_name == "annual_reports"


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
provider:
  name: "fake"
  embedding_model: "fake-embed"
  generation_model: "fake-gen"
  embedding_dimension: 256
  batch_size: 20
  retry:
    max_retries: 2
    initial_wait_seconds: 0.5
    max_wait_seconds: 2.0
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
        """,
        encoding="utf-8",
    )

    custom_env = tmp_path / ".env.test"
    custom_env.write_text("GEMINI_API_KEY=test_secret_key_123\n", encoding="utf-8")

    settings = load_settings(config_path=custom_yaml, env_path=custom_env)

    assert settings.provider.name == "fake"
    assert settings.provider.embedding_dimension == 256
    assert settings.provider.batch_size == 20
    assert settings.provider.retry.max_retries == 2
    assert settings.ingestion.chunk_size == 500
    assert settings.ingestion.min_page_chars == 10
    assert settings.storage.collection_name == "temp_collection"
    assert settings.gemini_api_key == "test_secret_key_123"
