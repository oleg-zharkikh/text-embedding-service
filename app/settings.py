from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
    )

    model_name: str = Field(
        default='BAAI/bge-m3',
        description='Sentence transformer model name'
    )

    device: str | None = Field(
        default=None,
        description='Device for model inference (cpu, cuda, mps). None for auto-detection'
    )

    batch_size: int = Field(
        default=32,
        description='Batch size for encoding',
        ge=1
    )

    normalize_embeddings: bool = Field(
        default=True,
        description='Normalize embeddings to unit length'
    )

    task_ttl_minutes: int = Field(
        default=60,
        description='Time to keep completed tasks before cleanup (minutes)',
        ge=1
    )

    max_texts_per_task: int = Field(
        default=10000,
        description='Maximum number of texts per A2A task',
        ge=1
    )

    worker_threads: int = Field(
        default=4,
        description='Number of worker threads for background tasks',
        ge=1
    )

    log_file: str = Field(
        default='app.log',
        description='Log file path'
    )

    host: str = Field(
        default='0.0.0.0',
        description='Server host'
    )

    port: int = Field(
        default=5000,
        description='Server port',
        ge=1,
        le=65535
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
