from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    CONSOLE = "console"
    JSON = "json"


class StorageProviderName(StrEnum):
    LOCAL = "local"
    S3 = "s3"


class AIExecutionDevice(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
        frozen=True,
    )

    app_name: str = "Music Intelligence API"
    app_env: Environment = Environment.DEVELOPMENT
    debug: bool = False
    api_prefix: str = "/api/v1"
    api_version: str = "v1"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    database_url: str = (
        "postgresql+asyncpg://music_api:local-development-only@localhost:5432/music_api"
    )
    redis_enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"

    jwt_secret_key: SecretStr | None = None
    jwt_algorithm: str = "HS256"
    jwt_access_token_minutes: int = Field(default=15, gt=0)
    jwt_refresh_token_days: int = Field(default=30, gt=0)

    cors_origins: list[str] = Field(default_factory=list)
    max_upload_size_bytes: int = Field(default=104_857_600, gt=0)
    max_audio_duration_seconds: int = Field(default=21_600, gt=0)
    max_audio_output_size_bytes: int = Field(default=1_073_741_824, gt=0)
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    soundtouch_path: str = "soundstretch"
    soundtouch_version: str = "2.3.2+ds1-1"
    audio_conversion_timeout_seconds: int = Field(default=21_600, gt=0)
    audio_conversion_max_concurrency: int = Field(default=2, ge=1, le=32)
    ai_model_dir: Path = Path("./models")
    ai_execution_device: AIExecutionDevice = AIExecutionDevice.AUTO
    ai_inference_max_concurrency: int = Field(default=1, ge=1, le=32)
    ai_inference_timeout_seconds: int = Field(default=21_600, gt=0)
    ai_allow_model_downloads: bool = False
    temp_dir: str = "./tmp"
    storage_dir: str = "./storage"
    storage_provider: StorageProviderName = StorageProviderName.LOCAL
    storage_s3_endpoint_url: str | None = None
    storage_s3_region: str = "us-east-1"
    storage_s3_bucket: str = "music-api"
    storage_s3_access_key_id: SecretStr | None = None
    storage_s3_secret_access_key: SecretStr | None = None
    storage_s3_use_tls: bool = True
    storage_s3_presigned_url_ttl_seconds: int = Field(default=300, ge=1, le=3600)
    user_storage_quota_bytes: int = Field(default=5_368_709_120, gt=0)
    upload_staging_ttl_seconds: int = Field(default=86_400, gt=0)
    storage_dev_auth_enabled: bool = True
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.CONSOLE
    docs_enabled: bool | None = None

    external_service_base_url: str | None = None
    external_service_api_key: SecretStr | None = None

    @field_validator("api_prefix")
    @classmethod
    def normalize_api_prefix(cls, value: str) -> str:
        normalized = "/" + value.strip("/")
        if normalized == "/":
            raise ValueError("API prefix must contain a path segment.")
        return normalized

    @field_validator("api_version")
    @classmethod
    def validate_api_version(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "/" in normalized:
            raise ValueError("API_VERSION must be a non-empty path segment.")
        return normalized

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, value: list[str]) -> list[str]:
        if "*" in value:
            raise ValueError(
                "CORS_ORIGINS must list explicit origins; wildcard origins are not allowed."
            )
        return value

    @field_validator("database_url")
    @classmethod
    def validate_database_scheme(cls, value: str) -> str:
        if not value.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
            raise ValueError("DATABASE_URL must use postgresql+asyncpg or sqlite+aiosqlite.")
        return value

    @field_validator("redis_url")
    @classmethod
    def validate_redis_scheme(cls, value: str) -> str:
        if not value.startswith(("redis://", "rediss://", "unix://")):
            raise ValueError("REDIS_URL must use redis://, rediss://, or unix://.")
        return value

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("LOG_LEVEL must be one of CRITICAL, ERROR, WARNING, INFO, DEBUG.")
        return normalized

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if self.app_env == Environment.PRODUCTION:
            explicitly_set = self.model_fields_set
            if "database_url" not in explicitly_set:
                raise ValueError("DATABASE_URL must be explicitly configured in production.")
            if not self.database_url.startswith("postgresql+asyncpg://"):
                raise ValueError("DATABASE_URL must use PostgreSQL in production.")
            if self.redis_enabled and "redis_url" not in explicitly_set:
                raise ValueError("REDIS_URL must be explicitly configured when Redis is enabled.")
            if self.debug:
                raise ValueError("DEBUG must be false in production.")
            if self.docs_enabled:
                raise ValueError("DOCS_ENABLED must be false in production.")
            if self.storage_provider == StorageProviderName.S3:
                if self.storage_s3_access_key_id is None:
                    raise ValueError("STORAGE_S3_ACCESS_KEY_ID is required for S3 in production.")
                if self.storage_s3_secret_access_key is None:
                    raise ValueError(
                        "STORAGE_S3_SECRET_ACCESS_KEY is required for S3 in production."
                    )
            if self.storage_dev_auth_enabled:
                raise ValueError("STORAGE_DEV_AUTH_ENABLED must be false in production.")
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION

    @property
    def should_enable_docs(self) -> bool:
        if self.docs_enabled is not None:
            return self.docs_enabled
        return not self.is_production


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
