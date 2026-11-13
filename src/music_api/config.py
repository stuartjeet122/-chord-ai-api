from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    CONSOLE = "console"
    JSON = "json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
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
    cors_origins: list[str] = Field(default_factory=list)
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.CONSOLE
    docs_enabled: bool | None = None

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

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("LOG_LEVEL must be one of CRITICAL, ERROR, WARNING, INFO, DEBUG.")
        return normalized

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
