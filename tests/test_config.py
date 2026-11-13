from __future__ import annotations

import pytest
from pydantic import ValidationError

from music_api.config import Environment, Settings


def test_settings_normalize_api_prefix() -> None:
    settings = Settings(api_prefix="api/v1/")

    assert settings.api_prefix == "/api/v1"


def test_settings_reject_invalid_port() -> None:
    with pytest.raises(ValidationError):
        Settings(api_port=70000)


def test_settings_reject_wildcard_cors_origin() -> None:
    with pytest.raises(ValidationError, match="explicit origins"):
        Settings(cors_origins=["*"])


def test_production_requires_explicit_database_configuration() -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL must be explicitly configured"):
        Settings(app_env=Environment.PRODUCTION)


def test_production_rejects_debug_and_public_docs() -> None:
    common = {
        "app_env": Environment.PRODUCTION,
        "database_url": "postgresql+asyncpg://service:secret@db.example/music",
        "redis_url": "redis://redis.example:6379/0",
    }

    with pytest.raises(ValidationError, match="DEBUG must be false"):
        Settings(**common, debug=True)
    with pytest.raises(ValidationError, match="DOCS_ENABLED must be false"):
        Settings(**common, docs_enabled=True)


def test_production_settings_require_redis_url_when_enabled() -> None:
    with pytest.raises(ValidationError, match="REDIS_URL must be explicitly configured"):
        Settings(
            app_env=Environment.PRODUCTION,
            database_url="postgresql+asyncpg://service:secret@db.example/music",
        )


def test_production_requires_postgresql() -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL must use PostgreSQL"):
        Settings(
            app_env=Environment.PRODUCTION,
            database_url="sqlite+aiosqlite:///production.db",
            redis_enabled=False,
        )


def test_production_can_disable_redis_and_docs() -> None:
    settings = Settings(
        app_env=Environment.PRODUCTION,
        database_url="postgresql+asyncpg://service:secret@db.example/music",
        redis_enabled=False,
        docs_enabled=False,
        storage_dev_auth_enabled=False,
    )

    assert settings.should_enable_docs is False
