from __future__ import annotations

from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from music_api.config import Environment, Settings
from music_api.db.session import Database
from music_api.main import create_app


def test_health_and_liveness_are_dependency_independent(client: TestClient) -> None:
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert client.get("/api/v1/health/live").json() == {"status": "alive"}


def test_readiness_checks_database_and_optional_redis(client: TestClient) -> None:
    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "database": "ok",
            "ffmpeg": "ok",
            "ffprobe": "ok",
        },
    }


def test_readiness_returns_service_unavailable_when_database_is_down(
    client: TestClient,
) -> None:
    async def fail_ping() -> None:
        raise OperationalError("SELECT 1", {}, OSError("database unavailable"))

    client.app.state.resources.database.ping = fail_ping

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"] == {
        "database": "unavailable",
        "ffmpeg": "ok",
        "ffprobe": "ok",
    }


def test_readiness_detects_unavailable_redis(settings) -> None:
    redis_settings = Settings(
        app_env=Environment.TESTING,
        database_url=settings.database_url,
        redis_enabled=True,
        redis_url="redis://localhost:6379/0",
    )
    app = create_app(redis_settings)

    with TestClient(app) as redis_client:

        async def fail_ping() -> bool:
            raise RedisConnectionError("redis unavailable")

        redis_client.app.state.resources.redis.ping = fail_ping
        response = redis_client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "database": "ok",
        "ffmpeg": "ok",
        "ffprobe": "ok",
        "redis": "unavailable",
    }


def test_readiness_reports_missing_audio_tools(settings) -> None:
    missing_settings = Settings(
        app_env=Environment.TESTING,
        database_url=settings.database_url,
        redis_enabled=False,
        temp_dir=settings.temp_dir,
        storage_dir=settings.storage_dir,
        ffmpeg_path="ffmpeg-not-installed",
        ffprobe_path="ffprobe-not-installed",
    )
    app = create_app(missing_settings)
    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "database": "ok",
        "ffmpeg": "unavailable",
        "ffprobe": "unavailable",
    }


async def test_database_session_lifecycle(settings) -> None:
    database = Database(settings.database_url)

    try:
        async with database.session_factory() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar_one() == 1
            assert session.is_active is True
    finally:
        await database.close()
