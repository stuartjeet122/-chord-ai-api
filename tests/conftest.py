from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine

from music_api.audio import models as audio_models  # noqa: F401
from music_api.config import Environment, Settings
from music_api.db.base import Base
from music_api.jobs import models as job_models  # noqa: F401
from music_api.main import create_app
from music_api.music import models as music_models  # noqa: F401
from music_api.storage import models as storage_models  # noqa: F401


def _create_schema(settings: Settings) -> None:
    async def create() -> None:
        engine = create_async_engine(settings.database_url)
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(create())


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        app_env=Environment.TESTING,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        redis_enabled=False,
        temp_dir=str(tmp_path / "tmp"),
        storage_dir=str(tmp_path / "storage"),
        docs_enabled=True,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    _create_schema(settings)
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def worker_settings(settings: Settings) -> Settings:
    options = settings.model_dump()
    options["redis_enabled"] = True
    return Settings(**options)


@pytest.fixture
def worker_client(worker_settings: Settings) -> Iterator[TestClient]:
    _create_schema(worker_settings)
    app = create_app(worker_settings)
    with TestClient(app) as test_client:
        yield test_client
