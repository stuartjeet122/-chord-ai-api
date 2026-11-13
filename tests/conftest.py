from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from music_api.config import Environment, Settings
from music_api.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(app_env=Environment.TESTING, docs_enabled=True)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
