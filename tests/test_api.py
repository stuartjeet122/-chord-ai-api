from __future__ import annotations

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from music_api.config import Settings
from music_api.main import create_app


def test_root_reports_service_and_docs(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Music Intelligence API is running.",
        "api": "/api/v1",
        "docs": "/api/v1/docs",
    }


def test_versioned_root_returns_api_information(client: TestClient) -> None:
    response = client.get("/api/v1/")

    assert response.status_code == 200
    assert response.json()["api_version"] == "v1"


def test_swagger_is_available(client: TestClient) -> None:
    docs = client.get("/api/v1/docs")
    assert docs.status_code == 200
    assert '"url": "./openapi.json"' in docs.text
    assert client.get("/api/v1/openapi.json").status_code == 200


def test_openapi_uses_app_base_path_for_reverse_proxy(
    settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_BASE_PATH", "/chord-ai-api/")

    with TestClient(create_app(settings)) as proxy_client:
        schema = proxy_client.get("/api/v1/openapi.json").json()

    assert schema["servers"] == [{"url": "/chord-ai-api"}]
    assert f"{schema['servers'][0]['url']}{next(iter(schema['paths']))}" == (
        "/chord-ai-api/api/v1/"
    )
