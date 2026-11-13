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


def test_openapi_shows_functional_endpoints_only(client: TestClient) -> None:
    schema = client.get("/api/v1/openapi.json").json()
    paths = schema["paths"]

    assert "/api/v1/" not in paths
    assert not any(path.startswith("/api/v1/health") for path in paths)
    assert len(paths) == 13
    assert {tag["name"] for tag in schema["tags"]} == {"Music theory", "Rhythm"}
    assert paths["/api/v1/theory/notes/parse"]["post"]["summary"] == "Parse a musical note"
    assert (
        paths["/api/v1/music/metronome/schedule"]["post"]["summary"]
        == "Generate a deterministic metronome schedule"
    )


def test_openapi_uses_app_base_path_for_reverse_proxy(
    settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_BASE_PATH", "/chord-ai-api/")

    with TestClient(create_app(settings)) as proxy_client:
        schema = proxy_client.get("/api/v1/openapi.json").json()

    assert schema["servers"] == [{"url": "/chord-ai-api"}]
    assert (
        f"{schema['servers'][0]['url']}/api/v1/theory/notes/parse"
        == "/chord-ai-api/api/v1/theory/notes/parse"
    )
    assert "/api/v1/theory/notes/parse" in schema["paths"]
