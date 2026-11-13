from __future__ import annotations

from fastapi.testclient import TestClient


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
