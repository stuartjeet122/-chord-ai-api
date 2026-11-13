from __future__ import annotations

import logging
from uuid import UUID

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError


def test_root_returns_safe_service_information(client: TestClient) -> None:
    response = client.get("/api/v1/")

    assert response.status_code == 200
    assert response.json() == {
        "application": "Music Intelligence API",
        "api_version": "v1",
        "environment": "testing",
    }


def test_docs_and_openapi_are_available_in_testing(client: TestClient) -> None:
    assert client.get("/api/v1/docs").status_code == 200
    schema = client.get("/api/v1/openapi.json")

    assert schema.status_code == 200
    assert schema.json()["info"]["version"] == "1.0.0"
    assert "/api/v1/health/ready" in schema.json()["paths"]


def test_not_found_uses_problem_details_and_request_id(client: TestClient) -> None:
    response = client.get("/api/v1/unknown")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.headers["x-request-id"] == response.json()["trace_id"]
    assert response.json()["code"] == "http_404"


def test_invalid_path_parameter_uses_consistent_validation_error(
    client: TestClient,
) -> None:
    async def validation_probe(item_id: int) -> dict[str, int]:
        return {"item_id": item_id}

    client.app.add_api_route(
        "/api/v1/_test-validation/{item_id}",
        validation_probe,
        methods=["GET"],
    )
    response = client.get("/api/v1/_test-validation/not-an-integer")

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "request_validation_failed"
    assert response.json()["errors"][0]["location"] == "path.item_id"
    assert "not-an-integer" not in response.text


def test_unexpected_error_does_not_expose_exception_details(
    client: TestClient,
) -> None:
    async def error_probe() -> None:
        raise RuntimeError("sensitive internal detail")

    client.app.add_api_route("/api/v1/_test-error", error_probe, methods=["GET"])
    response = client.get("/api/v1/_test-error")

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/problem+json")
    assert "sensitive internal detail" not in response.text
    assert response.headers["x-request-id"] == response.json()["trace_id"]


def test_http_error_does_not_echo_arbitrary_detail(client: TestClient) -> None:
    async def forbidden_probe() -> None:
        raise HTTPException(status_code=403, detail="token=private-value")

    client.app.add_api_route("/api/v1/_test-forbidden", forbidden_probe, methods=["GET"])
    response = client.get("/api/v1/_test-forbidden")

    assert response.status_code == 403
    assert "private-value" not in response.text
    assert response.json()["detail"] == "The operation is not permitted."


def test_database_exception_returns_safe_problem_response(client: TestClient) -> None:
    async def database_probe() -> None:
        raise OperationalError("SELECT 1", {}, RuntimeError("private connection detail"))

    client.app.add_api_route("/api/v1/_test-database", database_probe, methods=["GET"])
    response = client.get("/api/v1/_test-database")

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")
    assert "private connection detail" not in response.text
    assert response.json()["code"] == "database_unavailable"


def test_request_id_is_generated_and_returned(client: TestClient) -> None:
    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    UUID(response.headers["x-request-id"])


def test_request_log_contains_request_context(
    client: TestClient,
    caplog,
) -> None:
    with caplog.at_level(logging.INFO, logger="music_api.requests"):
        response = client.get("/api/v1/health/live")

    request_log = next(record for record in caplog.records if record.name == "music_api.requests")
    assert request_log.request_id == response.headers["x-request-id"]
    assert request_log.method == "GET"
    assert request_log.path == "/api/v1/health/live"
    assert request_log.status_code == 200
    assert request_log.duration_ms >= 0


def test_valid_incoming_request_id_is_preserved(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers={"X-Request-ID": "client-request-42"})

    assert response.headers["x-request-id"] == "client-request-42"


def test_invalid_incoming_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers={"X-Request-ID": "bad\r\nid"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] != "bad\r\nid"
