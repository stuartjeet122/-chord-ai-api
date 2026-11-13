from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_endpoints_do_not_require_external_services(client: TestClient) -> None:
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert client.get("/api/v1/health/live").json() == {"status": "alive"}
    assert client.get("/api/v1/health/ready").json() == {"status": "ready"}
