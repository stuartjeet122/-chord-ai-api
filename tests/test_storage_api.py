from __future__ import annotations

import asyncio
import uuid

from fastapi.testclient import TestClient

from conftest import _create_schema
from music_api.config import Environment, Settings
from music_api.main import create_app


def _headers(owner_id: uuid.UUID) -> dict[str, str]:
    return {"X-Dev-User-ID": str(owner_id)}


def _mp3_upload(
    client: TestClient,
    owner_id: uuid.UUID,
    filename: str = "track.mp3",
    content: bytes = b"ID3" + b"\x04\x00\x00test-audio-payload",
    **headers: str,
):
    return client.post(
        "/api/v1/assets",
        headers={**_headers(owner_id), **headers},
        files={"file": (filename, content, "text/plain")},
        data={"asset_type": "original_audio"},
    )


def test_audio_upload_persists_metadata_without_storage_key(client: TestClient) -> None:
    owner_id = uuid.uuid4()

    response = _mp3_upload(
        client,
        owner_id,
        filename="../../untrusted.mp3",
    )

    assert response.status_code == 201
    body = response.json()
    assert body["owner_id"] == str(owner_id)
    assert body["original_filename"] == "untrusted.mp3"
    assert body["asset_type"] == "original_audio"
    assert body["mime_type"] == "audio/mpeg"
    assert body["audio_format"] == "mp3"
    assert body["size_bytes"] == len(b"ID3" + b"\x04\x00\x00test-audio-payload")
    assert len(body["sha256"]) == 64
    assert "storage_key" not in body
    stored_keys = asyncio.run(client.app.state.resources.storage_provider.list_keys("assets/"))
    assert len(stored_keys) == 1


def test_upload_download_and_storage_usage_are_owner_scoped(client: TestClient) -> None:
    owner = uuid.uuid4()
    other_owner = uuid.uuid4()
    content = b"ID3" + b"\x04\x00\x00audio bytes"
    uploaded = _mp3_upload(client, owner, content=content)
    assert uploaded.status_code == 201
    asset_id = uploaded.json()["id"]

    metadata = client.get(f"/api/v1/assets/{asset_id}", headers=_headers(owner))
    assert metadata.status_code == 200
    assert (
        client.get(f"/api/v1/assets/{asset_id}", headers=_headers(other_owner)).status_code == 404
    )

    download = client.get(f"/api/v1/assets/{asset_id}/download", headers=_headers(owner))
    assert download.status_code == 200
    assert download.content == content
    assert download.headers["content-disposition"].startswith("attachment;")
    assert download.headers["x-content-type-options"] == "nosniff"

    listing = client.get("/api/v1/assets", headers=_headers(owner))
    assert listing.status_code == 200
    assert [asset["id"] for asset in listing.json()["items"]] == [asset_id]
    assert client.get("/api/v1/assets", headers=_headers(other_owner)).json()["items"] == []

    usage = client.get("/api/v1/storage/usage", headers=_headers(owner))
    assert usage.status_code == 200
    assert usage.json()["used_bytes"] == len(content)
    assert usage.json()["categories"]["original_audio"] == len(content)


def test_upload_rejects_unknown_file_signature(client: TestClient) -> None:
    response = _mp3_upload(client, uuid.uuid4(), content=b"not an audio file")

    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_file_content"


def test_upload_enforces_file_size_limit(tmp_path) -> None:
    owner = uuid.uuid4()
    settings = Settings(
        app_env=Environment.TESTING,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'size-limit.db'}",
        redis_enabled=False,
        temp_dir=str(tmp_path / "tmp"),
        storage_dir=str(tmp_path / "storage"),
        max_upload_size_bytes=8,
    )
    _create_schema(settings)
    with TestClient(create_app(settings)) as client:
        response = _mp3_upload(
            client,
            owner,
            content=b"ID3" + b"0123456789",
        )

    assert response.status_code == 413
    assert response.json()["code"] == "upload_too_large"


def test_upload_enforces_quota_and_idempotency(client: TestClient) -> None:
    owner = uuid.uuid4()
    content = b"ID3" + b"\x04\x00\x00same-payload"
    first = _mp3_upload(client, owner, content=content, **{"Idempotency-Key": "upload-1"})
    replay = _mp3_upload(client, owner, content=content, **{"Idempotency-Key": "upload-1"})
    conflict = _mp3_upload(
        client,
        owner,
        content=b"ID3" + b"\x04\x00\x00different",
        **{"Idempotency-Key": "upload-1"},
    )

    assert first.status_code == 201
    assert replay.status_code == 201
    assert replay.json()["id"] == first.json()["id"]
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "asset_conflict"


def test_upload_rejects_quota_exhaustion(tmp_path) -> None:
    owner = uuid.uuid4()
    settings = Settings(
        app_env=Environment.TESTING,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'quota.db'}",
        redis_enabled=False,
        temp_dir=str(tmp_path / "tmp"),
        storage_dir=str(tmp_path / "storage"),
        max_upload_size_bytes=1024,
        user_storage_quota_bytes=20,
    )
    _create_schema(settings)
    content = b"ID3" + b"\x04\x00\x00quota-payload"
    with TestClient(create_app(settings)) as client:
        first = _mp3_upload(client, owner, content=content)
        second = _mp3_upload(client, owner, content=content)
        usage = client.get("/api/v1/storage/usage", headers=_headers(owner))

    assert first.status_code == 201
    assert second.status_code == 413
    assert second.json()["code"] == "storage_quota_exceeded"
    assert usage.json()["used_bytes"] == len(content)
    assert usage.json()["reserved_bytes"] == 0


def test_derived_assets_prevent_parent_deletion_until_removed(client: TestClient) -> None:
    owner = uuid.uuid4()
    parent = _mp3_upload(client, owner)
    assert parent.status_code == 201
    child = client.post(
        "/api/v1/assets",
        headers=_headers(owner),
        files={"file": ("derived.mp3", b"ID3\x04\x00\x00derived-payload", "audio/mpeg")},
        data={"asset_type": "processed_audio", "parent_asset_id": parent.json()["id"]},
    )

    assert child.status_code == 201
    assert child.json()["parent_asset_id"] == parent.json()["id"]
    assert (
        client.delete(f"/api/v1/assets/{parent.json()['id']}", headers=_headers(owner)).status_code
        == 409
    )
    assert (
        client.delete(f"/api/v1/assets/{child.json()['id']}", headers=_headers(owner)).status_code
        == 204
    )
    assert (
        client.delete(f"/api/v1/assets/{parent.json()['id']}", headers=_headers(owner)).status_code
        == 204
    )


def test_development_auth_is_required_and_never_accepted_in_production(client: TestClient) -> None:
    assert client.get("/api/v1/assets").status_code == 401

    production = Settings(
        app_env=Environment.PRODUCTION,
        database_url="postgresql+asyncpg://service:secret@db.example/music",
        redis_enabled=False,
        storage_dev_auth_enabled=False,
        docs_enabled=False,
    )
    production_client = TestClient(create_app(production))
    response = production_client.get(
        "/api/v1/assets",
        headers=_headers(uuid.uuid4()),
    )

    assert response.status_code == 401
    assert response.json()["code"] == "http_401"
