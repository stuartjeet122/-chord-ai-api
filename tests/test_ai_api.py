from __future__ import annotations

import asyncio
import uuid

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from music_api.config import Settings
from music_api.storage.models import AudioAsset


def test_model_registry_and_hardware_are_available_without_loading_models(
    client: TestClient,
) -> None:
    models = client.get("/api/v1/ai/models")
    hardware = client.get("/api/v1/ai/hardware")

    assert models.status_code == 200
    assert {model["model_id"] for model in models.json()} >= {
        "moss-music-8b-instruct",
        "music-x-lab-ismir2019-chord",
        "spotify-basic-pitch",
        "demucs-htdemucs",
    }
    assert all(not model["enabled"] for model in models.json())
    assert hardware.status_code == 200
    assert hardware.json()["cpu_count"] >= 1
    assert hardware.json()["configured_device"] == "auto"
    assert "hostname" not in hardware.json()


def test_model_status_discloses_candidates_are_not_enabled(client: TestClient) -> None:
    response = client.get("/api/v1/ai/models/spotify-basic-pitch/status")

    assert response.status_code == 200
    assert response.json()["state"] == "disabled"


def test_unknown_model_returns_standard_not_found_error(client: TestClient) -> None:
    response = client.get("/api/v1/ai/models/not-registered")

    assert response.status_code == 404
    assert response.json()["code"] == "model_not_found"


def test_benchmark_dataset_registration_enforces_audio_ownership(
    client: TestClient,
    settings: Settings,
) -> None:
    owner_id = uuid.uuid4()
    asset_id = uuid.uuid4()
    engine = create_async_engine(settings.database_url)

    async def seed_asset() -> None:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            session.add(
                AudioAsset(
                    id=asset_id,
                    owner_id=owner_id,
                    asset_type="original_audio",
                    original_filename="benchmark.wav",
                    storage_key="test/benchmark.wav",
                    storage_provider="local",
                    mime_type="audio/wav",
                    file_extension="wav",
                    size_bytes=10,
                    sha256="a" * 64,
                    duration_seconds=2.0,
                    sample_rate=44_100,
                    channel_count=1,
                    status="available",
                )
            )
            await session.commit()
        await engine.dispose()

    asyncio.run(seed_asset())

    payload = {
        "dataset_id": "benchmark-set",
        "version": "1",
        "task": "chord_recognition",
        "license": "CC0-1.0",
        "samples": [
            {
                "asset_id": str(asset_id),
                "references": [
                    {
                        "start_seconds": 0.0,
                        "end_seconds": 1.0,
                        "symbol": "C",
                        "root": "C",
                        "quality": "major",
                    }
                ],
            }
        ],
    }

    registered = client.post(
        "/api/v1/ai/benchmarks/datasets",
        json=payload,
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    listed = client.get(
        "/api/v1/ai/benchmarks/datasets",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    duplicate = client.post(
        "/api/v1/ai/benchmarks/datasets",
        json=payload,
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    out_of_range = client.post(
        "/api/v1/ai/benchmarks/datasets",
        json={
            **payload,
            "dataset_id": "out-of-range-set",
            "samples": [
                {
                    **payload["samples"][0],
                    "references": [{**payload["samples"][0]["references"][0], "end_seconds": 3.0}],
                }
            ],
        },
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    unavailable_model = client.post(
        "/api/v1/ai/benchmarks",
        json={
            "dataset_id": "benchmark-set",
            "dataset_version": "1",
            "model_ids": ["music-x-lab-ismir2019-chord"],
            "task": "chord_recognition",
        },
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    rejected = client.post(
        "/api/v1/ai/benchmarks/datasets",
        json={**payload, "dataset_id": "foreign-set"},
        headers={"X-Dev-User-ID": str(uuid.uuid4())},
    )

    assert registered.status_code == 201
    assert registered.json()["sample_count"] == 1
    assert listed.status_code == 200
    assert [item["dataset_id"] for item in listed.json()] == ["benchmark-set"]
    assert duplicate.status_code == 409
    assert out_of_range.status_code == 422
    assert unavailable_model.status_code == 503
    assert rejected.status_code == 422
