from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from music_api.config import Environment, Settings
from music_api.db.base import Base
from music_api.storage import models as storage_models  # noqa: F401
from music_api.storage.models import AssetType, UploadReservation
from music_api.storage.providers import LocalStorageProvider
from music_api.storage.service import StorageService


class _Upload:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.offset = 0

    async def read(self, size: int) -> bytes:
        chunk = self.content[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


@pytest.mark.asyncio
async def test_reconcile_reports_integrity_issues_and_cleans_only_temporary_data(
    tmp_path: Path,
) -> None:
    settings = Settings(
        app_env=Environment.TESTING,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'reconcile.db'}",
        redis_enabled=False,
        temp_dir=str(tmp_path / "tmp"),
        storage_dir=str(tmp_path / "storage"),
        upload_staging_ttl_seconds=1,
    )
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    provider = LocalStorageProvider(settings.storage_dir)
    service = StorageService(settings, provider)
    owner_id = uuid.uuid4()

    async with sessions() as session:
        staged = await service.stage_upload(
            _Upload(b"ID3\x04\x00\x00recorded-audio"),
            filename="recording.mp3",
            asset_type=AssetType.ORIGINAL_AUDIO,
        )
        asset, _ = await service.create_asset(session, owner_id, staged)
        await provider.delete(asset.storage_key)

        stale = await service.stage_upload(
            _Upload(b"ID3\x04\x00\x00stale-upload"),
            filename="stale.mp3",
            asset_type=AssetType.ORIGINAL_AUDIO,
        )
        old_timestamp = (datetime.now(UTC) - timedelta(seconds=10)).timestamp()
        os.utime(stale.staging_path, (old_timestamp, old_timestamp))

        orphan_source = tmp_path / "orphan.bin"
        orphan_source.write_bytes(b"orphan")
        orphan_key = "assets/orphan/untracked/file.bin"
        await provider.upload_file(
            orphan_source,
            orphan_key,
            content_type="application/octet-stream",
            metadata={},
        )
        session.add(
            UploadReservation(
                owner_id=owner_id,
                size_bytes=1,
                created_at=datetime.now(UTC) - timedelta(seconds=10),
            )
        )
        await session.commit()

        result = await service.reconcile(session, staging_ttl_seconds=1)

    assert result["missing_asset_ids"] == [str(asset.id)]
    assert result["orphaned_storage_keys"] == [orphan_key]
    assert result["expired_staging_files_removed"] == [stale.staging_path.name]
    assert result["expired_reservations_removed"] == 1
    assert result["deleted_assets"] == []
    assert await provider.exists(orphan_key)
    await engine.dispose()
