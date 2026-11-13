from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from music_api.storage.providers import (
    LocalStorageProvider,
    StorageError,
    UnsafeStorageKeyError,
)


@pytest.mark.asyncio
async def test_local_provider_upload_read_metadata_copy_list_and_delete(
    tmp_path: Path,
) -> None:
    provider = LocalStorageProvider(tmp_path / "store")
    source = tmp_path / "source.bin"
    source.write_bytes(b"streamed payload")

    metadata = await provider.upload_file(
        source,
        "assets/user/song/asset.bin",
        content_type="application/octet-stream",
        metadata={},
    )

    assert metadata.size_bytes == len(b"streamed payload")
    assert await provider.exists("assets/user/song/asset.bin")
    object_metadata = await provider.metadata("assets/user/song/asset.bin")
    assert object_metadata is not None
    assert object_metadata.size_bytes == metadata.size_bytes
    assert object_metadata.content_type is None
    content = b"".join(
        [chunk async for chunk in provider.iter_bytes("assets/user/song/asset.bin", 4)]
    )
    assert content == b"streamed payload"

    copied = await provider.copy("assets/user/song/asset.bin", "assets/user/song/copy.bin")
    assert copied.size_bytes == metadata.size_bytes
    assert await provider.list_keys("assets/") == [
        "assets/user/song/asset.bin",
        "assets/user/song/copy.bin",
    ]
    assert await provider.delete("assets/user/song/copy.bin") is True
    assert await provider.delete("assets/user/song/copy.bin") is False
    assert await provider.metadata("assets/user/song/copy.bin") is None


@pytest.mark.parametrize(
    "key",
    ["../outside", "/absolute", "a/../../outside", "a\\..\\outside", "."],
)
def test_local_provider_rejects_unsafe_keys(tmp_path: Path, key: str) -> None:
    provider = LocalStorageProvider(tmp_path / "store")

    with pytest.raises(UnsafeStorageKeyError):
        asyncio.run(provider.exists(key))


@pytest.mark.asyncio
async def test_failed_local_atomic_upload_leaves_no_final_or_temporary_file(
    tmp_path: Path,
) -> None:
    provider = LocalStorageProvider(tmp_path / "store")
    with pytest.raises(StorageError):
        await provider.upload_file(
            tmp_path / "missing-source",
            "assets/u/s/a.mp3",
            content_type="audio/mpeg",
            metadata={},
        )

    assert await provider.list_keys() == []
    assert list((tmp_path / "store" / "assets" / "u" / "s").glob(".upload-*")) == []


@pytest.mark.asyncio
async def test_local_provider_rejects_symlink_escape(tmp_path: Path) -> None:
    provider = LocalStorageProvider(tmp_path / "store")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = provider.root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is not permitted by this Windows environment.")

    with pytest.raises(UnsafeStorageKeyError):
        await provider.exists("escape/secret")


def test_s3_configuration_requires_credentials_in_production() -> None:
    from pydantic import ValidationError

    from music_api.config import Environment, Settings, StorageProviderName

    with pytest.raises(ValidationError, match="STORAGE_S3_ACCESS_KEY_ID"):
        Settings(
            app_env=Environment.PRODUCTION,
            database_url="postgresql+asyncpg://service:secret@db.example/music",
            redis_enabled=False,
            storage_dev_auth_enabled=False,
            storage_provider=StorageProviderName.S3,
            docs_enabled=False,
        )
