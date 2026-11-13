from __future__ import annotations

import asyncio
import io
import subprocess
import uuid
from pathlib import Path

import pytest
import soundfile
from fastapi.testclient import TestClient
from mutagen.id3 import APIC, ID3, TALB, TIT2, TPE1
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from music_api.audio.metadata import extract_metadata
from music_api.audio.models import (
    AudioArtwork,
    AudioConversion,
    AudioMetadata,
    AudioProcessingRecord,
)
from music_api.audio.service import AudioIngestionService
from music_api.audio.toolchain import AudioToolchain, AudioToolError
from music_api.config import Settings
from music_api.db.session import Database
from music_api.jobs.models import OutboxEvent, ProcessingJob
from music_api.music.models import MusicAnalysisRecord
from music_api.storage.providers import build_storage_provider
from music_api.storage.service import StorageService


def _ffmpeg(settings: Settings, arguments: list[str]) -> None:
    subprocess.run(
        [settings.ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
        capture_output=True,
        timeout=30,
    )


def _make_wav(settings: Settings, path: Path, *, duration: float = 1.0) -> None:
    _ffmpeg(
        settings,
        [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={duration}",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extension", "encoder_args"),
    [
        ("wav", ["-c:a", "pcm_s16le"]),
        ("mp3", ["-c:a", "libmp3lame", "-b:a", "128k"]),
        ("flac", ["-c:a", "flac"]),
        ("m4a", ["-c:a", "aac", "-f", "ipod"]),
        ("aac", ["-c:a", "aac", "-f", "adts"]),
        ("ogg", ["-c:a", "libvorbis", "-f", "ogg"]),
        ("opus", ["-c:a", "libopus", "-f", "ogg"]),
        ("aiff", ["-c:a", "pcm_s16be", "-f", "aiff"]),
    ],
)
async def test_real_ffmpeg_formats_are_probed_and_decoded(
    settings: Settings,
    tmp_path: Path,
    extension: str,
    encoder_args: list[str],
) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / f"sample.{extension}"
    _make_wav(settings, source)
    _ffmpeg(settings, ["-i", str(source), *encoder_args, str(output)])

    tools = AudioToolchain(settings)
    probe = await tools.inspect(output)
    await tools.verify_decodable(output)

    assert probe.audio_stream_count == 1
    assert probe.duration_seconds == pytest.approx(1.0, abs=0.1)
    assert probe.channels == 2
    assert probe.sample_rate == 48000
    assert probe.file_size_bytes == output.stat().st_size
    assert probe.codec


@pytest.mark.asyncio
async def test_wav_conversion_verifies_profile_and_preserves_original(
    settings: Settings,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "converted.wav"
    _make_wav(settings, source, duration=0.75)
    original_bytes = source.read_bytes()
    tools = AudioToolchain(settings)
    source_probe = await tools.inspect(source)

    async def report_progress(_elapsed_seconds: float) -> bool:
        return True

    await tools.convert(
        source,
        output,
        {
            "output_format": "wav",
            "sample_rate": 44100,
            "channels": "original",
            "bit_depth": 24,
            "float32": False,
            "normalize": False,
            "allow_upsampling": False,
        },
        source_probe,
        report_progress,
    )
    converted = await tools.inspect(output)
    await tools.verify_decodable(output)

    assert converted.codec == "pcm_s24le"
    assert converted.sample_rate == 44100
    assert converted.channels == 2
    assert converted.bit_depth == 24
    assert converted.duration_seconds == pytest.approx(source_probe.duration_seconds, abs=0.02)
    assert source.read_bytes() == original_bytes


@pytest.mark.asyncio
async def test_real_ffmpeg_ebu_r128_loudness_normalization(
    settings: Settings,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "normalized.wav"
    _make_wav(settings, source, duration=3.0)
    tools = AudioToolchain(settings)

    async def report_progress(_elapsed_seconds: float) -> bool:
        return True

    report = await tools.normalize_loudness(
        source,
        output,
        target_lufs=-16,
        true_peak_dbfs=-1.5,
        loudness_range_lu=11,
        limit=True,
        sample_rate=48000,
        progress=report_progress,
    )
    normalized = await tools.inspect(output)
    await tools.verify_decodable(output)

    assert normalized.sample_rate == 48000
    assert normalized.channels == 2
    assert report["method"] == "ebu_r128_loudnorm"
    assert report["integrated_loudness_input_lufs"] < -16
    assert abs(report["integrated_loudness_output_lufs"] - (-16)) < 1.0
    assert report["true_peak_output_dbfs"] <= -1.0


def test_mutagen_extracts_tags_and_validated_embedded_artwork(
    settings: Settings,
    tmp_path: Path,
) -> None:
    source = tmp_path / "tagged.mp3"
    _make_wav(settings, tmp_path / "source.wav")
    _ffmpeg(
        settings,
        [
            "-i",
            str(tmp_path / "source.wav"),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(source),
        ],
    )
    cover = io.BytesIO()
    Image.new("RGB", (32, 32), color="blue").save(cover, format="PNG")
    tags = ID3()
    tags.add(TIT2(encoding=3, text=["Phase Four Fixture"]))
    tags.add(TPE1(encoding=3, text=["Synthetic Artist"]))
    tags.add(TALB(encoding=3, text=["Test Album"]))
    tags.add(APIC(encoding=3, mime="image/png", type=3, desc="front", data=cover.getvalue()))
    tags.save(source)

    probe = asyncio.run(AudioToolchain(settings).inspect(source))
    metadata = extract_metadata(source, probe)

    assert metadata.tags["title"] == "Phase Four Fixture"
    assert metadata.tags["artist"] == "Synthetic Artist"
    assert metadata.tags["album"] == "Test Album"
    assert len(metadata.artworks) == 1
    assert metadata.artworks[0].mime_type == "image/png"
    assert metadata.artworks[0].data == cover.getvalue()
    assert metadata.technical["sample_rate"] == 48000


@pytest.mark.asyncio
async def test_corrupt_audio_is_rejected_by_ffprobe(settings: Settings, tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.mp3"
    corrupt.write_bytes(b"ID3\x04\x00\x00not-a-real-mpeg-stream")

    with pytest.raises(AudioToolError):
        await AudioToolchain(settings).inspect(corrupt)


def _run_processor(settings: Settings, job_id: uuid.UUID) -> None:
    async def run() -> None:
        database = Database(settings.database_url)
        provider = build_storage_provider(settings)
        storage = StorageService(settings, provider)
        try:
            await AudioIngestionService(settings, database, provider, storage).process_job(job_id)
        finally:
            await provider.close()
            await database.close()

    asyncio.run(run())


def test_inspection_and_conversion_jobs_end_to_end(
    worker_client: TestClient,
    settings: Settings,
    tmp_path: Path,
) -> None:
    client = worker_client
    owner_id = uuid.uuid4()
    source = tmp_path / "tagged.mp3"
    _make_wav(settings, tmp_path / "source.wav")
    _ffmpeg(
        settings,
        [
            "-i",
            str(tmp_path / "source.wav"),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(source),
        ],
    )
    image = io.BytesIO()
    Image.new("RGB", (24, 24), color="green").save(image, format="PNG")
    tags = ID3()
    tags.add(TIT2(encoding=3, text=["End-to-end Track"]))
    tags.add(TPE1(encoding=3, text=["Fixture Artist"]))
    tags.add(APIC(encoding=3, mime="image/png", type=3, desc="front", data=image.getvalue()))
    tags.save(source)
    original_bytes = source.read_bytes()
    original_upload = client.post(
        "/api/v1/assets",
        headers={"X-Dev-User-ID": str(owner_id)},
        files={"file": ("source.mp3", original_bytes, "audio/mpeg")},
        data={"asset_type": "original_audio"},
    )
    assert original_upload.status_code == 201
    source_id = original_upload.json()["id"]

    inspect = client.post(
        f"/api/v1/audio/{source_id}/inspect",
        headers={"X-Dev-User-ID": str(owner_id), "Idempotency-Key": "inspect-fixture"},
    )
    assert inspect.status_code == 202
    replay = client.post(
        f"/api/v1/audio/{source_id}/inspect",
        headers={"X-Dev-User-ID": str(owner_id), "Idempotency-Key": "inspect-fixture"},
    )
    assert replay.status_code == 202
    assert replay.json()["id"] == inspect.json()["id"]

    inspect_job_id = uuid.UUID(inspect.json()["id"])
    _run_processor(settings, inspect_job_id)
    inspect_status = client.get(
        f"/api/v1/jobs/{inspect_job_id}",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    assert inspect_status.json()["status"] == "succeeded"
    assert inspect_status.json()["progress"] == 100

    metadata = client.get(
        f"/api/v1/audio/{source_id}/metadata",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    assert metadata.status_code == 200
    assert metadata.json()["tags"]["title"] == "End-to-end Track"
    assert metadata.json()["technical"]["codec"] == "mp3"
    assert len(metadata.json()["artworks"]) == 1
    artwork_id = metadata.json()["artworks"][0]["asset_id"]
    artwork_download = client.get(
        f"/api/v1/assets/{artwork_id}/download",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    assert artwork_download.status_code == 200
    assert artwork_download.content == image.getvalue()

    conversion = client.post(
        f"/api/v1/audio/{source_id}/convert",
        headers={"X-Dev-User-ID": str(owner_id), "Idempotency-Key": "convert-fixture"},
        json={
            "profile": "music_analysis_wav",
            "output_format": "wav",
            "channels": "original",
            "bit_depth": 24,
        },
    )
    assert conversion.status_code == 202
    conversion_job_id = uuid.UUID(conversion.json()["id"])
    _run_processor(settings, conversion_job_id)
    conversion_status = client.get(
        f"/api/v1/jobs/{conversion_job_id}",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    assert conversion_status.json()["status"] == "succeeded"
    output_id = conversion_status.json()["output_asset_id"]
    output = client.get(
        f"/api/v1/assets/{output_id}",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    assert output.json()["audio_format"] == "pcm_s24le"
    assert output.json()["sample_rate"] == 44100
    assert output.json()["channel_count"] == 2
    output_bytes = client.get(
        f"/api/v1/assets/{output_id}/download",
        headers={"X-Dev-User-ID": str(owner_id)},
    ).content
    assert output_bytes.startswith(b"RIFF")
    assert (
        client.get(
            f"/api/v1/audio/{source_id}/assets",
            headers={"X-Dev-User-ID": str(owner_id)},
        ).json()["assets"][0]["id"]
        == output_id
    )
    assert (
        client.get(
            f"/api/v1/assets/{source_id}/download",
            headers={"X-Dev-User-ID": str(owner_id)},
        ).content
        == original_bytes
    )

    async def assert_database_records() -> None:
        engine = create_async_engine(settings.database_url)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with sessions() as session:
            metadata_record = await session.get(AudioMetadata, uuid.UUID(source_id))
            artworks = list(
                (
                    await session.scalars(
                        select(AudioArtwork).where(
                            AudioArtwork.audio_asset_id == uuid.UUID(source_id)
                        )
                    )
                ).all()
            )
            conversion_record = await session.scalar(
                select(AudioConversion).where(AudioConversion.job_id == conversion_job_id)
            )
            outbox_events = list(
                (
                    await session.scalars(
                        select(OutboxEvent).where(
                            OutboxEvent.job_id.in_([inspect_job_id, conversion_job_id])
                        )
                    )
                ).all()
            )
            assert metadata_record is not None
            assert len(artworks) == 1
            assert conversion_record is not None
            assert conversion_record.output_asset_id == uuid.UUID(output_id)
            assert len(outbox_events) == 2
        await engine.dispose()

    asyncio.run(assert_database_records())


def test_dsp_jobs_persist_waveforms_trimmed_assets_and_history(
    worker_client: TestClient,
    settings: Settings,
    tmp_path: Path,
) -> None:
    client = worker_client
    owner_id = uuid.uuid4()
    source = tmp_path / "source.wav"
    _make_wav(settings, source, duration=1.0)
    upload = client.post(
        "/api/v1/assets",
        headers={"X-Dev-User-ID": str(owner_id)},
        files={"file": ("dsp-fixture.wav", source.read_bytes(), "audio/wav")},
        data={"asset_type": "original_audio"},
    )
    assert upload.status_code == 201
    asset_id = upload.json()["id"]

    waveform_job = client.post(
        f"/api/v1/audio/{asset_id}/waveform",
        headers={"X-Dev-User-ID": str(owner_id), "Idempotency-Key": "dsp-waveform"},
        json={"samples_per_second": 20},
    )
    assert waveform_job.status_code == 202
    waveform_job_id = uuid.UUID(waveform_job.json()["id"])
    _run_processor(settings, waveform_job_id)
    waveform_status = client.get(
        f"/api/v1/jobs/{waveform_job_id}",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    assert waveform_status.json()["status"] == "succeeded"
    waveform_response = client.get(
        f"/api/v1/audio/{asset_id}/waveform",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    assert waveform_response.status_code == 200
    waveform = waveform_response.json()
    assert waveform["duration_seconds"] == pytest.approx(1.0, abs=0.02)
    assert len(waveform["timestamps_seconds"]) == 20
    waveform_output_id = waveform_status.json()["output_asset_id"]

    async def simulate_retry() -> None:
        engine = create_async_engine(settings.database_url)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with sessions() as session:
            job = await session.get(ProcessingJob, waveform_job_id)
            assert job is not None
            job.status = "retry_wait"
            job.stage = "RETRY_WAIT"
            job.finished_at = None
            await session.commit()
        await engine.dispose()

    asyncio.run(simulate_retry())
    _run_processor(settings, waveform_job_id)
    retried_waveform_job = client.get(
        f"/api/v1/jobs/{waveform_job_id}",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    assert retried_waveform_job.json()["status"] == "succeeded"
    assert retried_waveform_job.json()["attempt_count"] == 2
    assert retried_waveform_job.json()["output_asset_id"] == waveform_output_id

    trim_job = client.post(
        f"/api/v1/audio/{asset_id}/trim",
        headers={"X-Dev-User-ID": str(owner_id), "Idempotency-Key": "dsp-trim"},
        json={"start_seconds": 0.2, "end_seconds": 0.7},
    )
    assert trim_job.status_code == 202
    trim_job_id = uuid.UUID(trim_job.json()["id"])
    _run_processor(settings, trim_job_id)
    trim_status = client.get(
        f"/api/v1/jobs/{trim_job_id}",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    assert trim_status.json()["status"] == "succeeded"
    trimmed_id = trim_status.json()["output_asset_id"]
    trimmed_info = client.get(
        f"/api/v1/assets/{trimmed_id}",
        headers={"X-Dev-User-ID": str(owner_id)},
    ).json()
    assert trimmed_info["duration_seconds"] == pytest.approx(0.5, abs=0.02)
    output_audio = client.get(
        f"/api/v1/assets/{trimmed_id}/download",
        headers={"X-Dev-User-ID": str(owner_id)},
    ).content
    output_path = tmp_path / "downloaded-trim.wav"
    output_path.write_bytes(output_audio)
    decoded, _ = soundfile.read(output_path, always_2d=True)
    assert decoded.shape[0] == 24_000

    history = client.get(
        f"/api/v1/audio/{asset_id}/processing-history",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    assert history.status_code == 200
    assert {row["operation"] for row in history.json()} == {"waveform", "trim"}

    async def verify_records() -> None:
        engine = create_async_engine(settings.database_url)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with sessions() as session:
            records = list(
                (
                    await session.scalars(
                        select(AudioProcessingRecord).where(
                            AudioProcessingRecord.source_asset_id == uuid.UUID(asset_id)
                        )
                    )
                ).all()
            )
            assert len(records) == 2
            assert all(record.status == "succeeded" for record in records)
        await engine.dispose()

    asyncio.run(verify_records())


def test_music_analysis_job_persists_versioned_results_and_timeline(
    worker_client: TestClient,
    settings: Settings,
    tmp_path: Path,
) -> None:
    source = tmp_path / "music-analysis.wav"
    _make_wav(settings, source, duration=2.0)
    owner_id = uuid.uuid4()
    headers = {"X-Dev-User-ID": str(owner_id)}
    uploaded = worker_client.post(
        "/api/v1/assets",
        headers=headers,
        files={"file": ("music-analysis.wav", source.read_bytes(), "audio/wav")},
        data={"asset_type": "original_audio"},
    )
    assert uploaded.status_code == 201
    asset_id = uploaded.json()["id"]

    queued = worker_client.post(
        f"/api/v1/music/{asset_id}/analyze",
        headers={**headers, "Idempotency-Key": "music-analysis-v1"},
        json={"tasks": ["chords", "key", "tempo", "beats"]},
    )
    assert queued.status_code == 202
    duplicate = worker_client.post(
        f"/api/v1/music/{asset_id}/analyze",
        headers={**headers, "Idempotency-Key": "music-analysis-v1"},
        json={"tasks": ["chords", "key", "tempo", "beats"]},
    )
    assert duplicate.json()["id"] == queued.json()["id"]
    job_id = uuid.UUID(queued.json()["id"])
    _run_processor(settings, job_id)

    analysis = worker_client.get(f"/api/v1/music/{asset_id}/analysis", headers=headers)
    chords = worker_client.get(f"/api/v1/music/{asset_id}/chords", headers=headers)
    key = worker_client.get(f"/api/v1/music/{asset_id}/key", headers=headers)
    tempo = worker_client.get(f"/api/v1/music/{asset_id}/tempo", headers=headers)
    job = worker_client.get(f"/api/v1/jobs/{job_id}", headers=headers)

    assert job.json()["status"] == "succeeded"
    assert analysis.status_code == 200
    assert analysis.json()["version"] == 1
    assert analysis.json()["source_sha256"] == uploaded.json()["sha256"]
    assert analysis.json()["status"] == "succeeded"
    assert chords.status_code == 200
    assert chords.json()["analysis_id"] == analysis.json()["id"]
    assert all(item["confidence"] is None for item in chords.json()["items"])
    assert key.status_code == 200
    assert tempo.status_code == 200
    assert tempo.json()["time_signature"] is None

    interpretation_path = f"/api/v1/music/{asset_id}/analysis/{analysis.json()['id']}/theory"
    interpretation = worker_client.post(
        interpretation_path,
        headers=headers,
        json={"key": "C major", "chord_symbols": ["C", "D7", "G", "Fm", "C"]},
    )
    assert interpretation.status_code == 201
    assert interpretation.json()["version"] == 1
    assert interpretation.json()["source"] == "provided_chords"
    assert interpretation.json()["result"]["chords"][1]["roman_numeral"] == "V/V"
    assert interpretation.json()["result"]["chords"][3]["classification"] == "borrowed"
    retrieved_interpretation = worker_client.get(
        f"{interpretation_path}?version=1",
        headers=headers,
    )
    assert retrieved_interpretation.status_code == 200
    assert retrieved_interpretation.json()["id"] == interpretation.json()["id"]
    other_owner = worker_client.get(
        interpretation_path,
        headers={"X-Dev-User-ID": str(uuid.uuid4())},
    )
    assert other_owner.status_code == 404
    unchanged_analysis = worker_client.get(f"/api/v1/music/{asset_id}/analysis", headers=headers)
    assert "theory" not in unchanged_analysis.json()["result"]

    next_queued = worker_client.post(
        f"/api/v1/music/{asset_id}/analyze",
        headers=headers,
        json={"tasks": ["chords"]},
    )
    assert next_queued.status_code == 202
    _run_processor(settings, uuid.UUID(next_queued.json()["id"]))
    comparison = worker_client.get(
        f"/api/v1/music/{asset_id}/analysis/compare",
        headers=headers,
    )
    latest = worker_client.get(f"/api/v1/music/{asset_id}/analysis", headers=headers)
    assert latest.json()["version"] == 2
    assert [row["version"] for row in comparison.json()["analyses"]] == [2, 1]


def test_queued_music_analysis_cancellation_updates_analysis_version(
    worker_client: TestClient,
    settings: Settings,
    tmp_path: Path,
) -> None:
    source = tmp_path / "music-cancel.wav"
    _make_wav(settings, source, duration=1.0)
    owner_id = uuid.uuid4()
    headers = {"X-Dev-User-ID": str(owner_id)}
    uploaded = worker_client.post(
        "/api/v1/assets",
        headers=headers,
        files={"file": ("music-cancel.wav", source.read_bytes(), "audio/wav")},
        data={"asset_type": "original_audio"},
    )
    assert uploaded.status_code == 201
    queued = worker_client.post(
        f"/api/v1/music/{uploaded.json()['id']}/analyze",
        headers=headers,
        json={"tasks": ["chords"]},
    )
    assert queued.status_code == 202
    job_id = uuid.UUID(queued.json()["id"])

    cancelled = worker_client.post(f"/api/v1/jobs/{job_id}/cancel", headers=headers)
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    async def verify_analysis_state() -> None:
        engine = create_async_engine(settings.database_url)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with sessions() as session:
            analysis = await session.scalar(
                select(MusicAnalysisRecord).where(MusicAnalysisRecord.job_id == job_id)
            )
            assert analysis is not None
            assert analysis.status == "cancelled"
        await engine.dispose()

    asyncio.run(verify_analysis_state())


def test_job_cancellation_and_owner_isolation(
    worker_client: TestClient,
    settings: Settings,
) -> None:
    client = worker_client
    owner_id = uuid.uuid4()
    other_owner = uuid.uuid4()
    upload = client.post(
        "/api/v1/assets",
        headers={"X-Dev-User-ID": str(owner_id)},
        files={
            "file": ("fixture.wav", b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 32, "audio/wav")
        },
        data={"asset_type": "original_audio"},
    )
    assert upload.status_code == 201
    source_id = upload.json()["id"]
    job = client.post(
        f"/api/v1/audio/{source_id}/inspect",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    assert job.status_code == 202
    job_id = job.json()["id"]

    assert (
        client.get(
            f"/api/v1/jobs/{job_id}",
            headers={"X-Dev-User-ID": str(other_owner)},
        ).status_code
        == 404
    )
    cancelled = client.post(
        f"/api/v1/jobs/{job_id}/cancel",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    _run_processor(settings, uuid.UUID(job_id))
    status_response = client.get(
        f"/api/v1/jobs/{job_id}",
        headers={"X-Dev-User-ID": str(owner_id)},
    )
    assert status_response.json()["status"] == "cancelled"
