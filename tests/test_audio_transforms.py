from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
import soundfile

import music_api.audio.dsp.worker as dsp_worker
from music_api.audio.dsp.schemas import AudioTransformRequest
from music_api.audio.dsp.worker import process_audio
from music_api.audio.toolchain import ProbeResult
from music_api.audio.transforms import resolve_transform_parameters


def _request(**values: object) -> AudioTransformRequest:
    return AudioTransformRequest.model_validate(values)


def test_transform_parameters_resolve_keys_bpm_and_fractional_semitones() -> None:
    key_shift = resolve_transform_parameters(_request(source_key="C major", target_key="D major"))
    bpm_shift = resolve_transform_parameters(_request(semitones=1.5, source_bpm=120, target_bpm=90))

    assert key_shift["semitones"] == 2
    assert key_shift["target_key"] == "D major"
    assert bpm_shift["semitones"] == 1.5
    assert bpm_shift["tempo_factor"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "values",
    [
        {"source_key": "C major"},
        {"source_key": "C major", "target_key": "D major", "semitones": 2},
        {"source_bpm": 120},
        {"tempo_factor": 0.75, "source_bpm": 120, "target_bpm": 90},
        {"start_seconds": 2, "end_seconds": 1},
        {"tempo_factor": 0.1},
        {"semitones": 25},
    ],
)
def test_transform_request_rejects_conflicting_or_out_of_range_options(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _request(**values)


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("soundstretch") is None, reason="SoundTouch CLI is unavailable")
async def test_soundtouch_transforms_real_audio_and_validates_pitch_tempo_and_section(
    tmp_path: Path,
) -> None:
    sample_rate = 16_000
    duration = 3.0
    source_time = np.arange(round(sample_rate * duration)) / sample_rate
    samples = np.column_stack(
        (
            0.25 * np.sin(2 * np.pi * 220 * source_time),
            0.15 * np.sin(2 * np.pi * 220 * source_time),
        )
    )
    source = tmp_path / "source.wav"
    soundfile.write(source, samples, sample_rate, subtype="FLOAT")
    probe = ProbeResult(
        container="wav",
        codec="pcm_f32le",
        codec_profile=None,
        duration_seconds=duration,
        bitrate=None,
        sample_rate=sample_rate,
        channels=2,
        channel_layout="stereo",
        bit_depth=32,
        audio_stream_count=1,
        file_size_bytes=source.stat().st_size,
        raw_format_tags={},
        variable_bitrate=None,
    )

    async def report(_fraction: float) -> bool:
        return True

    result = await process_audio(
        source,
        tmp_path,
        source_asset_id=uuid.uuid4(),
        operation="time_pitch",
        options={
            "semitones": 12,
            "tempo_factor": 0.8,
            "source_start_seconds": 0.5,
            "source_end_seconds": 2.5,
            "start_seconds": 0.5,
            "end_seconds": 2.5,
            "_selected_section": True,
            "edge_fade_ms": 10,
        },
        probe=probe,
        progress=report,
        soundtouch_path=shutil.which("soundstretch") or "soundstretch",
        soundtouch_version="installed",
    )
    output, output_rate = soundfile.read(result.outputs[0].path, always_2d=True)
    frequency_spectrum = np.abs(np.fft.rfft(output[1000:-1000, 0]))
    frequencies = np.fft.rfftfreq(len(output[1000:-1000, 0]), 1 / output_rate)

    assert output.shape[1] == 2
    assert output_rate == sample_rate
    assert result.result["source_start_seconds"] == pytest.approx(0.5)
    assert result.result["source_end_seconds"] == pytest.approx(2.5)
    assert result.result["output_duration_seconds"] == pytest.approx(2.5, abs=0.1)
    assert frequencies[int(np.argmax(frequency_spectrum))] == pytest.approx(440, abs=4)
    assert result.result["non_finite_sample_count"] == 0


def test_transform_endpoint_rejects_invalid_request(client) -> None:
    response = client.post(
        f"/api/v1/audio/{uuid.uuid4()}/transform",
        json={"source_key": "C major"},
        headers={"X-Dev-User-ID": str(uuid.uuid4())},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_transform_progress_is_monotonic_during_peak_adjustment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_rate = 8000
    source_time = np.arange(sample_rate) / sample_rate
    samples = (0.98 * np.sin(2 * np.pi * 440 * source_time)).astype(np.float32)
    source = tmp_path / "source.wav"
    soundfile.write(source, samples, sample_rate, subtype="FLOAT")
    probe = ProbeResult(
        container="wav",
        codec="pcm_f32le",
        codec_profile=None,
        duration_seconds=1.0,
        bitrate=None,
        sample_rate=sample_rate,
        channels=1,
        channel_layout="mono",
        bit_depth=32,
        audio_stream_count=1,
        file_size_bytes=source.stat().st_size,
        raw_format_tags={},
        variable_bitrate=None,
    )

    async def fake_soundstretch(
        input_path: Path,
        output_path: Path,
        *,
        executable: str,
        semitones: float,
        tempo_factor: float,
        progress,
        timeout_seconds: int,
        version: str,
    ) -> dict[str, object]:
        for fraction in (0.0, 0.5, 1.0):
            assert await progress(fraction)
        input_samples, input_rate = soundfile.read(input_path)
        soundfile.write(output_path, input_samples, input_rate, subtype="FLOAT")
        return {
            "engine": "test",
            "engine_version": "test",
            "tempo_factor": tempo_factor,
            "tempo_percent": 0.0,
            "semitones": semitones,
            "input_frames": sample_rate,
            "output_frames": sample_rate,
            "sample_rate": sample_rate,
            "channel_count": 1,
            "output_duration_seconds": 1.0,
            "output_subtype": "FLOAT",
        }

    monkeypatch.setattr(dsp_worker, "run_soundstretch", fake_soundstretch)
    progress_values: list[float] = []

    async def report(fraction: float) -> bool:
        progress_values.append(fraction)
        return True

    result = await process_audio(
        source,
        tmp_path,
        source_asset_id=uuid.uuid4(),
        operation="time_pitch",
        options={"semitones": 0.0, "tempo_factor": 1.0},
        probe=probe,
        progress=report,
    )

    assert progress_values == sorted(progress_values)
    assert progress_values[-1] == 1.0
    assert result.result["peak_adjustment_db"] < 0
    assert max(result.result["output_quality"]["peak_amplitude"]) <= 10 ** (-1 / 20)
